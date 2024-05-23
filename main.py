import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
import torchvision

torchvision.disable_beta_transforms_warning()
torch.manual_seed(42)

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from argparse import ArgumentParser
import os
import cv2
import numpy as np
from datetime import datetime
import shutil
import pickle
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import scipy
import pandas as pd

from loader import get_dataset
from model.model import HierarchicalProbUNet
from utils import *


class AtopyDataModule(pl.LightningDataModule):
    def __init__(self, cfg, test_dataset):
        super().__init__()
        self.cfg = cfg
        self.test_dataset = test_dataset

    def setup(self, stage):
        self.roi_train, self.roi_val = get_dataset("roi_train", wholebody=self.cfg.train.wholebody)
        self.wb_train, self.wb_val = get_dataset("wb_train")

        self.test = get_dataset(self.test_dataset, step_size=self.cfg.test.step)

    def train_dataloader(self):
        if self.cfg.train.wholebody:
            return CombinedLoader(
                [
                    DataLoader(
                        self.roi_train,
                        batch_size=self.cfg.train.batch_size[0],
                        num_workers=self.cfg.num_workers,
                        pin_memory=True,
                        shuffle=True,
                    ),
                    DataLoader(
                        self.wb_train,
                        batch_size=self.cfg.train.batch_size[1],
                        num_workers=self.cfg.num_workers,
                        pin_memory=True,
                        shuffle=True,
                    ),
                ],
                "max_size_cycle",
            )
        else:
            return DataLoader(
                self.roi_train,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
                shuffle=True,
            )

    def val_dataloader(self):
        if self.cfg.train.wholebody:
            return CombinedLoader(
                [
                    DataLoader(
                        self.roi_val,
                        batch_size=self.cfg.train.batch_size[0],
                        num_workers=self.cfg.num_workers,
                        pin_memory=True,
                    ),
                    DataLoader(
                        self.wb_val,
                        batch_size=self.cfg.train.batch_size[1],
                        num_workers=self.cfg.num_workers,
                        pin_memory=True,
                    ),
                ],
                "max_size_cycle",
            )
        else:
            return DataLoader(
                self.roi_val,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
            )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.cfg.test.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
        )


class LitModel(pl.LightningModule):
    def __init__(self, model, cfg, exp_name=None):
        super().__init__()

        self.model = model
        self.cfg = cfg
        self.mc_n = cfg.test.mc_n
        self.exp_name = exp_name
        self.cfg.exp_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        self.step = self.cfg.test.step

        wind = scipy.signal.windows.get_window("cosine", 256)
        wind = wind / np.average(wind)
        wind = np.expand_dims(wind, 1)
        wind = wind * wind.T
        self.window = torch.from_numpy(wind).unsqueeze(-1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.cfg.train.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_start(self):
        self.logger.log_hyperparams(dict(self.cfg.train))

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
            batch, wb_batch = batch

            wb_img = wb_batch["img"]  # BN3HW
            wb_seg = wb_batch["seg"]  # BNCHW
            wb_img = torch.flatten(wb_img, 0, 1)  # (BN)3HW
            wb_seg = torch.flatten(wb_seg, 0, 1)  # (BN)CHW

            img = batch["img"]  # B3HW
            seg = batch["seg"]  # BL1HW
            grade = batch["grade"].swapaxes(1, 2)[:, :, None, :, None, None]  # BL'1C11

            seg = seg.unsqueeze(1) * grade
            seg = torch.flatten(seg, 0, 2)  # (BL'L)CHW
            img = torch.flatten(img.unsqueeze(1).expand(-1, seg.shape[0] // img.shape[0], -1, -1, -1), 0, 1)
            # Segmentation and grade labels of an image are intended to be fed into the model at the same time,
            # to learn diverse possibilities that the image possesses.

            img = torch.cat([img, wb_img])
            seg = torch.cat([seg, wb_seg])

        else:
            img = batch["img"]  # B3HW
            seg = batch["seg"]  # BL1HW
            grade = batch["grade"].swapaxes(1, 2)[:, :, None, :, None, None]  # BL'1C11

            seg = seg.unsqueeze(1) * grade
            seg = torch.flatten(seg, 0, 2)  # (BL'L)CHW
            img = torch.flatten(img.unsqueeze(1).expand(-1, seg.shape[0] // img.shape[0], -1, -1, -1), 0, 1)

        loss_dict = self.model.sum_loss(seg, img)
        self.log_dict(
            {"train_" + k: v for k, v in loss_dict["summaries"].items()},
            sync_dist=True,
            on_epoch=True,
            on_step=False,
            batch_size=seg.shape[0],
        )

        return loss_dict["supervised_loss"]

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, list):
            batch, wb_batch = batch

            wb_img = wb_batch["img"]  # BN3HW
            wb_seg = wb_batch["seg"]  # BNCHW
            wb_img = torch.flatten(wb_img, 0, 1)  # (BN)3HW
            wb_seg = torch.flatten(wb_seg, 0, 1)  # (BN)CHW

            img = batch["img"]  # B3HW
            seg = batch["seg"]  # BL1HW
            grade = batch["grade"].swapaxes(1, 2)[:, :, None, :, None, None]  # BL'1C11

            seg = seg.unsqueeze(1) * grade
            seg = torch.flatten(seg, 0, 2)  # (BL'L)CHW
            img = torch.flatten(img.unsqueeze(1).expand(-1, seg.shape[0] // img.shape[0], -1, -1, -1), 0, 1)
            # Segmentation and grade labels of an image are intended to be fed into the model at the same time,
            # to learn diverse possibilities that the image possesses.

            img = torch.cat([img, wb_img])
            seg = torch.cat([seg, wb_seg])

        else:
            img = batch["img"]  # B3HW
            seg = batch["seg"]  # BL1HW
            grade = batch["grade"].swapaxes(1, 2)[:, :, None, :, None, None]  # BL'1C11

            seg = seg.unsqueeze(1) * grade
            seg = torch.flatten(seg, 0, 2)  # (BL'L)CHW
            img = torch.flatten(img.unsqueeze(1).expand(-1, seg.shape[0] // img.shape[0], -1, -1, -1), 0, 1)

        loss_dict = self.model.sum_loss(seg, img)
        self.log_dict(
            {"val_" + k: v for k, v in loss_dict["summaries"].items()},
            sync_dist=True,
            on_epoch=True,
            on_step=False,
            batch_size=seg.shape[0],
        )

    def on_validation_epoch_end(self) -> None:
        self.model._loss_kwargs["beta"] /= self.cfg.train.gamma

    def on_test_start(self):
        self.logger.log_hyperparams(dict(self.cfg.test))
        self.gt_area = []
        self.gt_severity = []
        self.gt_kappa = []
        self.gt_easi = []
        self.area = []
        self.severity = []
        self.kappa = []
        self.easi = []
        self.file_names = []

    def test_step(self, batch, batch_idx):
        # Load batch
        seg = batch["seg"]  # BLHW
        gt_area = batch["area"]  # BL1
        grade = batch["grade"]  # BCL'
        ori_img = batch["ori_img"].cpu().numpy()
        file_name = batch["file_name"]
        self.file_names.extend(file_name)
        num_classes = self.cfg.model.num_cuts + 1

        gt_area = batch["area"].transpose(0, 1).cpu().numpy()  # LB1
        # Ground Truth
        gt_severity = grade.permute(2, 0, 1).cpu().numpy()  # L'BC

        if "img" in batch.keys():
            area4k = seg.flatten(2).transpose(0, 1).cpu().numpy()  # LB(HW)
            gt_kappa = np.nan_to_num(
                np.array([fleiss_kappa(aggregate_raters(area4k[:, i].T)[0]) for i in range(area4k.shape[1])]),
                nan=1.0,
            )  # Complete agreement : k = 1
            preds = self.model.sample(batch["img"], self.mc_n).argmax(-1)  # NBCHW

            # Prediction; area and severity scores
            area = (preds.sum(2) > 1).reshape(preds.shape[:2] + (-1,)).cpu().numpy()  # NB(HW)
            kappa = np.nan_to_num(
                np.array([fleiss_kappa(aggregate_raters(area[:, i].T)[0]) for i in range(area.shape[1])]),
                nan=1.0,
            )
            area = area.astype(np.float32).mean(-1, keepdims=True)  # NB1
            mean_area = area.mean(0)
            severity = preds.float().mean((3, 4)).cpu().numpy() / mean_area  # NBC
            severity = 0.5 * (severity + severity.mean(2, keepdims=True))
            easi = area2score(mean_area) * severity  # NBC
            area = area.squeeze(-1)  # NB

            preds = F.one_hot(preds, num_classes=num_classes).float().mean(0).cpu().numpy()  # BCHW4

        elif "patches" in batch.keys():
            gt_kappa = np.array([1.0])
            assert self.cfg.test.batch_size == 1

            # Load Batch
            patches = batch["patches"][0]  # NyNx3HW
            mask = batch["mask"][0]  # 1HW
            pad = batch["pad_width"][0]

            # Inference with patch batch
            nx = patches.shape[1]
            preds = torch.zeros(
                (
                    8,
                    self.cfg.model.num_classes,
                    (patches.shape[0] - 1) * self.step + 256,
                    (patches.shape[1] - 1) * self.step + 256,
                    num_classes,
                ),
                dtype=torch.float,
            ).to(
                patches.device
            )  # NCHW

            window = self.window.to(patches.device)
            norm = torch.zeros_like(preds).to(patches.device)
            patches = patches.reshape((-1,) + patches.shape[2:])  # N3HW

            N = patches.shape[0]
            bs = self.cfg.test.max_num_patches

            cnt = 0
            for rot in range(4):
                for flip in [True, False]:
                    img = torch.rot90(patches, rot, dims=(2, 3))
                    if flip:
                        img = torch.flip(img, dims=(3,))
                    for i in range(0, N, bs):
                        end = min(i + bs, N)
                        pred = self.model.sample(img[i:end], 1, mean=True)[0]

                        if flip:
                            pred = torch.flip(pred, dims=(3,))
                        pred = torch.rot90(pred, -rot, dims=(2, 3)) * window

                        for j in range(end - i):
                            y = (i + j) // nx * self.step
                            x = (i + j) % nx * self.step
                            preds[cnt, :, y : y + 256, x : x + 256] += pred[j]
                            norm[cnt, :, y : y + 256, x : x + 256] += window

                    cnt += 1

            preds = preds / norm  # NCHW4

            # Unpad
            preds = (preds[:, :, pad[0, 0] : -pad[0, 1], pad[1, 0] : -pad[1, 1]] * mask.unsqueeze(-1)).argmax(-1)

            # Prediction; area and severity scores
            area = (
                (preds.reshape(preds.shape[:2] + (-1,))[:, :, mask.flatten().bool()].sum(1, keepdims=True) > 1)
                .float()
                .mean(-1)
                .cpu()
                .numpy()
            )  # N1
            kappa = np.array([1.0])
            mean_area = area.mean(0, keepdims=True)
            severity = np.nan_to_num(
                preds.reshape(preds.shape[:2] + (-1,))[:, :, mask.flatten().bool()].float().mean(-1).cpu().numpy() / mean_area
            )  # NC
            severity = 0.5 * (severity + severity.mean(1, keepdims=True))
            easi = (area2score(mean_area) * severity)[:, np.newaxis, :]  # N1C
            severity = severity[:, np.newaxis, :]  # N1C
            preds = F.one_hot(preds, num_classes=num_classes).float().mean(0, keepdims=True).cpu().numpy()  # BCHW4

        else:
            raise ValueError

        gt_easi = (gt_area[np.newaxis, ...] * gt_severity[:, np.newaxis, :, :]).reshape((-1,) + gt_severity.shape[1:])  # NBC
        gt_area = gt_area.squeeze(-1)  # NB

        area = area2score(area)
        severity = severity * area[:, :, np.newaxis].astype(bool)  # NBC

        # Gather results
        self.gt_area.append(gt_area)
        self.gt_severity.append(gt_severity)
        self.gt_kappa.append(gt_kappa)
        self.gt_easi.append(gt_easi)
        self.area.append(area)
        self.severity.append(np.nan_to_num(severity))
        self.kappa.append(kappa)
        self.easi.append(easi)

        # visualization
        B, C, H, W = preds.shape[:-1]
        gt = (
            F.one_hot(
                seg.reshape((B, 1, 1, -1, H, W)) * grade.reshape((B, C, -1, 1, 1, 1)),
                num_classes=num_classes,
            )
            .float()
            .mean((2, 3))
            .cpu()
            .numpy()
        )  # BCHW4

        # Add mean of gt and preds
        gt = np.concatenate([gt, gt.mean(1, keepdims=True)], axis=1)
        preds = np.concatenate([preds, preds.mean(1, keepdims=True)], axis=1)

        font_size = max(H // 256, 1)
        for i in range(B):
            resized_ori = cv2.resize(ori_img[i], (W, H))

            # Pred summary
            pred_img = self.vis_pred(resized_ori, preds[i])
            text = self.write_summary(area[:, i], severity[:, i])
            for j, t in enumerate(text):
                cv2.putText(
                    pred_img,
                    t,
                    (10, 25 * (j + 1) * font_size),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size / 2,
                    (255, 90, 0),
                    font_size,
                    cv2.LINE_AA,
                )

            # GT summary
            gt_img = self.vis_gt(resized_ori, gt[i])
            text = self.write_summary(gt_area[:, i], gt_severity[:, i])
            for j, t in enumerate(text):
                cv2.putText(
                    gt_img,
                    t,
                    (10, 25 * (j + 1) * font_size),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size / 2,
                    (255, 90, 0),
                    font_size,
                    cv2.LINE_AA,
                )

            # Concatenate prediction and GT and save it.
            cv2.imwrite(
                f"results/{self.exp_name}/{file_name[i]}",
                np.concatenate((pred_img, gt_img), axis=0),
            )

            # Write results

    def on_test_end(self):
        self.easi = np.concatenate(self.easi, axis=1)  # NBC
        with open(f"results/{self.exp_name}/results.pkl", "wb") as f:
            pickle.dump(
                {
                    "gt_area": np.concatenate(self.gt_area, axis=1),  # NB
                    "gt_severity": np.concatenate(self.gt_severity, axis=1),  # NBC
                    "gt_kappa": np.concatenate(self.gt_kappa),  # B
                    "gt_easi": np.concatenate(self.gt_easi, axis=1),  # NBC
                    "area": np.concatenate(self.area, axis=1),  # NB
                    "severity": np.concatenate(self.severity, axis=1),  # NBC
                    "kappa": np.concatenate(self.kappa),  # B
                    "easi": self.easi,
                },
                f,
            )

        pd.DataFrame(
            self.easi.mean(0), columns=["Erythema", "Edema/Papulation", "Excoriation", "Lichenification"], index=self.file_names
        ).to_excel(f"results/{self.exp_name}/easi.xlsx")

    def vis_gt(self, img, gt):
        return np.concatenate(
            (
                np.ones_like(img) * 255,
                np.swapaxes(
                    np.swapaxes(heatmap(img, gt), 1, 2).reshape(-1, img.shape[0], 3),
                    0,
                    1,
                ),
            ),
            axis=1,
        )

    def vis_pred(self, img, pred):
        return np.concatenate(
            (
                img,
                np.swapaxes(
                    np.swapaxes(heatmap(img, pred), 1, 2).reshape(-1, img.shape[0], 3),
                    0,
                    1,
                ),
            ),
            axis=1,
        )

    def write_summary(self, area, severity):
        (
            area_mean,
            area_std,
            severity_mean,
            severity_std,
            EASI_mean,
            EASI_std,
        ) = cal_EASI(area, severity)
        severity_lower_bound = np.clip(severity_mean - severity_std, 0, 3)
        severity_upper_bound = np.clip(severity_mean + severity_std, 0, 3)
        return [
            f"Area score : {max(area_mean-area_std, 0):.2f} - {min(area_mean+area_std, 6):.2f}",
            f"Erythema : {severity_lower_bound[0]:.2f} - {severity_upper_bound[0]:.2f}",
            f"Induration : {severity_lower_bound[1]:.2f} - {severity_upper_bound[1]:.2f}",
            f"Excoriation : {severity_lower_bound[2]:.2f} - {severity_upper_bound[2]:.2f}",
            f"Lichenification : {severity_lower_bound[3]:.2f} - {severity_upper_bound[3]:.2f}",
            f"EASI : {max(EASI_mean-EASI_std, 0):.2f} - {min(EASI_mean+EASI_std, 72):.2f}",
        ]


if __name__ == "__main__":
    # env
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    torch.set_float32_matmul_precision("high")

    # arguments
    parser = ArgumentParser()
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--phase", type=str, choices=["train", "test", "predict"])
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--cfg", type=str, default="config.yaml")
    parser.add_argument("--test-dataset", type=str, default="19_int")
    args = parser.parse_args()
    cfg = load_config(args.cfg)

    # dataset
    atopy = AtopyDataModule(cfg, args.test_dataset)

    # model
    model = HierarchicalProbUNet(
        latent_dims=cfg.model.latent_dims,
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        loss_kwargs=dict(cfg.train.loss_kwargs),
        num_cuts=cfg.model.num_cuts,
        weights=cfg.model.weights,
    )

    if args.checkpoint:
        litmodel = LitModel.load_from_checkpoint(args.checkpoint, model=model, cfg=cfg, map_location=lambda storage, loc: storage.cuda(1))
    else:
        litmodel = LitModel(model, cfg=cfg)

    # train
    if args.phase == "train":
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_last=True)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        trainer = pl.Trainer(
            max_epochs=cfg.train.max_epochs,
            devices=args.devices,
            accelerator="gpu",
            strategy="ddp_find_unused_parameters_true",
            callbacks=[checkpoint_callback, lr_monitor],
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
            gradient_clip_val=0.5,
        )
        trainer.fit(litmodel, datamodule=atopy)
        shutil.copy("config.yaml", f"{trainer.logger.log_dir}/config.yaml")
    # test
    else:
        litmodel.exp_name = args.test_dataset
        os.makedirs("results/" + args.test_dataset, exist_ok=True)
        trainer = pl.Trainer(devices=[args.devices])
        if args.phase == "test":
            trainer.test(litmodel, datamodule=atopy)
        else:
            trainer.predict(litmodel, datamodule=atopy)
