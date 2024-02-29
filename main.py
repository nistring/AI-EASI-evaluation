import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from argparse import ArgumentParser
import os
import cv2
import numpy as np
from datetime import datetime
import shutil
import pickle
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

from loader import get_dataset
from model.model import HierarchicalProbUNet
from model.model_utils import log_cumulative
from utils import *
from patchify import patchify


class AtopyDataModule(pl.LightningDataModule):
    def __init__(self, cfg, test_dataset):
        super().__init__()
        self.cfg = cfg
        self.test_dataset = test_dataset

    def setup(self, stage):
        self.train, self.val = get_dataset("train")
        # self.test = get_dataset("intra")
        self.test = get_dataset(self.test_dataset)
        # self.predict = CustomDataset("data/predict", step=self.cfg.predict.step)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.train.batch_size, num_workers=self.cfg.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.cfg.test.batch_size, num_workers=self.cfg.num_workers, pin_memory=False)

    # def predict_dataloader(self):
    #     return DataLoader(self.predict, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True)


class LitModel(pl.LightningModule):
    def __init__(self, model, cfg, exp_name=None):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.mc_n = cfg.test.mc_n
        self.exp_name = exp_name
        self.cfg.exp_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        self.step = self.cfg.test.step

        # Window for smooth blending
        window = torch.ones(256)
        window[: 256 - self.step] = torch.arange(256 - self.step) / self.step
        window = torch.tile(window[:, None], (1, 256))
        self.window = (window * torch.rot90(window) * torch.rot90(window, 2) * torch.rot90(window, 3))[..., None]  # 256 x 256 x 1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.cfg.train.gamma)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=self.cfg.train.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_start(self):
        self.logger.log_hyperparams(dict(self.cfg.train))

    def training_step(self, batch, batch_idx):
        seg = batch["seg"]  # BlHW (l : labeller)
        seg = seg[:, [self.current_epoch % seg.shape[1]]]
        img = batch["img"]  # BCHW
        grade = batch["grade"]  # BCl
        grade = grade[:, :, [self.current_epoch % grade.shape[2]], None]

        loss_dict = self.model.sum_loss(seg, grade, img)

        self.log_dict(
            {"train_" + k: v for k, v in loss_dict["summaries"].items()},
            sync_dist=True,
            on_epoch=True,
            on_step=False,
            batch_size=seg.shape[0],
        )

        return loss_dict["supervised_loss"]

    def validation_step(self, batch, batch_idx):
        seg = batch["seg"]  # B x labeller x H x W
        img = batch["img"]  # B x C x H x W
        grade = batch["grade"]  # B x C x cls_labellers
        summaries = None
        for i in range(seg.shape[1]):
            for j in range(grade.shape[-1]):
                # B1HW, BC11 -> BCHW
                loss_dict = self.model.sum_loss(seg[:, [i]], grade[:, :, [j], None], img)
                if summaries is None:
                    summaries = loss_dict["summaries"]
                else:
                    for k, v in loss_dict["summaries"].items():
                        summaries[k] += v
        for k, v in summaries.items():
            summaries[k] = v / seg.shape[1] / grade.shape[-1]

        self.log_dict({"val_" + k: v for k, v in summaries.items()}, sync_dist=True, on_epoch=True, on_step=False, batch_size=seg.shape[0])

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

    def test_step(self, batch, batch_idx):
        # Load batch
        seg = batch["seg"]  # B x L x H x W
        grade = batch["grade"]  # B x C x L'
        ori_img = batch["ori_img"].cpu().numpy()
        file_name = batch["file_name"]

        # Ground Truth
        gt_area = seg.reshape(seg.shape[:2] + (-1,)).permute(1, 0, 2).cpu().numpy()  # N x B x (H x W)
        # gt_kappa = np.nan_to_num(np.array([fleiss_kappa(aggregate_raters(gt_area[:, i].T)[0]) for i in range(gt_area.shape[1])]), nan=1.) # Complete agreement : k = 1
        gt_kappa = np.array([1.])
        gt_area = gt_area.astype(np.float32).mean(-1, keepdims=True)  # N x B x 1
        gt_severity = grade.permute(2, 0, 1).cpu().numpy()  # N x B x C

        if "img" in batch.keys():
            preds = self.model.sample(batch["img"], self.mc_n).argmax(-1)  # N x B x C x H x W

            # Prediction; area and severity scores
            area = preds.sum(2).bool().reshape(preds.shape[:2] + (-1,)).cpu().numpy()  # N x B x (H x W)
            kappa = np.nan_to_num(np.array([fleiss_kappa(aggregate_raters(area[:, i].T)[0]) for i in range(area.shape[1])]), nan=1.)
            area = area.astype(np.float32).mean(-1, keepdims=True)  # N x B x 1
            severity = preds.float().mean((3, 4)).cpu().numpy() / area  # N x B x C
            easi = area2score(area) * severity  # N x B x C
            area = area.squeeze(-1)  # N x B

        elif "patches" in batch.keys():
            assert self.cfg.test.batch_size == 1

            # Load Batch
            patches = batch["patches"][0]  # Ny x Nx x C x h x w x 3
            mask = batch["mask"][0]  # 1 x H x W
            pad = batch["pad_width"][0]

            # Inference with patch batch
            nx = patches.shape[1]
            preds = torch.zeros(
                (
                    self.mc_n,
                    self.cfg.model.num_classes,
                    (patches.shape[0] - 1) * self.step + 256,
                    (patches.shape[1] - 1) * self.step + 256,
                    self.cfg.model.num_cuts + 1,
                ),
                dtype=torch.float,
            ).to(
                patches.device
            )  # NCHW x num_cuts+1

            window = self.window.to(patches.device)
            patches = patches.reshape((-1,) + patches.shape[2:])  # NChw3

            valid_patches = self.valid_patches(mask[0], pad[:2])
            N = len(valid_patches)
            bs = min(self.cfg.test.max_num_patches, N)
            for i in range(0, N, bs):
                end = min(i + bs, N)
                pred = self.model.sample(patches[valid_patches[i:end]], self.mc_n)  # NbChwc

                for j in range(end - i):
                    y = valid_patches[i + j] // nx * self.step
                    x = valid_patches[i + j] % nx * self.step
                    preds[:, :, y : y + 256, x : x + 256] += pred[:, j] * window

            preds = preds.argmax(-1)  # N x C x H x W
            # Unpad
            preds = preds[:, :, pad[0, 0] : -pad[0, 1], pad[1, 0] : -pad[1, 1]] * mask

            # Prediction; area and severity scores
            gt_area /= mask.float().mean().item()
            masked_pred = preds.reshape(preds.shape[:2] + (-1,))[:, :, mask.flatten().bool()] # N x C x _
            area = masked_pred.sum(1, keepdims=True).bool().float().mean(-1).cpu().numpy() # N1
            kappa = np.array([1.]) # np.nan_to_num(np.array([fleiss_kappa(aggregate_raters(area.T)[0])]), nan=1.)
            severity = masked_pred.float().mean(-1).cpu().numpy() / area  # NC
            easi = (area2score(area) * severity)[:, np.newaxis, :]  # N1C
            severity = severity[:, np.newaxis, :]  # N1C
            preds = preds.unsqueeze(1)
        else:
            raise ValueError

        gt_easi = (area2score(gt_area)[np.newaxis, ...] * gt_severity[:, np.newaxis, :, :]).reshape((-1,) + gt_severity.shape[1:])  # N x B x C
        gt_area = gt_area.squeeze(-1)  # N x B

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
        B, C, H, W = preds.shape[1:]
        num_classes = self.cfg.model.num_cuts + 1
        preds = F.one_hot(preds, num_classes=num_classes).float().mean(0).cpu().numpy()  # B x C x H x W x 4
        gt = (
            F.one_hot(seg.reshape((B, 1, 1, -1, H, W)) * grade.reshape((B, C, -1, 1, 1, 1)), num_classes=num_classes)
            .float()
            .mean((2, 3))
            .cpu()
            .numpy()
        )  # B x C x H x W x 4

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
            cv2.imwrite(f"results/{self.exp_name}/{file_name[i]}", np.concatenate((pred_img, gt_img), axis=0))

    def on_test_end(self):
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
                    "easi": np.concatenate(self.easi, axis=1),  # N x B x C
                },
                f,
            )

    def on_predict_start(self):
        self.logger.log_hyperparams(dict(self.cfg.predict))

    def predict_step(self, batch, batch_idx):
        patches = batch["patches"][0]
        img = batch["ori_img"].cpu().numpy().astype(np.uint8)[0]
        file_name = batch["file_name"][0]
        mask = batch["mask"][[0], :, :, 0].bool().cpu().numpy()

        # Sort out patches to inference
        nx = patches.shape[1]
        patches = patches.reshape(-1, patches.shape[2], patches.shape[3], patches.shape[4])
        bs = self.cfg.test.batch_size
        valid = []
        for i in range(patches.shape[0]):
            if not torch.all(patches[i] == 0):
                valid.append(i)

        # Assign severity and lesion area
        severities = np.zeros(
            (self.mc_n, patches.shape[1] * self.step + 256, patches.shape[2] * self.step + 256, self.cfg.model.num_classes)
        )
        lesion_area = np.zeros(severities.shape[:-1])

        # Inference with patch batch
        for i in range(len(valid) // bs + 1):
            area, logits = self.model.sample(patches[valid[i * bs : min((i + 1) * bs, len(valid))]], self.mc_n)
            for j in range(bs):
                y = valid[i * bs + j] // nx
                x = valid[i * bs + j] % nx
                severities[:, y : y + 256, x : x + 256] += logits[:, j].cpu().numpy() * self.window
                lesion_area[:, y : y + 256, x : x + 256] += area[:, j].cpu().numpy() * self.window

        # Post processing
        lesion_area = (lesion_area >= 0.5) * mask
        preds = log_cumulative(severities.reshape(-1, severities.shape[-1])).argmax(-1).reshape(severities.shape) * lesion_area
        severities = torch.div(preds.sum((2, 3)), lesion_area.sum((2, 3)))  # N x C
        areas = lesion_area.sum((2, 3)) / mask.sum()  # N
        easi = area2score(areas) * severities.sum(1)  # N

        # Visualization
        img = heatmap(img, preds)
        for i, text in enumerate(
            [
                f"Area: {areas.mean() * 100:.1f}%(+/-{area.std() * 100:.1f})",
                f"Erythema: {severities[:, 0].mean():.2f}(+/-{severities[:, 0].std():.2f})",
                f"Papulation: {severities[:, 1].mean():.2f}(+/-{severities[:, 1].std():.2f})",
                f"Excoriation: {severities[:, 2].mean():.2f}(+/-{severities[:, 2].std():.2f})",
                f"Lichenification: {severities[:, 3].mean():.2f}(+/-{severities[:, 3].std():.2f})",
                f"EASI: {easi.mean():.1f}(+/-{easi.std():.1f})",
            ],
            start=1,
        ):
            cv2.putText(
                img,
                text,
                (25, 25 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return img

    def valid_patches(self, mask: torch.Tensor, pad: torch.Tensor):
        """_summary_

        Args:
            mask (torch.Tensor): H x W

        Returns:
            _type_: _description_
        """
        mask_patches = patchify(np.pad(mask.cpu().numpy(), pad.cpu().numpy()), (256, 256), self.step).reshape((-1, 256, 256))
        valid = []
        for i in range(mask_patches.shape[0]):
            if np.any(mask_patches[i] != 0):
                valid.append(i)
        return valid

    def vis_gt(self, img, gt):
        return np.concatenate(
            (
                np.ones_like(img) * 255,
                np.swapaxes(np.swapaxes(heatmap(img, gt), 1, 2).reshape(-1, img.shape[0], 3), 0, 1),
            ),
            axis=1,
        )

    def vis_pred(self, img, pred):
        return np.concatenate((img, np.swapaxes(np.swapaxes(heatmap(img, pred), 1, 2).reshape(-1, img.shape[0], 3), 0, 1)), axis=1)

    def write_summary(self, area, severity):
        area_mean, area_std, severity_mean, severity_std, EASI_mean, EASI_std = cal_EASI(area, severity)
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
    parser.add_argument("--test-dataset", type=str, default="intra")
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
        weights=torch.Tensor(cfg.model.weights),
    )

    # train
    if args.phase == "train":
        litmodel = LitModel(model, cfg=cfg)
        checkpoint_callback = ModelCheckpoint(monitor="val_loss")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        trainer = pl.Trainer(
            max_epochs=cfg.train.max_epochs,
            devices=args.devices,
            accelerator="gpu",
            strategy="ddp_find_unused_parameters_true",
            callbacks=[checkpoint_callback, lr_monitor],
            check_val_every_n_epoch=10,
            log_every_n_steps=1,
            gradient_clip_val=0.5,
        )
        if args.checkpoint:
            trainer.fit(litmodel, datamodule=atopy, ckpt_path=args.checkpoint)
        else:
            trainer.fit(litmodel, datamodule=atopy)
        shutil.copy("config.yaml", f"{trainer.logger.log_dir}/config.yaml")
    # test
    else:
        exp_name = create_dir(args.phase)
        litmodel = LitModel.load_from_checkpoint(args.checkpoint, model=model, cfg=cfg, exp_name=exp_name)
        trainer = pl.Trainer(devices=[args.devices])
        if args.phase == "test":
            trainer.test(litmodel, datamodule=atopy)
        else:
            trainer.predict(litmodel, datamodule=atopy)
