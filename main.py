import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

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
import scipy.signal
import pandas as pd

from loader import get_dataset
from model.model import HierarchicalProbUNet
from utils import *


class AtopyDataModule(pl.LightningDataModule):
    def __init__(self, cfg, test_dataset, synthetic):
        super().__init__()
        self.cfg = cfg
        self.test_dataset = test_dataset
        self.synthetic = synthetic

    def setup(self, stage):
        self.roi_train, self.roi_val = get_dataset("roi_train", wholebody=self.cfg.train.wholebody, synthetic=self.synthetic)
        if self.cfg.train.wholebody:
            self.wb_train, self.wb_val = get_dataset("wb_train")
        else:
            self.wb_train, self.wb_val = None, None

        self.test = get_dataset(self.test_dataset, step_size=self.cfg.test.step, synthetic=self.synthetic)

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

    def predict_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=1,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
        )


class LitModel(pl.LightningModule):
    def __init__(self, model, cfg, exp_name=None, synthetic=False):
        super().__init__()

        self.model = model
        self.cfg = cfg
        self.mc_n = cfg.test.mc_n
        self.exp_name = exp_name
        self.synthetic = synthetic
        self.cfg.exp_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        self.step = self.cfg.test.step

        # 2D cosine window function for smooth joining
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
        """
        This method is called at the beginning of the training process.
        It logs the hyperparameters used for training.
        """
        self.logger.log_hyperparams(dict(self.cfg.train))

    def training_step(self, batch, batch_idx):
        return self.forward_step("train", batch)
    def validation_step(self, batch, batch_idx):
        self.forward_step("val", batch)

    def on_test_start(self):
        # Results to be saved
        self.logger.log_hyperparams(dict(self.cfg.test))
        self.gt_area = []
        self.gt_severity = []
        self.gt_easi = []
        self.area = []
        self.severity = []
        self.easi = []
        self.file_names = []
        self.ged = []

    def test_step(self, batch, batch_idx):
        # Load batch
        seg = batch["seg"]  # BLHW
        gt_area = batch["area"]  # BL
        grade = batch["grade"]  # BL'C
        ori_img = batch["ori_img"].cpu().numpy()
        file_name = batch["file_name"]
        self.file_names.extend(file_name)
        num_classes = self.cfg.model.num_cuts + 1

        B, _, H, W = seg.shape
        C = grade.shape[-1]

        # Ground Truth
        gt_easi = (gt_area[:, :, None, None] * grade[:, None, :, :]).reshape(B, -1, C).cpu().numpy()  # BNC
        gt_area = gt_area.squeeze(-1).cpu().numpy()  # BL
        gt_severity = grade.cpu().numpy()  # BL'C

        gt = (seg.reshape((B, -1, 1, 1, H, W)) * grade.reshape((B, 1, -1, C, 1, 1))).reshape((B, -1, C, H, W))  # BNCHW

        # Prediction
        if "img" in batch.keys():  # Only ROI images
            area, severity, easi, preds = self.inference_patch(batch)

            self.ged.append(generalized_energy_distance(gt, preds))

            if self.synthetic == True:
                dSY = torch.zeros(B, self.mc_n).to(gt.device)
                for i in range(gt.shape[1]):
                    for j in range(self.mc_n):
                        dSY[:, j] += (gt[:, i] - preds[:, j]).float().abs().mean((1, 2, 3))
                topks = torch.topk(dSY, k=self.cfg.test.topk, dim=-1, largest=False)[1]  # Bk
                for i in range(preds.shape[0]):
                    np.save(f"results/synthetic/{file_name[i].split('.')[0] + '.npy'}", preds[i, topks[i]].cpu().numpy())
                return

            preds = F.one_hot(preds, num_classes=num_classes).float().mean(1).cpu().numpy()  # BCHW4
        elif "patches" in batch.keys():
            area, severity, easi, preds = self.inference_wholebody(batch, num_classes)
            preds = F.one_hot(preds, num_classes=num_classes).float().mean(0, keepdim=True).cpu().numpy()  # NCHW4
            self.ged.append(np.zeros(B))
        else:
            raise ValueError
        gt = F.one_hot(gt, num_classes=num_classes).float().mean(1).cpu().numpy()  # BCHW4

        # Gather results
        self.gt_area.append(gt_area)
        self.gt_severity.append(gt_severity)
        self.gt_easi.append(gt_easi)
        self.area.append(area)
        self.severity.append(np.nan_to_num(severity))
        self.easi.append(easi)

        # Add mean of gt and preds by class
        gt = np.concatenate([gt, gt.mean(1, keepdims=True)], axis=1)
        preds = np.concatenate([preds, preds.mean(1, keepdims=True)], axis=1)

        # Visualization
        font_size = max(H // 256, 1)
        for i in range(B):
            resized_ori = cv2.resize(ori_img[i], (W, H))

            # Pred summary
            pred_img = self.vis_pred(resized_ori, preds[i])
            text = self.write_summary(area[i], severity[i], easi[i])
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
            text = self.write_summary(gt_area[i], gt_severity[i], gt_easi[i])
            for j, t in enumerate(text):
                cv2.putText(
                    gt_img, t, (10, 25 * (j + 1) * font_size), cv2.FONT_HERSHEY_SIMPLEX, font_size / 2, (255, 90, 0), font_size, cv2.LINE_AA
                )

            # Concatenate prediction and GT and save it.
            cv2.imwrite(f"results/{self.exp_name}/{file_name[i]}", np.concatenate((pred_img, gt_img), axis=0))

    def on_test_end(self):
        # Write results
        self.easi = np.concatenate(self.easi)  # BNC
        with open(f"results/{self.exp_name}/results.pkl", "wb") as f:
            pickle.dump(
                {
                    "gt_area": np.concatenate(self.gt_area),  # BN
                    "gt_severity": np.concatenate(self.gt_severity),  # BNC
                    "gt_easi": np.concatenate(self.gt_easi),  # BNC
                    "area": np.concatenate(self.area),  # BN
                    "severity": np.concatenate(self.severity),  # BNC
                    "easi": self.easi,
                    "ged": np.concatenate(self.ged),  # B
                },
                f,
            )

        pd.DataFrame(
            self.easi.mean(1), columns=["Erythema", "Edema/Papulation", "Excoriation", "Lichenification"], index=self.file_names
        ).to_excel(f"results/{self.exp_name}/easi.xlsx")

    def predict_step(self, batch, batch_idx):
        # Load batch
        ori_img = batch["ori_img"][0].cpu().numpy()
        file_name = batch["file_name"][0]
        num_classes = self.cfg.model.num_cuts + 1

        # Prediction
        if not self.cfg.test.wholebody:
            area, severity, easi, preds = self.inference_patch(batch)
            preds = F.one_hot(preds, num_classes=num_classes).float()[0].cpu().numpy()  # NCHW4
        else:
            area, severity, easi, preds = self.inference_wholebody(batch, num_classes)
            preds = F.one_hot(preds, num_classes=num_classes).float().cpu().numpy()  # NCHW4

        # visualization
        N, C, H, W = preds.shape[:-1]

        # Add mean of gt and preds
        preds = np.concatenate([preds, preds.mean(1, keepdims=True)], axis=1)

        for i in range(N):
            hm_img = heatmap(cv2.resize(ori_img, (W, H)), preds[i])
            for j, sign in enumerate(["e", "i", "ex", "l", "mean"]):
                cv2.imwrite(f"figure/images/{file_name.split('.')[0]}_{sign}_{i}.jpg", hm_img[j])

    def forward_step(self, phase, batch):
        if isinstance(batch, list):  # Use whole-body images and ROI images jointly.
            batch, wb_batch = batch

            wb_img = wb_batch["img"]  # BN3HW
            wb_seg = wb_batch["seg"]  # BNHW
            wb_img = torch.flatten(wb_img, 0, 1)  # (BN)3HW
            wb_seg = torch.flatten(wb_seg, 0, 1)  # (BN)HW

        img = batch["img"]  # B3HW
        seg = batch["seg"][:, None, :, None, :, :]  # B1L1HW
        grade = batch["grade"][:, :, None, :, None, None]  # BL'1C11

        seg = torch.flatten(seg * grade, 1, 2)  # BNCHW
        if "syn" in batch.keys():
            seg = torch.cat([seg, batch["syn"]], axis=1)  # BN'CHW
        img = torch.flatten(img.unsqueeze(1).expand(-1, seg.shape[1], -1, -1, -1), 0, 1)  # (BN)3HW
        seg = torch.flatten(seg, 0, 1)  # (BN)CHW

        # Segmentation and grade labels of an image are intended to be fed into the model at the same time,
        # to learn diverse possibilities that the image possesses.
        if isinstance(batch, list):
            img = torch.cat([img, wb_img])
            seg = torch.cat([seg, wb_seg])

        loss_dict = self.model.sum_loss(seg, img)
        self.log_dict(
            {f"{phase}_" + k: v for k, v in loss_dict["summaries"].items()},
            sync_dist=True,
            on_epoch=True,
            on_step=False,
            batch_size=seg.shape[0],
        )
        return loss_dict["supervised_loss"]

    def inference_patch(self, batch):
        """
        Perform inference on a batch of patches.

        Args:
            batch (dict): A dictionary containing the input data for the model. 
                          It should have the key "img" for the input images.
                          - "img" (torch.Tensor): Tensor of shape (B, 3, H, W) representing the input images.

        Returns:
            tuple: A tuple containing the following elements:
                - area (torch.Tensor): Tensor of shape (B, L) representing the area scores.
                - severity (torch.Tensor): Tensor of shape (B, L, C) representing the severity scores.
                - easi (torch.Tensor): Tensor of shape (B, L, C) representing the EASI scores.
                - preds (torch.Tensor): Tensor of shape (B, N, C, H, W) representing the predictions.
        """
        preds = self.model.sample(batch["img"], self.mc_n).argmax(-1)  # BNCHW
        area, severity, easi = cal_EASI(preds.flatten(3, 4))
        return area, severity, easi, preds

    def inference_wholebody(self, batch, num_classes):
        """
        Perform inference on a batch of whole-body images.

        Args:
            batch (dict): A dictionary containing the input data for the model. 
                          It should have the keys "patches", "mask", and "pad_width".
                          - "patches" (torch.Tensor): Shape (Ny, Nx, 3, H, W) where Ny and Nx are the number of patches in y and x directions.
                          - "mask" (torch.Tensor): Shape (1, H, W) where H and W are the height and width of the whole-body image.
                          - "pad_width" (torch.Tensor): Shape (2, 2) containing the padding widths for height and width dimensions.
            num_classes (int): The number of classes for segmentation.

        Returns:
            tuple: A tuple containing the area, severity, EASI scores, and predictions.
                   - area (torch.Tensor): Shape (1, N) where N is the number of patches.
                   - severity (torch.Tensor): Shape (1, N, C) where C is the number of classes.
                   - easi (torch.Tensor): Shape (1, N, C) where C is the number of classes.
                   - preds (torch.Tensor): Shape (N, C, H, W) where N is the number of patches, C is the number of classes, and H, W are the height and width of the whole-body image.
        """
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

        # Do an inference step with 8 different orientiations to reduce possible variablity
        cnt = 0
        for rot in range(4):
            for flip in [True, False]:
                img = torch.rot90(patches, rot, dims=(2, 3))
                if flip:
                    img = torch.flip(img, dims=(3,))
                for i in range(0, N, bs):
                    end = min(i + bs, N)
                    pred = self.model.sample(img[i:end], 1, mean=True).squeeze(1)  # BCHW4

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
        preds = (preds[:, :, pad[0, 0] : -pad[0, 1], pad[1, 0] : -pad[1, 1]] * mask.unsqueeze(-1)).argmax(-1)  # NCHW

        # Prediction; area and severity scores
        area, severity, easi = cal_EASI(preds.reshape(preds.shape[:2] + (-1,))[:, :, mask.flatten().bool()].unsqueeze(0))
        return area, severity, easi, preds

    def vis_gt(self, img, gt):
        return np.concatenate(
            (
                np.ones_like(img) * 255,
                np.swapaxes(np.swapaxes(heatmap(img, gt), 1, 2).reshape(-1, img.shape[0], 3), 0, 1),
            ),
            axis=1,
        )

    def vis_pred(self, img, pred):
        return np.concatenate(
            (img, np.swapaxes(np.swapaxes(heatmap(img, pred), 1, 2).reshape(-1, img.shape[0], 3), 0, 1)),
            axis=1,
        )

    def write_summary(self, area, severity, easi):
        """
        Generate a summary of the area, severity, and EASI scores.

        Args:
            area (numpy.ndarray): Array of area scores.
            severity (numpy.ndarray): Array of severity scores.
            easi (numpy.ndarray): Array of EASI scores.

        Returns:
            list: A list of strings summarizing the area, severity, and EASI scores.
        """
        area_mean, area_std = area.mean(), area.std()
        severity_mean, severity_std = severity.mean(0), severity.std(0)
        easi_mean, easi_std = easi.mean(), easi.std()
        severity_lower_bound = np.clip(severity_mean - severity_std, 0, 3)
        severity_upper_bound = np.clip(severity_mean + severity_std, 0, 3)
        return [
            f"Area score : {max(area_mean-area_std, 0):.2f} - {min(area_mean+area_std, 6):.2f}",
            f"Erythema : {severity_lower_bound[0]:.2f} - {severity_upper_bound[0]:.2f}",
            f"Induration : {severity_lower_bound[1]:.2f} - {severity_upper_bound[1]:.2f}",
            f"Excoriation : {severity_lower_bound[2]:.2f} - {severity_upper_bound[2]:.2f}",
            f"Lichenification : {severity_lower_bound[3]:.2f} - {severity_upper_bound[3]:.2f}",
            f"EASI : {max(easi_mean-easi_std, 0):.2f} - {min(easi_mean+easi_std, 72):.2f}",
        ]


if __name__ == "__main__":
    # env
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.enabled = False   # The input data is not of contiguous type

    # arguments
    parser = ArgumentParser()
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--phase", type=str, choices=["train", "test", "predict"])
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--cfg", type=str, default="config.yaml")
    parser.add_argument("--test-dataset", type=str, default="19_int")
    parser.add_argument("--synthetic", action="store_true", help="Use/make synthetic data")
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    cfg.test.wholebody = True if args.test_dataset in ["20_int", "20_ext", "wb_predict"] else False

    # dataset
    atopy = AtopyDataModule(cfg, args.test_dataset, args.synthetic)

    # model
    model = HierarchicalProbUNet(
        latent_dims=cfg.model.latent_dims,
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        loss_kwargs=dict(cfg.train.loss_kwargs),
        num_cuts=cfg.model.num_cuts,
        weights=cfg.model.weights,
        imbalance=cfg.model.imbalance
    )

    if args.checkpoint:
        litmodel = LitModel.load_from_checkpoint(
            args.checkpoint, synthetic=args.synthetic, model=model, cfg=cfg, map_location=lambda storage, loc: storage.cuda()
        )
    else:
        litmodel = LitModel(model, synthetic=args.synthetic, cfg=cfg)

    # train
    if args.phase == "train":
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_last=True)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        trainer = pl.Trainer(
            max_epochs=cfg.train.max_epochs,
            devices=args.devices,
            accelerator="gpu",
            strategy="ddp",
            callbacks=[checkpoint_callback, lr_monitor],
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
            gradient_clip_val=0.5,
        )
        trainer.fit(litmodel, datamodule=atopy)
        shutil.copy("config.yaml", f"{trainer.logger.log_dir}/config.yaml")
    # test
    else:
        trainer = pl.Trainer(devices=[args.devices])
        if args.phase == "test":
            litmodel.exp_name = args.test_dataset
            if cfg.test.wholebody:
                cfg.test.batch_size = 1
            os.makedirs("results/" + args.test_dataset, exist_ok=True)
            trainer.test(litmodel, datamodule=atopy)
        else:
            os.makedirs("figure/images", exist_ok=True)
            trainer.predict(litmodel, datamodule=atopy)
