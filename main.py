import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from argparse import ArgumentParser
import os
import cv2
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
from datetime import datetime
import shutil

from loader import get_dataset
from model.model import HierarchicalProbUNet
from model.model_utils import log_cumulative
from utils import *


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
        return DataLoader(self.test, batch_size=self.cfg.test.batch_size, num_workers=self.cfg.num_workers, pin_memory=True)

    # def predict_dataloader(self):
    #     return DataLoader(self.predict, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True)


class LitModel(pl.LightningModule):
    def __init__(self, model, cfg, exp_name=None):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.mc_n = cfg.test.mc_n
        self.exp_name = exp_name
        self.automatic_optimization = False
        self.cfg.exp_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.cfg.train.gamma)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=self.cfg.train.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_start(self):
        self.logger.log_hyperparams(dict(self.cfg.train))

    def training_step(self, batch, batch_idx):
        seg = batch["seg"]  # B x labeller x H x W
        img = batch["img"]  # B x C x H x W
        grade = batch["grade"]  # B x C x cls_labellers
        opt = self.optimizers()

        summaries = None
        for i in range(seg.shape[1]):
            for j in range(grade.shape[-1]):
                # B1HW, BC11 -> BCHW
                loss_dict = self.model.sum_loss(seg[:, [i]] * grade[:, :, [j], None], img)

                # Backward
                opt.zero_grad()
                self.manual_backward(loss_dict["supervised_loss"])
                self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
                opt.step()

                if summaries is None:
                    summaries = loss_dict["summaries"]
                else:
                    for k, v in loss_dict["summaries"].items():
                        summaries[k] += v
        for k, v in summaries.items():
            summaries[k] = v / seg.shape[1] / grade.shape[-1]

        self.log_dict(
            {"train_" + k: v for k, v in summaries.items()}, sync_dist=True, on_epoch=True, on_step=False, batch_size=seg.shape[0]
        )

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        sch.step()

    def validation_step(self, batch, batch_idx):
        seg = batch["seg"]  # B x labeller x H x W
        img = batch["img"]  # B x C x H x W
        grade = batch["grade"]  # B x C x cls_labellers
        summaries = None
        for i in range(seg.shape[1]):
            for j in range(grade.shape[-1]):
                # B1HW, BC11 -> BCHW
                loss_dict = self.model.sum_loss(seg[:, [i]] * grade[:, :, [j], None], img)
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
        self.EIExL = []
        self.EIExL_std = []
        self.gt_EIExL = []
        self.gt_EIExL_std = []

    def test_step(self, batch, batch_idx):
        seg = batch["seg"]  # B x labeller x H x W
        img = batch["img"]  # B x 3 x H x W
        grade = batch["grade"]  # B x C x cls_labellers
        ori_img = batch["ori_img"]
        file_name = batch["file_name"]

        H, W = img.shape[2:]
        B, C = grade.shape[:2]
        num_classes = self.cfg.model.num_classes

        preds = self.model.sample(img, self.mc_n)  # N x B x H x W x C
        gt = seg.reshape((B, 1, 1, -1, H, W)) * grade.reshape((B, C, -1, 1, 1, 1))  # B x C x _ x _ H x W
        gt_EIExL_std = gt.float().std((2, 3)).mean((2, 3))  # B x C
        gt = F.one_hot(gt, num_classes=num_classes).float().mean((2, 3))  # B x C x H x W x 4
        area = preds.bool().float().mean((2, 3))
        EIExL = preds.float().mean((2, 3)) * area2score(area) / area  # N x B x C
        EIExL_std = preds.float().std(0).mean((1, 2))  # B x C
        gt_EIExL = (
            (grade.unsqueeze(-1) * area2score(seg.float().mean((2, 3))).reshape((B, 1, 1, -1))).reshape((B, C, -1)).permute(2, 0, 1)
        )  # N x B x C

        self.EIExL.append(np.nan_to_num(EIExL.cpu().numpy()))
        self.EIExL_std.append(EIExL_std.cpu().numpy())
        self.gt_EIExL.append(gt_EIExL.cpu().numpy())
        self.gt_EIExL_std.append(gt_EIExL_std.cpu().numpy())

        # visualization
        ori_img = ori_img.cpu().numpy()
        preds = (F.one_hot(preds, num_classes=self.cfg.model.num_classes).float().mean(0)).permute(0, 3, 1, 2, 4)  # B x C x H x W x 4
        for i in range(B):
            resized_ori = cv2.resize(ori_img[i], (256, 256))
            pred_img = np.concatenate(
                (resized_ori, np.swapaxes(np.swapaxes(heatmap(resized_ori, preds[i]), 1, 2).reshape(-1, 256, 3), 0, 1)), axis=1
            )
            gt_img = np.concatenate(
                (
                    np.ones_like(resized_ori) * 255,
                    np.swapaxes(np.swapaxes(heatmap(resized_ori, gt[i]), 1, 2).reshape(-1, 256, 3), 0, 1),
                ),
                axis=1,
            )
            cv2.imwrite(f"results/{self.exp_name}/{file_name[i]}", np.concatenate((pred_img, gt_img), axis=0))

    def on_test_end(self):
        with open(f"results/{self.exp_name}/results.pkl", "wb") as f:
            pickle.dump(
                {
                    "EIExL": np.concatenate(self.EIExL, axis=1),  # N x B x C
                    "EIExL_std": np.concatenate(self.EIExL_std, axis=0),  # B x C
                    "gt_EIExL": np.concatenate(self.gt_EIExL, axis=1),  # N x B x C
                    "gt_EIExL_std": np.concatenate(self.gt_EIExL_std, axis=0),  # B x C
                },
                f,
            )

    def on_predict_start(self):
        self.logger.log_hyperparams(dict(self.cfg.predict))
        self.step = self.cfg.predict.step

        # Window for smooth blending
        window = np.ones(256)
        window[: 256 - self.step] = np.arange(256 - self.step) / self.step
        window = np.tile(window[:, np.newaxis], (1, 256))
        self.window = window * np.rot90(window) * np.rot90(window, 2) * np.rot90(window, 3)

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
    )

    # train
    if args.phase == "train":
        litmodel = LitModel(model, cfg=cfg)
        checkpoint_callback = ModelCheckpoint(monitor="val_loss_mean")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        trainer = pl.Trainer(
            max_epochs=cfg.train.max_epochs,
            devices=args.devices,
            accelerator="gpu",
            strategy="ddp_find_unused_parameters_true",
            callbacks=[checkpoint_callback, lr_monitor],
            log_every_n_steps=1,
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
