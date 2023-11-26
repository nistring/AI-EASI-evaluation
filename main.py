import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)

from argparse import ArgumentParser
import os
import cv2
import numpy as np
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from copy import deepcopy
from datetime import datetime
import shutil

from loader import get_dataset, CustomDataset
from model.model import HierarchicalProbUNet
from utils import *
from scheduler import CosineAnnealingWarmUpRestarts

class AtopyDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage):
        self.train, self.val = get_dataset("Atopy_Segment_Train")
        self.test = get_dataset("Atopy_Segment_Test")
        # self.test = get_dataset("Atopy_Segment_Extra")
        self.predict = CustomDataset("data/predict", step=self.cfg.predict.step)

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

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True)


class LitModel(pl.LightningModule):
    def __init__(self, model, cfg, exp_name=None):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.mc_n = cfg.test.mc_n
        self.exp_name = exp_name
        self.cfg.exp_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0, weight_decay=self.cfg.train.weight_decay)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, self.cfg.train.T_0, eta_max=self.cfg.train.lr, T_up=self.cfg.train.T_up, gamma=self.cfg.train.gamma)
        return {"optimizer": optimizer,  "lr_scheduler": scheduler}

    def on_train_start(self):
        self.logger.log_hyperparams(dict(self.cfg.train))

    def training_step(self, batch, batch_idx):
        seg, img, grade, _ = batch

        mask = seg[:, [np.random.randint(seg.shape[1])]]
        loss_dict = self.model.sum_loss(
            torch.einsum("BLHW,BC->BCHW", mask, grade[..., np.random.randint(grade.shape[-1])]), img, lesion_area=mask
        )
        loss = loss_dict["supervised_loss"]
        summaries = loss_dict["summaries"]
        self.log_dict({"train_" + k: v for k, v in summaries.items()}, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seg, img, grade, _ = batch
        for i in range(seg.shape[1]):
            for j in range(grade.shape[-1]):
                loss_dict = self.model.sum_loss(torch.einsum("BLHW,BC->BCHW", seg[:, [i]], grade[..., j]), img, lesion_area=seg[:, [i]])
                loss = loss_dict["supervised_loss"]
                summaries = loss_dict["summaries"]

                self.log_dict({"val_" + k: v for k, v in summaries.items()}, sync_dist=True)

    def test_or_predict(self, img):

        preds, grades = self.model.sample(img, self.mc_n)
        # preds: (B x N x H x W)
        # grades: (B x C x num_cuts+1)
        weight = torch.Tensor([0.0, 0.1, 0.2, 0.3]).to(grades.device)
        mean_grade = (grades[:, :4] * weight / weight.sum()).sum((1, 2), keepdims=True)

        mean = torch.zeros_like(preds)
        mean[preds >= 0.5] = 1.0

        weighted_mean = mean.mean(1) * mean_grade
        std = mean.std(1)
        mean = mean.mean(1)

        entropy = preds.mean(1)
        entropy = -(entropy * torch.log(entropy) + (1 - entropy) * torch.log(1 - entropy))

        mutual_information = entropy + (preds * torch.log(preds) + (1 - preds) * torch.log(1 - preds)).mean(1)

        return mean, weighted_mean, std, entropy, mutual_information, grades

    def on_test_start(self):
        self.logger.log_hyperparams(dict(self.cfg.test))

        self.img = []
        self.gt = []
        self.gt_grade = []
        self.mean = []
        self.weighted_mean = []
        self.std = []
        self.entropy = []
        self.mi = []
        self.grade = []
        self.cohens_k = []

    def test_step(self, batch, batch_idx):
        seg, img, gt_grade, ori_img = batch

        seg = seg[0].bool().cpu().numpy()
        gt = np.ones_like(seg[0, 0], dtype=np.uint8)
        gt[seg[0, 1] * seg[1, 1]] = 2
        gt[seg[0, 0] * seg[1, 0]] = 0

        seg = seg.reshape((2, -1))
        cohens_k = cohen_kappa_score(seg[0], seg[1])

        # Sampling
        mean, weighted_mean, std, entropy, mi, grades = self.test_or_predict(img)

        self.img.append(ori_img.cpu().numpy())
        self.gt.append(gt)
        self.gt_grade.append(gt_grade.cpu().numpy())
        self.mean.append(mean.cpu().numpy())
        self.weighted_mean.append(weighted_mean.cpu().numpy())
        self.std.append(std.cpu().numpy())
        self.entropy.append(entropy.cpu().numpy())
        self.mi.append(mi.cpu().numpy())
        self.grade.append(grade.cpu().numpy())
        self.cohens_k.append(cohens_k)

    def on_test_end(self):
        results = {
            "img": np.concatenate(self.img, axis=0),
            "gt": np.stack(self.gt, axis=0),
            "gt_grade": np.concatenate(self.gt_grade, axis=0),
            "mean": np.stack(self.mean, axis=0),
            "weighted_mean": np.stack(self.weighted_mean, axis=0),
            "std": np.stack(self.std, axis=0),
            "entropy": np.stack(self.entropy, axis=0),
            "mi": np.stack(self.mi, axis=0),
            "grade": np.stack(self.grade, axis=0),
            "cohens_k": np.stack(self.cohens_k, axis=0),
        }
        with open(f"results/{self.exp_name}/results.pickle", "wb") as f:
            pickle.dump(results, f)

    def on_predict_start(self):
        self.logger.log_hyperparams(dict(self.cfg.predict))
        self.step = self.cfg.predict.step
        window = np.ones(256)
        window[: 256 - self.step] = np.arange(256 - self.step) / self.step
        window = np.tile(window[:, np.newaxis], (1, 256))
        self.window = window * np.rot90(window) * np.rot90(window, 2) * np.rot90(window, 3)

    def predict_step(self, batch, batch_idx):
        patches = batch["patches"][0]
        img = batch["ori_img"].cpu().numpy().astype(np.uint8)[0]
        file_name = batch["file_name"][0]
        mask = batch["mask"][0, :, :, 0].bool().cpu().numpy()

        nx = patches.shape[1]
        patches = patches.reshape(-1, patches.shape[2], patches.shape[3], patches.shape[4])
        bs = self.cfg.test.batch_size
        valid = []
        for i in range(patches.shape[0]):
            if not torch.all(patches[i] == 0):
                valid.append(i)

        pred = np.zeros((patches.shape[1] * self.step + 256, patches.shape[2] * self.step + 256))
        weighted_pred = np.zeros_like(pred)
        uncertainty = np.zeros_like(pred)

        for i in range(len(valid) // bs + 1):
            mean, weighted_mean, std, entropy, mi, grades = self.test_or_predict(patches[valid[i * bs : min((i + 1) * bs, len(valid))]])
            for j in range(bs):
                y = valid[i * bs + j] // nx
                x = valid[i * bs + j] % nx
                pred[y : y + 256, x : x + 256] += mean[j].cpu().numpy() * self.window
                weighted_pred[y : y + 256, x : x + 256] += weighted_mean[j].cpu().numpy() * self.window
                if self.cfg.predict.metric == "entropy":
                    u = entropy
                elif self.cfg.predict.metric == "mi":
                    u = mi
                else:
                    u = std
                uncertainty[y : y + 256, x : x + 256] += u[j].cpu().numpy() * self.window

        # Size
        height, width = img.shape[:2]

        # Thresholding
        pred = pred >= 0.5
        certainty = uncertainty < uncertainty[~mask].mean() + self.cfg.predict.th * uncertainty[~mask].std()

        # Draw contours
        img, area, easi = draw_contours(img, pred, certainty, weighted_pred, roi=~mask)

        # Text
        cv2.putText(
            img,
            f"entropy(sigma={self.cfg.predict.th:.2f}); area : {area[0]:.1f}-{area[1]:.1f}, EASI : {easi[0]:.1f}-{easi[1]:.1f}",
            (25, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        os.makedirs("results/predict", exist_ok=True)
        cv2.imwrite(os.path.join("results/predict", file_name), img)


if __name__ == "__main__":
    # arguments
    parser = ArgumentParser()
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--phase", type=str, choices=["train", "test", "predict"])
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--cfg", type=str, default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.cfg)

    # dataset
    atopy = AtopyDataModule(cfg)

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
        checkpoint_callback = ModelCheckpoint(monitor="val_loss")
        trainer = pl.Trainer(
            max_epochs=cfg.train.max_epochs,
            devices=args.devices,
            accelerator="gpu",
            strategy="ddp_find_unused_parameters_true",
            callbacks=[checkpoint_callback],
            log_every_n_steps=10,
            profiler="advanced"
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
