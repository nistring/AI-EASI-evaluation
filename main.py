import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    StochasticWeightAveraging,
    ModelCheckpoint,
)
from lightning.pytorch.tuner import Tuner

from argparse import ArgumentParser
import os
import cv2
import math
import numpy as np
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
import shutil

from loader import get_dataset, CustomDataset
from model.model import HierarchicalProbUNet
from utils import *


class AtopyDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage):
        self.train, self.val = get_dataset("Atopy_Segment_Train")
        self.test = get_dataset("Atopy_Segment_Test")
        # self.test = get_dataset("Atopy_Segment_Extra")
        self.predict = CustomDataset("data/predict")

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
        return DataLoader(self.test, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True)


class LitModel(pl.LightningModule):
    def __init__(self, model, cfg, exp_name=None):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.mc_n = cfg.test.mc_n
        self.mc_n -= self.mc_n % cfg.test.batch_size
        self.automatic_optimization = False
        self.exp_name = exp_name
        self.cfg.exp_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=self.cfg.train.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_start(self):
        self.logger.log_hyperparams(dict(self.cfg.train))

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        seg, img, grade, _ = batch
        for i in range(seg.shape[1]):
            loss_dict = self.model.sum_loss(seg[:, i], img, grade)
            loss = loss_dict["supervised_loss"]
            summaries = loss_dict["summaries"]

            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

            self.log_dict({"train_" + k: v for k, v in summaries.items()}, sync_dist=True)

        sch.step()

    def validation_step(self, batch, batch_idx):
        seg, img, grade, _ = batch
        for i in range(seg.shape[1]):
            loss_dict = self.model.sum_loss(seg[:, i], img, grade)
            loss = loss_dict["supervised_loss"]
            summaries = loss_dict["summaries"]

            self.log_dict({"val_" + k: v for k, v in summaries.items()}, sync_dist=True)

    def test_or_predict(self, img):
        b, c, h, w = img.shape
        img = img.expand(self.cfg.test.batch_size, c, h, w)

        preds, grades = self.model.sample(img, self.mc_n // self.cfg.test.batch_size)

        mean = torch.zeros_like(preds)
        mean[preds >= 0.5] = 1.0

        weighted_mean = (mean * grades).mean(0)
        std = mean.std(0)
        mean = mean.mean(0)

        entropy = preds.mean(0)
        entropy = -(entropy * torch.log(entropy) + (1 - entropy) * torch.log(1 - entropy))

        mutual_information = entropy + (preds * torch.log(preds) + (1 - preds) * torch.log(1 - preds)).mean(0)

        return mean, weighted_mean, std, entropy, mutual_information, grades.reshape(-1)

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
        mean, weighted_mean, std, entropy, mi, grade = self.test_or_predict(img)

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

    def predict_step(self, batch, batch_idx):
        patches = batch["patches"][0]
        img = batch["ori_img"].cpu().numpy().astype(np.uint8)[0]
        file_name = batch["file_name"][0]
        mask = batch["mask"][0, :, :, 0].bool().cpu().numpy()

        pred = np.zeros(((patches.shape[1] + 1) * 128, (patches.shape[2] + 1) * 128))
        weighted_pred = np.zeros_like(pred)
        uncertainty = np.zeros_like(pred)

        with tqdm(total=patches.shape[1] * patches.shape[2]) as pbar:
            for i in range(patches.shape[1]):
                for j in range(patches.shape[2]):
                    patch = patches[:, i, j]
                    if not torch.all(patch == 0):
                        mean, weighted_mean, std, entropy, mi, grade = self.test_or_predict(patch)
                        pred[i * 128 : i * 128 + 256, j * 128 : j * 128 + 256] += mean.cpu().numpy()
                        weighted_pred[i * 128 : i * 128 + 256, j * 128 : j * 128 + 256] += weighted_mean.cpu().numpy()
                        if self.cfg.predict.metric == "entropy":
                            u = entropy
                        elif self.cfg.predict.metric == "mi":
                            u = mi
                        else:
                            u = std
                        uncertainty[i * 128 : i * 128 + 256, j * 128 : j * 128 + 256] += u.cpu().numpy()
                    pbar.update(1)

        pred[128:-128] /= 2
        pred[:, 128:-128] /= 2
        weighted_pred[128:-128] /= 2
        weighted_pred[:, 128:-128] /= 2
        uncertainty[128:-128] /= 2
        uncertainty[:, 128:-128] /= 2

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

    assert cfg.test.mc_n >= cfg.test.batch_size, print("Number of MC trials must be greater than batch size!")

    # dataset
    atopy = AtopyDataModule(cfg)

    # model
    model = HierarchicalProbUNet(
        latent_dims=cfg.model.latent_dims,
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        loss_kwargs=dict(cfg.train.loss_kwargs),
        num_grades=cfg.model.num_grades,
        num_cuts=cfg.model.num_cuts
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
