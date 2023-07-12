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

from loader import get_dataset, CustomDataset
from model.model import HierarchicalProbUNet
from model.multitask_model import MultiTaskHPU
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.train.step_size, gamma=self.cfg.train.gamma)
        return [optimizer], [scheduler]

    def on_train_start(self):
        self.logger.log_hyperparams(dict(self.cfg.train))

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        seg, img, grade = batch
        for i in range(seg.shape[1]):
            loss_dict = self.model.sum_loss(seg[:, i], img, grade)
            loss = loss_dict["supervised_loss"]
            summaries = loss_dict["summaries"]

            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

            self.log_dict(
                {
                    "train_loss": loss,
                    "train_rec_per_pixel": summaries["ma_rec_loss_mean"],
                    "train_lagmul": summaries["lagmul"],
                    "train_kl_sum": summaries["kl_sum"],
                    "train_kl_0": summaries["kl_0"],
                    "train_kl_1": summaries["kl_1"],
                    "train_kl_2": summaries["kl_2"],
                    "train_kl_3": summaries["kl_3"],
                }
            )
`
    def validation_step(self, batch, batch_idx):
        seg, img, grade = batch
        for i in range(seg.shape[1]):
            loss_dict = self.model.sum_loss(seg[:, i], img, grade)
            loss = loss_dict["supervised_loss"]
            summaries = loss_dict["summaries"]

            self.log_dict(
                {
                    "val_loss": loss,
                    "val_rec_per_pixel": summaries["ma_rec_loss_mean"],
                    "val_lagmul": summaries["lagmul"],
                    "val_kl_sum": summaries["kl_sum"],
                    "val_kl_0": summaries["kl_0"],
                    "val_kl_1": summaries["kl_1"],
                    "val_kl_2": summaries["kl_2"],
                    "val_kl_3": summaries["kl_3"],
                },
                sync_dist=True,
            )

    def test_or_predict(self, img):
        b, c, w, h = img.shape
        img = img.expand(self.cfg.test.batch_size, c, w, h)

        preds, grade = self.model.sample(img, self.mc_n // self.cfg.test.batch_size)

        return quantify_uncertainty(preds), grade

    def on_test_start(self):
        self.logger.log_hyperparams(dict(self.cfg.test))
        df = pd.DataFrame(
            {
                "ac_2": [],
                "au_2": [],
                "ic_2": [],
                "iu_2": [],
                "c_1": [],
                "u_1": [],
                "ac_0": [],
                "au_0": [],
                "ic_0": [],
                "iu_0": [],
                "n_gt_2": [],
                "n_gt_1": [],
                "n_gt_0": [],
                "grade": [],
                "pred": [],
                "match": [],
                "cohen's k": [],
            }
        )
        self.results = {"std": df, "entropy": df.copy(deep=True), "mutual_information": df.copy(deep=True)}

    def test_step(self, batch, batch_idx):

        seg, img, gt_grade = batch

        seg = seg[0].bool().cpu()

        # accurate & certain; where two raters consented being accurate
        gt_2 = seg[0, 1] * seg[1, 1]
        # inaccurate & certain; where two raters consented being accurate
        gt_0 = seg[0, 0] * seg[1, 0]
        # uncertain; where two raters didn't consent
        gt_1 = (~gt_2) * (~gt_0)

        seg = seg.reshape((2, -1)).numpy()
        cohens_k = cohen_kappa_score(seg[0], seg[1])

        # Sampling
        (mean, std, entropy, mutual_information), grade = self.test_or_predict(img)

        # Binarize
        mean = (mean >= 0.5).cpu()
        std = (std >= std.mean()).cpu()
        entropy = (entropy >= entropy.mean()).cpu()
        mutual_information = (mutual_information >= mutual_information.mean()).cpu()

        gt_grade = torch.argmax(gt_grade[0]).item()
        grade = gt_grade if grade is None else torch.argmax(grade[0]).item()

        self.results["std"] = eval_perform(self.results["std"], (gt_2, gt_1, gt_0), mean, std, (gt_grade, grade), cohens_k)
        self.results["entropy"] = eval_perform(self.results["entropy"], (gt_2, gt_1, gt_0), mean, entropy, (gt_grade, grade), cohens_k)
        self.results["mutual_information"] = eval_perform(
            self.results["mutual_information"], (gt_2, gt_1, gt_0), mean, mutual_information, (gt_grade, grade), cohens_k
        )

    def on_test_end(self):
        self.results["std"].to_csv(f"results/{self.exp_name}/std.csv", index=False)
        self.results["entropy"].to_csv(f"results/{self.exp_name}/entropy.csv", index=False)
        self.results["mutual_information"].to_csv(f"results/{self.exp_name}/mutual_information.csv", index=False)

    def on_predict_start(self):
        self.logger.log_hyperparams(dict(self.cfg.test))
        for dir in ["entropy", "mutual_information", "prediction", "segmentation", "std"]:
            os.makedirs(f"vis/{self.exp_name}/{dir}")
        for dir in ["entropy", "mutual_information", "std"]:
            os.makedirs(f"vis/{self.exp_name}/prediction/{dir}")

    def predict_step(self, batch, batch_idx):
        img = batch["transformed"]
        ori_img = batch["ori_img"]
        file_name = batch["file_name"][0]

        b, c, w, h = img.shape

        ori_img = ori_img.cpu().numpy().astype(np.uint8)[0]
        height, width = ori_img.shape[:2]

        (mean, std, entropy, mutual_information), grade = self.test_or_predict(img)
        grade = "None" if grade is None else torch.argmax(grade[0]).item()

        std = normalize_uncertainty(std)
        entropy = normalize_uncertainty(entropy)
        mutual_information = normalize_uncertainty(mutual_information)

        # Visualization
        mean = visualize(mean, width, height, f"vis/{self.exp_name}/segmentation/{file_name}")
        std = 1 - visualize(1 - std, width, height, f"vis/{self.exp_name}/std/{file_name}")
        entropy = 1 - visualize(1 - entropy, width, height, f"vis/{self.exp_name}/entropy/{file_name}")
        mutual_information = 1 - visualize(1 - mutual_information, width, height, f"vis/{self.exp_name}/mutual_information/{file_name}")

        mean = mean >= 0.5
        scale = width / w / 2

        cv2.imwrite(
            f"vis/{self.exp_name}/prediction/std/{file_name}",
            draw_contours(deepcopy(ori_img), mean, std, grade, scale, self.cfg.test.uncertainty_th),
        )
        cv2.imwrite(
            f"vis/{self.exp_name}/prediction/entropy/{file_name}",
            draw_contours(deepcopy(ori_img), mean, entropy, grade, scale, self.cfg.test.uncertainty_th),
        )
        cv2.imwrite(
            f"vis/{self.exp_name}/prediction/mutual_information/{file_name}",
            draw_contours(deepcopy(ori_img), mean, mutual_information, grade, scale, self.cfg.test.uncertainty_th),
        )


if __name__ == "__main__":
    # arguments
    parser = ArgumentParser()
    parser.add_argument("--devices", type=int, default=3)
    parser.add_argument("--phase", type=str, choices=["train", "test", "predict"])
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--multi", action="store_true", help="Use multi-task u-net model")
    parser.add_argument("--cfg", type=str, default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.cfg)

    assert cfg.test.mc_n >= cfg.test.batch_size, print("Number of MC trials must be greater than batch size!")

    # dataset
    atopy = AtopyDataModule(cfg)

    # model
    if args.multi:
        model = MultiTaskHPU(
            latent_dims=cfg.model.latent_dims,
            in_channels=cfg.model.in_channels,
            num_classes=cfg.model.num_classes,
            loss_kwargs=dict(cfg.train.loss_kwargs),
        )
    else:
        model = HierarchicalProbUNet(
            in_channels=cfg.model.in_channels,
            num_classes=cfg.model.num_classes,
            loss_kwargs=dict(cfg.train.loss_kwargs),
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
        )
        if args.checkpoint:
            trainer.fit(litmodel, datamodule=atopy, ckpt_path=args.checkpoint)
        else:
            trainer.fit(litmodel, datamodule=atopy)
    # test
    else:
        exp_name = create_dir(args.phase)
        litmodel = LitModel.load_from_checkpoint(args.checkpoint, model=model, cfg=cfg, exp_name=exp_name)
        trainer = pl.Trainer(devices=[args.devices])
        if args.phase == "test":
            trainer.test(litmodel, datamodule=atopy)
        else:
            trainer.predict(litmodel, datamodule=atopy)
