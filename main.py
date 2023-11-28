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
from statsmodels.stats.inter_rater import fleiss_kappa
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_start(self):
        self.logger.log_hyperparams(dict(self.cfg.train))

    def training_step(self, batch, batch_idx):
        seg, img, grade, _ = batch

        mask = seg[:, [self.trainer.current_epoch % seg.shape[1]]]
        loss_dict = self.model.sum_loss(
            torch.einsum("BLHW,BC->BCHW", mask, grade[..., self.trainer.current_epoch % grade.shape[-1]]), img, lesion_area=mask
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

    def on_test_start(self):
        self.logger.log_hyperparams(dict(self.cfg.test))
        self.mean_std_entropy_mi = []
        self.kappa = []
        self.severities = []
        self.easi = []

    def test_step(self, batch, batch_idx):
        seg, img, grade, ori_img = batch

        # seg: B x labeller x H x W
        bs = seg.shape[0]
        labeller = seg.shape[1]
        seg = seg.sum(1)

        # area: N x B x H x W x 1
        # logits/preds: N x B x H x W x C
        area, logits = self.model.sample(img, self.mc_n)
        mask = area >= 0.5
        preds = lit_model.model.log_cumulative(logits.reshape(-1, logits.shape[-1])).argmax(-1).reshape(logits.shape) * mask
        severities = torch.div(preds.sum((2, 3)), mask.sum((2, 3)))  # N x B x C
        easi = area2score(mask.mean((2, 3, 4))) * severities.sum(-1)  # N x B

        # mean/std/entropy/mi : B x H x W
        area = area.squeeze(-1)
        mask = mask.squeeze(-1)
        mean = mask.mean(0)
        std = mask.std(0)
        entropy = area.mean(0)
        entropy = -(entropy * torch.log(entropy) + (1 - entropy) * torch.log(1 - entropy))
        mi = entropy + (area * torch.log(area) + (1 - area) * torch.log(1 - area)).mean(0)

        seg = seg.reshape(bs, -1)  # B x (H x W)
        seg_mask = seg.unsqueeze(0).expand(4, -1, -1)  # 4 x B x (H x W)
        mean_std_entropy_mi = torch.stack([mean, std, entropy, mi]).reshape(4, bs, -1)
        msem_list = []
        for i in range(labeller + 1):
            seg_mask_i = seg_mask == i
            msem_list.append((mean_std_entropy_mi * seg_mask_i).sum(-1) / seg_mask_i.sum(-1))

        seg = torch.stack((labeller - seg, seg), dim=-1).cpu().numpy()  # B x (H x W) x 2

        self.mean_std_entropy_mi.append(torch.stack(msem_list, dim=1).cpu.numpy())  # 4(mean, std, entropy, mi) x (labeller+1) x B
        self.kappa.extend([fleiss_kappa(seg[i]) for i in bs])  # B
        self.severities.append(severities.cpu().numpy())  # N x B x C
        self.easi.append(easi.cpu().numpy())  # N x B
        self.gt_severities.append(grade.permute(2, 0, 1).cpu.numpy())  # N x B x C

    def on_test_end(self):
        self.mean_std_entropy_mi = np.stack(self.mean_std_entropy_mi, dim=-1)
        results = {
            "mean": self.mean_std_entropy_mi[0],
            "std": self.mean_std_entropy_mi[1],
            "entropy": self.mean_std_entropy_mi[2],
            "mi": self.mean_std_entropy_mi[3],
            "kappa": np.array(self.kappa),
            "severities": np.concatenate(self.severities, axis=1),
            "easi": np.concatenate(self.easi, axis=1),
            "gt_severities": np.concatenate(self.gt_severities, axis=1),
        }
        with open(f"results/{self.exp_name}/results.pickle", "wb") as f:
            pickle.dump(results, f)

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
        lesion_area = np.zeros(severity_logits.shape[:-1])

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
        preds = (
            lit_model.model.log_cumulative(severities.reshape(-1, severities.shape[-1])).argmax(-1).reshape(severities.shape) * lesion_area
        )
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
            profiler="advanced",
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
