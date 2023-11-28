from argparse import ArgumentParser
import os
import cv2
import numpy as np
from utils import *
from main import LitModel
from model.model import HierarchicalProbUNet
import random
from loader import test_transforms
from sklearn.metrics import cohen_kappa_score
import imageio
from tqdm import tqdm
import torch



if __name__ == "__main__":
    # arguments
    parser = ArgumentParser()
    # parser.add_argument("--img-path", type=str, default="data/train/images/Atopy_Segment_Extra/Grade2/9115.jpg")
    parser.add_argument("--img-path", type=str, default="data/train/images/Atopy_Segment_Extra/Grade1/9753.jpg")
    parser.add_argument("--sample-n", type=int, default=100)
    parser.add_argument("--cfg", type=str, default="config.yaml")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--weight", type=str, default="lightning_logs/version_0/checkpoints/epoch=66-step=938.ckpt")

    args = parser.parse_args()
    cfg = load_config(args.cfg)

    print(f"Sampling {os.path.basename(args.img_path)}")

    # model
    model = HierarchicalProbUNet(
        latent_dims=cfg.model.latent_dims,
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        loss_kwargs=dict(cfg.train.loss_kwargs),
        num_cuts=cfg.model.num_cuts,
    )

    # Load model
    lit_model = LitModel.load_from_checkpoint(args.weight, model=model, cfg=cfg)
    lit_model.to(args.device)
    lit_model.eval()

    # Image transformation
    ori_img = cv2.imread(args.img_path)
    img = cv2.resize(ori_img, (512, 512))
    img = test_transforms(image=img)["image"].unsqueeze(0)
    img = img.expand(args.sample_n, -1, -1, -1).to(args.device)

    # Designate latent variables
    # z_q = torch.linspace(0, 3, args.sample_n).reshape(1, -1, 1, 1, 1).to(args.device)  # None
    # mean = True  # None
    z_q = None
    mean = [False, False, False, False]

    # Sampling
    # logits : B(N) x H x W x C+1
    area, logits = lit_model.model.sample(img, 1, z_q=z_q, mean=mean)[0]

    images = []
    for th in tqdm(np.arange(0.9, 0.0, -0.1)):
        for bias in np.arange(-1., 1.1, 0.1):
            # Postprocessing
            img = cv2.resize(ori_img, (256, 256))
            # preds: B(N) x H x W x C
            preds = (
                torch.cat(lit_model.model.log_cumulative(logits.reshape(-1, logits.shape[-1]) + bias), dim=-1).argmax(-1).reshape(logits.shape)
            ) * (area >= th)
            easi = preds.float().mean() * 24
            preds = heatmap(img, preds)

            result = np.concatenate([img] + [preds[i] for i in range(preds.shape[0])], axis=1)
            cv2.putText(
                result,
                f"Bias: {bias:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                result,
                f"Threshold: {th:.1f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                result,
                f"EASI: {easi:.2f}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            
            for i, text in enumerate(("Erythema", "Induration", "Excoriation", "Lichenification")):
                cv2.putText(
                    result,
                    text,
                    (256 * (i+1) + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            images.append(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    
    imageio.mimsave("samples.gif", images, fps=15)