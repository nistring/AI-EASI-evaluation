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
    parser.add_argument("--sample-n", type=int, default=30)
    parser.add_argument("--cfg", type=str, default="config.yaml")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--weight", type=str, default="weights/lr_e-5_exdecay_0.99/checkpoints/epoch=486-step=6818.ckpt")

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
    # mean = True  # None

    # Sampling
    # logits : B(N) x H x W x C+1
    area, logits = lit_model.model.sample(img, 1)
    area = area[0]
    logits = logits[0]

    images = []
    for th in tqdm(np.arange(0.9, 0.0, -0.1)):
        for bias in np.arange(-1., 1.1, 0.1):
            # Postprocessing
            img = cv2.resize(ori_img, (256, 256))
            # preds: B(N) x H x W x C
            preds = (
                torch.cat(lit_model.model.log_cumulative(logits.reshape(-1, logits.shape[-1]) + bias), dim=-1).argmax(-1).reshape(logits.shape)
            ) * (area >= th)
            preds = (F.one_hot(preds, num_classes=4).float().mean(0)).permute(2, 0, 1, 3)  # C x H x W x 4
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