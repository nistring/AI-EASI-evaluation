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

if __name__ == "__main__":
    # arguments
    parser = ArgumentParser()
    parser.add_argument("--img-path", type=str, default="data/train/images/Atopy_Segment_Test/Grade2/7205.jpg")
    parser.add_argument("--sample-n", type=int, default=16)
    parser.add_argument("--cfg", type=str, default="config.yaml")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--weight", type=str, default="weights/multi/version_3/checkpoints/epoch=149-step=7200.ckpt")

    args = parser.parse_args()
    cfg = load_config(args.cfg)

    print(f"Sampling {os.path.basename(args.img_path)}")

    # model
    model = HierarchicalProbUNet(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        loss_kwargs=dict(cfg.train.loss_kwargs),
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
    z_q = torch.linspace(0, 3, args.sample_n).reshape(1, -1, 1, 1, 1).to(args.device)  # None
    mean = True  # None
    # z_q = None
    # mean = None # False

    # Sampling
    preds, grades = lit_model.model.sample(img, 1, z_q=z_q, mean=mean)

    # Postprocessing
    mean = torch.zeros_like(preds)
    mean[preds >= 0.5] = 1.0
    mean = mean.cpu().numpy()
    grades = grades.reshape(-1).cpu().numpy()

    # Make GIF
    images = []
    img = cv2.resize(ori_img, (256, 256))
    for j in range(mean.shape[0]):
        result = np.concatenate((img, (cv2.cvtColor(mean[j], cv2.COLOR_GRAY2RGB) * 255).astype(np.uint8)), axis=1)

        cv2.putText(
            result,
            f"Predicted grade : {grades[j]:.2f}",
            (270, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        images.append(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    
    imageio.mimsave("samples.gif", images, fps=4)