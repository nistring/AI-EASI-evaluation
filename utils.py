import cv2
import pandas as pd
import torch
import numpy as np
import sys
import os
import yaml
from easydict import EasyDict as edict

def load_config(path):
    with open(path) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config

def draw_contours(img, prediction, uncertainty, grade, scale, thresholds):
    for th, color in zip(thresholds, [(0, 0, 255), (0, 128, 255), (0, 255, 255)]):
        mask = prediction * (uncertainty < th).astype(np.uint8)
        mask = cv2.GaussianBlur(mask, (int(scale * 8) * 2 + 1, int(scale * 8) * 2 + 1), 0) >= 0.5
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color=color, thickness=int(scale * 2))
    cv2.putText(
        img,
        f"class {grade}",
        (int(25 * scale), int(25 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 255, 0),
        int(scale * 2),
        cv2.LINE_AA,
    )
    return img


def visualize(img, width, height, path):
    img = cv2.resize(
        (img * 255).type(torch.uint8).cpu().numpy(),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    cv2.imwrite(path, img)
    return img.astype(np.float32) / 255


def normalize_uncertainty(uncertainty):
    if uncertainty.mean() == 0.0:
        return uncertainty
    else:
        return torch.clamp(uncertainty / 2 / uncertainty.mean(), 0, 1)


def quantify_uncertainty(preds):
    mean = torch.zeros_like(preds)
    mean[preds >= 0.5] = 1.0
    std = mean.std(0)
    mean = mean.mean(0)

    entropy = preds.mean(0)
    entropy = -(entropy * torch.log(entropy) + (1 - entropy) * torch.log(1 - entropy))

    mutual_information = entropy - (preds * torch.log(preds) + (1 - preds) * torch.log(1 - preds)).mean(0)
    return mean, std, entropy, mutual_information


def eval_perform(df, labels, predictions, binary_unc_map, grades, cohens_k):
    gt_2, gt_1, gt_0 = labels
    gt_grade, pred_grade = grades

    n_gt_2 = gt_2.sum().item()
    n_gt_1 = gt_1.sum().item()
    n_gt_0 = gt_0.sum().item()

    results = pd.DataFrame(
        {
            "ac_2": [(gt_2 * (~binary_unc_map) * predictions).sum().item()],
            "au_2": [(gt_2 * binary_unc_map * predictions).sum().item()],
            "ic_2": [(gt_2 * (~binary_unc_map) * (~predictions)).sum().item()],
            "iu_2": [(gt_2 * binary_unc_map * (~predictions)).sum().item()],
            "c_1": [(gt_1 * (~binary_unc_map)).sum().item()],
            "u_1": [(gt_1 * binary_unc_map).sum().item()],
            "ac_0": [(gt_0 * (~binary_unc_map) * predictions).sum().item()],
            "au_0": [(gt_0 * binary_unc_map * predictions).sum().item()],
            "ic_0": [(gt_0 * (~binary_unc_map) * (~predictions)).sum().item()],
            "iu_0": [(gt_0 * binary_unc_map * (~predictions)).sum().item()],
            "n_gt_2": [n_gt_2],
            "n_gt_1": [n_gt_1],
            "n_gt_0": [n_gt_0],
            "grade": [gt_grade],
            "pred": [pred_grade],
            "match": [int(gt_grade == pred_grade)],
            "cohen's k": [cohens_k],
        }
    )

    return pd.concat([df, results], axis=0)

def create_dir(phase):
    root = 'results' if phase == 'test' else 'vis'

    exp_name = 1
    while True:
        if not os.path.exists(f"{root}/{exp_name}"):
            os.mkdir(f"{root}/{exp_name}")
            break
        exp_name += 1
    
    with open(f"{root}/{exp_name}/config.txt", 'w') as f:
        f.write(' '.join(sys.argv))

    return exp_name