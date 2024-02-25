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


def cal_EASI(area, severity):
    """Calculates EASI from area(0-1) and severity

    Args:
        area (_type_): N
        severity (_type_): N x C

    Returns:
        _type_: Mean and std of area and severity score plus EASI
    """
    area = area2score(area)
    if area.shape[0] == severity.shape[0]:
        EASI = area * severity.sum(1)  # N
    else:
        EASI = area[np.newaxis, :] * severity.sum(1, keepdims=True)  # N x N'
    return area.mean(), area.std(), severity.mean(0), severity.std(0), EASI.mean(), EASI.std()


def area2score(area):
    score = 5 + 10 * (area - 0.9)
    score[area < 0.9] = 1 + 5 * (area[area < 0.9] - 0.1)
    score[area < 0.1] = 10 * area[area < 0.1]
    return score


def create_dir(phase):
    root = "results" if phase == "test" else "vis"

    exp_name = 1
    while True:
        if not os.path.exists(f"{root}/{exp_name}"):
            os.mkdir(f"{root}/{exp_name}")
            break
        exp_name += 1

    with open(f"{root}/{exp_name}/config.txt", "w") as f:
        f.write(" ".join(sys.argv))

    return exp_name


def heatmap(ori, pred):
    """_summary_

    Args:
        ori (torch.Tensor): H x W x 3
        pred (torch.HalfTensor): C x H x W x 4

    Returns:
        np.ndarray: C x H x W x 3
    """

    return np.clip(
        (pred[:, :, :, 1:, np.newaxis] * np.array([[255, 64, 64], [64, 255, 64], [64, 64, 255]])).sum(-2)
        + ori[np.newaxis, :, :, :] * pred[..., [0]],
        0.0,
        255,
    ).astype(np.uint8)
