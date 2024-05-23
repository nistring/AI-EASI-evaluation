import numpy as np
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
    if area.shape[0] == severity.shape[0]:
        EASI = area * severity.sum(1)  # N
    else:
        EASI = area[np.newaxis, :] * severity.sum(1, keepdims=True)  # N x N'
    return area.mean(), area.std(), severity.mean(0), severity.std(0), EASI.mean(), EASI.std()


def area2score(area):
    score = 6.25 * area + 0.375
    score[area > 0.9] = 6.
    score[area < 0.1] = 1.
    score[area < 0.01] = 0.
    return score


def heatmap(ori, pred):
    """_summary_

    Args:
        ori (torch.Tensor): H x W x 3
        pred (torch.HalfTensor): C x H x W x 4

    Returns:
        np.ndarray: C x H x W x 3
    """
    return np.clip(
        (pred[:, :, :, 1:, np.newaxis] * np.array([[0, 255, 127], [0, 255, 255], [0, 0, 255]])).sum(-2)
        + ori[np.newaxis, :, :, :] * pred[..., [0]],
        0.0,
        255,
    ).astype(np.uint8)
