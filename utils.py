import numpy as np
import yaml
from easydict import EasyDict as edict
import torch

def load_config(path):
    with open(path) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config


def cal_EASI(preds):
    # area
    area = (preds.sum(2) > 1).reshape(preds.shape[:2] + (-1,)).float().mean(-1, keepdims=True).cpu().numpy()  # BN1
    mean_area = area.mean(1, keepdims=True) # B11
    area = area2score(area)  # BN1

    # severity
    severity = preds.float().mean(-1).cpu().numpy() / mean_area  # BNC
    # severity = 0.5 * (severity + severity.mean(2, keepdims=True))  # BNC
    severity = np.where(np.isnan(severity), 0, severity * area.astype(bool))  # BNC

    # easi
    easi = area2score(mean_area) * severity  # BNC

    return area.squeeze(-1), severity, easi


def area2score(area):
    """A mapping function that converts 0-1 scale area proportion to area score for EASI.

    Args:
        area (np.ndarray): Proportion of area in 0-1 scale.

    Returns:
        np.ndarray: Converted area score.
    """
    score = 6.25 * area + 0.375
    score[area > 0.9] = 6.
    score[area < 0.1] = 1.
    score[area < 0.01] = 0.
    return score


def heatmap(ori, pred):
    """Overlays heatmap results on an original image.

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


def generalized_energy_distance(s, y):
    """Calculate the generalized energy distance(maximum mean discrepancy) with L1 distance as a kernel

    Args:
        s (torch.Tensor): samples; B x N x C x H x W
        y (torch.Tensor): ground truth; B x M x C x H x W

    Returns:
        torch.Tensor: Generalized energy distance for each batch.
    """
    B, N, C, H, W = s.shape
    M = y.shape[1]

    # 0-3 to 0-1 scale
    s = s / (C-1)
    y = y / (C-1)

    s_flat = s.view(B, N, -1)
    y_flat = y.view(B, M, -1)

    dSY = torch.cdist(s_flat, y_flat, p=1).mean((1, 2))
    dSS = torch.cdist(s_flat, s_flat, p=1).mean((1, 2))
    dYY = torch.cdist(y_flat, y_flat, p=1).mean((1, 2))

    ged = ((2 * dSY - dSS - dYY) / C / H / W).cpu().numpy()
    return ged