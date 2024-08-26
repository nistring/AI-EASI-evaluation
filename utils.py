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
    severity = 0.5 * (severity + severity.mean(2, keepdims=True))  # BNC
    severity = np.nan_to_num(severity * area.astype(bool))  # BNC

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
        _type_: _description_
    """
    B, N, C = s.shape[:3]
    M = y.shape[1]

    # 0-3 to 0-1 scale
    s = s / (C-1)
    y = y / (C-1)

    dSY, dSS, dYY = torch.zeros(B).to(s.device), torch.zeros(B).to(s.device), torch.zeros(B).to(s.device)

    for i in range(N):
        for j in range(M):
            dSY += (s[:, i]-y[:, j]).abs().mean((1, 2, 3)) # B
    dSY *= (2 / N / M)    

    for i in range(N):
        for j in range(N):
            dSS += (s[:, i]-s[:, j]).abs().mean((1, 2, 3)) # B
    dSS /= (N * N)

    for i in range(M):
        for j in range(M):
            dYY += (y[:, i]-y[:, j]).abs().mean((1, 2, 3)) # B
    dYY /= (M * M)

    ged = (dSY - dSS - dYY).cpu().numpy()
    return ged