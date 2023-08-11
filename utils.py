import cv2
import pandas as pd
import torch
import numpy as np
import sys
import os
import yaml
from easydict import EasyDict as edict
import pickle

def load_config(path):
    with open(path) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config

def draw_contours(img, prediction, certainty, weighted_mean:np.ndarray, scale=0.5, roi=None):

    if roi is None:
        roi = np.ones_like(prediction)

    # True & Certain
    mask = prediction & certainty & roi
    min_area = mask[roi].mean()
    min_easi = weighted_mean[mask].sum() / mask.sum() * area_score(min_area) * 4
    mask = cv2.GaussianBlur(mask.astype(np.uint8), (int(scale * 8) * 2 + 1, int(scale * 8) * 2 + 1), 0) >= 0.5
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color=(0, 0, 255), thickness=int(scale * 2))

    # Only Certain
    mask = prediction & roi
    max_area = mask[roi].mean()
    max_easi = weighted_mean[mask].sum() / mask.sum() * area_score(max_area) * 4
    mask = cv2.GaussianBlur(mask.astype(np.uint8), (int(scale * 8) * 2 + 1, int(scale * 8) * 2 + 1), 0) >= 0.5
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color=(0, 255, 255), thickness=int(scale * 2))

    return img, (min_area * 100, max_area * 100), (min_easi, max_easi)


def normalize_uncertainty(uncertainty):
    uncertainty -= uncertainty.min()
    if uncertainty.max() != 0:
        uncertainty /= uncertainty.max()
    return uncertainty


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

def area_score(area:float):
    if area < 0.1:
        return 10 * area
    elif area < 0.9:
        return 1 + 5 * (area - 0.1)
    else:
        return 5 + 10 * (area - 0.9)

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