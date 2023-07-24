from argparse import ArgumentParser
import os
import cv2
import numpy as np
from copy import deepcopy
from utils import *
from sklearn.metrics import roc_curve, RocCurveDisplay, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import pool

if __name__ == "__main__":
    # arguments
    parser = ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    args = parser.parse_args()

    step = 0.01
    start = -2.
    stop = 2.

    root = os.path.join("results", args.exp_name)

    # load results
    with open(os.path.join(root, "results.pickle"), "rb") as f:
        results = pickle.load(f)

    # Uncertain : True, certain : False
    gt = results["gt"]
    gt_2 = gt == 2
    gt_1 = gt == 1
    pred = results["mean"] >= 0.5

    fig, ax = plt.subplots()

    for metric in ["std", "entropy", "mi"]:
        print(f"Processing {metric}...")
        score = results[metric]
        mean = score.mean()
        std = score.std()

        f1 = []
        thresholds = np.arange(start, stop + step, step)
        for z in tqdm(thresholds):
            pred_2 = pred & (score < mean + z * std)
            pred_1 = score >= mean + z * std
            f1.append(2 * ((gt_2 & pred_2).sum() / (gt_2.sum() + pred_2.sum()) + (gt_1 & pred_1).sum() / (gt_1.sum() + pred_1.sum())))
        max_f1 = max(f1)
        optimal_th =  thresholds[f1.index(max_f1)]

        ax.plot(thresholds, f1, label=f"{metric}(th={optimal_th:.2f}, f1={max_f1:.2f})")

    ax.set_title(f"{args.exp_name}")
    ax.set_xlabel("Threshold(z score)")
    ax.set_ylabel("F1 score")
    ax.legend()
    plt.savefig(f"results/{args.exp_name}/ROC_{args.exp_name}.png")