from argparse import ArgumentParser
import os
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

if __name__ == "__main__":
    # arguments
    parser = ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    args = parser.parse_args()
    root = os.path.join("results", args.exp_name)
    sns.set_theme(style="ticks")

    # load results
    with open(os.path.join(root, "results.pkl"), "rb") as f:
        results = pickle.load(f)

    # severities
    EIExL = results["EIExL"].mean(0)
    gt_EIExL = results["gt_EIExL"].mean(0)

    B = EIExL.shape[0]
    dataset = np.array(["Erythema", "Induration", "Excoriation", "Lichenification"] * B)
    df = pd.DataFrame(data={"EASI": EIExL.reshape(-1), "gt_EASI":gt_EIExL.reshape(-1), "dataset":dataset})
    lm = sns.lmplot(data=df, x="gt_EASI", y="EASI", col="dataset", hue="dataset")
    lm.set(xlim=(-1, 19), ylim=(-1, 19))
    axes = lm.axes
    for i in range(len(["Erythema", "Induration", "Excoriation", "Lichenification"])):
        r, _ = pearsonr(gt_EIExL[:, i], EIExL[:, i])
        axes[0, i].annotate("r = {:.3f}".format(r),
                    xy=(.1, .9), xycoords=axes[0, i].transAxes)
    plt.savefig(f"results/{args.exp_name}_EASI_mean.png")

    EIExL = results["EIExL_std"]
    gt_EIExL = results["gt_EIExL_std"]
    B = EIExL.shape[0]
    dataset = np.array(["Erythema", "Induration", "Excoriation", "Lichenification"] * B)
    df = pd.DataFrame(data={"EASI": EIExL.reshape(-1), "gt_EASI":gt_EIExL.reshape(-1), "dataset":dataset})
    lm = sns.lmplot(data=df, x="gt_EASI", y="EASI", col="dataset", hue="dataset")
    lm.set(xlim=(-0.1, 1.5), ylim=(-0.1, 1.5))
    plt.margins(x=.1, y=.1)
    axes = lm.axes
    for i in range(len(["Erythema", "Induration", "Excoriation", "Lichenification"])):
        r, _ = pearsonr(gt_EIExL[:, i], EIExL[:, i])
        axes[0, i].annotate("r = {:.3f}".format(r),
                    xy=(.1, .9), xycoords=axes[0, i].transAxes)
    plt.savefig(f"results/{args.exp_name}_EASI_std.png")