from argparse import ArgumentParser
import os
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau


if __name__ == "__main__":
    # arguments
    parser = ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    args = parser.parse_args()
    root = os.path.join("results", args.exp_name)
    sns.set_theme()

    # load results
    with open(os.path.join(root, "results.pkl"), "rb") as f:
        results = pickle.load(f)

    # area
    area = results["area"].mean(0)
    gt_area = results["gt_area"].mean(0)
    df = pd.DataFrame(data={"Pred.": area, "True": gt_area})
    lm = sns.regplot(data=df, x="True", y="Pred.", robust=True, scatter_kws={"s": 2.}, truncate=False)
    lm.set(xlim=(0., 1.), ylim=(0., 1.))
    lm.set(title=f"PCC of {args.exp_name}'s area")
    lm.axes.axline((0, 0), slope=1.0, ls="--", color="black", alpha=0.5)

    r, p = pearsonr(gt_area, area, alternative="greater")
    lm.text(0.1, 0.9, "r = {:.3f}, p = {:.3e}".format(r, p))
    lm.text(0.1, 0.8, "Mean avg err = {:.3f}".format(np.mean(np.abs(gt_area - area))))
    plt.tight_layout()
    plt.savefig(f"figure/{args.exp_name}_area.png")
    plt.clf()

    # kappa
    kappa = results["kappa"]
    gt_kappa = results["gt_kappa"]
    df = pd.DataFrame(data={"Pred.": kappa, "True": gt_kappa})
    sns.scatterplot(data=df, x="True", y="Pred.", color="0.5", size=0.2, legend=False)
    lm = sns.kdeplot(data=df, x="True", y="Pred.", fill=True, color="k")
    lm.set(title=f"Kendall's tau on fleiss k of {args.exp_name}'s area")
    lm.set(xlim=(-1., 1.), ylim=(0., 1.))
    r, p = kendalltau(gt_kappa, kappa, alternative="greater")
    lm.text(-0.9, 0.1, "r = {:.3f}, p = {:.3e}".format(r, p))
    plt.tight_layout()
    plt.savefig(f"figure/{args.exp_name}_kappa.png")
    plt.clf()

    # Severity
    severity = results["severity"].mean(0)
    gt_severity = results["gt_severity"].mean(0)
    B = severity.shape[0]
    Sign = np.array(["Erythema", "Induration", "Excoriation", "Lichenification"] * B)

    df = pd.DataFrame(data={"Pred.": severity.reshape(-1), "True": gt_severity.reshape(-1), "Sign": Sign})
    lm = sns.lmplot(
        data=df, x="True", y="Pred.", col="Sign", hue="Sign", x_estimator=np.mean, robust=True, x_ci="sd", ci=None, truncate=False
    )
    lm.set(xlim=(-0.05, 3.05), ylim=(-0.05, 3.05))
    lm.figure.suptitle(f"PCC of {args.exp_name}'s severity score")
    axes = lm.axes
    for i in range(len(["Erythema", "Induration", "Excoriation", "Lichenification"])):
        axes[0, i].axline((0, 0), slope=1.0, ls="--", color="black", alpha=0.5)
        axes[0, i].annotate(
            "Mean avg err = {:.3f}".format(np.mean(np.abs(gt_severity[:, i] - severity[:, i]))),
            xy=(0.1, 0.9),
            xycoords=axes[0, i].transAxes,
        )
    plt.tight_layout()
    plt.savefig(f"figure/{args.exp_name}_severity.png")
    plt.clf()

    # EASI mean
    easi = results["easi"].mean(0)
    gt_easi = results["gt_easi"].mean(0)

    df = pd.DataFrame(data={"Pred.": easi.reshape(-1), "True": gt_easi.reshape(-1), "Sign": Sign})
    lm = sns.lmplot(data=df, x="True", y="Pred.", col="Sign", hue="Sign", robust=True, scatter_kws={"s": 2.}, legend=False, truncate=False)
    lm.set(xlim=(0., 18.), ylim=(0., 18.))
    lm.figure.suptitle(f"PCC of {args.exp_name}'s EASI score")
    axes = lm.axes
    for i in range(len(["Erythema", "Induration", "Excoriation", "Lichenification"])):
        r, p = pearsonr(gt_easi[:, i], easi[:, i], alternative="greater")
        axes[0, i].axline((0, 0), slope=1.0, ls="--", color="black", alpha=0.5)
        axes[0, i].annotate("r = {:.3f}, p = {:.3e}".format(r, p), xy=(0.1, 0.9), xycoords=axes[0, i].transAxes)
        axes[0, i].annotate(
            "Mean avg err = {:.3f}".format(np.mean(np.abs(gt_easi[:, i] - easi[:, i]))), xy=(0.1, 0.8), xycoords=axes[0, i].transAxes
        )
    plt.tight_layout()
    plt.savefig(f"figure/{args.exp_name}_easi.png")
    plt.clf()

    # EASI std
    easi = results["easi"].std(0)
    gt_easi = results["gt_easi"].std(0)

    df = pd.DataFrame(data={"Pred.": easi.reshape(-1), "True": gt_easi.reshape(-1), "Sign": Sign})
    g = sns.FacetGrid(df, col="Sign", hue="Sign", height=5)
    g.map(sns.kdeplot, "True", "Pred.")
    g.figure.suptitle(f"Kendall's tau on std of {args.exp_name}'s EASI score")
    g.set(xlim=(0., None), ylim=(0., None))
    axes = g.axes
    for i in range(len(["Erythema", "Induration", "Excoriation", "Lichenification"])):
        r, p = kendalltau(gt_easi[:, i], easi[:, i], alternative="greater")
        axes[0, i].annotate("r = {:.3f}, p = {:.3e}".format(r, p), xy=(0.1, 0.9), xycoords=axes[0, i].transAxes)
    plt.tight_layout()
    plt.savefig(f"figure/{args.exp_name}_easi_std.png")
    plt.clf()