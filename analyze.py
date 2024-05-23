from argparse import ArgumentParser
import os
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

robust = False

if __name__ == "__main__":
    # arguments
    parser = ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    args = parser.parse_args()
    root = os.path.join("results", args.exp_name)
    sns.set_theme()

    os.makedirs(f"figure/{args.exp_name}", exist_ok=True)
    # load results
    with open(os.path.join(root, "results.pkl"), "rb") as f:
        results = pickle.load(f)

    # area
    area = results["area"].mean(0)
    gt_area = results["gt_area"].mean(0)
    df = pd.DataFrame(data={"Pred.": area, "True": gt_area})
    lm = sns.regplot(data=df, x="True", y="Pred.", scatter_kws={"alpha":0.5}, robust=robust, x_jitter=0.1)
    lm.set(xlim=(-0.1, 6.1), ylim=(-0.1, 6.1))
    lm.set(title=f"PCC of {args.exp_name}'s area")
    lm.axes.axline((0, 0), slope=1.0, ls="--", color="black", alpha=0.5)

    r, p = pearsonr(gt_area, area, alternative="greater")
    lm.text(0.1, 5.0, "r = {:.3f}, p = {:.3e}".format(r, p))
    lm.text(0.1, 4.0, "Mean avg err = {:.3f}".format(np.mean(np.abs(gt_area - area))))
    plt.tight_layout()
    plt.savefig(f"figure/{args.exp_name}/{args.exp_name}_area.png")
    plt.clf()

    # kappa
    kappa = results["kappa"]
    gt_kappa = results["gt_kappa"]
    df = pd.DataFrame(data={"Pred.": kappa, "True": gt_kappa})
    lm = sns.scatterplot(data=df, x="True", y="Pred.", color="0.5", alpha=0.5, legend=False)
    lm.set(title=f"Spearman's rho on fleiss k of {args.exp_name}'s area")
    lm.set(xlim=(-1.05, 1.05), ylim=(-0.05, 1.05))
    r, p = spearmanr(gt_kappa, kappa, alternative="greater")
    lm.text(-0.9, 0.1, "r = {:.3f}, p = {:.3e}".format(r, p))
    plt.tight_layout()
    plt.savefig(f"figure/{args.exp_name}/{args.exp_name}_kappa.png")
    plt.clf()

    # EASI mean
    easi = results["easi"].sum(2).mean(0)
    gt_easi = results["gt_easi"].sum(2).mean(0)
    df = pd.DataFrame(data={"Pred.": easi, "True": gt_easi})
    lm = sns.regplot(data=df, x="True", y="Pred.", scatter_kws={"alpha":0.5}, robust=robust)
    lm.set(xlim=(-1., 73.), ylim=(-1., 73.))
    lm.set(title=f"PCC of {args.exp_name}'s EASI score")
    lm.axes.axline((0, 0), slope=1.0, ls="--", color="black", alpha=0.5)
    r, p = pearsonr(gt_easi, easi, alternative="greater")
    lm.text(5, 65, "r = {:.3f}, p = {:.3e}".format(r, p))
    lm.text(5, 55, "Mean avg err = {:.3f}".format(np.mean(np.abs(gt_easi - easi))))
    plt.tight_layout()
    plt.savefig(f"figure/{args.exp_name}/{args.exp_name}_easi.png")
    plt.clf()

    # EASI std
    easi = results["easi"].sum(2).std(0)
    gt_easi = results["gt_easi"].sum(2).std(0)
    df = pd.DataFrame(data={"Pred.": easi, "True": gt_easi})
    lm = sns.scatterplot(data=df, x="True", y="Pred.", color="0.5", alpha=0.5, legend=False)
    # lm = sns.kdeplot(data=df, x="True", y="Pred.", fill=True, color="k")
    lm.set(title=f"Spearman's rho on std of {args.exp_name}'s EASI score")
    lm.set(xlim=(-0.5, None), ylim=(-0.1, None))
    r, p = spearmanr(gt_easi, easi, alternative="greater")
    lm.text(0.5, 0.5, "r = {:.3f}, p = {:.3e}".format(r, p))
    plt.tight_layout()
    plt.savefig(f"figure/{args.exp_name}/{args.exp_name}_easi_std.png")
    plt.clf()

    # Severity
    severity = results["severity"].mean(0)
    gt_severity = results["gt_severity"].mean(0)
    B = severity.shape[0]
    Sign = np.array(["Erythema", "Induration", "Excoriation", "Lichenification"] * B)

    df = pd.DataFrame(data={"Pred.": severity.reshape(-1), "True": gt_severity.reshape(-1), "Sign": Sign})
    lm = sns.lmplot(
        data=df, x="True", y="Pred.", col="Sign", hue="Sign", x_jitter=.05, robust=robust, scatter_kws={"alpha":0.5}
    )
    lm.set(xlim=(-0.1, 3.1), ylim=(-0.1, 3.1))
    lm.figure.suptitle(f"PCC of {args.exp_name}'s severity score")
    axes = lm.axes
    for i in range(len(["Erythema", "Induration", "Excoriation", "Lichenification"])):
        r, p = pearsonr(gt_severity[:, i], severity[:, i], alternative="greater")
        axes[0, i].axline((0, 0), slope=1.0, ls="--", color="black", alpha=0.5)
        axes[0, i].annotate("r = {:.3f}, p = {:.3e}".format(r, p), xy=(0.1, 0.9), xycoords=axes[0, i].transAxes)
        axes[0, i].annotate(
            "Mean avg err = {:.3f}".format(np.mean(np.abs(gt_severity[:, i] - severity[:, i]))),
            xy=(0.1, 0.8),
            xycoords=axes[0, i].transAxes,
        )
    plt.tight_layout()
    plt.savefig(f"figure/{args.exp_name}/{args.exp_name}_severity.png")
    plt.clf()

    # EASI mean by sign
    easi = results["easi"].mean(0)
    gt_easi = results["gt_easi"].mean(0)
    df = pd.DataFrame(data={"Pred.": easi.reshape(-1), "True": gt_easi.reshape(-1), "Sign": Sign})
    lm = sns.lmplot(data=df, x="True", y="Pred.", col="Sign", hue="Sign", scatter_kws={"alpha":0.5}, legend=False, robust=robust)
    lm.set(xlim=(-0.5, 18.5), ylim=(-0.5, 18.5))
    lm.figure.suptitle(f"PCC of {args.exp_name}'s EASI score by sign")
    axes = lm.axes
    for i in range(len(["Erythema", "Induration", "Excoriation", "Lichenification"])):
        r, p = pearsonr(gt_easi[:, i], easi[:, i], alternative="greater")
        axes[0, i].axline((0, 0), slope=1.0, ls="--", color="black", alpha=0.5)
        axes[0, i].annotate("r = {:.3f}, p = {:.3e}".format(r, p), xy=(0.1, 0.9), xycoords=axes[0, i].transAxes)
        axes[0, i].annotate(
            "Mean avg err = {:.3f}".format(np.mean(np.abs(gt_easi[:, i] - easi[:, i]))), xy=(0.1, 0.8), xycoords=axes[0, i].transAxes
        )
    plt.tight_layout()
    plt.savefig(f"figure/{args.exp_name}/{args.exp_name}_easi_by_sign.png")
    plt.clf()

    # EASI std by sign
    easi = results["easi"].std(0)
    gt_easi = results["gt_easi"].std(0)

    df = pd.DataFrame(data={"Pred.": easi.reshape(-1), "True": gt_easi.reshape(-1), "Sign": Sign})
    lm = sns.lmplot(data=df, x="True", y="Pred.", col="Sign", hue="Sign", scatter_kws={"alpha":0.5}, legend=False, fit_reg=False)
    lm.figure.suptitle(f"Spearman's rho on std of {args.exp_name}'s EASI score by sign")
    lm.set(xlim=(-0.5, None), ylim=(0., None))
    axes = lm.axes
    for i in range(len(["Erythema", "Induration", "Excoriation", "Lichenification"])):
        r, p = spearmanr(gt_easi[:, i], easi[:, i], alternative="greater")
        axes[0, i].annotate("r = {:.3f}, p = {:.3e}".format(r, p), xy=(0.1, 0.9), xycoords=axes[0, i].transAxes)
    plt.tight_layout()
    plt.savefig(f"figure/{args.exp_name}/{args.exp_name}_easi_std_by_sign.png")
    plt.clf()