from argparse import ArgumentParser
import os
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
from multiprocessing import Pool

if __name__ == "__main__":
    robust = True
    # arguments
    def process_version(version_path):
        stats = []
        root_dir = Path(version_path)

        for results_path in root_dir.rglob("results.pkl"):
            figure_dir = root_dir / "figure" / results_path.parent.name
            sns.set_theme()

            figure_dir.mkdir(parents=True, exist_ok=True)
            # load results
            with open(os.path.join(results_path), "rb") as f:
                results = pickle.load(f)

            # Replace NaN values with 0 in results
            for key in results.keys():
                results[key] = np.nan_to_num(results[key])

            # area
            area = results["area"].mean(1)
            gt_area = results["gt_area"].mean(1)
            df = pd.DataFrame(data={"Pred.": area, "True": gt_area})
            lm = sns.regplot(data=df, x="True", y="Pred.", scatter_kws={"alpha":0.5}, robust=robust, x_jitter=0.1)
            lm.set(xlim=(-0.1, 6.1), ylim=(-0.1, 6.1))
            lm.set(xlabel="True area score", ylabel="Predicted area score", title='')
            lm.set(title=f"PCC of {results_path.parent.name}'s area")
            lm.axes.axline((0, 0), slope=1.0, ls="--", color="black", alpha=0.5)

            r, p = pearsonr(gt_area, area, alternative="greater")
            lm.text(0.1, 5.0, "r = {:.3f}, p = {:.3e}".format(r, p))
            lm.text(0.1, 4.0, "Mean avg err = {:.3f}".format(np.mean(np.abs(gt_area - area))))
            stats.append({
                "Metric": "Area",
                "Version": results_path.parent.name,
                "PCC": r,
                "P-value": p,
                "Mean avg err": np.mean(np.abs(gt_area - area))
            })
            plt.tight_layout()
            plt.savefig(figure_dir / f"{results_path.parent.name}_area.svg")
            plt.close()

            # EASI mean
            easi = results["easi"].sum(2).mean(1)
            gt_easi = results["gt_easi"].sum(2).mean(1)
            df = pd.DataFrame(data={"Pred.": easi, "True": gt_easi})
            lm = sns.regplot(data=df, x="True", y="Pred.", scatter_kws={"alpha":0.5}, robust=robust)
            lm.set(xlim=(-1., 73.), ylim=(-1., 73.))
            lm.set(xlabel="True EASI score", ylabel="Predicted EASI score", title='')
            lm.set(title=f"PCC of {results_path.parent.name}'s EASI score")
            lm.axes.axline((0, 0), slope=1.0, ls="--", color="black", alpha=0.5)
            r, p = pearsonr(gt_easi, easi, alternative="greater")
            lm.text(5, 65, "r = {:.3f}, p = {:.3e}".format(r, p))
            lm.text(5, 55, "Mean avg err = {:.3f}".format(np.mean(np.abs(gt_easi - easi))))
            stats.append({
                "Metric": "EASI mean",
                "Version": results_path.parent.name,
                "PCC": r,
                "P-value": p,
                "Mean avg err": np.mean(np.abs(gt_easi - easi))
            })
            plt.tight_layout()
            plt.savefig(figure_dir / f"{results_path.parent.name}_easi.svg")
            plt.close()

            # EASI std
            easi = results["easi"].sum(2).std(1)
            gt_easi = results["gt_easi"].sum(2).std(1)
            df = pd.DataFrame(data={"Pred.": easi, "True": gt_easi})
            lm = sns.scatterplot(data=df, x="True", y="Pred.", color="0.5", alpha=0.5, legend=False)
            lm.set(title=f"Spearman's rho on std of {results_path.parent.name}'s EASI score")
            lm.set(xlim=(-0.5, None), ylim=(-0.1, None))
            lm.set(xlabel="Standard deviation of the true EASI score", ylabel="Standard deviation of the predicted EASI score", title='')
            mean_std = gt_easi.mean()
            std_std = gt_easi.std()
            r, p = spearmanr(gt_easi, easi, alternative="greater")
            lm.text(0.5, 0.5, "r = {:.3f}, p = {:.3e}, std = {:.2f}({:.2f})".format(r, p, mean_std, std_std))
            stats.append({
                "Metric": "EASI std",
                "Version": results_path.parent.name,
                "Spearman's rho": r,
                "P-value": p
            })
            plt.tight_layout()
            plt.savefig(figure_dir / f"{results_path.parent.name}_easi_std.svg")
            plt.close()

            # Severity
            severity = results["severity"].mean(1)
            gt_severity = results["gt_severity"].mean(1)
            B = severity.shape[0]
            Sign = np.array(["Erythema", "Induration", "Excoriation", "Lichenification"] * B)

            df = pd.DataFrame(data={"Pred.": severity.reshape(-1), "True": gt_severity.reshape(-1), "Sign": Sign})
            lm = sns.lmplot(
                data=df, x="True", y="Pred.", col="Sign", hue="Sign", x_jitter=.05, robust=robust, scatter_kws={"alpha":0.5}
            )
            lm.set(xlim=(-0.1, 3.1), ylim=(-0.1, 3.1))
            lm.set(xlabel="True severity score by sign", ylabel="Predicted severity score by sign", title='')
            lm.figure.suptitle(f"PCC of {results_path.parent.name}'s severity score")
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
                stats.append({
                    "Metric": f"Severity {['Erythema', 'Induration', 'Excoriation', 'Lichenification'][i]}",
                    "Version": results_path.parent.name,
                    "PCC": r,
                    "P-value": p,
                    "Mean avg err": np.mean(np.abs(gt_severity[:, i] - severity[:, i]))
                })
            plt.tight_layout()
            plt.savefig(figure_dir / f"{results_path.parent.name}_severity.svg")
            plt.close()

            # EASI mean by sign
            easi = results["easi"].mean(1)
            gt_easi = results["gt_easi"].mean(1)
            df = pd.DataFrame(data={"Pred.": easi.reshape(-1), "True": gt_easi.reshape(-1), "Sign": Sign})
            lm = sns.lmplot(data=df, x="True", y="Pred.", col="Sign", hue="Sign", scatter_kws={"alpha":0.5}, legend=False, robust=robust)
            lm.set(xlim=(-0.5, 18.5), ylim=(-0.5, 18.5))
            lm.set(xlabel="True EASI score by sign", ylabel="Predicted EASI score by sign", title='')
            lm.figure.suptitle(f"PCC of {results_path.parent.name}'s EASI score by sign")
            axes = lm.axes
            for i in range(len(["Erythema", "Induration", "Excoriation", "Lichenification"])):
                r, p = pearsonr(gt_easi[:, i], easi[:, i], alternative="greater")
                axes[0, i].axline((0, 0), slope=1.0, ls="--", color="black", alpha=0.5)
                axes[0, i].annotate("r = {:.3f}, p = {:.3e}".format(r, p), xy=(0.1, 0.9), xycoords=axes[0, i].transAxes)
                axes[0, i].annotate(
                    "Mean avg err = {:.3f}".format(np.mean(np.abs(gt_easi[:, i] - easi[:, i]))), xy=(0.1, 0.8), xycoords=axes[0, i].transAxes
                )
                stats.append({
                    "Metric": f"EASI mean {['Erythema', 'Induration', 'Excoriation', 'Lichenification'][i]}",
                    "Version": results_path.parent.name,
                    "PCC": r,
                    "P-value": p,
                    "Mean avg err": np.mean(np.abs(gt_easi[:, i] - easi[:, i]))
                })
            plt.tight_layout()
            plt.savefig(figure_dir / f"{results_path.parent.name}_easi_by_sign.svg")
            plt.close()

            # EASI std by sign
            easi = results["easi"].std(1)
            gt_easi = results["gt_easi"].std(1)

            df = pd.DataFrame(data={"Pred.": easi.reshape(-1), "True": gt_easi.reshape(-1), "Sign": Sign})
            lm = sns.lmplot(data=df, x="True", y="Pred.", col="Sign", hue="Sign", scatter_kws={"alpha":0.5}, legend=False, fit_reg=False)
            lm.figure.suptitle(f"Spearman's rho on std of {results_path.parent.name}'s EASI score by sign")
            lm.set(xlim=(-0.5, None), ylim=(0., None))
            lm.set(xlabel="Standard deviation of the true EASI score", ylabel="Standard deviation of the predicted EASI score", title='')
            axes = lm.axes
            for i in range(len(["Erythema", "Induration", "Excoriation", "Lichenification"])):
                r, p = spearmanr(gt_easi[:, i], easi[:, i], alternative="greater")
                mean_std = gt_easi[:, i].mean()
                std_std = gt_easi[:, i].std()
                axes[0, i].annotate("r = {:.3f}, p = {:.3e}, std = {:.2f}({:.2f})".format(r, p, mean_std, std_std), xy=(0.1, 0.9), xycoords=axes[0, i].transAxes)
                stats.append({
                    "Metric": f"EASI std {['Erythema', 'Induration', 'Excoriation', 'Lichenification'][i]}",
                    "Version": results_path.parent.name,
                    "Spearman's rho": r,
                    "P-value": p
                })
            plt.tight_layout()
            plt.savefig(figure_dir / f"{results_path.parent.name}_easi_std_by_sign.svg")
            plt.close()

            # Histogram of GED
            if "ged" in results.keys():
                median = np.mean(results["ged"])
                df = pd.DataFrame(data={"MMD(L1)": results["ged"]})
                sns.histplot(data=df, x="MMD(L1)").set(title=f"Maximum mean discrepency with L1 norm: median = {median:.3f}")
                stats.append({
                    "Metric": "GED",
                    "Version": results_path.parent.name,
                    "Median": median
                })
                plt.tight_layout()
                plt.savefig(figure_dir / f"{results_path.parent.name}_ged.svg")
                plt.close()
        return stats

    root_dir = Path("lightning_logs/0.0_wb_new")
    versions = [version_path for version_path in root_dir.rglob("version_*/")]
    with Pool(len(versions)) as p:
        all_stats = p.map(process_version, versions)

    # Save statistics to CSV
    all_stats_flat = [item for sublist in all_stats for item in sublist]
    stats_df = pd.DataFrame(all_stats_flat)
    stats_df.to_csv(f"{root_dir.name}_statistics.csv", index=False)
