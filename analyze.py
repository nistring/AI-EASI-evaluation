from argparse import ArgumentParser
import os
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # arguments
    parser = ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    args = parser.parse_args()
    root = os.path.join("results", args.exp_name)

    # load results
    with open(os.path.join(root, "results.pkl"), "rb") as f:
        results = pickle.load(f)

    # penguins = sns.load_dataset("penguins")
    
    for cat in ["mean", "std", "entropy", "mi"]:
        data = results[cat]
        x = np.repeat(np.linspace(0.0, 1.0, num=data.shape[0]), data.shape[1])
        data = np.nan_to_num(data.reshape(-1))
        mask = data != 0
        x = [",".join(item) for item in x[mask].astype(str)]
        data = data[mask]
        g = sns.catplot(
            data=pd.DataFrame({cat: data, "lesion_prob":x}), kind="bar",
            x="lesion_prob", y=cat,
            errorbar="sd", palette="dark", alpha=.6, height=6
        )
        g.despine(left=True)
        # g.set_axis_labels("", "Body mass (g)")
        # g.legend.set_title("")
        plt.savefig(os.path.join(root, f"{cat}.png"))
        plt.clf()
