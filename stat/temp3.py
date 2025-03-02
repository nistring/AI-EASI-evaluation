import pandas as pd
import seaborn as sns
from pingouin import intraclass_corr
import matplotlib.pyplot as plt

# Load the Excel file
file_path = "results/Results_expert+AI.xlsx"
sheet_name = "종합"
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Define the assessments
assessments = ["조수익", "이지수", "나정임", "AI_mean"]

# Initialize a dictionary to store the results
results = {assessment: {} for assessment in assessments}

# Calculate the required values per ID for each assessment
# Fill NaN values in 'ID' column with the value above
df = df.ffill()
for assessment in assessments:
    grouped = df.groupby("ID")
    for id, group in grouped:
        hn_front = group[assessment][group["Weight"] == "HN_front"].item()
        hn_rear = group[assessment][group["Weight"] == "HN_rear"].item()
        hn_weighted_mean = (3 * hn_front + hn_rear) / 4
        trunk_mean = group[assessment][group["Weight"].str.startswith("Trunk")].mean()
        arm_mean = group[assessment][group["Weight"].str.startswith("Arm")].mean()
        leg_mean = group[assessment][group["Weight"].str.startswith("Leg")].mean()

        results[assessment][id] = {"HN": hn_weighted_mean, "Trunk": trunk_mean, "Arm": arm_mean, "Leg": leg_mean}
        results[assessment][id]["Total"] = group[f"{assessment}합계"].iloc[0].item()

# Calculate MAE of AI_mean to the mean of experts
for region in ["HN", "Trunk", "Arm", "Leg", "Total"]:
    experts = pd.DataFrame({assessment: [results[assessment][id][region] for id in results[assessment]] for assessment in assessments[:-1]})
    experts_mean = experts.mean(axis=1)
    experts_std = experts.std(axis=0)
    ai_mean = pd.Series([results["AI_mean"][id][region] for id in results["AI_mean"]])
    mae = (ai_mean - experts_mean).abs().mean()
    print(f"MAE of AI_mean to the mean of experts in {region}: {mae}", experts_std.mean())


# Initialize a dictionary to store the correlation matrices
correlation_matrices = {region: pd.DataFrame(index=assessments, columns=assessments) for region in ["HN", "Trunk", "Arm", "Leg", "Total"]}
# Calculate the intra-class correlation coefficients
for region in correlation_matrices:
    for assessment1 in assessments:
        for assessment2 in assessments:
            if assessment1 != assessment2:
                values = [results[assessment1][id][region] for id in results[assessment1]] + [
                    results[assessment2][id][region] for id in results[assessment2]
                ]
                icc = intraclass_corr(
                    data=pd.DataFrame(
                        {
                            "ID": list(results[assessment1].keys()) + list(results[assessment2].keys()),
                            "Value": values,
                            "Rater": ["Rater1"] * len(results[assessment1]) + ["Rater2"] * len(results[assessment2]),
                        }
                    ),
                    targets="ID",
                    raters="Rater",
                    ratings="Value",
                )
                correlation_matrices[region].loc[assessment1, assessment2] = icc.loc[icc["Type"] == "ICC2", "ICC"].item()
            else:
                correlation_matrices[region].loc[assessment1, assessment2] = 1
print(correlation_matrices)

# Plot the correlation matrices and save as SVG in the directory 'figure'
# fig, axes = plt.subplots(2, 4, figsize=(15, 10))
fig = plt.figure(figsize=(10, 5), layout="constrained")
subfigs = fig.subfigures(1, 2, width_ratios=[1,1])
regions = ["HN", "Trunk", "Arm", "Leg"]

axes = subfigs[0].subplots(2, 2)  # , sharex=True, sharey=True)

for ax, region, title in zip(axes.flatten(), regions, ["(a) Head and neck", "(b) Trunk", "(c) Upper limbs", "(d) Lower limbs"]):
    sns.heatmap(correlation_matrices[region].astype(float), annot=True, ax=ax, vmin=0.0, vmax=1, annot_kws={"size": 8}, cbar=False)
    ax.set_title(title, loc="left", size=8)
    ax.set_xticklabels(["1", "2", "3", "AI"], size=8, rotation=0)
    ax.set_yticklabels(["1", "2", "3", "AI"], size=8, rotation=0)
    ax.set_aspect("equal")

ax5 = subfigs[1].add_subplot(aspect="equal")
sns.heatmap(
    correlation_matrices["Total"].astype(float),
    annot=True,
    ax=ax5,
    vmin=0.0,
    vmax=1,
    cbar_kws={"ticks": [0, 1], "shrink": 0.8},
    annot_kws={"size": 10},
)
ax5.set_title("(e) ICC matrix of total EASI score", loc="left", size=10)
ax5.set_xticklabels(["1", "2", "3", "AI"], rotation=0, size=10)
ax5.set_yticklabels(["1", "2", "3", "AI"], rotation=0, size=10)
ax5.set_position([0.05, 0.1, 0.8, 0.8])  # Adjust the position and size of ax5

# plt.tight_layout()

# Save the figure as SVG
output_path = "figure/correlation_matrices.svg"
plt.savefig(output_path, format="svg")
