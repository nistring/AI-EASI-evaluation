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

# Print the results
for assessment, data in results.items():
    print(f"Assessment: {assessment}")
    for id, values in data.items():
        print(f'ID: {id}, HN: {values["HN"]}, Trunk: {values["Trunk"]}, Arm: {values["Arm"]}, Leg: {values["Leg"]}, Total: {values["Total"]}')

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
                    data=pd.DataFrame({"ID": list(results[assessment1].keys()) + list(results[assessment2].keys()), "Value": values, "Rater": ["Rater1"] * len(results[assessment1]) + ["Rater2"] * len(results[assessment2])}),
                    targets="ID",
                    raters="Rater",
                    ratings="Value",
                )
                correlation_matrices[region].loc[assessment1, assessment2] = icc.loc[icc["Type"]=="ICC2", "ICC"].item()
            else:
                correlation_matrices[region].loc[assessment1, assessment2] = 1
print(correlation_matrices)

# Plot the correlation matrices and save as SVG in the directory 'figure'
fig, axes = plt.subplots(3, 2, figsize=(15, 10))
regions = ['HN', 'Trunk', 'Arm', 'Leg', "Total"]
for ax, region in zip(axes.flatten(), regions):
    sns.heatmap(correlation_matrices[region].astype(float), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title(f'Intra-rater Correlation Matrix for {region}')
plt.tight_layout()

# Save the figure as SVG
output_path = "figure/correlation_matrices.svg"
plt.savefig(output_path, format='svg')
