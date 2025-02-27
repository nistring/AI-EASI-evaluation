import pandas as pd

# Load the CSV file
for file_path in ['weight2_statistics.csv']:
    # file_path = 'statistics.csv'
    df = pd.read_csv(file_path)

    # Define the number of groups and the size of each group
    num_groups = 1
    group_size = 32 * 5

    # Initialize lists to store the mean and std for each group
    means = []
    stds = []

    # Calculate mean and std for each group
    # Calculate mean and std for each group by metric and version
    for i in range(num_groups):
        group_name = f'Group {i + 1}'
        group_df = df.iloc[i * group_size:(i + 1) * group_size]
        group_means = group_df.groupby(['Metric', 'Version']).mean(numeric_only=True).reset_index()
        group_stds = group_df.groupby(['Metric', 'Version']).std(numeric_only=True).reset_index()
        group_means['Group'] = group_name
        group_stds['Group'] = group_name
        means.append(group_means)
        stds.append(group_stds)

    # Concatenate all group DataFrames
    means_df = pd.concat(means, ignore_index=True)
    stds_df = pd.concat(stds, ignore_index=True)

    # Print the results
    print("Means of each group by metric and version:")
    print(means_df)
    print("\nStandard deviations of each group by metric and version:")
    print(stds_df)

    means_df.to_csv(f'{file_path}_means.csv', index=False)
    stds_df.to_csv(f'{file_path}_stds.csv', index=False)

# for i in range(num_groups):
#     group_df = df.iloc[i * group_size:(i + 1) * group_size]
#     group_mean = group_df.mean(numeric_only=True)
#     group_std = group_df.std(numeric_only=True)
#     means.append(group_mean)
#     stds.append(group_std)

# # Convert lists to DataFrames for better readability
# means_df = pd.DataFrame(means)
# stds_df = pd.DataFrame(stds)

# # Print the results
# print("Means of each group:")
# print(means_df)
# print("\nStandard deviations of each group:")
# print(stds_df)