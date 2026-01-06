import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

csv_path = Path(r"E:\AUT\thesis\files\feature_reduction\unpaired_test\Significant_Features_Detailed.csv")


def main():
    # 1. Load the Data
    if not Path(csv_path).exists():
        print(f"Error: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 2. Randomly Select 4 Features
    # We check if there are at least 4 features, otherwise take all of them
    n_sample = min(4, len(df))
    selected_df = df.sample(n=n_sample)  # Add random_state=42 if you want fixed results

    print("Selected Features for Visualization:")
    print(selected_df[['Method', 'Feature']])

    # 3. Prepare Data for Plotting (Long Format)
    plot_data = []

    for index, row in selected_df.iterrows():
        # Create a combined label like "EMD ch08_delta"
        feature_label = f"{row['Method']} {row['Feature']}"

        # Append "Good Group" data
        plot_data.append({
            "Feature": feature_label,
            "Group": "Good Counters",
            "Mean": row['Mean_Difference_Good'],
            "SD": row['SD_Good']
        })

        # Append "Bad Group" data
        plot_data.append({
            "Feature": feature_label,
            "Group": "Bad Counters",
            "Mean": row['Mean_Difference_Bad'],
            "SD": row['SD_Bad']
        })

    plot_df = pd.DataFrame(plot_data)

    # 4. Create the Bar Chart
    sns.set_theme(style="whitegrid")

    # Get unique labels for X-axis ordering
    features = plot_df["Feature"].unique()
    groups = plot_df["Group"].unique()
    n_features = len(features)
    n_groups = len(groups)

    # Setup layout
    fig, ax = plt.subplots(figsize=(12, 7))
    bar_width = 0.35
    indices = np.arange(n_features)

    # Define colors
    colors = {"Good Counters": "#1f77b4", "Bad Counters": "#ff7f0e"}  # Blue and Orange

    # Loop to draw bars manually (to include custom error bars from CSV)
    for i, group in enumerate(groups):
        # Filter data for this group
        group_data = plot_df[plot_df["Group"] == group]

        # Ensure data is aligned with the 'features' order
        group_means = group_data.set_index("Feature").reindex(features)["Mean"]
        group_sds = group_data.set_index("Feature").reindex(features)["SD"]

        # Calculate position offset for grouped bars
        offset = (i - n_groups / 2 + 0.5) * bar_width

        # Draw Bar
        ax.bar(indices + offset, group_means, bar_width,
               label=group, color=colors[group],
               yerr=group_sds, capsize=5,  # <--- Draws the Error Bars (SD)
               alpha=0.9, edgecolor='black')

    # 5. Formatting
    ax.set_xlabel("Feature (Method + Channel_Band)", fontsize=12)
    ax.set_ylabel("Mean Reactivity (Task - Rest)", fontsize=12)
    ax.set_title("Comparison of Good vs Bad Counters (Mean ± SD)", fontsize=14, fontweight='bold')

    ax.set_xticks(indices)
    ax.set_xticklabels(features, rotation=15, ha='right')
    ax.legend(title="Group")
    ax.axhline(0, color='black', linewidth=0.8)  # Add a zero line

    save_path = Path(r"E:\AUT\thesis\files\feature_reduction\unpaired_test") / "Good_vs_Bad_Comparison.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Graph saved to: {save_path}")


if __name__ == "__main__":
    main()