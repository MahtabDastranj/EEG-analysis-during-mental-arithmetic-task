import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

csv_path = Path(r"E:\AUT\thesis\files\feature_reduction\unpaired_test\Significant_Features_Detailed.csv")
out_dir = Path(r"E:\AUT\thesis\files\feature_reduction\unpaired_test")
FEATURES_PER_GRAPH = 5


def main():
    # 1. Load the Data
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    total_features = len(df)
    print(f"Loaded {total_features} features.")

    # Calculate how many graphs we need
    # (e.g., 15 features // 5 = 3 graphs)
    num_chunks = (total_features + FEATURES_PER_GRAPH - 1) // FEATURES_PER_GRAPH

    sns.set_theme(style="whitegrid")
    colors = {"Good Counters": "#1f77b4", "Bad Counters": "#ff7f0e"}  # Blue and Orange

    # 2. Loop through chunks
    for i in range(num_chunks):
        start_idx = i * FEATURES_PER_GRAPH
        end_idx = min((i + 1) * FEATURES_PER_GRAPH, total_features)

        # Slice the dataframe for this batch
        batch_df = df.iloc[start_idx:end_idx]

        print(f"Processing Batch {i + 1}: Features {start_idx + 1} to {end_idx}")

        # 3. Prepare Data (Long Format)
        plot_data = []
        for index, row in batch_df.iterrows():
            feature_label = f"{row['Method']}\n{row['Feature']}"

            plot_data.append({
                "Feature": feature_label,
                "Group": "Good Counters",
                "Mean": row['Mean_Difference_Good'],
                "SD": row['SD_Good']
            })
            plot_data.append({
                "Feature": feature_label,
                "Group": "Bad Counters",
                "Mean": row['Mean_Difference_Bad'],
                "SD": row['SD_Bad']
            })

        plot_df = pd.DataFrame(plot_data)

        # 4. Create Plot
        features = plot_df["Feature"].unique()
        groups = plot_df["Group"].unique()
        n_features = len(features)
        n_groups = len(groups)

        fig, ax = plt.subplots(figsize=(12, 6))
        bar_width = 0.35
        indices = np.arange(n_features)

        for j, group in enumerate(groups):
            group_data = plot_df[plot_df["Group"] == group]

            # Align data
            group_means = group_data.set_index("Feature").reindex(features)["Mean"]
            group_sds = group_data.set_index("Feature").reindex(features)["SD"]

            offset = (j - n_groups / 2 + 0.5) * bar_width

            ax.bar(indices + offset, group_means, bar_width,
                   label=group, color=colors[group],
                   yerr=group_sds, capsize=5,
                   alpha=0.9, edgecolor='black')

        # 5. Formatting
        ax.set_xlabel("Feature", fontsize=12)
        ax.set_ylabel("Mean Reactivity (Task - Rest)", fontsize=12)

        # Dynamic Title indicating which features are shown
        ax.set_title(f"Comparison of Good vs Bad Counters (Batch {i + 1}: Features {start_idx + 1}-{end_idx})",
                     fontsize=14, fontweight='bold')

        ax.set_xticks(indices)
        ax.set_xticklabels(features, rotation=0, fontsize=11)
        ax.legend(title="Group")
        ax.axhline(0, color='black', linewidth=0.8)

        # 6. Save
        filename = f"Good_vs_Bad_Comparison_Batch_{i + 1}.png"
        save_path = out_dir / filename

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"Saved graph to: {save_path}")


if __name__ == "__main__":
    main()