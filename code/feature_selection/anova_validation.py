import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import random

raw_feature_dir = Path(r"E:\AUT\thesis\files\features")
anova_results_path = Path(r"E:\AUT\thesis\files\feature_reduction\anova\Method_Comparison_Friedman_Results.csv")
out_dir = Path(r"E:\AUT\thesis\files\viz")

methods = ["STFT", "CWT", "EMD"]
n_channels = 19
band_names = ["delta", "theta", "alpha", "beta", "gamma"]

sns.set_theme(style="whitegrid")


def channel_labels(n):
    return [f"ch{idx:02d}" for idx in range(1, n + 1)]


def get_data_for_specific_scenario(method, feature_name, scenario_label):
    """
    Loads raw data for ONE specific feature, filtering files
    based on the scenario (Task or Rest).
    """
    method_dir = raw_feature_dir / method
    values = []

    # Determine what to look for in filenames
    # The ANOVA file uses "Task_State" and "Rest_State"
    is_task_scenario = "task" in scenario_label.lower()

    for root, _, files in os.walk(method_dir):
        for file in files:
            if not file.lower().endswith('.csv'):
                continue

            # Task/rest separation
            filename_clean = file.lower()
            if is_task_scenario and "rest" in filename_clean:
                continue  # We want Task, but this file is Rest -> SKIP
            if not is_task_scenario and "rest" not in filename_clean:
                continue  # We want Rest, but this file is Task -> SKIP

            filepath = Path(root) / file

            try:
                # 1. Read
                mat = pd.read_csv(filepath, header=None)
                if mat.shape != (n_channels, len(band_names)):
                    continue

                # 2. Normalize (Relative Power)
                vals = mat.values.astype(np.float64)
                row_sums = vals.sum(axis=1)[:, np.newaxis]
                row_sums[row_sums == 0] = 1e-12
                mat_norm = vals / row_sums

                # 3. Extract the specific feature
                ch_idx = int(feature_name.split('_')[0].replace('ch', '')) - 1
                band_idx = band_names.index(feature_name.split('_')[1])

                val = mat_norm[ch_idx, band_idx]
                values.append(val)

            except Exception:
                continue

    return values


def main():
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load ANOVA Results
    if not anova_results_path.exists():
        print("Error: ANOVA results file not found.")
        return

    df_anova = pd.read_csv(anova_results_path)

    # 2. Randomly Pick 5 Rows (Feature + Scenario pairs)
    # We sample indices to keep the Feature and Scenario linked
    if len(df_anova) < 5:
        indices = df_anova.index
    else:
        indices = random.sample(list(df_anova.index), 5)

    selected_rows = df_anova.loc[indices]
    print(f"Selected {len(selected_rows)} items for visualization.")

    # 3. Calculate Mean/SD for these specific pairs
    plot_data = []

    for idx, row in selected_rows.iterrows():
        feat = row['feature']
        scenario = row['scenario']  # e.g., "Task_State"

        print(f"Processing: {feat} ({scenario})")

        for method in methods:
            # Get values strictly for this scenario
            vals = get_data_for_specific_scenario(method, feat, scenario)
            vals = np.array(vals)

            if len(vals) == 0:
                continue

            plot_data.append({
                "Label": f"{feat}\n({scenario})",  # Unique label for X-axis
                "Method": method,
                "Mean": np.mean(vals),
                "SD": np.std(vals)
            })

    df_plot = pd.DataFrame(plot_data)

    # 4. Visualization
    plt.figure(figsize=(14, 7))

    ax = sns.barplot(
        data=df_plot,
        x="Label",
        y="Mean",
        hue="Method",
        palette="viridis",
        edgecolor="black",
        alpha=0.9
    )

    # 5. Add Error Bars
    n_methods = len(methods)
    n_items = len(selected_rows)

    for i, method in enumerate(methods):
        method_data = df_plot[df_plot["Method"] == method]
        # Align data to X-axis labels
        # Note: We must ensure the order matches the unique labels on X-axis
        unique_labels = df_plot["Label"].unique()
        method_data = method_data.set_index("Label").reindex(unique_labels).reset_index()

        # Calculate positions
        bar_width = 0.8 / n_methods
        x_coords = np.arange(n_items) + (i - 1) * bar_width

        plt.errorbar(
            x=x_coords,
            y=method_data["Mean"],
            yerr=method_data["SD"],
            fmt='none',
            ecolor='black',
            capsize=5,
            elinewidth=1.5
        )

    plt.title("Method Comparison: Relative Power", fontsize=16, fontweight='bold')
    plt.ylabel("Relative Power", fontsize=12)
    plt.xlabel("Feature (Scenario)", fontsize=12)
    plt.legend(title="Extraction Method")

    # Save
    save_path = out_dir / "Scenario_Specific_BarChart.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Chart saved to: {save_path}")


if __name__ == "__main__":
    main()