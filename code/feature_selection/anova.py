import os
import re
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import friedmanchisquare
from statsmodels.stats.multitest import multipletests

root_feature_dir = Path(r"E:\AUT\thesis\files\features")
out_dir = Path(r"E:\AUT\thesis\files\feature_reduction\anova")
methods = ("STFT", "CWT", "EMD")
n_channels = 19
band_names = ["delta", "theta", "alpha", "beta", "gamma"]
out_dir.mkdir(parents=True, exist_ok=True)


def extract_id(path):
    stem = Path(path).stem
    m = re.search(r'(\d{2})', stem)
    if not m:
        raise ValueError(f"Cannot extract 2-digit participant id from filename: {stem}")
    return int(m.group(1))


def channel_labels(n):
    return [f"ch{idx:02d}" for idx in range(1, n + 1)]


def read_and_normalize_features(path):
    ''' Forms: participant id| state| it 95 normalized channels'''
    try:
        mat = pd.read_csv(path, header=None)
    except Exception:
        return None

    if mat.shape != (n_channels, len(band_names)):
        return None

    mat_values = mat.values.astype(np.float64)

    # For each channel, the amount for the 5 frequency bands are scaled to 1 for a better method evaluation
    row_sums = mat_values.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-12
    mat_norm = mat_values / row_sums

    df_norm = pd.DataFrame(mat_norm, columns=band_names, index=channel_labels(n_channels))

    flat = {}
    for ch in df_norm.index:
        for band in band_names:
            flat[f"{ch}_{band}"] = float(df_norm.loc[ch, band])

    pid = extract_id(path)
    state = "rest" if "rest" in str(path).lower() else "task"

    row = {
        "participant_id": pid,
        "state": state
    }
    row.update(flat)

    return row


def load_method_data(method_name):
    method_path = root_feature_dir / method_name
    rows = []

    for dirpath, _, filenames in os.walk(method_path):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                row = read_and_normalize_features(Path(dirpath) / fn)
                if row is not None:
                    rows.append(row)

    df = pd.DataFrame(rows)
    print(f"[{method_name}] Loaded DataFrame shape: {df.shape}")
    # Expected: (72 rows, 97 columns) → 36 subjects × 2 states
    return df


def compute_task_rest_difference(df, method_name):
    feature_cols = [c for c in df.columns if c not in ("participant_id", "state")]

    pivot = df.pivot(
        index="participant_id",
        columns="state",
        values=feature_cols
    )

    print(f"[{method_name}] Pivot table shape: {pivot.shape}")
    # Expected: (36 participants, 95 features × 2 states)

    diff = (pivot.xs("task", level=1, axis=1) - pivot.xs("rest", level=1, axis=1))

    print(f"[{method_name}] Task–Rest difference shape: {diff.shape}")
    # Expected: (36 participants, 95 features)

    return diff


# Load data
method_dfs = {method: load_method_data(method) for method in methods}

# Task – Rest differences
diffs = {
    method: compute_task_rest_difference(df, method)
    for method, df in method_dfs.items()
}

features = diffs[methods[0]].columns
print(f"Number of features: {len(features)}")  # Expected: 95

# Friedman tests
results = []
for i, feature in enumerate(features):
    data = np.column_stack([
        diffs[method][feature].values
        for method in methods
    ])

    valid_mask = ~np.isnan(data).any(axis=1)
    data = data[valid_mask]

    if data.shape[0] < 10:
        continue

    stat, p_value = friedmanchisquare(
        data[:, 0],
        data[:, 1],
        data[:, 2]
    )

    print(data.shape)

    results.append({
        "feature": feature,
        "friedman_statistic": stat,
        "p_value": p_value
    })


results_df = pd.DataFrame(results)
print(f"Results DataFrame shape: {results_df.shape}")  # Expected: (95 rows, 3 columns)

results_df["p_fdr"] = multipletests(
    results_df["p_value"],
    alpha=0.05,
    method="fdr_bh"
)[1]

results_df["significant"] = results_df["p_fdr"] < 0.05

output_path = out_dir / "friedman_method_comparison_results.csv"
results_df.to_csv(output_path, index=False)

print(f"\nAnalysis complete. Results saved to:\n{output_path}")
print(f"Significant features: {results_df['significant'].sum()} / {len(results_df)}")

random.seed(42)

# Get feature lists
significant_features = results_df.loc[results_df["significant"], "feature"].tolist()

nonsignificant_features = results_df.loc[~results_df["significant"], "feature"].tolist()

# Randomly sample 5 features from each group
sig_sample = random.sample(significant_features, min(5, len(significant_features)))

nonsig_sample = random.sample(nonsignificant_features, min(5, len(nonsignificant_features)))


def plot_feature_comparison(feature_list, title, filename):
    methods = ["STFT", "CWT", "EMD"]
    n_features = len(feature_list)

    x = np.arange(n_features)
    width = 0.25

    plt.figure(figsize=(12, 6))

    for i, method in enumerate(methods):
        means = []
        stds = []

        for feature in feature_list:
            values = diffs[method][feature].dropna().values
            means.append(np.mean(values))
            stds.append(np.std(values))

        plt.bar(
            x + i * width,
            means,
            width=width,
            yerr=stds,
            capsize=6,
            label=method
        )

    plt.xticks(
        ticks=x + width,
        labels=feature_list,
        rotation=30,
        ha="right"
    )

    plt.ylabel("Task − Rest Relative Power")
    plt.xlabel("Feature (Channel_Band)")
    plt.title(title)
    plt.legend(title="Extraction Method")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # 💾 Save figure
    save_path = out_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {save_path}")


plot_feature_comparison(sig_sample, title="Method Comparison: Significant Features",
                        filename="significant_features_comparison.png")
plot_feature_comparison(nonsig_sample,title="Method Comparison: Non-Significant Features",
                        filename="nonsignificant_features_comparison.png")
