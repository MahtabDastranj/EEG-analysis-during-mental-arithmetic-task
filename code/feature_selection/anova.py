import os
import re
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
    """Extracts the 2-digit participant ID from the filename."""
    stem = Path(path).stem
    m = re.search(r'(\d{2})', stem)
    if not m:
        raise ValueError(f"Cannot extract 2-digit participant id from filename: {stem}")
    return int(m.group(1))


def channel_labels(n):
    return [f"ch{idx:02d}" for idx in range(1, n + 1)]


def read_and_normalize_features(path):
    """
    Reads feature CSV file and normalizes to Relative Power (row-wise).
    Obtains relative power across frequency bands for each channel.
    Returns a flat dictionary suitable for DataFrame construction.
    """
    try:
        mat = pd.read_csv(path, header=None)
    except Exception:
        return None

    if mat.shape != (n_channels, len(band_names)):
        return None

    mat_values = mat.values.astype(np.float64)

    # Row-wise normalization (Relative Power)
    row_sums = mat_values.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-12
    mat_norm = mat_values / row_sums

    df_norm = pd.DataFrame(
        mat_norm,
        columns=band_names,
        index=channel_labels(n_channels)
    )

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
    """Loads all feature files for one method into a DataFrame."""
    method_path = root_feature_dir / method_name
    rows = []

    for dirpath, _, filenames in os.walk(method_path):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                row = read_and_normalize_features(Path(dirpath) / fn)
                if row is not None:
                    rows.append(row)

    return pd.DataFrame(rows)


def compute_task_rest_difference(df):
    """
    Computes task - rest per participant for all features.
    Output shape: participants × features
    """
    feature_cols = [c for c in df.columns if c not in ("participant_id", "state")]

    pivot = df.pivot(
        index="participant_id",
        columns="state",
        values=feature_cols
    )

    diff = (
        pivot.xs("task", level=1, axis=1) -
        pivot.xs("rest", level=1, axis=1)
    )

    return diff


method_dfs = {
    method: load_method_data(method)
    for method in methods
}

diffs = {
    method: compute_task_rest_difference(df)
    for method, df in method_dfs.items()
}

features = diffs[methods[0]].columns  # same feature set for all methods

results = []

for feature in features:

    data = np.column_stack([
        diffs[method][feature].values
        for method in methods
    ])  # shape: (36 (participants), 3)

    # Remove rows with missing values
    valid_mask = ~np.isnan(data).any(axis=1)
    data = data[valid_mask]

    if data.shape[0] < 10:
        continue  # insufficient data

    stat, p_value = friedmanchisquare(
        data[:, 0],
        data[:, 1],
        data[:, 2]
    )

    results.append({
        "feature": feature,
        "friedman_statistic": stat,
        "p_value": p_value
    })

results_df = pd.DataFrame(results)

results_df["p_fdr"] = multipletests(
    results_df["p_value"],
    alpha=0.05,
    method="fdr_bh"
)[1]

results_df["significant"] = results_df["p_fdr"] < 0.05

output_path = out_dir / "friedman_method_comparison_results.csv"
results_df.to_csv(output_path, index=False)

print(f"Analysis complete. Results saved to:\n{output_path}")
print(f"Significant features: {results_df['significant'].sum()} / {len(results_df)}")
