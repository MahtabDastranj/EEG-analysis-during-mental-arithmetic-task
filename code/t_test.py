from pathlib import Path
import os
import json
import re
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, norm
from statsmodels.stats.multitest import fdrcorrection

# --- CONFIGURATION ---
out_dir = Path(r'E:\AUT\thesis\files\feature_reduction')
labels_csv = Path(r"E:\AUT\thesis\EEG-analysis-during-mental-arithmetic-task\subject-info.csv")
root_feature_dir = Path(r"E:\AUT\thesis\files\features")

methods = ("STFT", "CWT", "EMD")
n_channels = 19
band_names = ["delta", "theta", "alpha", "beta", "gamma"]


# --- HELPER FUNCTIONS ---

def extract_id(path):
    stem = Path(path).stem
    m = re.search(r'(\d{2})', stem)
    if not m:
        raise ValueError(f"Cannot extract 2-digit participant id from filename: {stem}")
    return int(m.group(1))


def id_label_extraction(labels_csv):
    """
    Reads the CSV and separates IDs into two lists based on the quality column.
    Returns: {0: [list of bad counter IDs], 1: [list of good counter IDs]}
    """
    df = pd.read_csv(labels_csv, header=0)
    # Assuming Column 0 is ID and Column 5 is the Group Label (0 or 1)
    col0 = df.iloc[:, 0].astype(str).str.strip()
    pid = col0.str.extract(r'(\d{2})', expand=False).astype(int)
    qual = pd.to_numeric(df.iloc[:, 5], errors="coerce").astype(int)
    return {0: pid[qual == 0].tolist(), 1: pid[qual == 1].tolist()}


def channel_labels(n):
    return [f"ch{idx:02d}" for idx in range(1, n + 1)]


def read_features(path):
    """
    Reads one file (19x5). Flattens it into 1 row with 95 columns.
    """
    mat = pd.read_csv(path, header=None)
    if mat.shape != (n_channels, len(band_names)):
        # Safety check for file dimensions
        raise ValueError(f"{Path(path).name}: expected {(n_channels, len(band_names))}, got {mat.shape}")
    mat.columns = band_names
    mat.index = channel_labels(n_channels)

    flat = {}
    for ch in mat.index:
        for band in band_names:
            flat[f"{ch}_{band}"] = float(mat.at[ch, band])

    pid = extract_id(path)
    row = {"participant_id": pid}
    row.update(flat)
    return pd.DataFrame([row])


def stack_features(paths):
    if not paths:
        return pd.DataFrame(columns=["participant_id"])
    rows = [read_features(p) for p in sorted(paths)]
    all_df = pd.concat(rows, axis=0, ignore_index=True)
    # Average separate recordings if a participant appears twice in one folder
    num_cols = ["participant_id"] + [c for c in all_df.columns if c != "participant_id"]
    agg = all_df[num_cols].groupby("participant_id", as_index=False).mean(numeric_only=True)
    return agg


def find_files(method_root):
    rest_files, task_files = [], []
    for dirpath, _, filenames in os.walk(method_root):
        for fn in filenames:
            if not fn.lower().endswith(".csv"):
                continue
            fpath = Path(dirpath) / fn
            if "rest" in str(fpath).lower():
                rest_files.append(fpath)
            else:
                task_files.append(fpath)
    return {"rest": rest_files, "task": task_files}


def discover_feature_files(root, methods=methods):
    return {m: find_files(root / m) for m in methods}


# --- STATISTICAL ENGINE (METICULOUS MODE) ---

def descriptive_by_feature(merged, base_feats):
    """
    Performs Wilcoxon Signed-Rank Test for every feature.
    Calculates Effect Size r = Z / sqrt(N).
    """
    rows = []

    # Pre-calculate constants for efficiency
    n_pairs = merged.shape[0]  # This will be 10 for Bad, 26 for Good

    for f in base_feats:
        # Extract Task and Rest vectors
        a = merged[f + "_task"].to_numpy(dtype=float)
        b = merged[f + "_rest"].to_numpy(dtype=float)

        # Identify valid pairs (remove NaNs)
        mask = ~np.logical_or(np.isnan(a), np.isnan(b))
        valid_n = int(mask.sum())

        diff = (a - b)[mask] if valid_n > 0 else np.array([])

        # Descriptive Stats
        mean_rest = float(np.nanmean(b)) if valid_n > 0 else np.nan
        sd_rest = float(np.nanstd(b, ddof=1)) if valid_n > 1 else np.nan
        mean_task = float(np.nanmean(a)) if valid_n > 0 else np.nan
        sd_task = float(np.nanstd(a, ddof=1)) if valid_n > 1 else np.nan
        mean_diff = float(np.nanmean(diff)) if valid_n > 0 else np.nan
        sd_diff = float(np.nanstd(diff, ddof=1)) if valid_n > 1 else np.nan

        # --- WILCOXON TEST ---
        wil_stat = np.nan
        p_val = np.nan
        effect_size_r = np.nan

        # We need at least a few pairs to run a test.
        # Using 6 as a safe minimum for Wilcoxon to yield a p < 0.05 possibility.
        if valid_n >= 6:
            try:
                # method='approx' allows us to approximate Z-score from p-value easily
                # zero_method='wilcox' discards zero-differences (standard practice)
                res = wilcoxon(a[mask], b[mask], zero_method='wilcox', method='approx')
                wil_stat = res.statistic
                p_val = res.pvalue

                # --- EFFECT SIZE CALCULATION ---
                # r = Z / sqrt(N)
                # Recover Z-score from the p-value (inverse normal distribution)
                # We use 1 - p/2 because p-value is 2-tailed
                z_score = norm.ppf(1 - p_val / 2)
                effect_size_r = z_score / np.sqrt(valid_n)

            except Exception:
                # Fails if all differences are zero or N is too small
                pass

        rows.append((
            f, valid_n,
            mean_rest, sd_rest, mean_task, sd_task,
            mean_diff, sd_diff,
            wil_stat, float(p_val), float(effect_size_r)
        ))

    cols = [
        "feature", "n",
        "mean_rest", "sd_rest", "mean_task", "sd_task",
        "mean_diff", "sd_diff",
        "wilcoxon_stat", "p_val", "effect_size_r"
    ]
    return pd.DataFrame(rows, columns=cols)


def run_group_full_stats(rest_df, task_df, group_ids):
    """
    Runs stats for ONE group (Good or Bad) and ONE method.
    """
    # Filter data for specific group IDs
    rest_sub = rest_df[rest_df["participant_id"].isin(group_ids)].copy()
    task_sub = task_df[task_df["participant_id"].isin(group_ids)].copy()

    # Merge on ID
    merged = pd.merge(rest_sub, task_sub, on="participant_id", suffixes=("_rest", "_task"))

    # If not enough subjects in this group, return empty
    if merged.shape[0] < 5:
        return pd.DataFrame()

    # Identify features
    feat_cols_rest = [c for c in merged.columns if c.endswith("_rest")]
    base_feats = [c[:-5] for c in feat_cols_rest if (c[:-5] + "_task") in merged.columns]

    # 1. Run Descriptive + Wilcoxon
    stats_df = descriptive_by_feature(merged, base_feats)

    # 2. FDR CORRECTION (Benjamini-Hochberg)
    # We apply this to the 95 p-values of THIS specific group/method combination
    pvals = stats_df["p_val"].to_numpy(dtype=float)
    valid_mask = ~np.isnan(pvals)

    corrected = np.full_like(pvals, np.nan, dtype=float)
    rejects = np.full_like(pvals, False, dtype=bool)

    if valid_mask.sum() > 0:
        try:
            # alpha=0.05 is the standard threshold
            rej, p_corr = fdrcorrection(pvals[valid_mask], alpha=0.05, method='indep')
            corrected[valid_mask] = p_corr
            rejects[valid_mask] = rej
        except Exception:
            pass

    stats_df["p_fdr"] = corrected
    stats_df["significant"] = rejects

    # Sort: Significant first, then by effect size (descending)
    stats_df = stats_df.sort_values(by=["significant", "effect_size_r"], ascending=[False, False]).reset_index(
        drop=True)
    return stats_df


def process_method(method_name, rest_files, task_files, labels_dict):
    """
    Orchestrates the analysis for one Method (e.g., STFT).
    Splits into Group 0 (Bad) and Group 1 (Good).
    """
    if not rest_files or not task_files:
        print(f"Skipping {method_name}: Files missing.")
        return pd.DataFrame()

    rest_df = stack_features(rest_files)
    task_df = stack_features(task_files)

    # Ensure types
    rest_df['participant_id'] = rest_df['participant_id'].astype(int)
    task_df['participant_id'] = task_df['participant_id'].astype(int)

    # Find common features
    feats_rest = [c for c in rest_df.columns if c != "participant_id"]
    feats_task = [c for c in task_df.columns if c != "participant_id"]
    common_feats = sorted(set(feats_rest).intersection(feats_task))

    rest_df = rest_df[["participant_id"] + common_feats].copy()
    task_df = task_df[["participant_id"] + common_feats].copy()

    method_frames = []

    # --- LOOP THROUGH GROUPS (0: Bad, 1: Good) ---
    group_names = {0: "Bad_Counters", 1: "Good_Counters"}

    for label, id_list in labels_dict.items():
        group_name = group_names.get(label, f"Group_{label}")
        print(f"Processing Method: {method_name} | Group: {group_name} | N={len(id_list)}")

        group_df = run_group_full_stats(rest_df, task_df, id_list)

        if not group_df.empty:
            group_df.insert(0, "group", group_name)
            method_frames.append(group_df)

    if not method_frames:
        return pd.DataFrame()

    # Combine Good and Bad results for this Method
    full = pd.concat(method_frames, axis=0, ignore_index=True)
    full.insert(0, "method", method_name)

    # Save a CSV specifically for this method
    full.to_csv(out_dir / f"Results_{method_name}.csv", index=False)
    return full


def main():
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load IDs
    labels_dict = id_label_extraction(labels_csv)

    # Discover files
    discovered = discover_feature_files(root_feature_dir, methods=methods)

    all_results = []

    # Iterate through methods (STFT, CWT, EMD)
    for method in methods:
        paths = discovered.get(method, {})
        df = process_method(method, paths.get("rest", []), paths.get("task", []), labels_dict)
        if not df.empty:
            all_results.append(df)

    # Master File (Optional, creates a huge file with all methods and groups)
    if all_results:
        master_df = pd.concat(all_results, axis=0, ignore_index=True)
        master_df.to_csv(out_dir / "Results_MASTER_ALL.csv", index=False)
        print("\nDone! Analysis complete. Check the output directory.")


if __name__ == "__main__":
    main()