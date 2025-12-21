from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, norm
from statsmodels.stats.multitest import fdrcorrection

out_dir = Path(r"E:\AUT\thesis\files\feature_reduction\t-test")
labels_csv = Path(r'E:\AUT\thesis\EEG-analysis-during-mental-arithmetic-task\subject-info.csv')
root_feature_dir = Path(r"E:\AUT\thesis\files\features")

methods = ("STFT", "CWT", "EMD")
n_channels = 19
band_names = ["delta", "theta", "alpha", "beta", "gamma"]


def extract_id(path):
    """Extracts the 2-digit participant ID from the filename."""
    stem = Path(path).stem
    m = re.search(r'(\d{2})', stem)
    if not m:
        raise ValueError(f"Cannot extract 2-digit participant id from filename: {stem}")
    return int(m.group(1))


def id_label_extraction(labels_csv):
    """
    Reads subject-info.csv.
    Assumes Column 0 is ID and Column 5 is Group (0=Bad, 1=Good).
    Returns a dictionary mapping IDs to Groups and a clean DataFrame.
    """
    df = pd.read_csv(labels_csv, header=0)

    # Clean and extract IDs
    col0 = df.iloc[:, 0].astype(str).str.strip()
    pid = col0.str.extract(r'(\d{2})', expand=False).astype(int)

    # Extract Group Label (0 or 1)
    qual = pd.to_numeric(df.iloc[:, 5], errors="coerce").astype(int)

    # Create the dictionary for processing
    label_dict = {0: pid[qual == 0].tolist(), 1: pid[qual == 1].tolist()}

    # Create a nice summary DataFrame for the output file
    summary_df = pd.DataFrame({
        "participant_id": pid,
        "group_label": qual,
    })

    return label_dict, summary_df


def channel_labels(n):
    return [f"ch{idx:02d}" for idx in range(1, n + 1)]


def read_features(path):
    """Reads a 19x5 feature file and flattens it to a 1x95 row."""
    mat = pd.read_csv(path, header=None)
    if mat.shape != (n_channels, len(band_names)):
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
    """Combines multiple feature files into one DataFrame."""
    if not paths:
        return pd.DataFrame(columns=["participant_id"])
    rows = [read_features(p) for p in sorted(paths)]
    all_df = pd.concat(rows, axis=0, ignore_index=True)
    # Average if duplicates exist
    num_cols = ["participant_id"] + [c for c in all_df.columns if c != "participant_id"]
    agg = all_df[num_cols].groupby("participant_id", as_index=False).mean(numeric_only=True)
    return agg


def find_files(method_root):
    """Locates rest and task files in the directory."""
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


def descriptive_by_feature(merged, base_feats):
    """
    Calculates Descriptive Stats, Wilcoxon Test, and Effect Size (r).
    MODIFIED: Includes robust effect size calculation using Z-score.
    """
    rows = []

    for f in base_feats:
        a = merged[f + "_task"].to_numpy(dtype=float)
        b = merged[f + "_rest"].to_numpy(dtype=float)

        # Filter valid pairs (no NaNs)
        mask = ~np.logical_or(np.isnan(a), np.isnan(b))
        valid_n = int(mask.sum())

        diff = (a - b)[mask] if valid_n > 0 else np.array([])

        mean_rest = float(np.nanmean(b)) if valid_n > 0 else np.nan
        sd_rest = float(np.nanstd(b, ddof=1)) if valid_n > 1 else np.nan
        mean_task = float(np.nanmean(a)) if valid_n > 0 else np.nan
        sd_task = float(np.nanstd(a, ddof=1)) if valid_n > 1 else np.nan
        mean_diff = float(np.nanmean(diff)) if valid_n > 0 else np.nan
        sd_diff = float(np.nanstd(diff, ddof=1)) if valid_n > 1 else np.nan

        # --- WILCOXON SIGNED-RANK TEST ---
        wil_stat = np.nan
        p_val = np.nan
        effect_size_r = np.nan

        if valid_n >= 5: # Lowered to 5 to accommodate small 'Bad' group
            try:
                # Use 'approx' to ensure we can calculate a Z-score for r
                res = wilcoxon(a[mask], b[mask], zero_method='wilcox', method='approx')
                wil_stat = res.statistic
                p_val = res.pvalue

                # --- EFFECT SIZE (r = |Z| / sqrt(N)) ---
                # Since we used method='approx', we derive Z from the p-value
                z_score = np.abs(norm.ppf(p_val / 2))
                effect_size_r = z_score / np.sqrt(valid_n)

            except Exception:
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
    Runs analysis for ONE group and ONE method.
    Applies FDR correction to the 95 features within this specific batch.
    """
    rest_sub = rest_df[rest_df["participant_id"].isin(group_ids)].copy()
    task_sub = task_df[task_df["participant_id"].isin(group_ids)].copy()

    merged = pd.merge(rest_sub, task_sub, on="participant_id", suffixes=("_rest", "_task"))

    if merged.shape[0] < 5:
        return pd.DataFrame()

    feat_cols_rest = [c for c in merged.columns if c.endswith("_rest")]
    base_feats = [c[:-5] for c in feat_cols_rest if (c[:-5] + "_task") in merged.columns]

    # 1. Run Stats (includes Effect Size r)
    stats_df = descriptive_by_feature(merged, base_feats)

    # 2. Apply FDR Correction
    pvals = stats_df["p_val"].to_numpy(dtype=float)
    valid_mask = ~np.isnan(pvals)

    corrected = np.full_like(pvals, np.nan, dtype=float)
    rejects = np.full_like(pvals, False, dtype=bool)

    if valid_mask.sum() > 0:
        try:
            rej, p_corr = fdrcorrection(pvals[valid_mask], alpha=0.05, method='indep')
            corrected[valid_mask] = p_corr
            rejects[valid_mask] = rej
        except Exception:
            pass

    stats_df["p_fdr"] = corrected
    stats_df["significant"] = rejects

    # Sort: Largest Effect Size first
    stats_df = stats_df.sort_values(by=["effect_size_r"], ascending=False).reset_index(drop=True)
    return stats_df


def process_method(method_name, rest_files, task_files, labels_dict):
    """Splits analysis into Good vs Bad groups for the given method."""
    if not rest_files or not task_files:
        return pd.DataFrame()

    rest_df = stack_features(rest_files)
    task_df = stack_features(task_files)

    rest_df['participant_id'] = rest_df['participant_id'].astype(int)
    task_df['participant_id'] = task_df['participant_id'].astype(int)

    feats_rest = [c for c in rest_df.columns if c != "participant_id"]
    feats_task = [c for c in task_df.columns if c != "participant_id"]
    common_feats = sorted(set(feats_rest).intersection(feats_task))

    rest_df = rest_df[["participant_id"] + common_feats].copy()
    task_df = task_df[["participant_id"] + common_feats].copy()

    method_frames = []
    group_names = {0: "Bad_Counters", 1: "Good_Counters"}

    for label, id_list in labels_dict.items():
        group_name = group_names.get(label, f"Group_{label}")
        print(f"Processing {method_name} | {group_name} | N={len(id_list)}")

        group_df = run_group_full_stats(rest_df, task_df, id_list)

        if not group_df.empty:
            group_df.insert(0, "group", group_name)
            method_frames.append(group_df)

    if not method_frames:
        return pd.DataFrame()

    full = pd.concat(method_frames, axis=0, ignore_index=True)
    full.insert(0, "method", method_name)
    return full


def main():
    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. GENERATE PARTICIPANT GROUP FILE
    print("Loading participant labels...")
    labels_dict, summary_df = id_label_extraction(labels_csv)

    group_file_path = out_dir / "participant_groups.csv"
    summary_df.to_csv(group_file_path, index=False)
    print(f"File 1 Created: {group_file_path}")

    # 2. RUN ANALYSIS & GENERATE MASTER RESULTS FILE
    print("Discovering feature files...")
    discovered = discover_feature_files(root_feature_dir, methods=methods)

    all_results = []

    for method in methods:
        paths = discovered.get(method, {})
        df = process_method(method, paths.get("rest", []), paths.get("task", []), labels_dict)
        if not df.empty:
            all_results.append(df)

    if all_results:
        master_df = pd.concat(all_results, axis=0, ignore_index=True)

        # Save Master File
        master_path = out_dir / "Results.csv"
        master_df.to_csv(master_path, index=False)
        print(f"File 2 Created: {master_path}")

        # --- SELECTION OF MOST IMPORTANT FEATURES ---
        # We select features with p < 0.05 and a "Large" effect size (r > 0.5)
        # These are the ones that actually distinguish task from rest effectively.
        most_important = master_df[
            (master_df["p_val"] < 0.05) &
            (master_df["effect_size_r"] > 0.5)
        ].sort_values(by=["group", "effect_size_r"], ascending=[True, False])

        if not most_important.empty:
            important_path = out_dir / "Most_Significant_Features.csv"
            most_important.to_csv(important_path, index=False)
            print(f"File 3 Created: {important_path} ({len(most_important)} features)")
    else:
        print("No results generated. Check file paths and data.")

    print("\nAnalysis Complete.")


if __name__ == "__main__":
    main()