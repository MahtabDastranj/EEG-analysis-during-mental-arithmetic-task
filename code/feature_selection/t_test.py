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
    """Extracts the 2-digit participant ID from the filename string."""
    stem = Path(path).stem
    m = re.search(r'(\d{2})', stem)
    if not m:
        raise ValueError(f"Cannot extract 2-digit participant id from filename: {stem}")
    return int(m.group(1))


def id_label_extraction(labels_csv):
    """
    Reads subject-info.csv and maps IDs to performance groups.
    Assumes Column 0 is ID and Column 5 is Group (0=Bad, 1=Good).
    """
    df = pd.read_csv(labels_csv, header=0)

    # Clean and extract IDs
    col0 = df.iloc[:, 0].astype(str).str.strip()
    pid = col0.str.extract(r'(\d{2})', expand=False).astype(int)

    # Extract Group Label (0 or 1)
    qual = pd.to_numeric(df.iloc[:, 5], errors="coerce").astype(int)

    # Create dictionary for group processing
    label_dict = {0: pid[qual == 0].tolist(), 1: pid[qual == 1].tolist()}

    # Create summary for verification
    summary_df = pd.DataFrame({
        "participant_id": pid,
        "group_label": qual,
    })

    return label_dict, summary_df


def channel_labels(n):
    """Generates generic channel labels (ch01, ch02...)."""
    return [f"ch{idx:02d}" for idx in range(1, n + 1)]


def read_features(path):
    """
    Reads a 19x5 feature file and flattens it.
    IMPORTANT: Applies LOG TRANSFORMATION (log10) to absolute power.
    EEG power
    """
    mat = pd.read_csv(path, header=None)
    if mat.shape != (n_channels, len(band_names)):
        raise ValueError(f"{Path(path).name}: expected {(n_channels, len(band_names))}, got {mat.shape}")

    mat.columns = band_names
    mat.index = channel_labels(n_channels)

    flat = {}
    for ch in mat.index:
        for band in band_names:
            # We add 1e-12 (epsilon) to prevent log(0) errors
            val = float(mat.at[ch, band])
            flat[f"{ch}_{band}"] = np.log10(val + 1e-12)

    pid = extract_id(path)
    row = {"participant_id": pid}
    row.update(flat)
    return pd.DataFrame([row])


def stack_features(paths):
    """Combines all participant feature files into a single DataFrame."""
    if not paths:
        return pd.DataFrame(columns=["participant_id"])

    rows = [read_features(p) for p in sorted(paths)]
    all_df = pd.concat(rows, axis=0, ignore_index=True)

    # Average if duplicates exist for the same participant ID
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
    """Maps methods to their respective file paths."""
    return {m: find_files(root / m) for m in methods}


def descriptive_by_feature(merged, base_feats):
    """
    Calculates Descriptive Stats, Wilcoxon Test, and Effect Size (r).
    Direction indicates if log-power increased or decreased.
    """
    rows = []

    for f in base_feats:
        a = merged[f + "_task"].to_numpy(dtype=float)
        b = merged[f + "_rest"].to_numpy(dtype=float)

        # Filter valid pairs where both rest and task data exist
        mask = ~np.logical_or(np.isnan(a), np.isnan(b))
        valid_n = int(mask.sum())

        diff = (a - b)[mask] if valid_n > 0 else np.array([])

        # Descriptive stats (Log-Power means)
        mean_rest = float(np.nanmean(b)) if valid_n > 0 else np.nan
        sd_rest = float(np.nanstd(b, ddof=1)) if valid_n > 1 else np.nan
        mean_task = float(np.nanmean(a)) if valid_n > 0 else np.nan
        sd_task = float(np.nanstd(a, ddof=1)) if valid_n > 1 else np.nan
        mean_diff = float(np.nanmean(diff)) if valid_n > 0 else np.nan
        sd_diff = float(np.nanstd(diff, ddof=1)) if valid_n > 1 else np.nan

        # Task > Rest in log-scale means an increase in absolute power
        direction = "Task > Rest" if mean_diff > 0 else "Rest > Task" if mean_diff < 0 else "No Change"

        # WILCOXON SIGNED-RANK TEST (Non-parametric paired comparison)
        wil_stat = np.nan
        p_val = np.nan
        effect_size_r = np.nan

        if valid_n >= 5:
            try:
                res = wilcoxon(a[mask], b[mask], zero_method='wilcox', method='approx')
                wil_stat = res.statistic
                p_val = res.pvalue

                # EFFECT SIZE (r = |Z| / sqrt(N))
                # Clip p-value to avoid infinite Z-scores
                p_clipped = np.clip(p_val, 1e-15, 1.0)
                z_score = np.abs(norm.ppf(p_clipped / 2.0))
                effect_size_r = z_score / np.sqrt(valid_n)
            except Exception:
                pass

        rows.append((
            f, valid_n, direction,
            mean_rest, sd_rest, mean_task, sd_task,
            mean_diff, sd_diff,
            wil_stat, float(p_val), float(effect_size_r)
        ))

    cols = [
        "feature", "n", "direction",
        "mean_rest", "sd_rest", "mean_task", "sd_task",
        "mean_diff", "sd_diff",
        "wilcoxon_stat", "p_val", "effect_size_r"
    ]
    return pd.DataFrame(rows, columns=cols)


def run_group_full_stats(rest_df, task_df, group_ids):
    """Runs analysis for a specific participant group and applies FDR correction."""
    rest_sub = rest_df[rest_df["participant_id"].isin(group_ids)].copy()
    task_sub = task_df[task_df["participant_id"].isin(group_ids)].copy()

    merged = pd.merge(rest_sub, task_sub, on="participant_id", suffixes=("_rest", "_task"))

    if merged.shape[0] < 5:
        return pd.DataFrame()

    feat_cols_rest = [c for c in merged.columns if c.endswith("_rest")]
    base_feats = [c[:-5] for c in feat_cols_rest if (c[:-5] + "_task") in merged.columns]

    # Calculate stats
    stats_df = descriptive_by_feature(merged, base_feats)

    # Apply FDR (False Discovery Rate) Correction across all features for this group
    pvals = stats_df["p_val"].to_numpy(dtype=float)
    valid_mask = ~np.isnan(pvals)

    corrected = np.full_like(pvals, np.nan, dtype=float)
    rejects = np.full_like(pvals, False, dtype=bool)

    if valid_mask.sum() > 0:
        try:
            # Alpha = 0.05 is the standard threshold for significance
            rej, p_corr = fdrcorrection(pvals[valid_mask], alpha=0.05, method='indep')
            corrected[valid_mask] = p_corr
            rejects[valid_mask] = rej
        except Exception:
            pass

    stats_df["p_fdr"] = corrected
    stats_df["significant"] = rejects

    # Sorting by effect size helps identify the most discriminative features first
    stats_df = stats_df.sort_values(by=["effect_size_r"], ascending=False).reset_index(drop=True)
    return stats_df


def process_method(method_name, rest_files, task_files, labels_dict):
    """Processes a specific feature extraction method (STFT, CWT, or EMD)."""
    if not rest_files or not task_files:
        return pd.DataFrame()

    rest_df = stack_features(rest_files)
    task_df = stack_features(task_files)

    # Identify features common to both rest and task sets
    common_feats = sorted(set(rest_df.columns).intersection(task_df.columns).difference({"participant_id"}))
    rest_df = rest_df[["participant_id"] + common_feats].copy()
    task_df = task_df[["participant_id"] + common_feats].copy()

    method_frames = []
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

    full = pd.concat(method_frames, axis=0, ignore_index=True)
    full.insert(0, "method", method_name)
    return full


def main():
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Map participant IDs to their respective performance groups
    print("Loading participant labels...")
    labels_dict, summary_df = id_label_extraction(labels_csv)
    summary_df.to_csv(out_dir / "participant_groups.csv", index=False)

    # 2. Locate all feature files on disk
    print("Discovering feature files...")
    discovered = discover_feature_files(root_feature_dir, methods=methods)

    all_results = []

    # 3. Iterate through methods and run statistics
    for method in methods:
        paths = discovered.get(method, {})
        df = process_method(method, paths.get("rest", []), paths.get("task", []), labels_dict)
        if not df.empty:
            all_results.append(df)

    # 4. Save results to disk
    if all_results:
        master_df = pd.concat(all_results, axis=0, ignore_index=True)
        master_df.to_csv(out_dir / "Results.csv", index=False)

        # Generate a curated file of features with high statistical significance and large effect size
        # r >= 0.3 is generally considered a medium-to-large effect in neuroscience
        most_important = master_df[
            (master_df["p_val"] < 0.05) &
            (master_df["effect_size_r"] >= 0.3)
            ].sort_values(by=["group", "effect_size_r"], ascending=[True, False])

        if not most_important.empty:
            most_important.to_csv(out_dir / "Significant_Features.csv", index=False)
            print(f"Analysis Complete. Master file and Significant_Features file created.")
    else:
        print("No results generated. Check input directories.")


if __name__ == "__main__":
    main()