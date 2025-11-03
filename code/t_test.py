from pathlib import Path
import os
import json
import re
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, t, shapiro, wilcoxon
from statsmodels.stats.multitest import fdrcorrection

out_dir = Path(r'E:\AUT\thesis\files\feature_reduction')
labels_csv = Path(r"E:\AUT\thesis\EEG-analysis-during-mental-arithmetic-task\subject-info.csv")
root_feature_dir = Path(r"E:\AUT\thesis\files\features")

methods = ("STFT", "CWT", "EMD")
n_channels = 19
band_names = ["delta", "theta", "alpha", "beta", "gamma"]


def extract_id(path):
    stem = Path(path).stem
    m = re.search(r'(\d{2})', stem)
    if not m:
        raise ValueError(f"Cannot extract 2-digit participant id from filename: {stem}")
    return int(m.group(1))


def id_label_extraction(labels_csv):
    df = pd.read_csv(labels_csv, header=0)
    col0 = df.iloc[:, 0].astype(str).str.strip()
    pid = col0.str.extract(r'(\d{2})', expand=False).astype(int)
    qual = pd.to_numeric(df.iloc[:, 5], errors="coerce").astype(int)  # 0/1
    return {0: pid[qual == 0].tolist(), 1: pid[qual == 1].tolist()}


def cohen_d_paired(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    mask = ~np.logical_or(np.isnan(a), np.isnan(b))
    if mask.sum() < 2:
        return np.nan
    diff = a[mask] - b[mask]
    sd = diff.std(ddof=1)
    return np.nan if sd == 0 else diff.mean() / sd


def hedges_g_paired(d, n):
    if n <= 1 or np.isnan(d):
        return np.nan
    J = 1.0 - (3.0 / (4.0 * (n - 1) - 1.0))
    return d * J


def bh_fdr(pvals):
    # keep this helper, but we'll prefer statsmodels.fdrcorrection for final accuracy
    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return p
    m = p.size
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)
    q = p * m / ranks
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    out = np.empty_like(p, dtype=float)
    out[order] = np.clip(q_sorted, 0.0, 1.0)
    return out


def channel_labels(n):
    return [f"ch{idx:02d}" for idx in range(1, n + 1)]


def read_features(path):
    """Return one-row DataFrame: participant_id + chXX_band (95 cols)."""
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
    if not paths:
        return pd.DataFrame(columns=["participant_id"])
    rows = [read_features(p) for p in sorted(paths)]
    all_df = pd.concat(rows, axis=0, ignore_index=True)
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


def descriptive_by_feature(merged, base_feats):
    """
    For each base feature (e.g., 'ch01_delta'), compute:
      - descriptive stats for rest and task
      - paired t-test
      - Shapiro test on differences (if n >= 3)
      - Wilcoxon fallback if non-normal (and feasible)
      - Cohen's d, Hedges' g
      - 95% CI for mean difference
    After computing p-values we will apply FDR correction outside (so we return raw analysis p-values).
    """
    rows = []
    n_pairs = merged.shape[0]

    for f in base_feats:
        a = merged[f + "_task"].to_numpy(dtype=float)
        b = merged[f + "_rest"].to_numpy(dtype=float)
        # mask pairs where either is nan
        mask = ~np.logical_or(np.isnan(a), np.isnan(b))
        valid_n = int(mask.sum())

        diff = (a - b)[mask] if valid_n > 0 else np.array([])

        mean_rest = float(np.nanmean(b)) if valid_n > 0 else np.nan
        sd_rest = float(np.nanstd(b, ddof=1)) if valid_n > 1 else np.nan
        mean_task = float(np.nanmean(a)) if valid_n > 0 else np.nan
        sd_task = float(np.nanstd(a, ddof=1)) if valid_n > 1 else np.nan
        mean_diff = float(np.nanmean(diff)) if valid_n > 0 else np.nan
        sd_diff = float(np.nanstd(diff, ddof=1)) if valid_n > 1 else np.nan

        # paired t-test (operate on full arrays but nan_policy omitted by ttest_rel, use mask selection)
        if valid_n >= 2:
            try:
                t_stat, p_val_t = ttest_rel(a[mask], b[mask])
            except Exception:
                t_stat, p_val_t = np.nan, np.nan
        else:
            t_stat, p_val_t = np.nan, np.nan

        # Shapiro's normality test on differences when sample size >= 3 (scipy's shapiro needs n>=3)
        sh_p = np.nan
        if valid_n >= 3:
            try:
                sh_p = float(shapiro(diff)[1])
            except Exception:
                sh_p = np.nan

        # Wilcoxon fallback if non-normal and at least 2 paired samples
        wil_p = np.nan
        wil_stat = np.nan
        use_wilcoxon = False
        if valid_n >= 2 and not np.isnan(sh_p) and sh_p < 0.05:
            # attempt Wilcoxon (two-sided)
            try:
                # wilcoxon requires paired arrays with more than zero non-zero diffs in some SciPy versions:
                wil_stat, wil_p = wilcoxon(a[mask], b[mask])
                use_wilcoxon = True
            except Exception:
                wil_stat, wil_p = np.nan, np.nan
                use_wilcoxon = False

        # Decide which p to use for multiple comparison correction (prefer Wilcoxon when used)
        analysis_p = wil_p if use_wilcoxon and not np.isnan(wil_p) else p_val_t

        d_val = cohen_d_paired(a, b)
        g_val = hedges_g_paired(d_val, n_pairs)

        # CI for mean difference (t-based) when enough samples
        if valid_n > 1 and not np.isnan(sd_diff):
            se = sd_diff / np.sqrt(valid_n)
            tcrit = t.ppf(0.975, df=valid_n - 1)
            ci_low = float(mean_diff - tcrit * se)
            ci_high = float(mean_diff + tcrit * se)
        else:
            ci_low = np.nan
            ci_high = np.nan

        rows.append((
            f, valid_n,
            mean_rest, sd_rest, mean_task, sd_task,
            mean_diff, sd_diff, float(t_stat), float(p_val_t),
            sh_p, wil_stat, wil_p, bool(use_wilcoxon),
            float(d_val) if not np.isnan(d_val) else np.nan,
            float(g_val) if not np.isnan(g_val) else np.nan,
            ci_low, ci_high,
            float(analysis_p) if not np.isnan(analysis_p) else np.nan
        ))

    cols = [
        "feature", "n",
        "mean_rest", "sd_rest", "mean_task", "sd_task",
        "mean_diff", "sd_diff", "t", "p_t",
        "shapiro_p", "wilcoxon_stat", "p_wilcoxon", "wilcoxon_used",
        "d_paired", "g_paired", "ci95_low", "ci95_high",
        "analysis_p"
    ]
    return pd.DataFrame(rows, columns=cols)


def run_group_full_stats(rest_df, task_df, group_ids):
    rest_sub = rest_df[rest_df["participant_id"].isin(group_ids)].copy()
    task_sub = task_df[task_df["participant_id"].isin(group_ids)].copy()
    merged = pd.merge(rest_sub, task_sub, on="participant_id", suffixes=("_rest", "_task"))
    if merged.shape[0] < 2:
        return pd.DataFrame()

    feat_cols_rest = [c for c in merged.columns if c.endswith("_rest")]
    base_feats = [c[:-5] for c in feat_cols_rest if (c[:-5] + "_task") in merged.columns]

    stats_df = descriptive_by_feature(merged, base_feats)

    # Multiple comparison correction (Benjamini-Hochberg) on analysis_p
    # Use statsmodels.fdrcorrection which returns reject boolean array and corrected p-values
    pvals = stats_df["analysis_p"].to_numpy(dtype=float)
    # Some features may have NaN p-values (insufficient data). fdrcorrection does not accept NaNs.
    # We'll operate on the non-NaN subset, then put results back
    valid_mask = ~np.isnan(pvals)
    corrected = np.full_like(pvals, np.nan, dtype=float)
    rejects = np.full_like(pvals, False, dtype=bool)
    if valid_mask.sum() > 0:
        try:
            rej, p_corr = fdrcorrection(pvals[valid_mask], alpha=0.05, method='indep')
            corrected[valid_mask] = p_corr
            rejects[valid_mask] = rej
        except Exception:
            # fallback to local BH implementation if something goes wrong
            corrected_vals = bh_fdr(pvals[valid_mask])
            corrected[valid_mask] = corrected_vals
            rejects[valid_mask] = corrected_vals < 0.05

    stats_df["p_fdr"] = corrected
    stats_df["reject_fdr"] = rejects

    stats_df = stats_df.sort_values(by=["p_fdr", "t"], ascending=[True, False]).reset_index(drop=True)
    return stats_df


def process_method(method_name, rest_files, task_files, labels_dict):
    if not rest_files or not task_files:
        return pd.DataFrame()

    rest_df = stack_features(rest_files)
    task_df = stack_features(task_files)
    if rest_df.empty or task_df.empty:
        return pd.DataFrame()

    # ensure participant_id type consistency
    rest_df['participant_id'] = rest_df['participant_id'].astype(int)
    task_df['participant_id'] = task_df['participant_id'].astype(int)

    feats_rest = [c for c in rest_df.columns if c != "participant_id"]
    feats_task = [c for c in task_df.columns if c != "participant_id"]
    common_feats = sorted(set(feats_rest).intersection(feats_task))
    if not common_feats:
        return pd.DataFrame()

    rest_df = rest_df[["participant_id"] + common_feats].copy()
    task_df = task_df[["participant_id"] + common_feats].copy()

    method_frames = []
    for label in (0, 1):
        group_ids = labels_dict.get(label, [])
        group_df = run_group_full_stats(rest_df, task_df, group_ids)
        if group_df.empty:
            continue
        group_df.insert(1, "label_group", label)
        method_frames.append(group_df)

    if not method_frames:
        return pd.DataFrame(columns=[
            "method", "feature", "label_group", "n",
            "mean_rest", "sd_rest", "mean_task", "sd_task",
            "mean_diff", "sd_diff", "t", "p_t",
            "shapiro_p", "wilcoxon_stat", "p_wilcoxon", "wilcoxon_used",
            "d_paired", "g_paired", "ci95_low", "ci95_high",
            "analysis_p", "p_fdr", "reject_fdr"
        ])

    full = pd.concat(method_frames, axis=0, ignore_index=True)
    full.insert(0, "method", method_name)
    full.to_csv(out_dir / f"ttest_results_{method_name}.csv", index=False)
    return full


def main():
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_dict = id_label_extraction(labels_csv)
    (out_dir / "labels_dict.json").write_text(json.dumps(labels_dict, indent=2))

    discovered = discover_feature_files(root_feature_dir, methods=methods)

    all_frames = []
    for method in methods:
        paths = discovered.get(method, {})
        full_df = process_method(method, paths.get("rest", []), paths.get("task", []), labels_dict)
        if not full_df.empty:
            all_frames.append(full_df)

    if all_frames:
        all_df = pd.concat(all_frames, axis=0, ignore_index=True)
        all_df.to_csv(out_dir / "ttest_results_ALL.csv", index=False)


if __name__ == "__main__":
    main()
