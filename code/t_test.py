from pathlib import Path
import os
import json
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, t

out_dir = Path(r'E:\AUT\thesis\files\feature_reduction')
labels_csv = Path(r"E:\AUT\thesis\EEG-analysis-during-mental-arithmetic-task\subject-info.csv")
root_feature_dir = Path(r"E:\AUT\thesis\files\features")

methods = ("STFT", "CWT", "EMD")
n_channels = 19
band_names = ["delta", "theta", "alpha", "beta", "gamma"]  # ensure this matches your file column order


def extract_id(path):
    stem = Path(path).stem
    return int(stem[:2])  # assumes filenames start with 2-digit ID 00..35


def id_label_extraction(labels_csv):
    df = pd.read_csv(labels_csv, header=0)
    col0 = df.iloc[:, 0].astype(str).str.strip()
    pid = col0.str.extract(r'(\d{2})', expand=False).astype(int)
    qual = pd.to_numeric(df.iloc[:, 5], errors="coerce").astype(int)  # 0/1
    return {0: pid[qual == 0].tolist(), 1: pid[qual == 1].tolist()}


def cohen_d_paired(a, b):
    diff = np.asarray(a - b).reshape(-1)
    sd = diff.std(ddof=1)
    return np.nan if sd == 0 else diff.mean() / sd


def hedges_g_paired(d, n):
    if n <= 1 or np.isnan(d):
        return np.nan
    J = 1.0 - (3.0 / (4.0 * (n - 1) - 1.0))
    return d * J


def bh_fdr(pvals):
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
    rows = []
    n_pairs = merged.shape[0]
    for f in base_feats:
        a = merged[f + "_task"].to_numpy(dtype=float)
        b = merged[f + "_rest"].to_numpy(dtype=float)
        diff = a - b

        mean_rest = float(np.nanmean(b)); sd_rest = float(np.nanstd(b, ddof=1))
        mean_task = float(np.nanmean(a)); sd_task = float(np.nanstd(a, ddof=1))
        mean_diff = float(np.nanmean(diff)); sd_diff = float(np.nanstd(diff, ddof=1))

        t_stat, p_val = ttest_rel(a, b, nan_policy="omit")
        d_val = cohen_d_paired(a, b)
        g_val = hedges_g_paired(d_val, n_pairs)

        if n_pairs > 1 and not np.isnan(sd_diff):
            se = sd_diff / np.sqrt(n_pairs)
            tcrit = t.ppf(0.975, df=n_pairs - 1)
            ci_low = float(mean_diff - tcrit * se)
            ci_high = float(mean_diff + tcrit * se)
        else:
            ci_low = np.nan; ci_high = np.nan

        rows.append((f, n_pairs, mean_rest, sd_rest, mean_task, sd_task,
                     mean_diff, sd_diff, float(t_stat), float(p_val), d_val, g_val, ci_low, ci_high))

    return pd.DataFrame(rows, columns=[
        "feature", "n",
        "mean_rest", "sd_rest", "mean_task", "sd_task",
        "mean_diff", "sd_diff", "t", "p", "d_paired", "g_paired", "ci95_low", "ci95_high"
    ])


def run_group_full_stats(rest_df, task_df, group_ids):
    rest_sub = rest_df[rest_df["participant_id"].isin(group_ids)].copy()
    task_sub = task_df[task_df["participant_id"].isin(group_ids)].copy()
    merged = pd.merge(rest_sub, task_sub, on="participant_id", suffixes=("_rest", "_task"))
    if merged.shape[0] < 2:
        return pd.DataFrame()

    feat_cols_rest = [c for c in merged.columns if c.endswith("_rest")]
    base_feats = [c[:-5] for c in feat_cols_rest if (c[:-5] + "_task") in merged.columns]

    stats_df = descriptive_by_feature(merged, base_feats)
    stats_df["p_fdr"] = bh_fdr(stats_df["p"].to_numpy())
    stats_df = stats_df.sort_values(by=["p_fdr", "t"], ascending=[True, False]).reset_index(drop=True)
    return stats_df


def process_method(method_name, rest_files, task_files, labels_dict):
    if not rest_files or not task_files:
        return pd.DataFrame()

    rest_df = stack_features(rest_files)
    task_df = stack_features(task_files)
    if rest_df.empty or task_df.empty:
        return pd.DataFrame()

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
            "mean_diff", "sd_diff", "t", "p", "d_paired", "g_paired", "ci95_low", "ci95_high", "p_fdr"
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
