from pathlib import Path
import os
import json
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, t
from datetime import datetime

# ======================
# ====== CONFIG  =======
# ======================
SCRIPT_DIR = Path(".").resolve()
OUT_DIR = SCRIPT_DIR
LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Labels CSV: HAS header. 1st column = participant_id, 5th column = counting_quality in {0,1}
LABELS_CSV = Path(r"/path/to/your/labels.csv")  # <-- EDIT THIS

# Root folder with STFT/CWT/EMD; each method folder contains per-participant CSVs.
ROOT_DIR = Path(r"/path/to/your/features_root")  # <-- EDIT THIS
METHODS = ("STFT", "CWT", "EMD")

# Per-file matrix assumptions (no header in feature files)
N_CHANNELS = 19
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]  # must match your 5 column order
ALPHA_FDR = 0.05  # FDR threshold for "significant" outputs

# ======================
# ===== Logging ========
# ======================
RUN_REPORT = LOG_DIR / "RUN_REPORT.txt"

def log_line(path, text, mode="a"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        f.write(text.rstrip("\n") + "\n")

def log_header():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line(RUN_REPORT, "=" * 80, mode="w")
    log_line(RUN_REPORT, f"Run started: {ts}")
    log_line(RUN_REPORT, f"LABELS_CSV: {LABELS_CSV}")
    log_line(RUN_REPORT, f"ROOT_DIR:   {ROOT_DIR}")
    log_line(RUN_REPORT, f"OUT_DIR:    {OUT_DIR}")
    log_line(RUN_REPORT, f"LOG_DIR:    {LOG_DIR}")
    log_line(RUN_REPORT, f"Assumptions: 19x5 feature files (no header), labels CSV has header")
    log_line(RUN_REPORT, f"Band order: {BAND_NAMES}")
    log_line(RUN_REPORT, f"ID rule: first two filename chars 00..35 -> 0..35")
    log_line(RUN_REPORT, "=" * 80)

# ======================
# === Simple ID parse ==
# ======================
def extract_id_from_filename(path):
    """
    Assumes filename starts with a two-digit participant index: 00..35
    Returns integer 0..35
    """
    stem = Path(path).stem
    if len(stem) < 2 or not stem[:2].isdigit():
        raise ValueError(f"Filename must start with two digits (00..35): {Path(path).name}")
    pid = int(stem[:2])
    if not (0 <= pid <= 35):
        raise ValueError(f"Two-digit ID out of range 00..35 in: {Path(path).name}")
    return pid

# ======================
# === Label handling ===
# ======================
def read_labels_build_dict(labels_csv):
    """
    Labels CSV HAS header. Use column positions:
      col 1 (index 0): participant_id
      col 5 (index 4): counting_quality in {0,1}
    Returns dict {0: [ids], 1: [ids]}.
    """
    df = pd.read_csv(labels_csv, header=0)
    pid = df.iloc[:, 0].astype(int)
    qual = df.iloc[:, 4].astype(int)
    return {0: pid[qual == 0].tolist(), 1: pid[qual == 1].tolist()}

# ======================
# === Stats helpers  ===
# ======================
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
    m = max(1, len(pvals))
    order = np.argsort(pvals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)
    q = pvals * m / ranks
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    qvals = np.empty_like(pvals, dtype=float)
    qvals[order] = q_sorted
    return qvals

# ======================
# === IO & shaping   ===
# ======================
def channel_labels(n):
    return [f"ch{idx:02d}" for idx in range(1, n + 1)]

def read_19x5_feature_matrix(path):
    """
    Read a per-participant 19x5 CSV (NO HEADER) and return a single wide row:
    participant_id + (ch01_delta ... ch19_gamma) = 95 features.
    """
    mat = pd.read_csv(path, header=None)

    if mat.shape != (N_CHANNELS, len(BAND_NAMES)):
        raise ValueError(f"{Path(path).name}: expected shape {(N_CHANNELS, len(BAND_NAMES))}, got {mat.shape}")

    mat.columns = BAND_NAMES
    mat.index = channel_labels(N_CHANNELS)

    flat = {}
    for ch in mat.index:
        for band in BAND_NAMES:
            flat[f"{ch}_{band}"] = float(mat.at[ch, band])

    pid = extract_id_from_filename(path)
    row = {"participant_id": pid}
    row.update(flat)
    return pd.DataFrame([row])

def load_and_concat_participants(paths):
    """
    Stack per-participant 19x5 files into one table (one row per participant).
    Aggregates duplicates (if any) by mean.
    """
    if not paths:
        return pd.DataFrame(columns=["participant_id"])
    rows = [read_19x5_feature_matrix(p) for p in sorted(paths)]
    all_df = pd.concat(rows, axis=0, ignore_index=True)
    # Aggregate duplicates by mean (just in case)
    num_cols = ["participant_id"] + [c for c in all_df.columns if c != "participant_id"]
    agg = all_df[num_cols].groupby("participant_id", as_index=False).mean(numeric_only=True)
    return agg

# ======================
# === STFT-like discovery (rest/task by name) ===
# ======================
def discover_files_stft_style(method_root):
    """
    Walk method_root; any .csv with 'rest' in its path/name -> REST, else -> TASK.
    Returns dict {'rest': [...], 'task': [...]} of Path objects.
    """
    rest_files = []
    task_files = []
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

def discover_feature_files(root, methods=METHODS):
    return {m: discover_files_stft_style(root / m) for m in methods}

# ======================
# === Stat pipeline  ===
# ======================
def descriptive_by_feature(merged, base_feats):
    rows = []
    n_pairs = merged.shape[0]
    for f in base_feats:
        a = merged[f + "_task"].to_numpy(dtype=float)  # Task
        b = merged[f + "_rest"].to_numpy(dtype=float)  # Rest
        diff = a - b

        mean_rest = float(np.nanmean(b)); sd_rest = float(np.nanstd(b, ddof=1))
        mean_task = float(np.nanmean(a)); sd_task = float(np.nanstd(a, ddof=1))
        mean_diff = float(np.nanmean(diff)); sd_diff = float(np.nanstd(diff, ddof=1))

        t_stat, p_val = ttest_rel(a, b, nan_policy="omit")
        d_val = cohen_d_paired(a, b)
        g_val = hedges_g_paired(d_val, n_pairs)

        if n_pairs > 1 and not np.isnan(sd_diff):
            se = sd_diff / np.sqrt(n_pairs)
            tcrit = t.ppf(0.975, df=n_pairs - 1)  # 95% two-tailed
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
    base_feats = [c[:-5] for c in feat_cols_rest if (c[:-5] + "_task") in merged.columns]  # strip "_rest"

    stats_df = descriptive_by_feature(merged, base_feats)
    stats_df["p_fdr"] = bh_fdr(stats_df["p"].to_numpy())
    stats_df = stats_df.sort_values(by=["p_fdr", "t"], ascending=[True, False]).reset_index(drop=True)
    return stats_df

def process_method(method_name, rest_files, task_files, labels_dict):
    method_log = LOG_DIR / f"METHOD_{method_name}.txt"
    with open(method_log, "w", encoding="utf-8") as f:
        f.write(f"Processing method: {method_name}\n")
        f.write("-" * 80 + "\n")

    def mlog(s):
        log_line(method_log, s)

    if not rest_files:
        mlog("No REST files found.")
        return pd.DataFrame()
    if not task_files:
        mlog("No TASK files found.")
        return pd.DataFrame()

    mlog(f"REST files: {len(rest_files)} | TASK files: {len(task_files)}")

    rest_df = load_and_concat_participants(rest_files)
    task_df = load_and_concat_participants(task_files)

    if rest_df.empty or task_df.empty:
        mlog("Empty REST or TASK data after loading. Skipping.")
        return pd.DataFrame()

    feats_rest = [c for c in rest_df.columns if c != "participant_id"]
    feats_task = [c for c in task_df.columns if c != "participant_id"]
    common_feats = sorted(set(feats_rest).intersection(feats_task))
    if not common_feats:
        mlog("No common feature columns between REST and TASK.")
        return pd.DataFrame()

    # Reduce to common features
    rest_df = rest_df[["participant_id"] + common_feats].copy()
    task_df = task_df[["participant_id"] + common_feats].copy()

    method_frames = []
    sig_frames = []

    for label in (0, 1):
        group_ids = labels_dict.get(label, [])
        mlog(f"Label group {label}: candidate IDs = {len(group_ids)}")

        group_df = run_group_full_stats(rest_df, task_df, group_ids)
        if group_df.empty:
            mlog(f"Label {label}: insufficient matched pairs or no data. Skipped.")
            continue

        n_pairs = int(group_df["n"].iloc[0]) if not group_df.empty else 0
        mlog(f"Label {label}: matched pairs used = {n_pairs}, features tested = {group_df.shape[0]}")

        group_df.insert(1, "label_group", label)
        method_frames.append(group_df)
        sig_frames.append(group_df[group_df["p_fdr"] <= ALPHA_FDR].copy())

    if not method_frames:
        mlog("No results generated for any label in this method.")
        return pd.DataFrame(columns=[
            "method", "feature", "label_group", "n",
            "mean_rest", "sd_rest", "mean_task", "sd_task",
            "mean_diff", "sd_diff", "t", "p", "d_paired", "g_paired", "ci95_low", "ci95_high", "p_fdr"
        ])

    full = pd.concat(method_frames, axis=0, ignore_index=True)
    full.insert(0, "method", method_name)
    out_full = OUT_DIR / f"ttest_results_{method_name}.csv"
    full.to_csv(out_full, index=False)
    mlog(f"Full results saved to: {out_full}")

    sig = pd.concat(sig_frames, axis=0, ignore_index=True) if any(len(df) for df in sig_frames) else pd.DataFrame(columns=full.columns)
    out_sig = OUT_DIR / f"significant_{method_name}.csv"
    sig.to_csv(out_sig, index=False)
    mlog(f"Significant (p_fdr <= {ALPHA_FDR}) saved to: {out_sig}")

    return full

# ======================
# ====== Main ==========
# ======================
def main():
    log_header()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Labels dictionary
    labels_dict = read_labels_build_dict(LABELS_CSV)
    labels_json_path = OUT_DIR / "labels_dict.json"
    labels_json_path.write_text(json.dumps(labels_dict, indent=2))
    log_line(RUN_REPORT, f"Label dictionary saved: {labels_json_path}")
    log_line(RUN_REPORT, f"Label counts -> bad(0): {len(labels_dict.get(0, []))} | good(1): {len(labels_dict.get(1, []))}")

    # Discover files per method (no file-list logs, per your request)
    discovered = discover_feature_files(ROOT_DIR, methods=METHODS)
    log_line(RUN_REPORT, "Discovery complete for all methods.")

    # Process methods
    all_frames = []
    for method in METHODS:
        paths = discovered.get(method, {})
        full_df = process_method(method, paths.get("rest", []), paths.get("task", []), labels_dict)
        if not full_df.empty:
            all_frames.append(full_df)
            log_line(RUN_REPORT, f"[{method}] rows written: {full_df.shape[0]}")

    # Combined output
    if all_frames:
        all_df = pd.concat(all_frames, axis=0, ignore_index=True)
        out_all = OUT_DIR / "ttest_results_ALL.csv"
        all_df.to_csv(out_all, index=False)
        log_line(RUN_REPORT, f"Combined ALL results: {out_all}")

        sig_all = all_df[all_df["p_fdr"] <= ALPHA_FDR].copy()
        out_sig_all = OUT_DIR / "significant_ALL.csv"
        sig_all.to_csv(out_sig_all, index=False)
        log_line(RUN_REPORT, f"Combined significant (p_fdr <= {ALPHA_FDR}): {out_sig_all}")
        log_line(RUN_REPORT, f"Total significant rows: {sig_all.shape[0]}")
    else:
        log_line(RUN_REPORT, "No results generated for any method.")

    log_line(RUN_REPORT, "=" * 80)
    log_line(RUN_REPORT, "Run finished.")

if __name__ == "__main__":
    main()
