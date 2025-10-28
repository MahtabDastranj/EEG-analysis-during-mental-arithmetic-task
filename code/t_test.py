"""
Feature selection via paired t-tests (Task vs Rest) for Good (1) and Bad (0) counters,
computed separately for STFT, CWT, and EMD feature sets.

What it does:
1) Reads your labels CSV (1st col = participant_id, 5th col = counting_quality in {0,1}).
2) Builds a dict: {0: [...bad_ids], 1: [...good_ids]}.
3) For each method (STFT, CWT, EMD), loads REST and TASK CSVs with 'participant_id' + feature columns.
4) Within each label group (0,1), aligns participants and runs paired t-tests per feature (Task vs Rest).
5) Adds Benjamini–Hochberg FDR (p_fdr) and paired Cohen's d, exports per-method results to CSV,
   plus top-K by |t| per group.

Outputs (saved next to this script by default):
- labels_dict.json
- ttest_results_<METHOD>.csv
- ttest_top_<METHOD>_label<0-or-1>.csv

Notes:
- CSVs for each method must share identical feature column names between REST and TASK files.
- If a group has < 2 matched participants between REST and TASK, that group's stats are skipped.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel


SCRIPT_DIR = Path(".").resolve()
OUT_DIR = SCRIPT_DIR  # change if you want outputs elsewhere

# Path to the labels CSV:
# - Column 1: participant_id (index number)
# - Column 5: counting_quality (0=bad, 1=good)
LABELS_CSV = Path("/path/to/your/labels.csv")  # <-- EDIT THIS

# Feature file paths for each method.
# Each CSV must have a 'participant_id' column + feature columns with identical names in REST/TASK files.
FEATURE_PATHS = {
    "STFT": {
        "rest": Path("/path/to/STFT_rest.csv"),  # <-- EDIT THIS
        "task": Path("/path/to/STFT_task.csv"),  # <-- EDIT THIS
    },
    "CWT": {
        "rest": Path("/path/to/CWT_rest.csv"),   # <-- EDIT THIS
        "task": Path("/path/to/CWT_task.csv"),   # <-- EDIT THIS
    },
    "EMD": {
        "rest": Path("/path/to/EMD_rest.csv"),   # <-- EDIT THIS
        "task": Path("/path/to/EMD_task.csv"),   # <-- EDIT THIS
    },
}

# Number of top features to export (by absolute t-statistic) per label group.
TOP_K = 30


# ======================
# === Helper funcs  ====
# ======================
def read_labels_build_dict(labels_csv: Path) -> dict:
    """
    Read the labels CSV and build {0: [ids], 1: [ids]} from col1 (ID) and col5 (quality).
    Assumes CSV has no header.
    """
    df = pd.read_csv(labels_csv, header=None)
    pid = df.iloc[:, 0].astype(int)
    qual = df.iloc[:, 4].astype(int)
    label_dict = {
        0: pid[qual == 0].tolist(),
        1: pid[qual == 1].tolist()
    }
    return label_dict


def cohen_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d for paired samples, comparing arrays a (Task) vs b (Rest).
    d = mean(a-b) / std(a-b, ddof=1)
    """
    diff = a - b
    if diff.ndim != 1:
        diff = diff.reshape(-1)
    sd = diff.std(ddof=1)
    if sd == 0:
        return np.nan
    return diff.mean() / sd


def run_group_ttests(rest_df: pd.DataFrame, task_df: pd.DataFrame, group_ids: list) -> pd.DataFrame:
    """
    Paired t-tests (Task vs Rest) within the given label group.
    Both DataFrames must include 'participant_id' and identical feature columns.
    Returns a DataFrame with columns: feature, t, p, d, n, p_fdr (BH).
    """
    # Filter to participants in the group and align by participant_id
    rest_sub = rest_df[rest_df["participant_id"].isin(group_ids)].copy()
    task_sub = task_df[task_df["participant_id"].isin(group_ids)].copy()
    merged = pd.merge(rest_sub, task_sub, on="participant_id", suffixes=("_rest", "_task"))

    if merged.empty or merged.shape[0] < 2:
        return pd.DataFrame()  # Not enough pairs

    feat_cols_rest = [c for c in merged.columns if c.endswith("_rest")]
    feat_cols_task = [c for c in merged.columns if c.endswith("_task")]

    # Determine base feature names present in both
    base_feats = [c.replace("_rest", "") for c in feat_cols_rest]
    base_feats = [f for f in base_feats if (f + "_task") in feat_cols_task]

    results = []
    n_pairs = merged.shape[0]

    for f in base_feats:
        a = merged[f + "_task"].to_numpy(dtype=float)  # Task
        b = merged[f + "_rest"].to_numpy(dtype=float)  # Rest
        t_stat, p_val = ttest_rel(a, b, nan_policy="omit")
        d_val = cohen_d_paired(a, b)
        results.append((f, t_stat, p_val, d_val, n_pairs))

    res_df = pd.DataFrame(results, columns=["feature", "t", "p", "d", "n"])

    # Benjamini–Hochberg FDR
    p = res_df["p"].to_numpy()
    m = len(p)
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)
    fdr = p * m / ranks
    # enforce monotonicity on sorted p-values
    fdr_sorted = np.minimum.accumulate(fdr[order][::-1])[::-1]
    p_fdr = np.empty_like(p, dtype=float)
    p_fdr[order] = fdr_sorted
    res_df["p_fdr"] = p_fdr

    # Sort by |t| descending for readability
    res_df = res_df.sort_values(by="t", key=lambda s: np.abs(s), ascending=False).reset_index(drop=True)
    return res_df


def load_features_csv(path: Path) -> pd.DataFrame:
    """Load a features CSV that must include 'participant_id' + feature columns."""
    df = pd.read_csv(path)
    if "participant_id" not in df.columns:
        raise ValueError(f"'participant_id' column missing in {path}")
    return df


def process_method(method_name: str, rest_path: Path, task_path: Path, labels_dict: dict):
    """
    Run paired t-tests for the given method across both label groups (0,1);
    save results CSVs, plus top-K lists per group.
    """
    rest_df = load_features_csv(rest_path)
    task_df = load_features_csv(task_path)

    # Harmonize feature columns (intersection)
    feat_cols_rest = [c for c in rest_df.columns if c != "participant_id"]
    feat_cols_task = [c for c in task_df.columns if c != "participant_id"]
    common_feats = sorted(list(set(feat_cols_rest).intersection(feat_cols_task)))
    if not common_feats:
        raise ValueError(f"No common feature columns between REST and TASK for {method_name}.")
    rest_df = rest_df[["participant_id"] + common_feats].copy()
    task_df = task_df[["participant_id"] + common_feats].copy()

    all_results = []
    for label in (0, 1):
        group_ids = labels_dict.get(label, [])
        res_df = run_group_ttests(rest_df, task_df, group_ids)
        if res_df.empty:
            print(f"[{method_name}] No valid pairs for label {label}. Skipping.")
            continue

        res_df.insert(1, "label_group", label)  # 0=bad, 1=good
        all_results.append(res_df)

        # Save top-K for this label group
        top_df = res_df.head(TOP_K).copy()
        top_out = OUT_DIR / f"ttest_top_{method_name}_label{label}.csv"
        top_df.to_csv(top_out, index=False)

    if all_results:
        full = pd.concat(all_results, axis=0, ignore_index=True)
        out_path = OUT_DIR / f"ttest_results_{method_name}.csv"
        full.to_csv(out_path, index=False)
        print(f"[{method_name}] Saved: {out_path}")
    else:
        print(f"[{method_name}] No results generated.")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Labels -> dict
    labels_dict = read_labels_build_dict(LABELS_CSV)
    # (Optional) Save the dictionary so you can inspect it
    (OUT_DIR / "labels_dict.json").write_text(json.dumps(labels_dict, indent=2))
    print("Label dictionary:", labels_dict)

    # 2) Process each method
    for method, paths in FEATURE_PATHS.items():
        process_method(method, paths["rest"], paths["task"], labels_dict)


if __name__ == "__main__":
    main()
