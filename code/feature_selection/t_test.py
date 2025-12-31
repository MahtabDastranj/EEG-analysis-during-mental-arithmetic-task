from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, norm
from statsmodels.stats.multitest import fdrcorrection

# ================= CONFIGURATION =================
out_dir = Path(r"E:\AUT\thesis\files\feature_reduction\unpaired_test")
labels_csv = Path(r'E:\AUT\thesis\EEG-analysis-during-mental-arithmetic-task\subject-info.csv')
root_feature_dir = Path(r"E:\AUT\thesis\files\features")

methods = ("STFT", "CWT", "EMD")
n_channels = 19
band_names = ["delta", "theta", "alpha", "beta", "gamma"]


# =================================================

def extract_id(path):
    stem = Path(path).stem
    m = re.search(r'(\d{2})', stem)
    if not m:
        raise ValueError(f"Cannot extract 2-digit participant id from filename: {stem}")
    return int(m.group(1))


def get_participant_labels(labels_csv):
    df = pd.read_csv(labels_csv, header=0)
    col0 = df.iloc[:, 0].astype(str).str.strip()
    pids = col0.str.extract(r'(\d{2})', expand=False).astype(int)
    groups = pd.to_numeric(df.iloc[:, 5], errors="coerce").fillna(-1).astype(int)
    return dict(zip(pids, groups))


def channel_labels(n):
    return [f"ch{idx:02d}" for idx in range(1, n + 1)]


def read_features(path):
    try:
        mat = pd.read_csv(path, header=None)
    except Exception:
        return None

    if mat.shape != (n_channels, len(band_names)):
        return None

    mat.columns = band_names
    mat.index = channel_labels(n_channels)

    flat = {}
    for ch in mat.index:
        for band in band_names:
            val = float(mat.at[ch, band])
            flat[f"{ch}_{band}"] = np.log10(val + 1e-12)

    pid = extract_id(path)
    row = {"participant_id": pid}
    row.update(flat)
    return row


def load_all_data_for_method(method_dir):
    rest_rows = []
    task_rows = []
    for root, _, files in os.walk(method_dir):
        for file in files:
            if not file.lower().endswith(".csv"):
                continue
            fpath = Path(root) / file
            row = read_features(fpath)
            if not row:
                continue
            if "rest" in file.lower():
                rest_rows.append(row)
            else:
                task_rows.append(row)
    return pd.DataFrame(rest_rows), pd.DataFrame(task_rows)


def calculate_mann_whitney_stats(df_delta, label_map, method_name):
    # Map IDs to Groups and drop unknowns
    df_delta["group"] = df_delta["participant_id"].map(label_map)
    df_delta = df_delta.dropna(subset=["group"])

    # Split vectors
    df_good = df_delta[df_delta["group"] == 1]
    df_bad = df_delta[df_delta["group"] == 0]

    feature_cols = [c for c in df_delta.columns if c not in ["participant_id", "group"]]
    results = []

    print(f"[{method_name}] Processing {len(feature_cols)} features...")

    for feat in feature_cols:
        vec_good = df_good[feat].to_numpy()
        vec_bad = df_bad[feat].to_numpy()

        # 1. Descriptive Stats
        mean_good = np.mean(vec_good)
        mean_bad = np.mean(vec_bad)
        sd_good = np.std(vec_good, ddof=1)
        sd_bad = np.std(vec_bad, ddof=1)

        # 2. Determine "Task vs Rest" Direction for each group
        # Positive Delta means Task > Rest
        dir_good = "Task > Rest" if mean_good > 0 else "Rest > Task"
        dir_bad = "Task > Rest" if mean_bad > 0 else "Rest > Task"

        # 3. Determine Group Comparison Direction
        # Who had the higher value (more positive or less negative)?
        group_comp = "Good > Bad" if mean_good > mean_bad else "Bad > Good"

        # 4. Mann-Whitney U Test
        try:
            stat, p_val = mannwhitneyu(vec_good, vec_bad, alternative='two-sided')

            # Effect Size (r)
            n1, n2 = len(vec_good), len(vec_bad)
            p_clipped = max(p_val, 1e-15)
            z_score = abs(norm.ppf(p_clipped / 2))
            effect_size_r = z_score / np.sqrt(n1 + n2)

        except Exception:
            stat, p_val, effect_size_r = np.nan, np.nan, np.nan

        results.append({
            "Method": method_name,
            "Feature": feat,
            "Direction_Good": dir_good,
            "Direction_Bad": dir_bad,
            "Comparison": group_comp,
            "Mean_Difference_Good": mean_good,
            "Mean_Difference_Bad": mean_bad,
            "SD_Good": sd_good,
            "SD_Bad": sd_bad,
            "MW_Stat": stat,
            "P_Value": p_val,
            "Effect_Size_r": effect_size_r
        })

    return pd.DataFrame(results)


def main():
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Subject Labels...")
    label_map = get_participant_labels(labels_csv)

    all_results = []

    for method in methods:
        method_path = root_feature_dir / method
        if not method_path.exists():
            continue

        print(f"Processing Method: {method}")
        df_rest, df_task = load_all_data_for_method(method_path)

        if df_rest.empty or df_task.empty:
            continue

        # Average duplicates
        df_rest = df_rest.groupby("participant_id").mean().reset_index()
        df_task = df_task.groupby("participant_id").mean().reset_index()

        # Calculate Delta (Task - Rest)
        merged = pd.merge(df_rest, df_task, on="participant_id", suffixes=('_rest', '_task'))
        feature_cols = [c.replace('_rest', '') for c in merged.columns if '_rest' in c]

        df_delta = pd.DataFrame()
        df_delta["participant_id"] = merged["participant_id"]

        for feat in feature_cols:
            df_delta[feat] = merged[f"{feat}_task"] - merged[f"{feat}_rest"]

        # Run Stats
        method_res = calculate_mann_whitney_stats(df_delta, label_map, method)

        if not method_res.empty:
            # FDR Correction
            pvals = method_res["P_Value"].values
            valid_idx = ~np.isnan(pvals)
            reject = np.full(len(pvals), False, dtype=bool)
            p_fdr = np.full(len(pvals), np.nan)

            if np.sum(valid_idx) > 0:
                reject[valid_idx], p_fdr[valid_idx] = fdrcorrection(pvals[valid_idx], alpha=0.05)

            method_res["P_FDR"] = p_fdr
            method_res["Significant_FDR"] = reject
            all_results.append(method_res)

    if all_results:
        final_df = pd.concat(all_results, axis=0, ignore_index=True)

        # 1. Save Master File
        final_df.to_csv(out_dir / "All_Features_Unpaired_Results.csv", index=False)

        # 2. Filter for SIGNIFICANT features (P < 0.05)
        # We sort by Effect Size to put the strongest findings at the top
        sig_df = final_df[final_df["P_Value"] < 0.05].copy()
        sig_df = sig_df.sort_values(by="Effect_Size_r", ascending=False)

        # Save the requested file
        sig_path = out_dir / "Significant_Features_Detailed.csv"
        sig_df.to_csv(sig_path, index=False)

        print(f"\nSUCCESS!")
        print(f"Total Significant Features Found: {len(sig_df)}")
        print(f"Detailed CSV saved to: {sig_path}")
        print("Check columns 'Direction_Good' and 'Direction_Bad' for Task vs Rest info.")

    else:
        print("No results generated.")


if __name__ == "__main__":
    main()