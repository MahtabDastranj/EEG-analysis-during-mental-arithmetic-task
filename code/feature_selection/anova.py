from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import fdrcorrection

# ================= CONFIGURATION =================
root_feature_dir = Path(r"E:\AUT\thesis\files\features")
out_dir = Path(r"E:\AUT\thesis\files\feature_reduction\anova")
methods = ("STFT", "CWT", "EMD")
n_channels = 19
band_names = ["delta", "theta", "alpha", "beta", "gamma"]


# =================================================

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
    Reads feature file and normalizes to Relative Power (0-1).
    Returns a dictionary row for the dataframe.
    """
    try:
        mat = pd.read_csv(path, header=None)
    except Exception:
        return None

    if mat.shape != (n_channels, len(band_names)):
        return None

    # 1. Normalize to Relative Power (Row-wise)
    mat_values = mat.values.astype(np.float64)
    row_sums = mat_values.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1e-12  # Safety
    mat_norm = mat_values / row_sums

    # 2. Flatten to 1D Row
    df_norm = pd.DataFrame(mat_norm, columns=band_names, index=channel_labels(n_channels))
    flat = {}
    for ch in df_norm.index:
        for band in band_names:
            flat[f"{ch}_{band}"] = float(df_norm.at[ch, band])

    # 3. Add Metadata
    pid = extract_id(path)
    # Identify if this is a 'rest' or 'task' file
    state = "rest" if "rest" in str(path).lower() else "task"

    row = {"participant_id": pid, "state": state}
    row.update(flat)
    return row


def load_method_data(method_name):
    """Loads all files for a specific method into a DataFrame."""
    path = root_feature_dir / method_name
    all_rows = []
    for dirpath, _, filenames in os.walk(path):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                row = read_and_normalize_features(Path(dirpath) / fn)
                if row:
                    all_rows.append(row)
    return pd.DataFrame(all_rows)


def run_statistical_pipeline(df_stft, df_cwt, df_emd, scenario_label):
    """
    Runs Friedman -> Post-hoc Wilcoxon -> FDR pipeline.
    Expects 3 inputs of N=36 for each feature.
    """
    # Identify feature columns (excluding metadata)
    keys = ["participant_id", "state"]
    feature_cols = [c for c in df_stft.columns if c not in keys]

    results = []
    print(f"Running Statistics for: {scenario_label} (Subjects: {len(df_stft)})")

    for feat in feature_cols:
        # Extract the 3 vectors of 36x1
        v1 = df_stft[feat].values.astype(np.float64)  # STFT vector
        v2 = df_cwt[feat].values.astype(np.float64)  # CWT vector
        v3 = df_emd[feat].values.astype(np.float64)  # EMD vector

        # Safety Check: Skip flat lines (zero variance)
        if np.all(v1 == v1[0]) and np.all(v2 == v2[0]) and np.all(v3 == v3[0]):
            continue

        # --- STEP 1: Friedman Test (Omnibus) ---
        # Checks if there is ANY difference among the 3 methods
        try:
            stat, p_friedman = friedmanchisquare(v1, v2, v3)
        except ValueError:
            continue

        # --- STEP 2: Post-Hoc Wilcoxon (Pairwise) ---
        # Only strictly necessary if Friedman is significant, but good to report.
        def get_pval(x, y):
            try:
                # 'two-sided' is standard for "is there a difference?"
                return wilcoxon(x, y).pvalue
            except:
                return 1.0

        p_stft_cwt = get_pval(v1, v2)
        p_stft_emd = get_pval(v1, v3)
        p_cwt_emd = get_pval(v2, v3)

        # Store descriptive stats to see WHO is higher/lower
        results.append({
            "scenario": scenario_label,
            "feature": feat,
            "mean_STFT": np.mean(v1),
            "mean_CWT": np.mean(v2),
            "mean_EMD": np.mean(v3),
            "friedman_stat": stat,
            "p_friedman": p_friedman,
            "p_STFT_vs_CWT": p_stft_cwt,
            "p_STFT_vs_EMD": p_stft_emd,
            "p_CWT_vs_EMD": p_cwt_emd
        })

    if not results:
        return pd.DataFrame()

    res_df = pd.DataFrame(results)

    # 3: FDR Correction ---
    # We apply FDR to the Friedman P-values to control family-wise error across 95 features
    reject, p_fdr = fdrcorrection(res_df["p_friedman"], alpha=0.05)
    res_df["p_friedman_fdr"] = p_fdr
    res_df["significant"] = reject

    # Sort by the "Disagreement Score" (Friedman Statistic)
    return res_df.sort_values("friedman_stat", ascending=False)


def main():
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("Loading data from all methods...")
    all_data = {m: load_method_data(m) for m in methods}
    keys = ["participant_id", "state"]

    # 2. Synchronize Data (Ensure we have exactly 36 subjects for ALL methods)
    # We merge to find the intersection of subjects present in STFT, CWT, and EMD
    common = pd.merge(all_data["STFT"][keys], all_data["CWT"][keys], on=keys)
    common = pd.merge(common, all_data["EMD"][keys], on=keys)

    # Filter original dataframes to strictly match the common list
    df_stft = pd.merge(common, all_data["STFT"], on=keys).sort_values(keys)
    df_cwt = pd.merge(common, all_data["CWT"], on=keys).sort_values(keys)
    df_emd = pd.merge(common, all_data["EMD"], on=keys).sort_values(keys)

    # 3. Define Scenarios (Vectors of 36x1)
    # We run the test twice: once for Task data, once for Rest data.
    scenarios = [
        ("Task_State", df_stft["state"] == "task"),
        ("Rest_State", df_stft["state"] == "rest")
    ]

    all_results = []

    for label, mask in scenarios:
        # Filter for the 36 participants in this state
        sub_stft = df_stft[mask]
        sub_cwt = df_cwt[mask]
        sub_emd = df_emd[mask]

        if len(sub_stft) == 0:
            print(f"Skipping {label} (No data found)")
            continue

        # Run the statistics
        scenario_res = run_statistical_pipeline(sub_stft, sub_cwt, sub_emd, label)
        if not scenario_res.empty:
            all_results.append(scenario_res)

    # 4. Save Final Report
    if all_results:
        final_df = pd.concat(all_results, axis=0, ignore_index=True)
        final_path = out_dir / "Method_Comparison_Friedman_Results.csv"
        final_df.to_csv(final_path, index=False)
        print(f"\nSUCCESS. Results saved to:\n{final_path}")
        print("Check 'significant' column to see where methods disagree.")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()