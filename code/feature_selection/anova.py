from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
from statsmodels.stats.multitest import fdrcorrection


root_feature_dir = Path(r"E:\AUT\thesis\files\features")
out_dir = Path(r"E:\AUT\thesis\files\feature_reduction\anova")
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


def channel_labels(n):
    """Generates labels like ch01, ch02..."""
    return [f"ch{idx:02d}" for idx in range(1, n + 1)]


def read_and_normalize_features(path):
    """
    Reads a feature file and normalizes to Relative Power.
    This ensures all methods are on the same scale (0 to 1) for comparison.
    """
    mat = pd.read_csv(path, header=None)
    if mat.shape != (n_channels, len(band_names)):
        return None

    # Row-wise normalization (Relative Power)
    row_sums = mat.sum(axis=1).values[:, np.newaxis]
    row_sums[row_sums == 0] = 1e-12
    mat_norm = mat.values / row_sums

    df_norm = pd.DataFrame(mat_norm, columns=band_names, index=channel_labels(n_channels))

    # Flatten matrix to a single row dictionary
    flat = {}
    for ch in df_norm.index:
        for band in band_names:
            flat[f"{ch}_{band}"] = float(df_norm.at[ch, band])

    pid = extract_id(path)
    state = "rest" if "rest" in str(path).lower() else "task"

    row = {"participant_id": pid, "state": state}
    row.update(flat)
    return row


def load_method_data(method_name):
    """Iterates through folders to load all CSVs for a specific method."""
    path = root_feature_dir / method_name
    all_rows = []
    for dirpath, _, filenames in os.walk(path):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                row = read_and_normalize_features(Path(dirpath) / fn)
                if row:
                    all_rows.append(row)
    return pd.DataFrame(all_rows)


def run_friedman_for_scenario(df_stft, df_cwt, df_emd, scenario_label):
    """Executes Friedman Test and FDR correction for a specific data subset."""
    keys = ["participant_id", "state"]
    feature_cols = [c for c in df_stft.columns if c not in keys]
    results = []

    print(f"Processing Scenario: {scenario_label} (N={len(df_stft)})")

    for feat in feature_cols:
        v1 = df_stft[feat].values
        v2 = df_cwt[feat].values
        v3 = df_emd[feat].values

        try:
            # Friedman test: compares the rank distributions of the 3 methods
            stat, p_val = friedmanchisquare(v1, v2, v3)
            results.append({
                "scenario": scenario_label,
                "feature": feat,
                "mean_STFT": np.mean(v1),
                "mean_CWT": np.mean(v2),
                "mean_EMD": np.mean(v3),
                "chi_square": stat,
                "p_val": p_val
            })
        except ValueError:
            # Skip if all values are identical (no variance)
            continue

    if not results:
        return pd.DataFrame()

    res_df = pd.DataFrame(results)

    # Apply Benjamini-Hochberg (FDR) correction
    rej, p_fdr = fdrcorrection(res_df["p_val"], alpha=0.05)
    res_df["p_fdr"] = p_fdr
    res_df["significant"] = rej

    return res_df.sort_values("chi_square", ascending=False)


def main():
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data for all three methods
    all_data = {m: load_method_data(m) for m in methods}

    # 2. Alignment: Keep only records present in all 3 methods (Paired Data)
    keys = ["participant_id", "state"]
    common = pd.merge(all_data["STFT"][keys], all_data["CWT"][keys], on=keys)
    common = pd.merge(common, all_data["EMD"][keys], on=keys)

    df_stft = pd.merge(common, all_data["STFT"], on=keys).sort_values(keys)
    df_cwt = pd.merge(common, all_data["CWT"], on=keys).sort_values(keys)
    df_emd = pd.merge(common, all_data["EMD"], on=keys).sort_values(keys)

    # 3. Run Analysis for 3 distinct scenarios
    all_scenarios_results = []

    # Scenario 1: Task Only
    mask_task = df_stft["state"] == "task"
    res_task = run_friedman_for_scenario(
        df_stft[mask_task], df_cwt[mask_task], df_emd[mask_task], "Task_Only"
    )
    all_scenarios_results.append(res_task)

    # Scenario 2: Rest Only
    mask_rest = df_stft["state"] == "rest"
    res_rest = run_friedman_for_scenario(
        df_stft[mask_rest], df_cwt[mask_rest], df_emd[mask_rest], "Rest_Only"
    )
    all_scenarios_results.append(res_rest)

    # Scenario 3: Combined (Rest + Task)
    res_combined = run_friedman_for_scenario(
        df_stft, df_cwt, df_emd, "Combined_All"
    )
    all_scenarios_results.append(res_combined)

    # 4. Save and Summarize Results
    final_report = pd.concat(all_scenarios_results, axis=0, ignore_index=True)
    output_path = out_dir / "Methods_Comparison.csv"
    final_report.to_csv(output_path, index=False)

    print(f"\nAnalysis complete! Results saved to: {output_path}")

    # Print summary of significant features found in each scenario
    for scen in ["Task_Only", "Rest_Only", "Combined_All"]:
        sig_count = final_report[
            (final_report["scenario"] == scen) & (final_report["significant"])
            ].shape[0]
        print(f"Significant features in {scen}: {sig_count}/95")


if __name__ == "__main__":
    main()