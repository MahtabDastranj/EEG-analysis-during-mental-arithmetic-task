from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import fdrcorrection

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
root_feature_dir = Path(r"E:\AUT\thesis\files\features")
out_dir = Path(r"E:\AUT\thesis\files\feature_reduction\anova")
methods = ("STFT", "CWT", "EMD")
n_channels = 19
band_names = ["delta", "theta", "alpha", "beta", "gamma"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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
    Ensures methods are compared on a common scale.
    """
    try:
        mat = pd.read_csv(path, header=None)
    except Exception:
        return None

    if mat.shape != (n_channels, len(band_names)):
        return None

    # Row-wise normalization: P_band / P_total_in_range
    # Using float64 for higher precision during comparison
    mat_values = mat.values.astype(np.float64)
    row_sums = mat_values.sum(axis=1)[:, np.newaxis]

    # Avoid division by zero
    row_sums[row_sums == 0] = 1e-12
    mat_norm = mat_values / row_sums

    df_norm = pd.DataFrame(mat_norm, columns=band_names, index=channel_labels(n_channels))

    # Flatten matrix to a single row
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
    path = root_feature_dir / method_name
    all_rows = []
    for dirpath, _, filenames in os.walk(path):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                row = read_and_normalize_features(Path(dirpath) / fn)
                if row:
                    all_rows.append(row)
    return pd.DataFrame(all_rows)


# =============================================================================
# STATISTICAL ENGINE (FRIEDMAN + POST-HOC)
# =============================================================================

def run_friedman_for_scenario(df_stft, df_cwt, df_emd, scenario_label):
    """
    Executes Friedman Test and Post-hoc Wilcoxon comparisons.
    """
    keys = ["participant_id", "state"]
    feature_cols = [c for c in df_stft.columns if c not in keys]
    results = []

    print(f"Processing Scenario: {scenario_label} (N={len(df_stft)})")

    for feat in feature_cols:
        v1 = df_stft[feat].values.astype(np.float64)  # STFT
        v2 = df_cwt[feat].values.astype(np.float64)  # CWT
        v3 = df_emd[feat].values.astype(np.float64)  # EMD

        # Skip features with zero variance (identical values across all methods/subjects)
        if np.all(v1 == v1[0]) and np.all(v2 == v2[0]) and np.all(v3 == v3[0]):
            continue

        try:
            # 1. Global Friedman Test
            stat, p_val = friedmanchisquare(v1, v2, v3)

            # 2. Pairwise Post-hoc Wilcoxon Tests
            # This identifies WHICH methods actually disagree.
            def safe_wilcoxon(x, y):
                if np.array_equal(x, y): return 1.0
                try:
                    return wilcoxon(x, y).pvalue
                except:
                    return 1.0

            p_stft_cwt = safe_wilcoxon(v1, v2)
            p_stft_emd = safe_wilcoxon(v1, v3)
            p_cwt_emd = safe_wilcoxon(v2, v3)

            results.append({
                "scenario": scenario_label,
                "feature": feat,
                "mean_STFT": np.mean(v1),
                "mean_CWT": np.mean(v2),
                "mean_EMD": np.mean(v3),
                "chi_square": stat,
                "p_val": p_val,
                "p_STFT_vs_CWT": p_stft_cwt,
                "p_STFT_vs_EMD": p_stft_emd,
                "p_CWT_vs_EMD": p_cwt_emd
            })
        except ValueError:
            continue

    if not results:
        return pd.DataFrame()

    res_df = pd.DataFrame(results)

    # 3. FDR Correction on the Global P-values
    rej, p_fdr = fdrcorrection(res_df["p_val"], alpha=0.05)
    res_df["p_fdr"] = p_fdr
    res_df["significant"] = rej

    return res_df.sort_values("chi_square", ascending=False)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and align data
    all_data = {m: load_method_data(m) for m in methods}
    keys = ["participant_id", "state"]

    # Ensure subjects exist in all 3 methods for paired testing
    common = pd.merge(all_data["STFT"][keys], all_data["CWT"][keys], on=keys)
    common = pd.merge(common, all_data["EMD"][keys], on=keys)

    df_stft = pd.merge(common, all_data["STFT"], on=keys).sort_values(keys)
    df_cwt = pd.merge(common, all_data["CWT"], on=keys).sort_values(keys)
    df_emd = pd.merge(common, all_data["EMD"], on=keys).sort_values(keys)

    # 2. Run Scenarios
    all_scenarios_results = []

    # Scenarios: Task, Rest, and Both combined
    scenarios = [
        (df_stft["state"] == "task", "Task_Only"),
        (df_stft["state"] == "rest", "Rest_Only"),
        (np.ones(len(df_stft), dtype=bool), "Combined_All")
    ]

    for mask, label in scenarios:
        res = run_friedman_for_scenario(df_stft[mask], df_cwt[mask], df_emd[mask], label)
        if not res.empty:
            all_scenarios_results.append(res)

    # 3. Final Output
    if all_scenarios_results:
        final_report = pd.concat(all_scenarios_results, axis=0, ignore_index=True)
        final_report.to_csv(out_dir / "Methods_Comparison_Full.csv", index=False)
        print("\nAnalysis complete. Results saved with post-hoc p-values.")
    else:
        print("No results were generated.")


if __name__ == "__main__":
    main()