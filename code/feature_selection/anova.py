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
    return [f"ch{idx:02d}" for idx in range(1, n + 1)]


def read_and_normalize_features(path):
    """
    Reads a 19x5 feature file.
    Normalizes each channel to relative power (sum of bands = 1)
    to ensure STFT, CWT, and EMD are on the same scale for comparison.
    """
    mat = pd.read_csv(path, header=None)
    if mat.shape != (n_channels, len(band_names)):
        return None

    # Convert to Relative Power (Important for ANOVA between methods)
    # Each row (channel) will sum to 1.0
    row_sums = mat.sum(axis=1).values[:, np.newaxis]
    row_sums[row_sums == 0] = 1e-12  # Avoid division by zero
    mat_norm = mat.values / row_sums

    df_norm = pd.DataFrame(mat_norm, columns=band_names, index=channel_labels(n_channels))

    # Flatten to single row
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
    """Loads all CSVs for a specific method into a DataFrame."""
    path = root_feature_dir / method_name
    all_rows = []
    for dirpath, _, filenames in os.walk(path):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                row = read_and_normalize_features(Path(dirpath) / fn)
                if row:
                    all_rows.append(row)
    return pd.DataFrame(all_rows)


def main():
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading and aligning features from all 3 methods...")

    # 1. Load data for all methods
    data = {}
    for m in methods:
        data[m] = load_method_data(m)

    # 2. Find participants/states present in ALL three methods
    # This is required for the Friedman (paired) test
    keys = ["participant_id", "state"]
    common = pd.merge(data["STFT"][keys], data["CWT"][keys], on=keys)
    common = pd.merge(common, data["EMD"][keys], on=keys)

    print(f"Found {len(common)} matched records (participant-state pairs) across all methods.")

    # 3. Align the DataFrames
    df_stft = pd.merge(common, data["STFT"], on=keys).sort_values(keys)
    df_cwt = pd.merge(common, data["CWT"], on=keys).sort_values(keys)
    df_emd = pd.merge(common, data["EMD"], on=keys).sort_values(keys)

    # 4. Perform Friedman Test per Feature
    feature_cols = [c for c in df_stft.columns if c not in keys]
    results = []

    print("Applying Friedman Test (Non-parametric ANOVA)...")
    for feat in feature_cols:
        # Get values for this feature across the 3 methods
        v1 = df_stft[feat].values
        v2 = df_cwt[feat].values
        v3 = df_emd[feat].values

        try:
            # Friedman test compares the distributions of the 3 methods
            stat, p_val = friedmanchisquare(v1, v2, v3)

            # Descriptive stats for the paper
            m_stft, m_cwt, m_emd = np.mean(v1), np.mean(v2), np.mean(v3)

            results.append({
                "feature": feat,
                "mean_STFT": m_stft,
                "mean_CWT": m_cwt,
                "mean_EMD": m_emd,
                "chi_square": stat,
                "p_val": p_val
            })
        except ValueError:
            # Occurs if all values are identical
            continue

    anova_df = pd.DataFrame(results)

    # 5. Multiple Comparison Correction (FDR)
    if not anova_df.empty:
        rej, p_fdr = fdrcorrection(anova_df["p_val"], alpha=0.05)
        anova_df["p_fdr"] = p_fdr
        anova_df["significant"] = rej

        # 6. Save Results
        anova_path = out_dir / "Methods_ANOVA_Comparison.csv"
        anova_df.sort_values("chi_square", ascending=False).to_csv(anova_path, index=False)
        print(f"ANOVA Analysis Complete. Results saved to: {anova_path}")

        # Count significant differences
        sig_count = anova_df["significant"].sum()
        print(f"Out of 95 features, {sig_count} showed a significant difference between methods.")
    else:
        print("No valid features found for analysis.")


if __name__ == "__main__":
    main()