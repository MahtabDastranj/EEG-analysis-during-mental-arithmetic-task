from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
import random

root_feature_dir = Path(r"E:\AUT\thesis\files\features")
out_dir = Path(r"E:\AUT\thesis\files\feature_reduction\anova")
methods = ["STFT", "CWT", "EMD"]
n_channels = 19
band_names = ["delta", "theta", "alpha", "beta", "gamma"]
N_RANDOM_FEATURES = 3


def extract_id(path):
    stem = Path(path).stem
    m = re.search(r'(\d{2})', stem)
    if not m:
        return None
    return int(m.group(1))


def channel_labels(n):
    return [f"ch{idx:02d}" for idx in range(1, n + 1)]


def generate_all_feature_names():
    """Generates list of all 95 possible features (e.g., ch01_alpha)"""
    feats = []
    for ch in channel_labels(n_channels):
        for band in band_names:
            feats.append(f"{ch}_{band}")
    return feats


def get_delta_for_method_and_feature(method, feature_name):
    method_dir = root_feature_dir / method
    if not method_dir.exists():
        return {}

    # 1. Parse which row/col we need from the csv matrix
    try:
        ch_str, band_str = feature_name.split('_')
        ch_idx = int(ch_str.replace('ch', '')) - 1
        band_idx = band_names.index(band_str)
    except:
        return {}

    rest_vals = {}
    task_vals = {}

    # 2. Iterate all files in that method's folder
    for root, _, files in os.walk(method_dir):
        for file in files:
            if not file.lower().endswith('.csv'):
                continue

            pid = extract_id(file)
            if pid is None:
                continue

            try:
                # Read the CSV matrix
                filepath = Path(root) / file
                mat = pd.read_csv(filepath, header=None)

                if mat.shape != (n_channels, len(band_names)):
                    continue

                # Extract value
                val = float(mat.iloc[ch_idx, band_idx])

                if "rest" in file.lower():
                    rest_vals[pid] = val
                else:
                    task_vals[pid] = val
            except:
                continue

    # 3. Calculate Delta (Task - Rest)
    common_ids = set(rest_vals.keys()).intersection(task_vals.keys())
    delta_results = {}

    for pid in common_ids:
        delta_results[pid] = task_vals[pid] - rest_vals[pid]

    return delta_results


def main():
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(42)  # Set seed for reproducibility, remove if you want random every time

    # 1. Pick Random Features
    all_features = generate_all_feature_names()
    selected_features = random.sample(all_features, N_RANDOM_FEATURES)

    print(f"Randomly selected {N_RANDOM_FEATURES} features for verification:")
    for f in selected_features:
        print(f" - {f}")

    # 2. Build the Dataset
    final_rows = []

    # Loop through the randomly selected features
    for feat in selected_features:
        print(f"\nProcessing feature: {feat}")

        # Get data for all 3 methods
        data_stft = get_delta_for_method_and_feature("STFT", feat)
        data_cwt = get_delta_for_method_and_feature("CWT", feat)
        data_emd = get_delta_for_method_and_feature("EMD", feat)

        # Find participants present in ALL 3 methods (to ensure fair comparison)
        common_ids = set(data_stft.keys()) & set(data_cwt.keys()) & set(data_emd.keys())

        for pid in sorted(list(common_ids)):
            final_rows.append({
                "Participant_ID": pid,
                "Feature": feat,
                "STFT_Delta": data_stft[pid],
                "CWT_Delta": data_cwt[pid],
                "EMD_Delta": data_emd[pid]
            })

    # 3. Save to CSV
    if final_rows:
        df = pd.DataFrame(final_rows)

        # Sort for readability
        df = df.sort_values(by=["Feature", "Participant_ID"])

        save_path = out_dir / "Verification_Data_For_Supervisor.csv"
        df.to_csv(save_path, index=False)

        print("SUCCESS!")
        print(f"File created: {save_path}")
        print("=" * 50)
    else:
        print("Error: No common data found across methods.")


if __name__ == "__main__":
    main()