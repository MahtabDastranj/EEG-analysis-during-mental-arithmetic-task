from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import warnings

out_dir = r'E:\AUT\thesis\files\feature_reduction\method_comparison'
labels_csv = r'E:\AUT\thesis\EEG-analysis-during-mental-arithmetic-task\subject-info.csv'
root_feature_dir = r"E:\AUT\thesis\files\features"

methods = ("STFT", "CWT", "EMD")
n_channels = 19
band_names = ["delta", "theta", "alpha", "beta", "gamma"]

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


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
    qual = pd.to_numeric(df.iloc[:, 5], errors="coerce").astype(int)
    return {0: pid[qual == 0].tolist(), 1: pid[qual == 1].tolist()}


def channel_labels(n):
    return [f"ch{idx:02d}" for idx in range(1, n + 1)]


def read_features(path):
    mat = pd.read_csv(path, header=None)
    if mat.shape != (n_channels, len(band_names)):
        return None
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
        return pd.DataFrame()
    rows = [read_features(p) for p in sorted(paths)]
    rows = [r for r in rows if r is not None]  # Filter bad reads
    if not rows:
        return pd.DataFrame()

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


def run_friedman_on_triplet(df_stft, df_cwt, df_emd, feature_name):
    """
    Runs Friedman test on 3 matched vectors (STFT, CWT, EMD) for one feature.
    """
    # Align data by index (Participant ID) to ensure we compare the SAME person
    # Inner join: only keep participants present in ALL 3 methods
    aligned = pd.concat([df_stft, df_cwt, df_emd], axis=1, join='inner')

    # After concat, columns are [feature, feature, feature].
    # We assume the order matches STFT, CWT, EMD because of concat order.

    if aligned.shape[0] < 5:  # Need at least 5 subjects for a test
        return np.nan, np.nan, "Insufficient Data"

    vec_stft = aligned.iloc[:, 0].to_numpy()
    vec_cwt = aligned.iloc[:, 1].to_numpy()
    vec_emd = aligned.iloc[:, 2].to_numpy()

    # The Friedman Test
    stat, p = friedmanchisquare(vec_stft, vec_cwt, vec_emd)

    # Simple Logic for "Winner" (Highest Median)
    medians = {'STFT': np.median(vec_stft), 'CWT': np.median(vec_cwt), 'EMD': np.median(vec_emd)}
    winner = max(medians, key=medians.get)

    return stat, p, winner


def analyze_condition_group(data_map, group_ids, state, feature_list):
    """
    data_map: {'STFT': df, 'CWT': df, 'EMD': df} for the specific state
    """
    results = []

    # 1. Filter all DataFrames for this Group ID list
    filtered_map = {}
    for method in methods:
        df = data_map[method]
        # Keep only rows where participant_id is in group_ids
        sub_df = df[df['participant_id'].isin(group_ids)].set_index('participant_id')
        filtered_map[method] = sub_df

    # 2. Loop through every feature (ch01_delta, etc.)
    for feat in feature_list:
        try:
            # Extract just that feature column from each method
            s = filtered_map['STFT'][feat]
            c = filtered_map['CWT'][feat]
            e = filtered_map['EMD'][feat]

            stat, p, winner = run_friedman_on_triplet(s, c, e, feat)

            results.append({
                "feature": feat,
                "n_subjects": s.shape[0],  # Using STFT count as proxy (aligned)
                "friedman_stat": stat,
                "p_value": p,
                "significant": p < 0.05 if not np.isnan(p) else False,
                "highest_method": winner
            })
        except KeyError:
            # Feature missing in one of the files
            continue

    return pd.DataFrame(results)


def main():
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Groups
    labels_dict = id_label_extraction(labels_csv)
    group_names = {0: "Bad_Counters", 1: "Good_Counters"}

    # 2. Load Raw Data into Memory
    # Structure: raw_data['STFT']['rest'] -> DataFrame
    print("Loading raw feature data...")
    raw_data = {}

    # We need to discover common features across all files
    all_features = set()

    for m in methods:
        raw_data[m] = {}
        files = find_files(root_feature_dir / m)

        # Load Rest
        print(f"  Loading {m} Rest...")
        df_rest = stack_features(files['rest'])
        raw_data[m]['rest'] = df_rest

        # Load Task
        print(f"  Loading {m} Task...")
        df_task = stack_features(files['task'])
        raw_data[m]['task'] = df_task

        # Track feature names
        if not df_rest.empty:
            feats = [c for c in df_rest.columns if c != 'participant_id']
            all_features.update(feats)

    sorted_features = sorted(list(all_features))
    print(f"Found {len(sorted_features)} features to compare.")

    # 3. The Grand Loop: Group -> State -> Analysis
    for group_code, group_name in group_names.items():
        ids = labels_dict.get(group_code, [])
        print(f"\nAnalyzing Group: {group_name} (N={len(ids)})")

        for state in ['rest', 'task']:
            print(f"  State: {state.upper()}")

            # Prepare map for this specific state
            state_data_map = {
                'STFT': raw_data['STFT'][state],
                'CWT': raw_data['CWT'][state],
                'EMD': raw_data['EMD'][state]
            }

            results_df = analyze_condition_group(state_data_map, ids, state, sorted_features)

            if not results_df.empty:
                # Save File
                filename = f"Comparison_{group_name}_{state.upper()}.csv"
                save_path = out_dir / filename
                results_df.to_csv(save_path, index=False)
                print(f"    Saved: {filename}")

    print("\nProcessing Complete.")


if __name__ == "__main__":
    main()