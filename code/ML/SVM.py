from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

selected_features_path = Path(r"E:\AUT\thesis\files\feature_reduction\unpaired_test\Significant_Features_Detailed.csv")
labels_csv = Path(r'E:\AUT\thesis\EEG-analysis-during-mental-arithmetic-task\subject-info.csv')
root_feature_dir = Path(r"E:\AUT\thesis\files\features")
out_dir = Path(r"E:\AUT\thesis\files\classification\SVM")
n_channels = 19
band_names = ["delta", "theta", "alpha", "beta", "gamma"]


def extract_id(path):
    stem = Path(path).stem
    m = re.search(r'(\d{2})', stem)
    if not m:
        raise ValueError(f"Cannot extract ID: {stem}")
    return int(m.group(1))


def channel_labels(n):
    return [f"ch{idx:02d}" for idx in range(1, n + 1)]


def get_labels(labels_csv):
    """Returns a dictionary: {Participant_ID: Label (0 or 1)}"""
    df = pd.read_csv(labels_csv, header=0)
    col0 = df.iloc[:, 0].astype(str).str.strip()
    pids = col0.str.extract(r'(\d{2})', expand=False).astype(int)
    groups = pd.to_numeric(df.iloc[:, 5], errors="coerce").fillna(-1).astype(int)

    # Filter only valid 0 or 1 labels
    valid_mask = groups.isin([0, 1])
    return dict(zip(pids[valid_mask], groups[valid_mask]))


def read_single_feature_vector(method, feature_name):
    """
    Loads raw data for ONE specific feature across ALL subjects.
    Returns a Series: Index=ID, Value=(Task - Rest)
    """
    method_dir = root_feature_dir / method
    if not method_dir.exists():
        print(f"Warning: Directory not found {method_dir}")
        return None

    # Storage
    rest_vals = {}
    task_vals = {}

    # Parse Feature Name (e.g., ch01_alpha) to get indices
    try:
        ch_str, band_str = feature_name.split('_')
        ch_idx = int(ch_str.replace('ch', '')) - 1
        band_idx = band_names.index(band_str)
    except:
        print(f"Error parsing feature name: {feature_name}")
        return None

    # Iterate files
    for root, _, files in os.walk(method_dir):
        for file in files:
            if not file.lower().endswith('.csv'):
                continue

            try:
                pid = extract_id(file)

                # Read CSV
                path = Path(root) / file
                mat = pd.read_csv(path, header=None)

                if mat.shape != (n_channels, len(band_names)):
                    continue

                # Extract Specific Value
                val = float(mat.iloc[ch_idx, band_idx])
                log_val = np.log10(val + 1e-12)  # Apply Log Transform

                if "rest" in file.lower():
                    rest_vals[pid] = log_val
                else:
                    task_vals[pid] = log_val
            except:
                continue

    # Calculate Delta (Task - Rest)
    # Only for participants who have both files
    common_ids = set(rest_vals.keys()).intersection(task_vals.keys())
    delta_data = {}

    for pid in common_ids:
        delta_data[pid] = task_vals[pid] - rest_vals[pid]

    return pd.Series(delta_data, name=f"{method}_{feature_name}")


def main():
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Selection File
    print("Loading Selected Features...")
    if not selected_features_path.exists():
        print("Error: Selected features file not found.")
        return

    # Assuming columns are "Method" and "Feature" (or first 2 cols)
    df_select = pd.read_csv(selected_features_path)

    # Normalize column names just in case
    df_select.columns = [c.lower() for c in df_select.columns]

    # Limit to first 15 rows if the file is larger
    df_select = df_select.head(15)
    print(f"Selected {len(df_select)} features for classification.")

    # 2. Build the X Matrix (Features)
    feature_vectors = []

    for idx, row in df_select.iterrows():
        # Handle variations in column naming
        if 'method' in row and 'feature' in row:
            method = row['method']
            feat = row['feature']
        else:
            # Fallback: Assume col 0 is Method, col 1 is Feature
            method = row.iloc[0]
            feat = row.iloc[1]

        print(f"Loading data for: {method} - {feat}")
        vec = read_single_feature_vector(method, feat)
        if vec is not None:
            feature_vectors.append(vec)

    if not feature_vectors:
        print("No data loaded. Check feature names and paths.")
        return

    # Combine into DataFrame (Indices are Participant IDs)
    X_df = pd.concat(feature_vectors, axis=1)

    # 3. Build the y Vector (Labels)
    label_map = get_labels(labels_csv)

    # Align X and y
    # Only keep participants who have both Features AND a Label
    valid_ids = [pid for pid in X_df.index if pid in label_map]

    X_final = X_df.loc[valid_ids]
    y_final = [label_map[pid] for pid in valid_ids]
    y_final = np.array(y_final)

    print(f"\nFinal Dataset Shape: {X_final.shape}")
    print(f"Class Balance: {np.bincount(y_final)} (0=Bad, 1=Good)")

    # 4. Split Data (Stratified)
    # stratify=y ensures the ratio of Good/Bad is the same in Train and Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_final,
        test_size=0.20,
        random_state=42,
        stratify=y_final
    )

    print(f"Train Set: {len(y_train)} samples")
    print(f"Test Set:  {len(y_test)} samples")

    # 5. Preprocessing (Standard Scaling)
    # SVM is sensitive to scale. We fit on TRAIN and transform TEST.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Train Robust SVM
    # Kernel='rbf' allows for non-linear decision boundaries
    # C=1.0 is standard regularization
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)

    # 7. Evaluate
    y_pred = svm.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Bad Counters', 'Good Counters'])

    print("\n--- CLASSIFICATION RESULTS ---")
    print(f"Accuracy: {acc:.2f}")
    print("\nConfusion Matrix:")
    print(conf_mat)
    print("\nReport:")
    print(report)

    # 8. Visualize Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bad', 'Good'],
                yticklabels=['Bad', 'Good'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'SVM Confusion Matrix (Acc={acc:.2f})')

    save_path = out_dir / "SVM_Confusion_Matrix.png"
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Plot saved to: {save_path}")

    # Save X and y for record
    X_final.to_csv(out_dir / "X_matrix_used.csv")
    pd.DataFrame(y_final, index=X_final.index, columns=["label"]).to_csv(out_dir / "y_labels_used.csv")


if __name__ == "__main__":
    main()