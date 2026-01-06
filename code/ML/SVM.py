from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
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
    df = pd.read_csv(labels_csv, header=0)
    col0 = df.iloc[:, 0].astype(str).str.strip()
    pids = col0.str.extract(r'(\d{2})', expand=False).astype(int)
    groups = pd.to_numeric(df.iloc[:, 5], errors="coerce").fillna(-1).astype(int)
    valid_mask = groups.isin([0, 1])
    return dict(zip(pids[valid_mask], groups[valid_mask]))


def read_single_feature_vector(method, feature_name):
    method_dir = root_feature_dir / method
    if not method_dir.exists():
        return None

    rest_vals = {}
    task_vals = {}

    try:
        ch_str, band_str = feature_name.split('_')
        ch_idx = int(ch_str.replace('ch', '')) - 1
        band_idx = band_names.index(band_str)
    except:
        return None

    for root, _, files in os.walk(method_dir):
        for file in files:
            if not file.lower().endswith('.csv'):
                continue
            try:
                pid = extract_id(file)
                path = Path(root) / file
                mat = pd.read_csv(path, header=None)
                if mat.shape != (n_channels, len(band_names)):
                    continue
                val = float(mat.iloc[ch_idx, band_idx])
                log_val = np.log10(val + 1e-12)
                if "rest" in file.lower():
                    rest_vals[pid] = log_val
                else:
                    task_vals[pid] = log_val
            except:
                continue

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

    df_select = pd.read_csv(selected_features_path)
    df_select.columns = [c.lower() for c in df_select.columns]
    df_select = df_select.head(15)
    print(f"Selected {len(df_select)} features.")

    # 2. Build X Matrix
    feature_vectors = []
    for idx, row in df_select.iterrows():
        if 'method' in row and 'feature' in row:
            method = row['method']
            feat = row['feature']
        else:
            method = row.iloc[0]
            feat = row.iloc[1]
        vec = read_single_feature_vector(method, feat)
        if vec is not None:
            feature_vectors.append(vec)

    if not feature_vectors:
        print("No data loaded.")
        return

    X_df = pd.concat(feature_vectors, axis=1)
    label_map = get_labels(labels_csv)
    valid_ids = [pid for pid in X_df.index if pid in label_map]

    X_final = X_df.loc[valid_ids]
    y_final = np.array([label_map[pid] for pid in valid_ids])

    print(f"Class Balance: {np.bincount(y_final)} (0=Bad, 1=Good)")

    # 3. Stratified Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_final,
        test_size=0.20,
        random_state=42,
        stratify=y_final
    )

    # 4. Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ================= THE FIX IS HERE =================
    # class_weight='balanced' automatically adjusts weights
    # inversely proportional to class frequencies
    print("Training Balanced SVM...")
    svm = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',  # <--- CRITICAL FIX
        random_state=42
    )
    svm.fit(X_train_scaled, y_train)
    # ===================================================

    # 5. Evaluate
    y_train_pred = svm.predict(X_train_scaled)
    y_test_pred = svm.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("\n" + "=" * 30)
    print("      RESULTS (BALANCED)")
    print("=" * 30)
    print(f"Train Acc: {train_acc:.2f}")
    print(f"Test Acc:  {test_acc:.2f}")

    # 6. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cms = [confusion_matrix(y_train, y_train_pred), confusion_matrix(y_test, y_test_pred)]
    titles = [f'Train (Acc={train_acc:.2f})', f'Test (Acc={test_acc:.2f})']

    for i, ax in enumerate(axes):
        sns.heatmap(cms[i], annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'], ax=ax)
        ax.set_title(titles[i])
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig(out_dir / "Balanced_SVM_Results.png", dpi=300)
    plt.show()
    print("Results saved.")


if __name__ == "__main__":
    main()