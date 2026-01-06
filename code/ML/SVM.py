from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
selected_features_path = Path(r"E:\AUT\thesis\files\feature_reduction\unpaired_test\Significant_Features_Detailed.csv")
labels_csv = Path(r'E:\AUT\thesis\EEG-analysis-during-mental-arithmetic-task\subject-info.csv')
root_feature_dir = Path(r"E:\AUT\thesis\files\features")
out_dir = Path(r"E:\AUT\thesis\files\classification\SVM_CrossVal")
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

    X_final = X_df.loc[valid_ids].values
    y_final = np.array([label_map[pid] for pid in valid_ids])

    print(f"Final Dataset: N={len(y_final)} (Bad={np.sum(y_final == 0)}, Good={np.sum(y_final == 1)})")

    # 3. Setup Cross-Validation
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Storage for metrics across folds
    fold_metrics = {
        'accuracy': [],
        'precision_bad': [], 'recall_bad': [], 'f1_bad': [],
        'precision_good': [], 'recall_good': [], 'f1_good': [],
        'macro_f1': [],
        'n_support': []
    }

    # Aggregate Confusion Matrix
    total_cm = np.zeros((2, 2), dtype=int)

    print(f"\nRunning {k_folds}-Fold Cross-Validation...")
    print("-" * 60)

    fold_idx = 1
    for train_index, test_index in skf.split(X_final, y_final):
        X_train, X_test = X_final[train_index], X_final[test_index]
        y_train, y_test = y_final[train_index], y_final[test_index]

        # Standard Scaling (Fit on Train, Transform Test)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Balanced SVM
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
        svm.fit(X_train_scaled, y_train)

        # Predict
        y_pred = svm.predict(X_test_scaled)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1], zero_division=0)

        # Store
        fold_metrics['accuracy'].append(acc)
        fold_metrics['precision_bad'].append(p[0])
        fold_metrics['recall_bad'].append(r[0])
        fold_metrics['f1_bad'].append(f[0])
        fold_metrics['precision_good'].append(p[1])
        fold_metrics['recall_good'].append(r[1])
        fold_metrics['f1_good'].append(f[1])
        fold_metrics['macro_f1'].append(np.mean(f))
        fold_metrics['n_support'].append(np.sum(svm.n_support_))

        # Update Aggregate Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        total_cm += cm

        print(
            f"Fold {fold_idx}: Acc={acc:.2f} | Bad F1={f[0]:.2f} | Good F1={f[1]:.2f} | SuppVec={np.sum(svm.n_support_)}")
        fold_idx += 1

    # 4. Final Reporting
    print("-" * 60)
    print("CROSS-VALIDATION RESULTS (Average ± SD)")
    print("-" * 60)

    def report_stat(name, key):
        mean_val = np.mean(fold_metrics[key])
        std_val = np.std(fold_metrics[key])
        print(f"{name:<20}: {mean_val:.3f} ± {std_val:.3f}")

    report_stat("Accuracy", 'accuracy')
    report_stat("Macro F1-Score", 'macro_f1')
    print("--- Class 0 (Bad) ---")
    report_stat("Precision", 'precision_bad')
    report_stat("Recall", 'recall_bad')
    report_stat("F1-Score", 'f1_bad')
    print("--- Class 1 (Good) ---")
    report_stat("Precision", 'precision_good')
    report_stat("Recall", 'recall_good')
    report_stat("F1-Score", 'f1_good')

    print("-" * 60)
    avg_support = np.mean(fold_metrics['n_support'])
    print(f"Avg Support Vectors : {avg_support:.1f} (out of ~{len(y_final) * 0.8:.0f} training samples)")
    print("Interpretation: Lower is better. If nearly equal to training size, model is overfitting.")

    # 5. Visualize Aggregate Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
    plt.title(f'Aggregate Confusion Matrix ({k_folds}-Fold CV)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    save_path = out_dir / "CV_Aggregate_ConfusionMatrix.png"
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"\nPlot saved to: {save_path}")


if __name__ == "__main__":
    main()