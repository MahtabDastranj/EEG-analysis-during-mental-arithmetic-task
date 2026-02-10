from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
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
    if not m: raise ValueError(f"Cannot extract ID: {stem}")
    return int(m.group(1))


def get_labels(labels_csv):
    df = pd.read_csv(labels_csv, header=0)
    col0 = df.iloc[:, 0].astype(str).str.strip()
    pids = col0.str.extract(r'(\d{2})', expand=False).astype(int)
    groups = pd.to_numeric(df.iloc[:, 5], errors="coerce").fillna(-1).astype(int)
    valid_mask = groups.isin([0, 1])
    return dict(zip(pids[valid_mask], groups[valid_mask]))


def read_single_feature_vector(method, feature_name):
    """Computes task-rest"""
    method_dir = root_feature_dir / method
    if not method_dir.exists(): return None
    rest_vals, task_vals = {}, {}
    try:
        ch_str, band_str = feature_name.split('_')
        ch_idx = int(ch_str.replace('ch', '')) - 1
        band_idx = band_names.index(band_str)
    except:
        return None

    for root, _, files in os.walk(method_dir):
        for file in files:
            if not file.lower().endswith('.csv'): continue
            try:
                pid = extract_id(file)
                path = Path(root) / file
                mat = pd.read_csv(path, header=None)
                if mat.shape != (n_channels, len(band_names)): continue
                val = float(mat.iloc[ch_idx, band_idx])
                log_val = np.log10(val + 1e-12)
                if "rest" in file.lower():
                    rest_vals[pid] = log_val
                else:
                    task_vals[pid] = log_val
            except:
                continue

    # Match subjects and compute difference
    common_ids = set(rest_vals.keys()).intersection(task_vals.keys())
    delta_data = {pid: task_vals[pid] - rest_vals[pid] for pid in common_ids}
    return pd.Series(delta_data, name=f"{method}_{feature_name}")


def main():
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("Loading Selected Features...")
    if not selected_features_path.exists():
        print("Error: Selected features file not found.")
        return

    df_select = pd.read_csv(selected_features_path)
    df_select.columns = [c.lower() for c in df_select.columns]
    df_select = df_select.head(7)
    print(f"Using Top {len(df_select)} features.")

    feature_vectors = []
    for idx, row in df_select.iterrows():
        method = row['method'] if 'method' in row else row.iloc[0]
        feat = row['feature'] if 'feature' in row else row.iloc[1]
        vec = read_single_feature_vector(method, feat)
        if vec is not None: feature_vectors.append(vec)

    if not feature_vectors: return
    X_df = pd.concat(feature_vectors, axis=1)
    label_map = get_labels(labels_csv)
    valid_ids = [pid for pid in X_df.index if pid in label_map]
    X_final = X_df.loc[valid_ids].values
    y_final = np.array([label_map[pid] for pid in valid_ids])

    print(f"Data Loaded: N={len(y_final)} (Bad={np.sum(y_final == 0)}, Good={np.sum(y_final == 1)})")

    # 2. Grid Search (Performed on all data once)
    print("\nRunning Grid Search...")
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'gamma': [1, 0.1, 0.01, 0.001, 'scale'],
        'kernel': ['rbf'],
        'class_weight': ['balanced']
    }

    scaler_gs = StandardScaler()
    X_scaled_gs = scaler_gs.fit_transform(X_final)
    grid = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='f1_macro')
    grid.fit(X_scaled_gs, y_final)
    best_params = grid.best_params_
    print(f"Best Parameters: {best_params}")

    # 3. Leave-One-Subject-Out (LOSO) CV
    print(f"\nRunning LOSO Validation (N={len(y_final)} iterations)...")
    loo = LeaveOneOut()

    total_y_test = []
    total_y_pred = []
    train_accuracies = []
    support_vector_counts = [] # Track SVs per iteration

    for train_idx, test_idx in loo.split(X_final):
        X_train, X_test = X_final[train_idx], X_final[test_idx]
        y_train, y_test = y_final[train_idx], y_final[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        svm = SVC(**best_params, random_state=42)
        svm.fit(X_train_scaled, y_train)

        # Track the number of support vectors used to define the margin
        support_vector_counts.append(np.sum(svm.n_support_))

        total_y_test.extend(y_test)
        total_y_pred.append(svm.predict(X_test_scaled)[0])
        train_accuracies.append(accuracy_score(y_train, svm.predict(X_train_scaled)))

    # 4. Reporting
    avg_test_acc = accuracy_score(total_y_test, total_y_pred)
    avg_train_acc = np.mean(train_accuracies)
    mean_sv = np.mean(support_vector_counts)
    sv_perc = (mean_sv / (len(y_final) - 1)) * 100
    print("-" * 45)
    print(f"{'Average Train Acc':<25} {avg_train_acc:.1%}")
    print(f"{'Average Test Acc (LOSO)':<25} {avg_test_acc:.1%}")
    print(f"{'Generalization Gap':<25} {(avg_train_acc - avg_test_acc):.1%}")
    print(f"{'Mean Support Vectors':<25} {mean_sv:.1f} ({sv_perc:.1f}%)")
    print("-" * 45)

    print("\nDetailed Classification Report:")
    print(classification_report(total_y_test, total_y_pred, target_names=['Bad (0)', 'Good (1)'], digits=3))

    # 5. Confusion Matrix
    plt.figure(figsize=(7, 6))
    total_cm = confusion_matrix(total_y_test, total_y_pred, labels=[0, 1])
    group_names = ['TN', 'FP', 'FN', 'TP']
    group_counts = ["{0:0.0f}".format(value) for value in total_cm.flatten()]
    group_percentages = ["{0:.1%}".format(value) for value in total_cm.flatten() / np.sum(total_cm)]
    labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(total_cm, annot=labels, fmt='', cmap='Blues',
                xticklabels=['Bad (0)', 'Good (1)'], yticklabels=['Bad (0)', 'Good (1)'],
                cbar=False, annot_kws={"size": 14})

    plt.title(f'Confusion Matrix\n Accuracy: {avg_test_acc:.1%}', fontsize=15, fontweight='bold', pad=20)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    save_path = out_dir / "confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nMatrix saved to: {save_path}")


if __name__ == "__main__":
    main()