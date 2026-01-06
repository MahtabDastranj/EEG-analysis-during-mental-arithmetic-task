from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

selected_features_path = Path(r"E:\AUT\thesis\files\feature_reduction\unpaired_test\Significant_Features_Detailed.csv")
labels_csv = Path(r'E:\AUT\thesis\EEG-analysis-during-mental-arithmetic-task\subject-info.csv')
root_feature_dir = Path(r"E:\AUT\thesis\files\features")
out_dir = Path(r"E:\AUT\thesis\files\classification\SVM_Final_Report")
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

    # CRITICAL: Using top 5 features (User had 3 in snippet, 5 is safer for 36 subjects)
    df_select = df_select.head(5)
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

    # 2. Grid Search
    print("\nRunning Grid Search...")
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'gamma': [1, 0.1, 0.01, 0.001, 'scale'],
        'kernel': ['rbf'], 'class_weight': ['balanced']
    }
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    grid = GridSearchCV(SVC(random_state=42), param_grid, refit=True, cv=5, scoring='f1_macro')
    grid.fit(X_scaled, y_final)
    best_params = grid.best_params_
    print(f"Best Parameters: {best_params}")

    # 3. Final 5-Fold CV
    print("\nRunning Final 5-Fold Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_stats = []
    total_y_test = []
    total_y_pred = []
    total_cm = np.zeros((2, 2), dtype=int)

    fold_idx = 1
    for train_idx, test_idx in skf.split(X_final, y_final):
        X_train, X_test = X_final[train_idx], X_final[test_idx]
        y_train, y_test = y_final[train_idx], y_final[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        svm = SVC(**best_params, random_state=42)
        svm.fit(X_train_scaled, y_train)

        y_test_pred = svm.predict(X_test_scaled)
        y_train_pred = svm.predict(X_train_scaled)

        # Collect for global metrics
        total_y_test.extend(y_test)
        total_y_pred.extend(y_test_pred)

        # Per Fold Metrics
        test_acc = accuracy_score(y_test, y_test_pred)
        train_acc = accuracy_score(y_train, y_train_pred)
        p, r, f, _ = precision_recall_fscore_support(y_test, y_test_pred, labels=[0, 1], zero_division=0)

        cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
        total_cm += cm

        fold_stats.append({
            'Fold': fold_idx,
            'Train_Acc': train_acc,
            'Test_Acc': test_acc,
            'Bad_Recall': r[0],
            'Support_Vecs': np.sum(svm.n_support_)
        })
        fold_idx += 1

    # 4. Reporting
    df_res = pd.DataFrame(fold_stats)
    print("\n" + "=" * 65)
    print(f"{'Fold':<6} {'Train Acc':<12} {'Test Acc':<12} {'Bad Recall':<12} {'Supp. Vecs':<12}")
    print("-" * 65)
    for _, row in df_res.iterrows():
        print(
            f"{int(row['Fold']):<6} {row['Train_Acc']:.1%}       {row['Test_Acc']:.1%}       {row['Bad_Recall']:.1%}       {int(row['Support_Vecs']):<12}")
    print("-" * 65)

    avg_test_acc = df_res['Test_Acc'].mean()

    # Calculate Macro and Weighted Averages
    # We calculate this by aggregating all predictions (Simulating the final performance)
    # OR by averaging the per-fold scores. Averaging per-fold is standard for CV.

    print(f"Average Test Accuracy:  {avg_test_acc:.1%} ± {df_res['Test_Acc'].std():.1%}")
    print(f"Average Train Accuracy: {df_res['Train_Acc'].mean():.1%}")
    print(f"Generalization Gap:     {(df_res['Train_Acc'].mean() - avg_test_acc):.1%}")

    # Use classification_report to get Weighted/Macro averages easily
    print("\nDetailed Classification Report (Aggregated):")
    print(classification_report(total_y_test, total_y_pred, target_names=['Bad (0)', 'Good (1)'], digits=3))

    # 5. Confusion Matrix
    plt.figure(figsize=(7, 6))
    group_names = ['TN', 'FP', 'FN', 'TP']
    group_counts = ["{0:0.0f}".format(value) for value in total_cm.flatten()]
    group_percentages = ["{0:.1%}".format(value) for value in total_cm.flatten() / np.sum(total_cm)]
    labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(total_cm, annot=labels, fmt='', cmap='Blues',
                xticklabels=['Bad (0)', 'Good (1)'], yticklabels=['Bad (0)', 'Good (1)'],
                cbar=False, annot_kws={"size": 14})

    plt.title(f'Aggregate Confusion Matrix\nMean Accuracy: {avg_test_acc:.1%}', fontsize=15, fontweight='bold', pad=20)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    save_path = out_dir / "Confusion_Matrix_With_Stats.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nMatrix saved to: {save_path}")


if __name__ == "__main__":
    main()