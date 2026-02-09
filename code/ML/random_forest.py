from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report)
import seaborn as sns
import matplotlib.pyplot as plt

selected_features_path = Path(r"E:\AUT\thesis\files\feature_reduction\unpaired_test\Significant_Features_Detailed.csv")
labels_csv = Path(r'E:\AUT\thesis\EEG-analysis-during-mental-arithmetic-task\subject-info.csv')
root_feature_dir = Path(r"E:\AUT\thesis\files\features")
out_dir = Path(r"E:\AUT\thesis\files\classification\random_forest")

n_channels = 19
band_names = ["delta", "theta", "alpha", "beta", "gamma"]

def extract_id(path):
    stem = Path(path).stem
    m = re.search(r'(\d{2})', stem)
    if not m:
        raise ValueError(f"Cannot extract ID: {stem}")
    return int(m.group(1))


def get_labels(labels_csv):
    df = pd.read_csv(labels_csv, header=0)
    col0 = df.iloc[:, 0].astype(str).str.strip()
    pids = col0.str.extract(r'(\d{2})', expand=False).astype(int)
    groups = pd.to_numeric(df.iloc[:, 5], errors="coerce").fillna(-1).astype(int)
    valid_mask = groups.isin([0, 1])
    return dict(zip(pids[valid_mask], groups[valid_mask]))


def read_single_feature_vector(method, feature_name):
    """Computes task - rest (log power difference)"""
    method_dir = root_feature_dir / method
    if not method_dir.exists():
        return None

    rest_vals, task_vals = {}, {}

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

    common_ids = set(rest_vals).intersection(task_vals)
    delta_data = {pid: task_vals[pid] - rest_vals[pid] for pid in common_ids}
    return pd.Series(delta_data, name=f"{method}_{feature_name}")


def main():
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Selected Features...")
    df_select = pd.read_csv(selected_features_path)
    df_select.columns = [c.lower() for c in df_select.columns]
    df_select = df_select.head(7)
    print(f"Using Top {len(df_select)} features.")

    feature_vectors = []
    for _, row in df_select.iterrows():
        method = row['method']
        feat = row['feature']
        vec = read_single_feature_vector(method, feat)
        if vec is not None:
            feature_vectors.append(vec)

    if not feature_vectors:
        print("No valid features found.")
        return

    X_df = pd.concat(feature_vectors, axis=1) # Rows = participants, Columns = selected EEG features

    label_map = get_labels(labels_csv)
    valid_ids = [pid for pid in X_df.index if pid in label_map]

    X_final = X_df.loc[valid_ids].values  # (35, 5)
    y_final = np.array([label_map[pid] for pid in valid_ids])

    print(f"Data Loaded: N={len(y_final)} (Bad={np.sum(y_final == 0)}, Good={np.sum(y_final == 1)})")

    # GRID SEARCH
    print("\nRunning Random Forest Grid Search...")

    param_grid = {
        'n_estimators': [200, 500],  # trees
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [2, 3, 4],
        'max_features': ['sqrt'],
        'class_weight': ['balanced']
    }

    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

    grid = GridSearchCV(rf_base, param_grid, cv=5, scoring='f1_macro', refit=True)

    grid.fit(X_final, y_final)
    best_params = grid.best_params_

    print(f"Best Parameters: {best_params}")

    # 5-FOLD CV
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

        rf = RandomForestClassifier(
            **best_params,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X_train, y_train)

        y_test_pred = rf.predict(X_test)
        y_train_pred = rf.predict(X_train)

        total_y_test.extend(y_test)
        total_y_pred.extend(y_test_pred)

        test_acc = accuracy_score(y_test, y_test_pred)
        train_acc = accuracy_score(y_train, y_train_pred)

        p, r, f, _ = precision_recall_fscore_support(
            y_test, y_test_pred,
            labels=[0, 1],
            zero_division=0
        )

        cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
        total_cm += cm

        fold_stats.append({
            'Fold': fold_idx,
            'Train_Acc': train_acc,
            'Test_Acc': test_acc,
            'Bad_Recall': r[0]
        })

        fold_idx += 1

    # REPORT
    df_res = pd.DataFrame(fold_stats)

    print("\n" + "=" * 65)
    print(f"{'Fold':<6} {'Train Acc':<12} {'Test Acc':<12} {'Bad Recall':<12}")
    print("-" * 65)

    for _, row in df_res.iterrows():
        print(
            f"{int(row['Fold']):<6} "
            f"{row['Train_Acc']:.1%}       "
            f"{row['Test_Acc']:.1%}       "
            f"{row['Bad_Recall']:.1%}"
        )

    print("-" * 65)

    avg_test_acc = df_res['Test_Acc'].mean()

    print(f"Average Test Accuracy:  {avg_test_acc:.1%} ± {df_res['Test_Acc'].std():.1%}")
    print(f"Average Train Accuracy: {df_res['Train_Acc'].mean():.1%}")
    print(f"Generalization Gap:     {(df_res['Train_Acc'].mean() - avg_test_acc):.1%}")

    print("\nDetailed Classification Report (Aggregated):")
    print(classification_report(
        total_y_test,
        total_y_pred,
        target_names=['Bad (0)', 'Good (1)'],
        digits=3
    ))

    # CONFUSION MATRIX
    plt.figure(figsize=(7, 6))

    group_names = ['TN', 'FP', 'FN', 'TP']
    group_counts = [f"{v:d}" for v in total_cm.flatten()]
    group_percentages = [f"{v:.1%}" for v in total_cm.flatten() / total_cm.sum()]
    labels = [f"{n}\n{c}\n({p})" for n, c, p in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(
        total_cm,
        annot=labels,
        fmt='',
        cmap='Greens',
        xticklabels=['Bad (0)', 'Good (1)'],
        yticklabels=['Bad (0)', 'Good (1)'],
        cbar=False,
        annot_kws={"size": 14}
    )

    plt.title(f'Random Forest – Aggregate Confusion Matrix\nMean Accuracy: {avg_test_acc:.1%}',
              fontsize=15, fontweight='bold', pad=20)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    save_path = out_dir / "Confusion_Matrix_RF.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nMatrix saved to: {save_path}")


if __name__ == "__main__":
    main()
