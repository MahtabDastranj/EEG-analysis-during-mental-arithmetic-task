from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
selected_features_path = Path(r"E:\AUT\thesis\files\feature_reduction\unpaired_test\Significant_Features_Detailed.csv")
labels_csv = Path(r'E:\AUT\thesis\EEG-analysis-during-mental-arithmetic-task\subject-info.csv')
root_feature_dir = Path(r"E:\AUT\thesis\files\features")
out_dir = Path(r"E:\AUT\thesis\files\classification\SVM_GridSearch")
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

    # 1. Load Data
    print("Loading Selected Features...")
    if not selected_features_path.exists():
        print("Error: Selected features file not found.")
        return

    df_select = pd.read_csv(selected_features_path)
    df_select.columns = [c.lower() for c in df_select.columns]

    # --- CHANGE: TRY FEWER FEATURES ---
    # Sometimes 15 is too many for 36 people. Let's start with 10.
    df_select = df_select.head(10)
    print(f"Using Top {len(df_select)} features for classification.")

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

    print(f"Data Loaded: N={len(y_final)}")

    # 2. Setup Grid Search
    # We will test these combinations to see which gives the best MACRO F1 score
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001, 'scale'],
        'kernel': ['rbf'],
        'class_weight': ['balanced']
    }

    # Use Nested Cross-Validation logic:
    # We will just run GridSearch on the whole dataset to find "Best Params" first
    # Then run 5-Fold CV using those best params.

    print("\nRunning Grid Search to find best Hyperparameters...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5, scoring='f1_macro')
    grid.fit(X_scaled, y_final)

    best_params = grid.best_params_
    print(f"\nBest Parameters Found: {best_params}")
    print(f"Best Macro F1 during GridSearch: {grid.best_score_:.3f}")

    # 3. Validating Best Model with Detailed 5-Fold CV
    print("\nRunning Final 5-Fold CV with Best Parameters...")

    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_metrics = {'accuracy': [], 'f1_bad': [], 'f1_good': [], 'n_support': []}

    for train_index, test_index in skf.split(X_final, y_final):
        X_train, X_test = X_final[train_index], X_final[test_index]
        y_train, y_test = y_final[train_index], y_final[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train with BEST params
        svm = SVC(**best_params, random_state=42)
        svm.fit(X_train_scaled, y_train)

        y_pred = svm.predict(X_test_scaled)

        # Metrics
        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1], zero_division=0)
        fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        fold_metrics['f1_bad'].append(f[0])
        fold_metrics['f1_good'].append(f[1])
        fold_metrics['n_support'].append(np.sum(svm.n_support_))

    # 4. Final Report
    print("-" * 60)
    print(f"FINAL RESULTS (Using C={best_params['C']}, gamma={best_params['gamma']})")
    print("-" * 60)
    print(f"Accuracy     : {np.mean(fold_metrics['accuracy']):.3f} ± {np.std(fold_metrics['accuracy']):.3f}")
    print(f"Bad F1-Score : {np.mean(fold_metrics['f1_bad']):.3f}")
    print(f"Good F1-Score: {np.mean(fold_metrics['f1_good']):.3f}")
    print(f"Avg Supp Vec : {np.mean(fold_metrics['n_support']):.1f} (Lower is better)")

    # Check Improvement
    ratio = np.mean(fold_metrics['n_support']) / (len(y_final) * 0.8)
    print(f"Support Vector Ratio: {ratio:.1%}")
    if ratio < 0.6:
        print(">> GREAT! The model is no longer overfitting.")
    elif ratio > 0.8:
        print(">> WARNING: Model is still overfitting. Try reducing feature count.")


if __name__ == "__main__":
    main()