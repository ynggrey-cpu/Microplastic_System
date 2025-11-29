#!/usr/bin/env python3
"""
feature_selection_and_resampling.py

Usage:
- Edit the DATA_PATH variable or call load_data(df=your_dataframe).
- The script expects a column named 'Risk_Type' as the target.
- Installs required packages if missing: scikit-learn, pandas, numpy, imbalanced-learn, matplotlib, seaborn

What it does:
1. Loads preprocessed dataset (or accepts a DataFrame).
2. Reports class distribution for 'Risk_Type'.
3. Runs multiple feature selection methods:
   - Filter: ANOVA f_classif and mutual_info_classif
   - Filter (categorical): chi2 (requires non-negative; handled with MinMaxScaler)
   - Embedded: L1 LogisticRegression (SelectFromModel), RandomForest feature importances (SelectFromModel)
   - Wrapper: RFE with LogisticRegression
4. Aggregates top features from each method and shows votes/ranks.
5. Trains baseline models and evaluates F1-scores (macro/micro) with StratifiedKFold.
6. Provides resampling helpers (SMOTE, SMOTENC, RandomOverSampler, RandomUnderSampler) and compares results.
7. Runs GridSearchCV for basic hyperparameter tuning inside a pipeline (including sampling).
"""

import argparse
import warnings
from collections import Counter, defaultdict
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

# sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2, SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# imbalanced-learn
try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception as e:
    raise ImportError(
        "imblearn is required for resampling (pip install imbalanced-learn). Original error: {}".format(e)
    )

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# ---------- Configuration ----------
DATA_PATH = "preprocessed_data.csv"  # change to your preprocessed csv path or pass a df to load_data()
TARGET_COL = "Risk_Type"
RANDOM_STATE = 42
N_JOBS = -1
TOP_K = 20  # top features to display from each method
# -----------------------------------


def load_data(path: Optional[str] = None, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Load dataset from CSV or accept a DataFrame directly.
    Ensures TARGET_COL exists.
    """
    if df is not None:
        data = df.copy()
    elif path is not None:
        data = pd.read_csv(path)
    else:
        raise ValueError("Provide either path or df")

    if TARGET_COL not in data.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in data. Columns: {data.columns.tolist()}")

    return data


def show_class_distribution(y: pd.Series, plot: bool = True) -> None:
    counts = y.value_counts()
    print("Class distribution (counts):")
    print(counts)
    print("\nClass distribution (percent):")
    print((counts / counts.sum() * 100).round(2))

    if plot:
        plt.figure(figsize=(8, 4))
        sns.barplot(x=counts.index.astype(str), y=counts.values)
        plt.title("Risk_Type class distribution")
        plt.ylabel("Count")
        plt.xlabel("Risk_Type")
        plt.show()


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].copy()
    # encode target if needed
    if y.dtype == object or y.dtype.name == "category":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index, name=y.name)
        # store mapping if you need
        print("Target label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
    return X, y


def identify_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"Identified {len(num_cols)} numeric columns and {len(cat_cols)} categorical columns.")
    return num_cols, cat_cols


def run_filter_methods(X: pd.DataFrame, y: pd.Series, num_cols: List[str], cat_cols: List[str], top_k: int = TOP_K) -> Dict[str, List[Tuple[str, float]]]:
    results = {}

    if len(num_cols) > 0:
        # ANOVA f_classif (numerical inputs)
        skb_f = SelectKBest(score_func=f_classif, k=min(top_k, len(num_cols)))
        skb_f.fit(X[num_cols], y)
        scores_f = sorted(zip(num_cols, skb_f.scores_), key=lambda x: -np.nan_to_num(x[1]))
        results['f_classif'] = scores_f[:top_k]

        # mutual_info_classif
        skb_mi = SelectKBest(score_func=mutual_info_classif, k=min(top_k, len(num_cols)))
        skb_mi.fit(X[num_cols], y)
        scores_mi = sorted(zip(num_cols, skb_mi.scores_), key=lambda x: -np.nan_to_num(x[1]))
        results['mutual_info'] = scores_mi[:top_k]
    else:
        results['f_classif'] = []
        results['mutual_info'] = []

    if len(cat_cols) > 0:
        # chi2 requires non-negative data, we'll MinMax scale categorical encoded as integers or one-hot
        # Try label-encoding categories into integer codes (works if they are ordinal-ish)
        X_cat_encoded = X[cat_cols].apply(lambda col: col.astype('category').cat.codes)
        scaler = MinMaxScaler()
        X_cat_scaled = scaler.fit_transform(X_cat_encoded.fillna(0))
        skb_chi = SelectKBest(score_func=chi2, k=min(top_k, len(cat_cols)))
        skb_chi.fit(X_cat_scaled, y)
        scores_chi = sorted(zip(cat_cols, skb_chi.scores_), key=lambda x: -np.nan_to_num(x[1]))
        results['chi2_categorical'] = scores_chi[:top_k]
    else:
        results['chi2_categorical'] = []

    return results


def run_embedded_methods(X: pd.DataFrame, y: pd.Series, top_k: int = TOP_K) -> Dict[str, List[Tuple[str, float]]]:
    results = {}

    # L1 Logistic Regression (sparse) - good for linear relationships
    try:
        lr = LogisticRegression(penalty='l1', solver='saga', max_iter=5000, random_state=RANDOM_STATE, class_weight='balanced')
        sfm_lr = SelectFromModel(lr, max_features=top_k)
        sfm_lr.fit(X.fillna(0), y)
        # coefficients absolute value
        if hasattr(sfm_lr.estimator_, 'coef_'):
            coefs = np.abs(sfm_lr.estimator_.coef_).mean(axis=0)
            feature_names = X.columns[sfm_lr.estimator_.coef_.shape[1] - len(coefs):] if False else X.columns
            scores = sorted(zip(X.columns, coefs), key=lambda x: -x[1])
            results['l1_logistic'] = scores[:top_k]
        else:
            # fallback to selected features only
            sel = X.columns[sfm_lr.get_support()]
            results['l1_logistic'] = [(f, 1.0) for f in sel[:top_k]]
    except Exception as e:
        print("L1 logistic failed:", e)
        results['l1_logistic'] = []

    # RandomForest feature importances
    try:
        rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=N_JOBS, class_weight='balanced')
        rf.fit(X.fillna(0), y)
        importances = rf.feature_importances_
        scores_rf = sorted(zip(X.columns, importances), key=lambda x: -x[1])
        results['random_forest'] = scores_rf[:top_k]
    except Exception as e:
        print("RandomForest embedded method failed:", e)
        results['random_forest'] = []

    return results


def run_wrapper_methods(X: pd.DataFrame, y: pd.Series, top_k: int = TOP_K) -> Dict[str, List[Tuple[str, float]]]:
    results = {}
    # RFE with logistic regression
    try:
        lr = LogisticRegression(penalty='l2', solver='liblinear', max_iter=2000, random_state=RANDOM_STATE, class_weight='balanced')
        n_features_to_select = min(top_k, X.shape[1])
        rfe = RFE(estimator=lr, n_features_to_select=n_features_to_select, step=0.1)
        rfe.fit(X.fillna(0), y)
        ranking = list(zip(X.columns, rfe.ranking_))
        # ranking 1 means selected
        selected = [(name, 1.0) for name, rank in ranking if rank == 1]
        # for ordering present features by importance if estimator has coef_
        if hasattr(rfe.estimator_, 'coef_'):
            coefs = np.abs(rfe.estimator_.coef_).mean(axis=0)
            scores = sorted(zip(X.columns, coefs), key=lambda x: -x[1])
            results['rfe_logistic'] = scores[:top_k]
        else:
            results['rfe_logistic'] = selected[:top_k]
    except Exception as e:
        print("RFE failed:", e)
        results['rfe_logistic'] = []

    return results


def aggregate_feature_rankings(method_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Aggregate rankings from multiple methods into a single DataFrame with votes and average rank.
    method_results: mapping method_name -> list of (feature, score)
    """
    votes = defaultdict(lambda: [])
    for method, items in method_results.items():
        for rank, (feat, score) in enumerate(items, start=1):
            votes[feat].append(rank)

    agg = []
    for feat, ranks in votes.items():
        agg.append({
            'feature': feat,
            'votes': len(ranks),
            'avg_rank': np.mean(ranks)
        })
    df_agg = pd.DataFrame(agg).sort_values(['votes', 'avg_rank'], ascending=[False, True])
    return df_agg


def evaluate_models(X: pd.DataFrame, y: pd.Series, feature_subset: Optional[List[str]] = None, cv_splits: int = 5) -> Dict[str, float]:
    """
    Train and evaluate LogisticRegression and RandomForest using StratifiedKFold CV, report macro/micro F1.
    Returns summary dict.
    """
    results = {}
    X_use = X[feature_subset] if feature_subset is not None else X
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=5000, solver='saga', class_weight='balanced', random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=N_JOBS, class_weight='balanced')
    }

    for name, model in models.items():
        f1_macro = cross_val_score(model, X_use.fillna(0), y, scoring='f1_macro', cv=skf, n_jobs=N_JOBS).mean()
        f1_micro = cross_val_score(model, X_use.fillna(0), y, scoring='f1_micro', cv=skf, n_jobs=N_JOBS).mean()
        results[name] = {'f1_macro': float(f1_macro), 'f1_micro': float(f1_micro)}
        print(f"{name}: f1_macro={f1_macro:.4f}, f1_micro={f1_micro:.4f}")
    return results


def resample_data(X: pd.DataFrame, y: pd.Series, method: str = 'none', categorical_features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Resample X,y using method:
      - 'none' : no resampling
      - 'oversample' : RandomOverSampler
      - 'undersample' : RandomUnderSampler
      - 'smote' : SMOTE (numeric only)
      - 'smotenc' : SMOTENC (for datasets with categorical_features indices)
    categorical_features: list of column names which are categorical (required for smotenc)
    """
    if method == 'none':
        return X, y

    if method == 'oversample':
        ros = RandomOverSampler(random_state=RANDOM_STATE)
        X_res, y_res = ros.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)

    if method == 'undersample':
        rus = RandomUnderSampler(random_state=RANDOM_STATE)
        X_res, y_res = rus.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)

    if method == 'smote':
        sm = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = sm.fit_resample(X.select_dtypes(include=[np.number]), y)
        # combine numeric resampled with categorical repeated rows if any
        # If dataset has categorical columns we need SMOTENC; here we'll just return numeric-only resampled
        return pd.DataFrame(X_res, columns=X.select_dtypes(include=[np.number]).columns), pd.Series(y_res, name=y.name)

    if method == 'smotenc':
        if not categorical_features:
            raise ValueError("categorical_features must be provided for smotenc")
        cat_idx = [list(X.columns).index(c) for c in categorical_features]
        smnc = SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE)
        X_res, y_res = smnc.fit_resample(X.fillna(0), y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)

    raise ValueError("Unknown resampling method: " + method)


def tune_model_with_resampling(X: pd.DataFrame, y: pd.Series, sampler: Optional[str] = None, categorical_features: Optional[List[str]] = None) -> Tuple[GridSearchCV, Dict]:
    """
    Create a pipeline (optional sampler -> scaler -> classifier) and run GridSearchCV for RandomForest and LogisticRegression.
    sampler: 'none', 'oversample', 'smote', 'smotenc'
    Returns fitted gridsearch (for last estimator) and best_params summary.
    """
    # Example grid for RandomForest
    if sampler and sampler != 'none':
        if sampler == 'oversample':
            sampler_obj = RandomOverSampler(random_state=RANDOM_STATE)
        elif sampler == 'smote':
            sampler_obj = SMOTE(random_state=RANDOM_STATE)
        elif sampler == 'smotenc':
            if not categorical_features:
                raise ValueError("Pass categorical_features for smotenc")
            cat_idx = [list(X.columns).index(c) for c in categorical_features]
            sampler_obj = SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE)
        else:
            sampler_obj = None
    else:
        sampler_obj = None

    pipe_steps = []
    if sampler_obj is not None:
        pipe_steps.append(('sampler', sampler_obj))
    pipe_steps.append(('scaler', StandardScaler()))
    pipe_steps.append(('clf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)))

    pipeline = ImbPipeline(pipe_steps)

    param_grid = {
        'clf__n_estimators': [100, 300],
        'clf__max_depth': [None, 10, 30],
        'clf__class_weight': [None, 'balanced']
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipeline, param_grid, scoring='f1_macro', cv=skf, n_jobs=N_JOBS, verbose=1)
    gs.fit(X.fillna(0), y)
    print("Best params:", gs.best_params_)
    print("Best score (f1_macro):", gs.best_score_)
    return gs, {'best_params': gs.best_params_, 'best_score': float(gs.best_score_)}


def demo_flow(path: Optional[str] = None, df: Optional[pd.DataFrame] = None):
    # Load
    data = load_data(path, df)
    X, y = split_features_target(data)
    show_class_distribution(y)

    num_cols, cat_cols = identify_column_types(X)

    print("\nRunning filter methods...")
    filter_results = run_filter_methods(X, y, num_cols, cat_cols, top_k=TOP_K)
    for k, v in filter_results.items():
        print(f"\nTop results for {k}:")
        for feat, score in v[:10]:
            print(f"  {feat}: {score:.6f}")

    print("\nRunning embedded methods...")
    embedded_results = run_embedded_methods(X, y, top_k=TOP_K)
    for k, v in embedded_results.items():
        print(f"\nTop results for {k}:")
        for feat, score in v[:10]:
            print(f"  {feat}: {score:.6f}")

    print("\nRunning wrapper methods (RFE)...")
    wrapper_results = run_wrapper_methods(X, y, top_k=TOP_K)
    for k, v in wrapper_results.items():
        print(f"\nTop results for {k}:")
        for feat, score in v[:10]:
            print(f"  {feat}: {score:.6f}")

    # Aggregate feature rankings
    method_results_combined = {**filter_results, **embedded_results, **wrapper_results}
    agg = aggregate_feature_rankings(method_results_combined)
    print("\nAggregated feature ranking (top 30):")
    print(agg.head(30))

    # Use top N aggregated features to evaluate models
    top_features = agg['feature'].head(30).tolist()
    print("\nEvaluating models using top aggregated features:")
    eval_results_top = evaluate_models(X, y, feature_subset=top_features, cv_splits=5)

    print("\nEvaluate baseline models on all features:")
    eval_results_all = evaluate_models(X, y, feature_subset=None, cv_splits=5)

    # Investigate class imbalance and resampling
    print("\nComparing resampling strategies (oversample, undersample, smote, smotenc if categorical available):")
    resample_methods = ['none', 'oversample', 'undersample']
    if len(num_cols) > 0:
        resample_methods.append('smote')
    if len(cat_cols) > 0:
        resample_methods.append('smotenc')

    resample_summary = {}
    for method in resample_methods:
        print(f"\nResampling method: {method}")
        try:
            X_res, y_res = resample_data(X, y, method=method, categorical_features=cat_cols if method == 'smotenc' else None)
            print(f"After resampling distribution: {Counter(y_res)}")
            # Evaluate on resampled data using top_features intersection (ensure features exist)
            features_to_use = [f for f in top_features if f in X_res.columns]
            print("Features used:", len(features_to_use))
            res_eval = evaluate_models(X_res, y_res, feature_subset=features_to_use if features_to_use else None, cv_splits=5)
            resample_summary[method] = res_eval
        except Exception as e:
            print("Resampling failed for method", method, "error:", e)

    # Example hyperparameter tuning with resampling in pipeline
    print("\nRunning GridSearchCV with RandomOverSampler + RandomForest (this may take a while)...")
    try:
        gs, gs_summary = tune_model_with_resampling(X[top_features], y, sampler='oversample', categorical_features=cat_cols)
    except Exception as e:
        print("GridSearchCV failed:", e)
        gs_summary = {}

    print("\nDone. Summaries:")
    print("Evaluation with top features:", eval_results_top)
    print("Evaluation with all features:", eval_results_all)
    print("Resampling summary:", resample_summary)
    print("GridSearch summary:", gs_summary)

    # Optionally save top features to CSV
    agg.to_csv("feature_ranking_aggregated.csv", index=False)
    print("\nAggregated feature ranking saved to feature_ranking_aggregated.csv")

    # Return important objects if programmatic access desired
    return {
        'agg': agg,
        'filter_results': filter_results,
        'embedded_results': embedded_results,
        'wrapper_results': wrapper_results,
        'eval_top': eval_results_top,
        'eval_all': eval_results_all,
        'resample_summary': resample_summary,
        'gridsearch_summary': gs_summary
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature selection, class imbalance investigation, resampling, and basic tuning for Risk_Type")
    parser.add_argument("--path", type=str, default=DATA_PATH, help="Path to preprocessed CSV file (must contain 'Risk_Type')")
    args = parser.parse_args()
    demo_flow(path=args.path)
