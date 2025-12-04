import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from pandas.errors import EmptyDataError, ParserError

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Microplastic Risk Analysis", layout="wide")

NUMERIC_COLS = [
    "MP_Count_per_L",
    "Risk_Score",
    "Microplastic_Size_mm_midpoint",
    "Density_midpoint",
]

CATEGORICAL_COLS = [
    "Location",
    "Shape",
    "Polymer_Type",
    "pH",
    "Salinity",
    "Industrial_Activity",
    "Population_Density",
    "Author",
]

TARGET_RISK_TYPE = "Risk_Type"
TARGET_RISK_LEVEL = "Risk_Level"

# -------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None, path: str = "Microplastic.csv"):
    src = uploaded_file if uploaded_file is not None else path
    encodings_to_try = ["latin1", "utf-8", "cp1252"]

    last_err = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(src, encoding=enc)
            return df
        except (UnicodeDecodeError, EmptyDataError, ParserError) as e:
            last_err = e
            continue
        except FileNotFoundError:
            if uploaded_file is None:
                raise

    if last_err is not None:
        raise last_err
    return None

# -------------------------------------------------------
# PREPROCESSING
# -------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            df[col] = s.fillna(s.median())
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            mode_val = df[col].mode(dropna=True)
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])
    return df

def cap_outliers_iqr(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        clipped = np.where(s < low, low, s)
        clipped = np.where(clipped > high, high, clipped)
        df[col] = clipped
    return df

def transform_skewed(df: pd.DataFrame, cols):
    df = df.copy()
    cols_present = [c for c in cols if c in df.columns]
    if not cols_present:
        return df, pd.Series(dtype=float), []
    for col in cols_present:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    skewness = df[cols_present].skew(numeric_only=True)
    skewed_cols = skewness[skewness.abs() > 1].index.tolist()
    for col in skewed_cols:
        min_val = df[col].min()
        if pd.isna(min_val):
            continue
        shift = (abs(min_val) + 1e-6) if min_val <= 0 else 0
        df[col] = np.log1p(df[col] + shift)
    return df, skewness, skewed_cols

def scale_numeric(df: pd.DataFrame, cols):
    df = df.copy()
    scaler = StandardScaler()
    cols_present = [c for c in cols if c in df.columns]
    if cols_present:
        df[cols_present] = scaler.fit_transform(df[cols_present])
    return df, scaler

def preprocess_for_model(df: pd.DataFrame):
    df = df.copy()
    if TARGET_RISK_TYPE in df.columns and TARGET_RISK_LEVEL in df.columns:
        df = df.dropna(subset=[TARGET_RISK_TYPE, TARGET_RISK_LEVEL])
    df = handle_missing_values(df)
    df = cap_outliers_iqr(df, NUMERIC_COLS)
    df, skewness, skewed_cols = transform_skewed(df, NUMERIC_COLS)
    df, _ = scale_numeric(df, NUMERIC_COLS)
    y_type = df[TARGET_RISK_TYPE] if TARGET_RISK_TYPE in df.columns else None
    y_level = df[TARGET_RISK_LEVEL] if TARGET_RISK_LEVEL in df.columns else None
    drop_cols = [c for c in [TARGET_RISK_TYPE, TARGET_RISK_LEVEL] if c in df.columns]
    feature_df = df.drop(columns=drop_cols)
    existing_cat_cols = [c for c in CATEGORICAL_COLS if c in feature_df.columns]
    X = pd.get_dummies(feature_df, columns=existing_cat_cols, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df, X, y_type, y_level, skewness, skewed_cols

# -------------------------------------------------------
# SPLIT HELPERS
# -------------------------------------------------------
def merge_rare_classes(y: pd.Series, min_count: int = 2, other_label: str = "Other"):
    y = pd.Series(y).copy()
    counts = y.value_counts(dropna=True)
    rare = counts[counts < min_count].index
    y = y.where(~y.isin(rare), other_label)
    return y

def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    y = pd.Series(y)
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in the target.")
    counts = y.value_counts()
    min_class = int(counts.min())
    n = len(y)
    k = y.nunique()
    if min_class < 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        return (X_train, X_test, y_train, y_test), False, float(test_size)
    min_test_size = k / n
    max_test_size = 1 - (k / n)
    ts = float(test_size)
    ts = max(ts, min_test_size)
    if max_test_size > 0:
        ts = min(ts, max_test_size)
    for ts_try in [ts, 0.2, 0.15, 0.1, 0.05]:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=ts_try, random_state=random_state, stratify=y
            )
            return (X_train, X_test, y_train, y_test), True, float(ts_try)
        except ValueError:
            continue
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    return (X_train, X_test, y_train, y_test), False, float(test_size)

# -------------------------------------------------------
# MODELING
# -------------------------------------------------------
def train_models(X, y, test_size=0.2):
    y = pd.Series(y)
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in the target to train models.")
    before_counts = y.value_counts()
    y_merged = merge_rare_classes(y, min_count=2, other_label="Other")
    after_counts = y_merged.value_counts()
    merge_note = None
    if not before_counts.equals(after_counts):
        merge_note = {"before": before_counts, "after": after_counts}
    (X_train, X_test, y_train, y_test), used_stratify, final_test_size = safe_train_test_split(
        X, y_merged, test_size=test_size, random_state=42
    )
    split_note = (
        f"✅ Stratified split used (test_size={final_test_size:.2f})."
        if used_stratify
        else f"⚠️ Non-stratified split used (test_size={final_test_size:.2f}) because some classes are too small."
    )
    split_info = {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_counts": y_train.value_counts(),
        "y_test_counts": y_test.value_counts(),
        "used_stratify": used_stratify,
        "final_test_size": final_test_size,
    }
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, multi_class="auto"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    metrics_list = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics_list.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (weighted)": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall (weighted)": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1-score (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        })
    metrics_df = pd.DataFrame(metrics_list).set_index("Model")
    return models, metrics_df, split_info, split_note, merge_note

def smote_and_tune_logreg(X, y, test_size=0.2):
    y = pd.Series(y)
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in the target to run SMOTE and tuning.")
    before_counts = y.value_counts()
    y_merged = merge_rare_classes(y, min_count=2, other_label="Other")
    after_counts = y_merged.value_counts()
    merge_note = None
    if not before_counts.equals(after_counts):
        merge_note = {"before": before_counts, "after": after_counts}
    (X_train, X_test, y_train, y_test), used_stratify, final_test_size = safe_train_test_split(
        X, y_merged, test_size=test_size, random_state=42
    )
    split_note = (
        f"✅ Stratified split used (test_size={final_test_size:.2f})."
        if used_stratify
        else f"⚠️ Non-stratified split used (test_size={final_test_size:.2f}) because some classes are too small."
    )
    split_info = {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_counts": y_train.value_counts(),
        "y_test_counts": y_test.value_counts(),
        "used_stratify": used_stratify,
        "final_test_size": final_test_size,
    }
    smote_used = True
    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
    except ValueError:
        smote_used = False
        X_res, y_res = X_train, y_train
    param_grid = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]}
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, multi_class="auto"),
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=5,
        n_jobs=-1,
    )
    grid.fit(X_res, y_res)
    best_lr = grid.best_estimator_
    y_pred = best_lr.predict(X_test)
    tuned_metrics = pd.DataFrame([{
        "Model": "LogReg (tuned + SMOTE)" if smote_used else "LogReg (tuned, no SMOTE)",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (weighted)": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-score (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }]).set_index("Model")
    return best_lr, tuned_metrics, grid.best_params_, split_info, split_note, merge_note, smote_used

# -------------------------------------------------------
# LEAKAGE-SAFE CV HELPERS
# -------------------------------------------------------
def build_preprocess_pipeline(df_raw: pd.DataFrame):
    numeric_features = [c for c in NUMERIC_COLS if c in df_raw.columns]
    categorical_features = [c for c in CATEGORICAL_COLS if c in df_raw.columns]
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor

def run_cross_validation(df_raw: pd.DataFrame, target_col: str, model_name: str,
                         n_splits: int = 5, stratified: bool = True, use_smote: bool = False):
    if target_col not in df_raw.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    df = df_raw.dropna(subset=[target_col]).copy()
    y = df[target_col]
    X = df.drop(columns=[c for c in [TARGET_RISK_TYPE, TARGET_RISK_LEVEL] if c in df.columns])
    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes in the target for cross-validation.")
    y = merge_rare_classes(y, min_count=2, other_label="Other")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, multi_class="auto"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    if model_name not in models:
        raise ValueError("Unknown model selected.")
    model = models[model_name]
    if stratified:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    preprocessor = build_preprocess_pipeline(df_raw)
    if use_smote:
        pipe = ImbPipeline(steps=[
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", model),
        ])
    else:
        pipe = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", model),
        ])
    scoring = {
        "accuracy": "accuracy",
        "precision_w": "precision_weighted",
        "recall_w": "recall_weighted",
        "f1_w": "f1_weighted",
    }
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise")
    summary = {}
    for k in scoring.keys():
        arr = scores[f"test_{k}"]
        summary[k] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    summary_df = pd.DataFrame(summary).T
    summary_df = summary_df.rename(index={
        "accuracy": "Accuracy",
        "precision_w": "Precision (weighted)",
        "recall_w": "Recall (weighted)",
        "f1_w": "F1-score (weighted)",
    })
    return summary_df, scores

# -------------------------------------------------------
# VISUALS
# -------------------------------------------------------
def plot_hist_box(df, col):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if len(s) == 0:
        axes[0].text(0.5, 0.5, f"No numeric data for {col}", ha="center", va="center")
        axes[1].text(0.5, 0.5, f"No numeric data for {col}", ha="center", va="center")
    else:
        sns.histplot(s, kde=True, ax=axes[0])
        axes[0].set_title(f"Histogram of {col}")
        sns.boxplot(x=s, ax=axes[1])
        axes[1].set_title(f"Boxplot of {col}")
    plt.tight_layout()
    return fig

def plot_scatter(df, x_col, y_col):
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    fig, ax = plt.subplots(figsize=(6, 4))
    if mask.sum() == 0:
        ax.text(0.5,  0.5, f"No numeric data for {x_col} and {y_col}", ha="center", va="center")
    else:
        ax.scatter(x[mask], y[mask], alpha=0.7)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")
    plt.tight_layout()
    return fig

def plot_metrics_bar(metrics_df, title_suffix=""):
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_df[["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-score (weighted)"]].plot(
        kind="bar", ax=ax
    )
    ax.set_title(f"Model Performance {title_suffix}")
    ax.set_ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig

def plot_box_by_category_readable(
    df,
    value_col,
    category_col,
    top_n=8,
    other_label="Other",
    figsize=(12, 5),
    horizontal=True,
):
    val = pd.to_numeric(df[value_col], errors="coerce")
    cat = (
        df[category_col]
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    data = pd.DataFrame({value_col: val, category_col: cat}).dropna(subset=[value_col, category_col])
    fig, ax = plt.subplots(figsize=figsize)
    if data.empty:
        ax.text(0.5, 0.5, f"No usable data for {value_col} by {category_col}", ha="center", va="center")
        plt.tight_layout()
        return fig
    counts = data[category_col].value_counts()
    keep = counts.head(top_n).index
    data[category_col] = np.where(data[category_col].isin(keep), data[category_col], other_label)
    order = data.groupby(category_col)[value_col].median().sort_values().index.tolist()
    if horizontal:
        sns.boxplot(data=data, y=category_col, x=value_col, order=order, ax=ax)
    else:
        sns.boxplot(data=data, x=category_col, y=value_col, order=order, ax=ax)
        ax.tick_params(axis="x", labelrotation=35)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    ax.set_title(f"{value_col} by {category_col} (Top {top_n} + {other_label})")
    plt.tight_layout()
    return fig

def plot_categorical_topn_bar(
    series: pd.Series,
    title: str,
    top_n: int = 15,
    other_label: str = "Other",
    figsize=(10, 6),
):
    s = series.dropna().astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan}).dropna()
    counts = s.value_counts()
    if counts.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No category data available", ha="center", va="center")
        plt.tight_layout()
        return fig, counts
    top = counts.head(top_n)
    remainder = counts.iloc[top_n:].sum()
    if remainder > 0:
        top = pd.concat([top, pd.Series({other_label: remainder})])
    fig, ax = plt.subplots(figsize=figsize)
    top.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Count")
    ax.set_ylabel(series.name if series.name else "Category")
    plt.tight_layout()
    return fig, counts

# -------------------------------------------------------
# APP
# -------------------------------------------------------
def main():
    st.title("Microplastic Risk Prediction – Streamlit App")
    st.markdown(
        """
        This app demonstrates the analysis and modeling workflow for predicting **Risk_Type**
        and **Risk_Level** using microplastic and environmental features.
        """
    )

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Data Overview & Task 1",
            "Preprocessing (Task 2)",
            "Feature Selection & Relevance (Task 3 & 6)",
            "Classification Modeling (Tasks 4, 5 & 7)",
            "Polymer Type Distribution",
            "SMOTE & Hyperparameter Tuning (Risk_Type)",
            "Cross Validation (K-Fold)",
        ],
    )

    st.sidebar.subheader("Data source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Microplastic CSV",
        type=["csv"],
        help="If you don't upload anything, the app will try to use 'Microplastic.csv' from the app folder."
    )

    with st.spinner("Loading data..."):
        try:
            df_raw = load_data(uploaded_file=uploaded_file)
        except UnicodeDecodeError:
            st.error("⚠️ Unable to decode the file. Please upload a proper CSV (text).")
            st.stop()
        except EmptyDataError:
            st.error("⚠️ The uploaded file appears empty/unreadable as CSV.")
            st.stop()
        except ParserError:
            st.error("⚠️ The file is not a valid CSV format. Re-export as CSV and try again.")
            st.stop()
        except FileNotFoundError:
            df_raw = None

    if df_raw is None:
        st.error("❌ No dataset found. Upload a CSV or add 'Microplastic.csv' beside app.py.")
        st.stop()

    # -------------- The page logic here is *identical* to your full original code -----------------
    # (Your tabs, all processing, all visuals, are preserved as in the reference above!)

    # -------------------- PAGE 1 --------------------
    if page == "Data Overview & Task 1":
        st.header("Data Overview & Task 1: Risk_Score Analysis")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Raw Data",
            "Risk_Score Distribution",
            "MP_Count vs Risk_Score",
            "Risk_Score by Risk_Level",
        ])

        with tab1:
            st.subheader("Raw Dataset (first 10 rows)")
            st.dataframe(df_raw.head(10))
            st.markdown(f"**Shape:** `{df_raw.shape[0]}` rows × `{df_raw.shape[1]}` columns")
            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - This section verifies that the dataset is loaded successfully.
                - The first few rows help confirm that the expected columns are present and values look reasonable.
                """
            )

        with tab2:
            if "Risk_Score" in df_raw.columns:
                st.subheader("Distribution of Risk_Score (Histogram & Boxplot)")
                st.pyplot(plot_hist_box(df_raw, "Risk_Score"))
                st.markdown("**Interpretation:**")
                st.markdown(
                    """
                    - The histogram shows frequency of Risk_Score values.
                    - The boxplot summarizes dispersion and flags potential outliers.
                    """
                )
            else:
                st.info("Column 'Risk_Score' not found in the dataset.")

        with tab3:
            if "MP_Count_per_L" in df_raw.columns and "Risk_Score" in df_raw.columns:
                st.subheader("Relationship between Risk_Score and MP_Count_per_L")
                st.pyplot(plot_scatter(df_raw, "MP_Count_per_L", "Risk_Score"))
                st.markdown("**Interpretation:**")
                st.markdown(
                    """
                    - If an upward trend exists, higher microplastic concentration tends to correspond to higher risk.
                    - If scattered, other variables likely contribute to risk.
                    """
                )
            else:
                st.info("Columns 'MP_Count_per_L' and/or 'Risk_Score' not found.")

        with tab4:
            if "Risk_Level" in df_raw.columns and "Risk_Score" in df_raw.columns:
                st.subheader("Difference in Risk_Score by Risk_Level (Boxplot)")
                st.pyplot(
                    plot_box_by_category_readable(
                        df_raw,
                        value_col="Risk_Score",
                        category_col="Risk_Level",
                        top_n=8,
                        figsize=(12, 5),
                        horizontal=True,
                    )
                )
                st.markdown("**Interpretation:**")
                st.markdown(
                    """
                    - This compares Risk_Score distributions across Risk_Level categories.
                    - Overlap suggests borderline samples or thresholds that may require refinement.
                    """
                )
            else:
                st.info("Columns 'Risk_Level' and/or 'Risk_Score' not found.")

    # -------------------- PAGE 2 --------------------
    elif page == "Preprocessing (Task 2)":
        st.header("Task 2: Preprocessing")
        df_clean, X, y_type, y_level, skewness, skewed_cols = preprocess_for_model(df_raw)

        tab1, tab2, tab3, tab4 = st.tabs([
            "Before Preprocessing",
            "After Preprocessing",
            "Skewness",
            "Encoded Features",
        ])

        with tab1:
            numeric_present = [c for c in NUMERIC_COLS if c in df_raw.columns]
            if numeric_present:
                st.subheader("Before Preprocessing – Descriptive Stats")
                st.write(df_raw[numeric_present].describe())
                st.markdown("**Interpretation:**")
                st.markdown(
                    """
                    - Baseline statistics before cleaning.
                    - Large ranges and extreme maxima suggest outliers and skewness.
                    """
                )
            else:
                st.info("No numeric columns found for descriptive stats.")

        with tab2:
            numeric_present_clean = [c for c in NUMERIC_COLS if c in df_clean.columns]
            if numeric_present_clean:
                st.subheader("After Preprocessing – Descriptive Stats")
                st.write(df_clean[numeric_present_clean].describe())
                st.markdown("**Interpretation:**")
                st.markdown(
                    """
                    - Outlier capping and transformations stabilize numeric distributions.
                    - Scaling improves comparability across features.
                    """
                )
            else:
                st.info("No cleaned numeric columns found.")

        with tab3:
            st.subheader("Skewness (Before Transform)")
            st.write(skewness)
            if len(skewed_cols) > 0:
                st.write("Skewed columns transformed (log1p):")
                st.write(skewed_cols)
            else:
                st.write("No columns exceeded skewness threshold.")
            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - Skewness measures distribution asymmetry.
                - Log-transform helps reduce long tails and improve stability.
                """
            )

        with tab4:
            st.subheader("Encoded Feature Matrix (X) – First 10 Rows")
            st.dataframe(X.head(10))
            st.write("Shape of X:", X.shape)
            st.markdown("**Interpretation:**")
            st.markdown(
                """
                - Categorical variables are converted into numeric features (one-hot encoding).
                - The feature matrix is ready for machine learning.
                """
            )

    # -------------------- PAGE 3 --------------------
    elif page == "Feature Selection & Relevance (Task 3 & 6)":
        st.header("Tasks 3 & 6: Feature Selection / Relevance")
        _, X, y_type, y_level, _, _ = preprocess_for_model(df_raw)

        st.markdown(
            """
            A **Random Forest** model estimates feature importance.
            Higher importance indicates stronger contribution to prediction.
            """
        )

        tab_rt, tab_rl = st.tabs(["Risk_Type Feature Importance", "Risk_Level Feature Importance"])

        with tab_rt:
            if y_type is not None:
                st.subheader("Random Forest Feature Importance – Risk_Type")
                rf_rt = RandomForestClassifier(n_estimators=200, random_state=42)
                rf_rt.fit(X, y_type)
                importances_rt = pd.Series(rf_rt.feature_importances_, index=X.columns).sort_values(ascending=False)

                st.write("Top 10 features (Risk_Type):")
                st.dataframe(importances_rt.head(10))
                fig = plt.figure(figsize=(8, 4))
                importances_rt.head(10).sort_values().plot(kind="barh")
                plt.title("Top 10 Feature Importances (Risk_Type)")
                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("**Interpretation:**")
                st.markdown("- The top-ranked features contribute most strongly to Risk_Type prediction.")
            else:
                st.warning("Risk_Type column not found.")

        with tab_rl:
            if y_level is not None:
                st.subheader("Random Forest Feature Importance – Risk_Level")
                rf_rl = RandomForestClassifier(n_estimators=200, random_state=42)
                rf_rl.fit(X, y_level)
                importances_rl = pd.Series(rf_rl.feature_importances_, index=X.columns).sort_values(ascending=False)

                st.write("Top 10 features (Risk_Level):")
                st.dataframe(importances_rl.head(10))
                fig = plt.figure(figsize=(8, 4))
                importances_rl.head(10).sort_values().plot(kind="barh")
                plt.title("Top 10 Feature Importances (Risk_Level)")
                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("**Interpretation:**")
                st.markdown("- The top-ranked features contribute most strongly to Risk_Level prediction.")
            else:
                st.warning("Risk_Level column not found.")

    # -------------------- PAGE 4 --------------------
    elif page == "Classification Modeling (Tasks 4, 5 & 7)":
        st.header("Tasks 4, 5 & 7: Classification Modeling")
        _, X, y_type, y_level, _, _ = preprocess_for_model(df_raw)

        tab1, tab2 = st.tabs(["Risk-Type Models", "Risk-Level Models"])

        with tab1:
            if y_type is None:
                st.warning("Risk_Type column not found; cannot train models for Risk-Type.")
            else:
                st.subheader("Models for Risk-Type")
                _, metrics_rt, split_info_rt, split_note_rt, merge_note_rt = train_models(X, y_type)

                st.write("Performance Metrics – Risk-Type")
                st.dataframe(metrics_rt.style.format("{:.3f}"))
                st.pyplot(plot_metrics_bar(metrics_rt, "(Risk-Type)"))
                st.info(split_note_rt)

                if merge_note_rt is not None:
                    with st.expander("Rare-class merging details (small classes → 'Other')"):
                        st.write("Before:")
                        st.write(merge_note_rt["before"])
                        st.write("After:")
                        st.write(merge_note_rt["after"])

                st.markdown("**Train–Test Split (Risk-Type):**")
                st.markdown(
                    f"""
                    - Training set shape: `{split_info_rt['X_train_shape']}`  
                    - Test set shape: `{split_info_rt['X_test_shape']}`
                    """
                )
                st.write("Class distribution in **training set**:")
                st.write(split_info_rt["y_train_counts"])
                st.write("Class distribution in **test set**:")
                st.write(split_info_rt["y_test_counts"])

                st.markdown("**Interpretation:**")
                st.markdown(
                    """
                    - Compare models using F1-score when classes are imbalanced.
                    - Stratified split preserves proportions; rare-class merging improves stability.
                    """
                )

        with tab2:
            if y_level is None:
                st.warning("Risk_Level column not found; cannot train models for Risk-Level.")
            else:
                st.subheader("Models for Risk-Level")
                _, metrics_rl, split_info_rl, split_note_rl, merge_note_rl = train_models(X, y_level)

                st.write("Performance Metrics – Risk-Level")
                st.dataframe(metrics_rl.style.format("{:.3f}"))
                st.pyplot(plot_metrics_bar(metrics_rl, "(Risk-Level)"))
                st.info(split_note_rl)

                if merge_note_rl is not None:
                    with st.expander("Rare-class merging details (small classes → 'Other')"):
                        st.write("Before:")
                        st.write(merge_note_rl["before"])
                        st.write("After:")
                        st.write(merge_note_rl["after"])

                st.markdown("**Train–Test Split (Risk-Level):**")
                st.markdown(
                    f"""
                    - Training set shape: `{split_info_rl['X_train_shape']}`  
                    - Test set shape: `{split_info_rl['X_test_shape']}`
                    """
                )
                st.write("Class distribution in **training set**:")
                st.write(split_info_rl["y_train_counts"])
                st.write("Class distribution in **test set**:")
                st.write(split_info_rl["y_test_counts"])

                st.markdown("**Interpretation:**")
                st.markdown(
                    """
                    - Metrics evaluate the ability to classify Risk_Level categories.
                    - Small classes may be merged into 'Other' to enable stable splitting.
                    """
                )

        st.subheader("Overall Interpretation")
        st.markdown(
            """
            - Evaluation quality depends on whether each class appears in both train and test sets.
            - The app automatically attempts stratification and stabilizes evaluation using rare-class merging.
            """
        )

    # -------------------- PAGE 5 --------------------
    elif page == "Polymer Type Distribution":
        st.header("Polymer Type Distribution")
        df = handle_missing_values(df_raw)

        if "Polymer_Type" in df.columns:
            polymer = df["Polymer_Type"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
            polymer = polymer.dropna()
            vc = polymer.value_counts()

            tabA, tabB = st.tabs(["Counts Table", "Readable Plot (Top N + Other)"])

            with tabA:
                st.subheader("Value Counts of Polymer_Type")
                st.dataframe(vc.rename("count"))
                st.markdown(
                    """
                    **Interpretation:**
                    - This table lists each Polymer_Type and its frequency.
                    - Many unique text labels can appear due to inconsistent naming; normalization helps consolidate categories.
                    """
                )

            with tabB:
                st.subheader("Bar Plot of Polymer_Type Distribution (Readable)")
                top_n = st.slider("Show Top N polymer types", min_value=5, max_value=30, value=15, step=1)
                fig, _ = plot_categorical_topn_bar(
                    polymer,
                    title=f"Distribution of Polymer_Type (Top {top_n} + Other)",
                    top_n=top_n,
                    other_label="Other",
                    figsize=(10, 7),
                )
                st.pyplot(fig)
                st.markdown(
                    f"""
                    **Interpretation:**
                    - This plot shows the **Top {top_n} most common** polymer types and groups the remaining types under **'Other'**.
                    - A horizontal bar chart is used so long polymer names remain readable.
                    """
                )
        else:
            st.warning("Column 'Polymer_Type' not found in the dataset.")

    # -------------------- PAGE 6 --------------------
    elif page == "SMOTE & Hyperparameter Tuning (Risk_Type)":
        st.header("Address Class Imbalance & Tune Logistic Regression (Risk-Type)")
        _, X, y_type, _, _, _ = preprocess_for_model(df_raw)

        if y_type is None:
            st.warning("Risk_Type column not found; cannot run SMOTE or tuning.")
            return

        tab1, tab2 = st.tabs(["Original Distribution & Base Models", "SMOTE + Tuning & Comparison"])

        with tab1:
            st.subheader("Class Distribution of Risk-Type (Original)")
            st.write(pd.Series(y_type).value_counts())

            _, base_metrics_rt, split_info_base_rt, split_note_base, merge_note_base = train_models(X, y_type)

            st.subheader("Base Models Performance (Risk-Type)")
            st.dataframe(base_metrics_rt.style.format("{:.3f}"))
            st.pyplot(plot_metrics_bar(base_metrics_rt, "(Risk-Type – Base)"))
            st.info(split_note_base)

            if merge_note_base is not None:
                with st.expander("Rare-class merging details (small classes → 'Other')"):
                    st.write("Before:")
                    st.write(merge_note_base["before"])
                    st.write("After:")
                    st.write(merge_note_base["after"])

            st.markdown("**Train–Test Split (Base):**")
            st.markdown(
                f"""
                - Training set shape: `{split_info_base_rt['X_train_shape']}`  
                - Test set shape: `{split_info_base_rt['X_test_shape']}`
                """
            )

        with tab2:
            st.subheader("SMOTE + Hyperparameter Tuning")
            with st.spinner("Running SMOTE + GridSearchCV..."):
                best_lr, tuned_metrics, best_params, split_info_smote, split_note_smote, merge_note_smote, smote_used = (
                    smote_and_tune_logreg(X, y_type)
                )

            st.write("Best Hyperparameters (Logistic Regression):")
            st.json(best_params)
            st.info(split_note_smote)

            if not smote_used:
                st.warning("SMOTE could not be applied due to very small minority classes; tuning continued without SMOTE.")

            if merge_note_smote is not None:
                with st.expander("Rare-class merging details (small classes → 'Other')"):
                    st.write("Before:")
                    st.write(merge_note_smote["before"])
                    st.write("After:")
                    st.write(merge_note_smote["after"])

            st.subheader("Tuned Logistic Regression Performance")
            st.dataframe(tuned_metrics.style.format("{:.3f}"))

            try:
                _, base_metrics_compare, _, _, _ = train_models(X, y_type)
                combined = pd.concat([base_metrics_compare, tuned_metrics])
                st.subheader("Comparison: Tuned Logistic Regression vs Base Models")
                st.dataframe(combined.style.format("{:.3f}"))
                st.pyplot(plot_metrics_bar(combined, "(Risk-Type – Base vs Tuned)"))
            except ValueError:
                st.warning("Could not generate base-vs-tuned comparison due to limited class sizes.")

    # -------------------- PAGE 7 --------------------
    elif page == "Cross Validation (K-Fold)":
        st.header("Cross Validation (K-Fold / Stratified K-Fold)")

        st.markdown(
            """
            This page runs **leakage-safe cross-validation** using a preprocessing **Pipeline**.
            - For classification, **Stratified K-Fold** is recommended (keeps class balance per fold).
            - If enabled, **SMOTE is applied inside each fold** (correct way).
            """
        )

        target = st.selectbox("Select target", [TARGET_RISK_TYPE, TARGET_RISK_LEVEL])
        model_name = st.selectbox("Select model", ["Logistic Regression", "Random Forest", "Gradient Boosting"])
        n_splits = st.slider("Number of folds (k)", min_value=3, max_value=10, value=5, step=1)
        stratified = st.checkbox("Use Stratified K-Fold (recommended for classification)", value=True)

       
