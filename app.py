import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

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

NUMERIC_MODEL_COLS = [
    "MP_Count_per_L_num",
    "Risk_Score_num",
    "Microplastic_Size_mm_midpoint",
    "Density_midpoint",
    "Salinity_num",
    "pH_num",
]

# -------------------------------------------------------
# UTILS (ROBUST DATA CLEANUP)
# -------------------------------------------------------
def extract_midpoint(val):
    """Extract midpoint from range strings like '0.1–5.0', '1.3–1.4', or '0.91�1.05'. If single value, return as float."""
    if pd.isna(val): return np.nan
    s = str(val).replace("�", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"[^\d\.\-/]", "", s)
    if "-" in s:
        try:
            numbers = [float(n) for n in s.split("-") if n.strip() != ""]
            if len(numbers) == 2:
                return sum(numbers) / 2.0
        except:
            return np.nan
    if "/" in s:
        nums = re.findall(r"[\d\.]+", s)
        try:
            return float(nums[0]) if nums else np.nan
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan

def clean_salinity(val):
    """Extract numeric value from '33 ppt', '33 PSU', etc."""
    if pd.isna(val): return np.nan
    s = str(val)
    nums = re.findall(r"\d+\.?\d*", s)
    return float(nums[0]) if nums else np.nan

def clean_density(val):
    return extract_midpoint(val)

def clean_mp_count(val):
    s = str(val)
    nums = re.findall(r"[\d\.]+", s)
    if nums:
        try:
            return float(nums[0])
        except:
            return np.nan
    return np.nan

def clean_pH(val):
    s = str(val)
    match = re.search(r"\d+\.?\d*", s)
    if match:
        return float(match.group())
    return np.nan

def clean_numeric_columns(df):
    df_new = df.copy()
    # Rebuild numeric columns as midpoints
    if "Microplastic_Size_mm" in df_new.columns:
        df_new["Microplastic_Size_mm_midpoint"] = df_new["Microplastic_Size_mm"].apply(extract_midpoint)
    elif "Microplastic_Size_mm_midpoint" in df_new.columns:
        df_new["Microplastic_Size_mm_midpoint"] = df_new["Microplastic_Size_mm_midpoint"].apply(extract_midpoint)
    else:
        df_new["Microplastic_Size_mm_midpoint"] = np.nan

    if "Density" in df_new.columns:
        df_new["Density_midpoint"] = df_new["Density"].apply(clean_density)
    elif "Density_midpoint" in df_new.columns:
        df_new["Density_midpoint"] = df_new["Density_midpoint"].apply(clean_density)
    else:
        df_new["Density_midpoint"] = np.nan

    if "Salinity" in df_new.columns:
        df_new["Salinity_num"] = df_new["Salinity"].apply(clean_salinity)
    else:
        df_new["Salinity_num"] = np.nan

    if "MP_Count_per_L" in df_new.columns:
        df_new["MP_Count_per_L_num"] = df_new["MP_Count_per_L"].apply(clean_mp_count)
    else:
        df_new["MP_Count_per_L_num"] = np.nan

    if "Risk_Score" in df_new.columns:
        df_new["Risk_Score_num"] = pd.to_numeric(df_new["Risk_Score"], errors="coerce")
    else:
        df_new["Risk_Score_num"] = np.nan

    if "pH" in df_new.columns:
        df_new["pH_num"] = df_new["pH"].apply(clean_pH)
    else:
        df_new["pH_num"] = np.nan

    return df_new

# -------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------
@st.cache_data
def load_data(uploaded_file=None, path: str = "Microplastic.csv"):
    src = uploaded_file if uploaded_file is not None else path
    encodings_to_try = ["utf-8", "latin1", "cp1252"]
    last_err = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(src, encoding=enc, na_values=["N/A","ND","","-"])
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
def preprocess_for_model(df: pd.DataFrame):
    df = clean_numeric_columns(df)
    # Drop rows only if missing the targets
    targets = [TARGET_RISK_TYPE, TARGET_RISK_LEVEL]
    df = df.dropna(subset=[t for t in targets if t in df.columns])
    # Fill missing values in numeric columns with median
    for col in NUMERIC_MODEL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            med = df[col].median()
            df[col] = df[col].fillna(med)
        else:
            df[col] = np.nan

    # Fill missing in categorical columns with mode
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            mode_val = df[col].mode(dropna=True)
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])
            else:
                df[col] = df[col].fillna("Unknown")
        else:
            df[col] = "Unknown"

    # Encode features: get_dummies for categorical
    features = NUMERIC_MODEL_COLS.copy()
    for col in CATEGORICAL_COLS:
        features.append(col)
    feature_df = df[features]
    X = pd.get_dummies(feature_df, columns=[c for c in CATEGORICAL_COLS if c in feature_df.columns], drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    y_type = df[TARGET_RISK_TYPE] if TARGET_RISK_TYPE in df.columns else None
    y_level = df[TARGET_RISK_LEVEL] if TARGET_RISK_LEVEL in df.columns else None
    return df, X, y_type, y_level

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if y.nunique()>1 else None)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    metrics_list = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics_list.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        })
    metrics_df = pd.DataFrame(metrics_list).set_index("Model")
    return models, metrics_df

# -------------------------------------------------------
# VISUALS
# -------------------------------------------------------
def plot_metrics_bar(metrics_df, title_suffix=""):
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_df[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
        kind="bar", ax=ax
    )
    ax.set_title(f"Model Performance {title_suffix}")
    ax.set_ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig

# -------------------------------------------------------
# APP
# -------------------------------------------------------
def main():
    st.title("Microplastic Risk Prediction – Streamlit App")
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", [
        "Raw Data",
        "Preprocessing",
        "Classification Modeling",
    ])
    uploaded_file = st.sidebar.file_uploader(
        "Upload Microplastic CSV",
        type=["csv"],
        help="If you don't upload anything, the app will try to use 'Microplastic.csv' from the app folder."
    )
    try:
        df_raw = load_data(uploaded_file=uploaded_file)
    except Exception as e:
        st.error(f"❌ Could not read data: {e}")
        st.stop()

    if df_raw is None or df_raw.empty:
        st.error("No dataset found. Please check file and format.")
        st.stop()

    if page == "Raw Data":
        st.subheader("Raw Data Preview (first 15 rows)")
        st.dataframe(df_raw.head(15))
        st.write(f"Shape: {df_raw.shape}")
        st.write("Columns:", list(df_raw.columns))

    elif page == "Preprocessing":
        st.subheader("Cleaned Numeric Columns (first 15 rows)")
        df_clean = clean_numeric_columns(df_raw)
        st.dataframe(df_clean.head(15))
        st.write(f"Numeric Feature Columns: {NUMERIC_MODEL_COLS}")

        st.write("Column value distributions:")
        for col in NUMERIC_MODEL_COLS:
            if col in df_clean.columns:
                st.write(f"**{col}**", df_clean[col].describe())

    elif page == "Classification Modeling":
        df, X, y_type, y_level = preprocess_for_model(df_raw)
        st.subheader("Classification Modeling")
        tab1, tab2 = st.tabs(["Risk-Type", "Risk-Level"])

        with tab1:
            if y_type is not None:
                st.info("Training models for Risk-Type...")
                models, metrics_rt = train_models(X, y_type)
                st.write(metrics_rt.style.format("{:.3f}"))
                st.pyplot(plot_metrics_bar(metrics_rt, "(Risk-Type)"))
            else:
                st.warning("Risk_Type column unavailable.")

        with tab2:
            if y_level is not None:
                st.info("Training models for Risk-Level...")
                models, metrics_rl = train_models(X, y_level)
                st.write(metrics_rl.style.format("{:.3f}"))
                st.pyplot(plot_metrics_bar(metrics_rl, "(Risk-Level)"))
            else:
                st.warning("Risk_Level column unavailable.")

if __name__ == "__main__":
    main()
