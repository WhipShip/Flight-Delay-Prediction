# -*- coding: utf-8 -*-
"""
Smartphone Price Prediction — Interactive Streamlit Dashboard
Implemented with rigorous 5-Fold Cross Validation and Complete Analytics
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import ast
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import itertools

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score, cross_validate,
    cross_val_predict
)
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    mean_squared_error, r2_score, confusion_matrix,
    ConfusionMatrixDisplay, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)

import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smartphone ML Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & CLEANING (Identical to final script)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean():
    df = pd.read_csv('Smartphone_Specifications_Dataset.csv')

    cols_to_drop = [
        'refresh_rate_hz', 'memory_card_max_gb', 'memory_card_type',
        'memory_card_supported', 'display_type', 'fast_charge_w',
        'sim_type', 'chipset', 'storage_gb', 'VoLTE'
    ]
    df_cleaned = df.drop(columns=cols_to_drop, errors='ignore')

    core_mapping = {r'(?i).*octa.*': 8, r'(?i).*hexa.*': 6, r'(?i).*quad.*': 4}
    df_cleaned['core_type'] = df_cleaned['core_type'].replace(core_mapping, regex=True)
    df_cleaned['core_type'] = pd.to_numeric(df_cleaned['core_type'], errors='coerce')

    res_split = df_cleaned['resolution'].str.split('x', expand=True)
    width = pd.to_numeric(res_split[0], errors='coerce')
    height = pd.to_numeric(res_split[1], errors='coerce')
    df_cleaned['total_pixels'] = width * height
    df_cleaned.drop(columns=['resolution'], inplace=True, errors='ignore')

    df_cleaned['network_type'] = df_cleaned['network_type'].str.rstrip('g')
    df_cleaned['network_type'] = pd.to_numeric(df_cleaned['network_type'])

    df_cleaned['total_rear_camera_mp'] = df_cleaned['rear_camera_mp_list'].apply(
        lambda x: sum(ast.literal_eval(x)) if pd.notna(x) else 0
    )
    df_cleaned.drop(columns=['rear_camera_mp_list'], inplace=True, errors='ignore')

    df_cleaned = df_cleaned[df_cleaned['price'] <= 400000]
    if 'ram_gb' in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned['ram_gb'] <= 32]

    # Imputations exactly as provided
    ghz_mapping = {'apple iphone 14 pro max': 3.46, 'samsung galaxy a34 5g': 2.60, 'apple iphone 14 pro': 3.46,
                   'tesla pi phone': 3.20, 'google pixel 6a': 2.80, 'google pixel 7a': 2.85,
                   'apple iphone 15 pro max': 3.78, 'samsung galaxy a54 5g': 2.40, 'vivo y02': 2.00,
                   'samsung galaxy a75 5g': 2.40, 'samsung galaxy s22 fe 5g': 2.80, 'apple iphone xr': 2.49,
                   'apple iphone 14 mini': 3.23, 'samsung galaxy s24 ultra': 3.39, 'letv y2 pro': 2.00,
                   'apple iphone 15 ultra': 3.78, 'nokia x50 5g': 2.40, 'apple iphone 15 pro': 3.78,
                   'samsung galaxy a15': 2.20, 'samsung galaxy f14': 2.40, 'oneplus nord 5': 2.80,
                   'apple iphone 15': 3.46, 'vivo s17 pro': 3.10, 'google pixel 8': 3.00,
                   'samsung galaxy s23 fe 5g': 2.80, 'samsung galaxy m51s 5g': 2.40, 'samsung galaxy m14': 2.40,
                   'vivo t2 pro 5g': 2.80, 'huawei nova y61': 2.20, 'iqoo z9': 2.80, 'apple iphone 15 plus': 3.46,
                   'samsung galaxy m35': 2.40, 'samsung galaxy a05': 2.00, 'google pixel 8 pro': 3.00,
                   'samsung galaxy m52s 5g': 2.40}
    df_cleaned['clock_ghz'] = df_cleaned['clock_ghz'].fillna(df_cleaned['model'].map(ghz_mapping))
    df_cleaned.dropna(subset=['clock_ghz'], inplace=True)

    os_mapping = {'oppo find n fold': 'Android v11', 'vivo x fold 5g': 'Android v12', 'oppo find n2 5g': 'Android v13',
                  'xiaomi mix fold 2 5g': 'Android v12', 'samsung galaxy z flip 3': 'Android v11',
                  'samsung galaxy z fold 4': 'Android v12', 'royole flexpai 2': 'Android v10',
                  'oppo find n flip': 'Android v13', 'oppo find n2 flip': 'Android v13',
                  'vivo x fold plus': 'Android v12', 'samsung galaxy z flip 4 5g': 'Android v12',
                  'samsung galaxy z fold 3': 'Android v11', 'lg wing 5g': 'Android v10', 'oukitel wp21': 'Android v12',
                  'lg v60 thinq': 'Android v10', 'oppo find n 5g': 'Android v11', 'oppo x': 'Android v11',
                  'asus rog phone 6d ultimate': 'Android v12', 'huawei mate xs 2': 'EMUI v12',
                  'xiaomi mi mix fold': 'Android v10', 'cat s22 flip': 'Android v11',
                  'royole flexpai 3 5g': 'Android v11', 'huawei mate x': 'Android v9', 'vivo x fold 2': 'Android v13'}
    df_cleaned['os'] = df_cleaned['os'].fillna(df_cleaned['model'].map(os_mapping))

    fc_mapping = {'oppo find n2 5g': 32.0, 'google pixel 7 pro 5g': 10.8, 'google pixel 7 5g': 10.8,
                  'google pixel 6 pro': 11.1, 'samsung galaxy z flip 3': 10.0, 'oppo find n flip': 32.0,
                  'oppo find n2 flip': 32.0, 'jio jiophone 2': 0.3, 'vertu signature touch': 2.1,
                  'xiaomi redmi mix alpha': 0.0, 'nokia 8000 4g': 0.0, 'samsung galaxy z flip 4 5g': 10.0,
                  'lg wing 5g': 32.0, 'oukitel wp21': 20.0, 'lg v60 thinq': 10.0, 'asus rog phone 6d ultimate': 12.0,
                  'huawei mate xs 2': 10.7, 'apple iphone se 4': 12.0, 'google pixel 8': 10.5, 'leitz phone 2': 12.6,
                  'cat s22 flip': 2.0, 'royole flexpai 3 5g': 32.0, 'huawei mate x': 0.0, 'itel a23s': 0.3}
    df_cleaned['front_camera_mp'] = df_cleaned['front_camera_mp'].fillna(df_cleaned['model'].map(fc_mapping))

    rc_max_mapping = {'oppo find n2 5g': 50.0, 'samsung galaxy z flip 3': 12.0, 'oppo find n flip': 50.0,
                      'oppo find n2 flip': 50.0, 'nokia 8000 4g': 2.0, 'samsung galaxy z flip 4 5g': 12.0,
                      'lg wing 5g': 64.0, 'oukitel wp21': 64.0, 'lg v60 thinq': 64.0,
                      'asus rog phone 6d ultimate': 50.0, 'huawei mate xs 2': 50.0, 'apple iphone se 4': 48.0,
                      'cat s22 flip': 5.0, 'royole flexpai 3 5g': 64.0, 'huawei mate x': 40.0, 'itel a23s': 2.0}
    df_cleaned['rear_camera_max_mp'] = df_cleaned['rear_camera_max_mp'].fillna(df_cleaned['model'].map(rc_max_mapping))

    rc_tot_mapping = {'oppo find n2 5g': 130.0, 'samsung galaxy z flip 3': 24.0, 'oppo find n flip': 58.0,
                      'oppo find n2 flip': 58.0, 'nokia 8000 4g': 2.0, 'samsung galaxy z flip 4 5g': 24.0,
                      'lg wing 5g': 89.0, 'oukitel wp21': 86.0, 'lg v60 thinq': 77.0,
                      'asus rog phone 6d ultimate': 68.0, 'huawei mate xs 2': 71.0, 'apple iphone se 4': 48.0,
                      'cat s22 flip': 5.0, 'royole flexpai 3 5g': 88.0, 'huawei mate x': 64.0, 'itel a23s': 2.0}
    df_cleaned['total_rear_camera_mp'] = df_cleaned['total_rear_camera_mp'].fillna(
        df_cleaned['model'].map(rc_tot_mapping))

    df_cleaned.dropna(subset=['core_type', 'battery_mah', 'total_pixels'], inplace=True)
    df_cleaned.drop(columns='model', inplace=True, errors='ignore')

    df_cleaned = pd.get_dummies(df_cleaned, columns=['os'], drop_first=True, dtype=int)
    bool_columns = ['NFC', 'ir_blaster']
    df_cleaned[bool_columns] = df_cleaned[bool_columns].astype(int)

    return df_cleaned, bool_columns


# ──────────────────────────────────────────────────────────────────────────────
# 2. CACHED CV FUNCTIONS FOR REGRESSION (Strictly folds, preserving all OOF)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def run_static_regression_models(df_cleaned, bool_columns):
    y_log = np.log1p(df_cleaned['price'])
    X = df_cleaned.drop(columns=['price'])

    cols_to_scale = [c for c in X.columns if c not in bool_columns and not c.startswith('os_')]
    kf = KFold(n_splits=5, shuffle=True, random_state=8)

    n_samples = len(X)
    oof_actuals = np.zeros(n_samples)
    oof_folds = np.zeros(n_samples)
    oof_linear = np.zeros(n_samples)
    oof_rf = np.zeros(n_samples)

    cv_lin_train = []
    cv_rf_train = []

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train_log, y_test_log = y_log.iloc[train_index], y_log.iloc[test_index]

        oof_actuals[test_index] = np.expm1(y_test_log)
        oof_folds[test_index] = fold

        scaler = StandardScaler()
        X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

        # Linear
        lr = LinearRegression().fit(X_train, y_train_log)
        oof_linear[test_index] = np.expm1(lr.predict(X_test))
        cv_lin_train.append(r2_score(np.expm1(y_train_log), np.expm1(lr.predict(X_train))))

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=8).fit(X_train, y_train_log)
        oof_rf[test_index] = np.expm1(rf.predict(X_test))
        cv_rf_train.append(r2_score(np.expm1(y_train_log), np.expm1(rf.predict(X_train))))

    lin_test_r2 = r2_score(oof_actuals, oof_linear)
    rf_test_r2 = r2_score(oof_actuals, oof_rf)

    return {
        'oof_actuals': oof_actuals, 'oof_folds': oof_folds,
        'oof_linear': oof_linear, 'oof_rf': oof_rf,
        'lin_train': np.mean(cv_lin_train), 'lin_test': lin_test_r2,
        'rf_train': np.mean(cv_rf_train), 'rf_test': rf_test_r2
    }


@st.cache_data
def run_ridge_cv(df_cleaned, bool_columns, alpha):
    y_log = np.log1p(df_cleaned['price'])
    X = df_cleaned.drop(columns=['price'])
    cols_to_scale = [c for c in X.columns if c not in bool_columns and not c.startswith('os_')]
    kf = KFold(n_splits=5, shuffle=True, random_state=8)

    oof_ridge = np.zeros(len(X))
    cv_train = []
    fold_scores = []

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train_log, y_test_log = y_log.iloc[train_index], y_log.iloc[test_index]

        scaler = StandardScaler()
        X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

        ridge = Ridge(alpha=alpha).fit(X_train, y_train_log)
        preds = np.expm1(ridge.predict(X_test))
        oof_ridge[test_index] = preds

        cv_train.append(r2_score(np.expm1(y_train_log), np.expm1(ridge.predict(X_train))))
        fold_scores.append(r2_score(np.expm1(y_test_log), preds))

    test_r2 = r2_score(np.expm1(y_log), oof_ridge)
    return np.mean(cv_train), test_r2, oof_ridge, fold_scores


@st.cache_data
def run_poly_cv(df_cleaned, bool_columns, degree):
    y_log = np.log1p(df_cleaned['price'])
    X = df_cleaned.drop(columns=['price'])
    cols_to_scale = [c for c in X.columns if c not in bool_columns and not c.startswith('os_')]
    kf = KFold(n_splits=5, shuffle=True, random_state=8)

    oof_poly = np.zeros(len(X))
    cv_train = []

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train_log, y_test_log = y_log.iloc[train_index], y_log.iloc[test_index]

        scaler = StandardScaler()
        X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

        poly = PolynomialFeatures(degree=degree)
        X_p_train = poly.fit_transform(X_train)
        X_p_test = poly.transform(X_test)

        m = LinearRegression().fit(X_p_train, y_train_log)

        train_pred = np.expm1(np.clip(m.predict(X_p_train), None, 21))
        test_pred = np.expm1(np.clip(m.predict(X_p_test), None, 21))

        oof_poly[test_index] = test_pred
        cv_train.append(r2_score(np.expm1(y_train_log), train_pred))

    test_r2 = r2_score(np.expm1(y_log), oof_poly)
    return np.mean(cv_train), test_r2, oof_poly


@st.cache_data
def run_poly_ridge_cv(df_cleaned, degree, alpha):
    y_log = np.log1p(df_cleaned['price'])
    X = df_cleaned.drop(columns=['price'])
    kf = KFold(n_splits=5, shuffle=True, random_state=8)

    oof_pr = np.zeros(len(X))
    cv_train = []

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train_log, y_test_log = y_log.iloc[train_index], y_log.iloc[test_index]

        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_p_train = poly.fit_transform(X_train)
        X_p_test = poly.transform(X_test)

        scaler = StandardScaler()
        X_p_train_s = scaler.fit_transform(X_p_train)
        X_p_test_s = scaler.transform(X_p_test)

        m = Ridge(alpha=alpha).fit(X_p_train_s, y_train_log)

        train_pred = np.expm1(np.clip(m.predict(X_p_train_s), None, 21))
        test_pred = np.expm1(np.clip(m.predict(X_p_test_s), None, 21))

        oof_pr[test_index] = test_pred
        cv_train.append(r2_score(np.expm1(y_train_log), train_pred))

    test_r2 = r2_score(np.expm1(y_log), oof_pr)
    return np.mean(cv_train), test_r2, oof_pr


# ──────────────────────────────────────────────────────────────────────────────
# 3. CACHED CLASSIFICATION FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def categorize_price(price):
    if price < 15000:
        return "Budget"
    elif price < 60000:
        return "Mid"
    else:
        return "Premium"


def categorize_rating(r):
    if r >= 83:
        return 'High'
    elif r >= 78:
        return 'Medium'
    else:
        return 'Low'


@st.cache_data
def run_classification_models(df_cleaned):
    # ── Price Classification ────────────────────────────────────────────────
    df_price = df_cleaned.copy()
    df_price['Price_Category'] = df_price['price'].apply(categorize_price)
    X_price = df_price.drop(columns=['price', 'rating', 'Price_Category', 'model', 'os'], errors='ignore').fillna(0)
    y_price = df_price['Price_Category']

    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_price, y_price, test_size=0.2, random_state=43,
                                                                stratify=y_price)

    # Weaker Logistic Baseline (Unscaled intentionally as per script to show failure)
    lr_pl = Pipeline([
        ('logistic', LogisticRegression(max_iter=100, solver='lbfgs', C=0.1, random_state=42))
    ])
    kf_class = KFold(n_splits=5, shuffle=True, random_state=43)
    lr_cv_preds = cross_val_predict(lr_pl, X_price, y_price, cv=kf_class)
    lr_acc = accuracy_score(y_price, lr_cv_preds)

    # RF Feature Selection & Training
    selector = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_p, y_train_p)
    top_features = pd.Series(selector.feature_importances_, index=X_price.columns).sort_values(ascending=False).head(
        10).index.tolist()

    X_train_p_top = X_train_p[top_features]
    X_test_p_top = X_test_p[top_features]
    rfc = RandomForestClassifier(n_estimators=200, random_state=43).fit(X_train_p_top, y_train_p)
    rfc_preds = rfc.predict(X_test_p_top)
    rfc_acc = accuracy_score(y_test_p, rfc_preds)

    # Naive Bayes Price
    scaler_nb = StandardScaler()
    X_train_s_nb = scaler_nb.fit_transform(X_train_p)
    X_test_s_nb = scaler_nb.transform(X_test_p)
    nb_model_p = GaussianNB().fit(X_train_s_nb, y_train_p)
    nb_preds_p = nb_model_p.predict(X_test_s_nb)
    nb_acc_p = accuracy_score(y_test_p, nb_preds_p)

    # ── Rating Classification ───────────────────────────────────────────────
    df_rating = df_cleaned.copy()
    df_rating['Rating_Category'] = df_rating['rating'].apply(categorize_rating)

    features_knn = ['price', 'ram_gb', 'battery_mah', 'screen_size_in', 'rear_camera_count',
                    'rear_camera_max_mp', 'front_camera_mp', 'clock_ghz', 'total_pixels',
                    'total_rear_camera_mp', 'NFC', 'ir_blaster', 'network_type'] + [c for c in df_cleaned.columns if
                                                                                    c.startswith('os_')]

    X_rating = df_rating[features_knn]
    y_rating = df_rating['Rating_Category']

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_rating, y_rating, test_size=0.2, random_state=42,
                                                                stratify=y_rating)

    scaler_r = StandardScaler()
    X_train_r_s = scaler_r.fit_transform(X_train_r)
    X_test_r_s = scaler_r.transform(X_test_r)

    # NB Rating
    nb_model_r = GaussianNB().fit(X_train_r_s, y_train_r)
    nb_preds_r = nb_model_r.predict(X_test_r_s)
    nb_acc_r = accuracy_score(y_test_r, nb_preds_r)

    return {
        'X_price': X_price, 'y_price': y_price, 'top_features': top_features,
        'y_test_p': y_test_p, 'rfc_preds': rfc_preds, 'nb_preds_p': nb_preds_p, 'lr_cv_preds': lr_cv_preds,
        'lr_acc': lr_acc, 'rfc_acc': rfc_acc, 'nb_acc_p': nb_acc_p,
        'X_rating': X_rating, 'y_rating': y_rating, 'y_test_r': y_test_r,
        'X_train_r_s': X_train_r_s, 'X_test_r_s': X_test_r_s, 'y_train_r': y_train_r,
        'nb_preds_r': nb_preds_r, 'nb_acc_r': nb_acc_r, 'scaler_r': scaler_r, 'features_knn': features_knn
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ──────────────────────────────────────────────────────────────────────────────
try:
    df_cleaned, bool_columns = load_and_clean()
except FileNotFoundError:
    st.error("Dataset not found. Place `Smartphone_Specifications_Dataset.csv` in the same directory.")
    st.stop()

st.sidebar.title("📱 ML Dashboard")
page = st.sidebar.radio("Navigate", ["📊 EDA", "📈 Regression Models", "🏷️ Classification Models"])

# ==============================================================================
# PAGE 1 — EDA
# ==============================================================================
if page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis & Feature Insights")

    # 1. Price Skewness Transformation
    st.subheader("Why We Transform Data: Visualizing the Target Log Transformation")
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df_cleaned['price'], bins=40, kde=True, color='crimson', ax=axes1[0])
    axes1[0].set_title('1A. Raw Price Distribution\n(Notice the heavy Right-Skew)')
    axes1[0].set_xlabel('Actual Price (₹)')

    sns.histplot(np.log1p(df_cleaned['price']), bins=40, kde=True, color='seagreen', ax=axes1[1])
    axes1[1].set_title('1B. Log-Transformed Price\n(Closer to Normal Distribution)')
    axes1[1].set_xlabel('Log(Price + 1)')
    col_f1_left, col_f1_center, col_f1_right = st.columns([1, 3, 1])
    with col_f1_center:
        st.pyplot(fig1)
    plt.close(fig1)
    st.divider()

    # 2. Box Plot
    st.subheader("Price Outlier Visualization")
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.boxplot(df_cleaned['price'], vert=False, patch_artist=True, boxprops=dict(facecolor='#2563EB', alpha=0.6))
    ax2.set_xlabel("Price (₹)")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x / 1000:.0f}k"))
    ax2.grid(True, linestyle='--', alpha=0.5)
    col_f2_left, col_f2_center, col_f2_right = st.columns([1, 3, 1])
    with col_f2_center:
        st.pyplot(fig2)
    plt.close(fig2)
    st.divider()

    # 3. Correlation Heatmap
    st.subheader("Smartphone Feature Correlation Heatmap")
    numerical_df = df_cleaned.iloc[:, :15]
    corr_matrix = numerical_df.corr()
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt='.2f', square=True, linewidths=0.5, ax=ax3)
    col_f3_left, col_f3_center, col_f3_right = st.columns([1, 3, 1])
    with col_f3_center:
        st.pyplot(fig3)
    plt.close(fig3)
    st.divider()

    # 4. Feature Space Comparison (Linear vs RF) - INTERACTIVE DROPDOWN
    st.subheader("Interactive Model Feature Comparison vs Price")

    # Get all numerical features (exclude 'price' and binary variables like NFC/OS)
    valid_features = [col for col in df_cleaned.columns
                      if col != 'price'
                      and df_cleaned[col].nunique() > 2
                      and not col.startswith('os_')]

    # Streamlit Dropdown
    selected_feature = st.selectbox(
        "Select a feature to visualize its relationship with Price:",
        options=valid_features,
        index=valid_features.index('total_pixels') if 'total_pixels' in valid_features else 0
    )

    # Set up single plot
    fig4, ax4 = plt.subplots(figsize=(8, 4))

    # Isolate data
    X_single = df_cleaned[[selected_feature]]
    y_val = df_cleaned['price']
    X_train_ft, X_test_ft, y_train_ft, y_test_ft = train_test_split(X_single, y_val, test_size=0.2, random_state=43)

    # Train models on single feature
    lr = LinearRegression().fit(X_train_ft, y_train_ft)
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=43).fit(X_train_ft, y_train_ft)

    # Generate line data
    x_range = pd.DataFrame(np.linspace(df_cleaned[selected_feature].min(), df_cleaned[selected_feature].max(), 500),
                           columns=[selected_feature])

    # Plot Scatter and Lines
    sns.scatterplot(data=df_cleaned, x=selected_feature, y='price', alpha=0.7, color='gray', ax=ax4)
    ax4.plot(x_range, lr.predict(x_range), 'r--', lw=2,
             label=f'Linear (Test R²: {r2_score(y_test_ft, lr.predict(X_test_ft)):.3f})')
    ax4.plot(x_range, rf.predict(x_range), color='#2b5b84', lw=3,
             label=f'RF (Test R²: {r2_score(y_test_ft, rf.predict(X_test_ft)):.3f})')

    # Formatting
    feature_name_clean = selected_feature.replace("_", " ").title()
    ax4.set_title(f'{feature_name_clean} vs Price', fontsize=14, pad=10)
    ax4.set_xlabel(feature_name_clean, fontsize=12)
    ax4.set_ylabel('Price (₹)', fontsize=12)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x / 1000:.0f}k"))
    ax4.legend(fontsize=10)
    ax4.grid(True, ls='--', alpha=0.5)

    col_f4_left, col_f4_center, col_f4_right = st.columns([1, 3, 1])
    with col_f4_center:
        st.pyplot(fig4)
    plt.close(fig4)

# ==============================================================================
# PAGE 2 — REGRESSION
# ==============================================================================
elif page == "📈 Regression Models":
    st.title("📈 5-Fold Cross Validated Regression")

    with st.spinner("Aggregating 100% of data across 5 folds..."):
        reg_static = run_static_regression_models(df_cleaned, bool_columns)

    oof_actuals = reg_static['oof_actuals']
    oof_folds = reg_static['oof_folds']
    mn, mx = oof_actuals.min(), oof_actuals.max()

    # 1. Interactive Ridge Analysis
    st.subheader("Ridge CV Stability (Alphas: 0.01 to 1000)")
    alphas = [0.01, 0.1, 10, 100, 1000]

    # Precompute all alphas for graph
    alpha_perfs = {}
    for a in alphas:
        _, t_r2, _, folds = run_ridge_cv(df_cleaned, bool_columns, a)
        alpha_perfs[a] = {'test_r2': t_r2, 'folds': folds}
    best_a = max(alpha_perfs, key=lambda k: alpha_perfs[k]['test_r2'])

    fig_ridge, ax_ridge = plt.subplots(figsize=(8, 4))
    folds_labels = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    colors = cm.viridis(np.linspace(0, 0.9, len(alphas)))
    for i, a in enumerate(alphas):
        cum_avg = np.cumsum(alpha_perfs[a]['folds']) / np.arange(1, 6)
        is_win = (a == best_a)
        lw = 3.5 if is_win else 1.5
        ax_ridge.plot(folds_labels, cum_avg, marker='o', lw=lw, alpha=(1.0 if is_win else 0.5), color=colors[i],
                      label=f"Alpha {a} {'(WINNER)' if is_win else ''}")
    ax_ridge.set_title("Model Stability: Cumulative Average Test R² Across Folds")
    ax_ridge.grid(True, ls=':', alpha=0.6)
    ax_ridge.legend(loc='lower right')
    col_fr_left, col_fr_center, col_fr_right = st.columns([1, 3, 1])
    with col_fr_center:
        st.pyplot(fig_ridge)
    plt.close(fig_ridge)

    # Sliders for Poly & Poly-Ridge
    col_sl1, col_sl2, col_sl3 = st.columns(3)
    with col_sl1:
        poly_d = st.slider("Polynomial Degree", 1, 3, 2)
    with col_sl2:
        pr_deg = st.slider("Poly-Ridge Degree", 1, 3, 2)
    with col_sl3:
        pr_alpha = st.select_slider("Poly-Ridge Alpha", options=alphas, value=100)

    # Run dynamic models
    r_tr, r_te, oof_ridge, _ = run_ridge_cv(df_cleaned, bool_columns, best_a)
    p_tr, p_te, oof_poly = run_poly_cv(df_cleaned, bool_columns, poly_d)
    pr_tr, pr_te, oof_pr = run_poly_ridge_cv(df_cleaned, pr_deg, pr_alpha)

    # Metrics Summary
    st.markdown("### Test R² Score Comparison")
    m_cols = st.columns(5)
    m_cols[0].metric("Linear", f"{reg_static['lin_test']:.4f}")
    m_cols[1].metric(f"Ridge (α={best_a})", f"{r_te:.4f}")
    m_cols[2].metric(f"Poly (d={poly_d})", f"{p_te:.4f}")
    m_cols[3].metric(f"Poly-Ridge", f"{pr_te:.4f}")
    m_cols[4].metric("Random Forest", f"{reg_static['rf_test']:.4f}")

    # Bar chart (Train vs Test) exactly as requested
    st.subheader("Model Performance Comparison: Train vs. Test R² Scores")
    models = ['Linear', 'Ridge', 'Poly', 'Poly-Ridge', 'Random Forest']
    tr_scores = [reg_static['lin_train'], r_tr, p_tr, pr_tr, reg_static['rf_train']]
    te_scores = [reg_static['lin_test'], r_te, np.clip(p_te, -0.1, 1.0), pr_te, reg_static['rf_test']]

    fig_bar, ax_bar = plt.subplots(figsize=(9, 4))
    x = np.arange(len(models))
    w = 0.35
    r1 = ax_bar.bar(x - w / 2, tr_scores, w, label='Train R²', color='#2563EB')
    r2 = ax_bar.bar(x + w / 2, te_scores, w, label='Test R²', color='#DC2626')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(models)
    ax_bar.set_ylim(-0.2, 1.1)
    ax_bar.axhline(0, color='black')
    ax_bar.legend()

    for i, rect in enumerate(r2):
        if models[i] == 'Poly' and p_te < 0:
            ax_bar.annotate('FAIL', (rect.get_x() + rect.get_width() / 2, 0), xytext=(0, -20),
                            textcoords="offset points", ha='center', color='red', weight='bold')
        else:
            ax_bar.annotate(f'{rect.get_height():.2f}', (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha='center')
    for rect in r1:
        ax_bar.annotate(f'{rect.get_height():.2f}', (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center')
    col_fb_left, col_fb_center, col_fb_right = st.columns([1, 3, 1])
    with col_fb_center:
        st.pyplot(fig_bar)
    plt.close(fig_bar)
    st.divider()

    # Best vs Worst Folds (2x2 Grid)
    st.subheader("Actual vs. Predicted Price: Best vs. Worst Folds (Out-of-Fold)")
    fig_grid, axes_grid = plt.subplots(2, 2, figsize=(10, 7))
    axes_grid = axes_grid.flatten()
    preds_list = [reg_static['oof_linear'], oof_ridge, oof_pr, reg_static['oof_rf']]
    titles = ["Linear Regression", "Ridge Regression", "Poly-Ridge Regression", "Random Forest"]

    for i, (ax, preds, title) in enumerate(zip(axes_grid, preds_list, titles)):
        fold_r2 = {f: r2_score(oof_actuals[oof_folds == f], preds[oof_folds == f]) for f in range(1, 6)}
        bf = max(fold_r2, key=fold_r2.get)
        wf = min(fold_r2, key=fold_r2.get)

        b_idx, w_idx = (oof_folds == bf), (oof_folds == wf)
        o_idx = ~(b_idx | w_idx)

        ax.scatter(oof_actuals[o_idx], preds[o_idx], color='lightgray', alpha=0.4, s=15, label="Other Folds")
        ax.scatter(oof_actuals[w_idx], preds[w_idx], color='crimson', alpha=0.85, s=35,
                   label=f"Worst Fold ({wf}) R²: {fold_r2[wf]:.3f}")
        ax.scatter(oof_actuals[b_idx], preds[b_idx], color='seagreen', alpha=0.85, s=35,
                   label=f"Best Fold ({bf}) R²: {fold_r2[bf]:.3f}")

        ax.plot([mn, mx], [mn, mx], "k--", lw=1.5)
        ax.plot([mn, mx], [mn * 0.8, mx * 0.8], "grey", ls=":")
        ax.plot([mn, mx], [mn * 1.2, mx * 1.2], "grey", ls=":")
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Actual Price (₹)")
        ax.set_ylabel("Predicted Price (₹)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x / 1000:.0f}k"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x / 1000:.0f}k"))
        ax.set_xlim([mn, mx * 1.05]);
        ax.set_ylim([mn, mx * 1.15])
        ax.legend(loc="upper left")
    plt.tight_layout()
    # Squeeze the plot into the center by adding empty spaces on the left and right
    # [1, 3, 1] means: 20% empty space, 60% plot, 20% empty space.
    left_spacer, center_col, right_spacer = st.columns([1, 3, 1])

    with center_col:
        st.pyplot(fig_grid)
    plt.close(fig_grid)

    # ==============================================================================
    # REGRESSION INFERENCE (PREDICT EXACT PRICE)
    # ==============================================================================
    st.divider()
    st.header("🔮 Predict Exact Price for a New Phone")
    st.caption("Enter the specifications below to predict the exact price using the Random Forest model.")

    # Separate OS features from standard features
    X_reg = df_cleaned.drop(columns=['price'])
    os_features = [c for c in X_reg.columns if c.startswith('os_')]
    non_os_features = [c for c in X_reg.columns if not c.startswith('os_')]

    # Create clean OS names for the dropdown
    os_options = ["Other / Default OS"] + [c.replace('os_', '') for c in os_features]

    with st.form("reg_inference"):
        cols_reg = st.columns(3)
        reg_inputs = {}

        # 1. Ask for standard features
        for idx, feat in enumerate(non_os_features):
            with cols_reg[idx % 3]:
                if feat in ['NFC', 'ir_blaster']:
                    reg_inputs[feat] = st.selectbox(feat, [0, 1], key=f"reg_inf_{feat}")
                else:
                    reg_inputs[feat] = st.number_input(feat, value=float(X_reg[feat].median()), format="%.2f",
                                                       key=f"reg_inf_{feat}")

        # 2. Add the single OS Dropdown
        with cols_reg[len(non_os_features) % 3]:
            selected_os = st.selectbox("Operating System (OS)", options=os_options, key="reg_inf_os")

        submit_reg = st.form_submit_button("Predict Exact Price")

    if submit_reg:
        # Automatically distribute the 1s and 0s based on the single OS selection
        for feat in os_features:
            reg_inputs[feat] = 1 if selected_os != "Other / Default OS" and feat == f"os_{selected_os}" else 0

        with st.spinner("Calculating price..."):
            y_log_master = np.log1p(df_cleaned['price'])
            rf_master = RandomForestRegressor(n_estimators=100, random_state=8).fit(X_reg, y_log_master)

            # Create Dataframe and enforce exact column order
            inp_df = pd.DataFrame([reg_inputs])[X_reg.columns]
            pred_log = rf_master.predict(inp_df)[0]
            pred_price = np.expm1(pred_log)

            st.success(f"💰 **Predicted Exact Price: ₹{pred_price:,.0f}**")

# ==============================================================================
# PAGE 3 — CLASSIFICATION
# ==============================================================================
elif page == "🏷️ Classification Models":
    st.title("🏷️ Classification Models")

    with st.spinner("Training classification models..."):
        c_data = run_classification_models(df_cleaned)

    # 1. Price Category --------------------------------------------------------
    st.header("1. Pricing Categories (Budget < 15k, Mid < 60k, Premium)")

    st.subheader("Why Logistic Regression Fails (Feature Overlap)")
    fig_f, ax_f = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x=c_data['X_price'][c_data['top_features'][0]], y=c_data['X_price'][c_data['top_features'][1]],
                    hue=c_data['y_price'], palette="Set1", alpha=0.6, s=80, ax=ax_f)
    ax_f.set_title(f"Overlapping Feature Space: {c_data['top_features'][0]} vs {c_data['top_features'][1]}")
    col_ff_left, col_ff_center, col_ff_right = st.columns([1, 3, 1])
    with col_ff_center:
        st.pyplot(fig_f)
    plt.close(fig_f)

    # LogReg Failure Matrix (normalize='true')
    col1, col2 = st.columns(2)
    with col1:
        st.write("Logistic Regression Errors (Normalized)")
        fig_lr, ax_lr = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(c_data['y_price'], c_data['lr_cv_preds'],
                                                display_labels=['Budget', 'Mid', 'Premium'], cmap='Reds',
                                                normalize='true', ax=ax_lr)
        st.pyplot(fig_lr)
        plt.close(fig_lr)

    with col2:
        st.write("Random Forest Success")
        fig_rf, ax_rf = plt.subplots(figsize=(6, 5))
        cm_rf = confusion_matrix(c_data['y_test_p'], c_data['rfc_preds'], labels=['Budget', 'Mid', 'Premium'])
        sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", xticklabels=['Budget', 'Mid', 'Premium'],
                    yticklabels=['Budget', 'Mid', 'Premium'], ax=ax_rf)
        st.pyplot(fig_rf)
        plt.close(fig_rf)

    # Pricing Metrics Chart
    st.subheader("Pricing Models Performance Comparison")
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    lr_scores = [c_data['lr_acc'],
                 precision_score(c_data['y_price'], c_data['lr_cv_preds'], average='weighted', zero_division=0),
                 recall_score(c_data['y_price'], c_data['lr_cv_preds'], average='weighted', zero_division=0),
                 f1_score(c_data['y_price'], c_data['lr_cv_preds'], average='weighted', zero_division=0)]
    nb_scores = [c_data['nb_acc_p'],
                 precision_score(c_data['y_test_p'], c_data['nb_preds_p'], average='weighted', zero_division=0),
                 recall_score(c_data['y_test_p'], c_data['nb_preds_p'], average='weighted', zero_division=0),
                 f1_score(c_data['y_test_p'], c_data['nb_preds_p'], average='weighted', zero_division=0)]
    rfc_scores = [c_data['rfc_acc'],
                  precision_score(c_data['y_test_p'], c_data['rfc_preds'], average='weighted', zero_division=0),
                  recall_score(c_data['y_test_p'], c_data['rfc_preds'], average='weighted', zero_division=0),
                  f1_score(c_data['y_test_p'], c_data['rfc_preds'], average='weighted', zero_division=0)]

    fig_pm, ax_pm = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics))
    w = 0.25
    ax_pm.bar(x - w, lr_scores, w, label='Logistic Baseline', color='#e74c3c', edgecolor='black')
    ax_pm.bar(x, nb_scores, w, label='Naive Bayes', color='#f1c40f', edgecolor='black')
    ax_pm.bar(x + w, rfc_scores, w, label='Random Forest', color='#2ecc71', edgecolor='black')
    ax_pm.set_xticks(x);
    ax_pm.set_xticklabels(metrics)
    ax_pm.legend(loc='lower right')
    ax_pm.set_ylim(0, 1.1)
    col_fpm_left, col_fpm_center, col_fpm_right = st.columns([1, 3, 1])
    with col_fpm_center:
        st.pyplot(fig_pm)
    plt.close(fig_pm)
    st.divider()

    # 2. Rating Category -------------------------------------------------------
    st.header("2. Rating Categories (Low, Medium ≥ 78, High ≥ 83)")

    # ─── INTERACTIVE KNN ──────────────────────────────────────────────────────
    st.subheader("Interactive KNN")
    st.caption("Adjust K to see how it affects accuracy.")
    k_val = st.slider("Select K for KNN:", 1, 15, 3, 2)

    knn = KNeighborsClassifier(n_neighbors=k_val).fit(c_data['X_train_r_s'], c_data['y_train_r'])
    knn_preds = knn.predict(c_data['X_test_r_s'])

    st.metric(f"KNN Accuracy (K={k_val})", f"{accuracy_score(c_data['y_test_r'], knn_preds):.4f}")

    fig_kc, ax_kc = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(c_data['y_test_r'], knn_preds, labels=['Low', 'Medium', 'High'], ax=ax_kc,
                                            cmap="Blues")
    plt.tight_layout()

    # Center the KNN plot
    col_kc_left, col_kc_center, col_kc_right = st.columns([1, 3, 1])
    with col_kc_center:
        st.pyplot(fig_kc)
    plt.close(fig_kc)

    st.divider()

    # ==============================================================================
    # 3. NAIVE BAYES BASELINES (SIDE-BY-SIDE)
    # ==============================================================================
    st.header("3. Naive Bayes Baselines")
    st.caption("Comparison of the Naive Bayes model's performance on both classification tasks.")

    col_nb1, col_nb2 = st.columns(2)

    # --- Naive Bayes: PRICE ---
    with col_nb1:
        st.subheader("Price Bracket Prediction")
        st.metric("NB Price Accuracy", f"{c_data['nb_acc_p']:.4f}")

        fig_nb_p, ax_nb_p = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            c_data['y_test_p'],
            c_data['nb_preds_p'],
            labels=['Budget', 'Mid', 'Premium'],
            ax=ax_nb_p,
            cmap="Oranges"
        )
        plt.tight_layout()
        st.pyplot(fig_nb_p)
        plt.close(fig_nb_p)

    # --- Naive Bayes: RATING ---
    with col_nb2:
        st.subheader("Rating Bracket Prediction")
        st.metric("NB Rating Accuracy", f"{c_data['nb_acc_r']:.4f}")

        fig_nb_r, ax_nb_r = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            c_data['y_test_r'],
            c_data['nb_preds_r'],
            labels=['Low', 'Medium', 'High'],
            ax=ax_nb_r,
            cmap="YlGnBu"
        )
        plt.tight_layout()
        st.pyplot(fig_nb_r)
        plt.close(fig_nb_r)

    st.divider()

    # ==============================================================================
    # 4. DYNAMIC INFERENCE FORMS (SPLIT)
    # ==============================================================================
    st.header("4. 🔮 Classification Inference Dashboard")

    col_inf1, col_inf2 = st.columns(2)

    # ─── FORM A: PREDICT PRICE BRACKET ─────────────────────────────────────────
    with col_inf1:
        st.subheader("Predict Price Bracket")
        st.caption("Predicts Budget (<15k), Mid (<60k), or Premium. (Price is hidden)")

        # Detect OS features required for Price model
        pc_os_features = [c for c in c_data['top_features'] if c.startswith('os_')]
        pc_non_os_features = [c for c in c_data['top_features'] if not c.startswith('os_')]
        pc_os_options = ["Other / Default OS"] + [c.replace('os_', '') for c in pc_os_features]

        with st.form("price_bracket_inference"):
            pc_inputs = {}
            for feat in pc_non_os_features:
                if feat in ['NFC', 'ir_blaster']:
                    pc_inputs[feat] = st.selectbox(feat, [0, 1], key=f"pc_inf_{feat}")
                else:
                    pc_inputs[feat] = st.number_input(feat, value=float(c_data['X_price'][feat].median()),
                                                      format="%.2f", key=f"pc_inf_{feat}")

            # Single OS Dropdown (if OS features made the top 10)
            if pc_os_features:
                pc_selected_os = st.selectbox("Operating System (OS)", options=pc_os_options, key="pc_inf_os")
            else:
                pc_selected_os = "Other / Default OS"

            submit_pc = st.form_submit_button("Predict Price Bracket")

        if submit_pc:
            for feat in pc_os_features:
                pc_inputs[feat] = 1 if pc_selected_os != "Other / Default OS" and feat == f"os_{pc_selected_os}" else 0

            inp_pc_df = pd.DataFrame([pc_inputs])[c_data['top_features']]
            rf_full = RandomForestClassifier(n_estimators=200, random_state=43).fit(
                c_data['X_price'][c_data['top_features']], c_data['y_price'])
            pred_prc = rf_full.predict(inp_pc_df)[0]
            st.success(f"💰 **Predicted Price Bracket:** {pred_prc}")

    # ─── FORM B: PREDICT RATING BRACKET ────────────────────────────────────────
    with col_inf2:
        st.subheader("Predict Rating Bracket")
        st.caption("Predicts Low, Medium, or High Rating. (Rating is hidden)")

        # Detect OS features required for Rating model
        rc_os_features = [c for c in c_data['features_knn'] if c.startswith('os_')]
        rc_non_os_features = [c for c in c_data['features_knn'] if not c.startswith('os_')]
        rc_os_options = ["Other / Default OS"] + [c.replace('os_', '') for c in rc_os_features]

        with st.form("rating_bracket_inference"):
            rc_inputs = {}
            for feat in rc_non_os_features:
                if feat in ['NFC', 'ir_blaster']:
                    rc_inputs[feat] = st.selectbox(feat, [0, 1], key=f"rc_inf_{feat}")
                else:
                    rc_inputs[feat] = st.number_input(feat, value=float(c_data['X_rating'][feat].median()),
                                                      format="%.2f", key=f"rc_inf_{feat}")

            # Single OS Dropdown
            if rc_os_features:
                rc_selected_os = st.selectbox("Operating System (OS)", options=rc_os_options, key="rc_inf_os")
            else:
                rc_selected_os = "Other / Default OS"

            submit_rc = st.form_submit_button("Predict Rating Bracket")

        if submit_rc:
            for feat in rc_os_features:
                rc_inputs[feat] = 1 if rc_selected_os != "Other / Default OS" and feat == f"os_{rc_selected_os}" else 0

            inp_rc_df = pd.DataFrame([rc_inputs])[c_data['features_knn']]

            scaler_full = StandardScaler()
            X_rating_s_full = scaler_full.fit_transform(c_data['X_rating'])
            knn_full = KNeighborsClassifier(n_neighbors=k_val).fit(X_rating_s_full, c_data['y_rating'])

            inp_rc_s = scaler_full.transform(inp_rc_df)
            pred_rat = knn_full.predict(inp_rc_s)[0]
            st.success(f"⭐ **Predicted Rating Profile:** {pred_rat}")