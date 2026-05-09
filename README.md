# 📱 Smartphone Price Prediction

> Final project for the **Data Science and Methodology Course** at the **Egyptian University of Informatics**

This project applies supervised machine learning — both regression and classification — to a dataset of real-world smartphone specifications in order to predict device prices and categorize phones into meaningful market tiers.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
  - [1. Data Cleaning](#1-data-cleaning)
  - [2. Scaling & Transformation](#2-scaling--transformation)
  - [3. Regression Models](#3-regression-models)
  - [4. Classification Models](#4-classification-models)
  - [5. Visualization](#5-visualization)
- [Results Summary](#results-summary)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)

---

## Overview

Smartphone pricing is driven by a complex mix of hardware specifications, brand positioning, and feature trade-offs. This project tackles that complexity through two parallel ML tracks:

- **Regression** — Predict a smartphone's exact price (₹) from its specifications.
- **Classification** — Assign each device to a market tier: **Budget**, **Mid-Range**, **Premium**, or **Flagship**.

Both tracks use 5-fold cross-validation to ensure robust, generalizable performance estimates.

---

## Dataset

**Source:** `Smartphone_Specifications_Dataset.csv`

**Key features used:**

| Feature | Description |
|---|---|
| `ram_gb` | RAM in gigabytes |
| `battery_mah` | Battery capacity |
| `clock_ghz` | Processor clock speed (GHz) |
| `core_type` | Number of CPU cores (4, 6, or 8) |
| `total_pixels` | Display resolution (width × height) |
| `network_type` | Network generation (4 or 5) |
| `front_camera_mp` | Front camera megapixels |
| `rear_camera_max_mp` | Highest rear camera megapixels |
| `total_rear_camera_mp` | Sum of all rear camera MPs |
| `NFC` | NFC support (binary) |
| `ir_blaster` | IR blaster support (binary) |
| `os_*` | One-hot encoded OS dummies |

**Target variable:** `price` (Indian Rupees ₹)

---

## Project Pipeline

### 1. Data Cleaning

Raw data required extensive preprocessing before modelling:

- **Dropped high-missingness columns** — `refresh_rate_hz`, `memory_card_max_gb`, `memory_card_type`, `display_type`, `fast_charge_w`, `sim_type`, `chipset`, `storage_gb`, `VoLTE`
- **Core type normalization** — Converted text labels (`Octa`, `Hexa`, `Quad`) to integers (8, 6, 4) via regex
- **Resolution parsing** — Split `resolution` string (e.g. `1080x2400`) into `total_pixels` (width × height), then dropped the original column
- **Camera MP aggregation** — Parsed JSON-like `rear_camera_mp_list` strings and summed values into `total_rear_camera_mp`
- **Missing value imputation** — Used a manually curated dictionary (sourced with the aid of Gemini) to fill missing `clock_ghz`, `os`, `front_camera_mp`, `rear_camera_max_mp`, and `total_rear_camera_mp` for specific named models
- **Outlier removal** — Capped prices at ₹400,000 and RAM at 32 GB to remove unrealistic entries
- **Encoding** — One-hot encoded `os`, converted `NFC` and `ir_blaster` booleans to integers (0/1)

---

### 2. Scaling & Transformation

- **Log-transformation** applied to the price target (`np.log1p`) to correct heavy right-skew before any model training
- **StandardScaler** applied to continuous features only (boolean and dummy columns left untouched)
- Scaling was performed **inside each cross-validation fold** (fit on train, transform on test) to prevent data leakage
- An 80/20 train-test split (`random_state=8`) was used for final evaluation

---

### 3. Regression Models

All regression models predict **log-price** during training, then predictions are back-transformed via `np.expm1` for evaluation in original rupee scale. Performance is measured using **R²**.

| Model | Description |
|---|---|
| **Linear Regression** | Baseline OLS regression; 5-fold CV benchmark |
| **Ridge Regression** | L2-regularized linear model; alpha tuned across a search grid |
| **Polynomial Regression** | Feature expansion to degree *d*; best degree selected by CV |
| **Poly-Ridge Regression** | Polynomial features + Ridge regularization; joint (degree, alpha) grid search |
| **Random Forest Regression** | Ensemble of decision trees; captures non-linear feature interactions |

Model selection criterion: **highest average Test R² across 5 folds**.

---

### 4. Classification Models

Phones are assigned to price tiers using the following thresholds (₹):

| Category | Price Range |
|---|---|
| 🟢 Budget | < ₹15,000 |
| 🔵 Mid-Range | ₹15,000 – ₹30,000 |
| 🟠 Premium | ₹30,000 – ₹60,000 |
| 🔴 Flagship | > ₹60,000 |

**Models evaluated:**

| Model | Notes |
|---|---|
| **Logistic Regression** | Baseline; demonstrates limitations with complex feature spaces |
| **K-Nearest Neighbors (KNN)** | K swept across values; K=3 yielded best accuracy |
| **Random Forest Classifier — Model 1** | Classifies by rating category |
| **Random Forest Classifier — Model 2** | Classifies by price tier using top-10 features selected by feature importance |
| **Gaussian Naive Bayes** | Probabilistic baseline across all four tiers |

Random Forest Classification (Model 2) uses a two-stage approach: a temporary forest first ranks feature importance, and only the top 10 features are passed to the final model.

---

### 5. Visualization

Key visualizations produced throughout the project:

- **Correlation Heatmap** — Pearson correlation matrix for all numerical features
- **Price Distribution** — Before/after log-transformation comparison (raw skew vs. normalized)
- **Ridge Alpha Stability Plot** — Cumulative average R² per fold for each alpha candidate
- **Regression Model Comparison** — Side-by-side bar charts of Train vs. Test R² for all regression models
- **KNN K-vs-Accuracy Curve** — Elbow plot to select optimal K
- **Confusion Matrices** — For Logistic Regression and KNN classifiers
- **Classification Report Heatmaps** — Precision, recall, F1 per class for Random Forest and Naive Bayes
- **Feature Scatter Plots** — Individual feature relationships with price overlaid with regression curves

---

## Results Summary

### Regression

| Model | Avg Train R² | Avg Test R² |
|---|---|---|
| Linear Regression | 0.78 | 0.69 |
| Ridge Regression | 0.78 | 0.70 |
| Polynomial Regression | 0.94 | -inf |
| Poly-Ridge Regression | 0.84 | 0.71 |
| **Random Forest** | 0.96 | 0.83 |

### Classification

| Model | Accuracy |
|---|---|
| Logistic Regression | 72.37% |
| KNN (K=3) | — |
| Random Forest | 85.26% |
| Gaussian Naive Bayes | — |

---

## Tech Stack

- **Language:** Python 3
- **Environment:** Google Colab / Local IDE
- **Core Libraries:**
  - `pandas`, `numpy` — data manipulation
  - `scikit-learn` — modelling, pipelines, cross-validation
  - `matplotlib`, `seaborn` — visualization
  - `streamlit` — interactive web dashboard UI

---

## Getting Started

First, clone the repository to your local machine:
```bash
git clone https://github.com/WhipShip/Smartphone-Price-Prediction.git
cd Smartphone-Price-Prediction
```

### Running the Notebook (Google Colab)

1. Upload the `Smartphone_Specifications_Dataset.csv` to your Google Drive.
2. Open the `.ipynb` file in Google Colab.
3. Update the dataset path in the notebook's loading cell to match your Drive location.
4. Execute the cells top to bottom to step through the data cleaning, modeling, and visualization phases.

### Running the Interactive Dashboard (Local)

The repository includes a fully interactive Streamlit web dashboard (`dashboard.py`) that allows you to explore the data, adjust model parameters in real-time, and input custom phone specifications for live price predictions.

1. **Install the required libraries** using your terminal or command prompt:
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn
   ```
2. **Ensure the dataset is present**: Make sure `Smartphone_Specifications_Dataset.csv` is located in the exact same folder as the dashboard python file.
3. **Launch the dashboard**:
   ```bash
   streamlit run dashboard.py
   ```
4. A browser window will automatically open at `http://localhost:8501/` where you can interact with the models.

---

## Authors

Developed as a final project for the **Data Science and Methodology Course**
Team:
- Moahmed Ezzat Elshanouri
- Maryam Mohamed Rageh
- Abdelrahman ElKhashab
- Mariam Mohamed Hassan
- Abdullah Helmy
**Egyptian University of Informatics**
