# Credit Risk Prediction: ML Pipeline

Production-ready machine learning pipeline for predicting credit risk on SBA loans. Analyzes 1.57M loans (1991-2019) with 96.5% ROC-AUC using XGBoost.

## Quick Start

### Prerequisites
```bash
python --version  # 3.9+
pip install -r requirements.txt
```

### Run Pipeline
```bash
cd notebooks
jupyter notebook credit_risk_pipeline.ipynb
```

Execute: Kernel → Restart & Run All

**Expected Output:**
- Data loaded: 1,573,141 loans
- Features engineered: 23 total
- Best model: XGBoost (0.9652 ROC-AUC)
- Model saved: `models/credit_risk_model_updated.pkl`

## Project Structure

```
CreditRiskPrediction/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── CONTRIBUTING.md                    # Contribution guidelines
├── notebooks/
│   └── credit_risk_pipeline.ipynb    # Main ML pipeline
├── data/
│   ├── raw/                           # Raw CSV files
│   └── processed/                     # Generated during pipeline
├── models/
│   ├── credit_risk_model_updated.pkl  # Best model (XGBoost)
│   └── PredictionModel.ipynb          # Reference implementation
└── visualizations/                    # Tableau files
```

## Key Features

### Data
- **1.57M loans** across 28 years (1991-2019)
- **13.33% default rate** with 6.5:1 class imbalance
- **3 time periods** capturing economic cycles

### Features (23 Total)

**18 Base Features:**
- Loan characteristics, risk indicators, business profile, geography, temporal patterns

**5 Engineered Features:**
- GuaranteeRatio, HasJobs, LoanPerJob, InterestTermProduct, LoanSize

### ML Methodology

- ✅ Split BEFORE preprocessing (prevents data leakage)
- ✅ Fit transformers on train only
- ✅ Stratified 60/20/20 train-val-test split
- ✅ Class imbalance handling (scale_pos_weight=6.5)

### Model Performance

| Model | Val AUC | Test AUC |
|-------|---------|----------|
| Logistic Regression | 0.8121 | 0.8122 |
| Random Forest | ~0.83 | ~0.83 |
| Gradient Boosting | ~0.87 | ~0.87 |
| **XGBoost** | **0.9653** | **0.9652** |

**Best Model:** XGBoost with minimal overfitting (0.0001 gap)

## Pipeline Architecture

```
Raw Data (1.57M loans)
         ↓
Load & Explore (3 time periods)
         ↓
Feature Engineering (18 base + 5 engineered)
         ↓
Train-Test Split (60/20/20 stratified) ⭐ CRITICAL
         ↓
Preprocessing (Fit on Train Only)
    ├── Imputation (median/most_frequent)
    ├── Encoding (label/target encoding)
    ├── Feature engineering (5 derived features)
    └── Scaling (StandardScaler)
         ↓
Model Training (4 models)
         ↓
Model Evaluation (Validation → Test)
         ↓
Deployment (Save artifacts)
```

## Feature Engineering

### Base Features (18)

**Loan Characteristics:**
- GrossApproval, SBAGuaranteedApproval, TermInMonths, JobsSupported

**Risk Indicators:**
- InitialInterestRate

**Business Profile:**
- NaicsCode, BusinessType

**Program Details:**
- Program, DeliveryMethod

**Geographic:**
- BorrState, ProjectState

**Temporal:**
- ApprovalYear, ApprovalMonth, ApprovalQuarter
- DisbursementYear, DisbursementMonth, DaysToDisbursement
- TimePeriod

### Engineered Features (5)

| Feature | Formula | Purpose |
|---------|---------|---------|
| GuaranteeRatio | SBA Guarantee / Gross Approval | SBA coverage percentage |
| HasJobs | JobsSupported > 0 | Job creation indicator |
| LoanPerJob | Gross Approval / Jobs Supported | Capital efficiency metric |
| InterestTermProduct | Interest Rate × Term Months | Total interest burden |
| LoanSize | Binned: 0-50k, 50-150k, 150-350k, 350k+ | Loan size categories |

## Deployment

### Load Model
```python
import pickle

with open("models/credit_risk_model_updated.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
scaler = artifacts["scaler"]
imputer_num = artifacts["imputer_num"]
imputer_cat = artifacts["imputer_cat"]
```

### Make Predictions

```python
# Preprocess new data (assuming X_new is a DataFrame with required features)
X_new_processed = X_new.copy()

# Apply imputation
X_new_processed[numeric_features] = imputer_num.transform(X_new_processed[numeric_features])
X_new_processed[categorical_features] = imputer_cat.transform(X_new_processed[categorical_features])

# Scale features
X_new_scaled = scaler.transform(X_new_processed)

# Get predictions
predictions = model.predict(X_new_scaled)
probabilities = model.predict_proba(X_new_scaled)[:, 1]

# Interpret results
for idx, prob in enumerate(probabilities):
    risk_level = "HIGH" if prob > 0.5 else "LOW"
    print(f"Loan {idx}: {risk_level} risk ({prob:.2%})")
```

See **GUIDE_DEPLOYMENT.md** for complete deployment instructions.

## Data Leakage Prevention

**Checklist:**
- ✅ Split BEFORE preprocessing
- ✅ Fit imputers on train only
- ✅ Fit encoders on train only
- ✅ Fit scaler on train only
- ✅ Transform val/test with train statistics
- ✅ No information leakage between sets

## Quality Metrics

| Criterion | Score | Status |
|-----------|-------|--------|
| Data Leakage | 0 (None) | ✅ Pass |
| Preprocessing Order | Correct | ✅ Pass |
| Validation Strategy | 3-way split | ✅ Pass |
| Class Imbalance | Handled | ✅ Pass |
| Model Performance | 0.965 AUC | ✅ Pass |


## Technical Stack

**Python 3.9+**
- Data Processing: pandas, numpy
- ML: scikit-learn, xgboost
- Visualization: matplotlib, seaborn
- Utilities: jupyter, joblib


## Project Statistics

| Metric | Value |
|--------|-------|
| Total Loans | 1,573,141 |
| Time Period | 28 years (1991-2019) |
| Features | 23 (18 base + 5 engineered) |
| Models Trained | 4 |
| Best Model AUC | 0.9652 |
| Default Rate | 13.33% |
| Class Imbalance | 6.5:1 |
| Data Leakage | None |
| Production Ready | ✅ Yes |

## Next Steps

### Recommended Improvements

1. Hyperparameter tuning with GridSearchCV
2. 5-fold cross-validation for robustness
3. SHAP analysis for feature importance
4. Model ensemble combining multiple models
5. Threshold optimization for business metrics

### Future Enhancements

- Add LightGBM and CatBoost models
- Implement stacking ensemble
- Add fairness analysis
- Build real-time prediction API

## License

MIT License - See LICENSE file for details

---

**Status:** ✅ Production Ready | **Version:** 2.0 | **Last Updated:** March 17, 2026
