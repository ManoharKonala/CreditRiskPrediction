# CreditRiskPrediction

A comprehensive project for predicting credit risk using machine learning and visualizing results with Tableau. This repository provides a full workflow from data analysis and model building to interactive business dashboards.

---

## 📂 Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Interactive Usage](#interactive-usage)
- [Model Details](#model-details)
- [Visualization Insights](#visualization-insights)
- [Requirements](#requirements)
- [Contributing](#contributing)

---

## 📝 Project Overview

Credit risk prediction is crucial for financial institutions to assess the likelihood of a borrower defaulting on a loan. This project leverages machine learning to automate and improve the accuracy of credit risk assessment, and uses Tableau for rich, interactive data visualizations.

---

## 📊 Data

The model utilizes `foia-7afy2010-fy2019-asof-221231.csv` as its input dataset. This dataset is not included in the repository and needs to be acquired separately. It can be obtained from a public FOIA request. [Link to data source or description if available]

---

## 🗂 Repository Structure

| Path | Description |
|------|-------------|
| [`Ml Model/PredictionModel.ipynb`](Ml%20Model/PredictionModel.ipynb) | Jupyter notebook for data cleaning, feature engineering, model training, evaluation, and prediction. Includes code cells, visualizations, and markdown explanations. |
| [`Ml Model/credit_risk_model_updated.pkl`](Ml%20Model/credit_risk_model_updated.pkl) | Trained machine learning model (pickle file) for direct use in production or further analysis. |
| [`GrossApproval&NaisDescription.twbx`](GrossApproval%26NaisDescription.twbx) | Tableau packaged workbook with dashboards on gross approvals and NAIS descriptions. |
| [`Visualizations Using Tableau.twb`](Visualizations%20Using%20Tableau.twb) | Tableau workbook with additional visualizations and insights. |

---

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ManoharKonala/CreditRiskPrediction.git
   cd CreditRiskPrediction
   ```
2. **Open the Jupyter notebook:**
   - Launch Jupyter Notebook or VS Code.
   - Open [`Ml Model/PredictionModel.ipynb`](Ml%20Model/PredictionModel.ipynb).
3. **Explore Tableau dashboards:**
   - Open `.twb` and `.twbx` files in Tableau Desktop for interactive visualizations.

---

## 🖱️ Interactive Usage

- **Run the notebook step-by-step:**
  - Each cell is annotated with markdown for clarity.
  - Modify parameters and re-run cells to experiment with different model settings.
- **Use the trained model for predictions:**
  ```python
  import pickle
  with open('Ml Model/credit_risk_model_updated.pkl', 'rb') as f:
      model = pickle.load(f)
  # Example: model.predict(your_data)
  ```
- **Interact with Tableau dashboards:**
  - Filter, drill down, and explore trends in the Tableau workbooks for actionable insights.

---

## 🤖 Model Details
- **Notebook highlights:**
  - Data preprocessing: handling missing values, encoding categorical variables, feature scaling.
  - Utilizes `TargetEncoder` for handling high-cardinality categorical features and `SelectKBest` for feature selection.
  - Employs SHAP (SHapley Additive exPlanations) for model interpretability, providing insights into feature contributions.
  - Evaluates model performance using various metrics, including classification reports and ROC AUC score, considering prediction thresholds such as 0.5 and 0.8.
  - Model training: includes model selection, hyperparameter tuning, and evaluation metrics (accuracy, ROC-AUC, etc.).
  - Model export: trained model is saved as `credit_risk_model_updated.pkl` for reuse.
- **How to retrain:**
  - Update the notebook with new data and re-run all cells.
  - Save the new model as a `.pkl` file for deployment.

---

## 📤 Notebook Outputs

The Jupyter notebook `Ml Model/PredictionModel.ipynb` generates the following key files:

-   `risky_borrowers_threshold_0.7.csv`: A CSV file listing borrowers identified as high risk (based on an 0.8 probability threshold in the code), including their risk probability and top contributing factors from SHAP analysis.
-   `total_risk_summary.csv`: A CSV file providing a summary of total loans, actual risky loans, and predicted risky loans at different thresholds.
-   `shap_summary_2.png`: An image file containing the SHAP summary plot, visualizing global feature importance.

---

## 📊 Visualization Insights
- **GrossApproval&NaisDescription.twbx:**
  - Interactive dashboards for gross approvals and NAIS descriptions.
  - Useful for business users to understand approval trends and risk segments.
- **Visualizations Using Tableau.twb:**
  - Additional charts and dashboards for deeper data exploration.
  - Use Tableau’s filters and drill-down features for custom analysis.

---

## 📦 Requirements
- Python 3.x
- Jupyter Notebook
- pandas, scikit-learn, matplotlib, seaborn (see notebook for full list)
- Tableau Desktop (for `.twb`/`.twbx` files)

---

## 🤝 Contributing
- Fork the repository and submit pull requests for improvements.
- Open issues for questions, suggestions, or bug reports.

---

For more details, see the code, notebooks, and dashboards in this repository. Enjoy exploring and predicting credit risk interactively!