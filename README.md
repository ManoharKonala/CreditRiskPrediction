# CreditRiskPrediction

A project for predicting credit risk using machine learning and visualizing results with Tableau.

---

## 📂 Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Requirements](#requirements)
- [Highlights](#highlights)

---

## 🗂 Project Structure

- [`Ml Model/PredictionModel.ipynb`](https://github.com/ManoharKonala/CreditRiskPrediction/blob/main/Ml%20Model/PredictionModel.ipynb)  
  Jupyter notebook for data preprocessing, model training, and evaluation.

- [`Ml Model/credit_risk_model_updated.pkl`](https://github.com/ManoharKonala/CreditRiskPrediction/blob/main/Ml%20Model/credit_risk_model_updated.pkl)  
  Trained machine learning model (pickle file) for deployment or inference.

- [`GrossApproval&NaisDescription.twbx`](https://github.com/ManoharKonala/CreditRiskPrediction/blob/main/GrossApproval%26NaisDescription.twbx)  
  Tableau dashboard for gross approvals and NAIS descriptions.

- [`Visualizations Using Tableau.twb`](https://github.com/ManoharKonala/CreditRiskPrediction/blob/main/Visualizations%20Using%20Tableau.twb)  
  Additional Tableau visualizations.

---

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ManoharKonala/CreditRiskPrediction.git
   cd CreditRiskPrediction
   ```

2. **Open the Jupyter notebook:**
   - Launch Jupyter Notebook or VS Code.
   - Open [`Ml Model/PredictionModel.ipynb`](https://github.com/ManoharKonala/CreditRiskPrediction/blob/main/Ml%20Model/PredictionModel.ipynb).

3. **Explore Tableau dashboards:**
   - Open `.twb` and `.twbx` files in Tableau Desktop.

---

## 🛠 Usage

- Run the notebook to understand and reproduce the credit risk prediction workflow.
- Use the `.pkl` model for making predictions on new data:
  ```python
  import pickle

  with open('Ml Model/credit_risk_model_updated.pkl', 'rb') as f:
      model = pickle.load(f)
  # model.predict(your_data)
  ```
- Open Tableau files for interactive data visualizations.

---

## 📦 Requirements

- Python (with Jupyter Notebook)
- Required Python libraries (see notebook)
- Tableau Desktop

---

## ✨ Highlights

- End-to-end credit risk prediction workflow
- Ready-to-use trained machine learning model
- Interactive Tableau dashboards for business insights

---

Feel free to reach out for questions or contributions!
