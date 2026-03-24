# 💳 Credit Risk Engine Dashboard

An end-to-end credit risk modeling and decision engine combining machine learning, financial risk metrics, and an interactive dashboard for real-time analysis.

---

## 🚀 Overview

This project simulates a real-world credit risk workflow used in banking and fintech environments. It estimates the probability of default (PD) using a logistic regression model and extends it into a full financial risk framework including expected loss and risk-based pricing.

The solution is deployed as an interactive dashboard built with Streamlit, allowing users to simulate client profiles, analyze risk drivers, and perform batch scoring.

---

## ⚙️ Key Features

- 📊 **Probability of Default (PD)** using Logistic Regression  
- 💸 **Loss Given Default (LGD)** as a configurable parameter  
- 📉 **Expected Loss (EL)** calculation  
- 💰 **Risk-based Pricing** (interest rate estimation)  
- 🔍 **Model Explainability** (feature impact analysis)  
- 📈 **Risk Visualization** (distribution vs client position)  
- 📁 **Batch Scoring** via CSV upload  
- 🎨 **Fintech-style Dashboard UI**

---

## 🧠 Methodology

The model follows a standard credit risk framework:

\[
EL = PD \times LGD \times EAD
\]

Where:

- **PD (Probability of Default):** Estimated using logistic regression  
- **LGD (Loss Given Default):** User-defined loss severity  
- **EAD (Exposure at Default):** Loan amount  

The pricing component is derived as:

\[
Interest Rate = Base Rate + PD \times Risk Premium
\]

This structure reflects common practices in credit risk management and lending decision systems.

---

## 📊 Model Details

- Model: Logistic Regression (scikit-learn)  
- Feature Engineering:
  - Debt-to-Income Ratio  
  - Income per Age  
- Scaling: StandardScaler  
- Evaluation Metrics:
  - ROC-AUC  
  - KS Statistic  
  - Precision / Recall  

---

## 🖥️ Dashboard Capabilities

The Streamlit dashboard allows users to:

- Simulate individual client risk profiles  
- Visualize probability of default  
- Understand model drivers through explainability  
- Estimate expected loss and pricing  
- Upload datasets and perform batch scoring  

---

## 📂 Project Structure
credit-risk-dashboard/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

🌐 Live Demo

👉 https://credit-risk-dashboard-2seq9ttdxk5ewune7bmzcg.streamlit.app/

💡 Business Applications

This type of system can be applied to:

Credit approval and underwriting
Risk-based pricing strategies
Portfolio risk monitoring
Expected loss estimation
Decision support systems in fintech
🚀 Future Improvements
Model calibration (Platt scaling / isotonic regression)
Advanced models (Gradient Boosting, XGBoost)
SHAP-based explainability
Portfolio segmentation and clustering
Integration with real datasets
👨‍💻 Author

Ricardo RM
Actuarial & Financial Data Analysis
