# 💳 Credit Risk Engine

This project implements an end-to-end credit risk engine combining machine learning and financial modeling.

## 🚀 Features

- Probability of Default (PD) using Logistic Regression
- Loss Given Default (LGD)
- Expected Loss calculation
- Risk-based pricing (interest rate)
- Model explainability
- Interactive dashboard
- Batch scoring (CSV upload)

## 🧠 Methodology

The model estimates credit risk using a supervised learning approach and extends it into a full financial risk framework:

EL = PD × LGD × EAD

## 📊 Tech Stack

- Python
- Streamlit
- scikit-learn
- pandas
- matplotlib

## ▶️ Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
