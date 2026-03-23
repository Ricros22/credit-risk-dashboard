# =========================================
# LIBRERÍAS
# =========================================
import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# =========================================
# CONFIG
# =========================================
st.set_page_config(page_title="Credit Risk Engine", layout="wide")

# =========================================
# ESTILO FINTECH
# =========================================
st.markdown("""
    <style>
    body {
        background-color: #0E1117;
        color: white;
    }
    .stMetric {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================
# CARGA MODELO
# =========================================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("💳 Credit Risk Engine")

# =========================================
# INPUTS
# =========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Client Profile")
    
    income = st.slider("Income ($)", 10_000, 100_000, 30_000, step=1_000)
    age = st.slider("Age", 18, 70, 40)
    loan_amount = st.slider("Loan Amount ($)", 1_000, 50_000, 15_000, step=1_000)
    credit_score = st.slider("Credit Score", 300, 850, 650)

with col2:
    st.subheader("Risk Parameters")
    
    lgd = st.slider("LGD", 0.1, 0.9, 0.4)
    base_rate = st.slider("Base Interest Rate", 0.01, 0.15, 0.05)
    risk_premium = st.slider("Risk Premium", 0.1, 0.5, 0.2)

# =========================================
# FEATURE ENGINEERING
# =========================================
debt_to_income = loan_amount / income
income_per_age = income / age

X = np.array([[income, age, loan_amount, credit_score,
               debt_to_income, income_per_age]])

X_scaled = scaler.transform(X)

# =========================================
# MODELO
# =========================================
pd_proba = model.predict_proba(X_scaled)[0][1]
ead = loan_amount
expected_loss = pd_proba * lgd * ead
interest_rate = base_rate + pd_proba * risk_premium

# =========================================
# KPIs
# =========================================
st.subheader("📊 Risk Metrics")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric("PD (Default Risk)", f"{pd_proba:.2%}")
kpi2.metric("LGD", f"{lgd:.2%}")
kpi3.metric("Expected Loss", f"${expected_loss:,.0f}")
kpi4.metric("Interest Rate", f"{interest_rate:.2%}")

score = int((1 - pd_proba) * 850)
kpi5.metric("Credit Score", f"{score:,}")

# =========================================
# SEMÁFORO
# =========================================
st.subheader("Risk Level")

if pd_proba < 0.2:
    st.markdown("🟢 **Low Risk Client**")
elif pd_proba < 0.5:
    st.markdown("🟡 **Medium Risk Client**")
else:
    st.markdown("🔴 **High Risk Client**")

st.progress(float(pd_proba))

# =========================================
# DISTRIBUCIÓN
# =========================================
st.subheader("📊 Risk Distribution")

simulated = np.random.beta(2, 5, 1000)

fig, ax = plt.subplots()
ax.hist(simulated, bins=30)
ax.axvline(pd_proba, linestyle='dashed')
ax.set_title("Population Risk vs Client")
st.pyplot(fig)

# =========================================
# EXPLICABILIDAD
# =========================================
st.subheader("🔍 Model Explainability")

feature_names = [
    "income", "age", "loan_amount", "credit_score",
    "debt_to_income", "income_per_age"
]

coefficients = model.coef_[0]
impact = coefficients * X_scaled[0]

explain_df = pd.DataFrame({
    "Feature": feature_names,
    "Impact": impact
}).sort_values(by="Impact", ascending=False)

st.dataframe(explain_df.style.format({"Impact": "{:.4f}"}))

# gráfica
fig2, ax2 = plt.subplots()
ax2.barh(explain_df["Feature"], explain_df["Impact"])
ax2.set_title("Feature Impact on Risk")
st.pyplot(fig2)

# =========================================
# BATCH SCORING
# =========================================
st.subheader("📁 Batch Scoring")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    df["debt_to_income"] = df["loan_amount"] / df["income"]
    df["income_per_age"] = df["income"] / df["age"]
    
    X_batch = df[[
        "income", "age", "loan_amount", "credit_score",
        "debt_to_income", "income_per_age"
    ]]
    
    X_scaled = scaler.transform(X_batch)
    
    df["PD"] = model.predict_proba(X_scaled)[:, 1]
    df["LGD"] = lgd
    df["EAD"] = df["loan_amount"]
    df["Expected_Loss"] = df["PD"] * df["LGD"] * df["EAD"]
    df["Interest_Rate"] = base_rate + df["PD"] * risk_premium
    df["Score"] = ((1 - df["PD"]) * 850).astype(int)

    st.dataframe(df.style.format({
        "PD": "{:.2%}",
        "LGD": "{:.2%}",
        "Expected_Loss": "${:,.0f}",
        "Interest_Rate": "{:.2%}",
        "Score": "{:,}"
    }))

    st.download_button(
        "Download Results",
        df.to_csv(index=False),
        "scored_clients.csv"
    )