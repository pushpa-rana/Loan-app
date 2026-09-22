import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Loan Risk Assessment",
    page_icon="🏦",
    layout="wide"
)

with open("style.css", "r") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

model = joblib.load("loan_model.joblib")

st.markdown(
    '<div class="main-title">🏦 Loan Risk Assessment System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning based loan approval prediction and risk analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">👤 Applicant Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:
    married = st.selectbox(
        "Marital Status",
        ["Yes", "No"]
    )

with col3:
    dependents = st.selectbox(
        "Number of Dependents",
        ["0", "1", "2", "3+"]
    )

col4, col5 = st.columns(2)

with col4:
    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

with col5:
    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

st.markdown(
    '<div class="section-title">💰 Financial Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000,
        step=500
    )

with col2:
    coapplicant_income = st.number_input(
        "Co-applicant Income",
        min_value=0,
        value=0,
        step=500
    )

with col3:
    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=150,
        step=10
    )

col4, col5 = st.columns(2)

with col4:
    loan_amount_term = st.selectbox(
        "Loan Term (Months)",
        [360, 180, 120, 84, 60, 36]
    )

with col5:
    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

st.markdown(
    '<div class="section-title">📊 Credit Information</div>',
    unsafe_allow_html=True
)

credit_history = st.selectbox(
    "Credit History",
    ["Good", "Poor", "Not Available"]
)

total_income = applicant_income + coapplicant_income

if total_income > 0:
    loan_to_income_ratio = loan_amount / total_income
else:
    loan_to_income_ratio = 0

loan_term_years = loan_amount_term / 12

if loan_amount > 0 and loan_amount_term > 0:
    monthly_rate = 0.08 / 12
    principal = loan_amount * 1000

    estimated_emi = (
        principal *
        monthly_rate *
        (1 + monthly_rate) ** loan_amount_term
    ) / (
        (1 + monthly_rate) ** loan_amount_term - 1
    )
else:
    estimated_emi = 0

st.markdown(
    '<div class="section-title">📋 Financial Summary</div>',
    unsafe_allow_html=True
)

summary1, summary2, summary3, summary4 = st.columns(4)

with summary1:
    st.metric(
        "Total Income",
        f"{total_income:,.0f}"
    )

with summary2:
    st.metric(
        "Loan Amount",
        f"{loan_amount:,.0f}"
    )

with summary3:
    st.metric(
        "Loan / Income",
        f"{loan_to_income_ratio:.2f}"
    )

with summary4:
    st.metric(
        "Estimated EMI",
        f"{estimated_emi:,.0f}"
    )

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "🔍 ANALYZE LOAN APPLICATION"
)

if predict_button:

    gender_value = gender.lower()
    married_value = married.lower()

    if dependents == "3+":
        dependents_value = 3
    else:
        dependents_value = int(dependents)

    education_value = education.lower()
    self_employed_value = self_employed.lower()
    property_area_value = property_area.lower()

    if credit_history == "Good":
        credit_history_value = 1
    elif credit_history == "Poor":
        credit_history_value = 0
    else:
        credit_history_value = np.nan

    input_data = pd.DataFrame({
        "gender": [gender_value],
        "married": [married_value],
        "dependents": [dependents_value],
        "education": [education_value],
        "self_employed": [self_employed_value],
        "applicantincome": [applicant_income],
        "coapplicantincome": [coapplicant_income],
        "loanamount": [loan_amount],
        "loan_amount_term": [loan_amount_term],
        "credit_history": [credit_history_value],
        "property_area": [property_area_value]
    })

    input_data["total_income"] = (
        input_data["applicantincome"] +
        input_data["coapplicantincome"]
    )

    input_data["loan_to_income_ratio"] = (
        input_data["loanamount"] /
        input_data["total_income"].replace(0, np.nan)
    )

    input_data["loan_term_years"] = (
        input_data["loan_amount_term"] / 12
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    rejection_probability = probabilities[0] * 100
    approval_probability = probabilities[1] * 100

    st.markdown(
        '<div class="section-title">📌 Prediction Result</div>',
        unsafe_allow_html=True
    )

    if prediction == 1:

        st.markdown(
            f"""<div class="approved-card">
<div class="result-title">✅ LOAN APPROVED</div>
<div class="probability">
Model Approval Probability:
<strong>{approval_probability:.1f}%</strong>
</div>
<div class="probability">
Model Rejection Probability:
<strong>{rejection_probability:.1f}%</strong>
</div>
</div>""",
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""<div class="rejected-card">
<div class="result-title">❌ LOAN NOT APPROVED</div>
<div class="probability">
Model Approval Probability:
<strong>{approval_probability:.1f}%</strong>
</div>
<div class="probability">
Model Rejection Probability:
<strong>{rejection_probability:.1f}%</strong>
</div>
</div>""",
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">🔎 Factors Influencing the Prediction</div>',
        unsafe_allow_html=True
    )

    factors = []

    if credit_history == "Good":
        factors.append(
            "🟢 Good credit history supports loan approval."
        )
    elif credit_history == "Poor":
        factors.append(
            "🔴 Poor credit history negatively affects the prediction."
        )
    else:
        factors.append(
            "🟠 Credit history is unavailable, increasing uncertainty."
        )

    if loan_to_income_ratio > 0.40:
        factors.append(
            "🔴 The requested loan is relatively high compared with total income."
        )
    elif loan_to_income_ratio < 0.20:
        factors.append(
            "🟢 The requested loan is relatively small compared with total income."
        )
    else:
        factors.append(
            "🟡 The loan-to-income ratio is within a moderate range."
        )

    if total_income >= 10000:
        factors.append(
            "🟢 Higher combined applicant income supports repayment capacity."
        )
    elif total_income < 5000:
        factors.append(
            "🔴 Lower combined income may reduce repayment capacity."
        )
    else:
        factors.append(
            "🟡 Combined income provides a moderate repayment capacity."
        )

    if education == "Graduate":
        factors.append(
            "🟢 Graduate education is associated with a positive model signal."
        )
    else:
        factors.append(
            "🟡 Education level contributes to the overall model assessment."
        )

    if dependents_value >= 3:
        factors.append(
            "🟠 A higher number of dependents may increase financial obligations."
        )
    else:
        factors.append(
            "🟢 Number of dependents does not indicate a high dependency burden."
        )

    if loan_amount_term >= 360:
        factors.append(
            "🟡 A longer loan term spreads repayment over a longer period."
        )
    else:
        factors.append(
            "🟢 A shorter loan term results in a shorter repayment period."
        )

    for factor in factors:
        st.write(factor)

    st.info(
        "ℹ️ These factors describe patterns used by the machine-learning "
        "system. They are not guaranteed reasons a real financial institution "
        "would approve or reject a loan."
    )

st.markdown(
    """
    <div class="footer">
        Loan Risk Assessment System |
        Python • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
