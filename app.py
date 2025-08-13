import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from streamlit_extras.add_vertical_space import add_vertical_space


# Custom Streamlit page config
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the trained model and scaler
model = joblib.load("Churn_model.pkl")
scaler = joblib.load("scaler.pkl")


# Add a stylish header
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2em;
    }
    .stButton>button {
        background-color: #2E8B57;
        color: white;
        font-size: 1.1rem;
        border-radius: 8px;
        padding: 0.5em 2em;
    }
    .stSuccess {
        background-color: #e6ffe6;
        border-left: 5px solid #2E8B57;
    }
    </style>
    <div class="main-title">💳 Bank Customer Churn Prediction</div>
    <div class="subtitle">Predict whether a customer will leave your bank using machine learning</div>
    """,
    unsafe_allow_html=True
)


# Sidebar for user info and branding
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
        width=120,
    )
    st.markdown("<h3 style='color:#2E8B57;'>Welcome!</h3>", unsafe_allow_html=True)
    st.write("Fill in the customer details below to predict churn.")
    add_vertical_space(2)
    st.markdown("---")

# Main input form
with st.form("churn_form"):
    st.markdown("<h4 style='color:#2E8B57;'>Customer Information</h4>", unsafe_allow_html=True)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1, help="Customer's credit score")
    age = st.number_input("Age", min_value=18, max_value=100, step=1, help="Customer's age")
    tenure = st.number_input("Tenure (Years)", min_value=0, max_value=50, step=1, help="Years with the bank")
    balance = st.number_input("Balance", min_value=0.0, step=100.0, help="Account balance")
    num_products = st.number_input("Number of Products", min_value=1, max_value=4, step=1, help="Number of bank products")
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"], help="Does the customer have a credit card?")
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"], help="Is the customer active?")
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=100.0, help="Estimated annual salary")
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"], help="Customer's country")
    gender = st.selectbox("Gender", ["Female", "Male"], help="Customer's gender")
    submitted = st.form_submit_button("🔮 Predict Churn")


# When Predict button is clicked
if submitted:
    # Log transform Age
    import numpy as np
    log_age = 0 if age <= 0 else np.log(age)

    # Create DataFrame with raw input
    input_df = pd.DataFrame({
        "CreditScore": [credit_score],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [1 if has_cr_card == "Yes" else 0],
        "IsActiveMember": [1 if is_active_member == "Yes" else 0],
        "EstimatedSalary": [estimated_salary],
        "LogAge": [log_age],
        "Geography": [geography],
        "Gender": [gender]
    })

    # One-hot encode Geography and Gender
    input_df = pd.get_dummies(input_df, columns=["Geography", "Gender"])

    # Ensure all columns match training features
    expected_cols = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'LogAge',
        'Geography_France', 'Geography_Germany', 'Geography_Spain',
        'Gender_Female', 'Gender_Male'
    ]
    for col in expected_cols:
        if col not in input_df:
            input_df[col] = 0  # Add missing dummy columns

    # Reorder columns to match training
    input_df = input_df[expected_cols]

    # Scale numeric features
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.markdown("<h3 style='color:#d9534f;'>🚨 Customer Will Churn</h3>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;'><img src='https://cdn-icons-png.flaticon.com/512/564/564619.png' width='80'></div>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:#2E8B57;'>✅ Customer Will Stay</h3>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;'><img src='https://cdn-icons-png.flaticon.com/512/190/190411.png' width='80'></div>", unsafe_allow_html=True)
