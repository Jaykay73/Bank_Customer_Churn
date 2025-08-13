# Bank Customer Churn Prediction

A **Streamlit** web application that predicts whether a bank customer will churn using a trained machine learning model.  
The app takes customer details as input and provides a prediction with an intuitive interface.

---

## Features
- Interactive **Streamlit** UI for customer data entry
- Pre-trained **Random Forest** model
- **One-hot encoding** for categorical data
- **Feature scaling** using `StandardScaler`
- Easy deployment to **Streamlit Cloud**, Heroku, or Render
- Modular code structure for maintainability

---

## Project Structure
bank_churn_prediction/
│
├── app.py # Streamlit app entry point
│
├── models/
│ ├── Churn_model.pkl # Trained model
│ ├── scaler.pkl # Scaler for numeric features
│ └── expected_columns.pkl # Column order for input alignment
│
├── src/
│ ├── init.py
│ ├── preprocess.py # Data preprocessing functions
│ ├── predict.py # Model loading & prediction functions
│ └── utils.py # Utility helpers
│
├── data/
│ ├── raw/ # Raw dataset
│ └── processed/ # Cleaned dataset for training
│
├── notebooks/
│ └── Bank_Churn_Prediction.ipynb # Notebook for EDA & training
│
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── .gitignore # Ignore unnecessary files
