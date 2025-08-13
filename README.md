# Bank Customer Churn Prediction

A **Streamlit** web application that predicts whether a bank customer will churn using a trained machine learning model.
The app takes customer details as input and provides a prediction with an intuitive interface.

---

## Features

* Interactive **Streamlit** UI for customer data entry
* Trained various models like **Random Forest**, **Logistic Regression**, **SVM Classifier**, **Gradient Boost** amongst others.
* Performed hyperparameter tuning and cross validation to select best params for best performing model
* **One-hot encoding** for categorical data
* **Feature scaling** using `StandardScaler`
* Easy deployment to **Streamlit Cloud**, Heroku, or Render
* Modular code structure for maintainability

---

## Project Structure

```
bank_customer_churn/
│
├── app.py                       # Streamlit app entry point
│
├── models/
│   ├── Churn_model.pkl          # Trained model
│   ├── scaler.pkl               # Scaler for numeric features
│   └── feature_names.pkl     # Column order for input alignment
│
├── data/
│   ├── raw/                     # Raw dataset
│   └── processed/               # Cleaned dataset for training
│
├── notebooks/
│   └── Bank_Churn_Prediction.ipynb   # Notebook for EDA & training
│
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Ignore unnecessary files
```

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/bank-churn-prediction.git
cd bank-churn-prediction
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Run Locally

```bash
streamlit run app.py
```

Then open your browser at: **[http://localhost:8501](http://localhost:8501)**

---

## Deployment (Streamlit Cloud)

1. Push your project to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io/).
3. Connect your GitHub repo and deploy.
---

## Model Training
Model and scaler are saved as `.pkl` files in the `models/` directory.
The chosen model after hyperparameter tuning and cross validation was a Gradient Booster Classifier

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

* [Streamlit](https://streamlit.io/)
* [Scikit-learn](https://scikit-learn.org/)
* Dataset: Bank customer churn dataset (adapted for ML)
