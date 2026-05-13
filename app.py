import streamlit as st
import joblib
import pandas as pd
import json
from pathlib import Path

# Konfigurasi Halaman
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

# Path model (menyesuaikan struktur folder Anda)
MODEL_DIR = Path(__file__).parent / "model"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "model.pkl")
    feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
    prep_info = joblib.load(MODEL_DIR / "preprocessing_info.pkl")
    with open(MODEL_DIR / "model_meta.json") as f:
        meta = json.load(f)
    return model, feature_cols, prep_info, meta

try:
    model, feature_cols, prep_info, meta = load_artifacts()
    THRESHOLD = meta["threshold"]
    RAW_CAT_COLS = prep_info["raw_cat_cols"]
    RAW_BINARY_COLS = prep_info["raw_binary_cols"]
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# --- UI STREAMLIT ---
st.title("📱 Telco Customer Churn Prediction")
st.write(f"Model Version: {meta['version']}")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        
    with col2:
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        
    with col3:
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        MonthlyCharges = st.number_input("Monthly Charges", value=50.0)
        TotalCharges = st.number_input("Total Charges", value=500.0)

    submit = st.form_submit_button("Predict Churn Risk")

if submit:
    # Preprocessing (Logika sama dengan FastAPI Anda)
    input_data = {
        "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner,
        "Dependents": Dependents, "tenure": tenure, "PhoneService": PhoneService,
        "MultipleLines": MultipleLines, "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity, "TechSupport": TechSupport,
        "Contract": Contract, "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        # Default kolom lain yang mungkin dibutuhkan model Anda
        "OnlineBackup": "No", "DeviceProtection": "No", 
        "StreamingTV": "No", "StreamingMovies": "No"
    }
    
    df = pd.DataFrame([input_data])
    all_cat_cols = RAW_CAT_COLS + RAW_BINARY_COLS
    df_encoded = pd.get_dummies(df, columns=all_cat_cols)
    df_aligned = df_encoded.reindex(columns=feature_cols, fill_value=0)
    
    # Prediksi
    prob = float(model.predict_proba(df_aligned)[:, 1][0])
    is_churn = prob >= THRESHOLD
    
    # Tampilkan Hasil
    st.divider()
    if is_churn:
        st.error(f"### 🚨 HIGH RISK: Customer likely to churn!")
    else:
        st.success(f"### ✅ LOW RISK: Customer likely to stay.")
        
    st.metric("Churn Probability", f"{prob*100:.2f}%")