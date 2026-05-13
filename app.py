import streamlit as st
import joblib
import pandas as pd
import json
from pathlib import Path

# Konfigurasi Halaman
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

# Path model
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

st.title("📱 Telco Customer Churn Prediction")

# Form Input
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    with col2:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=1)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

    # Baris tombol
    c_sub, c_clr = st.columns([1, 5])
    with c_sub:
        submit = st.form_submit_button("Submit")
    with c_clr:
        # Tombol clear di Streamlit Cloud biasanya otomatis me-refresh form jika diletakkan di luar atau menggunakan state
        clear = st.form_submit_button("Clear")

if submit:
    st.divider()
    st.subheader("Output Prediksi")
    
    # Mapping input ke DataFrame (Pastikan nama key/kolom sesuai dengan training model)
    input_dict = {
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SeniorCitizen": 0 # Tambahan jika model mewajibkan fitur ini
    }
    
    df = pd.DataFrame([input_dict])
    
    # Preprocessing
    all_cat_cols = RAW_CAT_COLS + RAW_BINARY_COLS
    df_encoded = pd.get_dummies(df, columns=[c for c in all_cat_cols if c in df.columns])
    df_aligned = df_encoded.reindex(columns=feature_cols, fill_value=0)
    
    # Prediksi
    prob = float(model.predict_proba(df_aligned)[:, 1][0])
    is_churn = prob >= THRESHOLD
    
    # Hasil
    if is_churn:
        st.error(f"### 🚨 HIGH RISK: Customer likely to churn!")
    else:
        st.success(f"### ✅ LOW RISK: Customer likely to stay.")
        
    st.metric("Churn Probability", f"{prob*100:.2f}%")
    st.write("Data Detail:", df)