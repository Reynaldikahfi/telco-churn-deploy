import streamlit as st
import joblib
import pandas as pd
import json
from pathlib import Path

# Konfigurasi Halaman
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

# --- FUNGSI RESET ---
def reset_form():
    st.session_state["gender"] = "Male"
    st.session_state["partner"] = "No"
    st.session_state["dependents"] = "No"
    st.session_state["phone_service"] = "No"
    st.session_state["multiple_lines"] = "No"
    st.session_state["internet_service"] = "DSL"
    st.session_state["online_security"] = "No"
    st.session_state["online_backup"] = "No"
    st.session_state["device_protection"] = "No"
    st.session_state["tech_support"] = "No"
    st.session_state["streaming_tv"] = "No"
    st.session_state["streaming_movies"] = "No"
    st.session_state["contract"] = "Month-to-month"
    st.session_state["paperless_billing"] = "No"
    st.session_state["payment_method"] = "Electronic check"
    st.session_state["tenure"] = 1
    st.session_state["monthly_charges"] = 50.0
    st.session_state["total_charges"] = 500.0

# Inisialisasi state jika belum ada
if "gender" not in st.session_state:
    reset_form()

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

# Form Input menggunakan key dari session_state
with st.form("prediction_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        partner = st.selectbox("Partner", ["Yes", "No"], key="partner")
        dependents = st.selectbox("Dependents", ["Yes", "No"], key="dependents")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], key="phone_service")
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"], key="multiple_lines")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet_service")
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key="online_security")
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key="online_backup")
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], key="device_protection")

    with col2:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key="tech_support")
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key="streaming_tv")
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], key="streaming_movies")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], key="paperless_billing")
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], key="payment_method")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, key="tenure")
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, key="monthly_charges")
        total_charges = st.number_input("Total Charges", min_value=0.0, key="total_charges")

    col_btn1, col_btn2 = st.columns([1, 8])
    with col_btn1:
        submit = st.form_submit_button("Submit")
    with col_btn2:
        # Tombol Clear menggunakan callback untuk mereset state
        clear = st.form_submit_button("Clear", on_click=reset_form)

if submit:
    st.success("Analisis Selesai!")