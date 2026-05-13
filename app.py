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
with st.form("prediction_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    
    with col1:
        # Untuk selectbox, pilihan pertama adalah default. 
        # Anda bisa menambahkan "-" sebagai pilihan kosong jika model mengizinkan.
        gender = st.selectbox("Gender", ["Male", "Female"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    with col2:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
        # Mengubah 'value' menjadi 0 agar terlihat kosong/bersih
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=0)
        monthly_charges = st.number_input("Monthly Charges", min_value=0, value=0)
        total_charger = st.number_input("Total Charger", min_value=0, value=0)

    # Baris tombol
    c_sub, c_clr = st.columns([1, 5])
    with c_sub:
        submit = st.form_submit_button("Submit")
    with c_clr:
        clear = st.form_submit_button("Clear")
        if clear:
            st.rerun()

if submit:
    # Cek jika user belum mengisi data (opsional)
    if tenure == 0 and monthly_charges == 0:
        st.warning("Mohon isi data tenure dan charges terlebih dahulu.")
    else:
        st.divider()
        st.subheader("Output Prediksi")
        
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
            "TotalCharger": total_charger,
        }
        
        df = pd.DataFrame([input_dict])
        all_cat_cols = RAW_CAT_COLS + RAW_BINARY_COLS
        df_encoded = pd.get_dummies(df, columns=[c for c in all_cat_cols if c in df.columns])
        df_aligned = df_encoded.reindex(columns=feature_cols, fill_value=0)
        
        prob = float(model.predict_proba(df_aligned)[:, 1][0])
        is_churn = prob >= THRESHOLD
        
        if is_churn:
            st.error(f"### 🚨 HIGH RISK: Customer likely to churn!")
        else:
            st.success(f"### ✅ LOW RISK: Customer likely to stay.")
            
        st.metric("Churn Probability", f"{prob*100:.2f}%")