import streamlit as st
import pandas as pd
import joblib

# Load model (pastikan path file sesuai dengan di GitHub)
# model = joblib.load('model/model_churn.pkl')

st.set_page_config(page_title="Telco Churn Prediction", layout="centered")

st.title("Telco Customer Churn Prediction")
st.write("Silakan masukkan data pelanggan di bawah ini:")

# Menggunakan form agar lebih rapi
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
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
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=0.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=0.0)

    # Tombol Submit
    submitted = st.form_submit_button("Submit")

# Logika Output setelah klik Submit
if submitted:
    st.subheader("Output Prediksi")
    
    # Membuat DataFrame dari input (sesuaikan dengan format model Anda)
    input_data = pd.DataFrame({
        'gender': [gender],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        # ... tambahkan kolom lainnya sesuai urutan fitur model ...
    })
    
    st.write("Data yang dimasukkan:")
    st.dataframe(input_data)
    
    # Contoh pemanggilan prediksi:
    # prediction = model.predict(input_data)
    # st.success(f"Hasil Prediksi: {prediction[0]}")