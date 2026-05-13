# =============================================================
# colab_export.py
# Jalankan file ini di GOOGLE COLAB (bagian akhir notebook EDA kamu)
# Tujuan: export semua artifact yang dibutuhkan untuk deployment
# =============================================================
#
# CARA PAKAI:
#   Paste kode ini ke cell baru di Colab, jalankan setelah training selesai.
#   File output akan tersimpan di /content/ dan bisa kamu download.
#
# OUTPUT FILES:
#   model.pkl        → model XGBoost hasil training
#   encoders.pkl     → semua LabelEncoder yang sudah di-fit
#   feature_cols.pkl → urutan kolom yang dipakai model
#   model_meta.json  → metadata: threshold, versi, metrics
# =============================================================

import joblib
import json
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# STEP 1: Definisikan kolom kategorikal dan fitur
# ------------------------------------------------------------------
# Sesuaikan dengan kolom yang kamu encode di notebook

CAT_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

# Urutan kolom HARUS sama persis dengan saat training
FEATURE_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# ------------------------------------------------------------------
# STEP 2: Fit ulang encoder dari raw data (sebelum encoding)
# ------------------------------------------------------------------
# PENTING: kamu perlu df_raw = dataframe SEBELUM di-encode
# Ganti 'df_raw' dengan nama variable dataframe aslimu

df_raw = pd.read_csv('Telco-Customer-Churn.csv')
df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')

encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    le.fit(df_raw[col].astype(str))
    encoders[col] = le
    print(f"  {col}: {list(le.classes_)}")

print("\nEncoder berhasil dibuat.")

# ------------------------------------------------------------------
# STEP 3: Simpan semua artifact
# ------------------------------------------------------------------

# Simpan model (ganti 'xgb' dengan nama variable model XGBoost kamu)
# Pastikan variabel 'xgb' sudah ada dari proses training sebelumnya
joblib.dump(xgb, '/content/model.pkl')
print("model.pkl tersimpan")

# Simpan encoders
joblib.dump(encoders, '/content/encoders.pkl')
print("encoders.pkl tersimpan")

# Simpan urutan kolom
joblib.dump(FEATURE_COLS, '/content/feature_cols.pkl')
print("feature_cols.pkl tersimpan")

# Simpan metadata model
meta = {
    "model_name": "XGBoost Telco Churn",
    "version": "1.0.0",
    "threshold": float(THRESHOLD),   # ganti THRESHOLD dengan variable threshold kamu
    "feature_cols": FEATURE_COLS,
    "cat_cols": CAT_COLS,
    "metrics": {
        "recall_churn": 0.925,        # isi dari hasil evaluasi kamu
        "precision_churn": 0.433,
        "f1_churn": 0.590,
        "roc_auc": 0.85
    }
}
with open('/content/model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
print("model_meta.json tersimpan")

# ------------------------------------------------------------------
# STEP 4: Verifikasi artifact bisa di-load kembali
# ------------------------------------------------------------------
print("\n--- Verifikasi ---")
_model    = joblib.load('/content/model.pkl')
_encoders = joblib.load('/content/encoders.pkl')
_cols     = joblib.load('/content/feature_cols.pkl')

# Test prediksi 1 baris dummy
dummy = {col: df_raw[col].iloc[0] for col in FEATURE_COLS}
dummy_df = pd.DataFrame([dummy])
dummy_df['TotalCharges'] = pd.to_numeric(dummy_df['TotalCharges'], errors='coerce').fillna(0)
for col in CAT_COLS:
    dummy_df[col] = _encoders[col].transform(dummy_df[col].astype(str))

pred = _model.predict_proba(dummy_df[_cols])[:, 1]
print(f"Test prediksi dummy berhasil: prob_churn = {pred[0]:.4f}")
print("\nSemua artifact OK. Silakan download 4 file dari panel Files di kiri Colab.")
