# Telco Churn — Deployment Guide (Windows)

## XGBoost Model · FastAPI · Docker · GitHub Actions

---

## Struktur Project

```
telco-churn-deploy/
│
├── colab_export.py          ← (1) Jalankan di Colab untuk export model
│
├── app/
│   ├── main.py              ← (2) FastAPI service — endpoint prediksi
│   └── monitoring.py        ← (5) Prometheus metrics
│
├── model/                   ← Taruh file hasil export Colab di sini
│   ├── model.pkl
│   ├── preprocessing_info.pkl
│   ├── feature_cols.pkl
│   └── model_meta.json
│
├── notebooks/               ← Taruh file .ipynb dari Colab di sini
├── data/                    ← Taruh dataset CSV di sini
├── tests/
│   └── test_api.py          ← Unit & integration tests
│
├── Dockerfile               ← (3) Container image
├── docker-compose.yml       ← (3) Multi-service stack
├── requirements.txt         ← Production dependencies
├── requirements-dev.txt     ← Development dependencies
└── .github/
    └── workflows/
        └── ci-cd.yml        ← (4) GitHub Actions pipeline
```

---

## Langkah-langkah Deploy

### Phase 1 — Export model dari Colab

1. Buka notebook EDA kamu di Google Colab
2. Tambahkan cell baru di bagian paling bawah
3. Paste isi file `colab_export.py`, jalankan
4. Download 4 file yang muncul di panel Files kiri Colab:
   - `model.pkl`
   - `preprocessing_info.pkl`
   - `feature_cols.pkl`
   - `model_meta.json`
5. Taruh semua file ke folder `model\` di dalam project

---

### Phase 2 — Jalankan API lokal di VSCode

**Buka PowerShell di VSCode** (Terminal → New Terminal), lalu:

```powershell
# Cek posisi folder — pastikan ada Dockerfile dan requirements.txt
Get-ChildItem

# Izinkan menjalankan script (jalankan sekali saja)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
.\venv\Scripts\Activate.ps1

# Setelah aktif, terminal berubah jadi:
# (venv) PS E:\RCK\ML\telco-churn-deploy>

# Install semua dependencies
pip install -r requirements-dev.txt

# Jalankan server FastAPI
uvicorn app.main:app --reload --port 8000
```

Buka browser: **http://localhost:8000/docs** → Swagger UI otomatis tersedia untuk test prediksi.

Untuk **stop server**: tekan `Ctrl + C` di terminal.

---

### Phase 3 — Docker

> Pastikan **Docker Desktop** sudah terinstall dan sedang berjalan sebelum menjalankan perintah ini.
> Download: https://www.docker.com/products/docker-desktop

```powershell
# Build image Docker
docker build -t churn-api:v1.0 .

# Jalankan container tunggal (untuk test cepat)
docker run -p 8000:8000 churn-api:v1.0

# --- ATAU ---

# Jalankan semua service sekaligus (API + MLflow UI)
docker compose up --build

# Jalankan di background (tidak memblokir terminal)
docker compose up --build -d

# Lihat log real-time
docker compose logs -f churn-api

# Stop semua service
docker compose down
```

Setelah `docker compose up`:

- API → **http://localhost:8000**
- MLflow UI → **http://localhost:5000**

---

### Phase 4 — CI/CD (GitHub Actions)

```powershell
# Inisialisasi Git (jika belum)
git init
git add .
git commit -m "first commit"

# Hubungkan ke GitHub repo kamu
git remote add origin https://github.com/USERNAME/telco-churn-deploy.git
git branch -M main
git push -u origin main
```

Setelah push, setup secrets di GitHub:

1. Buka repo di GitHub → **Settings → Secrets and variables → Actions**
2. Klik **New repository secret**, tambahkan dua secret:
   - `DOCKERHUB_USERNAME` → username Docker Hub kamu
   - `DOCKERHUB_TOKEN` → token dari https://hub.docker.com/settings/security
3. Setiap push ke `main` akan otomatis: **test → build → push Docker image**

---

### Phase 5 — Testing

```powershell
# Pastikan venv aktif dulu
.\venv\Scripts\Activate.ps1

# Jalankan semua unit test
pytest tests/ -v

# Jalankan test dengan output lebih ringkas
pytest tests/ -v --tb=short
```

**Test endpoint manual via PowerShell** (pastikan server sudah jalan di terminal lain):

```powershell
# Test endpoint /health
Invoke-RestMethod -Uri http://localhost:8000/health -Method Get

# Test endpoint /predict
$body = @{
    gender           = "Female"
    SeniorCitizen    = 0
    Partner          = "Yes"
    Dependents       = "No"
    tenure           = 1
    PhoneService     = "No"
    MultipleLines    = "No phone service"
    InternetService  = "DSL"
    OnlineSecurity   = "No"
    OnlineBackup     = "Yes"
    DeviceProtection = "No"
    TechSupport      = "No"
    StreamingTV      = "No"
    StreamingMovies  = "No"
    Contract         = "Month-to-month"
    PaperlessBilling = "Yes"
    PaymentMethod    = "Electronic check"
    MonthlyCharges   = 29.85
    TotalCharges     = 29.85
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/predict `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

---

## Perintah Berguna Sehari-hari (PowerShell)

```powershell
# Aktifkan venv (wajib setiap buka terminal baru)
.\venv\Scripts\Activate.ps1

# Nonaktifkan venv
deactivate

# Lihat package yang terinstall
pip list

# Cek apakah server berjalan
Invoke-RestMethod http://localhost:8000/health

# Lihat container Docker yang berjalan
docker ps

# Stop semua container
docker compose down

# Hapus image lama dan build ulang
docker compose down
docker rmi churn-api:v1.0
docker build -t churn-api:v1.0 .
```

---

## Troubleshooting

| Error                                            | Penyebab                            | Solusi                                                         |
| ------------------------------------------------ | ----------------------------------- | -------------------------------------------------------------- |
| `cannot be loaded, running scripts is disabled`  | PowerShell blokir script            | Jalankan `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `PathNotFound: venv\Scripts\activate`            | Format perintah salah               | Gunakan `.\venv\Scripts\Activate.ps1` (pakai `.\` dan `.ps1`)  |
| `ModuleNotFoundError: No module named 'fastapi'` | venv belum aktif                    | Jalankan `.\venv\Scripts\Activate.ps1` dulu                    |
| `FileNotFoundError: model.pkl`                   | File belum di-copy ke `model\`      | Download hasil `colab_export.py` dan taruh di folder `model\`  |
| `Address already in use: port 8000`              | Port sudah dipakai proses lain      | Jalankan `uvicorn app.main:app --reload --port 8001`           |
| `docker: Cannot connect`                         | Docker Desktop belum jalan          | Buka Docker Desktop, tunggu sampai icon-nya hijau              |
| `422 Unprocessable Entity`                       | Field input salah / tipe data salah | Cek schema di http://localhost:8000/docs                       |
