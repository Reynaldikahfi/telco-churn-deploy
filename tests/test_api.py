# =============================================================
# tests/test_api.py  —  Unit & Integration Tests
# =============================================================
#
# Cara jalankan:
#   pytest tests/ -v
#   pytest tests/ -v --tb=short   (output lebih ringkas)
#
# Test ini menggunakan TestClient FastAPI — tidak perlu server aktif.
# =============================================================

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Tambahkan root project ke sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ------------------------------------------------------------------
# Fixture: data customer valid untuk testing
# ------------------------------------------------------------------
VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}

# ------------------------------------------------------------------
# Import app (akan load model — pastikan model/ folder ada)
# ------------------------------------------------------------------
try:
    from app.main import app
    client = TestClient(app)
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model artifact tidak ada")
class TestHealthEndpoints:

    def test_root_returns_200(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "api" in resp.json()

    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_metadata_has_threshold(self):
        resp = client.get("/metadata")
        assert resp.status_code == 200
        data = resp.json()
        assert "threshold" in data
        assert 0 < data["threshold"] < 1


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model artifact tidak ada")
class TestPredictEndpoint:

    def test_valid_customer_returns_prediction(self):
        resp = client.post("/predict", json=VALID_CUSTOMER)
        assert resp.status_code == 200
        data = resp.json()
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "risk_label" in data
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_risk_label_valid_values(self):
        resp = client.post("/predict", json=VALID_CUSTOMER)
        assert resp.json()["risk_label"] in ["High", "Medium", "Low"]

    def test_invalid_gender_returns_422(self):
        bad = {**VALID_CUSTOMER, "gender": "Unknown"}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422   # Unprocessable Entity

    def test_missing_required_field_returns_422(self):
        bad = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_total_charges_none_is_handled(self):
        """TotalCharges boleh kosong — harus diisi otomatis dengan median."""
        no_tc = {**VALID_CUSTOMER, "TotalCharges": None}
        resp = client.post("/predict", json=no_tc)
        assert resp.status_code == 200

    def test_batch_predict(self):
        batch = {"customers": [VALID_CUSTOMER, VALID_CUSTOMER]}
        resp = client.post("/predict/batch", json=batch)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_customers"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_limit_exceeded(self):
        batch = {"customers": [VALID_CUSTOMER] * 501}
        resp = client.post("/predict/batch", json=batch)
        assert resp.status_code == 400


# ------------------------------------------------------------------
# Test preprocessing (tidak perlu model)
# ------------------------------------------------------------------
class TestPreprocessingLogic:

    def test_valid_customer_dict_structure(self):
        """Pastikan schema VALID_CUSTOMER punya semua field yang diperlukan."""
        required = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService",
            "MonthlyCharges", "TotalCharges"
        ]
        for field in required:
            assert field in VALID_CUSTOMER, f"Field '{field}' hilang dari test data"

    def test_monthly_charges_is_positive(self):
        assert VALID_CUSTOMER["MonthlyCharges"] > 0

    def test_tenure_is_non_negative(self):
        assert VALID_CUSTOMER["tenure"] >= 0
