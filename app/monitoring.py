# =============================================================
# app/monitoring.py  —  Phase 5: Observability
# =============================================================
#
# Menambahkan Prometheus metrics ke FastAPI.
# Import dan pakai di main.py untuk expose /metrics endpoint.
#
# Metrics yang di-track:
#   churn_predictions_total     → counter: berapa prediksi sudah dibuat
#   churn_probability_histogram → distribusi skor probabilitas
#   request_latency_seconds     → latency tiap endpoint
#   predicted_churn_rate        → gauge: % customer diprediksi churn
# =============================================================

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time

# ------------------------------------------------------------------
# Definisi metrics
# ------------------------------------------------------------------

# Counter: terus naik, tidak bisa turun
PREDICTIONS_TOTAL = Counter(
    'churn_predictions_total',
    'Total prediksi yang telah dilakukan',
    ['result']   # label: "churn" atau "no_churn"
)

# Histogram: distribusi nilai — cocok untuk probabilitas & latency
CHURN_PROB_HISTOGRAM = Histogram(
    'churn_probability',
    'Distribusi skor probabilitas churn',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

LATENCY_HISTOGRAM = Histogram(
    'request_latency_seconds',
    'Latency request per endpoint',
    ['endpoint'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# Gauge: nilai yang bisa naik turun (realtime)
CHURN_RATE_GAUGE = Gauge(
    'predicted_churn_rate',
    'Persentase customer diprediksi churn (rolling 1000 request terakhir)'
)

# ------------------------------------------------------------------
# State sederhana untuk rolling churn rate
# ------------------------------------------------------------------
_recent_predictions = []
MAX_WINDOW = 1000


def record_prediction(prob: float, is_churn: bool):
    """Panggil ini setiap kali ada prediksi baru."""
    global _recent_predictions

    # Update counter
    label = "churn" if is_churn else "no_churn"
    PREDICTIONS_TOTAL.labels(result=label).inc()

    # Update histogram probabilitas
    CHURN_PROB_HISTOGRAM.observe(prob)

    # Update rolling churn rate
    _recent_predictions.append(int(is_churn))
    if len(_recent_predictions) > MAX_WINDOW:
        _recent_predictions.pop(0)

    rate = sum(_recent_predictions) / len(_recent_predictions)
    CHURN_RATE_GAUGE.set(rate)


def track_latency(endpoint: str):
    """Context manager untuk ukur latency."""
    class LatencyTracker:
        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *args):
            elapsed = time.time() - self.start
            LATENCY_HISTOGRAM.labels(endpoint=endpoint).observe(elapsed)

    return LatencyTracker()


# ------------------------------------------------------------------
# Endpoint /metrics — dibaca oleh Prometheus scraper
# ------------------------------------------------------------------
def metrics_endpoint():
    """
    Tambahkan ke main.py:

        from app.monitoring import metrics_endpoint
        app.add_route("/metrics", metrics_endpoint)
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
