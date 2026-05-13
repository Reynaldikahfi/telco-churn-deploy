# =============================================================
# Dockerfile  —  Phase 3: Containerization
# =============================================================
#
# Build:  docker build -t churn-api:v1.0 .
# Run:    docker run -p 8000:8000 churn-api:v1.0
#
# Multi-stage build:
#   Stage 1 (builder): install dependencies di layer terpisah
#   Stage 2 (runtime): copy hanya yang diperlukan → image lebih kecil
# =============================================================

# ── Stage 1: builder ──────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install dependencies dulu (di-cache jika requirements.txt tidak berubah)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Non-root user untuk keamanan
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Copy installed packages dari builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy kode dan model
COPY app/     ./app/
COPY model/   ./model/

# Ganti ke non-root
USER appuser

# Expose port API
EXPOSE 8000

# Health check — Docker akan restart container jika /health tidak OK
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Jalankan server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
