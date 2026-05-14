# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps for SimpleITK (needs libGL on some platforms)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY pipeline.py app.py ./

# Optional: copy pre-built frontend (see README)
# COPY frontend/dist ./frontend/dist

# Jobs stored here — mount a volume in production for persistence
RUN mkdir -p /app/jobs

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", \
     "--limit-max-requests", "1000", \
     "--timeout-keep-alive", "30"]