FROM python:3.11-slim

# System deps for SimpleITK
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY pipeline.py app.py index.html ./

# HuggingFace Spaces requires port 7860
ENV PORT=7860
ENV JOBS_ROOT=/app/jobs

# Create jobs directory
RUN mkdir -p /app/jobs

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]