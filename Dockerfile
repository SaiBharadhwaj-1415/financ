# ==========================================================
# Base Image
# ==========================================================
FROM python:3.11-slim

# ==========================================================
# Environment Setup
# ==========================================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=7860
ENV HOST=0.0.0.0

# ==========================================================
# Working Directory
# ==========================================================
WORKDIR /app

# ==========================================================
# Install System Dependencies
# ==========================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libxext6 libxrender1 \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# ==========================================================
# Install Python Dependencies
# ==========================================================
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================================
# Copy Application Code
# ==========================================================
COPY . .

# ==========================================================
# Start Script
# ==========================================================
RUN echo '#!/bin/bash' > /start.sh && \
    echo 'echo "🚀 Starting Financial AI Backend..."' >> /start.sh && \
    echo 'uvicorn main:app --host 0.0.0.0 --port 7860' >> /start.sh && \
    chmod +x /start.sh

# ==========================================================
# Expose Port & Run App
# ==========================================================
EXPOSE 7860
CMD ["/bin/bash", "/start.sh"]
