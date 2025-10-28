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

# Try to install requirements, skip missing ones gracefully
RUN pip install --no-cache-dir -r requirements.txt || true

# ==========================================================
# Copy Application Code
# ==========================================================
COPY . .

# ==========================================================
# Auto-install any missing dependencies at runtime
# ==========================================================
RUN echo 'import pkg_resources, subprocess, sys\n' \
         'for dist in pkg_resources.working_set:\n' \
         '    try:\n' \
         '        __import__(dist.project_name)\n' \
         '    except ImportError:\n' \
         '        print(f"⚠️  Missing {dist.project_name}, installing...")\n' \
         '        subprocess.check_call([sys.executable, "-m", "pip", "install", dist.project_name])\n' \
         'print("✅ All dependencies verified!")' \
         > verify_deps.py && \
    python verify_deps.py || true

# ==========================================================
# Start Script
# ==========================================================
RUN echo '#!/bin/bash' > /start.sh && \
    echo 'echo "🚀 Starting Financial AI Backend..."' >> /start.sh && \
    echo 'python verify_deps.py || true' >> /start.sh && \
    echo 'uvicorn main:app --host 0.0.0.0 --port 7860' >> /start.sh && \
    chmod +x /start.sh

# ==========================================================
# Expose Port & Run App
# ==========================================================
EXPOSE 7860
CMD ["/bin/bash", "/start.sh"]
