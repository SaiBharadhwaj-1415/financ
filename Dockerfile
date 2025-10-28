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
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel

# Try to install from requirements, skip missing gracefully
RUN pip install --no-cache-dir -r requirements.txt || true

# Ensure python-multipart exists
RUN pip install --no-cache-dir python-multipart

# ==========================================================
# Copy Application Code
# ==========================================================
COPY . .

# ==========================================================
# Auto-install missing deps at runtime
# ==========================================================
RUN printf 'import importlib, subprocess, sys\n' \
           'print("🔍 Checking dependencies...")\n' \
           'missing=[]\n' \
           'reqs=subprocess.getoutput("pip freeze").splitlines()\n' \
           'for name in ["fastapi", "uvicorn", "python-multipart"]:\n' \
           '    try:\n' \
           '        importlib.import_module(name)\n' \
           '    except ImportError:\n' \
           '        print(f"⚠️ Missing {name}, installing...")\n' \
           '        subprocess.check_call([sys.executable, "-m", "pip", "install", name])\n' \
           'print("✅ Dependencies ready!")\n' \
           > verify_deps.py

# ==========================================================
# Start Script
# ==========================================================
RUN echo '#!/bin/bash' > /start.sh && \
    echo 'echo "🚀 Starting Financial AI Backend..."' >> /start.sh && \
    echo 'python verify_deps.py || true' >> /start.sh && \
    echo 'exec uvicorn main:app --host 0.0.0.0 --port 7860' >> /start.sh && \
    chmod +x /start.sh

# ==========================================================
# Expose Port & Run App
# ==========================================================
EXPOSE 7860
CMD ["/bin/bash", "/start.sh"]
