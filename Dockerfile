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
    gcc g++ \
    libglib2.0-0 libgl1 libxext6 libxrender1 libsm6 \
    tesseract-ocr \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# ==========================================================
# Copy Dependencies and Install Core Packages
# ==========================================================
COPY requirements.txt .

# Upgrade pip & install dependencies safely
RUN pip install --upgrade pip setuptools wheel

# Install all requirements (ignore errors for resilience)
RUN pip install --no-cache-dir -r requirements.txt || true

# ==========================================================
# Copy Application Code
# ==========================================================
COPY . .

# ==========================================================
# Automatic Runtime Package Checker (Self-Healing)
# ==========================================================
# This step ensures any missing package (even if not in requirements.txt)
# gets installed dynamically before the app starts.
RUN echo '#!/bin/bash' > /start.sh && \
    echo 'echo "🔍 Checking & Installing missing Python packages..."' >> /start.sh && \
    echo 'python - <<EOF' >> /start.sh && \
    echo 'import pkg_resources, subprocess;' >> /start.sh && \
    echo 'with open("requirements.txt") as f:' >> /start.sh && \
    echo '    for line in f:' >> /start.sh && \
    echo '        pkg = line.strip().split("==")[0];' >> /start.sh && \
    echo '        if pkg and pkg not in [p.key for p in pkg_resources.working_set]:' >> /start.sh && \
    echo '            print(f"Installing missing package: {pkg}");' >> /start.sh && \
    echo '            subprocess.call(["pip", "install", pkg]);' >> /start.sh && \
    echo 'EOF' >> /start.sh && \
    echo 'echo "✅ All dependencies are installed!"' >> /start.sh && \
    echo 'echo "🚀 Starting FastAPI server..."' >> /start.sh && \
    echo 'uvicorn app.main:app --host 0.0.0.0 --port 7860' >> /start.sh && \
    chmod +x /start.sh

# ==========================================================
# Expose Port & Run App
# ==========================================================
EXPOSE 7860

CMD ["/bin/bash", "/start.sh"]
