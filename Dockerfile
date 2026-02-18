FROM python:3.12-slim

WORKDIR /app

# System deps for OpenCV (headless) and general runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/

# Data volume mount point
VOLUME /app/data

# Health check: verify the collector log was updated in the last 3 minutes
HEALTHCHECK --interval=120s --timeout=5s --retries=3 \
    CMD python -c "import os, time; f='/app/data/collector.log'; exit(0 if os.path.exists(f) and time.time()-os.path.getmtime(f)<180 else 1)"

ENTRYPOINT ["python", "-u", "collect.py"]
