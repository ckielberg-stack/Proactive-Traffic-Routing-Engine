FROM python:3.12-slim

WORKDIR /app

# System deps for OpenCV (headless) and general runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy canonical application code
COPY config.py main.py main_loop.py retention.py roi_helper.py ./
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/
COPY camera_config.json vms_config.json ./

# Data volume mount point
VOLUME /app/data

# Health check: verify the canonical app is serving requests
HEALTHCHECK --interval=120s --timeout=5s --retries=3 \
    CMD python -c "import json, urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3); exit(0 if r.status == 200 and json.load(r).get('status') == 'ok' else 1)"

ENTRYPOINT ["python", "-u", "main.py"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
