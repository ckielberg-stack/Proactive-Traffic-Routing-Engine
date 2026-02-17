FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py collect.py discover_cameras.py dashboard.py ./
COPY static/ ./static/

# Data volume mount point
VOLUME /app/data

# Health check: verify the collector log was updated in the last 3 minutes
HEALTHCHECK --interval=120s --timeout=5s --retries=3 \
    CMD python -c "import os, time; f='/app/data/collector.log'; exit(0 if os.path.exists(f) and time.time()-os.path.getmtime(f)<180 else 1)"

ENTRYPOINT ["python", "-u", "collect.py"]
