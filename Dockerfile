# Energy Demand Forecasting API
# Build:  docker build -t energy-api .
# Run:    docker run --rm -p 8000:8000 energy-api        then open http://localhost:8000/docs
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Dependencies first, so this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Source code and the trained model (the API does not need the data or MLflow).
COPY src/ src/
COPY models/ models/

# Run as a normal user, not root.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# `docker ps` shows (healthy) once /health answers. Uses Python because the slim image has no curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=3)"

# Platforms such as Render set $PORT (10000 by default); locally it falls back to 8000.
CMD ["sh", "-c", "exec uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]
