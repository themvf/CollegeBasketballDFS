FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

COPY apps/api/requirements.deploy.txt /tmp/requirements.deploy.txt

RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.deploy.txt

COPY src /app/src
COPY apps/api /app/apps/api

WORKDIR /app/apps/api

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
