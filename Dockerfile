FROM python:3.11-slim AS builder

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md LICENSE requirements.txt ./
COPY noshow_iq ./noshow_iq

RUN python -m pip install --upgrade pip && \
    python -m pip install --prefix=/install -r requirements.txt && \
    python -m pip install --prefix=/install .


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --shell /usr/sbin/nologin appuser
WORKDIR /app

COPY --from=builder /install /usr/local
COPY noshow_iq ./noshow_iq
COPY artifacts ./artifacts

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn noshow_iq.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
