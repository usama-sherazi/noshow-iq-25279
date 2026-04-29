---
title: NoShowIQ
sdk: docker
app_port: 8000
---

# NoShowIQ (noshow-iq-25279)

NoShowIQ is a prediction API for clinic appointment no-shows.

## Live
- Hugging Face Space: (add after deploy)

## CI
![ci-cd](https://github.com/usama-sherazi/noshow-iq-25279/actions/workflows/ci-cd.yml/badge.svg)

## Dataset
- `KaggleV2-May-2016.csv` (Medical Appointment No-Shows)

## Dev setup

```bash
python -m venv .venv
# activate venv (PowerShell)
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Run API (local)

```bash
uvicorn noshow_iq.api:app --reload --port 8000
```

## Run UI (local / deployed)
Open the dashboard at:
- `/` (root path)

## Endpoints
- `GET /health`
- `POST /predict`
- `GET /history`
- `GET /stats`

