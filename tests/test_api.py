from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from noshow_iq import api as api_module
from noshow_iq.model import train


def _train_tmp_model(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        [
            {
                "PatientId": 1,
                "AppointmentID": 10,
                "Gender": "F",
                "ScheduledDay": "2016-04-29T18:38:08Z",
                "AppointmentDay": "2016-04-30T00:00:00Z",
                "Age": 22,
                "Neighbourhood": "JARDIM DA PENHA",
                "Scholarship": 0,
                "Hipertension": 0,
                "Diabetes": 0,
                "Alcoholism": 0,
                "Handcap": 0,
                "SMS_received": 1,
                "No-show": "No",
            },
            {
                "PatientId": 2,
                "AppointmentID": 11,
                "Gender": "M",
                "ScheduledDay": "2016-04-28T10:00:00Z",
                "AppointmentDay": "2016-05-10T00:00:00Z",
                "Age": 50,
                "Neighbourhood": "CENTRO",
                "Scholarship": 1,
                "Hipertension": 1,
                "Diabetes": 0,
                "Alcoholism": 0,
                "Handcap": 0,
                "SMS_received": 0,
                "No-show": "Yes",
            },
            {
                "PatientId": 3,
                "AppointmentID": 12,
                "Gender": "F",
                "ScheduledDay": "2016-04-28T10:00:00Z",
                "AppointmentDay": "2016-04-29T00:00:00Z",
                "Age": 35,
                "Neighbourhood": "CENTRO",
                "Scholarship": 0,
                "Hipertension": 0,
                "Diabetes": 1,
                "Alcoholism": 1,
                "Handcap": 0,
                "SMS_received": 0,
                "No-show": "No",
            },
            {
                "PatientId": 4,
                "AppointmentID": 13,
                "Gender": "M",
                "ScheduledDay": "2016-04-28T10:00:00Z",
                "AppointmentDay": "2016-05-01T00:00:00Z",
                "Age": 70,
                "Neighbourhood": "CENTRO",
                "Scholarship": 0,
                "Hipertension": 1,
                "Diabetes": 1,
                "Alcoholism": 0,
                "Handcap": 0,
                "SMS_received": 1,
                "No-show": "Yes",
            },
        ]
    )
    model_path = tmp_path / "model.joblib"
    train(df, model_path=model_path)
    return model_path


def test_health_ok():
    client = TestClient(api_module.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_ok(tmp_path: Path, monkeypatch):
    model_path = _train_tmp_model(tmp_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    client = TestClient(api_module.app)
    payload = {
        "ScheduledDay": "2016-04-29T18:38:08Z",
        "AppointmentDay": "2016-04-30T00:00:00Z",
        "Age": 22,
        "Gender": "F",
        "Neighbourhood": "JARDIM DA PENHA",
        "SMS_received": 1,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"risk", "probability", "recommendation"}
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_validation_error():
    client = TestClient(api_module.app)
    r = client.post("/predict", json={"Age": 10})
    assert r.status_code == 422


def test_predict_model_missing_returns_503(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(Path("does_not_exist.joblib")))
    client = TestClient(api_module.app)
    payload = {
        "ScheduledDay": "2016-04-29T18:38:08Z",
        "AppointmentDay": "2016-04-30T00:00:00Z",
        "Age": 22,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 503


def test_history_returns_list(tmp_path: Path, monkeypatch):
    model_path = _train_tmp_model(tmp_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    client = TestClient(api_module.app)

    payload = {
        "ScheduledDay": "2016-04-29T18:38:08Z",
        "AppointmentDay": "2016-04-30T00:00:00Z",
        "Age": 22,
    }
    _ = client.post("/predict", json=payload)

    r = client.get("/history")
    assert r.status_code == 200
    body = r.json()
    assert "items" in body
    assert isinstance(body["items"], list)


def test_stats_has_expected_keys():
    client = TestClient(api_module.app)
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {
        "total_predictions",
        "high_risk_count",
        "low_risk_count",
        "average_probability",
        "last_trained",
    }
