import pandas as pd

from noshow_iq.model import evaluate, train


def test_evaluate_returns_both_class_metrics():
    metrics = evaluate([0, 0, 1, 1], [0, 1, 1, 1])
    assert 0.0 <= metrics.precision_show <= 1.0
    assert 0.0 <= metrics.recall_show <= 1.0
    assert 0.0 <= metrics.f1_show <= 1.0
    assert 0.0 <= metrics.precision_no_show <= 1.0
    assert 0.0 <= metrics.recall_no_show <= 1.0
    assert 0.0 <= metrics.f1_no_show <= 1.0


def test_train_smoke_dataframe(tmp_path):
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
    out = train(df, model_path=tmp_path / "model.joblib")
    assert "metrics" in out
    assert (tmp_path / "model.joblib").exists()
