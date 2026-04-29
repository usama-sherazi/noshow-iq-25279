import sys
from typing import Any, Dict

import httpx


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def _ok(msg: str) -> None:
    print(f"OK: {msg}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        return _fail("Usage: python smoke_test.py <base_url>")

    base_url = argv[1].rstrip("/")
    timeout = 20.0

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        r = client.get(f"{base_url}/health")
        if r.status_code != 200 or r.json().get("status") != "ok":
            return _fail(f"/health failed: {r.status_code} {r.text}")
        _ok("/health")

        payload: Dict[str, Any] = {
            "ScheduledDay": "2016-04-29T18:38:08Z",
            "AppointmentDay": "2016-04-30T00:00:00Z",
            "Age": 22,
            "Gender": "F",
            "Neighbourhood": "JARDIM DA PENHA",
            "SMS_received": 1,
        }
        r = client.post(f"{base_url}/predict", json=payload)
        if r.status_code != 200:
            return _fail(f"/predict failed: {r.status_code} {r.text}")
        body = r.json()
        if not {"risk", "probability", "recommendation"} <= set(body.keys()):
            return _fail(f"/predict unexpected response: {body}")
        _ok("/predict")

        r = client.get(f"{base_url}/stats")
        if r.status_code != 200:
            return _fail(f"/stats failed: {r.status_code} {r.text}")
        s = r.json()
        required = {
            "total_predictions",
            "high_risk_count",
            "low_risk_count",
            "average_probability",
            "last_trained",
        }
        if not required <= set(s.keys()):
            return _fail(f"/stats missing keys: {s}")
        _ok("/stats")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
