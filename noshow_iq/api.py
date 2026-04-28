from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from pymongo import MongoClient
from dotenv import load_dotenv

from noshow_iq.model import DEFAULT_MODEL_PATH, predict as model_predict
from noshow_iq.preprocess import preprocess_dataframe

load_dotenv()

app = FastAPI(title="NoShowIQ")

DEFAULT_DB_NAME = "noshow_iq"

def _mongo_client(mongo_uri: str) -> MongoClient:
    tls_insecure = os.getenv("MONGO_TLS_INSECURE", "").strip().lower() in {"1", "true", "yes"}
    if tls_insecure:
        return MongoClient(mongo_uri, tlsAllowInvalidCertificates=True)
    return MongoClient(mongo_uri)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime,)):
        return value.replace(microsecond=0).isoformat()
    if hasattr(value, "isoformat") and callable(getattr(value, "isoformat")):
        try:
            return value.isoformat()
        except Exception:
            pass
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _row_to_json_safe_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k): _json_safe(v) for k, v in row.items()}


def _risk_level(probability: float) -> str:
    # Simple, explainable thresholds. Tune later with real validation.
    if probability >= 0.7:
        return "high"
    if probability >= 0.4:
        return "medium"
    return "low"


def _recommendation(risk: str) -> str:
    if risk == "high":
        return "Call patient to confirm and consider overbooking."
    if risk == "medium":
        return "Send reminder message and confirm contact details."
    return "Standard reminder is sufficient."


class AppointmentIn(BaseModel):
    # Require the two key timestamps so we can compute days_in_advance/weekday.
    scheduled_day: datetime = Field(alias="ScheduledDay")
    appointment_day: datetime = Field(alias="AppointmentDay")

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class PredictOut(BaseModel):
    risk: str
    probability: float
    recommendation: str


class PredictionStore:
    def insert_prediction(self, doc: Dict[str, Any]) -> None:  # pragma: no cover
        raise NotImplementedError

    def history(self, limit: int = 20) -> List[Dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def stats(self) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


class MongoPredictionStore(PredictionStore):
    def __init__(self, mongo_uri: str, db_name: str = DEFAULT_DB_NAME) -> None:
        self._client = _mongo_client(mongo_uri)
        self._db = self._client[db_name]
        self._predictions = self._db["predictions"]

    def insert_prediction(self, doc: Dict[str, Any]) -> None:
        self._predictions.insert_one(doc)

    def history(self, limit: int = 20) -> List[Dict[str, Any]]:
        cursor = (
            self._predictions.find({}, sort=[("timestamp", -1)], limit=int(limit))
        )
        items: List[Dict[str, Any]] = []
        for d in cursor:
            d["_id"] = str(d.get("_id"))
            items.append(d)
        return items

    def stats(self) -> Dict[str, Any]:
        pipeline = [
            {
                "$facet": {
                    "pred": [
                        {
                            "$group": {
                                "_id": None,
                                "total_predictions": {"$sum": 1},
                                "high_risk_count": {
                                    "$sum": {"$cond": [{"$eq": ["$risk", "high"]}, 1, 0]}
                                },
                                "low_risk_count": {
                                    "$sum": {"$cond": [{"$eq": ["$risk", "low"]}, 1, 0]}
                                },
                                "average_probability": {"$avg": "$probability"},
                            }
                        }
                    ],
                    "train": [
                        {
                            "$lookup": {
                                "from": "training_runs",
                                "let": {},
                                "pipeline": [
                                    {"$sort": {"timestamp": -1}},
                                    {"$limit": 1},
                                    {"$project": {"_id": 0, "timestamp": 1}},
                                ],
                                "as": "last",
                            }
                        },
                        {"$unwind": {"path": "$last", "preserveNullAndEmptyArrays": True}},
                        {"$project": {"_id": 0, "last_trained": "$last.timestamp"}},
                        {"$limit": 1},
                    ],
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "pred": {"$first": "$pred"},
                    "last_trained": {"$first": "$train.last_trained"},
                }
            },
            {
                "$project": {
                    "total_predictions": {"$ifNull": ["$pred.total_predictions", 0]},
                    "high_risk_count": {"$ifNull": ["$pred.high_risk_count", 0]},
                    "low_risk_count": {"$ifNull": ["$pred.low_risk_count", 0]},
                    "average_probability": {"$ifNull": ["$pred.average_probability", 0.0]},
                    "last_trained": "$last_trained",
                }
            },
        ]
        out = list(self._predictions.aggregate(pipeline, allowDiskUse=False))
        if not out:
            return {
                "total_predictions": 0,
                "high_risk_count": 0,
                "low_risk_count": 0,
                "average_probability": 0.0,
                "last_trained": None,
            }
        return out[0]


class InMemoryPredictionStore(PredictionStore):
    def __init__(self) -> None:
        self._items: List[Dict[str, Any]] = []

    def insert_prediction(self, doc: Dict[str, Any]) -> None:
        self._items.append(doc)

    def history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(reversed(self._items[-int(limit) :]))

    def stats(self) -> Dict[str, Any]:
        return {
            "total_predictions": len(self._items),
            "high_risk_count": 0,
            "low_risk_count": 0,
            "average_probability": 0.0,
            "last_trained": None,
        }


_MEM_STORE = InMemoryPredictionStore()


def get_store() -> PredictionStore:
    mongo_uri = os.getenv("MONGO_URI")
    if mongo_uri:
        db_name = os.getenv("MONGO_DB_NAME", DEFAULT_DB_NAME)
        return MongoPredictionStore(mongo_uri, db_name=db_name)
    return _MEM_STORE


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictOut)
def predict(payload: AppointmentIn, store: PredictionStore = Depends(get_store)):
    model_path = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    raw = payload.model_dump(by_alias=True)

    try:
        result = model_predict(raw, model_path=model_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model artifact not found. Train the model first.",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    probability = float(result["probability"])
    risk = _risk_level(probability)
    recommendation = _recommendation(risk)

    cleaned_features: Dict[str, Any] = {}
    try:
        prep = preprocess_dataframe(pd.DataFrame([raw]))
        cleaned_features = _row_to_json_safe_dict(prep.df_clean.iloc[0].to_dict())
    except Exception:
        cleaned_features = {}

    doc = {
        "timestamp": _utc_now_iso(),
        "raw_input": raw,
        "cleaned_features": cleaned_features,
        "risk": risk,
        "probability": probability,
        "recommendation": recommendation,
    }
    try:
        store.insert_prediction(doc)
    except Exception:
        # Keep API responsive even if Mongo is down; Day 4 will harden this.
        pass

    return PredictOut(risk=risk, probability=probability, recommendation=recommendation)


@app.get("/history")
def history(store: PredictionStore = Depends(get_store)):
    return {"items": store.history(limit=20)}


@app.get("/stats")
def stats(store: PredictionStore = Depends(get_store)):
    return store.stats()
