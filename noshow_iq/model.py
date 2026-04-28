from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from noshow_iq.preprocess import preprocess_csv, preprocess_dataframe

DEFAULT_DB_NAME = "noshow_iq"
DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")


@dataclass(frozen=True)
class EvalMetrics:
    precision_show: float
    recall_show: float
    f1_show: float
    precision_no_show: float
    recall_no_show: float
    f1_no_show: float


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _pick_feature_columns(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    # Keep a stable, explainable set of features; only use columns that exist.
    numeric_candidates = [
        "age",
        "scholarship",
        "hipertension",
        "hypertension",  # defensive
        "diabetes",
        "alcoholism",
        "handicap",
        "sms_received",
        "days_in_advance",
        "weekday",
    ]
    categorical_candidates = [
        "gender",
        "neighbourhood",
        "neighborhood",  # defensive
    ]

    cols = set(df.columns.astype(str))
    numeric = [c for c in numeric_candidates if c in cols]
    categorical = [c for c in categorical_candidates if c in cols]
    return numeric, categorical


def _build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # handles the 80/20 imbalance baseline
        n_jobs=None,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])


def save_model(model: Pipeline, path: Path | str = DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    _ensure_parent_dir(path)
    joblib.dump(model, path)
    return path


def load_model(path: Path | str = DEFAULT_MODEL_PATH) -> Pipeline:
    return joblib.load(Path(path))


def evaluate(y_true: Iterable[int], y_pred: Iterable[int]) -> EvalMetrics:
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)

    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    # label 0 = show (No), label 1 = no-show (Yes)
    return EvalMetrics(
        precision_show=float(precision[0]),
        recall_show=float(recall[0]),
        f1_show=float(f1[0]),
        precision_no_show=float(precision[1]),
        recall_no_show=float(recall[1]),
        f1_no_show=float(f1[1]),
    )


def _get_mongo_collection(mongo_uri: str, db_name: str = DEFAULT_DB_NAME):
    tls_insecure = os.getenv("MONGO_TLS_INSECURE", "").strip().lower() in {"1", "true", "yes"}
    if tls_insecure:
        client = MongoClient(mongo_uri, tlsAllowInvalidCertificates=True)
    else:
        client = MongoClient(mongo_uri)
    return client[db_name]["training_runs"]


def log_training_run(
    *,
    mongo_uri: Optional[str],
    training_size: int,
    metrics: EvalMetrics,
    imbalance_technique: str,
    db_name: str = DEFAULT_DB_NAME,
) -> bool:
    uri = mongo_uri or os.getenv("MONGO_URI")
    if not uri:
        return False

    doc: Dict[str, Any] = {
        "timestamp": _utc_now_iso(),
        "training_size": int(training_size),
        "metrics": asdict(metrics),
        "imbalance_technique": imbalance_technique,
    }

    try:
        col = _get_mongo_collection(uri, db_name=db_name)
        col.insert_one(doc)
        return True
    except Exception:
        # Don't hard-fail local training if Mongo isn't available yet.
        return False


def train(
    data: str | Path | pd.DataFrame,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    mongo_uri: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    if isinstance(data, (str, Path)):
        prep = preprocess_csv(str(data))
    else:
        prep = preprocess_dataframe(data)

    if prep.y is None:
        raise ValueError("Target column missing; expected a no-show column in the dataset.")

    df = prep.df_clean
    y = prep.y.astype(int)

    numeric_features, categorical_features = _pick_feature_columns(df)
    if not numeric_features and not categorical_features:
        raise ValueError("No usable feature columns found after preprocessing.")

    X = df[numeric_features + categorical_features].copy()

    if y.nunique() < 2:
        raise ValueError("Need at least 2 classes to train a binary classifier.")

    # Stratified split is ideal, but tiny samples (or very rare minority class)
    # can make it impossible; fall back to non-stratified split.
    class_counts = y.value_counts()
    can_stratify = bool(class_counts.min() >= 2)
    # StratifiedShuffleSplit also requires at least 1 sample per class in test.
    n_classes = int(y.nunique())
    n_samples = int(len(y))
    n_test = int(round(float(test_size) * n_samples))
    if n_test < n_classes:
        can_stratify = False
    stratify = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    model = _build_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    saved_path = save_model(model, model_path)

    _logged = log_training_run(
        mongo_uri=mongo_uri,
        training_size=int(len(X_train)),
        metrics=metrics,
        imbalance_technique="class_weight=balanced",
    )

    return {
        "model_path": str(saved_path),
        "metrics": asdict(metrics),
        "training_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "mongo_logged": bool(_logged),
        "features": {"numeric": numeric_features, "categorical": categorical_features},
    }


def predict(
    record: Dict[str, Any] | pd.DataFrame,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    model = load_model(model_path)

    if isinstance(record, dict):
        df_raw = pd.DataFrame([record])
    else:
        df_raw = record.copy()

    prep = preprocess_dataframe(df_raw)
    df = prep.df_clean

    # Ensure inference uses the same columns the model was trained on.
    trained_cols = getattr(model, "feature_names_in_", None)
    if trained_cols is not None:
        X = df.reindex(columns=list(trained_cols), fill_value=np.nan)
    else:
        numeric_features, categorical_features = _pick_feature_columns(df)
        X = df[numeric_features + categorical_features].copy()

    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)

    return {"probability": proba, "predicted_class": pred}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train NoShowIQ model and save artifacts.")
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to Kaggle CSV (KaggleV2-May-2016.csv).",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Where to save joblib model.",
    )
    parser.add_argument("--mongo-uri", default=None, help="Mongo URI (or set MONGO_URI env var).")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args(argv)

    result = train(
        args.csv,
        model_path=args.model_path,
        mongo_uri=args.mongo_uri,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
