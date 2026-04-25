from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessResult:
    df_raw: pd.DataFrame
    df_clean: pd.DataFrame
    X: pd.DataFrame
    y: Optional[pd.Series]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def norm(col: str) -> str:
        col = str(col).strip().lower()
        col = col.replace("-", "_").replace(" ", "_")
        col = col.replace("__", "_")
        return col

    df = df.copy()
    df.columns = [norm(c) for c in df.columns]

    # Kaggle file uses `no-show` (hyphen) which becomes `no_show` here.
    # Some versions may contain slight variations; map them defensively.
    rename_map = {
        "no_show": "no_show",
        "no_show_": "no_show",
        "noshow": "no_show",
        "no_show?": "no_show",
        "appointmentid": "appointment_id",
        "patientid": "patient_id",
        "scheduledday": "scheduled_day",
        "appointmentday": "appointment_day",
        "sms_received": "sms_received",
        "handcap": "handicap",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["scheduled_day", "appointment_day"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def _clean_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "age" not in df.columns:
        return df

    age = pd.to_numeric(df["age"], errors="coerce")
    # Dataset is messy: negative ages and implausible ages exist.
    age = age.where((age >= 0) & (age <= 115))

    # Fill missing ages with median (robust baseline; justify in report).
    median_age = float(np.nanmedian(age.to_numpy(dtype=float)))
    df["age"] = age.fillna(median_age).astype(int)
    return df


def _clean_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    df = df.copy()
    if "no_show" not in df.columns:
        return df, None

    # Kaggle: "No" means showed up, "Yes" means no-show.
    raw = df["no_show"].astype(str).str.strip().str.lower()
    y = raw.map({"yes": 1, "no": 0})
    df = df.drop(columns=["no_show"])
    return df, y


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "scheduled_day" in df.columns and "appointment_day" in df.columns:
        # Required feature: days between booking and appointment.
        delta = df["appointment_day"] - df["scheduled_day"]
        df["days_in_advance"] = delta.dt.total_seconds() / (24 * 3600)
        df["days_in_advance"] = df["days_in_advance"].round().astype("Int64")

        # Second feature (simple + effective): weekday of appointment.
        df["weekday"] = df["appointment_day"].dt.weekday.astype("Int64")

    return df


def load_raw_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_dataframe(df_raw: pd.DataFrame) -> PreprocessResult:
    df0 = df_raw.copy()

    df = _normalize_columns(df0)
    df = _parse_dates(df)
    df = _clean_age(df)
    df, y = _clean_target(df)
    df = _feature_engineering(df)

    # Minimal missing handling: keep datetimes, fill remaining numeric NaNs.
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=["number", "Int64", "float", "int"]).columns
    for c in numeric_cols:
        if df_clean[c].isna().any():
            df_clean[c] = df_clean[c].fillna(df_clean[c].median())

    X = df_clean
    return PreprocessResult(df_raw=df0, df_clean=df_clean, X=X, y=y)


def preprocess_csv(path: str) -> PreprocessResult:
    df_raw = load_raw_csv(path)
    return preprocess_dataframe(df_raw)

