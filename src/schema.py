"""Nomes canónicos das colunas (alinhados ao CSV Pima) e conversão desde/para o Supabase."""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COL = "Outcome"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COL]

# Chave normalizada (minúsculas, sem underscores) -> nome canónico
_NORM_TO_CANONICAL = {
    "pregnancies": "Pregnancies",
    "glucose": "Glucose",
    "bloodpressure": "BloodPressure",
    "skinthickness": "SkinThickness",
    "insulin": "Insulin",
    "bmi": "BMI",
    "diabetespedigreefunction": "DiabetesPedigreeFunction",
    "age": "Age",
    "outcome": "Outcome",
}

_CANONICAL_TO_SNAKE = {
    "Pregnancies": "pregnancies",
    "Glucose": "glucose",
    "BloodPressure": "blood_pressure",
    "SkinThickness": "skin_thickness",
    "Insulin": "insulin",
    "BMI": "bmi",
    "DiabetesPedigreeFunction": "diabetes_pedigree_function",
    "Age": "age",
    "Outcome": "outcome",
}


def _norm_key(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def canonicalize_diabetes_df(df: pd.DataFrame) -> pd.DataFrame:
    """Aceita colunas em PascalCase (CSV) ou snake_case (PostgREST típico)."""
    rename = {}
    for c in df.columns:
        nk = _norm_key(c)
        if nk in _NORM_TO_CANONICAL:
            rename[c] = _NORM_TO_CANONICAL[nk]
        elif c in ALL_COLUMNS:
            rename[c] = c
        else:
            rename[c] = c
    out = df.rename(columns=rename)
    missing = [c for c in ALL_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Colunas em falta após normalização: {missing}. Colunas recebidas: {list(df.columns)}")
    return out[ALL_COLUMNS].copy()


def to_snake_case_records(df: pd.DataFrame) -> list[dict]:
    """Registos para INSERT no Postgres com nomes em snake_case (tabela sem aspas)."""
    df = canonicalize_diabetes_df(df)
    df_snake = df.rename(columns=_CANONICAL_TO_SNAKE)
    rows = df_snake.to_dict(orient="records")
    # JSON não aceita numpy types
    clean = []
    for row in rows:
        clean.append({k: (int(v) if k == "outcome" and pd.notna(v) else float(v) if pd.notna(v) else None) for k, v in row.items()})
    return clean
