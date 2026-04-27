from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from src.schema import FEATURE_COLUMNS, TARGET_COL

BASE_DIR = Path(__file__).resolve().parent.parent
INVALID_ZERO_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

PROCESSED_PARQUET = BASE_DIR / "data" / "processed" / "features.parquet"
LEGACY_CSV = BASE_DIR / "data" / "diabetes.csv"


class InvalidZeroToNaN(BaseEstimator, TransformerMixin):
    """Zeros em colunas clínicas inválidas viram NaN; saída numérica para os próximos passos."""

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            self.feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)
        return self

    def transform(self, X):
        if hasattr(X, "columns"):
            cols = [c for c in self.feature_names_in_ if c in X.columns]
            df = X[cols].copy()
        else:
            df = pd.DataFrame(X, columns=self.feature_names_in_)
        for c in INVALID_ZERO_COLS:
            if c in df.columns:
                df[c] = df[c].replace(0, np.nan)
        return df.astype(np.float64).to_numpy()


def default_dataset_path() -> Path:
    if PROCESSED_PARQUET.is_file():
        return PROCESSED_PARQUET
    return LEGACY_CSV


def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    p = Path(path) if path else default_dataset_path()
    if not p.is_file():
        raise FileNotFoundError(
            f"Dataset não encontrado: {p}. Rode `python3 -m src.ingestion` e `python3 -m src.preprocessing` "
            "ou coloque data/processed/features.parquet."
        )
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def load_and_split(
    path: str | Path | None = None,
    test_size: float = 0.2,
    seed: int = 42,
):
    df = load_dataset(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Coluna alvo '{TARGET_COL}' ausente no dataset.")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COL]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
    print(y_train.value_counts())
