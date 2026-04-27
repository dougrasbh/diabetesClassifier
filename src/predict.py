"""Inferência local a partir de model.pkl (útil para scripts e testes)."""
from pathlib import Path

import joblib
import pandas as pd


def load_model(path: Path | str | None = None):
    root = Path(__file__).resolve().parent.parent
    p = Path(path) if path else root / "model.pkl"
    if not p.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {p}")
    return joblib.load(p)


def predict_proba(X: pd.DataFrame, model=None):
    model = model or load_model()
    return model.predict_proba(X)


def predict(X: pd.DataFrame, model=None):
    model = model or load_model()
    return model.predict(X)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python -m src.predict <caminho.csv com as 8 features>")
        sys.exit(1)
    df = pd.read_csv(sys.argv[1])
    m = load_model()
    print(m.predict(df))
