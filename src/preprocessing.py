"""
Transformações analíticas com DuckDB sobre o snapshot bruto → data/processed/features.parquet.
"""
from pathlib import Path

import duckdb
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PARQUET = BASE_DIR / "data" / "raw" / "diabetes_raw.parquet"
RAW_CSV = BASE_DIR / "data" / "raw" / "diabetes_raw.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "features.parquet"


def main():
    if RAW_PARQUET.is_file():
        from_clause = f"read_parquet('{RAW_PARQUET.as_posix()}')"
    elif RAW_CSV.is_file():
        from_clause = f"read_csv_auto('{RAW_CSV.as_posix()}')"
    else:
        raise FileNotFoundError(
            f"Arquivo bruto não encontrado. Esperado {RAW_PARQUET} ou {RAW_CSV}. "
            "Rode: python -m src.ingestion"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    sql = f"""
    COPY (
        SELECT
            CAST(Pregnancies AS DOUBLE) AS Pregnancies,
            CAST(Glucose AS DOUBLE) AS Glucose,
            CAST(BloodPressure AS DOUBLE) AS BloodPressure,
            CAST(SkinThickness AS DOUBLE) AS SkinThickness,
            CAST(Insulin AS DOUBLE) AS Insulin,
            CAST(BMI AS DOUBLE) AS BMI,
            CAST(DiabetesPedigreeFunction AS DOUBLE) AS DiabetesPedigreeFunction,
            CAST(Age AS DOUBLE) AS Age,
            CAST(Outcome AS INTEGER) AS Outcome
        FROM {from_clause}
        WHERE Glucose IS NOT NULL
    ) TO '{PROCESSED_PATH.as_posix()}' (FORMAT PARQUET);
    """
    con.execute(sql)
    con.close()

    n = len(pd.read_parquet(PROCESSED_PATH))
    print(f"Processado com DuckDB → {PROCESSED_PATH} ({n} linhas)")


if __name__ == "__main__":
    main()
