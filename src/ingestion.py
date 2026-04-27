"""
Download exclusivo do Supabase → data/raw/diabetes_raw.parquet.
Requer SUPABASE_URL e SUPABASE_KEY no .env.

Para popular a tabela a partir do CSV:
  python3 -m src.upload_supabase
"""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.schema import canonicalize_diabetes_df

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "diabetes_raw.parquet"


def _require_supabase_env() -> tuple[str, str, str]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    table = os.getenv("SUPABASE_DIABETES_TABLE", "diabetes")
    if not url or not key:
        raise SystemExit(
            "Ingestão usa apenas Supabase. Defina SUPABASE_URL e SUPABASE_KEY no .env.\n"
            "Para enviar o CSV à base: python3 -m src.upload_supabase"
        )
    return url, key, table


def fetch_from_supabase() -> pd.DataFrame:
    from supabase import create_client

    url, key, table = _require_supabase_env()
    client = create_client(url, key)
    res = client.table(table).select("*").execute()
    if not res.data:
        raise RuntimeError(
            f"Tabela '{table}' está vazia. Carregue dados com: python3 -m src.upload_supabase"
        )
    df = pd.DataFrame(res.data)
    return canonicalize_diabetes_df(df)


def main():
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Ingestão via Supabase…")
    df = fetch_from_supabase()
    df.to_parquet(RAW_PATH, index=False)
    print(f"Salvo: {RAW_PATH} ({len(df)} linhas)")


if __name__ == "__main__":
    main()
