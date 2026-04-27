"""
Carrega um CSV (formato Pima Indians) e insere as linhas na tabela do Supabase.

Requer tabela em snake_case (ver scripts/create_diabetes_table.sql).
Uso:
  python3 -m src.upload_supabase
  python3 -m src.upload_supabase --csv caminho/dados.csv --replace
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.schema import canonicalize_diabetes_df, to_snake_case_records

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = BASE_DIR / "data" / "diabetes.csv"
BATCH_SIZE = 300


def _require_env() -> tuple[str, str, str]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    table = os.getenv("SUPABASE_DIABETES_TABLE", "diabetes")
    if not url or not key:
        raise SystemExit(
            "Defina SUPABASE_URL e SUPABASE_KEY no .env (recomenda-se a service_role para insert)."
        )
    return url, key, table


def _delete_all(client, table: str) -> None:
    """Apaga todas as linhas (Outcome só é 0 ou 1 no dataset)."""
    client.table(table).delete().neq("outcome", 999).execute()


def main():
    parser = argparse.ArgumentParser(description="CSV → Supabase (tabela diabetes)")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Caminho do CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Apaga todas as linhas da tabela antes de inserir",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV não encontrado: {args.csv}")

    from supabase import create_client

    url, key, table = _require_env()
    client = create_client(url, key)

    df = canonicalize_diabetes_df(pd.read_csv(args.csv))

    if args.replace:
        print(f"A limpar tabela '{table}'…")
        _delete_all(client, table)

    records = to_snake_case_records(df)
    n = len(records)
    for i in range(0, n, BATCH_SIZE):
        chunk = records[i : i + BATCH_SIZE]
        try:
            client.table(table).insert(chunk).execute()
        except Exception as e:
            raise RuntimeError(f"Erro Supabase no lote {i}-{i + len(chunk)}: {e}") from e
        print(f"Inseridas {min(i + BATCH_SIZE, n)}/{n} linhas…")

    print(f"Concluído: {n} linhas na tabela '{table}'.")


if __name__ == "__main__":
    main()
