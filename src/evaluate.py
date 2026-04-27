import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

from src.prepare_data import FEATURE_COLUMNS, load_and_split

load_dotenv()

DAGSHUB_USER = os.getenv("DAGSHUB_USER")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"


def main():
    if not all([DAGSHUB_USER, DAGSHUB_REPO, DAGSHUB_TOKEN]):
        raise SystemExit(
            "Defina DAGSHUB_USER, DAGSHUB_REPO e DAGSHUB_TOKEN no arquivo .env"
        )

    mlflow.set_tracking_uri(
        f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
    )
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

    model_uri = os.getenv(
        "MLFLOW_MODEL_URI", "models:/DiabetesClassifier@production"
    )

    print(f"Baixando modelo: {model_uri}")
    print(f"Fonte: https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}")

    pipeline = mlflow.sklearn.load_model(model_uri)
    print("Modelo carregado com sucesso!")

    _, X_test, _, y_test = load_and_split()
    y_pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, proba)

    print("\n" + "=" * 50)
    print("Avaliação do modelo em produção (holdout local):")
    print(classification_report(y_test, y_pred))
    print(f"Acurácia : {acc:.2%}")
    print(f"F1 (weighted): {f1:.3f}")
    print(f"ROC-AUC: {auc:.3f}")
    print("=" * 50)

    out_model = ROOT / "model.pkl"
    joblib.dump(pipeline, out_model)
    print(f"\nModelo salvo em {out_model}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = ARTIFACTS_DIR / "model_metrics.json"
    payload = {
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "roc_auc": float(auc),
        "model_uri": model_uri,
        "features": list(FEATURE_COLUMNS),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Métricas salvas em {metrics_path}")


if __name__ == "__main__":
    main()
