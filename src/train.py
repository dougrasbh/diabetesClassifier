import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.prepare_data import (
    FEATURE_COLUMNS,
    InvalidZeroToNaN,
    default_dataset_path,
    load_and_split,
)

load_dotenv()

DAGSHUB_USER = os.getenv("DAGSHUB_USER")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REGISTERED_MODEL = "DiabetesClassifier"


def build_pipeline(estimator) -> Pipeline:
    return Pipeline(
        [
            ("invalid_zeros", InvalidZeroToNaN()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    )


def run_configs():
    return [
        {
            "run_name": "exp-1-logistic-balanced",
            "model_tag": "LogisticRegression",
            "pipeline": build_pipeline(
                LogisticRegression(
                    max_iter=200,
                    class_weight="balanced",
                    solver="lbfgs",
                )
            ),
            "extra_params": {"solver": "lbfgs", "class_weight": "balanced"},
        },
        {
            "run_name": "exp-2-random-forest",
            "model_tag": "RandomForestClassifier",
            "pipeline": build_pipeline(
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    random_state=42,
                    class_weight="balanced",
                )
            ),
            "extra_params": {"n_estimators": 200, "max_depth": 8},
        },
        {
            "run_name": "exp-3-gradient-boosting",
            "model_tag": "GradientBoostingClassifier",
            "pipeline": build_pipeline(
                GradientBoostingClassifier(
                    n_estimators=120,
                    max_depth=3,
                    learning_rate=0.08,
                    random_state=42,
                )
            ),
            "extra_params": {"n_estimators": 120, "max_depth": 3, "learning_rate": 0.08},
        },
    ]


def evaluate_binary(pipeline: Pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, proba)
    return acc, f1w, auc, y_pred


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
    mlflow.set_experiment("diabetes-classification")

    data_path = default_dataset_path()
    X_train, X_test, y_train, y_test = load_and_split(data_path)

    print(f"Dataset: {data_path}")
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

    best = None
    best_auc = -1.0

    for cfg in run_configs():
        run_name = cfg["run_name"]
        pipeline = cfg["pipeline"]

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "model_family": cfg["model_tag"],
                    "test_size": 0.2,
                    "data_path": str(data_path),
                    **cfg["extra_params"],
                }
            )

            pipeline.fit(X_train, y_train)
            acc, f1w, auc, y_pred = evaluate_binary(pipeline, X_test, y_test)

            print("\n" + "=" * 50)
            print(run_name)
            print(classification_report(y_test, y_pred))
            print(f"accuracy={acc:.4f} f1_weighted={f1w:.4f} roc_auc={auc:.4f}")
            print("=" * 50)

            mlflow.log_metrics(
                {"accuracy": acc, "f1_weighted": f1w, "roc_auc": auc}
            )
            mlflow.set_tags({"model": cfg["model_tag"], "domain": "diabetes"})

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL,
            )

            if auc > best_auc:
                best_auc = auc
                best = {
                    "run_name": run_name,
                    "model_tag": cfg["model_tag"],
                    "accuracy": acc,
                    "f1_weighted": f1w,
                    "roc_auc": auc,
                }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_by_roc_auc": best,
        "features": FEATURE_COLUMNS,
        "target": "Outcome (1 = diabetes)",
        "note": "Promova a versão desejada no Model Registry do DagsHub para @production.",
    }
    (ARTIFACTS_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nResumo salvo em {ARTIFACTS_DIR / 'training_summary.json'}")
    print(
        f"\nMelhor run (por ROC-AUC no holdout): {best['run_name']} "
        f"AUC={best_auc:.4f}"
    )
    print(
        "\nPróximo passo: no DagsHub → Models → DiabetesClassifier, "
        "promova o melhor modelo para o stage 'Production'."
    )
    print(f"Experiments: https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}")


if __name__ == "__main__":
    main()
