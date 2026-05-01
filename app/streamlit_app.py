import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
# O pipeline em model.pkl referencia classes em `src.*`; o Streamlit precisa do raiz no PYTHONPATH.
_root = str(ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)

MODEL_PATH = ROOT / "model.pkl"
METRICS_PATH = ROOT / "artifacts" / "model_metrics.json"
TRAINING_SUMMARY_PATH = ROOT / "artifacts" / "training_summary.json"
DATA_FOR_EDA = ROOT / "data" / "processed" / "features.parquet"
FALLBACK_EDA = ROOT / "data" / "diabetes.csv"

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


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        st.error(
            f"Arquivo **model.pkl** não encontrado em `{MODEL_PATH}`. "
            "Execute `python -m src.evaluate` (com modelo em @production no MLflow) "
            "ou coloque um modelo treinado na raiz do projeto. "
            "No Docker, passe `--build-arg DAGSHUB_*` para o build correr `src.evaluate` e embutir o modelo."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


def load_metrics_block():
    if METRICS_PATH.is_file():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return None


def load_training_summary():
    if TRAINING_SUMMARY_PATH.is_file():
        return json.loads(TRAINING_SUMMARY_PATH.read_text(encoding="utf-8"))
    return None


def load_eda_frame():
    if DATA_FOR_EDA.is_file():
        return pd.read_parquet(DATA_FOR_EDA)
    if FALLBACK_EDA.is_file():
        return pd.read_csv(FALLBACK_EDA)
    return None


st.set_page_config(
    page_title="Predição de Diabetes",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Predição de diabetes (Pima Indians)")
st.markdown(
    "Classificação binária (**Outcome**): risco de diabetes com base em variáveis clínicas. "
    "O modelo é um pipeline sklearn (limpeza de zeros, imputação por mediana, normalização + classificador) "
    "carregado de `model.pkl` (gerado manualmente via `python -m src.evaluate`)."
)

metrics = load_metrics_block()
summary = load_training_summary()
if metrics:
    st.success(
        f"Métricas no holdout local (modelo em produção): "
        f"accuracy **{metrics['accuracy']:.3f}** · "
        f"F1 weighted **{metrics['f1_weighted']:.3f}** · "
        f"ROC-AUC **{metrics['roc_auc']:.3f}**"
    )
elif summary and summary.get("best_by_roc_auc"):
    b = summary["best_by_roc_auc"]
    st.info(
        f"Último treino local — melhor experimento (ROC-AUC): **{b.get('run_name')}** "
        f"(AUC **{b.get('roc_auc', 0):.3f}**). "
        "Execute `python -m src.evaluate` após promover um modelo no MLflow para ver métricas do artefato em produção."
    )

tab_pred, tab_model, tab_eda = st.tabs(["Predição", "Sobre o modelo", "EDA"])

with tab_pred:
    pipeline = load_model()

    st.subheader("Dados do paciente")
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.slider("Pregnancies", 0, 20, 6, 1)
        glucose = st.slider("Glucose", 0, 200, 148, 1)
        blood_pressure = st.slider("Blood Pressure", 0, 130, 72, 1)
        skin_thickness = st.slider("Skin Thickness", 0, 100, 35, 1)
    with col2:
        insulin = st.slider("Insulin", 0, 900, 0, 1)
        bmi = st.slider("BMI", 0.0, 70.0, 33.6, 0.1)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.627, 0.001)
        age = st.slider("Age", 0, 100, 50, 1)

    if st.button("Prever", type="primary", use_container_width=True):
        input_data = pd.DataFrame(
            [
                {
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": blood_pressure,
                    "SkinThickness": skin_thickness,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "DiabetesPedigreeFunction": dpf,
                    "Age": age,
                }
            ]
        )

        proba = pipeline.predict_proba(input_data)[0][1]

        st.divider()
        if proba < 0.3:
            st.success(f"Baixo risco ({proba:.2%})")
        elif proba < 0.7:
            st.warning(f"Risco moderado ({proba:.2%})")
        else:
            st.error(f"Alto risco ({proba:.2%})")

        st.markdown("### Probabilidade (classe positiva)")
        st.progress(float(proba))

    st.divider()
    st.subheader("Predição em lote (CSV)")
    st.markdown(
        "Colunas obrigatórias: `Pregnancies, Glucose, BloodPressure, SkinThickness, "
        "Insulin, BMI, DiabetesPedigreeFunction, Age`"
    )
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if not all(c in df.columns for c in FEATURE_COLUMNS):
            st.error("O CSV não contém todas as colunas necessárias.")
        else:
            pipeline = load_model()
            df = df.copy()
            df["prediction"] = pipeline.predict(df[FEATURE_COLUMNS])
            df["probability"] = pipeline.predict_proba(df[FEATURE_COLUMNS])[:, 1]
            st.success(f"{len(df)} linhas analisadas.")
            st.dataframe(df, use_container_width=True)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Baixar resultados",
                data=csv_bytes,
                file_name="resultado_diabetes.csv",
                mime="text/csv",
                use_container_width=True,
            )

with tab_model:
    st.markdown("### Features esperadas (ordem não importa; nomes devem coincidir)")
    st.code("\n".join(FEATURE_COLUMNS), language="text")
    st.markdown("### Alvo")
    st.write("`Outcome` binário: 1 indica diabetes (rótulo positivo no treino).")
    if summary:
        st.markdown("### Resumo do último `src.train`")
        st.json(summary)
    st.markdown(
        "### Fluxo MLOps (roteiro)\n"
        "Supabase → ingestão → DuckDB (`data/processed`) → `dvc repro` → "
        "MLflow (≥3 modelos) → promover **Production** → `evaluate` → `model.pkl` → Streamlit/Docker."
    )

with tab_eda:
    df_eda = load_eda_frame()
    if df_eda is None:
        st.warning("Nenhum dataset encontrado para EDA (rode ingestão/pré-processamento).")
    else:
        st.caption(f"Amostra: {len(df_eda)} linhas")
        st.dataframe(df_eda.describe(), use_container_width=True)
        numeric = df_eda.select_dtypes(include="number").columns.tolist()
        if "Outcome" in numeric:
            numeric.remove("Outcome")
        pick = st.selectbox("Histograma", numeric, index=min(1, len(numeric) - 1))
        s = df_eda[pick].dropna()
        if len(s) > 0:
            counts, edges = np.histogram(s, bins=15)
            labels = [f"{edges[i]:.0f}-{edges[i + 1]:.0f}" for i in range(len(counts))]
            st.bar_chart(pd.Series(counts, index=labels, name="frequência"))
