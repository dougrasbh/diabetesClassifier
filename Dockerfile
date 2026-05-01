# syntax=docker/dockerfile:1
# Estágio 1: credenciais só existem aqui; a imagem final não inclui DAGSHUB_*.
FROM python:3.11-slim AS model-builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG DAGSHUB_USER
ARG DAGSHUB_REPO
ARG DAGSHUB_TOKEN
# Opcional; se vazio, não exportar (senão evaluate.py veria "" e ignorava o default).
ARG MLFLOW_MODEL_URI

ENV DAGSHUB_USER=$DAGSHUB_USER \
    DAGSHUB_REPO=$DAGSHUB_REPO \
    DAGSHUB_TOKEN=$DAGSHUB_TOKEN

RUN if [ -n "$MLFLOW_MODEL_URI" ]; then export MLFLOW_MODEL_URI="$MLFLOW_MODEL_URI"; fi && \
    python3 -m src.evaluate

# Estágio 2: aplicação + artefatos do evaluate, sem tokens DagsHub
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY --from=model-builder /app/model.pkl /app/model.pkl
COPY --from=model-builder /app/artifacts/model_metrics.json /app/artifacts/model_metrics.json

ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
