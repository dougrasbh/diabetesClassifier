FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG DAGSHUB_USER
ARG DAGSHUB_REPO
ARG DAGSHUB_TOKEN

ENV DAGSHUB_USER=$DAGSHUB_USER
ENV DAGSHUB_REPO=$DAGSHUB_REPO
ENV DAGSHUB_TOKEN=$DAGSHUB_TOKEN

# Opcional: ex. models:/DiabetesClassifier/latest se @production for legado
ARG MLFLOW_MODEL_URI=
ENV MLFLOW_MODEL_URI=$MLFLOW_MODEL_URI

RUN python3 -m src.evaluate

EXPOSE 8501

CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
