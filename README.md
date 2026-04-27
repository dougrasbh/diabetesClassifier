# Classificador de diabetes — pipeline MLOps (roteiro)

Projeto de **classificação binária** (presença de diabetes) com o dataset **Pima Indians Diabetes**, seguindo o fluxo do roteiro: **Supabase → ingestão → DuckDB → DVC → MLflow (DagsHub) → Docker → Render → Streamlit**.

## Problema e métricas

- **Objetivo:** prever `Outcome` (0 ou 1) a partir de variáveis clínicas e demográficas tabulares.
- **Métricas principais:** acurácia, F1 (weighted) e **ROC-AUC** no conjunto de teste (holdout 20%).
- **Fonte dos dados:** tabela no **Supabase**. O CSV local serve só como ficheiro de origem para o **upload** inicial.

## Arquitetura

```
Supabase  →  src/ingestion.py  →  data/raw/*.parquet
                ↓
DuckDB (src/preprocessing.py)  →  data/processed/features.parquet
                ↓
dvc repro  →  src/train.py  →  MLflow (≥3 modelos)  →  Model Registry
                ↓
src/evaluate.py  →  model.pkl + artifacts/model_metrics.json
                ↓
Streamlit (app/streamlit_app.py)  →  Docker  →  Render
```

## Pré-requisitos

- Python 3.11+
- Conta [DagsHub](https://dagshub.com) (MLflow + opcionalmente remote DVC)
- Opcional: projeto [Supabase](https://supabase.com) com tabela contendo as mesmas colunas do CSV de referência

## Configuração

```bash
cp .env.example .env
# Preencha DAGSHUB_USER, DAGSHUB_REPO, DAGSHUB_TOKEN
# Obrigatório para ingestão: SUPABASE_URL, SUPABASE_KEY (recomenda-se service_role para upload)
# SUPABASE_DIABETES_TABLE=diabetes
pip install -r requirements.txt
```

## Supabase: criar tabela, enviar CSV e ingerir

1. No Supabase, executa o SQL em [`scripts/create_diabetes_table.sql`](scripts/create_diabetes_table.sql) (cria `public.diabetes` em snake_case).
2. Carrega o dataset para a tabela a partir do CSV (usa `SUPABASE_*` no `.env`; **service_role** evita bloqueios de RLS no insert):

```bash
python3 -m src.upload_supabase                    # default: data/diabetes.csv
python3 -m src.upload_supabase --replace          # apaga linhas existentes e reinsere
python3 -m src.upload_supabase --csv /caminho/outro.csv
```

3. O resto do pipeline **só lê** do Supabase na ingestão:

```bash
python3 -m src.ingestion
```

## Reproduzir o pipeline (DVC)

```bash
# Remote de armazenamento no DagsHub (uma vez por máquina)
dvc remote add -d dagshub https://dagshub.com/<SEU_USUARIO>/<SEU_REPO>.dvc
dvc remote modify --local dagshub auth basic
dvc remote modify --local dagshub user <token_dagshub>
dvc remote modify --local dagshub password <token_dagshub>

python3 -m dvc repro
```

Estágios em [`dvc.yaml`](dvc.yaml): `ingest` → `process` → `train`. O estágio `ingest` **exige** Supabase configurado (variáveis no `.env` ou no ambiente do CI).

Após o treino, promova no DagsHub o melhor modelo **sklearn** (versão criada pelo `train.py`) para o stage **Production** do registry `DiabetesClassifier`. Se `@production` ainda apontar para um artefato antigo (ex.: XGBoost), defina uma URI explícita:

```bash
# Ex.: versão específica ou a mais recente registrada
export MLFLOW_MODEL_URI="models:/DiabetesClassifier/latest"
python3 -m src.evaluate   # gera model.pkl e artifacts/model_metrics.json
```

No **Docker build**, passe `--build-arg MLFLOW_MODEL_URI=models:/DiabetesClassifier/latest` se o alias `@production` ainda não tiver sido atualizado para um modelo sklearn deste pipeline.

## Experimentos (MLflow)

Três famílias de modelo são treinadas em sequência em [`src/train.py`](src/train.py):

1. Regressão logística (`class_weight=balanced`)
2. Random Forest
3. Gradient Boosting

Compare runs no DagsHub → **Experiments** e escolha a versão a promover para `@production`.

## Interface Streamlit

```bash
python3 -m src.evaluate   # necessário para baixar o modelo em produção (ou use um model.pkl já existente)
streamlit run app/streamlit_app.py
```

Abas: **Predição**, **Sobre o modelo** (features e resumo de treino), **EDA**.

## Docker (local / Render)

```bash
docker build \
  --build-arg DAGSHUB_USER=seu_usuario \
  --build-arg DAGSHUB_REPO=seu_repo \
  --build-arg DAGSHUB_TOKEN=seu_token \
  -t diabetes-classifier .

docker run -p 8501:8501 -e PORT=8501 diabetes-classifier
```

No **Render**: Web Service com **Docker**, porta conforme `PORT` (o `CMD` usa `${PORT:-8501}`). Defina `DAGSHUB_USER`, `DAGSHUB_REPO`, `DAGSHUB_TOKEN` nas variáveis de ambiente (o build executa `evaluate` e embute `model.pkl`).

## Estrutura de pastas

```
├── app/
│   └── streamlit_app.py
├── artifacts/              # gerado pelo treino / evaluate
├── data/
│   ├── diabetes.csv        # exemplo para upload → Supabase
│   ├── raw/                # DVC (gerado)
│   └── processed/          # DVC (gerado)
├── notebooks/              # EDA adicional (opcional)
├── scripts/
│   └── create_diabetes_table.sql
├── src/
│   ├── schema.py
│   ├── ingestion.py
│   ├── upload_supabase.py
│   ├── preprocessing.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── dvc.yaml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Segurança

Não commite `.env` nem tokens. Se um token vazou, revogue-o no DagsHub/Supabase e gere outro.
