# Classificador de diabetes — pipeline MLOps (roteiro)

Projeto de **classificação binária** (presença de diabetes) com o dataset **Pima Indians Diabetes**, no fluxo: **Supabase → ingestão → DuckDB → DVC → MLflow (DagsHub) → Docker → Render → Streamlit**.

## Índice

1. [Problema e métricas](#problema-e-métricas)  
2. [Arquitetura](#arquitetura)  
3. [Pré-requisitos](#pré-requisitos)  
4. [Configuração do ambiente](#configuração-do-ambiente)  
5. [DagsHub (passo a passo manual)](#dagshub-passo-a-passo-manual)  
6. [Supabase: tabela, upload e ingestão](#supabase-tabela-upload-e-ingestão)  
7. [Processamento com DuckDB](#processamento-com-duckdb)  
8. [Pipeline local e DVC](#pipeline-local-e-dvc)  
9. [Experimentos e modelo em produção no MLflow](#experimentos-e-modelo-em-produção-no-mlflow)  
10. [Interface Streamlit](#interface-streamlit)  
11. [Docker e Render](#docker-e-render)  
12. [Estrutura de pastas](#estrutura-de-pastas)  
13. [Segurança](#segurança)  

---

## Problema e métricas

- **Objetivo:** prever `Outcome` (0 ou 1) a partir de variáveis clínicas e demográficas tabulares.  
- **Variável-alvo:** `Outcome`.  
- **Métricas principais:** acurácia, F1 (ponderado) e **ROC-AUC** no conjunto de teste (holdout de 20%).  
- **Fonte dos dados:** tabela no **Supabase**. O CSV local (`data/diabetes.csv`) serve apenas como arquivo de origem para o **upload** inicial.

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

- **Python** 3.11 ou superior  
- **Conta no [GitHub](https://github.com)** (login no DagsHub via GitHub)  
- **Opcional:** projeto no **[Supabase](https://supabase.com)** com tabela nas mesmas colunas do CSV de referência  

## Configuração do ambiente

```bash
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Edite o `.env` e preencha, no mínimo:

- **DagsHub / MLflow:** `DAGSHUB_USER`, `DAGSHUB_REPO`, `DAGSHUB_TOKEN` (veja a seção [DagsHub](#dagshub-passo-a-passo-manual)).  
- **Supabase (ingestão e upload):** `SUPABASE_URL`, `SUPABASE_KEY` (recomenda-se *service role* para insert); opcionalmente `SUPABASE_DIABETES_TABLE` (padrão: `diabetes`).  

`SUPABASE_URL` e `SUPABASE_KEY` ficam **somente** no `.env`; o arquivo está listado no `.gitignore` e **não** deve ser versionado.

## DagsHub (passo a passo manual)

### Criar repositório no DagsHub

O **[DagsHub](https://dagshub.com)** integra Git ao rastreamento de experimentos (MLflow). Siga os passos:

1. Acesse [dagshub.com](https://dagshub.com) e faça login com a conta GitHub.  
2. Clique em **New Repository**.  
3. Crie manualmente o repositório com o nome **`diabetes-app`** e use no `.env` exatamente `DAGSHUB_REPO=diabetes-app`.  
4. Mantenha as opções padrão e clique em **Create Repository**.  
5. O servidor MLflow do repositório segue o padrão `https://dagshub.com/<USUARIO>/<REPOSITORIO>.mlflow`. **Não é obrigatório copiar essa URL:** os scripts `src/train.py` e `src/evaluate.py` montam o *tracking URI* a partir de `DAGSHUB_USER` e `DAGSHUB_REPO` no `.env`.

> **Nota:** o DagsHub provisiona automaticamente um servidor MLflow gratuito por repositório. Não é necessário instalar nem hospedar um servidor MLflow à parte.

### Gerar token de acesso

1. No DagsHub: avatar → **User Settings** → **Tokens**.  
2. **New Token** → dê um nome (ex.: `aula-mlops`) e confirme.  
3. Copie o token (ele aparece **apenas uma vez**) e salve em `DAGSHUB_TOKEN` no `.env`.

## Supabase: tabela, upload e ingestão

1. No Supabase, abra o **SQL Editor**, copie o conteúdo de [`scripts/create_diabetes_table.sql`](scripts/create_diabetes_table.sql), cole e execute o script (cria `public.diabetes` em *snake_case*).  
2. Envie o CSV para a tabela usando as variáveis `SUPABASE_*` no `.env` (a chave **service_role** reduz bloqueios de RLS no *insert*):

```bash
python3 -m src.upload_supabase
```

3. O restante do pipeline **lê** os dados do Supabase apenas na ingestão:

```bash
python3 -m src.ingestion
```

O `src/ingestion.py` usa o cliente oficial do Supabase, monta um `pandas.DataFrame` e grava a camada bruta (*raw*) usada nas etapas seguintes.

## Processamento com DuckDB

No `src/preprocessing.py`, o DuckDB é usado como motor analítico embarcado para:

- limpeza e padronização das colunas;  
- transformações em SQL sobre os dados;  
- preparação das *features* para o treino.  

A saída é um Parquet em `data/processed/` (adequado para versionamento com DVC).

```bash
python3 -m src.preprocessing
```

**Verificação:** confira se existe `data/processed/features.parquet` antes de treinar.

## Pipeline local e DVC

O fluxo está em [`dvc.yaml`](dvc.yaml): estágios `ingest` → `process` → `train`. O estágio `ingest` exige Supabase configurado (variáveis no `.env` ou no ambiente de CI).

Para rodar **na mão** as mesmas etapas (sem depender do comando `dvc repro` neste guia):

```bash
python3 -m src.ingestion
python3 -m src.preprocessing
python3 -m src.train
```

### Avaliação e `model.pkl` (antes do Docker)

Antes de `python3 -m src.evaluate`, associe o modelo aprovado ao alias **production** no MLflow (passos em [Definir o alias production no MLflow](#definir-o-alias-production-no-mlflow-antes-do-evaluate)). Assim o download usa o modelo **aprovado para produção**, não só o mais recente.

Se o alias `production` ainda não apontar para o artefato desejado, defina uma URI explícita só para esse download:

```bash
export MLFLOW_MODEL_URI="models:/DiabetesClassifier/latest"
python3 -m src.evaluate
```

Confirme que existem `model.pkl` e `artifacts/model_metrics.json` na raiz do projeto antes do `docker build`.

## Experimentos e modelo em produção no MLflow

Em [`src/train.py`](src/train.py) são treinadas, em sequência, três famílias de modelo:

1. Regressão logística (`class_weight=balanced`)  
2. *Random Forest*  
3. *Gradient Boosting*  

Cada *run* registra métricas como **accuracy**, **f1_weighted** e **roc_auc** (detalhes no próprio `train.py`).

### Escolher o melhor modelo (DagsHub)

1. Acesse `https://dagshub.com/<SEU_USUARIO>/diabetes-app` (substitua `<SEU_USUARIO>` por `DAGSHUB_USER` no `.env`; o repositório deve ser **`diabetes-app`**, como em `DAGSHUB_REPO`).  
2. Aba **Experiments**.  
3. Os **3 runs** aparecem com **accuracy** e **f1_weighted** (e demais métricas do MLflow).  
4. Ordene pela coluna **f1_weighted** (maior → menor).  
5. Selecione dois *runs* → **Compare** para comparar parâmetros.  
6. Escolha o *run* com maior **F1 (ponderado)** como candidato a modelo de produção.

### Definir o alias production no MLflow (antes do evaluate)

1. No DagsHub, clique em **Go to MLflow UI** (canto superior direito na aba **Experiments**).  
2. No menu do MLflow: **Models**.  
3. Abra **`DiabetesClassifier`** (nome registrado pelo `src/train.py`).  
4. Veja as versões (Version 1, 2, 3…). Identifique a versão ligada ao *run* com maior **f1_weighted** (o número da versão costuma aumentar se os experimentos foram feitos em sequência; confira sempre as métricas no *run*).  
5. Na linha dessa versão: **⋮** → **Add alias**.  
6. Digite **`production`** e confirme. A versão passa a exibir o alias **production**.  
7. O `src/evaluate.py` usa por padrão `models:/DiabetesClassifier@production`. Opcional: variável de ambiente `MLFLOW_MODEL_URI` para outro artefato.

Em seguida:

```bash
python3 -m src.evaluate
```

## Interface Streamlit

```bash
streamlit run app/streamlit_app.py
```

Abas: **Predição**, **Sobre o modelo** (*features* e resumo de treino), **EDA**.

## Docker e Render

O `Dockerfile` usa **multi-stage build**: em um estágio intermediário executa-se `python3 -m src.evaluate` com credenciais passadas por **`--build-arg`**. Na imagem final entram `model.pkl` e `artifacts/model_metrics.json`; **não** são definidos `ENV` com `DAGSHUB_*` no estágio final (evita expor o token em *runtime*). **Atenção:** *build args* podem aparecer nos logs de build; em produção considere *secrets* do BuildKit.

**Build** (ajuste usuário, repositório e token):

```bash
docker build \
  --build-arg DAGSHUB_USER=seu_usuario \
  --build-arg DAGSHUB_REPO=diabetes-app \
  --build-arg DAGSHUB_TOKEN=seu_token \
  --build-arg MLFLOW_MODEL_URI=models:/DiabetesClassifier@production \
  -t diabetes-classifier .
```

O argumento `MLFLOW_MODEL_URI` é **opcional**; omita-o para usar o padrão do `src/evaluate.py`.

**Execução local** (o Streamlit usa `PORT` se existir; caso contrário, `8501`):

```bash
docker run -p 8501:8501 -e PORT=8501 diabetes-classifier
```

No **Render**, configure os **Docker Build Arguments** com os mesmos valores dos `--build-arg`. O Render define `PORT`; o contêiner usa `--server.address=0.0.0.0` para aceitar tráfego externo.

## Estrutura de pastas

```
├── app/
│   └── streamlit_app.py
├── artifacts/              # treino / evaluate
├── data/
│   ├── diabetes.csv        # exemplo para upload → Supabase
│   ├── raw/                # gerado (DVC)
│   └── processed/          # gerado (DVC)
├── notebooks/              # EDA opcional
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

Não faça *commit* de `.env` nem de tokens. Se algum token vazar, revogue-o no DagsHub e no Supabase e gere outro.
