# ViralSense: Social Media Virality Prediction & Trend Saturation Detection Engine

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://mlflow.org/)
[![Codespaces](https://img.shields.io/badge/GitHub-Codespaces-brightgreen)](https://github.com/features/codespaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ViralSense is an end-to-end MLOps pipeline featuring continual learning capabilities, engineered to forecast early virality scores for short-form video content (YouTube Shorts & TikTok) and identify trend lifecycle stages (early-rise, peak, and saturated). The system addresses real-world constraints such as label latency, concept drift, and API quota limits through an automated, staged machine learning workflow.

---

## Key Objectives & Problem Domain

The lifecycle of short-form digital content is exceptionally brief, often decaying within days or a few weeks before reaching audience fatigue. ViralSense resolves this bottleneck by:
* Predicting content virality early via interaction growth rates (views, likes, comments, shares).
* Classifying trend phases using the slope of virality scores over time.
* Mitigating label latency by isolating immature observations from final evaluation datasets during a 3–7 day maturation window.
* Monitoring covariate shift and concept drift to trigger adaptive model retraining.
* Assisting content creators and digital marketing teams in discovering emerging viral opportunities and optimizing promotional content strategies.

---

## Data Pipeline Architecture

Data is ingested periodically via an automated ETL pipeline, cleaned, feature-engineered, and versioned using DVC prior to consumption by machine learning models.

```mermaid
flowchart LR
    A["Dynamic Ingestion Sources<br/>(YouTube Shorts / TikTok FYP)"] -->|Extract| B["data/raw/<br/>(JSON Snapshots)"]
    B -->|Transform: Clean| C["data/processed/<br/>(clean_videos.parquet)"]
    C -->|Transform: Features| D["data/processed/<br/>(features_matrix.parquet)"]
    D -->|Version Control| E["DVC + Git Tagging<br/>(data-v1.0.0)"]
```

For complete technical specifications and architectural design, refer to **[docs/data_pipeline_architecture.md](docs/data_pipeline_architecture.md)**.

---

## Directory Structure

This repository adheres to the **Cookiecutter Data Science** standard:

```text
├── .devcontainer/         # GitHub Codespaces container setup and reproducibility config
├── config/                # Pipeline parameter files (config.yaml)
├── data/
│   ├── external/          # Auxiliary trend signals (Google Trends, TikTok trends)
│   ├── processed/         # Cleaned, feature-engineered datasets (Parquet)
│   └── raw/               # Raw batch snapshots extracted from APIs and feeds
│       ├── tiktok/        # TikTok public FYP snapshots
│       └── youtube/       # YouTube Shorts snapshots
├── docs/                  # Technical design and architecture documentation
├── models/                # Serialized model artifacts and experiment weights
├── notebooks/             # Exploratory Data Analysis (EDA) and prototype notebooks
├── src/                   # Production-grade source code
│   ├── __init__.py
│   ├── data/              # Data ingestion and validation modules
│   │   ├── clean.py       # Data cleaning, deduplication, and schema validation
│   │   ├── ingest_tiktok.py  # TikTok live FYP and trending ingestion
│   │   └── ingest_youtube.py # YouTube Data API v3 and trending ingestion
│   ├── features/          # Tabular metric computation and text vectorization
│   │   └── build_features.py # Velocity, engagement rates, and label generation
│   └── models/            # Staged regression and classification pipelines
├── .env.example           # Template for environment variables and API keys
├── .gitignore             # Standard Python/ML git ignore configuration
├── LICENSE                # Project license (MIT)
├── requirements.txt       # Core project dependencies
└── README.md              # Project documentation
```

---

## Pipeline Execution Guide

### 1. Environment Setup
Install project dependencies and configure environment variables:
```bash
cp .env.example .env
pip install -r requirements.txt
```

### 2. Data Ingestion (Raw Extraction)
Execute scheduled extractors for YouTube Shorts and TikTok:
```bash
# Ingest YouTube Shorts (trending charts and broad query search)
python src/data/ingest_youtube.py

# Ingest TikTok (public FYP feed and trending streams)
python src/data/ingest_tiktok.py
```

### 3. Data Cleaning & Sanitization
Deduplicate records, filter out content exceeding 60 seconds, and handle missing values:
```bash
python src/data/clean.py
```

### 4. Feature Engineering
Compute velocity metrics (views/hour, likes/hour), engagement ratios, content signals, and virality ground-truth labels:
```bash
python src/features/build_features.py
```

---

## Data Versioning Strategy (DVC)

Dataset releases follow semantic versioning (`v1.0.0`, `v1.1.0`) and are tracked using Data Version Control (DVC) to keep the Git repository lightweight:

```bash
# Track raw snapshots and processed datasets with DVC
dvc add data/raw data/processed

# Track DVC pointer files in Git
git add data/raw.dvc data/processed.dvc .gitignore
git commit -m "feat(data): track snapshot dataset v1.0.0"
git tag -a "data-v1.0.0" -m "Snapshot dataset release v1.0.0"
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.