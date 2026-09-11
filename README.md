# ViralSense: Social Media Virality Prediction & Trend Saturation Detection Engine
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://mlflow.org/)
[![Codespaces](https://img.shields.io/badge/GitHub-Codespaces-brightgreen)](https://github.com/features/codespaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ViralSense is an end-to-end MLOps pipeline featuring continual learning capabilities, engineered to forecast early virality scores for short-form video content and identify trend lifecycle stages (early-rise, peak, and saturated)[cite: 1]. The system addresses real-world constraints such as label latency, concept drift, and API quota limits through an automated, staged machine learning workflow[cite: 1].

---

## Key Objectives & Problem Domain
The lifecycle of short-form digital content is exceptionally brief, often decaying within days or a few weeks before reaching audience fatigue[cite: 1]. ViralSense resolves this bottleneck by:
* Predicting content virality early via interaction growth rates (views, likes, comments, shares)[cite: 1].
* Classifying trend phases using the slope of virality scores over time[cite: 1].
* Mitigating label latency by isolating immature observations from final evaluation datasets during a 3–7 day maturation window[cite: 1].
* Monitoring covariate shift and concept drift to trigger adaptive model retraining[cite: 1].

---

## Directory Structure
This repository adheres to the **Cookiecutter Data Science** standard:

```text
├── .devcontainer/         # GitHub Codespaces container setup and reproducibility config
├── config/                # Pipeline parameter files (config.yaml)
├── data/
│   ├── external/          # Auxiliary trend signals (Google Trends, TikTok trends)
│   ├── processed/         # Cleaned, feature-engineered datasets
│   └── raw/               # Raw batch snapshots extracted from the YouTube API
├── models/                # Serialized model artifacts and experiment weights
├── notebooks/             # Exploratory Data Analysis (EDA) and prototype notebooks
├── src/                   # Production-grade source code
│   ├── __init__.py
│   ├── data/              # Data ingestion and validation modules (Airflow jobs)
│   ├── features/          # Tabular metric computation and text vectorization
│   └── models/            # Staged regression and classification pipelines
├── .env.example           # Template for environment variables and API keys
├── .gitignore             # Standard Python/ML git ignore configuration
├── requirements.txt       # Core project dependencies
└── README.md              # Project documentation
