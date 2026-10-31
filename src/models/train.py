"""Modul Pelatihan Model dan Pelacakan Eksperimen dengan MLflow - ViralSense MLOps.

Modul ini bertanggung jawab untuk:
1. Memuat dataset ter-versioning (LK-05) dari features_matrix.parquet.
2. Mempersiapkan stratified train-test split dan preprocessing pipeline.
3. Melatih model klasifikasi (Random Forest, Gradient Boosting, SVM, Logistic Regression).
4. Melakukan logging parameter, metrik, artefak grafik, dan model ke MLflow.
5. Menjalankan suite eksperimen komparatif (minimal 3-5 runs).
6. Mendaftarkan model terbaik (Champion) ke MLflow Model Registry.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Backend non-interaktif tanpa display GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Suppress noisy agent hints from MLflow
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("viralsense-train")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Memuat konfigurasi pipeline dari file YAML."""
    full_path = PROJECT_ROOT / config_path if not Path(config_path).is_absolute() else Path(config_path)
    if not full_path.exists():
        logger.warning(f"File konfigurasi {full_path} tidak ditemukan. Menggunakan konfigurasi default.")
        return {}
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dataset_version() -> str:
    """Mendeteksi tag versi dataset terkini dari Git atau DVC."""
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if tag:
            return tag
    except Exception:
        pass
    return "data-v3.0"


def load_and_validate_dataset(
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, List[str], str]:
    """Memuat dataset fitur dan memvalidasi keberadaan target serta fitur."""
    storage_cfg = config.get("storage", {})
    feat_path_str = storage_cfg.get("features_data_path", "data/processed/features_matrix.parquet")
    feat_path = PROJECT_ROOT / feat_path_str if not Path(feat_path_str).is_absolute() else Path(feat_path_str)

    if not feat_path.exists():
        # Fallback coba dari clean data jika features_matrix belum dibuat
        clean_path_str = storage_cfg.get("clean_data_path", "data/processed/clean_videos.parquet")
        clean_path = PROJECT_ROOT / clean_path_str
        if clean_path.exists():
            logger.info(f"Features matrix belum ada. Membaca {clean_path} dan membuat fitur...")
            from src.features.build_features import extract_features, save_features_matrix
            clean_df = pd.read_parquet(clean_path)
            feat_df = extract_features(clean_df, config)
            save_features_matrix(feat_df, str(feat_path))
        else:
            raise FileNotFoundError(
                f"Dataset fitur tidak ditemukan di {feat_path} maupun {clean_path}. "
                "Jalankan modul preprocessing & build_features terlebih dahulu."
            )
    else:
        feat_df = pd.read_parquet(feat_path)

    training_cfg = config.get("model_training", {})
    target_col = training_cfg.get("target_column", "is_viral")
    feature_cols = training_cfg.get(
        "feature_columns",
        [
            "duration_seconds",
            "video_age_hours",
            "engagement_rate",
            "comment_to_like_ratio",
            "caption_length",
            "word_count",
            "has_hook_question",
        ],
    )

    if target_col not in feat_df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan di dalam dataset.")

    missing_cols = [c for c in feature_cols if c not in feat_df.columns]
    if missing_cols:
        raise ValueError(f"Kolom fitur berikut tidak ditemukan di dataset: {missing_cols}")

    logger.info(
        f"Dataset berhasil dimuat. Total sampel: {len(feat_df)}, "
        f"Fitur: {len(feature_cols)}, Distribusi target: {feat_df[target_col].value_counts().to_dict()}"
    )
    return feat_df, feature_cols, target_col


def split_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Membagi data ke set latih dan uji menggunakan stratified split."""
    X = df[feature_cols].copy()
    y = df[target_col].copy().astype(int)

    # Imputasi nilai tak terhingga atau NaN jika ada
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        f"Data terbagi: Train={X_train.shape[0]} sampel (Positif: {y_train.sum()}), "
        f"Test={X_test.shape[0]} sampel (Positif: {y_test.sum()})"
    )
    return X_train, X_test, y_train, y_test


def create_model(model_type: str, hyperparams: Dict[str, Any]) -> Any:
    """Menginstansiasi estimator Scikit-Learn berdasarkan tipe model dan hyperparameter."""
    model_type_norm = model_type.lower()
    random_state = hyperparams.get("random_state", 42)

    if model_type_norm in ["rf", "random_forest"]:
        n_estimators = int(hyperparams.get("n_estimators", 100))
        max_depth = hyperparams.get("max_depth", None)
        if max_depth is not None:
            max_depth = int(max_depth)
        min_samples_split = int(hyperparams.get("min_samples_split", 2))
        criterion = hyperparams.get("criterion", "gini")
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            random_state=random_state,
        )

    elif model_type_norm in ["gbm", "gradient_boosting"]:
        n_estimators = int(hyperparams.get("n_estimators", 100))
        learning_rate = float(hyperparams.get("learning_rate", 0.1))
        max_depth = int(hyperparams.get("max_depth", 3))
        subsample = float(hyperparams.get("subsample", 1.0))
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
        )

    elif model_type_norm in ["svm", "svc"]:
        kernel = hyperparams.get("kernel", "rbf")
        c_val = float(hyperparams.get("C", 1.0))
        gamma = hyperparams.get("gamma", "scale")
        return SVC(
            kernel=kernel,
            C=c_val,
            gamma=gamma,
            probability=True,
            random_state=random_state,
        )

    elif model_type_norm in ["lr", "logistic_regression"]:
        c_val = float(hyperparams.get("C", 1.0))
        penalty = hyperparams.get("penalty", "l2")
        solver = hyperparams.get("solver", "lbfgs")
        max_iter = int(hyperparams.get("max_iter", 1000))
        return LogisticRegression(
            C=c_val,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
        )

    else:
        raise ValueError(
            f"Tipe model '{model_type}' tidak didukung. "
            "Pilihan valid: 'random_forest', 'gradient_boosting', 'svm', 'logistic_regression'."
        )


def evaluate_predictions(
    y_true: pd.Series, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Menghitung metrik performa klasifikasi secara komprehensif."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = 0.5
    else:
        metrics["roc_auc"] = 0.5
    return metrics


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, title: str) -> plt.Figure:
    """Membuat visualisasi confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
    fig.colorbar(cax)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                x=j,
                y=i,
                s=str(cm[i, j]),
                va="center",
                ha="center",
                fontsize=14,
                fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Viral (0)", "Viral (1)"])
    ax.set_yticklabels(["Non-Viral (0)", "Viral (1)"])
    ax.set_xlabel("Predicted Label", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title(title, pad=15, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true: pd.Series, y_proba: np.ndarray, roc_auc: float, title: str) -> plt.Figure:
    """Membuat visualisasi Kurva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Random Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title(title, pad=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: List[str], importances: np.ndarray, title: str
) -> plt.Figure:
    """Membuat grafik horizontal feature importance."""
    indices = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(len(indices)), importances[indices], color="#3B82F6", align="center")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Relative Importance", fontweight="bold")
    ax.set_title(title, pad=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def train_single_run(
    model_type: str,
    hyperparams: Dict[str, Any],
    run_name: Optional[str] = None,
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """Menjalankan satu sesi pelatihan model lengkap dengan pelacakan MLflow."""
    config = load_config(config_path)
    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = mlflow_cfg.get("experiment_name", "viralsense-virality-prediction")

    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(experiment_name)
    except mlflow.exceptions.MlflowException as e:
        if "deleted" in str(e).lower():
            client = mlflow.tracking.MlflowClient()
            for exp in client.search_experiments(view_type=mlflow.entities.ViewType.ALL):
                if exp.name == experiment_name and exp.lifecycle_stage == "deleted":
                    client.restore_experiment(exp.experiment_id)
                    mlflow.set_experiment(experiment_name)
                    break
        else:
            raise


    df, feature_cols, target_col = load_and_validate_dataset(config)
    training_cfg = config.get("model_training", {})
    test_size = float(training_cfg.get("test_size", 0.2))
    random_state = int(hyperparams.get("random_state", training_cfg.get("random_state", 42)))

    X_train, X_test, y_train, y_test = split_data(
        df, feature_cols, target_col, test_size=test_size, random_state=random_state
    )

    dataset_version = get_dataset_version()
    if not run_name:
        run_name = f"{model_type}_{hyperparams.get('n_estimators', hyperparams.get('C', 'default'))}"

    estimator = create_model(model_type, hyperparams)
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )

    logger.info(f"=== Memulai MLflow Run: '{run_name}' (Model: {model_type}) ===")
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # 1. Logging Tags & Metadata
        mlflow.set_tag("model_family", model_type)
        mlflow.set_tag("dataset_version", dataset_version)
        mlflow.set_tag("target_column", target_col)
        mlflow.set_tag("pipeline_author", "ViralSense MLOps Team")

        # 2. Logging Parameters
        logged_params = {
            "model_type": model_type,
            "test_size": test_size,
            "random_state": random_state,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_features": len(feature_cols),
            "feature_columns": ",".join(feature_cols),
            "dataset_version": dataset_version,
        }
        for k, v in hyperparams.items():
            logged_params[f"param_{k}"] = v

        mlflow.log_params(logged_params)

        # 3. Model Fitting
        pipeline.fit(X_train, y_train)

        # 4. Predictions & Metrics Computation
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        y_train_proba = None
        y_test_proba = None
        if hasattr(pipeline, "predict_proba"):
            y_train_proba = pipeline.predict_proba(X_train)[:, 1]
            y_test_proba = pipeline.predict_proba(X_test)[:, 1]

        test_metrics = evaluate_predictions(y_test, y_test_pred, y_test_proba)
        train_metrics = evaluate_predictions(y_train, y_train_pred, y_train_proba)

        logged_metrics = {
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1_score": test_metrics["f1_score"],
            "roc_auc": test_metrics["roc_auc"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1_score": train_metrics["f1_score"],
        }
        mlflow.log_metrics(logged_metrics)

        # 5. Logging Artifact Figures
        # Confusion Matrix
        cm_fig = plot_confusion_matrix(
            y_test, y_test_pred, title=f"Confusion Matrix - {run_name}"
        )
        mlflow.log_figure(cm_fig, "plots/confusion_matrix.png")
        plt.close(cm_fig)

        # ROC Curve
        if y_test_proba is not None:
            roc_fig = plot_roc_curve(
                y_test, y_test_proba, test_metrics["roc_auc"], title=f"ROC Curve - {run_name}"
            )
            mlflow.log_figure(roc_fig, "plots/roc_curve.png")
            plt.close(roc_fig)

        # Feature Importance (bila didukung)
        clf = pipeline.named_steps["classifier"]
        if hasattr(clf, "feature_importances_"):
            fi_fig = plot_feature_importance(
                feature_cols, clf.feature_importances_, title=f"Feature Importance - {run_name}"
            )
            mlflow.log_figure(fi_fig, "plots/feature_importance.png")
            plt.close(fi_fig)

        # 6. Logging Model Artifact with Signature & Input Example
        signature = infer_signature(X_train, y_train_pred)
        input_example = X_train.iloc[:3].copy()

        try:
            # MLflow 3.x syntax
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                signature=signature,
                input_example=input_example,
            )
        except TypeError:
            # Fallback for MLflow < 3.0
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )


        logger.info(
            f"Run '{run_name}' berhasil! Run ID: {run_id} | "
            f"Accuracy: {logged_metrics['accuracy']:.4f} | "
            f"F1-Score: {logged_metrics['f1_score']:.4f} | "
            f"ROC-AUC: {logged_metrics['roc_auc']:.4f}"
        )

        return {
            "run_id": run_id,
            "run_name": run_name,
            "model_type": model_type,
            "hyperparams": hyperparams,
            "metrics": logged_metrics,
        }


def run_experiment_suite(
    config_path: str = "config/config.yaml",
) -> List[Dict[str, Any]]:
    """Menjalankan minimal 3-5 variasi eksperimen dengan hyperparameter berbeda."""
    logger.info("=================================================================")
    logger.info("   MEMULAI SUITE EKSPERIMEN LK-06: 5 VARIASI HYPERPARAMETER     ")
    logger.info("=================================================================")

    experiment_presets = [
        {
            "run_name": "run1_rf_baseline_n50_d3",
            "model_type": "random_forest",
            "hyperparams": {
                "n_estimators": 50,
                "max_depth": 3,
                "min_samples_split": 2,
                "criterion": "gini",
                "random_state": 42,
            },
        },
        {
            "run_name": "run2_rf_deep_n150_d8",
            "model_type": "random_forest",
            "hyperparams": {
                "n_estimators": 150,
                "max_depth": 8,
                "min_samples_split": 4,
                "criterion": "gini",
                "random_state": 42,
            },
        },
        {
            "run_name": "run3_gbm_standard_lr0.1_n100",
            "model_type": "gradient_boosting",
            "hyperparams": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "subsample": 0.9,
                "random_state": 42,
            },
        },
        {
            "run_name": "run4_gbm_tuned_lr0.05_n200",
            "model_type": "gradient_boosting",
            "hyperparams": {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 4,
                "subsample": 0.85,
                "random_state": 42,
            },
        },
        {
            "run_name": "run5_svm_rbf_c1.0",
            "model_type": "svm",
            "hyperparams": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "random_state": 42,
            },
        },
    ]

    results = []
    for preset in experiment_presets:
        res = train_single_run(
            model_type=preset["model_type"],
            hyperparams=preset["hyperparams"],
            run_name=preset["run_name"],
            config_path=config_path,
        )
        results.append(res)

    print("\n" + "=" * 90)
    print(f"{'RINGKASAN PERBANDINGAN EKSPERIMEN LK-06 (MLFLOW)':^90}")
    print("=" * 90)
    header = f"{'Run Name':<30} | {'Model':<16} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<7} | {'F1-Score':<8} | {'ROC-AUC':<8}"
    print(header)
    print("-" * 90)
    for r in results:
        m = r["metrics"]
        print(
            f"{r['run_name']:<30} | {r['model_type']:<16} | {m['accuracy']:<9.4f} | "
            f"{m['precision']:<9.4f} | {m['recall']:<7.4f} | {m['f1_score']:<8.4f} | {m['roc_auc']:<8.4f}"
        )
    print("=" * 90 + "\n")

    return results


def register_best_model(
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """Memilih run terbaik berdasarkan F1-Score & ROC-AUC lalu mendaftarkannya ke Model Registry."""
    config = load_config(config_path)
    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = mlflow_cfg.get("experiment_name", "viralsense-virality-prediction")
    reg_model_name = mlflow_cfg.get("registered_model_name", "ViralSense-ViralityClassifier")

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        raise ValueError(f"Eksperimen '{experiment_name}' tidak ditemukan di MLflow store.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.f1_score DESC", "metrics.roc_auc DESC"],
        max_results=50,
    )

    if not runs:
        raise ValueError(f"Tidak ada runs yang ditemukan di eksperimen '{experiment_name}'.")

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_run_name = best_run.data.tags.get("mlflow.runName", "unknown")
    best_f1 = best_run.data.metrics.get("f1_score", 0.0)
    best_auc = best_run.data.metrics.get("roc_auc", 0.0)
    best_acc = best_run.data.metrics.get("accuracy", 0.0)

    logger.info(
        f"Model Terbaik Terpilih: '{best_run_name}' (Run ID: {best_run_id}) "
        f"dengan F1-Score: {best_f1:.4f}, ROC-AUC: {best_auc:.4f}, Accuracy: {best_acc:.4f}"
    )

    # Daftarkan ke Model Registry
    model_uri = f"runs:/{best_run_id}/model"
    reg_result = mlflow.register_model(model_uri=model_uri, name=reg_model_name)
    logger.info(
        f"Model berhasil didaftarkan ke Model Registry '{reg_model_name}' versi {reg_result.version}!"
    )

    # Tambahkan tag / deskripsi pada model version
    try:
        client.set_model_version_tag(
            name=reg_model_name,
            version=reg_result.version,
            key="stage",
            value="Champion",
        )
        client.set_model_version_tag(
            name=reg_model_name,
            version=reg_result.version,
            key="selection_metric",
            value=f"F1-Score: {best_f1:.4f}, ROC-AUC: {best_auc:.4f}",
        )
    except Exception as e:
        logger.warning(f"Gagal menambahkan tag ke model version: {e}")

    # Simpan metadata model ke JSON untuk luaran deployment
    metadata = {
        "registered_model_name": reg_model_name,
        "model_version": str(reg_result.version),
        "champion_run_id": best_run_id,
        "champion_run_name": best_run_name,
        "model_type": best_run.data.params.get("model_type", "unknown"),
        "metrics": {
            "f1_score": best_f1,
            "roc_auc": best_auc,
            "accuracy": best_acc,
            "precision": best_run.data.metrics.get("precision", 0.0),
            "recall": best_run.data.metrics.get("recall", 0.0),
        },
        "hyperparameters": {
            k.replace("param_", ""): v
            for k, v in best_run.data.params.items()
            if k.startswith("param_")
        },
        "dataset_lineage": {
            "dataset_version": best_run.data.tags.get("dataset_version", "data-v3.0"),
            "target_column": best_run.data.tags.get("target_column", "is_viral"),
            "features": best_run.data.params.get("feature_columns", "").split(","),
            "n_train_samples": best_run.data.params.get("n_train_samples"),
            "n_test_samples": best_run.data.params.get("n_test_samples"),
        },
        "deployment_status": "Candidate / Champion",
    }

    out_dir = PROJECT_ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_meta_path = out_dir / "model_metadata.json"
    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model metadata berhasil disimpan ke {out_meta_path}")

    return metadata


def parse_cli_args():
    """Menguraikan argumen command-line."""
    parser = argparse.ArgumentParser(
        description="Pelatihan Model Machine Learning & MLflow Tracking - ViralSense LK-06"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path ke file konfigurasi pipeline (default: config/config.yaml)",
    )
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Jalankan seluruh suite 5 variasi eksperimen secara berurutan",
    )
    parser.add_argument(
        "--register_best",
        action="store_true",
        help="Pilih model dengan metrik terbaik dan daftarkan ke Model Registry",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="rf",
        choices=["rf", "gbm", "svm", "lr", "random_forest", "gradient_boosting", "svc", "logistic_regression"],
        help="Tipe model untuk run tunggal",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Nama eksplisit untuk MLflow Run",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Jumlah estimators untuk pohon (Random Forest / Gradient Boosting)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Kedalaman maksimal pohon",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate untuk Gradient Boosting",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        help="Kernel untuk SVM ('rbf', 'linear', 'poly')",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Parameter regulasi C (SVM / Logistic Regression)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.85,
        help="Subsample ratio untuk Gradient Boosting",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Daftarkan model dari run ini langsung ke MLflow Model Registry",
    )
    return parser.parse_args()


def main():
    """Fungsi eksekusi utama."""
    args = parse_cli_args()

    if args.run_all:
        results = run_experiment_suite(config_path=args.config)
        if args.register_best:
            register_best_model(config_path=args.config)
    elif args.register_best:
        register_best_model(config_path=args.config)
    else:
        hyperparams = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "kernel": args.kernel,
            "C": args.C,
        }
        res = train_single_run(
            model_type=args.model_type,
            hyperparams=hyperparams,
            run_name=args.run_name,
            config_path=args.config,
        )
        if args.register:
            cfg = load_config(args.config)
            reg_name = cfg.get("mlflow", {}).get("registered_model_name", "ViralSense-ViralityClassifier")
            run_id = res["run_id"]
            model_uri = f"runs:/{run_id}/model"
            logger.info(f"Mendaftarkan model run {run_id} ke Model Registry '{reg_name}'...")
            reg_result = mlflow.register_model(model_uri=model_uri, name=reg_name)
            logger.info(
                f"Model berhasil didaftarkan sebagai versi {reg_result.version} di '{reg_name}'!"
            )
            client = mlflow.tracking.MlflowClient()
            client.set_model_version_tag(
                name=reg_name,
                version=reg_result.version,
                key="model_version_role",
                value="Candidate Update v2",
            )
            client.set_model_version_tag(
                name=reg_name,
                version=reg_result.version,
                key="selection_metric",
                value=f"F1-Score: {res['metrics']['f1_score']:.4f}, ROC-AUC: {res['metrics']['roc_auc']:.4f}",
            )


if __name__ == "__main__":
    main()
