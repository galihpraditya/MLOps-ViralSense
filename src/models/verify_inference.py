"""Skrip Pengujian dan Verifikasi Kesiapan Inferensi Produksi - ViralSense MLOps.

Modul ini memverifikasi bahwa model yang berstatus 'Production' di MLflow Model Registry
dapat dimuat dan dieksekusi secara programatik menggunakan `mlflow.pyfunc.load_model`.
Skrip ini mensimulasikan lingkungan layanan inferensi (serving engine) dalam menangani
payload video YouTube Shorts secara real-time maupun batch.
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

# Suppress MLflow noisy hints
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import mlflow
import mlflow.pyfunc
import mlflow.sklearn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("viralsense-inference")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_inference_samples() -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Menyediakan dataset simulasi skenario video Shorts untuk uji inferensi."""
    scenarios = [
        {
            "case_id": "TC-01",
            "scenario": "High Velocity Viral Short (Strong Hook + High Engagement)",
            "expected_label": 1,
            "features": {
                "duration_seconds": 15,
                "video_age_hours": 4.5,
                "engagement_rate": 0.125,
                "comment_to_like_ratio": 0.18,
                "caption_length": 95,
                "word_count": 14,
                "has_hook_question": 1,
            },
        },
        {
            "case_id": "TC-02",
            "scenario": "Low Engagement Organic Short (Mature Age + Low Interaction)",
            "expected_label": 0,
            "features": {
                "duration_seconds": 58,
                "video_age_hours": 144.0,
                "engagement_rate": 0.008,
                "comment_to_like_ratio": 0.015,
                "caption_length": 210,
                "word_count": 32,
                "has_hook_question": 0,
            },
        },
        {
            "case_id": "TC-03",
            "scenario": "Trending Mystery/Humor Short (Fast Hook + High Comments)",
            "expected_label": 1,
            "features": {
                "duration_seconds": 22,
                "video_age_hours": 8.0,
                "engagement_rate": 0.095,
                "comment_to_like_ratio": 0.22,
                "caption_length": 110,
                "word_count": 17,
                "has_hook_question": 1,
            },
        },
        {
            "case_id": "TC-04",
            "scenario": "Promotional Catalog / Product Showcase (Flat Engagement)",
            "expected_label": 0,
            "features": {
                "duration_seconds": 45,
                "video_age_hours": 72.0,
                "engagement_rate": 0.015,
                "comment_to_like_ratio": 0.02,
                "caption_length": 340,
                "word_count": 48,
                "has_hook_question": 0,
            },
        },
    ]

    records = [s["features"] for s in scenarios]
    df = pd.DataFrame(records)
    # Penyelarasan tipe data sesuai signature schema MLflow
    df["duration_seconds"] = df["duration_seconds"].astype(np.int32)
    df["video_age_hours"] = df["video_age_hours"].astype(np.float64)
    df["engagement_rate"] = df["engagement_rate"].astype(np.float64)
    df["comment_to_like_ratio"] = df["comment_to_like_ratio"].astype(np.float64)
    df["caption_length"] = df["caption_length"].astype(np.int64)
    df["word_count"] = df["word_count"].astype(np.int64)
    df["has_hook_question"] = df["has_hook_question"].astype(np.int64)
    return df, scenarios


def load_production_model(
    model_name: str = "ViralSense-ViralityClassifier",
    stage: str = "Production",
    tracking_uri: str = "sqlite:///mlflow.db",
) -> Any:
    """Memuat model terdaftar pada stage tertentu menggunakan pyfunc loader."""
    mlflow.set_tracking_uri(tracking_uri)

    # Coba muat berdasarkan stage resmi
    model_uri_stage = f"models:/{model_name}/{stage}"
    logger.info(f"Mencoba memuat model produksi dari URI: '{model_uri_stage}'...")

    try:
        model = mlflow.pyfunc.load_model(model_uri_stage)
        logger.info(f"Model berhasil dimuat dari stage '{stage}' via mlflow.pyfunc!")
        return model, model_uri_stage
    except Exception as e_stage:
        logger.warning(f"Gagal memuat via stage '{model_uri_stage}': {e_stage}")

        # Fallback coba via alias '@production' atau versi aktif
        alias_uri = f"models:/{model_name}@production"
        logger.info(f"Mencoba memuat model fallback dari alias: '{alias_uri}'...")
        try:
            model = mlflow.pyfunc.load_model(alias_uri)
            logger.info(f"Model berhasil dimuat dari alias '@production' via mlflow.pyfunc!")
            return model, alias_uri
        except Exception as e_alias:
            logger.error(f"Gagal memuat via alias: {e_alias}")
            raise RuntimeError(f"Tidak dapat memuat model '{model_name}' pada stage '{stage}'.")


def run_inference_verification(
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """Menjalankan suite pengujian inferensi komprehensif pada model Production."""
    config_file = PROJECT_ROOT / config_path
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    model_name = mlflow_cfg.get("registered_model_name", "ViralSense-ViralityClassifier")

    logger.info("=========================================================================")
    logger.info("       MEMULAI SUITE VERIFIKASI INFERENSI MODEL REGISTRY PRODUCTION       ")
    logger.info("=========================================================================")

    # 1. Pemuatan Model
    start_load = time.perf_counter()
    model, resolved_uri = load_production_model(
        model_name=model_name, stage="Production", tracking_uri=tracking_uri
    )
    load_time_ms = (time.perf_counter() - start_load) * 1000

    # Pemuatan Scikit-Learn flavor untuk ekstraksi probabilitas jika didukung
    sklearn_model = None
    try:
        sklearn_model = mlflow.sklearn.load_model(resolved_uri)
    except Exception:
        pass

    # 2. Persiapan Data Uji
    df_samples, scenarios = get_inference_samples()

    # 3. Uji Inferensi Tunggal (Single-Record Latency)
    single_record = df_samples.iloc[[0]]
    start_single = time.perf_counter()
    single_pred = model.predict(single_record)
    single_latency_ms = (time.perf_counter() - start_single) * 1000

    # 4. Uji Inferensi Batch (Batch Latency)
    start_batch = time.perf_counter()
    batch_preds = model.predict(df_samples)
    batch_latency_ms = (time.perf_counter() - start_batch) * 1000
    avg_per_sample_ms = batch_latency_ms / len(df_samples)

    # 5. Probabilitas Konfidensi
    if sklearn_model is not None and hasattr(sklearn_model, "predict_proba"):
        probas = sklearn_model.predict_proba(df_samples)[:, 1]
    else:
        probas = np.array([float(p) for p in batch_preds])

    # 6. Evaluasi dan Pelaporan
    results_table = []
    print("\n" + "=" * 115)
    print(f"{'HASIL PENGUJIAN INFERENSI MODEL PRODUCTION':^115}")
    print(f"{f'Model URI: {resolved_uri} | Waktu Pemuatan: {load_time_ms:.2f} ms':^115}")
    print("=" * 115)
    header = f"{'Case ID':<8} | {'Scenario':<42} | {'Exp':<5} | {'Pred':<5} | {'Label':<10} | {'Probability':<12} | {'Status':<8}"
    print(header)
    print("-" * 115)

    all_matched = True
    for i, sc in enumerate(scenarios):
        pred_val = int(batch_preds[i])
        prob_val = float(probas[i])
        exp_val = sc["expected_label"]
        status = "PASSED" if pred_val == exp_val else "REVIEW"
        if pred_val != exp_val:
            all_matched = False

        label_name = "VIRAL (1)" if pred_val == 1 else "NON-VIRAL(0)"
        row = f"{sc['case_id']:<8} | {sc['scenario'][:42]:<42} | {exp_val:<5} | {pred_val:<5} | {label_name:<10} | {prob_val * 100:6.2f}%     | {status:<8}"
        print(row)
        results_table.append({
            "case_id": sc["case_id"],
            "scenario": sc["scenario"],
            "expected": exp_val,
            "predicted": pred_val,
            "probability": prob_val,
            "status": status,
        })

    print("=" * 115)
    print(f"Single-record Latency : {single_latency_ms:.2f} ms")
    print(f"Batch Processing Time : {batch_latency_ms:.2f} ms ({len(df_samples)} records)")
    print(f"Throughput Simulasi   : {1000.0 / avg_per_sample_ms:.1f} requests/detik")
    print("=" * 115 + "\n")

    summary = {
        "model_uri": resolved_uri,
        "load_time_ms": round(load_time_ms, 2),
        "single_latency_ms": round(single_latency_ms, 2),
        "batch_latency_ms": round(batch_latency_ms, 2),
        "average_latency_per_sample_ms": round(avg_per_sample_ms, 2),
        "all_tests_passed": all_matched,
        "test_cases": results_table,
    }

    if all_matched and single_latency_ms < 50.0:
        logger.info("VERIFIKASI INFERENSI BERHASIL: Model Production siap melayani inferensi real-time!")
    else:
        logger.info(f"Verifikasi inferensi selesai. Status kelulusan tes: {all_matched}")

    return summary


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Verifikasi Kesiapan Inferensi Model Production MLflow")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path konfigurasi pipeline")
    args = parser.parse_args()

    run_inference_verification(config_path=args.config)


if __name__ == "__main__":
    main()
