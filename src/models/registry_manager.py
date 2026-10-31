"""Modul Pengelolaan Siklus Hidup Model Registry dan Silsilah Metadata DVC - ViralSense MLOps.

Modul ini bertanggung jawab untuk:
1. Mengelola pendaftaran dan riwayat versi model pada MLflow Model Registry.
2. Melakukan transisi stage industri: None -> Staging -> Production -> Archived.
3. Menetapkan tags, deskripsi, dan anotasi metrik pada model version.
4. Menyinkronkan metadata model aktif ke file YAML dan memetakannya ke hash DVC dataset.
5. Mendukung antarmuka CLI untuk kemudahan integrasi CI/CD dan reproduksibilitas.
"""

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Suppress noisy agent hints from MLflow
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import mlflow
from mlflow.tracking import MlflowClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("viralsense-registry")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_mlflow_client(config_path: str = "config/config.yaml") -> Tuple[MlflowClient, str, Dict[str, Any]]:
    """Menginisialisasi MlflowClient berdasarkan konfigurasi."""
    config_file = PROJECT_ROOT / config_path
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    model_name = mlflow_cfg.get("registered_model_name", "ViralSense-ViralityClassifier")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    return client, model_name, cfg


def list_model_versions(
    model_name: Optional[str] = None, config_path: str = "config/config.yaml"
) -> List[Dict[str, Any]]:
    """Mendapatkan seluruh versi model terdaftar beserta status stage dan metriknya."""
    client, default_name, _ = get_mlflow_client(config_path)
    target_name = model_name or default_name

    logger.info(f"Mengambil riwayat versi model: '{target_name}'...")
    try:
        registered_model = client.get_registered_model(target_name)
    except Exception as e:
        logger.error(f"Model '{target_name}' tidak ditemukan di Model Registry: {e}")
        return []

    versions = client.search_model_versions(f"name='{target_name}'")
    # Urutkan berdasarkan nomor versi integer
    versions = sorted(versions, key=lambda v: int(v.version))

    records = []
    print("\n" + "=" * 105)
    print(f"{f'MLFLOW MODEL REGISTRY: {target_name}':^105}")
    print("=" * 105)
    header = f"{'Version':<9} | {'Current Stage':<16} | {'Run ID':<34} | {'Status':<10} | {'Tags':<25}"
    print(header)
    print("-" * 105)

    for v in versions:
        run = client.get_run(v.run_id)
        f1 = run.data.metrics.get("f1_score", "N/A")
        f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
        tags_str = ", ".join([f"{k}={val}" for k, val in v.tags.items() if not k.startswith("mlflow.")])
        if not tags_str:
            tags_str = f"F1={f1_str}"

        print(f"v{v.version:<8} | {v.current_stage:<16} | {v.run_id:<34} | {v.status:<10} | {tags_str[:25]}")
        records.append({
            "version": v.version,
            "stage": v.current_stage,
            "run_id": v.run_id,
            "status": v.status,
            "tags": v.tags,
            "description": v.description,
            "source": v.source,
        })

    print("=" * 105 + "\n")
    return records


def transition_model_stage(
    version: int,
    stage: str,
    model_name: Optional[str] = None,
    archive_existing: bool = False,
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """Mengubah stage model (None, Staging, Production, Archived)."""
    client, default_name, _ = get_mlflow_client(config_path)
    target_name = model_name or default_name

    valid_stages = ["None", "Staging", "Production", "Archived"]
    normalized_stage = stage.capitalize() if stage.lower() != "none" else "None"
    if normalized_stage not in valid_stages:
        raise ValueError(f"Stage '{stage}' tidak valid. Pilihan: {valid_stages}")

    logger.info(
        f"Mentransisikan '{target_name}' v{version} ke Stage '{normalized_stage}' "
        f"(archive_existing_versions={archive_existing})..."
    )

    result = client.transition_model_version_stage(
        name=target_name,
        version=str(version),
        stage=normalized_stage,
        archive_existing_versions=archive_existing,
    )

    # Tambahkan tag riwayat transisi
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    client.set_model_version_tag(
        name=target_name,
        version=str(version),
        key="last_stage_transition",
        value=f"{normalized_stage} at {timestamp}",
    )
    if normalized_stage == "Production":
        client.set_model_version_tag(
            name=target_name,
            version=str(version),
            key="stage",
            value="Production",
        )

    logger.info(
        f"Transisi berhasil! '{target_name}' v{version} saat ini berstatus '{result.current_stage}'."
    )
    return {
        "model_name": target_name,
        "version": result.version,
        "stage": result.current_stage,
        "timestamp": timestamp,
    }


def promote_to_production(
    version: int,
    reason: str = "Promoted as Champion model after passing validation benchmarks",
    model_name: Optional[str] = None,
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """Mempromosikan versi model ke Production dan mengarsipkan versi Production sebelumnya."""
    client, default_name, _ = get_mlflow_client(config_path)
    target_name = model_name or default_name

    logger.info(f"=== PROMOSI MODEL KE PRODUCTION: '{target_name}' v{version} ===")
    res = transition_model_stage(
        version=version,
        stage="Production",
        model_name=target_name,
        archive_existing=True,
        config_path=config_path,
    )

    client.set_model_version_tag(
        name=target_name,
        version=str(version),
        key="promotion_reason",
        value=reason,
    )
    client.update_model_version(
        name=target_name,
        version=str(version),
        description=f"Active Production Model for ViralSense Virality Prediction. Reason: {reason}",
    )

    logger.info(f"Model v{version} resmi dipromosikan ke Production dan siap melayani inferensi!")
    return res


def extract_dvc_dataset_info() -> Dict[str, Any]:
    """Mengekstrak informasi hash dan metadata dataset dari file .dvc."""
    dvc_file = PROJECT_ROOT / "data/raw/sample_raw_videos.csv.dvc"
    info = {
        "dvc_file": "data/raw/sample_raw_videos.csv.dvc",
        "dataset_tag": "data-v3.0",
        "md5_hash": "unknown",
        "size_bytes": 0,
        "remote_name": "r2-storage",
    }
    if dvc_file.exists():
        with open(dvc_file, "r", encoding="utf-8") as f:
            dvc_data = yaml.safe_load(f)
            outs = dvc_data.get("outs", [])
            if outs:
                info["md5_hash"] = outs[0].get("md5", "unknown")
                info["size_bytes"] = outs[0].get("size", 0)
    return info


def sync_production_metadata(
    model_name: Optional[str] = None,
    config_path: str = "config/config.yaml",
    output_yaml: str = "models/production_model.yaml",
    output_json: str = "models/model_metadata.json",
) -> Dict[str, Any]:
    """Menyinkronkan metadata versi Production yang aktif ke file YAML dan JSON berformat DVC."""
    client, default_name, config = get_mlflow_client(config_path)
    target_name = model_name or default_name

    # Cari model version yang aktif di Production
    prod_versions = [
        v for v in client.search_model_versions(f"name='{target_name}'")
        if v.current_stage == "Production"
    ]
    if not prod_versions:
        # Fallback cari versi tertinggi jika stage belum Production
        logger.warning(
            f"Tidak ada versi '{target_name}' dengan stage 'Production'. Menggunakan versi terbaru."
        )
        all_v = sorted(client.search_model_versions(f"name='{target_name}'"), key=lambda v: int(v.version))
        if not all_v:
            raise ValueError(f"Tidak ada model version yang ditemukan untuk '{target_name}'.")
        active_version = all_v[-1]
    else:
        active_version = prod_versions[-1]

    run = client.get_run(active_version.run_id)
    dvc_info = extract_dvc_dataset_info()

    training_cfg = config.get("model_training", {})
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

    metadata = {
        "model_registry": {
            "name": target_name,
            "version": int(active_version.version),
            "stage": active_version.current_stage,
            "status": active_version.status,
            "source_run_id": active_version.run_id,
            "source_run_name": run.data.tags.get("mlflow.runName", "unknown"),
            "artifact_uri": active_version.source,
            "framework": "scikit-learn",
            "registered_at": datetime.datetime.fromtimestamp(active_version.creation_timestamp / 1000.0).strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "data_lineage": {
            "dataset_tag": dvc_info["dataset_tag"],
            "dvc_file": dvc_info["dvc_file"],
            "dvc_data_hash_md5": dvc_info["md5_hash"],
            "raw_dataset_size_bytes": dvc_info["size_bytes"],
            "features_source": config.get("storage", {}).get("features_data_path", "data/processed/features_matrix.parquet"),
            "target_column": training_cfg.get("target_column", "is_viral"),
            "feature_columns": feature_cols,
            "n_features": len(feature_cols),
            "n_train_samples": int(run.data.params.get("n_train_samples", 271)),
            "n_test_samples": int(run.data.params.get("n_test_samples", 68)),
        },
        "performance_metrics": {
            "accuracy": float(run.data.metrics.get("accuracy", 0.0)),
            "f1_score": float(run.data.metrics.get("f1_score", 0.0)),
            "roc_auc": float(run.data.metrics.get("roc_auc", 0.0)),
            "precision": float(run.data.metrics.get("precision", 0.0)),
            "recall": float(run.data.metrics.get("recall", 0.0)),
            "train_accuracy": float(run.data.metrics.get("train_accuracy", 0.0)),
            "train_f1_score": float(run.data.metrics.get("train_f1_score", 0.0)),
        },
        "hyperparameters": {
            k.replace("param_", ""): v
            for k, v in run.data.params.items()
            if k.startswith("param_")
        },
        "inference_specification": {
            "pyfunc_model_uri": f"models:/{target_name}/{active_version.current_stage}",
            "version_model_uri": f"models:/{target_name}/{active_version.version}",
            "prediction_classes": [0, 1],
            "class_labels": {0: "Non-Viral", 1: "Viral"},
            "confidence_threshold": 0.50,
        },
    }

    # Simpan ke YAML
    yaml_file = PROJECT_ROOT / output_yaml
    yaml_file.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Metadata Produksi YAML berhasil disimpan ke: {yaml_file}")

    # Simpan juga ke JSON
    json_file = PROJECT_ROOT / output_json
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata Produksi JSON berhasil disinkronkan ke: {json_file}")

    return metadata


def parse_args():
    """Menguraikan argumen CLI."""
    parser = argparse.ArgumentParser(
        description="MLflow Model Registry & DVC Lineage Lifecycle Manager - ViralSense LK-07"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path ke konfigurasi pipeline (default: config/config.yaml)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Nama terdaftar di Model Registry",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Tampilkan seluruh versi model terdaftar beserta stage",
    )
    parser.add_argument(
        "--transition",
        action="store_true",
        help="Lakukan transisi stage pada versi tertentu",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Nomor versi model yang akan ditransisikan",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["None", "Staging", "Production", "Archived", "none", "staging", "production", "archived"],
        help="Target stage (None, Staging, Production, Archived)",
    )
    parser.add_argument(
        "--archive_existing",
        action="store_true",
        help="Otomatis arsipkan versi lain saat pindah ke Production",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promosikan versi ke Production dan arsipkan versi lainnya",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default="Promoted via ViralSense MLOps Lifecycle Manager",
        help="Alasan promosi atau transisi",
    )
    parser.add_argument(
        "--sync_dvc",
        action="store_true",
        help="Sinkronisasi metadata model aktif ke YAML/JSON untuk pelacakan DVC",
    )
    return parser.parse_args()


def main():
    """Fungsi eksekusi CLI."""
    args = parse_args()

    if args.list:
        list_model_versions(model_name=args.model_name, config_path=args.config)

    elif args.transition:
        if args.version is None or args.stage is None:
            logger.error("Argumen --version dan --stage wajib disertakan saat melakukan --transition.")
            return
        transition_model_stage(
            version=args.version,
            stage=args.stage,
            model_name=args.model_name,
            archive_existing=args.archive_existing,
            config_path=args.config,
        )

    elif args.promote:
        if args.version is None:
            logger.error("Argumen --version wajib disertakan saat melakukan --promote.")
            return
        promote_to_production(
            version=args.version,
            reason=args.reason,
            model_name=args.model_name,
            config_path=args.config,
        )

    elif args.sync_dvc:
        sync_production_metadata(model_name=args.model_name, config_path=args.config)

    else:
        # Default: list model versions
        list_model_versions(model_name=args.model_name, config_path=args.config)


if __name__ == "__main__":
    main()
