"""Modul Evaluasi Otomatis dan Validasi Ambang Batas LK-01 - ViralSense MLOps.

Modul ini bertindak sebagai Automated Gatekeeper dalam alur kerja CI/CD (Code as a Trigger):
1. Memuat run kandidat terbaru dari MLflow / candidate_run.json.
2. Menguji inferensi latensi menggunakan skenario benchmark YouTube Shorts.
3. Memvalidasi metrik terhadap kriteria keberhasilan LK-01:
   - Akurasi >= 82% (0.82)
   - Macro F1-Score >= 0.80
   - Latensi Inferensi < 200 ms (p95)
   - ROC-AUC >= 0.85
4. Jika Lolos Validasi (PASSED):
   - Mendaftarkan model ke MLflow Model Registry.
   - Mentransisikan status model ke stage 'Staging'.
   - Menyematkan metadata lineage (DVC hash, commit SHA, metrik evaluasi).
   - Menghasilkan ringkasan laporan evaluasi format Markdown & JSON.
5. Jika Gagal Validasi (FAILED):
   - Menghentikan alur kerja dengan exit code non-nol untuk mencegah silent failure di produksi.
"""

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# Suppress noisy agent hints
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("viralsense-eval")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Memuat konfigurasi pipeline dari file YAML."""
    full_path = PROJECT_ROOT / config_path if not Path(config_path).is_absolute() else Path(config_path)
    if not full_path.exists():
        logger.warning(f"File konfigurasi {full_path} tidak ditemukan. Menggunakan konfigurasi default.")
        return {}
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_git_commit_info() -> Dict[str, str]:
    """Mengambil commit SHA dan branch terkini dari Git."""
    info = {"commit_sha": "unknown", "branch": "unknown", "commit_msg": ""}
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
        info["commit_sha"] = sha[:8]
    except Exception:
        pass

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["branch"] = branch
    except Exception:
        pass

    try:
        msg = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%B"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["commit_msg"] = msg.split("\n")[0]
    except Exception:
        pass

    return info


def get_dvc_dataset_hash() -> str:
    """Mengambil hash dataset raw yang dilacak DVC."""
    dvc_file = PROJECT_ROOT / "data/raw/sample_raw_videos.csv.dvc"
    if dvc_file.exists():
        try:
            with open(dvc_file, "r", encoding="utf-8") as f:
                dvc_data = yaml.safe_load(f)
                outs = dvc_data.get("outs", [])
                if outs:
                    return str(outs[0].get("md5", "unknown"))
        except Exception:
            pass
    return "unknown"


def resolve_candidate_run(
    run_id: Optional[str] = None, config_path: str = "config/config.yaml"
) -> Tuple[str, Dict[str, Any]]:
    """Menentukan run ID model kandidat yang akan divalidasi."""
    cfg = load_config(config_path)
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = mlflow_cfg.get("experiment_name", "viralsense-virality-prediction")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # 1. Bila run_id diberikan via CLI
    if run_id:
        run = client.get_run(run_id)
        return run_id, run.data.to_dictionary()

    # 2. Coba periksa file candidate_run.json
    cand_path_str = cfg.get("evaluation", {}).get("candidate_run_path", "models/candidate_run.json")
    cand_file = PROJECT_ROOT / cand_path_str
    if cand_file.exists():
        try:
            with open(cand_file, "r", encoding="utf-8") as f:
                cand_data = json.load(f)
                saved_run_id = cand_data.get("run_id")
                if saved_run_id:
                    run = client.get_run(saved_run_id)
                    logger.info(f"Menggunakan kandidat run dari {cand_path_str}: {saved_run_id}")
                    return saved_run_id, run.data.to_dictionary()
        except Exception as e:
            logger.warning(f"Gagal membaca {cand_file}: {e}")

    # 3. Fallback: Cari run terbaru di MLflow experiment
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        raise ValueError(f"Experiment MLflow '{experiment_name}' tidak ditemukan.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"Tidak ada run yang ditemukan di dalam eksperimen '{experiment_name}'.")

    latest_run = runs[0]
    logger.info(f"Menggunakan run MLflow terbaru: '{latest_run.info.run_name}' ({latest_run.info.run_id})")
    return latest_run.info.run_id, latest_run.data.to_dictionary()


def measure_inference_latency(
    run_id: str,
    n_iterations: int = 20,
) -> Dict[str, float]:
    """Mengukur latensi inferensi sampel tunggal dan batch p95."""
    from src.models.verify_inference import get_inference_samples

    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Memuat model kandidat dari {model_uri} untuk benchmark latensi...")
    model = mlflow.pyfunc.load_model(model_uri)
    df_samples, _ = get_inference_samples()

    # Warmup
    _ = model.predict(df_samples.iloc[[0]])

    latencies = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = model.predict(df_samples.iloc[[0]])
        latencies.append((time.perf_counter() - start) * 1000.0)

    # Batch latency
    start_batch = time.perf_counter()
    _ = model.predict(df_samples)
    batch_time_ms = (time.perf_counter() - start_batch) * 1000.0

    return {
        "single_latency_avg_ms": float(np.mean(latencies)),
        "single_latency_p95_ms": float(np.percentile(latencies, 95)),
        "batch_latency_ms": float(batch_time_ms),
        "batch_per_sample_ms": float(batch_time_ms / len(df_samples)),
    }


def evaluate_candidate_model(
    run_id: Optional[str] = None,
    config_path: str = "config/config.yaml",
    auto_stage: bool = True,
) -> Dict[str, Any]:
    """Menjalankan evaluasi komprehensif terhadap ambang batas performa LK-01."""
    cfg = load_config(config_path)
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    model_name = mlflow_cfg.get("registered_model_name", "ViralSense-ViralityClassifier")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Resolusi Run ID kandidat
    candidate_run_id, run_data = resolve_candidate_run(run_id, config_path)
    metrics = run_data.get("metrics", {})
    params = run_data.get("params", {})
    tags = run_data.get("tags", {})

    accuracy = float(metrics.get("accuracy", 0.0))
    f1_score = float(metrics.get("f1_score", 0.0))
    roc_auc = float(metrics.get("roc_auc", 0.0))
    precision = float(metrics.get("precision", 0.0))
    recall = float(metrics.get("recall", 0.0))

    # Benchmark Latensi
    latency_stats = measure_inference_latency(candidate_run_id)
    latency_p95 = latency_stats["single_latency_p95_ms"]

    # Ambang batas dari konfigurasi LK-01
    eval_cfg = cfg.get("evaluation", {})
    thresholds = eval_cfg.get("thresholds", {})
    min_accuracy = float(thresholds.get("min_accuracy", 0.82))
    min_f1 = float(thresholds.get("min_f1_score", 0.80))
    max_latency = float(thresholds.get("max_latency_ms", 200.0))
    min_roc_auc = float(thresholds.get("min_roc_auc", 0.85))

    # Pengujian Gerbang Kualitas (Quality Gates)
    gates = [
        {
            "metric": "Accuracy",
            "actual": accuracy,
            "threshold": f">= {min_accuracy:.2%}",
            "passed": accuracy >= min_accuracy,
            "rule": "LK-01 Tabel 4.1: Target Akurasi >= 82%",
        },
        {
            "metric": "Macro F1-Score",
            "actual": f1_score,
            "threshold": f">= {min_f1:.4f}",
            "passed": f1_score >= min_f1,
            "rule": "LK-01 Tabel 4.1: Target F1-Score >= 0.80",
        },
        {
            "metric": "ROC-AUC Score",
            "actual": roc_auc,
            "threshold": f">= {min_roc_auc:.4f}",
            "passed": roc_auc >= min_roc_auc,
            "rule": "Stabilitas separasi probabilitas kelas viral",
        },
        {
            "metric": "Inference Latency (p95)",
            "actual": latency_p95,
            "threshold": f"< {max_latency:.1f} ms",
            "passed": latency_p95 < max_latency,
            "rule": "LK-01 Tabel 4.1: Latensi Inferensi < 200 ms (p95)",
        },
    ]

    all_passed = all(g["passed"] for g in gates)
    verdict = "PASSED" if all_passed else "FAILED"

    # Cetak Tampilan Evaluasi CLI yang Rapi
    print("\n" + "=" * 105)
    print(f"{'VIRALSENSE MLOPS - AUTOMATED MODEL EVALUATION & VALIDATION GATE':^105}")
    print(f"{'Kriteria Keberhasilan & Ambang Batas LK-01':^105}")
    print("=" * 105)
    print(f"Candidate Run ID : {candidate_run_id}")
    print(f"Run Name         : {tags.get('mlflow.runName', 'unknown')}")
    print(f"Model Type       : {params.get('model_type', 'unknown')}")
    print(f"Dataset Version  : {tags.get('dataset_version', 'unknown')}")
    print("-" * 105)
    print(f"{'Metrik':<26} | {'Nilai Aktual':<14} | {'LK-01 Threshold':<18} | {'Status':<10} | {'Aturan':<30}")
    print("-" * 105)
    for g in gates:
        act_str = f"{g['actual']:.4f}" if isinstance(g["actual"], float) else str(g["actual"])
        if g["metric"] == "Accuracy":
            act_str = f"{g['actual']:.2%}"
        elif "Latency" in g["metric"]:
            act_str = f"{g['actual']:.2f} ms"

        stat_str = "✅ PASS" if g["passed"] else "❌ FAIL"
        print(f"{g['metric']:<26} | {act_str:<14} | {g['threshold']:<18} | {stat_str:<10} | {g['rule']:<30}")
    print("=" * 105)
    print(f"VERDIK VALIDASI AKHIR: {verdict.upper()} ({'Lolos kualifikasi Staging' if all_passed else 'Ditolak: Terjadi degradasi performa'})")
    print("=" * 105 + "\n")

    # Informasi Git & DVC
    git_info = get_git_commit_info()
    dvc_hash = get_dvc_dataset_hash()

    new_model_version = None
    staged_stage = None

    # TAHAP 4: AUTO-REGISTRY UPDATE BILA VALIDASI LOLOS
    if all_passed and auto_stage:
        logger.info(f"Model lolos seluruh pengujian! Mendaftarkan ke MLflow Model Registry '{model_name}'...")
        model_uri = f"runs:/{candidate_run_id}/model"
        reg_result = mlflow.register_model(model_uri=model_uri, name=model_name)
        new_model_version = str(reg_result.version)

        # Update deskripsi model version
        client.update_model_version(
            name=model_name,
            version=new_model_version,
            description=(
                f"Candidate model automatically validated via CI/CD (Code as a Trigger). "
                f"Accuracy={accuracy:.2%}, F1={f1_score:.4f}, Latency={latency_p95:.2f}ms. "
                f"Git Commit: {git_info['commit_sha']}."
            ),
        )

        # Transisikan ke stage Staging
        logger.info(f"Mentransisikan '{model_name}' v{new_model_version} ke stage 'Staging'...")
        client.transition_model_version_stage(
            name=model_name,
            version=new_model_version,
            stage="Staging",
            archive_existing_versions=False,
        )
        staged_stage = "Staging"

        # Sematkan metadata tags yang kaya pada model version
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tags_to_set = {
            "stage": "Staging",
            "validation_status": "PASSED",
            "trigger_event": "CI/CD Code as a Trigger",
            "git_commit": git_info["commit_sha"],
            "git_branch": git_info["branch"],
            "dvc_dataset_hash": dvc_hash,
            "eval_accuracy": f"{accuracy:.4f}",
            "eval_f1_score": f"{f1_score:.4f}",
            "eval_latency_p95_ms": f"{latency_p95:.2f}",
            "evaluated_at": now_str,
        }
        for k, v in tags_to_set.items():
            client.set_model_version_tag(
                name=model_name, version=new_model_version, key=k, value=str(v)
            )

        logger.info(
            f"🎉 SUKSES! Model '{model_name}' versi {new_model_version} kini resmi aktif di stage 'Staging'!"
        )

    # Siapkan Laporan Evaluasi
    report_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_run_id": candidate_run_id,
        "run_name": tags.get("mlflow.runName", "unknown"),
        "model_name": model_name,
        "model_version": new_model_version,
        "current_stage": staged_stage or "None",
        "verdict": verdict,
        "all_passed": all_passed,
        "metrics": {
            "accuracy": accuracy,
            "f1_score": f1_score,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "single_latency_avg_ms": latency_stats["single_latency_avg_ms"],
            "single_latency_p95_ms": latency_p95,
            "batch_latency_ms": latency_stats["batch_latency_ms"],
        },
        "quality_gates": gates,
        "lineage": {
            "git_commit": git_info["commit_sha"],
            "git_branch": git_info["branch"],
            "git_commit_msg": git_info["commit_msg"],
            "dvc_dataset_hash": dvc_hash,
        },
    }

    # Simpan laporan JSON
    out_json = PROJECT_ROOT / eval_cfg.get("evaluation_report_path", "models/evaluation_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Simpan ringkasan Markdown (berguna untuk GitHub Actions Job Summary)
    out_md = PROJECT_ROOT / eval_cfg.get("evaluation_summary_md", "models/evaluation_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# 🧪 ViralSense Model Evaluation & Quality Gate Report\n\n")
        f.write(f"**Status Validasi:** {'🟢 **PASSED (Promoted to Staging)**' if all_passed else '🔴 **FAILED (Rejected)**'}\n\n")
        f.write(f"- **Model Registry:** `{model_name}` (Version: `v{new_model_version or 'N/A'}`)\n")
        f.write(f"- **Stage:** `{staged_stage or 'Rejected'}`\n")
        f.write(f"- **Candidate Run ID:** `{candidate_run_id}`\n")
        f.write(f"- **Git Commit:** `{git_info['commit_sha']}` (`{git_info['branch']}`)\n")
        f.write(f"- **DVC Dataset Hash:** `{dvc_hash}`\n\n")
        f.write("### Kriteria Ambang Batas LK-01\n\n")
        f.write("| Metrik | Nilai Aktual | Ambang Batas (LK-01) | Status |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for g in gates:
            act = f"{g['actual']:.2%}" if g["metric"] == "Accuracy" else (f"{g['actual']:.2f} ms" if "Latency" in g["metric"] else f"{g['actual']:.4f}")
            st = "✅ PASSED" if g["passed"] else "❌ FAILED"
            f.write(f"| {g['metric']} | {act} | {g['threshold']} | {st} |\n")
        f.write("\n")
        if all_passed:
            f.write("> [!NOTE]\n")
            f.write("> Model kandidat memenuhi seluruh standar kualitas dan latensi produksi. Model secara otomatis didaftarkan dan dipromosikan ke tahap **Staging**.\n")
        else:
            f.write("> [!CAUTION]\n")
            f.write("> Model kandidat gagal memenuhi salah satu kriteria ambang batas minimum. Promosi otomatis dibatalkan untuk mencegah *silent failure* pada sistem produksi.\n")

    logger.info(f"Laporan evaluasi berhasil ditulis ke {out_json} dan {out_md}")
    return report_data


def parse_args():
    """Menguraikan argumen CLI."""
    parser = argparse.ArgumentParser(
        description="Automated Model Evaluation & Validation Gatekeeper - ViralSense LK-08"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="ID Run MLflow spesifik yang akan dievaluasi (default: kandidat run terbaru)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path ke file konfigurasi pipeline",
    )
    parser.add_argument(
        "--no_stage",
        action="store_true",
        help="Jangan otomatis transisi ke Staging meskipun lolos evaluasi",
    )
    return parser.parse_args()


def main():
    """Entry point eksekusi skrip evaluasi."""
    args = parse_args()
    try:
        report = evaluate_candidate_model(
            run_id=args.run_id,
            config_path=args.config,
            auto_stage=not args.no_stage,
        )
        if not report["all_passed"]:
            logger.error("Evaluasi model gagal memenuhi kriteria ambang batas LK-01! Menghentikan pipeline.")
            sys.exit(1)
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Terjadi kesalahan fatal selama evaluasi model: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
