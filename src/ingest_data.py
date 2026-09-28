"""Skrip Pengumpul Data (Data Ingestion) Otomatis - ViralSense MLOps.

Mengambil data dinamis secara berkala dari sumber yang telah ditentukan (YouTube Shorts dan TikTok FYP)
menggunakan Requests, YouTube API, dan yt-dlp.
Mendukung simulasi periodik dengan penyimpanan snapshot berbasis timestamp (append-only)
sehingga data lama tidak tertimpa secara destruktif, siap mendukung konsep Continual Learning.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# Tambahkan direktori root ke sys.path jika belum ada
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingest_tiktok import extract_tiktok_videos
from src.data.ingest_youtube import extract_youtube_shorts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest_data")


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Memuat konfigurasi pipeline dari file YAML."""
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        logger.warning(f"File konfigurasi {full_path} tidak ditemukan, menggunakan nilai default.")
        return {}
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_platform_snapshot(
    data: List[Dict[str, Any]],
    platform: str,
    raw_dir: Path,
) -> Path:
    """Menyimpan snapshot data mentah per platform dengan struktur direktori berbasis tanggal dan timestamp."""
    now = datetime.utcnow()
    date_str = now.strftime("%Y-%m-%d")
    timestamp_str = now.strftime("%H%M%S")

    target_dir = raw_dir / platform / date_str
    target_dir.mkdir(parents=True, exist_ok=True)

    output_file = target_dir / f"{platform}_trending_{timestamp_str}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Snapshot mentah [{platform}] ({len(data)} video) disimpan di: {output_file}")
    return output_file


def export_raw_csv_summary(
    all_records: List[Dict[str, Any]],
    raw_dir: Path,
) -> Path:
    """Mengekspor ringkasan sampel data mentah ke CSV untuk kemudahan inspeksi langsung."""
    if not all_records:
        return raw_dir / "sample_raw_videos.csv"

    raw_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_records)

    # Simpan snapshot CSV ber-timestamp
    now_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    timestamped_csv = raw_dir / f"raw_snapshot_{now_str}.csv"
    df.to_csv(timestamped_csv, index=False)

    # Simpan juga ke sample_raw_videos.csv sebagai pointer sampel utama
    sample_csv = raw_dir / "sample_raw_videos.csv"
    df.to_csv(sample_csv, index=False)

    logger.info(f"Ringkasan sampel CSV disimpan di: {sample_csv} & {timestamped_csv}")
    return sample_csv


def run_ingestion_cycle(
    source: str,
    config: Dict[str, Any],
    raw_dir: Path,
) -> List[Dict[str, Any]]:
    """Menjalankan satu siklus penarikan data dinamis untuk sumber yang ditentukan."""
    combined_records: List[Dict[str, Any]] = []

    # 1. Ekstraksi YouTube Shorts
    if source in ("all", "youtube"):
        logger.info("--> Menarik data dinamis YouTube Shorts...")
        try:
            yt_records = extract_youtube_shorts(config=config)
            if yt_records:
                save_platform_snapshot(yt_records, "youtube", raw_dir)
                combined_records.extend(yt_records)
            else:
                logger.warning("Tidak ada data YouTube Shorts yang diperoleh.")
        except Exception as exc:
            logger.error(f"Gagal saat ekstraksi YouTube Shorts: {exc}", exc_info=True)

    # 2. Ekstraksi TikTok FYP / Trending
    if source in ("all", "tiktok"):
        logger.info("--> Menarik data dinamis TikTok FYP...")
        try:
            tt_records = extract_tiktok_videos(config=config)
            if tt_records:
                save_platform_snapshot(tt_records, "tiktok", raw_dir)
                combined_records.extend(tt_records)
            else:
                logger.warning("Tidak ada data TikTok yang diperoleh.")
        except Exception as exc:
            logger.error(f"Gagal saat ekstraksi TikTok FYP: {exc}", exc_info=True)

    return combined_records


def print_summary_table(records: List[Dict[str, Any]]) -> None:
    """Mencetak ringkasan hasil data ingestion ke layar terminal."""
    if not records:
        print("\n[PERINGATAN] Tidak ada record data yang berhasil ditarik.")
        return

    print("\n" + "=" * 80)
    print(f"RINGKASAN DATA INGESTION BERHASIL ({len(records)} TOTAL VIDEO DIAMBIL):")
    print("=" * 80)
    print(f"{'Platform':<10} | {'Video ID':<20} | {'Views':<12} | {'Likes':<10} | {'Judul'}")
    print("-" * 80)

    for item in records[:10]:
        platform = item.get("platform", "unknown")
        vid_id = str(item.get("video_id", ""))[:18]
        views = f"{int(item.get('view_count', 0)):,}"
        likes = f"{int(item.get('like_count', 0)):,}"
        title = str(item.get("title", ""))[:32]
        print(f"{platform:<10} | {vid_id:<20} | {views:<12} | {likes:<10} | {title}...")

    if len(records) > 10:
        print(f"... dan {len(records) - 10} video lainnya.")
    print("=" * 80 + "\n")


def parse_arguments() -> argparse.Namespace:
    """Parsing argumen baris perintah (CLI)."""
    parser = argparse.ArgumentParser(
        description="Skrip Penarikan Data (Data Ingestion) Otomatis ViralSense (YouTube Shorts & TikTok)."
    )
    parser.add_argument(
        "--source",
        choices=["all", "youtube", "tiktok"],
        default="all",
        help="Sumber data dinamis yang akan ditarik (default: all).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Jumlah siklus penarikan untuk simulasi periodik / Continual Learning (default: 1).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Jeda waktu (detik) antar-siklus saat menjalankan simulasi periodik (default: 5 detik).",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Direktori penyimpanan data mentah (default: data/raw).",
    )
    return parser.parse_args()


def main() -> None:
    """Fungsi utama orchestrator data ingestion."""
    args = parse_arguments()
    config = load_config()

    raw_dir = PROJECT_ROOT / args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("============================================================")
    logger.info("   MEMULAI DATA INGESTION OTOMATIS - VIRALSENSE PIPELINE   ")
    logger.info("============================================================")
    logger.info(f"Target Sumber     : {args.source.upper()}")
    logger.info(f"Direktori Raw     : {raw_dir}")
    logger.info(f"Jumlah Siklus     : {args.runs}")
    logger.info(f"Interval Siklus   : {args.interval} detik")

    total_accumulated_records: List[Dict[str, Any]] = []

    for run_idx in range(1, args.runs + 1):
        if args.runs > 1:
            logger.info(f"\n[Siklus {run_idx}/{args.runs}] Menjalankan snapshot periodik...")

        batch_records = run_ingestion_cycle(
            source=args.source,
            config=config,
            raw_dir=raw_dir,
        )
        total_accumulated_records.extend(batch_records)

        # Jika masih ada siklus berikutnya, tunggu sesuai interval
        if run_idx < args.runs:
            logger.info(f"Menunggu jeda interval {args.interval} detik sebelum siklus berikutnya...")
            time.sleep(args.interval)

    # Ekspor konsolidasi ke CSV sampel data mentah
    if total_accumulated_records:
        export_raw_csv_summary(total_accumulated_records, raw_dir)

    print_summary_table(total_accumulated_records)
    logger.info("Proses Data Ingestion selesai secara sukses.")


if __name__ == "__main__":
    main()
