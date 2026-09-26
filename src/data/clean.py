"""Modul pembersihan dan validasi data (Cleaning) untuk pipeline ViralSense."""

import glob
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Memuat konfigurasi pipeline dari file YAML."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_raw_snapshots(raw_dir: str = "data/raw") -> pd.DataFrame:
    """Membaca seluruh snapshot file raw JSON yang tersimpan di data/raw/."""
    json_pattern = str(Path(raw_dir) / "**" / "*.json")
    all_files = glob.glob(json_pattern, recursive=True)

    if not all_files:
        logger.warning(f"Tidak ada file raw data yang ditemukan di {raw_dir}")
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for filepath in all_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records.extend(data)
        except Exception as exc:
            logger.error(f"Gagal membaca file {filepath}: {exc}")

    logger.info(f"Total {len(records)} baris data mentah dibaca dari {len(all_files)} file snapshot.")
    return pd.DataFrame(records)


def clean_raw_data(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Melakukan pembersihan, deduplikasi, dan validasi format data video pendek."""
    if df.empty:
        logger.warning("DataFrame kosong, melewati proses pembersihan.")
        return df

    cleaning_cfg = cfg.get("cleaning", {})
    dedup_keys = cleaning_cfg.get("deduplicate_keys", ["platform", "video_id"])
    max_duration = cleaning_cfg.get("max_video_duration_seconds", 60)

    initial_count = len(df)

    # 1. Deduplikasi record
    df = df.drop_duplicates(subset=dedup_keys, keep="last")
    logger.info(f"Deduplikasi selesai: {initial_count} -> {len(df)} record.")

    # 2. Filter durasi format short-form (maksimal 60 detik)
    if "duration_seconds" in df.columns:
        df = df[df["duration_seconds"] <= max_duration]

    # 3. Penanganan Missing Values pada kolom numerik
    numeric_cols = ["view_count", "like_count", "comment_count", "duration_seconds"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 4. Penanganan kolom teks
    text_cols = ["title", "description"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    # 5. Standarisasi tanggal
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)

    if "ingested_at" in df.columns:
        df["ingested_at"] = pd.to_datetime(df["ingested_at"], errors="coerce", utc=True)

    logger.info(f"Pembersihan data selesai. Data bersih siap pakai: {len(df)} baris.")
    return df


def save_clean_data(df: pd.DataFrame, output_path: str = "data/processed/clean_videos.parquet") -> Path:
    """Menyimpan data bersih dalam format Parquet terkompresi."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file, index=False)
    logger.info(f"Data bersih berhasil disimpan ke: {out_file}")
    return out_file


if __name__ == "__main__":
    config = load_config()
    raw_directory = config.get("storage", {}).get("raw_dir", "data/raw")
    clean_target_path = config.get("storage", {}).get("clean_data_path", "data/processed/clean_videos.parquet")

    raw_df = load_raw_snapshots(raw_dir=raw_directory)
    if not raw_df.empty:
        # Gunakan modul preprocess untuk pipeline pembersihan dan tokenisasi NLP lengkap
        try:
            from src.preprocess import preprocess_data
            cleaned_df = preprocess_data(raw_df, config)
        except Exception:
            cleaned_df = clean_raw_data(raw_df, config)
        save_clean_data(cleaned_df, clean_target_path)
    else:
        logger.info("Jalankan modul ingestion terlebih dahulu untuk membuat data mentah.")

