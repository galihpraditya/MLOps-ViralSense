"""Modul rekayasa fitur (Feature Engineering) untuk memprediksi viralitas video."""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Memuat konfigurasi pipeline dari file YAML."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def extract_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Mengekstrak fitur-fitur analitik, rasio keterlibatan, dan sinyal viralitas."""
    if df.empty:
        logger.warning("DataFrame kosong, melewati feature engineering.")
        return df

    feat_df = df.copy()
    features_cfg = cfg.get("features", {})
    percentile_threshold = features_cfg.get("virality_percentile_threshold", 0.90)

    # 1. Menghitung usia video dalam jam (Age in Hours)
    if "published_at" in feat_df.columns and "ingested_at" in feat_df.columns:
        diff_seconds = (feat_df["ingested_at"] - feat_df["published_at"]).dt.total_seconds()
        # Hindari pembagian dengan nol menggunakan clipping minimal 0.1 jam (6 menit)
        feat_df["video_age_hours"] = np.maximum(diff_seconds / 3600.0, 0.1)
    else:
        feat_df["video_age_hours"] = 1.0

    # 2. Fitur Kecepatan Metrik (Velocity Metrics)
    feat_df["views_velocity"] = feat_df["view_count"] / feat_df["video_age_hours"]
    feat_df["likes_velocity"] = feat_df["like_count"] / feat_df["video_age_hours"]

    # 3. Rasio Keterlibatan (Engagement Rates)
    views_safe = np.maximum(feat_df["view_count"], 1.0)
    likes_safe = np.maximum(feat_df["like_count"], 1.0)

    # Bobot engagement: Komentar bernilai 2x dari Likes
    feat_df["engagement_rate"] = (feat_df["like_count"] + 2.0 * feat_df["comment_count"]) / views_safe
    feat_df["comment_to_like_ratio"] = feat_df["comment_count"] / likes_safe

    # 4. Sinyal Teks & Hook Sederhana
    full_text = feat_df["title"].fillna("") + " " + feat_df["description"].fillna("")
    feat_df["caption_length"] = full_text.str.len()
    feat_df["word_count"] = full_text.str.split().str.len()

    # Deteksi hook pertanyaan pada kalimat/judul awal
    feat_df["has_hook_question"] = full_text.str.contains(r"\?", regex=True).astype(int)

    # 5. Pembentukan Label Target (Ground Truth: is_viral)
    if len(feat_df) > 1:
        threshold_val = feat_df["views_velocity"].quantile(percentile_threshold)
        feat_df["is_viral"] = (feat_df["views_velocity"] >= threshold_val).astype(int)
    else:
        feat_df["is_viral"] = 0

    logger.info(f"Ekstraksi fitur selesai. Bentuk matriks: {feat_df.shape}")
    return feat_df


def save_features_matrix(df: pd.DataFrame, output_path: str = "data/processed/features_matrix.parquet") -> Path:
    """Menyimpan matriks fitur siap latih ke file Parquet."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file, index=False)
    logger.info(f"Matriks fitur berhasil disimpan ke: {out_file}")
    return out_file


if __name__ == "__main__":
    config = load_config()
    clean_path = config.get("storage", {}).get("clean_data_path", "data/processed/clean_videos.parquet")
    features_target_path = config.get(
        "storage", {}).get("features_data_path", "data/processed/features_matrix.parquet"
    )

    clean_file = Path(clean_path)
    if clean_file.exists():
        clean_df = pd.read_parquet(clean_file)
        feature_matrix = extract_features(clean_df, config)
        save_features_matrix(feature_matrix, features_target_path)
    else:
        logger.warning(f"File data bersih {clean_path} belum ditemukan. Jalankan clean.py terlebih dahulu.")

