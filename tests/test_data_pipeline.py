"""Unit Tests untuk Pengujian Pipeline Data Ingestion dan Preprocessing ViralSense."""

import glob
import unittest
from pathlib import Path
import pandas as pd

from src.preprocess import (
    clean_missing_values,
    clean_text_for_nlp,
    tokenize,
    remove_stopwords,
    preprocess_data,
    load_config,
)


class TestViralSensePipeline(unittest.TestCase):
    """Test suite untuk validasi fungsionalitas ingestion dan preprocessing."""

    def test_raw_data_snapshots_exist(self):
        """Memastikan minimal ada file snapshot JSON mentah di data/raw/."""
        json_snapshots = glob.glob("data/raw/**/*.json", recursive=True)
        csv_samples = glob.glob("data/raw/*.csv")
        self.assertGreater(len(json_snapshots), 0, "Harus ada minimal 1 file snapshot JSON di data/raw/")
        self.assertGreater(len(csv_samples), 0, "Harus ada file sampel CSV di data/raw/")

    def test_missing_value_imputation(self):
        """Memastikan penanganan nilai kosong numerik diisi 0 dan teks diisi string kosong."""
        sample_df = pd.DataFrame([
            {
                "platform": "youtube",
                "video_id": "test_01",
                "title": None,
                "view_count": None,
                "like_count": "invalid",
                "duration_seconds": None,
            }
        ])
        cleaned = clean_missing_values(sample_df)
        self.assertEqual(cleaned.loc[0, "view_count"], 0)
        self.assertEqual(cleaned.loc[0, "like_count"], 0)
        self.assertEqual(cleaned.loc[0, "duration_seconds"], 0)
        self.assertEqual(cleaned.loc[0, "title"], "")

    def test_tokenization_and_stopwords(self):
        """Memastikan tokenisasi memecah kata dan menghapus stopwords bahasa Indonesia & Inggris."""
        text = "Ini adalah konten video yang sangat viral dan amazing"
        cleaned_text = clean_text_for_nlp(text)
        tokens = tokenize(cleaned_text)
        clean_tokens = remove_stopwords(tokens)

        # Stopwords 'ini', 'adalah', 'yang', 'sangat', 'dan' harus dihapus
        self.assertIn("konten", clean_tokens)
        self.assertIn("video", clean_tokens)
        self.assertIn("viral", clean_tokens)
        self.assertIn("amazing", clean_tokens)
        self.assertNotIn("ini", clean_tokens)
        self.assertNotIn("adalah", clean_tokens)
        self.assertNotIn("yang", clean_tokens)
        self.assertNotIn("dan", clean_tokens)

    def test_duration_filtering(self):
        """Memastikan video berdurasi lebih dari 60 detik disaring keluar."""
        cfg = {"cleaning": {"max_video_duration_seconds": 60}}
        df = pd.DataFrame([
            {"platform": "tiktok", "video_id": "v1", "duration_seconds": 30, "title": "short"},
            {"platform": "tiktok", "video_id": "v2", "duration_seconds": 120, "title": "long"},
        ])
        processed = preprocess_data(df, cfg)
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed.iloc[0]["video_id"], "v1")

    def test_processed_dataset_integrity(self):
        """Memastikan dataset hasil akhir memiliki skema yang valid dan dapat dibaca."""
        pq_path = Path("data/processed/clean_videos.parquet")
        csv_path = Path("data/processed/clean_videos.csv")
        self.assertTrue(pq_path.exists(), "clean_videos.parquet harus ada")
        self.assertTrue(csv_path.exists(), "clean_videos.csv harus ada")

        df = pd.read_parquet(pq_path)
        self.assertGreater(len(df), 0, "Dataset bersih tidak boleh kosong")
        required_cols = ["platform", "video_id", "view_count", "like_count", "tokens", "clean_tokens"]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Kolom {col} wajib ada pada dataset bersih")


if __name__ == "__main__":
    unittest.main()
