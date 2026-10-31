"""Unit tests untuk pengujian kesiapan inferensi model Production MLflow - ViralSense LK-07."""

import os
import unittest
import numpy as np
import pandas as pd

# Suppress noisy agent hints
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import mlflow
import mlflow.pyfunc
from src.models.verify_inference import get_inference_samples, load_production_model


class TestProductionInference(unittest.TestCase):
    """Test suite untuk memvalidasi model Production di Model Registry."""

    @classmethod
    def setUpClass(cls):
        """Memuat model Production sekali untuk seluruh test suite."""
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        cls.model_name = "ViralSense-ViralityClassifier"
        cls.model, cls.resolved_uri = load_production_model(
            model_name=cls.model_name, stage="Production"
        )
        cls.df_samples, cls.scenarios = get_inference_samples()

    def test_production_model_loaded(self):
        """Memastikan model berhasil dimuat dari stage Production."""
        self.assertIsNotNone(self.model)
        self.assertIn("Production", self.resolved_uri)

    def test_single_inference_prediction(self):
        """Memastikan inferensi sampel tunggal mengembalikan output biner valid."""
        single_input = self.df_samples.iloc[[0]]
        pred = self.model.predict(single_input)
        self.assertEqual(len(pred), 1)
        self.assertIn(int(pred[0]), [0, 1])

    def test_batch_inference_predictions(self):
        """Memastikan batch inferensi mengembalikan dimensi dan tipe data yang benar."""
        preds = self.model.predict(self.df_samples)
        self.assertEqual(len(preds), len(self.df_samples))
        for p in preds:
            self.assertIn(int(p), [0, 1])

    def test_inference_latency_budget(self):
        """Memastikan latensi inferensi per sampel di bawah ambang SLA (50 ms)."""
        import time
        start = time.perf_counter()
        _ = self.model.predict(self.df_samples.iloc[[0]])
        latency_ms = (time.perf_counter() - start) * 1000
        self.assertLess(latency_ms, 50.0, f"Latensi {latency_ms:.2f}ms melebihi batas 50ms")

    def test_viral_vs_non_viral_sensitivity(self):
        """Memverifikasi sensitivitas model: video engagement tinggi harus diprediksi viral (1)."""
        preds = self.model.predict(self.df_samples)
        # TC-01 adalah kasus konten viral dengan sinyal engagement kuat
        self.assertEqual(int(preds[0]), 1, "Kasus TC-01 harus terdeteksi sebagai Viral (1)")
        # TC-02 adalah kasus konten non-viral (flat)
        self.assertEqual(int(preds[1]), 0, "Kasus TC-02 harus terdeteksi sebagai Non-Viral (0)")


if __name__ == "__main__":
    unittest.main()

