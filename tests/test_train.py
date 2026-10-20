"""Unit tests untuk modul pelatihan model dan integrasi MLflow (train.py)."""

import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.models.train import (
    create_model,
    evaluate_predictions,
    load_and_validate_dataset,
    load_config,
    split_data,
    train_single_run,
)


class TestTrainPipeline(unittest.TestCase):
    """Test suite untuk memvalidasi alur pelatihan dan metrik model."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("config/config.yaml")

    def test_load_and_validate_dataset(self):
        """Memverifikasi bahwa dataset fitur berhasil dimuat dan berisi target."""
        df, feature_cols, target_col = load_and_validate_dataset(self.config)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn(target_col, df.columns)
        for col in feature_cols:
            self.assertIn(col, df.columns)

    def test_split_data_shapes_and_stratification(self):
        """Memverifikasi rasio pembagian data latih dan uji."""
        df, feature_cols, target_col = load_and_validate_dataset(self.config)
        X_train, X_test, y_train, y_test = split_data(
            df, feature_cols, target_col, test_size=0.2, random_state=42
        )
        total = len(df)
        self.assertEqual(len(X_train) + len(X_test), total)
        self.assertAlmostEqual(len(X_test) / total, 0.2, delta=0.05)
        # Verifikasi stratifikasi: kedua set memiliki kelas positif
        self.assertGreater(y_train.sum(), 0)
        self.assertGreater(y_test.sum(), 0)

    def test_create_model_types(self):
        """Memverifikasi bahwa factory function menghasilkan estimator Scikit-Learn yang tepat."""
        rf = create_model("rf", {"n_estimators": 10, "max_depth": 2})
        self.assertEqual(rf.__class__.__name__, "RandomForestClassifier")

        gbm = create_model("gbm", {"n_estimators": 10, "learning_rate": 0.05})
        self.assertEqual(gbm.__class__.__name__, "GradientBoostingClassifier")

        svm = create_model("svm", {"kernel": "linear", "C": 0.5})
        self.assertEqual(svm.__class__.__name__, "SVC")

        lr = create_model("lr", {"C": 1.0})
        self.assertEqual(lr.__class__.__name__, "LogisticRegression")

    def test_evaluate_predictions_metrics(self):
        """Memverifikasi kalkulasi metrik evaluasi."""
        y_true = pd.Series([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0])
        y_proba = np.array([0.1, 0.8, 0.2, 0.4, 0.1])

        metrics = evaluate_predictions(y_true, y_pred, y_proba)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertEqual(metrics["accuracy"], 0.8)
        self.assertGreater(metrics["roc_auc"], 0.5)

    def test_train_single_run_execution(self):
        """Memverifikasi eksekusi satu run pelatihan dengan MLflow tracking."""
        res = train_single_run(
            model_type="rf",
            hyperparams={"n_estimators": 10, "max_depth": 2, "random_state": 42},
            run_name="unit_test_run",
            config_path="config/config.yaml",
        )
        self.assertIn("run_id", res)
        self.assertIn("metrics", res)
        self.assertIn("f1_score", res["metrics"])
        self.assertIn("accuracy", res["metrics"])
        self.assertGreater(res["metrics"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
