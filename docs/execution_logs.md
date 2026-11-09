# 📋 Log Eksekusi & Bukti Pemicu Otomatis (Code as a Trigger) - ViralSense

Dokumen ini menyediakan tautan resmi (*Action URLs*) dan riwayat log eksekusi lengkap dari alur kerja otomatisasi **GitHub Actions** untuk tugas MLOps LK-08: **"Code as a Trigger"**.

---

## 1. Tautan Resmi GitHub Actions (Action URLs)

Berikut adalah URL publik aktif yang membuktikan keberhasilan eksekusi end-to-end tanpa intervensi manual:

| Komponen / Event | Status | Durasi | Tautan Langsung (GitHub Actions) |
| :--- | :---: | :---: | :--- |
| **🚀 Run Pemicu Simulasi Perubahan (Push)** | **`PASSED`** | 2m 26s | **[Workflow Run #42386114410](https://github.com/galihpraditya/MLOps-ViralSense/actions/runs/42386114410)** |
| ├── **Job 1: 🧪 Automated Unit Testing (pytest)** | `PASSED` | 1m 16s | [Job #128443736401 Log](https://github.com/galihpraditya/MLOps-ViralSense/actions/runs/42386114410/job/128443736401) |
| └── **Job 2: 🚀 Retraining, Evaluation & Registry** | `PASSED` | 1m 10s | [Job #128444056211 Log](https://github.com/galihpraditya/MLOps-ViralSense/actions/runs/42386114410/job/128444056211) |
| **🔀 Pull Request #8 (feat/mlops-code-as-trigger)** | **`ALL PASS`** | - | **[Pull Request #8 on GitHub](https://github.com/galihpraditya/MLOps-ViralSense/pull/8)** |
| **📦 Build & Evaluation Artifacts** | `READY` | - | [viralsense-evaluation-artifacts](https://github.com/galihpraditya/MLOps-ViralSense/actions/runs/42386114410/artifacts/3655110825) |

---

## 2. Rincian Log Eksekusi per Tahapan

### Tahap 1: Automated Unit Testing (Pytest Gate)
* **Job Name**: `🧪 Tahap 1 - Automated Unit Testing`
* **Runner OS**: `ubuntu-latest` (Python 3.10.22)
* **Command**: `pytest -v tests/ --durations=5`

```text
============================= test session starts ==============================
platform linux -- Python 3.10.22, pytest-9.2.0, pluggy-1.6.0
rootdir: /home/runner/work/MLOps-ViralSense/MLOps-ViralSense
plugins: hydra-core-1.3.6, anyio-4.15.1
collected 15 items

tests/test_data_pipeline.py::TestViralSensePipeline::test_duration_filtering PASSED [  6%]
tests/test_data_pipeline.py::TestViralSensePipeline::test_missing_value_imputation PASSED [ 13%]
tests/test_data_pipeline.py::TestViralSensePipeline::test_processed_dataset_integrity PASSED [ 20%]
tests/test_data_pipeline.py::TestViralSensePipeline::test_raw_data_snapshots_exist PASSED [ 26%]
tests/test_data_pipeline.py::TestViralSensePipeline::test_tokenization_and_stopwords PASSED [ 33%]
tests/test_inference.py::TestProductionInference::test_batch_inference_predictions SKIPPED [ 40%]
tests/test_inference.py::TestProductionInference::test_inference_latency_budget SKIPPED [ 46%]
tests/test_inference.py::TestProductionInference::test_production_model_loaded SKIPPED [ 53%]
tests/test_inference.py::TestProductionInference::test_single_inference_prediction SKIPPED [ 60%]
tests/test_inference.py::TestProductionInference::test_viral_vs_non_viral_sensitivity SKIPPED [ 66%]
tests/test_train.py::TestTrainPipeline::test_create_model_types PASSED   [ 73%]
tests/test_train.py::TestTrainPipeline::test_evaluate_predictions_metrics PASSED [ 80%]
tests/test_train.py::TestTrainPipeline::test_load_and_validate_dataset PASSED [ 86%]
tests/test_train.py::TestTrainPipeline::test_split_data_shapes_and_stratification PASSED [ 93%]
tests/test_train.py::TestTrainPipeline::test_train_single_run_execution PASSED [100%]

=================== 10 passed, 5 skipped, 1 warning in 6.09s ===================
```

---

### Tahap 2: DVC Data Sync & Automated Training
* **Job Name**: `🚀 Tahap 2-4 - Retraining, Evaluation & Registry`
* **Dataset Hash (DVC MD5)**: `4ad531da424ea3e6a966acb6c4a5b648` (Sample Raw Videos)
* **Command**: `python train.py`
* **Hyperparameters (dari `config.yaml`)**:
  - `model_type`: `gbm`
  - `n_estimators`: `230`
  - `learning_rate`: `0.045`
  - `max_depth`: `4`
  - `subsample`: `0.85`

```text
2026-11-07 15:25:09 - [INFO] - Dataset berhasil dimuat. Total sampel: 339, Fitur: 7, Distribusi target: {0: 305, 1: 34}
2026-11-07 15:25:09 - [INFO] - Data terbagi: Train=271 sampel (Positif: 27), Test=68 sampel (Positif: 7)
2026-11-07 15:25:09 - [INFO] - === Memulai MLflow Run: 'gbm_230' (Model: gbm) ===
2026-11-07 15:25:20 - [INFO] - Run 'gbm_230' berhasil! Run ID: 8a1699a77ec341b3ad13da51df4aa326 | Accuracy: 0.9559 | F1-Score: 0.8000 | ROC-AUC: 0.9508
2026-11-07 15:25:20 - [INFO] - Metadata candidate run tersimpan ke: /home/runner/work/MLOps-ViralSense/MLOps-ViralSense/models/candidate_run.json
```

---

### Tahap 3: Automated Model Evaluation against LK-01 Benchmarks
* **Command**: `python src/models/evaluate.py`
* **Evaluator**: `src.models.evaluate.evaluate_candidate_model`

```text
=========================================================================================================
                     VIRALSENSE MLOPS - AUTOMATED MODEL EVALUATION & VALIDATION GATE                     
                               Kriteria Keberhasilan & Ambang Batas LK-01                                
=========================================================================================================
Candidate Run ID : 8a1699a77ec341b3ad13da51df4aa326
Run Name         : gbm_230
Model Type       : gbm
Dataset Version  : data-v3.0
---------------------------------------------------------------------------------------------------------
Metrik                     | Nilai Aktual   | LK-01 Threshold    | Status     | Aturan                        
---------------------------------------------------------------------------------------------------------
Accuracy                   | 95.59%         | >= 82.00%          | ✅ PASS     | LK-01 Tabel 4.1: Target Akurasi >= 82%
Macro F1-Score             | 0.8000         | >= 0.8000          | ✅ PASS     | LK-01 Tabel 4.1: Target F1-Score >= 0.80
ROC-AUC Score              | 0.9508         | >= 0.8500          | ✅ PASS     | Stabilitas separasi probabilitas kelas viral
Inference Latency (p95)    | 0.97 ms        | < 200.0 ms         | ✅ PASS     | LK-01 Tabel 4.1: Latensi Inferensi < 200 ms (p95)
=========================================================================================================
VERDIK VALIDASI AKHIR: PASSED (Lolos kualifikasi Staging)
=========================================================================================================
```

---

### Tahap 4: Auto-Registry Staging & Live Inference Verification
* **Target Registry**: `ViralSense-ViralityClassifier`
* **Target Stage**: `Staging`
* **Command**: `python src/models/verify_inference.py --stage Staging`

```text
2026-11-07 15:25:22 - [INFO] - Model lolos seluruh pengujian! Mendaftarkan ke MLflow Model Registry 'ViralSense-ViralityClassifier'...
2026-11-07 15:25:22 - [INFO] - Mentransisikan 'ViralSense-ViralityClassifier' v4 ke stage 'Staging'...
2026-11-07 15:25:22 - [INFO] - 🎉 SUKSES! Model 'ViralSense-ViralityClassifier' versi 4 kini resmi aktif di stage 'Staging'!

===================================================================================================================
                                    HASIL PENGUJIAN INFERENSI MODEL PRODUCTION                                     
                Model URI: models:/ViralSense-ViralityClassifier/Staging | Waktu Pemuatan: 1874.81 ms               
===================================================================================================================
Case ID  | Scenario                                   | Exp   | Pred  | Label        | Probability  | Status  
-------------------------------------------------------------------------------------------------------------------
TC-01    | High Velocity Viral Short (Strong Hook)    | 1     | 1     | VIRAL (1)    |  91.19%      | PASSED  
TC-02    | Low Engagement Organic Short (Mature Age)  | 0     | 0     | NON-VIRAL(0) |   0.49%      | PASSED  
TC-03    | Trending Mystery/Humor Short (Fast Hook)   | 1     | 1     | VIRAL (1)    |  61.65%      | PASSED  
TC-04    | Promotional Catalog / Product Showcase     | 0     | 0     | NON-VIRAL(0) |  15.92%      | PASSED  
===================================================================================================================
Single-record Latency : 2.24 ms (< 50 ms budget)
Batch Processing Time : 1.94 ms (4 records)
Throughput Simulasi   : 2066.9 requests/detik
===================================================================================================================
2026-11-07 15:25:24 - [INFO] - VERIFIKASI INFERENSI BERHASIL: Model Production siap melayani inferensi real-time!
```

---

## 3. Panduan Pengambilan Rekaman Layar (Screen Recording Guide)

Jika pengumpulan tugas mewajibkan rekaman layar (*screen recording* / video berdurasi singkat 1–2 menit) ke LMS/Google Classroom, ikuti langkah berikut:

1. Buka browser dan kunjungi: **[Workflow Run #42386114410](https://github.com/galihpraditya/MLOps-ViralSense/actions/runs/42386114410)**.
2. Sorot bagian grafik alur kerja (**Workflow Graph**):
   - Tunjukkan job `🧪 Tahap 1 - Automated Unit Testing` bertanda centang hijau (`1m 16s`).
   - Tunjukkan panah dependensi (`needs`) yang mengalir ke job `🚀 Tahap 2-4 - Retraining, Evaluation & Registry` (`1m 10s`).
3. Klik job **Tahap 2-4 - Retraining, Evaluation & Registry**:
   - Buka dropdown langkah `⚖️ Tahap 3 & 4 - Model Evaluation & MLflow Registry (Staging)` untuk memperlihatkan tabel ASCII validasi ambang batas LK-01 (`Accuracy 95.59%`, `F1 0.8000`, `Latency 0.97 ms`).
   - Buka dropdown `🔍 Tahap 4 (Lanjutan) - Verifikasi Inferensi Model Staging` untuk memperlihatkan inferensi pada model `Staging`.
4. Buka tab **Pull Requests** dan klik **[Pull Request #8](https://github.com/galihpraditya/MLOps-ViralSense/pull/8)**:
   - Perlihatkan bahwa semua *status checks* (4/4 successful) hijau.
