# ViralSense Model Metadata Documentation

Dokumentasi spesifikasi teknis dan metadata dari model terbaik (*Champion Model*) yang terpilih melalui proses *hyperparameter tuning* dan tercatat pada **MLflow Model Registry**.

---

## 1. Ringkasan Model Jawara (Champion Model)

| Atribut | Nilai Spesifikasi |
| :--- | :--- |
| **Model Registry Name** | `ViralSense-ViralityClassifier` |
| **Model Version** | `v1` |
| **Status Deployment** | `Champion` / `Production Candidate` |
| **Algoritma / Family** | Gradient Boosting Classifier (`sklearn.ensemble.GradientBoostingClassifier`) |
| **Source MLflow Run ID** | `63fbe9bd15b8408e8e2b1040d90bd865` |
| **Source MLflow Run Name** | `run4_gbm_tuned_lr0.05_n200` |
| **Tujuan Prediksi** | Klasifikasi Biner Viralitas YouTube Shorts (`is_viral`: 0 = Non-Viral, 1 = Viral) |

---

## 2. Metrik Evaluasi Performa (Test Set Evaluation)

| Metrik Evaluasi | Nilai | Penjelasan / Interpretasi |
| :--- | :--- | :--- |
| **Accuracy** | **95.59%** | Proporsi total prediksi benar dari seluruh sampel video pada set uji. |
| **F1-Score (Primary)** | **0.8000** | Keseimbangan harmonis antara Precision dan Recall pada kelas minoritas (*is_viral = 1*). |
| **ROC-AUC** | **0.9508** | Kemampuan diskriminasi model membedakan video yang berpotensi viral vs non-viral pada berbagai ambang probabilitas. |
| **Precision** | **75.00%** | Dari seluruh video yang diprediksi viral, 75% benar-benar viral (minim false positive). |
| **Recall** | **85.71%** | Dari seluruh video viral aktual, 85.71% berhasil dideteksi oleh model (minim false negative). |

---

## 3. Konfigurasi Hyperparameter Terbaik

Hyperparameter yang dikunci untuk model versi 1:
```json
{
  "n_estimators": 200,
  "learning_rate": 0.05,
  "max_depth": 4,
  "subsample": 0.85,
  "random_state": 42
}
```

---

## 4. Silsilah Data (Data Lineage & Feature Schema)

* **Dataset Versioning**: `data-v3.0` (dilacak via DVC & Git Tagging dari `sample_raw_videos.csv`).
* **Fitur Input (7 Variabel Prediktor)**:
  1. `duration_seconds` (int): Durasi video dalam detik ($\le 60$ detik).
  2. `video_age_hours` (float): Usia video dalam jam sejak waktu publikasi hingga penarikan snapshot.
  3. `engagement_rate` (float): Rasio keterlibatan $\frac{\text{likes} + 2 \times \text{comments}}{\max(\text{views}, 1)}$.
  4. `comment_to_like_ratio` (float): Rasio interaksi komentar terhadap penyuka.
  5. `caption_length` (int): Panjang total karakter judul dan deskripsi.
  6. `word_count` (int): Jumlah kata dalam teks konten.
  7. `has_hook_question` (int/binary): Indikator apakah judul/deskripsi memuat tanda tanya `?` (konten berbasis interaksi hook).
* **Ukuran Sampel**: Train = 271 baris (Positif: 27), Test = 68 baris (Positif: 7).
* **Pipeline Preprocessing**: `StandardScaler()` untuk standarisasi skala seluruh fitur numerik.

---

## 5. Panduan Pemuatan Model untuk Deployment (Inference)

Model dapat dimuat langsung dari MLflow Model Registry menggunakan Python SDK:

```python
import mlflow
import pandas as pd

# Konfigurasi tracking URI ke backend SQLite
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Muat model Champion versi 1 dari Registry
model_uri = "models:/ViralSense-ViralityClassifier/1"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Contoh payload input inferensi
sample_input = pd.DataFrame([{
    "duration_seconds": 15,
    "video_age_hours": 6.5,
    "engagement_rate": 0.085,
    "comment_to_like_ratio": 0.12,
    "caption_length": 120,
    "word_count": 18,
    "has_hook_question": 1
}])

# Prediksi kelas dan probabilitas
prediction = loaded_model.predict(sample_input)
probability = loaded_model.predict_proba(sample_input)[:, 1]

print(f"Prediksi: {'VIRAL' if prediction[0] == 1 else 'NON-VIRAL'}")
print(f"Probabilitas Viralitas: {probability[0]:.2%}")
```
