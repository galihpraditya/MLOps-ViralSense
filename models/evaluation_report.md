# 🧪 ViralSense Model Evaluation & Quality Gate Report

**Status Validasi:** 🟢 **PASSED (Promoted to Staging)**

- **Model Registry:** `ViralSense-ViralityClassifier` (Version: `v3`)
- **Stage:** `Staging`
- **Candidate Run ID:** `f2243426abe645098b98fb53a2207de0`
- **Git Commit:** `eafe3e47` (`feat/mlops-code-as-trigger`)
- **DVC Dataset Hash:** `4ad531da424ea3e6a966acb6c4a5b648`

### Kriteria Ambang Batas LK-01

| Metrik | Nilai Aktual | Ambang Batas (LK-01) | Status |
| :--- | :--- | :--- | :--- |
| Accuracy | 95.59% | >= 82.00% | ✅ PASSED |
| Macro F1-Score | 0.8000 | >= 0.8000 | ✅ PASSED |
| ROC-AUC Score | 0.9625 | >= 0.8500 | ✅ PASSED |
| Inference Latency (p95) | 1.93 ms | < 200.0 ms | ✅ PASSED |

> [!NOTE]
> Model kandidat memenuhi seluruh standar kualitas dan latensi produksi. Model secara otomatis didaftarkan dan dipromosikan ke tahap **Staging**.
