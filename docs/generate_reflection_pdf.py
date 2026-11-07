#!/usr/bin/env python3
"""Script untuk menghasilkan dokumen Refleksi Singkat MLOps format PDF.

Topik: Urgensi 'Code as a Trigger' dalam Mencegah Kegagalan Sistem di Produksi (Silent Failures)
Penulis: Galih Praditya Kurniawan (NIM: 245150201111017)
Mata Kuliah: Machine Learning Operations (MLOps)
Fakultas Ilmu Komputer, Universitas Brawijaya
"""

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class NumberedCanvas(canvas.Canvas):
    """Canvas kustom untuk menambahkan footer nomor halaman dan header formal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4B5563"))

        # Header (mulai halaman 2)
        if self._pageNumber > 1:
            self.drawString(
                2.0 * cm,
                28.3 * cm,
                "Refleksi MLOps: Urgensi 'Code as a Trigger' Mencegah Silent Failures - ViralSense",
            )
            self.setStrokeColor(colors.HexColor("#D1D5DB"))
            self.setLineWidth(0.5)
            self.line(2.0 * cm, 28.1 * cm, 19.0 * cm, 28.1 * cm)

        # Footer (seluruh halaman)
        self.setStrokeColor(colors.HexColor("#E5E7EB"))
        self.setLineWidth(0.5)
        self.line(2.0 * cm, 1.8 * cm, 19.0 * cm, 1.8 * cm)

        footer_text = f"Halaman {self._pageNumber} dari {page_count}"
        self.drawRightString(19.0 * cm, 1.3 * cm, footer_text)
        self.drawString(
            2.0 * cm,
            1.3 * cm,
            "Galih Praditya Kurniawan (245150201111017) | MLOps FILKOM Universitas Brawijaya",
        )
        self.restoreState()


def build_pdf(filename: str):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.2 * cm,
    )

    styles = getSampleStyleSheet()

    # Kustomisasi Styles
    primary_color = colors.HexColor("#1E3A8A")  # Navy Blue
    secondary_color = colors.HexColor("#1F2937")  # Dark Slate
    accent_color = colors.HexColor("#2563EB")  # Royal Blue

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=20,
        textColor=primary_color,
        alignment=1,  # Center
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4B5563"),
        alignment=1,
        spaceAfter=14,
    )

    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=secondary_color,
    )

    heading1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True,
    )

    heading2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=accent_color,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13.5,
        textColor=secondary_color,
        alignment=4,  # Justify
        spaceAfter=6,
    )

    callout_style = ParagraphStyle(
        "Callout_Text",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#1E40AF"),
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.white,
        alignment=1,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=secondary_color,
    )

    story = []

    # 1. Judul dan Identitas Penulis
    story.append(Paragraph("REFLEKSI SINGKAT MLOPS:<br/>URGENSI PRINSIP 'CODE AS A TRIGGER' DALAM MENCEGAH KEGAGALAN SISTEM DI PRODUKSI (SILENT FAILURES)", title_style))
    story.append(Paragraph("Studi Kasus: Sistem Prediksi Skor Viralitas dan Deteksi Kejenuhan Konten Media Sosial (ViralSense)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=8, spaceBefore=0))

    # Info Penulis Box
    author_table_data = [
        [
            Paragraph("<b>Penyusun:</b> Galih Praditya Kurniawan", meta_style),
            Paragraph("<b>Dosen Pengampu:</b> Rizal Setya Perdana, S.Kom., M.Kom., Ph.D.", meta_style),
        ],
        [
            Paragraph("<b>NIM:</b> 245150201111017", meta_style),
            Paragraph("<b>Mata Kuliah:</b> Machine Learning Operations (MLOps)", meta_style),
        ],
        [
            Paragraph("<b>Program Studi:</b> Magister Ilmu Komputer", meta_style),
            Paragraph("<b>Institusi:</b> Fakultas Ilmu Komputer, Universitas Brawijaya", meta_style),
        ],
    ]
    author_table = Table(author_table_data, colWidths=[8.0 * cm, 9.0 * cm])
    author_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F4F6")),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, -1), (-1, -1), 1, colors.HexColor("#D1D5DB")),
    ]))
    story.append(author_table)
    story.append(Spacer(1, 10))

    # 2. Ringkasan Eksekutif (Callout Box)
    callout_data = [[
        Paragraph(
            "<b>Intisari Refleksi:</b> Berbeda dari perangkat lunak tradisional di mana cacat logika menimbulkan pesan kesalahan fatal (crash/HTTP 500), kegagalan pada sistem Machine Learning sering kali bersifat senyap (<b>silent failures</b>)—layanan tetap mengembalikan status HTTP 200 OK namun menghasilkan prediksi yang rusak, terdegradasi, atau bahkan acak. Alur kerja otomatisasi <b>'Code as a Trigger'</b> menjadi benteng pertahanan krusial untuk mencegah silent failures melalui verifikasi kode otomatis (pytest), pelatihan ulang deterministik dengan silsilah data terkelola (DVC), evaluasi ketat berbasis ambang batas teknis (LK-01), dan promosi berjenjang pada Model Registry (Staging).",
            callout_style,
        )
    ]]
    callout_table = Table(callout_data, colWidths=[17.0 * cm])
    callout_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#93C5FD")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(callout_table)
    story.append(Spacer(1, 10))

    # 3. Bagian 1: Fenomena Silent Failures pada Sistem Machine Learning
    story.append(Paragraph("1. Fenomena Silent Failures dalam Machine Learning Produksi", heading1_style))
    story.append(Paragraph(
        "Dalam paradigma rekayasa perangkat lunak konvensional (Software Engineering 1.0), integritas sistem didominasi oleh kebenaran sintaksis dan alur eksekusi logika kode program. Jika terjadi regresi seperti dereferensi pointer null, kesalahan tipe variabel, atau pengecualian basis data, sistem akan secara instan melempar <i>loud error</i> (seperti kode status HTTP 500, crash process, atau log exception traceback) yang langsung disadari oleh operator melalui sistem monitoring kesehatan server.",
        body_style,
    ))
    story.append(Paragraph(
        "Sebaliknya, pada sistem Machine Learning (Software 2.0), model inferensi pada dasarnya hanyalah serangkaian perkalian matriks bobot matematis. Ketika kode pra-pemrosesan diubah secara ceroboh, normalisasi fitur bergeser, atau hyperparameter yang tidak tepat di-commit ke repositori, model yang dilatih ulang tetap dapat dimuat oleh serving container (FastAPI/Triton) dan tetap merespons permintaan klien dengan respons HTTP 200 OK. Namun secara substansial, output prediksi skor viralitas yang dihasilkan menjadi tidak akurat, bias, atau bahkan terdegradasi menjadi tebakan acak. Inilah yang diistilahkan sebagai <b>Silent Failures</b>.",
        body_style,
    ))
    story.append(Paragraph(
        "Pada kasus proyek <b>ViralSense</b>—platform analitik waktu-nyata yang memprediksi momentum viralitas video pendek YouTube Shorts—konsekuensi dari silent failure sangat fatal bagi agensi pemasaran digital dan kreator. Kreator konten yang menginvestasikan jutaan rupiah untuk memproduksi video berdasarkan rekomendasi model yang cacat akan mengalami kegagalan distribusi audiens, rasio kegagalan konten membengkak di atas 25%, dan keputusan operasional menjadi tersesat tanpa disadari oleh tim teknis selama berminggu-minggu.",
        body_style,
    ))

    # 4. Bagian 2: Anatomi dan Akar Penyebab Silent Failures
    story.append(Paragraph("2. Anatomi Akar Penyebab Silent Failures pada Pipeline ML", heading1_style))
    story.append(Paragraph(
        "Berdasarkan analisis arsitektur end-to-end ViralSense, terdapat empat simpul kerentanan utama yang dapat memicu terjadinya kegagalan senyap apabila alur kerja tidak diotomatisasi secara ketat:",
        body_style,
    ))
    story.append(Paragraph(
        "<b>a. Perubahan Logika Rekayasa Fitur & Kebocoran Data (Feature Engineering Drift):</b> Penyesuaian skrip imputasi nilai kosong (<i>clean_missing_values</i>) atau formula ekstraksi rasio (misalnya rasio komentar terhadap like atau laju kecepatan penayangan) dapat secara tidak sengaja memodifikasi skala input inference. Jika kode transformasi training tidak selaras dengan inference serving, model akan memproses fitur dengan makna numerik yang cacat (<i>training-serving skew</i>).",
        body_style,
    ))
    story.append(Paragraph(
        "<b>b. Degradasi Tersembunyi Akibat Hyperparameter Suboptimal:</b> Modifikasi kecil pada parameter regularisasi (seperti penurunan kedalaman pohon <code>max_depth</code> yang terlalu agresif atau perubahan <code>learning_rate</code>) dapat menyebabkan model mengalami underfitting atau overfitting parah. Tanpa evaluasi ambang batas otomatis, model yang performanya anjlok ini dapat secara tidak sengaja terdorong ke produksi menggantikan model Champion yang sudah terbukti stabil.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>c. Pergeseran Data dan Latensi Label (Data & Concept Drift):</b> Dinamika tren video pendek berubah dalam siklus mingguan. Apabila model dilatih menggunakan snapshot data usang yang tidak terhubung dengan silsilah versi DVC yang sahih, model baru akan kehilangan kepekaan terhadap pola audio latar dan slang terbaru di YouTube Shorts.",
        body_style,
    ))

    # 5. Bagian 3: Solusi "Code as a Trigger"
    story.append(Paragraph("3. Mengapa Prinsip 'Code as a Trigger' Merupakan Solusi Kritis", heading1_style))
    story.append(Paragraph(
        "Prinsip <b>'Code as a Trigger'</b> mendikte bahwa setiap modifikasi pada basis kode—sekecil apa pun perubahannya—harus diposisikan sebagai hipotesis yang belum terbukti dan wajib diverifikasi oleh orkestrator CI/CD secara otomatis tanpa toleransi intervensi manual. Pada implementasi ViralSense melalui GitHub Actions (<code>.github/workflows/mlops-automation.yaml</code>), sistem membangun empat lapis pertahanan (four-tier defenses):",
        body_style,
    ))

    # Sub-tahap
    story.append(Paragraph("<b>Tahap 1: Verifikasi Integritas Kode Otomatis (Pytest Fail-Fast Gate)</b>", heading2_style))
    story.append(Paragraph(
        "Sebelum sumber daya komputasi dialokasikan untuk melatih model, job <code>automated-testing</code> mengeksekusi suite uji komprehensif pada modul pra-pemrosesan data, validasi skema, factory model, dan fungsi metrik. Jika ada perubahan sintaksis atau regresi fungsional yang merusak kontrak masukan/keluaran, pipeline langsung dihentikan seketika (<i>fail-fast</i>), mencegah terbuangnya waktu dan kuota retraining.",
        body_style,
    ))

    story.append(Paragraph("<b>Tahap 2: Pelatihan Ulang Otomatis dengan Reproduksibilitas Data DVC</b>", heading2_style))
    story.append(Paragraph(
        "Melalui orkestrasi <code>train.py</code>, retraining dijalankan menggunakan dataset fitur yang terlacak silsilahnya secara deterministik via DVC (Data Version Control). Setiap baris data latih dipetakan ke hash MD5 data raw yang tersinkronisasi ke Cloudflare R2 Object Storage. Hal ini mengeliminasi anomali 'model bekerja di komputer lokal tetapi gagal di produksi' karena variasi versi dataset lokal.",
        body_style,
    ))

    story.append(Paragraph("<b>Tahap 3: Gerbang Evaluasi Kuantitatif Berbasis Ambang Batas LK-01</b>", heading2_style))
    story.append(Paragraph(
        "Tahap inilah yang menjadi penangkal utama silent failures. Modul <code>src/models/evaluate.py</code> secara ketat menguji kandidat model baru terhadap ambang batas teknis yang telah dibakukan dalam dokumen <b>LK-01 Tabel 4.1</b>:",
        body_style,
    ))
    story.append(Paragraph(
        "• <b>Target Akurasi $\ge 82\%$ (0.82)</b>: Memastikan ketepatan klasifikasi umum terhadap populasi video.<br/>"
        "• <b>Macro F1-Score $\ge 0.80$</b>: Menjamin model seimbang dan tidak bias ke kelas mayoritas non-viral.<br/>"
        "• <b>Latensi Inferensi Single-Sample $< 200\text{ ms (p95)}$</b>: Memenuhi Service Level Agreement (SLA) layanan inferensi real-time.<br/>"
        "• <b>ROC-AUC $\ge 0.85$</b>: Memastikan pemisahan skor probabilitas antar kelas viral vs non-viral sangat tajam.",
        body_style,
    ))
    story.append(Paragraph(
        "Jika satu saja ambang batas ini dilanggar (misalnya F1-Score turun ke 0.77 akibat perubahan hyperparameter yang buruk), skrip evaluasi melempar sinyal kegagalan (exit code 1), menandai run sebagai FAILED, dan membatalkan seluruh proses selanjutnya. Tidak ada model cacat yang dapat lolos secara senyap.",
        body_style,
    ))

    story.append(Paragraph("<b>Tahap 4: Pembaruan Model Registry Berjenjang (Stage: Staging)</b>", heading2_style))
    story.append(Paragraph(
        "Model yang berhasil membuktikan keunggulannya tidak langsung dipublikasikan ke trafik pengguna (Production), melainkan secara otomatis didaftarkan ke MLflow Model Registry dengan stage <b>Staging</b>. Di tahap ini, metadata silsilah (commit SHA, branch, DVC hash, dan metrik hasil uji) diikat secara permanen pada model version. Hal ini memberikan transparansi audit yang menyeluruh dan memungkinkan tim teknis melakukan Root Cause Analysis (RCA) dalam hitungan menit jika terjadi anomali di masa depan.",
        body_style,
    ))

    # 6. Bagian 4: Tabel Matriks Moda Kegagalan vs Mekanisme Pencegahan
    story.append(Spacer(1, 4))
    story.append(Paragraph("4. Matriks Komparasi: Moda Kegagalan vs Mekanisme Proteksi Otomatis", heading1_style))

    table_data = [
        [
            Paragraph("<b>Moda Silent Failure</b>", table_header_style),
            Paragraph("<b>Manifestasi Tanpa Otomatisasi</b>", table_header_style),
            Paragraph("<b>Mekanisme Pencegahan Otomatis</b>", table_header_style),
            Paragraph("<b>Hasil Akhir Proteksi</b>", table_header_style),
        ],
        [
            Paragraph("<b>Regresi Kode Pra-pemrosesan</b>", table_cell_style),
            Paragraph("Fungsi imputasi mengembalikan nilai NaN atau tipe data salah. Inferensi tetap jalan dengan imputasi default 0.", table_cell_style),
            Paragraph("Job <code>automated-testing</code> (pytest) mengecek skema, imputasi numerik, dan tokenisasi.", table_cell_style),
            Paragraph("✅ Build dibatalkan pada Tahap 1 sebelum retraining dimulai.", table_cell_style),
        ],
        [
            Paragraph("<b>Degradasi Performa Model (Underfit/Overfit)</b>", table_cell_style),
            Paragraph("Hyperparameter buruk menyebabkan F1-Score turun drastis. Model cacat dideploy manual karena 'tidak ada error'.", table_cell_style),
            Paragraph("Gerbang evaluasi otomatis (<code>evaluate.py</code>) menguji metrik terhadap ambang batas ketat LK-01 (F1 $\ge 0.80$, Akurasi $\ge 82\%$).", table_cell_style),
            Paragraph("✅ Model ditolak (FAILED), tidak didaftarkan ke Staging.", table_cell_style),
        ],
        [
            Paragraph("<b>Pelanggaran Batas Latensi (SLA Spike)</b>", table_cell_style),
            Paragraph("Arsitektur pohon terlalu dalam membuat waktu inferensi membengkak ke 350 ms, melumpuhkan gateway API di bawah beban tinggi.", table_cell_style),
            Paragraph("Uji latensi empiris 20 iterasi pada skenario riil YouTube Shorts memverifikasi latensi p95 $< 200\text{ ms}$.", table_cell_style),
            Paragraph("✅ Menjamin model yang dipromosikan ke Staging selalu siap melayani beban real-time.", table_cell_style),
        ],
        [
            Paragraph("<b>Ketidakpastian Asal-usul Model (Lineage Loss)</b>", table_cell_style),
            Paragraph("Model dideploy dari laptop engineer tanpa catatan kode commit, dataset, atau hyperparameter mana yang dipakai.", table_cell_style),
            Paragraph("MLflow Model Registry otomatis mengikat model version dengan Git commit SHA dan DVC dataset MD5 hash.", table_cell_style),
            Paragraph("✅ 100% auditabilitas dan reproduksibilitas eksperimen terjamin.", table_cell_style),
        ],
    ]

    failure_table = Table(table_data, colWidths=[3.2 * cm, 4.4 * cm, 5.0 * cm, 4.4 * cm])
    failure_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), primary_color),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ("PADDING", (0, 0), (-1, -1), 4.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
    ]))
    story.append(failure_table)
    story.append(Spacer(1, 8))

    # 7. Bagian 5: Kesimpulan
    story.append(Paragraph("5. Kesimpulan dan Pembelajaran", heading1_style))
    story.append(Paragraph(
        "Demonstrasi prinsip <b>'Code as a Trigger'</b> pada proyek ViralSense membuktikan bahwa otomatisasi bukan sekadar kemudahan operasional (convenience), melainkan prasyarat mutlak untuk menjaga keandalan sistem kecerdasan buatan pada level produksi. Dengan mengalihkan beban verifikasi dari intuisi manual manusia ke alur kerja berbasis kode yang deterministik, tim pengembang dapat melakukan iterasi eksperimen secara cepat dan percaya diri tanpa kekhawatiran merusak kualitas layanan yang sedang berjalan.",
        body_style,
    ))
    story.append(Paragraph(
        "Pemisahan tanggung jawab antara tahap pengujian kode (unit testing), retraining bersilsilah (DVC), gerbang validasi ambang batas (LK-01 Evaluation Gate), dan promosi bertahap ke MLflow Staging menutup celah terjadinya kegagalan senyap (silent failures) di seluruh siklus hidup model, mewujudkan sistem MLOps yang tangguh, efisien, dan berintegritas tinggi.",
        body_style,
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Dokumen Refleksi PDF berhasil dibuat: {filename}")


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent / "Refleksi_Otomatisasi_Pemicu_MLOps_Silent_Failures.pdf"
    build_pdf(str(out_path))
