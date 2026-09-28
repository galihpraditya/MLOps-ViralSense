"""Skrip Prapemrosesan Data (Preprocessing) Otomatis - ViralSense MLOps.

Membersihkan data mentah (raw data) yang dikumpulkan oleh modul ingestion:
1. Penggabungan snapshot raw data dari berbagai platform.
2. Penanganan missing values (imputasi numerik & teks).
3. Deduplikasi record berdasarkan composite key (platform, video_id).
4. Pemfilteran durasi format short-form (<= 60 detik).
5. Pembersihan teks (URL, mention, simbol/karakter khusus).
6. Tokenisasi (Tokenization) dan ekstraksi tagar/hashtag.
7. Penghapusan Stopwords (Stopword Removal) dwibahasa (Bahasa Indonesia & Inggris).
8. Penyimpanan output data bersih ke format Parquet dan CSV di data/processed/.
"""

import argparse
import glob
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
import yaml

# Tambahkan direktori root ke sys.path jika belum ada
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess")

# Daftar Stopwords Bahasa Indonesia Komprehensif (in-memory, aman offline tanpa dependensi eksternal)
INDONESIAN_STOPWORDS: Set[str] = {
    "ada", "adalah", "adanya", "adapun", "agak", "agar", "akan", "akankah", "akhir", "akhiri", "akhirnya",
    "aku", "akulah", "amat", "amatlah", "anda", "andalah", "antar", "antara", "antaranya", "apa", "apaan",
    "apabila", "apakah", "apalagi", "apatah", "artinya", "asal", "asalkan", "atas", "atau", "ataukah",
    "ataupun", "awal", "awalnya", "bagai", "bagaikan", "bagaimana", "bagaimanakah", "bagaimanapun", "bagi",
    "bahkan", "bahwa", "bahwasanya", "baik", "bakal", "bakalan", "balik", "banyak", "bapak", "barangkali",
    "baru", "bawah", "beberapa", "begini", "beginian", "beginikah", "beginilah", "begitu", "begitukah",
    "begitulah", "begitupun", "belakang", "belakangan", "belum", "belumlah", "benar", "benarkah", "benarlah",
    "berada", "berakhir", "berakhirlah", "berakhirnya", "berapa", "berapakah", "berapalah", "berapapun",
    "berarti", "berawal", "berbagai", "berdatangan", "beri", "berikan", "berikut", "berikutnya", "berjumlah",
    "berkali", "berkata", "berkehendak", "berkeinginan", "berkenaan", "berlainan", "berlalu", "berlangsung",
    "berlebihan", "bermacam", "bermaksud", "bermula", "bersama", "bersiap", "bertanya", "berturut",
    "bertutur", "berujar", "berupa", "besar", "betul", "betulkah", "biasa", "biasanya", "bila", "bilakah",
    "bisa", "bisakah", "boleh", "bolehkah", "bolehlah", "buat", "bukan", "bukankah", "bukanlah", "bukannya",
    "cuma", "dahulu", "dalam", "dan", "dapat", "dari", "daripada", "dekat", "demi", "demikian", "demikianlah",
    "dengan", "depan", "di", "dia", "diakhiri", "diakhirinya", "dialah", "diantara", "diantaranya", "diberi",
    "diberikan", "diberikannya", "dibuat", "dibuatnya", "didapat", "didatangkan", "digunakan", "dihendaki",
    "dini", "dipastikan", "diperbuat", "diperbuatnya", "dipergunakan", "diperkirakan", "diperlihatkan",
    "dipersoalkan", "dipunyai", "diri", "dirinya", "disebut", "disebutkan", "disebutkannya", "disini",
    "disinilah", "ditambahkan", "ditandaskan", "ditanya", "ditanyai", "ditanyakan", "ditegaskan", "ditujukan",
    "ditunjuk", "ditunjuki", "ditunjukkan", "ditunjukkannya", "ditunjuknya", "dituturkan", "dituturkannya",
    "diucapkan", "diucapkannya", "diungkapkan", "dong", "dulu", "empat", "enggak", "gak", "nggak", "entah",
    "entahlah", "hal", "hampir", "hanya", "hanyalah", "hari", "harus", "haruslah", "harusnya", "hendak",
    "hendaklah", "hendaknya", "hingga", "ia", "ialah", "ibarat", "ibaratkan", "ibaratnya", "ibu", "ikut",
    "ingat", "ingin", "inginkah", "inginkan", "ini", "inikah", "inilah", "itu", "itukah", "itulah", "jadi",
    "jadilah", "jadinya", "jangan", "jangankan", "janganlah", "jauh", "jawab", "jawaban", "jawabnya", "jelas",
    "jelaskan", "jelaslah", "jelasnya", "jika", "jikalau", "juga", "jumlah", "jumlahnya", "justru", "kala",
    "kalau", "kalaulah", "kalaupun", "kalian", "kami", "kamilah", "kamu", "kamulah", "kan", "kapan", "kapankah",
    "kapanpun", "karena", "karenanya", "kasus", "kata", "katakan", "katakanlah", "katanya", "ke", "keadaan",
    "kebetulan", "kecil", "kedua", "keduanya", "keinginan", "kelen", "kelak", "kelihatan", "kelihatannya",
    "kelima", "keluar", "kembali", "kemudian", "kemungkinan", "kemungkinannya", "kenapa", "kepada", "kepadanya",
    "kesampaian", "keseluruhan", "keseluruhannya", "keterlaluan", "ketika", "khususnya", "kini", "kinilah",
    "kira", "kiranya", "kita", "kitalah", "kok", "kurang", "lagi", "lagian", "lah", "lain", "lainnya",
    "lalu", "lama", "lamanya", "lanjut", "lanjutnya", "lebih", "lewat", "lihat", "luar", "macam", "maka",
    "makanya", "makin", "malah", "malahan", "mampu", "mampukah", "mana", "manakala", "manalagi", "masih",
    "masihkah", "masing", "masuk", "mata", "mau", "maupun", "melainkan", "melakukan", "melalui", "melihat",
    "melihatnya", "memang", "memastikan", "memberi", "memberikan", "membuat", "memerlukan", "memihak",
    "meminta", "memintakan", "memisalkan", "memperbuat", "mempergunakan", "memperkirakan", "memperlihatkan",
    "mempersiapkan", "mempersoalkan", "mempertanyakan", "mempunyai", "memulai", "memungkinkan", "menaiki",
    "menandaskan", "menanti", "menantikan", "menanya", "menanyai", "menanyakan", "mendapat", "mendapatkan",
    "mendatang", "mendatangi", "mendatangkan", "menegaskan", "mengakhiri", "mengapa", "mengatakan", "mengatakannya",
    "mengenai", "mengerjakan", "mengetahui", "menggunakan", "menghendaki", "mengibaratkan", "mengibaratkannya",
    "mengingat", "mengingatkan", "menginginkan", "mengira", "mengucapkan", "mengucapkannya", "mengungkapkan",
    "menjadi", "menjawab", "menuju", "menunjuk", "menunjuki", "menunjukkan", "menunjuknya", "menurut",
    "menurutnya", "menuturkan", "menyampaikan", "menyangkut", "menyatakan", "menyebutkan", "menyeluruh",
    "menyiapkan", "merasa", "mereka", "merekalah", "merupakan", "meski", "meskipun", "meyakini", "meyakinkan",
    "minta", "mirip", "misal", "misalkan", "misalnya", "mula", "mulai", "mulailah", "mulanya", "mungkin",
    "mungkinkah", "nah", "naik", "namun", "nanti", "nantinya", "nyaris", "nyatanya", "olah", "oleh",
    "olehnya", "pada", "padahal", "padanya", "pak", "paling", "panjang", "pantas", "para", "pasti", "pastilah",
    "penting", "pentingnya", "per", "percuma", "perlu", "perlukah", "perlunya", "pernah", "persoalan", "pertama",
    "pertamakali", "pula", "pulang", "pun", "punya", "rasa", "rasanya", "rata", "rupanya", "saat", "saatnya",
    "saja", "sajalah", "saling", "sama", "sampai", "sana", "sangat", "sangatlah", "satu", "saya", "sayalah",
    "se", "sebab", "sebabnya", "sebagai", "sebagaimana", "sebagainya", "sebagian", "sebaik", "sebaiknya",
    "sebaliknya", "sebanyak", "sebegini", "sebegitu", "sebelum", "sebelumnya", "sebenarnya", "seberapa",
    "sebesar", "sebetulnya", "sebisanya", "sebuah", "sebut", "sebutlah", "sebutnya", "secara", "secukupnya",
    "sedang", "sedangkan", "sedemikian", "sedikit", "sedikitnya", "seenaknya", "segala", "segalanya", "segera",
    "seharusnya", "sehingga", "seingat", "sejak", "sejauh", "sejenak", "sejumlah", "sekadar", "sekadarnya",
    "sekali", "sekalian", "sekaligus", "sekalipun", "sekarang", "sekaranglah", "sekecil", "seketika", "sekiranya",
    "sekitar", "sekitarnya", "sekurang", "sekurangnya", "sela", "selain", "selalu", "selama", "selamanya",
    "selanjutnya", "seluruh", "seluruhnya", "semacam", "semakin", "semampu", "semampunya", "semasa", "semasih",
    "semata", "semaunya", "sementara", "semisal", "semisalnya", "sempat", "semua", "semuanya", "semula",
    "sendiri", "sendirian", "sendirinya", "seolah", "seorang", "sepanjang", "sepantasnya", "sepantasnyalah",
    "seperlunya", "seperti", "sepertinya", "sepihak", "sering", "seringnya", "serta", "serupa", "sesaat",
    "sesama", "sesampai", "sesegera", "sesekali", "seseorang", "sesuatu", "sesuatunya", "sesudah", "sesudahnya",
    "setelah", "setempat", "setengah", "seterusnya", "setiap", "setiba", "setibanya", "setidak", "setidaknya",
    "setinggi", "seusai", "sewaktu", "siap", "siapa", "siapakah", "siapapun", "sini", "sinilah", "soal",
    "soalnya", "suatu", "sudah", "sudahkah", "sudahlah", "supaya", "tadi", "tadinya", "tahu", "tahun",
    "tak", "tanpa", "tanya", "tanyakan", "tanyanya", "tapi", "tegas", "tegasnya", "telah", "tempat", "tengah",
    "tentang", "tentu", "tentulah", "tentunya", "tepat", "terakhir", "terasa", "terbanyak", "terdahulu",
    "terdapat", "terdiri", "terhadap", "terhadapnya", "teringat", "terjadi", "terjadilah", "terjadinya", "terkira",
    "terlalu", "terlebih", "terlihat", "termasuk", "ternyata", "tersampaikan", "tersebut", "tersebutlah",
    "tertentu", "tertuju", "terus", "terutama", "tetap", "tetapi", "tiap", "tiba", "tidak", "tidakkah",
    "tidaklah", "toh", "waduh", "wah", "wahai", "waktu", "waktunya", "walau", "walaupun", "wong", "yaitu",
    "yakin", "yakni", "yang", "ya", "yuk", "nih", "deh", "kan", "tuh", "kah",
}

# Stopwords Bahasa Inggris Standar
ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
    "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
    "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves",
}

ALL_STOPWORDS: Set[str] = INDONESIAN_STOPWORDS.union(ENGLISH_STOPWORDS)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Memuat konfigurasi pipeline dari file YAML."""
    full_path = PROJECT_ROOT / config_path
    if full_path.exists():
        with open(full_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    return {}


def load_all_raw_data(raw_dir: Path) -> pd.DataFrame:
    """Membaca seluruh snapshot data mentah (JSON dan CSV) yang tersimpan di data/raw/."""
    json_files = glob.glob(str(raw_dir / "**" / "*.json"), recursive=True)
    csv_files = [f for f in glob.glob(str(raw_dir / "*.csv")) if "sample" in f or "snapshot" in f]

    records: List[Dict[str, Any]] = []

    # 1. Baca seluruh file JSON snapshot
    for filepath in json_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records.extend(data)
                elif isinstance(data, dict):
                    records.append(data)
        except Exception as exc:
            logger.warning(f"Lewati file rusak {filepath}: {exc}")

    if records:
        logger.info(f"Membaca {len(records)} baris data dari {len(json_files)} file JSON snapshot.")
        df = pd.DataFrame(records)
    elif csv_files:
        # Fallback membaca CSV jika JSON belum ada
        logger.info(f"Membaca data mentah dari CSV fallback: {csv_files[0]}")
        df = pd.read_csv(csv_files[0])
    else:
        logger.warning(f"Tidak ada file raw data yang ditemukan di direktori {raw_dir}")
        return pd.DataFrame()

    return df


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Menangani missing values untuk seluruh atribut numerik dan teks."""
    df = df.copy()

    # Kolom numerik diisi dengan 0 dan dikonversi ke tipe data yang sesuai
    numeric_columns = ["view_count", "like_count", "comment_count", "share_count", "duration_seconds"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            if col in ["view_count", "like_count", "comment_count", "share_count"]:
                df[col] = df[col].astype("int64")
            elif col == "duration_seconds":
                df[col] = df[col].astype("int32")

    # Kolom teks diisi dengan string kosong dan dibersihkan dari whitespace ekstra
    text_columns = ["title", "description", "channel_name", "channel_id", "platform", "video_id"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    return df


def extract_hashtags_from_text(text: str) -> List[str]:
    """Mengekstrak seluruh tagar (#hashtag) dari teks."""
    if not text:
        return []
    hashtags = re.findall(r"#\w+", text.lower())
    return [tag.lstrip("#") for tag in hashtags]


def clean_text_for_nlp(text: str) -> str:
    """Membersihkan teks mentah: menghapus URL, username mention, tanda baca berlebih, dan karakter aneh."""
    if not text:
        return ""

    # Hapus URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Hapus mention (@username)
    text = re.sub(r"@\w+", "", text)
    # Hapus simbol khusus dan angka yang berdiri sendiri, sisakan huruf dan spasi
    text = re.sub(r"[^\w\s]", " ", text)
    # Ganti underscore dengan spasi jika tersisa
    text = text.replace("_", " ")
    # Normalisasi spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()
    # Case folding (huruf kecil)
    return text.lower()


def tokenize(text: str) -> List[str]:
    """Melakukan tokenisasi teks menjadi daftar kata (tokens)."""
    if not text:
        return []
    # Tokenisasi berbasis regex kata alfanumerik
    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    return tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Menghapus stopwords bahasa Indonesia dan bahasa Inggris dari daftar token."""
    return [token for token in tokens if token not in ALL_STOPWORDS and len(token) > 1]


def preprocess_data(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Pipeline prapemrosesan data mentah lengkap: pembersihan, deduplikasi, tokenisasi, & stopword removal."""
    if df.empty:
        logger.warning("DataFrame kosong, melewati tahap prapemrosesan.")
        return df

    initial_count = len(df)
    logger.info(f"Memulai prapemrosesan untuk {initial_count} baris data mentah...")

    # 1. Penanganan Missing Values
    df = clean_missing_values(df)

    # 2. Deduplikasi record berdasarkan platform dan video_id
    dedup_subset = ["platform", "video_id"]
    valid_keys = [k for k in dedup_subset if k in df.columns]
    if valid_keys:
        df = df.drop_duplicates(subset=valid_keys, keep="last")
        logger.info(f"Deduplikasi selesai: {initial_count} -> {len(df)} records tersisa.")

    # 3. Penyaringan durasi short-form (maksimal 60 detik)
    max_duration = cfg.get("cleaning", {}).get("max_video_duration_seconds", 60)
    if "duration_seconds" in df.columns:
        # Hanya saring jika durasi > 0 dan > max_duration
        df = df[(df["duration_seconds"] <= max_duration) | (df["duration_seconds"] == 0)]
        logger.info(f"Filter durasi short-form (<= {max_duration} detik): {len(df)} records tersisa.")

    # 4. Standardisasi timestamp
    time_columns = ["published_at", "ingested_at"]
    for col in time_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # 5. Gabungkan judul dan deskripsi untuk representasi konten teks lengkap
    title_series = df["title"].astype(str) if "title" in df.columns else pd.Series([""] * len(df), index=df.index)
    desc_series = df["description"].astype(str) if "description" in df.columns else pd.Series([""] * len(df), index=df.index)
    df["full_content_text"] = (title_series + " " + desc_series).str.strip()

    # 6. Ekstraksi Tagar / Hashtags
    df["extracted_hashtags"] = df["full_content_text"].apply(extract_hashtags_from_text)

    # 7. Pembersihan Teks NLP
    df["cleaned_text"] = df["full_content_text"].apply(clean_text_for_nlp)

    # 8. Tokenisasi (Tokenization)
    df["tokens"] = df["cleaned_text"].apply(tokenize)
    df["token_count"] = df["tokens"].apply(len)

    # 9. Penghapusan Stopwords (Stopword Removal)
    df["clean_tokens"] = df["tokens"].apply(remove_stopwords)
    df["clean_token_count"] = df["clean_tokens"].apply(len)
    df["clean_text_final"] = df["clean_tokens"].apply(lambda tok_list: " ".join(tok_list))

    logger.info(f"Prapemrosesan berhasil. Total data bersih siap latih/fitur: {len(df)} baris.")
    return df


def save_processed_data(df: pd.DataFrame, processed_dir: Path) -> Dict[str, Path]:
    """Menyimpan data hasil prapemrosesan ke dalam format Parquet dan CSV."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = processed_dir / "clean_videos.parquet"
    csv_file = processed_dir / "clean_videos.csv"

    # Simpan ke Parquet untuk efisiensi pipeline ML
    df.to_parquet(parquet_file, index=False)

    # Simpan versi CSV untuk kemudahan inspeksi langsung
    # Untuk kolom list (tokens, hashtags), format string agar CSV rapi
    df_csv = df.copy()
    if "extracted_hashtags" in df_csv.columns:
        df_csv["extracted_hashtags"] = df_csv["extracted_hashtags"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    if "tokens" in df_csv.columns:
        df_csv["tokens"] = df_csv["tokens"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    if "clean_tokens" in df_csv.columns:
        df_csv["clean_tokens"] = df_csv["clean_tokens"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

    df_csv.to_csv(csv_file, index=False)

    logger.info(f"Data bersih disimpan ke Parquet: {parquet_file}")
    logger.info(f"Data bersih disimpan ke CSV:     {csv_file}")
    return {"parquet": parquet_file, "csv": csv_file}


def print_preprocessing_summary(df: pd.DataFrame) -> None:
    """Mencetak sampel dan statistik pembersihan data."""
    if df.empty:
        return

    print("\n" + "=" * 80)
    print(f"BUKTI HASIL PRAPEMROSESAN DATA (TOTAL: {len(df)} BARIS BERSIH):")
    print("=" * 80)
    for idx, (_, row) in enumerate(df.head(3).iterrows(), 1):
        print(f"[{idx}] Platform     : {row.get('platform', 'N/A').upper()} | ID: {row.get('video_id', 'N/A')}")
        print(f"    Judul Asli   : {row.get('title', '')[:65]}...")
        print(f"    Clean Text   : {row.get('clean_text_final', '')[:65]}...")
        print(f"    Tokens Awal  : {row.get('tokens', [])[:8]}")
        print(f"    Clean Tokens : {row.get('clean_tokens', [])[:8]}")
        print(f"    Views/Likes  : {int(row.get('view_count', 0)):,} views | {int(row.get('like_count', 0)):,} likes")
        print("-" * 80)
    print("=" * 80 + "\n")


def parse_arguments() -> argparse.Namespace:
    """Parsing argumen baris perintah (CLI)."""
    parser = argparse.ArgumentParser(
        description="Skrip Prapemrosesan Data Otomatis ViralSense (Tokenization, Stopwords, Missing Values)."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Direktori input data mentah (default: data/raw).",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Direktori output data bersih (default: data/processed).",
    )
    return parser.parse_args()


def main() -> None:
    """Fungsi utama orchestrator preprocessing."""
    args = parse_arguments()
    config = load_config()

    raw_dir = PROJECT_ROOT / args.raw_dir
    processed_dir = PROJECT_ROOT / args.processed_dir

    logger.info("============================================================")
    logger.info("    MEMULAI AUTOMASI PRAPEMROSESAN DATA - VIRALSENSE       ")
    logger.info("============================================================")
    logger.info(f"Direktori Raw Input  : {raw_dir}")
    logger.info(f"Direktori Processed  : {processed_dir}")

    # 1. Baca seluruh snapshot data mentah
    raw_df = load_all_raw_data(raw_dir=raw_dir)

    if raw_df.empty:
        logger.error("Data mentah tidak ditemukan. Jalankan 'python src/ingest_data.py' terlebih dahulu.")
        sys.exit(1)

    # 2. Proses prapemrosesan data
    clean_df = preprocess_data(raw_df, config)

    # 3. Simpan data bersih
    save_processed_data(clean_df, processed_dir)

    # 4. Tampilkan ringkasan verifikasi
    print_preprocessing_summary(clean_df)
    logger.info("Automasi prapemrosesan berhasil diselesaikan.")


if __name__ == "__main__":
    main()
