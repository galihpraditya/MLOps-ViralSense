"""Modul perolehan data dinamis (Ingestion) untuk TikTok secara luas (Broad FYP/Trending Discovery)."""

import json
import logging
import os
import re
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_env_file(env_path: str = ".env") -> None:
    """Membaca file .env secara otomatis ke dalam os.environ."""
    env_file = Path(env_path)
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key, val = key.strip(), val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val


load_env_file()


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Memuat konfigurasi pipeline dari file YAML."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def extract_tiktok_from_feed(
    regions: Optional[List[str]] = None,
    batch_limit: int = 30,
) -> List[Dict[str, Any]]:
    """Mengekstrak video TikTok yang sedang viral/FYP secara luas via Feed Rekomendasi Publik tanpa akun kaku."""
    regions = regions or ["US", "ID"]
    all_videos: List[Dict[str, Any]] = []
    seen_ids = set()
    now_str = datetime.utcnow().isoformat() + "Z"

    user_agent = os.getenv(
        "SCRAPER_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    )

    for region in regions:
        if len(all_videos) >= batch_limit:
            break

        url = f"https://www.tikwm.com/api/feed/list?region={region}&count={batch_limit}"
        logger.info(f"Mengambil feed video viral/FYP TikTok region '{region}'...")

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": user_agent,
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                items = data.get("data", [])
                if not items and isinstance(data.get("data"), dict):
                    items = data.get("data", {}).get("videos", [])

                logger.info(f"Menerima {len(items)} item dari feed TikTok region '{region}'.")

                for item in items:
                    v_id = str(item.get("video_id") or item.get("id") or "")
                    if not v_id or v_id in seen_ids:
                        continue

                    title = item.get("title") or item.get("content_desc") or "Trending TikTok Short"
                    author_data = item.get("author") or {}
                    uploader_id = author_data.get("unique_id") or "creator"
                    uploader_name = author_data.get("nickname") or uploader_id

                    views = int(item.get("play_count") or 0)
                    likes = int(item.get("digg_count") or 0)
                    comments = int(item.get("comment_count") or 0)
                    shares = int(item.get("share_count") or 0)
                    duration = int(item.get("duration") or 30)

                    # Filter durasi format short (<= 60 detik)
                    if duration > 60:
                        continue

                    # Ekstraksi timestamp rilis
                    create_time = item.get("create_time")
                    if create_time:
                        published_at = datetime.utcfromtimestamp(create_time).isoformat() + "Z"
                    else:
                        published_at = now_str

                    # Ekstraksi tagar
                    parsed_tags = re.findall(r"#(\w+)", title)
                    if not parsed_tags:
                        parsed_tags = ["tiktok", "fyp", "viral", region.lower()]

                    seen_ids.add(v_id)
                    all_videos.append(
                        {
                            "platform": "tiktok",
                            "video_id": f"tt_{v_id}",
                            "channel_id": uploader_id,
                            "channel_name": f"@{uploader_name}",
                            "title": title[:120],
                            "description": title[:300],
                            "published_at": published_at,
                            "duration_seconds": duration,
                            "view_count": views,
                            "like_count": likes,
                            "comment_count": comments,
                            "share_count": shares,
                            "tags": parsed_tags[:5],
                            "ingested_at": now_str,
                            "data_source_type": "TIKTOK_LIVE_FYP_FEED",
                        }
                    )
        except Exception as exc:
            logger.warning(f"Gagal mengambil feed TikTok region '{region}': {exc}")

    return all_videos


def extract_tiktok_from_apify(
    api_token: str,
    hashtags: List[str],
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Mengekstrak video TikTok berdasarkan tagar tren menggunakan Apify Actor."""
    import urllib.parse

    logger.info(f"Menggunakan Apify Actor untuk mengekstrak tagar tren TikTok: {hashtags[:3]}")
    all_videos: List[Dict[str, Any]] = []
    now_str = datetime.utcnow().isoformat() + "Z"

    # Panggil Apify REST API untuk actor tiktok-scraper
    run_url = f"https://api.apify.com/v2/acts/clockworks~tiktok-scraper/run-sync-get-dataset-items?token={api_token}"
    payload = {
        "hashtags": hashtags,
        "resultsPerPage": limit,
    }

    try:
        req = urllib.request.Request(
            run_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            items = json.loads(resp.read().decode("utf-8"))
            for item in items:
                v_id = str(item.get("id") or "")
                title = item.get("text") or "TikTok Viral"
                all_videos.append(
                    {
                        "platform": "tiktok",
                        "video_id": f"tt_{v_id}",
                        "channel_id": item.get("authorMeta", {}).get("name", "creator"),
                        "channel_name": f"@{item.get('authorMeta', {}).get('nickName', 'creator')}",
                        "title": title[:120],
                        "description": title[:300],
                        "published_at": item.get("createTimeISO") or now_str,
                        "duration_seconds": int(item.get("videoMeta", {}).get("duration", 30)),
                        "view_count": int(item.get("playCount", 0)),
                        "like_count": int(item.get("diggCount", 0)),
                        "comment_count": int(item.get("commentCount", 0)),
                        "share_count": int(item.get("shareCount", 0)),
                        "tags": item.get("hashtags", ["tiktok", "viral"]),
                        "ingested_at": now_str,
                        "data_source_type": "TIKTOK_APIFY_LIVE",
                    }
                )
    except Exception as exc:
        logger.warning(f"Gagal memanggil Apify TikTok Scraper: {exc}")

    return all_videos


def extract_tiktok_videos(
    config: Optional[Dict[str, Any]] = None,
    apify_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fungsi orkestrator ekstraksi video TikTok dengan strategi penemuan tren luas (tanpa akun statis)."""
    cfg = config or load_config()
    tt_cfg = cfg.get("data_sources", {}).get("tiktok", {})

    if not tt_cfg.get("enabled", True):
        logger.info("TikTok Ingestion dinonaktifkan dalam config.yaml.")
        return []

    regions = tt_cfg.get("feed_regions", ["US", "ID"])
    batch_limit = tt_cfg.get("batch_limit", 30)
    hashtags = tt_cfg.get("hashtags", ["fyp", "viral", "trending", "racuntiktok"])
    apify_token = apify_token or os.getenv("APIFY_API_TOKEN")

    records: List[Dict[str, Any]] = []

    # 1. Coba penarikan live FYP Feed publik secara luas
    records = extract_tiktok_from_feed(regions=regions, batch_limit=batch_limit)

    # 2. Jika tersedia Apify Token dan record feed masih sedikit, perkaya dengan Apify
    if apify_token and len(records) < 10:
        logger.info("Memperkaya data tren TikTok menggunakan Apify...")
        apify_records = extract_tiktok_from_apify(
            api_token=apify_token,
            hashtags=hashtags,
            limit=batch_limit,
        )
        # Gabungkan dan hilangkan duplikat ID
        existing_ids = {r["video_id"] for r in records}
        for ar in apify_records:
            if ar["video_id"] not in existing_ids:
                records.append(ar)
                existing_ids.add(ar["video_id"])

    # 3. Fallback sampel jika koneksi jaringan seluruhnya offline
    if not records:
        logger.warning("Koneksi TikTok offline, menyusun baseline broad dataset.")
        now_str = datetime.utcnow().isoformat() + "Z"
        sample_creators = ["kuliner_hits", "ide_bisnis_muda", "racun_outfit_indo", "daily_hacks_id"]
        for i, creator in enumerate(sample_creators):
            records.append(
                {
                    "platform": "tiktok",
                    "video_id": f"tt_viral_sample_{i+1}",
                    "channel_id": creator,
                    "channel_name": f"@{creator}",
                    "title": f"Ide konten dan racun produk viral trending #{i+1}",
                    "description": "Rekomendasi tren produk terlaris di TikTok minggu ini",
                    "published_at": now_str,
                    "duration_seconds": 25 + (i * 5),
                    "view_count": 350000 * (i + 1),
                    "like_count": 28000 * (i + 1),
                    "comment_count": 950 * (i + 1),
                    "share_count": 420 * (i + 1),
                    "tags": ["fyp", "viral", "racuntiktok"],
                    "ingested_at": now_str,
                    "data_source_type": "TIKTOK_SYNTHETIC_FALLBACK",
                }
            )

    logger.info(f"Total video TikTok viral berhasil diekstrak: {len(records)} video.")
    return records


def save_raw_data(data: List[Dict[str, Any]], platform: str = "tiktok", raw_dir: str = "data/raw") -> Path:
    """Menyimpan snapshot data mentah dalam format JSON append-only berbasis tanggal."""
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    timestamp_str = datetime.utcnow().strftime("%H%M%S")
    target_dir = Path(raw_dir) / platform / date_str
    target_dir.mkdir(parents=True, exist_ok=True)

    output_file = target_dir / f"{platform}_trending_{timestamp_str}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Berhasil menyimpan {len(data)} data mentah TikTok ke {output_file}")
    return output_file


if __name__ == "__main__":
    pipeline_cfg = load_config()
    raw_storage_dir = pipeline_cfg.get("storage", {}).get("raw_dir", "data/raw")

    logger.info("=== MEMULAI INGESTION TIKTOK (BROAD VIRAL/FYP) ===")
    extracted_records = extract_tiktok_videos(config=pipeline_cfg)

    if extracted_records:
        saved_file_path = save_raw_data(extracted_records, platform="tiktok", raw_dir=raw_storage_dir)
        print("\n" + "=" * 70)
        print(f"BUKTI INGESTION TIKTOK BERHASIL ({len(extracted_records)} TOTAL VIDEO):")
        print("=" * 70)
        for idx, item in enumerate(extracted_records[:3], 1):
            print(f"[{idx}] Kreator/Akun : {item['channel_name']}")
            print(f"    Judul/Caption: {item['title']}")
            print(f"    Video ID     : {item['video_id']}")
            print(f"    Durasi       : {item.get('duration_seconds', 0)} detik")
            print(f"    Views        : {item['view_count']:,} views")
            print(f"    Likes        : {item['like_count']:,} likes")
            print(f"    Published At : {item['published_at']}")
            print(f"    Tipe Sumber  : {item.get('data_source_type', 'N/A')}")
            print("-" * 70)
