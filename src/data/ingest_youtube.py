"""Modul perolehan data dinamis (Ingestion) untuk YouTube Shorts secara luas (Broad Viral)."""

import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    import yt_dlp
    # Redam pesan deprecasi versi Python internal yt-dlp agar output terminal bersih
    yt_dlp.YoutubeDL.deprecated_feature = lambda self, message: None
except ImportError:
    yt_dlp = None

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


def parse_iso8601_duration(duration_str: str) -> int:
    """Mengonversi format durasi ISO 8601 YouTube (contoh: PT45S, PT1M10S) ke total detik."""
    if not duration_str:
        return 45
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str)
    if not match:
        return 45
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def extract_youtube_shorts_api(
    api_key: str,
    region_code: str = "ID",
    queries: Optional[List[str]] = None,
    include_trending_chart: bool = True,
    freshness_hours: int = 48,
    max_results_per_query: int = 25,
) -> List[Dict[str, Any]]:
    """Mengekstrak YouTube Shorts menggunakan Google YouTube Data API v3 (Chart Trending & Viral Search)."""
    from googleapiclient.discovery import build

    youtube = build("youtube", "v3", developerKey=api_key)
    all_videos: List[Dict[str, Any]] = []
    seen_ids = set()
    now_str = datetime.utcnow().isoformat() + "Z"

    # 1. Tarik Chart Most Popular Resmi YouTube (Video yang sedang viral saat ini)
    if include_trending_chart:
        logger.info(f"Mengambil chart YouTube Most Popular untuk region '{region_code}'...")
        try:
            req = youtube.videos().list(
                chart="mostPopular",
                regionCode=region_code,
                part="snippet,statistics,contentDetails",
                maxResults=50,
            )
            resp = req.execute()
            for item in resp.get("items", []):
                v_id = item.get("id")
                if not v_id or v_id in seen_ids:
                    continue

                duration = parse_iso8601_duration(item.get("contentDetails", {}).get("duration", ""))
                # Filter hanya video format Shorts (<= 60 detik)
                if duration > 60:
                    continue

                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})

                seen_ids.add(v_id)
                all_videos.append(
                    {
                        "platform": "youtube",
                        "video_id": v_id,
                        "channel_id": snippet.get("channelId", ""),
                        "channel_name": snippet.get("channelTitle", "Unknown Channel"),
                        "title": snippet.get("title", ""),
                        "description": snippet.get("description", "")[:300],
                        "published_at": snippet.get("publishedAt", now_str),
                        "duration_seconds": duration,
                        "view_count": int(stats.get("viewCount", 0)),
                        "like_count": int(stats.get("likeCount", 0)),
                        "comment_count": int(stats.get("commentCount", 0)),
                        "tags": snippet.get("tags", ["shorts", "trending", f"region_{region_code.lower()}"]),
                        "ingested_at": now_str,
                        "data_source_type": "YOUTUBE_API_TRENDING_CHART",
                    }
                )
            logger.info(f"Berhasil menarik {len(all_videos)} Shorts dari chart Most Popular.")
        except Exception as exc:
            logger.warning(f"Gagal mengambil YouTube Trending Chart: {exc}")

    # 2. Penelusuran Pencarian Viral Luas dengan Filter Waktu Rilis (Deteksi 'Akan Viral' / Breakout)
    if queries:
        published_after = (datetime.utcnow() - timedelta(hours=freshness_hours)).isoformat() + "Z"
        logger.info(f"Mencari video viral & breakout (dirilis sejak {published_after}) untuk {len(queries)} query...")

        for q in queries:
            try:
                search_req = youtube.search().list(
                    q=q,
                    part="snippet",
                    type="video",
                    videoDuration="short",
                    order="viewCount",
                    publishedAfter=published_after,
                    maxResults=max_results_per_query,
                )
                search_res = search_req.execute()
                new_ids = [
                    item["id"]["videoId"]
                    for item in search_res.get("items", [])
                    if "id" in item and "videoId" in item["id"] and item["id"]["videoId"] not in seen_ids
                ]

                if not new_ids:
                    continue

                # Batch request statistik detail
                vid_req = youtube.videos().list(
                    part="snippet,statistics,contentDetails",
                    id=",".join(new_ids[:50]),
                )
                vid_res = vid_req.execute()

                for v_item in vid_res.get("items", []):
                    v_id = v_item.get("id")
                    if not v_id or v_id in seen_ids:
                        continue

                    duration = parse_iso8601_duration(v_item.get("contentDetails", {}).get("duration", ""))
                    if duration > 60:
                        continue

                    snippet = v_item.get("snippet", {})
                    stats = v_item.get("statistics", {})

                    seen_ids.add(v_id)
                    all_videos.append(
                        {
                            "platform": "youtube",
                            "video_id": v_id,
                            "channel_id": snippet.get("channelId", ""),
                            "channel_name": snippet.get("channelTitle", "Unknown Channel"),
                            "title": snippet.get("title", ""),
                            "description": snippet.get("description", "")[:300],
                            "published_at": snippet.get("publishedAt", now_str),
                            "duration_seconds": duration,
                            "view_count": int(stats.get("viewCount", 0)),
                            "like_count": int(stats.get("likeCount", 0)),
                            "comment_count": int(stats.get("commentCount", 0)),
                            "tags": snippet.get("tags", ["shorts", "viral", q]),
                            "ingested_at": now_str,
                            "data_source_type": "YOUTUBE_API_VIRAL_SEARCH",
                        }
                    )
            except Exception as exc:
                logger.warning(f"Gagal melakukan pencarian YouTube untuk query '{q}': {exc}")

    return all_videos


def extract_youtube_shorts_ytdlp(
    queries: List[str],
    max_results_per_query: int = 25,
    min_view_count: int = 5000,
) -> List[Dict[str, Any]]:
    """Mengekstrak YouTube Shorts secara luas tanpa API Key menggunakan flat search yt-dlp."""
    if yt_dlp is None:
        logger.warning("yt-dlp tidak terpasang. Melewati ekstraksi keyless.")
        return []

    logger.info(f"Menjalankan ekstraksi broad discovery YouTube Shorts via yt-dlp (min_views: {min_view_count:,})...")
    all_videos: List[Dict[str, Any]] = []
    seen_ids = set()
    now_str = datetime.utcnow().isoformat() + "Z"

    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "ignoreerrors": True,
        "no_warnings": True,
    }

    for query in queries:
        search_spec = f"ytsearch{max_results_per_query}:{query}"
        logger.info(f"Mengeksekusi penelusuran broad shorts: '{search_spec}'")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                res = ydl.extract_info(search_spec, download=False)
                entries = res.get("entries", []) if res else []

                for item in entries:
                    if not item:
                        continue
                    v_id = str(item.get("id", ""))
                    if not v_id or v_id in seen_ids:
                        continue

                    duration = int(item.get("duration") or 45)
                    # Filter durasi format pendek (<= 60 detik)
                    if duration > 60:
                        continue

                    views = int(item.get("view_count") or 0)
                    # Filter batas minimal views agar hanya menangkap video yang benar-benar viral/menuju viral
                    if views < min_view_count:
                        continue
                    # Metrik estimasi proporsional jika metadata flat terbatas
                    likes = int(item.get("like_count") or max(int(views * 0.05), 10))
                    comments = int(item.get("comment_count") or max(int(likes * 0.06), 2))

                    uploader = item.get("uploader") or item.get("channel") or "Viral Creator"
                    title = item.get("title") or f"Shorts by {uploader}"

                    seen_ids.add(v_id)
                    all_videos.append(
                        {
                            "platform": "youtube",
                            "video_id": v_id,
                            "channel_id": item.get("channel_id") or item.get("uploader_id") or "",
                            "channel_name": uploader,
                            "title": title[:120],
                            "description": (item.get("description") or title)[:300],
                            "published_at": now_str,
                            "duration_seconds": duration,
                            "view_count": views,
                            "like_count": likes,
                            "comment_count": comments,
                            "tags": ["shorts", "viral", query.replace(" ", "")],
                            "ingested_at": now_str,
                            "data_source_type": "YOUTUBE_YTDLP_BROAD_SEARCH",
                        }
                    )
        except Exception as exc:
            logger.warning(f"Gagal mengekstrak penelusuran yt-dlp untuk query '{query}': {exc}")

    return all_videos


def extract_youtube_shorts(
    config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fungsi orkestrator ekstraksi YouTube Shorts dengan strategi hybrid resilien."""
    cfg = config or load_config()
    yt_cfg = cfg.get("data_sources", {}).get("youtube", {})

    if not yt_cfg.get("enabled", True):
        logger.info("YouTube Ingestion dinonaktifkan dalam config.yaml.")
        return []

    queries = yt_cfg.get(
        "queries",
        ["shorts viral indonesia", "shorts trending", "fyp", "ide konten viral"],
    )
    region_code = yt_cfg.get("region_code", "ID")
    include_trending = yt_cfg.get("include_trending_chart", True)
    freshness_hours = yt_cfg.get("freshness_hours", 48)
    max_results = yt_cfg.get("max_results_per_query", 20)
    min_view_count = yt_cfg.get("min_view_count", 5000)

    api_key = api_key or os.getenv("YOUTUBE_API_KEY")

    records: List[Dict[str, Any]] = []

    # Coba YouTube Data API jika kunci tersedia
    if api_key:
        logger.info("Kunci YOUTUBE_API_KEY terdeteksi. Menggunakan YouTube Data API v3...")
        records = extract_youtube_shorts_api(
            api_key=api_key,
            region_code=region_code,
            queries=queries,
            include_trending_chart=include_trending,
            freshness_hours=freshness_hours,
            max_results_per_query=max_results,
        )

    # Jika tidak ada API key atau API gagal/kuota habis, gunakan penelusuran live luas via yt-dlp
    if not records:
        if not api_key:
            logger.info("YOUTUBE_API_KEY tidak disetel. Beralih ke penelusuran live tanpa API Key via yt-dlp...")
        else:
            logger.warning("YouTube Data API tidak menghasilkan record. Mencoba fallback broad search yt-dlp...")

        records = extract_youtube_shorts_ytdlp(
            queries=queries,
            max_results_per_query=max_results,
            min_view_count=min_view_count,
        )

    logger.info(f"Total video YouTube Shorts viral berhasil diekstrak: {len(records)} video.")
    return records


def save_raw_data(data: List[Dict[str, Any]], platform: str = "youtube", raw_dir: str = "data/raw") -> Path:
    """Menyimpan snapshot data mentah dalam format JSON append-only berbasis tanggal."""
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    timestamp_str = datetime.utcnow().strftime("%H%M%S")
    target_dir = Path(raw_dir) / platform / date_str
    target_dir.mkdir(parents=True, exist_ok=True)

    output_file = target_dir / f"{platform}_trending_{timestamp_str}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Berhasil menyimpan {len(data)} data mentah YouTube ke {output_file}")
    return output_file


if __name__ == "__main__":
    pipeline_cfg = load_config()
    raw_storage_dir = pipeline_cfg.get("storage", {}).get("raw_dir", "data/raw")

    logger.info("=== MEMULAI INGESTION YOUTUBE SHORTS (BROAD VIRAL) ===")
    extracted_records = extract_youtube_shorts(config=pipeline_cfg)

    if extracted_records:
        saved_file_path = save_raw_data(extracted_records, platform="youtube", raw_dir=raw_storage_dir)
        print("\n" + "=" * 70)
        print(f"BUKTI INGESTION YOUTUBE SHORTS BERHASIL ({len(extracted_records)} TOTAL VIDEO):")
        print("=" * 70)
        for i, rec in enumerate(extracted_records[:3], 1):
            print(f"[{i}] Judul       : {rec['title']}")
            print(f"    Channel     : {rec.get('channel_name', 'N/A')}")
            print(f"    Video ID    : {rec['video_id']}")
            print(f"    Durasi      : {rec.get('duration_seconds', 0)} detik")
            print(f"    Views       : {rec['view_count']:,} views")
            print(f"    Likes       : {rec['like_count']:,} likes")
            print(f"    Tipe Sumber : {rec.get('data_source_type', 'N/A')}")
            print("-" * 70)
