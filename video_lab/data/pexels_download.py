"""Download niche training clips from the Pexels Videos API.

Attribution (required by Pexels):
- Show a link to https://www.pexels.com (UI / README / logs)
- Credit videographers when possible
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from video_lab import DATA_ROOT, RAW_DIR, ensure_dirs

PEXELS_VIDEO_SEARCH = "https://api.pexels.com/v1/videos/search"
PEXELS_INDEX_PATH = DATA_ROOT / "pexels_index.jsonl"
PEXELS_HOME = "https://www.pexels.com"


def _api_key() -> str:
    key = (os.environ.get("PEXELS_API_KEY") or os.environ.get("PEXELS_API") or "").strip()
    if not key:
        raise RuntimeError(
            "Set PEXELS_API_KEY in .env (get a free key at https://www.pexels.com/api/)."
        )
    return key


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return (s or "clip")[:max_len]


def load_pexels_index(path: Path | None = None) -> dict[str, dict]:
    path = path or PEXELS_INDEX_PATH
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            name = str(row.get("filename") or "")
            if name:
                out[name] = row
    return out


def append_pexels_index(rows: list[dict], path: Path | None = None) -> Path:
    path = path or PEXELS_INDEX_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def caption_from_pexels_meta(filename: str, index: dict[str, dict] | None = None) -> str | None:
    index = index if index is not None else load_pexels_index()
    row = index.get(filename)
    if not row:
        return None
    return str(row.get("caption") or "").strip() or None


def _request_json(url: str, api_key: str, retries: int = 3) -> dict:
    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": api_key,
                "User-Agent": "movie-AI-video-lab/1.0 (training data; Videos provided by Pexels)",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                remaining = resp.headers.get("X-Ratelimit-Remaining")
                if remaining is not None and int(remaining) < 5:
                    time.sleep(2.0)
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                time.sleep(5.0 * (attempt + 1))
                continue
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Pexels HTTP {e.code}: {body[:300]}") from e
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Pexels request failed: {last_err}")


def _pick_video_file(
    video: dict,
    *,
    prefer_max_width: int = 1280,
    min_width: int = 360,
) -> dict | None:
    """Pick an MP4 near ~720–1280 on the long edge (portrait + landscape CDNs)."""
    files = [f for f in (video.get("video_files") or []) if str(f.get("file_type", "")).startswith("video/mp4")]
    files = [f for f in files if f.get("link") and f.get("width") and f.get("height")]
    if not files:
        return None

    def long_edge(f: dict) -> int:
        return max(int(f["width"]), int(f["height"]))

    # Prefer HD under/near prefer_max_width; avoid UHD/4K when a smaller file exists
    under = [f for f in files if long_edge(f) <= prefer_max_width and long_edge(f) >= min_width]
    pool = under or [f for f in files if long_edge(f) <= 1920] or files
    quality_rank = {"hd": 0, "sd": 1, "uhd": 2}

    def score(f: dict) -> tuple:
        q = quality_rank.get(str(f.get("quality") or "").lower(), 3)
        le = long_edge(f)
        over = max(0, le - prefer_max_width)
        return (over, abs(le - prefer_max_width), q, -le)

    return min(pool, key=score)


def _caption_from_video(video: dict, query: str) -> str:
    """Build training caption from Pexels page URL slug + credit."""
    page_url = str(video.get("url") or "")
    slug = ""
    m = re.search(r"/video/([^/]+)-(\d+)/?$", page_url)
    if m:
        slug = m.group(1).replace("-", " ").strip()
    user = video.get("user") or {}
    author = str(user.get("name") or "Pexels creator").strip()
    duration = int(video.get("duration") or 0)
    base = slug or f"{query}, cinematic stock footage"
    return f"{base}, {duration}s. Video by {author} on Pexels"


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "movie-AI-video-lab/1.0"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)


def search_videos(
    query: str,
    *,
    api_key: str | None = None,
    page: int = 1,
    per_page: int = 80,
    orientation: str = "landscape",
    size: str = "medium",
) -> dict:
    key = api_key or _api_key()
    params = {
        "query": query,
        "page": page,
        "per_page": min(80, max(1, per_page)),
        "orientation": orientation,
        "size": size,
    }
    url = f"{PEXELS_VIDEO_SEARCH}?{urllib.parse.urlencode(params)}"
    return _request_json(url, key)


def download_pexels_videos(
    query: str,
    *,
    target_count: int = 200,
    out_dir: Path | None = None,
    min_duration: int = 3,
    max_duration: int = 15,
    orientation: str = "landscape",
    size: str = "medium",
    prefer_max_width: int = 1280,
    sleep_s: float = 0.35,
    log_fn=None,
) -> dict:
    """Search + download MP4s into raw/. Returns summary dict."""

    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    ensure_dirs()
    out_dir = Path(out_dir or RAW_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    api_key = _api_key()
    query = (query or "").strip().lstrip("=").strip() or "nature"

    log(f"Videos provided by Pexels - {PEXELS_HOME}")
    log(f"Query={query!r} target={target_count} duration={min_duration}-{max_duration}s -> {out_dir}")

    existing = load_pexels_index()
    have_ids = {int(r["pexels_id"]) for r in existing.values() if r.get("pexels_id") is not None}
    downloaded = 0
    skipped = 0
    pages = 0
    new_rows: list[dict] = []
    page = 1
    per_page = 80

    while downloaded < target_count:
        pages += 1
        data = search_videos(
            query,
            api_key=api_key,
            page=page,
            per_page=per_page,
            orientation=orientation,
            size=size,
        )
        videos = data.get("videos") or []
        if not videos:
            log(f"No more results at page {page}.")
            break
        log(f"Page {page}: {len(videos)} candidates (total_results={data.get('total_results')})")

        for video in videos:
            if downloaded >= target_count:
                break
            vid = int(video["id"])
            if vid in have_ids:
                skipped += 1
                continue
            duration = int(video.get("duration") or 0)
            if duration < min_duration or duration > max_duration:
                skipped += 1
                continue
            file_info = _pick_video_file(video, prefer_max_width=prefer_max_width)
            if not file_info:
                skipped += 1
                continue

            user = video.get("user") or {}
            author = str(user.get("name") or "Pexels creator").strip()
            author_url = str(user.get("url") or PEXELS_HOME).strip()
            page_url = str(video.get("url") or PEXELS_HOME).strip()
            filename = f"pexels_{vid}_{_slug(query)}.mp4"
            dest = out_dir / filename
            if dest.exists() and dest.stat().st_size > 10_000:
                skipped += 1
                have_ids.add(vid)
                continue

            caption = _caption_from_video(video, query)
            try:
                _download_file(str(file_info["link"]), dest)
            except Exception as e:
                log(f"  fail id={vid}: {e}")
                if dest.exists():
                    dest.unlink(missing_ok=True)
                continue

            row = {
                "filename": filename,
                "path": str(dest.resolve()),
                "pexels_id": vid,
                "query": query,
                "caption": caption,
                "duration": duration,
                "width": int(file_info.get("width") or video.get("width") or 0),
                "height": int(file_info.get("height") or video.get("height") or 0),
                "fps": file_info.get("fps"),
                "photographer": author,
                "photographer_url": author_url,
                "pexels_url": page_url,
                "credit": f"Video by {author} on Pexels - {page_url}",
                "attribution_note": f"Videos provided by Pexels ({PEXELS_HOME})",
            }
            new_rows.append(row)
            have_ids.add(vid)
            downloaded += 1
            log(f"  [{downloaded}/{target_count}] {filename} - {row['credit']}")
            time.sleep(sleep_s)

        if not data.get("next_page"):
            log("Reached last search page.")
            break
        page += 1
        time.sleep(max(sleep_s, 0.5))  # gentle on 200 req/hour search quota

    if new_rows:
        append_pexels_index(new_rows)
        log(f"Attribution index updated: {PEXELS_INDEX_PATH}")

    summary = {
        "downloaded": downloaded,
        "skipped": skipped,
        "pages": pages,
        "out_dir": str(out_dir),
        "index": str(PEXELS_INDEX_PATH),
        "pexels": PEXELS_HOME,
    }
    log(
        f"Done. downloaded={downloaded} skipped={skipped} pages={pages}. "
        f"Videos provided by Pexels - {PEXELS_HOME}"
    )
    return summary
