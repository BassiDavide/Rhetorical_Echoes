"""
Rehydrate total_corpus_replication/ with real text fetched from YouTube.
Usage:
    python rehydrate.py --api-key YOUR_KEY
    YOUTUBE_API_KEY=YOUR_KEY python rehydrate.py --only comments
    python rehydrate.py --only transcripts
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

YOUTUBE_COMMENTS_URL = "https://www.googleapis.com/youtube/v3/comments"
BATCH_SIZE = 50


def iter_jsonl_files(root):
    return sorted(Path(root).rglob("*.jsonl"))


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            json.dump(record, fh, ensure_ascii=False)
            fh.write("\n")


# ---------------------------------------------------------------------------
# Comments (using YouTube Data API v3)
# ---------------------------------------------------------------------------

def fetch_comment_batch(comment_ids, api_key):
    """Return {comment_id: text} for up to 50 comment IDs in one API call."""
    params = {"part": "snippet", "id": ",".join(comment_ids), "key": api_key, "maxResults": 50}
    url = f"{YOUTUBE_COMMENTS_URL}?{urllib.parse.urlencode(params)}"

    for attempt in range(5):
        try:
            with urllib.request.urlopen(url) as resp:
                data = json.load(resp)
            break
        except urllib.error.HTTPError as e:
            if e.code == 403:
                raise RuntimeError(
                    "YouTube API returned 403 (quota exceeded, or the key is "
                    "invalid / YouTube Data API v3 is not enabled for it). "
                    "See README.md."
                ) from e
            if e.code == 429 or e.code >= 500:
                time.sleep(2 ** attempt)
                continue
            raise
    else:
        raise RuntimeError(f"Failed to fetch comment batch after retries (first id: {comment_ids[0]}).")

    result = {}
    for item in data.get("items", []):
        snippet = item["snippet"]
        result[item["id"]] = snippet.get("textOriginal", snippet.get("textDisplay"))
    return result


def rehydrate_comments_file(src_path, dst_path, api_key, sleep_s):
    records = list(read_jsonl(src_path))
    ids = [r["CommentID"] for r in records]

    text_by_id = {}
    for i in range(0, len(ids), BATCH_SIZE):
        batch = ids[i : i + BATCH_SIZE]
        text_by_id.update(fetch_comment_batch(batch, api_key))
        if sleep_s:
            time.sleep(sleep_s)

    missing = 0
    out = []
    for record in records:
        record = dict(record)
        text = text_by_id.get(record["CommentID"])
        if text is None:
            missing += 1  # deleted/removed comment, or no longer available
        record["CommentText"] = text
        out.append(record)

    write_jsonl(dst_path, out)
    return len(out), missing


# ---------------------------------------------------------------------------
# Transcripts (using youtube-transcript-api)
# ---------------------------------------------------------------------------

def rehydrate_transcripts_file(src_path, dst_path):
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as e:
        raise RuntimeError(
            "youtube-transcript-api is not installed. Run: pip install youtube-transcript-api"
        ) from e

    records = list(read_jsonl(src_path))
    failed = 0
    out = []
    for record in records:
        record = dict(record)
        video_id = record.get("videoId") or record.get("VideoID")
        try:
            segments = YouTubeTranscriptApi.get_transcript(video_id)
            record["CommentText"] = " ".join(seg["text"] for seg in segments)
        except Exception as e:  # noqa: BLE001 - report and keep going
            record["CommentText"] = None
            record["_rehydration_error"] = str(e)
            failed += 1
        record["text_source"] = (
            "youtube_transcript_api (public auto-captions; punctuation/casing "
            "may differ from the punctuation-restored transcript the original "
            "span offsets were computed on)"
        )
        out.append(record)

    write_jsonl(dst_path, out)
    return len(out), failed



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default="total_corpus_replication")
    parser.add_argument("--output-dir", default="total_corpus_replication_rehydrated")
    parser.add_argument("--api-key", default=os.environ.get("YOUTUBE_API_KEY"))
    parser.add_argument("--only", choices=["comments", "transcripts", "all"], default="all")
    parser.add_argument("--sleep", type=float, default=0.05, help="Seconds between comment API calls.")
    parser.add_argument("--limit-files", type=int, default=None, help="Process only the first N files (for testing).")
    args = parser.parse_args()

    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)

    if args.only in ("comments", "all"):
        if not args.api_key:
            raise SystemExit("Missing YouTube API key: pass --api-key or set YOUTUBE_API_KEY.")
        files = iter_jsonl_files(in_root / "comments")[: args.limit_files]
        total, missing = 0, 0
        for src in files:
            dst = out_root / src.relative_to(in_root)
            n, m = rehydrate_comments_file(src, dst, args.api_key, args.sleep)
            total, missing = total + n, missing + m
            print(f"[comments] {src.relative_to(in_root)}: {n} records, {m} missing/deleted")
        print(f"[comments] TOTAL: {total} records, {missing} missing/deleted")

    if args.only in ("transcripts", "all"):
        files = iter_jsonl_files(in_root / "transcripts")[: args.limit_files]
        total, failed = 0, 0
        for src in files:
            dst = out_root / src.relative_to(in_root)
            n, f = rehydrate_transcripts_file(src, dst)
            total, failed = total + n, failed + f
            print(f"[transcripts] {src.relative_to(in_root)}: {n} records, {f} failed")
        print(f"[transcripts] TOTAL: {total} records, {failed} failed")


if __name__ == "__main__":
    main()
