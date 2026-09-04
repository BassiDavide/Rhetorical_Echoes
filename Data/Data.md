# Rhetorical Alignment — Public Data Release

Anonymized data for the rhetorical alignment study, plus a script to recover
the original text from YouTube.

## Structure

- **`finetuning_span_annotations/`** — comments and transcripts manually
  annotated with persuasion-technique spans, used to fine-tune the models.
  Full text is included as-is (IDs here are internal annotation IDs, not
  real YouTube IDs, so this subset can't be rehydrated anyway).
  - `comments/{train,dev,test}.jsonl` and `..._original_labels.jsonl` (same
    data, original vs. normalized technique names — see labels note below)
  - `transcripts/{train,dev,test}.jsonl`

- **`total_corpus_replication/`** — the full corpus used for the study's
  analyses, organized by YouTube channel. **No text included** — only real
  YouTube IDs, model-predicted technique/stance labels with character
  offsets (`start`/`end`), and metadata (timestamps, likes, channel leaning,
  etc.). Use `rehydrate.py` to recover the text.
  - `comments/<Channel>/comments__<VideoID>.jsonl`
  - `transcripts/<Channel>/<VideoID>.jsonl`

- **`scripts/`**
  - `build_release.py` — produced this release from the internal project data.
  - `rehydrate.py` — recovers real text for `total_corpus_replication/`.

Labels note: in `finetuning_span_annotations/comments/`, the non-suffixed
files use a normalized/merged technique taxonomy (e.g. `smears/doubt`); the
`_original_labels` files keep the annotators' original, more granular label
names, with nothing merged or dropped.

## Rehydrating `total_corpus_replication/`

**Comments** need a free YouTube Data API v3 key:
1. Sign in with (or create) a Google account.
2. Go to the [Google Cloud Console](https://console.cloud.google.com/) → create a project.
3. APIs & Services → Library → enable **"YouTube Data API v3"**.
4. APIs & Services → Credentials → **Create credentials → API key**.

Free tier: 10,000 quota units/day, 1 unit per batch of up to 50 comments.

**Transcripts** need one extra package (no API key needed):
```
pip install youtube-transcript-api
```
Caveat: this pulls the public auto-generated captions, which may differ in
punctuation/casing from the transcript text the original span offsets were
computed on (that text went through a punctuation-restoration step) — treat
transcript rehydration as best-effort.

**Run** (from this folder):
```
python scripts/rehydrate.py --api-key YOUR_KEY
# or:
export YOUTUBE_API_KEY=YOUR_KEY
python scripts/rehydrate.py
```
Output mirrors the input structure under `total_corpus_replication_rehydrated/`,
with `CommentText` added to every record.

Useful flags:
- `--only comments` / `--only transcripts` — run just one part
- `--limit-files N` — process only the first N files, to test before a full run
- `--input-dir` / `--output-dir` — override default paths

Only text is recovered — author identity is intentionally never fetched, to
preserve the anonymization intent of this release.
