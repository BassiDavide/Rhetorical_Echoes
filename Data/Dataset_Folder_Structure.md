# Dataset folder

Anonymized data for the rhetorical alignment study, plus a script to recover the original text from YouTube.

## Structure of the folder

- **`finetuning_span_annotations/`** — comments and transcripts manually annotated with technique spans, used to fine-tune the models.
  - `comments/{train,dev,test}.jsonl` and `..._original_labels.jsonl` (same data, original vs. normalized technique names — see labels note below)
  - `transcripts/{train,dev,test}.jsonl`

- **`total_corpus_replication/`** — the full corpus used for the study's analyses, organized by YouTube channel.
**No text included** — only real YouTube IDs, model-predicted stance/techniques labels with character offsets (`start`/`end`), and metadata (timestamps, likes, channel leaning, etc.). 
Use `rehydrate.py` to recover the text. The sub-folders are organized as follows: 
  - `comments/<Channel>/comments__<VideoID>.jsonl`
  - `transcripts/<Channel>/<VideoID>.jsonl`

- **`scripts/`**
  - `rehydrate.py` — recovers real text for `total_corpus_replication/`.

Labels note: in `finetuning_span_annotations/comments/`, the non-suffixed files use a normalized/merged technique taxonomy (e.g. `smears/doubt`); the `_original_labels` files keep the annotators' original, more granular label names, with nothing merged or dropped (from Piskorski et al. 2025).

## Rehydrating `total_corpus_replication/`

**Comments** need a free YouTube Data API v3 key:
1. Sign in with (or create) a Google account.
2. Go to the [Google Cloud Console](https://console.cloud.google.com/) → create a project.
3. APIs & Services → Library → enable **"YouTube Data API v3"**.
4. APIs & Services → Credentials → **Create credentials → API key**.

**Transcripts** need one extra package:
```
pip install youtube-transcript-api
```
Caveat: this pulls the public auto-generated captions, which has no punctuation. We restored their punctuation with the PunctuationModel of Guhr et al. (2021). This may differ in punctuation/casing from the transcript text the original span offsets were computed on (that text went through a punctuation-restoration step) — treat transcript rehydration as best-effort, you can mail the corresponding author for additional additional support on this matter (bassidavide94@gmail.com).
