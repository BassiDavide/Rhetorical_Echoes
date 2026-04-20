#!/usr/bin/env python3
import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from span_singlehead.data_processing import SpanDataProcessor
from scripts.labels import LABEL_LIST


DEFAULT_INPUT_ROOT = Path(
    "ADD_PATH"
)
DEFAULT_OUTPUT_FILE = Path(
    "ADD_PATH"
)


def find_zip_paths(root: Path) -> List[Path]:
    return sorted(root.rglob("inception-document*.zip"))


def extract_text(data: Dict, sofa_id: str) -> str:
    referenced = data.get("_referenced_fss", {})
    sofa = referenced.get(sofa_id)
    if isinstance(sofa, dict):
        text = sofa.get("sofaString")
        if isinstance(text, str):
            return text
    for value in referenced.values():
        if isinstance(value, dict) and isinstance(value.get("sofaString"), str):
            return value["sofaString"]
    return ""


def parse_one_zip(
    zip_path: Path,
    processor: SpanDataProcessor,
    raw_counts: Counter,
    normalized_counts: Counter,
    dropped_counts: Counter,
) -> Optional[Dict]:
    with zipfile.ZipFile(zip_path) as zf:
        try:
            payload = json.loads(zf.read("CURATION_USER.json"))
        except KeyError:
            print(f"[warn] missing CURATION_USER.json in {zip_path}", file=sys.stderr)
            return None

    views = payload.get("_views") or {}
    view = views.get("_InitialView") or (next(iter(views.values())) if views else None)
    if view is None:
        print(f"[warn] no view in {zip_path}", file=sys.stderr)
        return None

    meta = (view.get("DocumentMetaData") or [{}])[0]
    doc_title = meta.get("documentTitle") or zip_path.parent.name
    raw_doc_id = doc_title[:-4] if doc_title.endswith(".txt") else doc_title
    doc_id = f"Transcript_{raw_doc_id}"
    sofa_id = str(meta.get("sofa", "1"))
    text = extract_text(payload, sofa_id)

    annotations = []
    for ann in view.get("Persuasiontechnique", []) or []:
        raw_label = ann.get("technique")
        if raw_label:
            raw_counts[raw_label] += 1
        norm = processor.normalize_technique(raw_label)
        if norm is None:
            if raw_label:
                dropped_counts[raw_label] += 1
            continue
        normalized_counts[norm] += 1

        try:
            start = int(ann.get("begin"))
            end = int(ann.get("end"))
        except (TypeError, ValueError):
            continue

        start = max(0, min(start, len(text)))
        end = max(0, min(end, len(text)))
        if end <= start:
            continue

        annotations.append(
            {
                "technique": norm,
                "start": start,
                "end": end,
                "text_span": text[start:end],
            }
        )

    return {
        "CommentID": doc_id,
        "RawCommentID": raw_doc_id,
        "Source": "transcripts",
        "CommentText": text,
        "annotations": annotations,
        "Mapping": {
            "batch_name": "Transcripts",
            "type": "TRANSCRIPTS",
            "new_filename": doc_title,
            "original_filename": doc_title,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert transcript INCEpTION exports into JSONL records compatible with Span_SingleHead."
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        print(f"[error] input root not found: {input_root}", file=sys.stderr)
        return 1

    zip_paths = find_zip_paths(input_root)
    if not zip_paths:
        print(f"[error] no inception-document*.zip found under {input_root}", file=sys.stderr)
        return 1

    processor = SpanDataProcessor(LABEL_LIST)
    raw_counts = Counter()
    normalized_counts = Counter()
    dropped_counts = Counter()

    items: List[Dict] = []
    for zip_path in zip_paths:
        item = parse_one_zip(zip_path, processor, raw_counts, normalized_counts, dropped_counts)
        if item is not None:
            items.append(item)

    with output_file.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_annotations = sum(len(item["annotations"]) for item in items)
    print(f"Wrote {len(items)} documents to {output_file}")
    print(f"Total normalized annotations: {total_annotations}")
    if dropped_counts:
        print(f"Dropped annotations: {sum(dropped_counts.values())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

