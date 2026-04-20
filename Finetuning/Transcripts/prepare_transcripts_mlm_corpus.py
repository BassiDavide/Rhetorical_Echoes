#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple
from urllib.parse import unquote


VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,20}$")
WS_RE = re.compile(r"\s+")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build train/val text corpora for transcript MLM. "
            "Can exclude supervised dev/test documents to reduce leakage."
        )
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="Root folder with transcript .txt files (recursive).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output folder for mlm_train.txt / mlm_val.txt and manifest.",
    )
    parser.add_argument(
        "--dev-docs-jsonl",
        default="",
        help="Supervised dev docs JSONL (optional, used for exclusion).",
    )
    parser.add_argument(
        "--test-docs-jsonl",
        default="",
        help="Supervised test docs JSONL (optional, used for exclusion).",
    )
    parser.add_argument(
        "--exclude-supervised-devtest",
        action="store_true",
        help="Exclude docs matching supervised dev/test identifiers.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument(
        "--dedupe-text",
        action="store_true",
        help="Dedupe by normalized text hash.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return WS_RE.sub(" ", text).strip()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def canonical_name_variants(name: str) -> Set[str]:
    out: Set[str] = set()
    raw = Path(name).name
    dec = unquote(raw)
    for cand in {raw, dec}:
        c = cand.strip()
        if not c:
            continue
        out.add(c.casefold())
        stem = c[:-4] if c.casefold().endswith(".txt") else c
        out.add(stem.casefold())
        if "__" in stem:
            tail = stem.split("__")[-1]
            out.add(tail.casefold())
        for pref in ("train_", "dev_", "test_", "sample_"):
            if stem.casefold().startswith(pref):
                out.add(stem[len(pref) :].casefold())
    return out


def extract_video_ids_from_name(name: str) -> Set[str]:
    ids: Set[str] = set()
    for variant in canonical_name_variants(name):
        cand = variant
        if ".en." in cand:
            cand = cand.split(".en.")[0]
        elif cand.endswith(".en"):
            cand = cand[:-3]
        if "__" in cand:
            cand = cand.split("__")[-1]
        if VIDEO_ID_RE.fullmatch(cand):
            ids.add(cand)
    return ids


def build_exclusion_sets(paths: Iterable[Path]) -> Tuple[Set[str], Set[str]]:
    excluded_names: Set[str] = set()
    excluded_video_ids: Set[str] = set()
    for path in paths:
        if not path or not path.exists():
            continue
        for row in iter_jsonl(path):
            for key in ("CommentID", "RawCommentID"):
                value = row.get(key)
                if value:
                    excluded_names.update(canonical_name_variants(value))
                    excluded_video_ids.update(extract_video_ids_from_name(value))
            mapping = row.get("Mapping", {})
            if isinstance(mapping, dict):
                for mk in ("original_filename", "new_filename"):
                    value = mapping.get(mk)
                    if value:
                        excluded_names.update(canonical_name_variants(value))
                        excluded_video_ids.update(extract_video_ids_from_name(value))
    return excluded_names, excluded_video_ids


def file_video_id(path: Path) -> str:
    stem = path.name
    if stem.casefold().endswith(".txt"):
        stem = stem[:-4]
    if stem.endswith(".en"):
        stem = stem[:-3]
    return stem


def should_exclude(path: Path, excluded_names: Set[str], excluded_video_ids: Set[str]) -> bool:
    name_keys = canonical_name_variants(path.name)
    if name_keys & excluded_names:
        return True
    vid = file_video_id(path)
    if vid.casefold() in excluded_video_ids:
        return True
    return False


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_root.exists():
        raise FileNotFoundError(f"source root not found: {source_root}")
    if not (0.0 < args.val_ratio < 0.5):
        raise ValueError("--val-ratio must be in (0, 0.5)")

    excluded_names: Set[str] = set()
    excluded_video_ids: Set[str] = set()
    if args.exclude_supervised_devtest:
        dev_path = Path(args.dev_docs_jsonl) if args.dev_docs_jsonl else None
        test_path = Path(args.test_docs_jsonl) if args.test_docs_jsonl else None
        excluded_names, excluded_video_ids = build_exclusion_sets([dev_path, test_path])
        excluded_video_ids = {x.casefold() for x in excluded_video_ids}
        print(
            f"[Exclusion] enabled: names={len(excluded_names)} video_ids={len(excluded_video_ids)} "
            f"(from dev/test docs)"
        )

    all_files = sorted(source_root.rglob("*.txt"))
    kept_docs: List[Tuple[Path, str]] = []
    seen_texts: Set[str] = set()
    skipped_empty = 0
    skipped_short = 0
    skipped_excluded = 0
    skipped_dedupe = 0

    for path in all_files:
        if args.exclude_supervised_devtest and should_exclude(path, excluded_names, excluded_video_ids):
            skipped_excluded += 1
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        text = normalize_text(raw)
        if not text:
            skipped_empty += 1
            continue
        if len(text) < args.min_chars:
            skipped_short += 1
            continue
        if args.dedupe_text:
            key = text.casefold()
            if key in seen_texts:
                skipped_dedupe += 1
                continue
            seen_texts.add(key)
        kept_docs.append((path, text))

    if not kept_docs:
        raise RuntimeError("No documents kept for MLM corpus.")

    rng = random.Random(args.seed)
    rng.shuffle(kept_docs)

    n_val = max(1, int(round(len(kept_docs) * args.val_ratio)))
    n_val = min(n_val, max(1, len(kept_docs) - 1))
    val_docs = kept_docs[:n_val]
    train_docs = kept_docs[n_val:]

    train_path = output_dir / "mlm_train.txt"
    val_path = output_dir / "mlm_val.txt"
    train_manifest = output_dir / "mlm_train_files.jsonl"
    val_manifest = output_dir / "mlm_val_files.jsonl"

    with train_path.open("w", encoding="utf-8") as f_txt, train_manifest.open("w", encoding="utf-8") as f_meta:
        for p, t in train_docs:
            f_txt.write(t + "\n")
            f_meta.write(json.dumps({"path": str(p), "video_id": file_video_id(p)}, ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as f_txt, val_manifest.open("w", encoding="utf-8") as f_meta:
        for p, t in val_docs:
            f_txt.write(t + "\n")
            f_meta.write(json.dumps({"path": str(p), "video_id": file_video_id(p)}, ensure_ascii=False) + "\n")

    manifest: Dict[str, object] = {
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "total_txt_files": len(all_files),
        "kept_docs": len(kept_docs),
        "train_docs": len(train_docs),
        "val_docs": len(val_docs),
        "min_chars": args.min_chars,
        "dedupe_text": bool(args.dedupe_text),
        "exclude_supervised_devtest": bool(args.exclude_supervised_devtest),
        "skipped": {
            "excluded_devtest": skipped_excluded,
            "empty": skipped_empty,
            "too_short": skipped_short,
            "deduped": skipped_dedupe,
        },
        "files": {
            "train_text": str(train_path),
            "val_text": str(val_path),
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
        },
    }
    with (output_dir / "mlm_corpus_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(
        "[MLM Corpus] "
        f"total_files={len(all_files)} kept={len(kept_docs)} "
        f"train={len(train_docs)} val={len(val_docs)} "
        f"excluded={skipped_excluded} short={skipped_short} deduped={skipped_dedupe}"
    )
    print(f"Wrote: {train_path} and {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
