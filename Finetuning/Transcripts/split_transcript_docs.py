#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_INPUT = Path("ADD_PATH")
DEFAULT_OUTPUT_DIR = Path("ADD_PATH")


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def has_prefix(record: Dict, prefix: str) -> bool:
    raw_id = (record.get("RawCommentID") or "").strip()
    return raw_id.startswith(prefix)


def split_records(
    records: List[Dict],
    seed: int,
    dev_ratio: float,
    test_ratio: float,
    test_prefix: str,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    rng = random.Random(seed)

    prefix_test = [r for r in records if has_prefix(r, test_prefix)] if test_prefix else []
    test_ids = {r["CommentID"] for r in prefix_test}
    remainder = [r for r in records if r["CommentID"] not in test_ids]

    if prefix_test:
        rng.shuffle(remainder)
        dev_size = max(1, int(round(len(remainder) * dev_ratio)))
        dev_size = min(dev_size, max(1, len(remainder) - 1))
        dev = remainder[:dev_size]
        train = remainder[dev_size:]
        test = prefix_test
    else:
        shuffled = list(records)
        rng.shuffle(shuffled)
        n = len(shuffled)
        test_size = max(1, int(round(n * test_ratio)))
        dev_size = max(1, int(round(n * dev_ratio)))
        if test_size + dev_size >= n:
            test_size = max(1, n // 5)
            dev_size = max(1, n // 5)
        test = shuffled[:test_size]
        dev = shuffled[test_size : test_size + dev_size]
        train = shuffled[test_size + dev_size :]

    return train, dev, test


def assert_disjoint(train: List[Dict], dev: List[Dict], test: List[Dict]) -> None:
    def ids(rows: List[Dict]) -> set:
        return {row["CommentID"] for row in rows}

    train_ids, dev_ids, test_ids = ids(train), ids(dev), ids(test)
    assert train_ids.isdisjoint(dev_ids), "train/dev overlap"
    assert train_ids.isdisjoint(test_ids), "train/test overlap"
    assert dev_ids.isdisjoint(test_ids), "dev/test overlap"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create transcript document-level train/dev/test splits.")
    parser.add_argument("--input-jsonl", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--test-prefix",
        default="sample_",
        help="If non-empty and found in RawCommentID, those docs are used as test.",
    )
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input_jsonl))
    train, dev, test = split_records(
        rows,
        seed=args.seed,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        test_prefix=args.test_prefix,
    )
    assert_disjoint(train, dev, test)

    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "train_docs.jsonl", train)
    write_jsonl(out_dir / "dev_docs.jsonl", dev)
    write_jsonl(out_dir / "test_docs.jsonl", test)

    print(f"Wrote train/dev/test docs to {out_dir}")
    print(f"  train: {len(train)}")
    print(f"  dev:   {len(dev)}")
    print(f"  test:  {len(test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
