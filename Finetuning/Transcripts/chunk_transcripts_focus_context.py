#!/usr/bin/env python3
import argparse
import bisect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from transformers import AutoTokenizer


DEFAULT_TOKENIZER = (
    "ADD_PATH"
)

PREV_MARKER = "<<PREV_CTX>>\n"
FOCUS_MARKER = "\n<<FOCUS>>\n"
NEXT_MARKER = "\n<<NEXT_CTX>>\n"


@dataclass
class TargetSegment:
    start_char: int
    end_char: int
    start_tok: int
    end_tok: int

    @property
    def token_len(self) -> int:
        return self.end_tok - self.start_tok + 1


def read_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sentence_end_positions(text: str) -> List[int]:
    boundaries = set()
    for m in re.finditer(r"[.!?]+(?:\s+|$)", text):
        boundaries.add(m.end())
    for m in re.finditer(r"\n+", text):
        boundaries.add(m.end())
    return sorted(boundaries)


def get_token_offsets(tokenizer, text: str) -> Tuple[List[int], List[int]]:
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = enc["offset_mapping"]
    token_starts = [int(s) for s, _ in offsets]
    token_ends = [int(e) for _, e in offsets]
    return token_starts, token_ends


def build_safe_boundaries(
    text: str,
    annotations: List[Dict],
    token_starts: List[int],
    token_ends: List[int],
    use_sentence_boundaries: bool,
    use_annotation_boundaries: bool,
) -> List[int]:
    token_bounds = {0, len(text)}
    token_bounds.update(token_starts)
    token_bounds.update(token_ends)

    safe = {0, len(text)}
    if use_sentence_boundaries:
        for pos in sentence_end_positions(text):
            if pos in token_bounds:
                safe.add(pos)
    if use_annotation_boundaries:
        for ann in annotations:
            s = int(ann["start"])
            e = int(ann["end"])
            if s in token_bounds:
                safe.add(s)
            if e in token_bounds:
                safe.add(e)
    return sorted(safe)


def first_ge(sorted_values: List[int], value: int) -> int:
    idx = bisect.bisect_left(sorted_values, value)
    if idx >= len(sorted_values):
        return sorted_values[-1]
    return sorted_values[idx]


def choose_target_end(
    cur_char: int,
    start_tok: int,
    token_ends: List[int],
    target_max_tokens: int,
    safe_boundaries: List[int],
    sentence_boundaries: List[int],
    annotations: List[Dict],
    enforce_sentence_completion: bool,
    preserve_annotation_integrity: bool,
) -> int:
    end_tok_excl = min(start_tok + target_max_tokens, len(token_ends))
    base_boundary = first_ge(safe_boundaries, token_ends[end_tok_excl - 1])
    boundary = base_boundary

    # Keep any annotation that already crosses the initial target boundary.
    # This avoids transitive expansion through later annotations.
    if preserve_annotation_integrity:
        desired_end = boundary
        for ann in annotations:
            ann_start = int(ann["start"])
            ann_end = int(ann["end"])
            if ann_start < cur_char:
                continue
            if ann_start < base_boundary < ann_end:
                if ann_end > desired_end:
                    desired_end = ann_end
        boundary = first_ge(safe_boundaries, desired_end)

    # Also keep the full sentence containing the resulting boundary.
    if enforce_sentence_completion and sentence_boundaries and boundary not in sentence_boundaries:
        sentence_end = first_ge(sentence_boundaries, boundary)
        if sentence_end >= boundary:
            boundary = first_ge(safe_boundaries, sentence_end)

    return boundary


def build_target_segments(
    text: str,
    annotations: List[Dict],
    token_starts: List[int],
    token_ends: List[int],
    safe_boundaries: List[int],
    sentence_boundaries: List[int],
    target_max_tokens: int,
    target_min_tokens: int,
    enforce_sentence_completion: bool,
    preserve_annotation_integrity: bool,
) -> List[TargetSegment]:
    segments: List[TargetSegment] = []
    cur_char = 0
    while cur_char < len(text):
        start_tok = bisect.bisect_left(token_starts, cur_char)
        if start_tok >= len(token_starts):
            break
        end_char = choose_target_end(
            cur_char,
            start_tok,
            token_ends,
            target_max_tokens=target_max_tokens,
            safe_boundaries=safe_boundaries,
            sentence_boundaries=sentence_boundaries,
            annotations=annotations,
            enforce_sentence_completion=enforce_sentence_completion,
            preserve_annotation_integrity=preserve_annotation_integrity,
        )
        if end_char <= cur_char:
            end_char = min(len(text), cur_char + 1)
        end_tok = bisect.bisect_right(token_ends, end_char) - 1
        if end_tok < start_tok:
            end_tok = start_tok
        segments.append(TargetSegment(cur_char, end_char, start_tok, end_tok))
        if end_char >= len(text):
            break
        cur_char = end_char

    # If the tail segment is too short, slide it backward to hit target_min_tokens when possible.
    # This may introduce overlap with the previous target segment, but keeps target size stable.
    if len(segments) >= 1 and segments[-1].token_len < target_min_tokens:
        tail = segments[-1]
        if tail.end_tok >= target_min_tokens - 1:
            new_start_tok = tail.end_tok - target_min_tokens + 1
            new_start_char = token_starts[new_start_tok]
            segments[-1] = TargetSegment(
                start_char=new_start_char,
                end_char=tail.end_char,
                start_tok=new_start_tok,
                end_tok=tail.end_tok,
            )

    return segments


def remap_target_annotations_with_markers(
    annotations: List[Dict],
    target_start: int,
    target_end: int,
    chunk_text: str,
    target_offset: int,
) -> List[Dict]:
    out = []
    for ann in annotations:
        ann_start = int(ann["start"])
        ann_end = int(ann["end"])
        if ann_end <= target_start or ann_start >= target_end:
            continue
        # We keep supervision on the target area only.
        local_start = target_offset + (max(ann_start, target_start) - target_start)
        local_end = target_offset + (min(ann_end, target_end) - target_start)
        if local_end <= local_start:
            continue
        out.append(
            {
                "technique": ann["technique"],
                "start": int(local_start),
                "end": int(local_end),
                "text_span": chunk_text[local_start:local_end],
            }
        )
    return out


def chunk_record_focus_context(
    record: Dict,
    tokenizer,
    target_max_tokens: int,
    target_min_tokens: int,
    context_ratio: float,
    chunk_max_tokens: int,
    add_focus_markers: bool,
    use_sentence_boundaries: bool,
    use_annotation_boundaries: bool,
    preserve_annotation_integrity: bool,
) -> List[Dict]:
    text = record.get("CommentText", "")
    annotations = sorted(record.get("annotations", []), key=lambda x: (int(x["start"]), int(x["end"])))
    if not text:
        return []

    token_starts, token_ends = get_token_offsets(tokenizer, text)
    if not token_starts:
        return []
    sentence_boundaries = sentence_end_positions(text) if use_sentence_boundaries else []

    safe_boundaries = build_safe_boundaries(
        text=text,
        annotations=annotations,
        token_starts=token_starts,
        token_ends=token_ends,
        use_sentence_boundaries=use_sentence_boundaries,
        use_annotation_boundaries=use_annotation_boundaries,
    )

    target_segments = build_target_segments(
        text=text,
        annotations=annotations,
        token_starts=token_starts,
        token_ends=token_ends,
        safe_boundaries=safe_boundaries,
        sentence_boundaries=sentence_boundaries,
        target_max_tokens=target_max_tokens,
        target_min_tokens=target_min_tokens,
        enforce_sentence_completion=use_sentence_boundaries,
        preserve_annotation_integrity=preserve_annotation_integrity,
    )

    chunks: List[Dict] = []
    base_side_ctx_tokens = max(0, int(round(target_max_tokens * context_ratio)))
    for chunk_idx, seg in enumerate(target_segments):
        target_tokens = seg.token_len
        side_ctx_tokens = base_side_ctx_tokens

        ctx_start_tok = max(0, seg.start_tok - side_ctx_tokens)
        ctx_end_tok = min(len(token_starts) - 1, seg.end_tok + side_ctx_tokens)

        def build_chunk_text(_ctx_start_tok: int, _ctx_end_tok: int):
            _ctx_start_char = token_starts[_ctx_start_tok]
            _ctx_end_char = token_ends[_ctx_end_tok]
            _prev_ctx = text[_ctx_start_char : seg.start_char]
            _target_txt = text[seg.start_char : seg.end_char]
            _next_ctx = text[seg.end_char : _ctx_end_char]
            if add_focus_markers:
                _chunk_text = PREV_MARKER + _prev_ctx + FOCUS_MARKER + _target_txt + NEXT_MARKER + _next_ctx
                _target_offset = len(PREV_MARKER) + len(_prev_ctx) + len(FOCUS_MARKER)
            else:
                _chunk_text = _prev_ctx + _target_txt + _next_ctx
                _target_offset = len(_prev_ctx)
            return _chunk_text, _target_offset, _ctx_start_char, _ctx_end_char

        chunk_text, target_offset, ctx_start_char, ctx_end_char = build_chunk_text(ctx_start_tok, ctx_end_tok)

        # Keep target intact; shrink context if chunk exceeds model token budget.
        if chunk_max_tokens > 0:
            for _ in range(2000):
                chunk_tokens = len(
                    tokenizer(
                        chunk_text,
                        add_special_tokens=False,
                        truncation=False,
                    )["input_ids"]
                )
                if chunk_tokens <= chunk_max_tokens:
                    break
                left_ctx = seg.start_tok - ctx_start_tok
                right_ctx = ctx_end_tok - seg.end_tok
                if left_ctx <= 0 and right_ctx <= 0:
                    break
                if right_ctx >= left_ctx and right_ctx > 0:
                    ctx_end_tok -= 1
                elif left_ctx > 0:
                    ctx_start_tok += 1
                elif right_ctx > 0:
                    ctx_end_tok -= 1
                chunk_text, target_offset, ctx_start_char, ctx_end_char = build_chunk_text(ctx_start_tok, ctx_end_tok)

        chunk_annotations = remap_target_annotations_with_markers(
            annotations=annotations,
            target_start=seg.start_char,
            target_end=seg.end_char,
            chunk_text=chunk_text,
            target_offset=target_offset,
        )

        chunk = {
            "CommentID": f"{record['CommentID']}__chunk{chunk_idx:04d}",
            "ParentCommentID": record["CommentID"],
            "RawCommentID": record.get("RawCommentID", record["CommentID"]),
            "Source": record.get("Source", "transcripts"),
            "CommentText": chunk_text,
            "annotations": chunk_annotations,
            "ChunkMeta": {
                "chunk_index": chunk_idx,
                "chunk_mode": "focus_context",
                "target_start_char": seg.start_char,
                "target_end_char": seg.end_char,
                "target_start_token": seg.start_tok,
                "target_end_token": seg.end_tok,
                "target_token_len": target_tokens,
                "context_start_char": ctx_start_char,
                "context_end_char": ctx_end_char,
                "left_context_tokens": seg.start_tok - ctx_start_tok,
                "right_context_tokens": ctx_end_tok - seg.end_tok,
                "context_ratio": context_ratio,
                "target_min_tokens": target_min_tokens,
                "target_max_tokens": target_max_tokens,
                "focus_markers": bool(add_focus_markers),
                "target_offset_in_chunk_chars": target_offset,
                "chunk_max_tokens": chunk_max_tokens,
            },
        }
        if "Mapping" in record:
            chunk["Mapping"] = dict(record["Mapping"])
        chunks.append(chunk)
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Chunk transcripts into target-focused windows with left/right context. "
            "Supervision is applied to target segment annotations."
        )
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--target-max-tokens", type=int, default=256)
    parser.add_argument("--target-min-tokens", type=int, default=256)
    parser.add_argument("--context-ratio", type=float, default=0.2)
    parser.add_argument(
        "--chunk-max-tokens",
        type=int,
        default=0,
        help="Optional hard cap for final chunk token length; context is reduced first.",
    )
    parser.add_argument("--add-focus-markers", action="store_true")
    parser.add_argument("--no-sentence-boundaries", action="store_true")
    parser.add_argument("--no-annotation-boundaries", action="store_true")
    parser.add_argument("--allow-splitting-annotations", action="store_true")
    args = parser.parse_args()

    if args.target_max_tokens < 2:
        raise ValueError("--target-max-tokens must be >= 2")
    if args.target_min_tokens < 1:
        raise ValueError("--target-min-tokens must be >= 1")
    if args.target_min_tokens > args.target_max_tokens:
        raise ValueError("--target-min-tokens cannot exceed --target-max-tokens")
    if args.context_ratio < 0.0 or args.context_ratio > 1.0:
        raise ValueError("--context-ratio must be in [0, 1]")
    if args.chunk_max_tokens < 0:
        raise ValueError("--chunk-max-tokens must be >= 0")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Fast tokenizer is required for offset mappings.")

    rows = read_jsonl(Path(args.input_jsonl))
    out: List[Dict] = []
    for row in rows:
        out.extend(
            chunk_record_focus_context(
                row,
                tokenizer=tokenizer,
                target_max_tokens=args.target_max_tokens,
                target_min_tokens=args.target_min_tokens,
                context_ratio=args.context_ratio,
                chunk_max_tokens=args.chunk_max_tokens,
                add_focus_markers=args.add_focus_markers,
                use_sentence_boundaries=not args.no_sentence_boundaries,
                use_annotation_boundaries=not args.no_annotation_boundaries,
                preserve_annotation_integrity=not args.allow_splitting_annotations,
            )
        )

    write_jsonl(Path(args.output_jsonl), out)
    docs = len(rows)
    avg_chunks = (len(out) / docs) if docs else 0.0
    print(f"Input docs: {docs}")
    print(f"Output chunks: {len(out)}")
    print(f"Avg chunks/doc: {avg_chunks:.2f}")
    print(
        "Config: "
        f"target_min={args.target_min_tokens}, "
        f"target_max={args.target_max_tokens}, "
        f"context_ratio={args.context_ratio}, "
        f"chunk_max_tokens={args.chunk_max_tokens}, "
        f"focus_markers={bool(args.add_focus_markers)}"
    )
    print(f"Wrote: {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
