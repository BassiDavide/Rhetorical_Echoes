#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"

COMMENTS_MLM_ROOT = Path("ADD_PATH")
COMMENTS_ROOT = Path("ADD_PATH")
TRANS_INPUT_ROOT = Path("ADD_PATH")
TRANS_MLM_SOURCE_ROOT = Path(
    "ADD_PATH"
)
COMMENTS_MIXED_TRAIN = COMMENTS_ROOT / "Def_Anon_Splits" / "train.jsonl"

MODEL_ALIASES = {
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "deberta-v3-large": "microsoft/deberta-v3-large",
    "roberta-base": "FacebookAI/roberta-base",
    "roberta-large": "FacebookAI/roberta-large",
    "modernbert-base": "answerdotai/ModernBERT-base",
    "modernbert-large": "answerdotai/ModernBERT-large",
}
DEFAULT_SEEDS = [42, 43, 44]
COMMENT_PIPELINE_VERSION = "nature_legacy_v1"
TRANSCRIPT_MLM_PIPELINE_VERSION = "shared_transcript_mlm_v3"
TRANSCRIPT_SPAN_PIPELINE_VERSION = "shared_transcript_span_v3"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full comments+transcripts pipeline for one or more models.")
    parser.add_argument("--models", nargs="+", required=True, help="Model aliases or raw Hugging Face model ids.")
    parser.add_argument("--output-root", default=str(ROOT_DIR / "outputs"))
    parser.add_argument("--shared-data-dir", default=str(ROOT_DIR / "data"))
    parser.add_argument(
        "--allow-long-context",
        action="store_true",
        help="If set, let models with larger context windows use larger transcript-side lengths.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def slugify(name: str) -> str:
    cleaned = name.lower().replace("/", "__")
    cleaned = re.sub(r"[^a-z0-9._-]+", "-", cleaned)
    return cleaned.strip("-")


def resolve_model(name: str) -> Tuple[str, str]:
    hf_name = MODEL_ALIASES.get(name, name)
    slug = slugify(name if name in MODEL_ALIASES else hf_name)
    return hf_name, slug


def resolve_comment_splits() -> Dict[str, Path]:
    candidate_dirs = [
        COMMENTS_ROOT / "Def_splits",
        COMMENTS_ROOT / "Def_Anon_Splits",
    ]
    for split_dir in candidate_dirs:
        train_path = split_dir / "train.jsonl"
        dev_path = split_dir / "dev.jsonl"
        test_path = split_dir / "test.jsonl"
        if train_path.exists() and dev_path.exists() and test_path.exists():
            return {
                "train": train_path,
                "dev": dev_path,
                "test": test_path,
            }
    raise FileNotFoundError(
        "Could not find comment splits. Checked: "
        + ", ".join(str(path) for path in candidate_dirs)
    )


def supports_long_context(model_slug: str, model_name: str) -> bool:
    text = f"{model_slug} {model_name}".casefold()
    return "modernbert" in text


def build_model_plan(*, model_slug: str, model_name: str, allow_long_context: bool) -> Dict[str, object]:
    plan: Dict[str, object] = {
        "comments_mlm_max_length": 512,
        "comments_span_max_length": 320,
        "transcript_mlm_max_length": 256,
        "transcript_mlm_batch_size": 8,
        "transcript_mlm_grad_accum": 2,
        "chunk_target_min_tokens": 256,
        "chunk_target_max_tokens": 256,
        "chunk_max_tokens": 512,
        "transcript_span_max_length": 512,
        "transcript_span_batch_size": 4,
        "transcript_span_grad_accum": 2,
        "mixed_steps_per_epoch": 118,
        "long_context_enabled": False,
    }
    if allow_long_context and supports_long_context(model_slug, model_name):
        plan.update(
            {
                "transcript_mlm_max_length": 512,
                "transcript_mlm_batch_size": 4,
                "transcript_mlm_grad_accum": 4,
                "chunk_target_min_tokens": 512,
                "chunk_target_max_tokens": 512,
                "chunk_max_tokens": 1024,
                "transcript_span_max_length": 1024,
                "transcript_span_batch_size": 2,
                "transcript_span_grad_accum": 4,
                "mixed_steps_per_epoch": 0,
                "long_context_enabled": True,
            }
        )
    return plan


def run_cmd(cmd: Sequence[str], *, env: Dict[str, str]) -> None:
    print("\n[Run]", " ".join(cmd))
    subprocess.run(list(cmd), check=True, env=env)


def result_exists(path: Path) -> bool:
    return (path / "test_results.json").exists()


def read_f1(path: Path) -> float:
    with (path / "test_results.json").open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload.get("eval_token_f1_macro", 0.0))


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def comment_run_usable(path: Path) -> bool:
    if not (path / "test_results.json").exists():
        return False
    config_path = path / "training_config.json"
    if not config_path.exists():
        return False
    try:
        payload = read_json(config_path)
    except Exception:
        return False
    return payload.get("comment_pipeline_version") == COMMENT_PIPELINE_VERSION


def transcript_mlm_usable(path: Path, *, expected_init_model_dir: Path) -> bool:
    final_model = path / "final_model"
    report_path = path / "mlm_training_report.json"
    if not final_model.exists() or not report_path.exists():
        return False
    try:
        payload = read_json(report_path)
    except Exception:
        return False
    return (
        payload.get("def_com_trans_pipeline_version") == TRANSCRIPT_MLM_PIPELINE_VERSION
        and payload.get("init_model_dir") == str(expected_init_model_dir)
    )


def transcript_run_usable(path: Path, *, expected_init_model_dir: Path, expected_encoder_init_from_mlm: Path) -> bool:
    if not (path / "test_results.json").exists():
        return False
    config_path = path / "training_config.json"
    if not config_path.exists():
        return False
    try:
        payload = read_json(config_path)
    except Exception:
        return False
    return (
        payload.get("def_com_trans_pipeline_version") == TRANSCRIPT_SPAN_PIPELINE_VERSION
        and payload.get("init_model_dir") == str(expected_init_model_dir)
        and payload.get("encoder_init_from_mlm") == str(expected_encoder_init_from_mlm)
    )


def write_summary(path: Path, stage_name: str, runs: List[Dict[str, object]]) -> None:
    f1_values = [float(run["eval_token_f1_macro"]) for run in runs]
    payload = {
        "stage": stage_name,
        "runs": runs,
        "mean_f1": float(mean(f1_values)) if f1_values else 0.0,
        "std_f1": float(pstdev(f1_values)) if len(f1_values) > 1 else 0.0,
        "n_runs": len(runs),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def ensure_shared_transcript_inputs(shared_data_dir: Path, env: Dict[str, str], skip_existing: bool) -> Tuple[Path, Path, Path]:
    raw_dir = shared_data_dir / "raw"
    split_dir = shared_data_dir / "splits_docs"
    mlm_dir = shared_data_dir / "mlm_corpus"
    raw_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    mlm_dir.mkdir(parents=True, exist_ok=True)

    raw_jsonl = raw_dir / "transcripts_full.jsonl"
    if not skip_existing or not raw_jsonl.exists():
        run_cmd(
            [
                sys.executable,
                str(SCRIPTS_DIR / "prepare_transcripts_jsonl.py"),
                "--input-root",
                str(TRANS_INPUT_ROOT),
                "--output-file",
                str(raw_jsonl),
            ],
            env=env,
        )

    train_docs = split_dir / "train_docs.jsonl"
    dev_docs = split_dir / "dev_docs.jsonl"
    test_docs = split_dir / "test_docs.jsonl"
    if not skip_existing or not (train_docs.exists() and dev_docs.exists() and test_docs.exists()):
        run_cmd(
            [
                sys.executable,
                str(SCRIPTS_DIR / "split_transcript_docs.py"),
                "--input-jsonl",
                str(raw_jsonl),
                "--output-dir",
                str(split_dir),
                "--seed",
                "42",
                "--dev-ratio",
                "0.2",
                "--test-prefix",
                "sample_",
            ],
            env=env,
        )

    mlm_train = mlm_dir / "mlm_train.txt"
    mlm_val = mlm_dir / "mlm_val.txt"
    if not skip_existing or not (mlm_train.exists() and mlm_val.exists()):
        run_cmd(
            [
                sys.executable,
                str(SCRIPTS_DIR / "prepare_transcript_mlm_corpus.py"),
                "--source-root",
                str(TRANS_MLM_SOURCE_ROOT),
                "--output-dir",
                str(mlm_dir),
                "--dev-docs-jsonl",
                str(dev_docs),
                "--test-docs-jsonl",
                str(test_docs),
                "--exclude-supervised-devtest",
                "--val-ratio",
                "0.02",
                "--seed",
                "42",
                "--min-chars",
                "200",
                "--dedupe-text",
            ],
            env=env,
        )

    return split_dir, mlm_dir, raw_jsonl


def ensure_model_chunks(
    *,
    chunk_dir: Path,
    tokenizer_ref: Path,
    split_dir: Path,
    target_min_tokens: int,
    target_max_tokens: int,
    chunk_max_tokens: int,
    env: Dict[str, str],
    skip_existing: bool,
) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "dev", "test"]:
        input_path = split_dir / f"{split_name}_docs.jsonl"
        output_path = chunk_dir / f"{split_name}.jsonl"
        if skip_existing and output_path.exists():
            continue
        run_cmd(
            [
                sys.executable,
                str(SCRIPTS_DIR / "chunk_transcripts_focus_context.py"),
                "--input-jsonl",
                str(input_path),
                "--output-jsonl",
                str(output_path),
                "--tokenizer",
                str(tokenizer_ref),
                "--target-min-tokens",
                str(target_min_tokens),
                "--target-max-tokens",
                str(target_max_tokens),
                "--context-ratio",
                "0.2",
                "--chunk-max-tokens",
                str(chunk_max_tokens),
                "--add-focus-markers",
            ],
            env=env,
        )


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    shared_data_dir = Path(args.shared_data_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    shared_data_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT_DIR}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    split_dir, mlm_corpus_dir, _ = ensure_shared_transcript_inputs(
        shared_data_dir,
        env,
        args.skip_existing,
    )
    comments_splits = resolve_comment_splits()

    pipeline_manifest = {"models": []}

    for requested_name in args.models:
        model_name, model_slug = resolve_model(requested_name)
        model_plan = build_model_plan(
            model_slug=model_slug,
            model_name=model_name,
            allow_long_context=args.allow_long_context,
        )
        model_root = output_root / model_slug
        comments_mlm_dir = model_root / "comments_mlm"
        comments_root = model_root / "comments"
        chunks_dir = model_root / "transcript_chunks" / "focus256_ctx20"
        transcript_mlm_root = model_root / "transcript_mlm"
        transcript_root = model_root / "transcripts"
        comments_root.mkdir(parents=True, exist_ok=True)
        transcript_mlm_root.mkdir(parents=True, exist_ok=True)
        transcript_root.mkdir(parents=True, exist_ok=True)

        comments_mlm_final = comments_mlm_dir / "final_model"
        if not args.skip_existing or not comments_mlm_final.exists():
            run_cmd(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "train_comments_mlm.py"),
                    "--root-dir",
                    str(COMMENTS_MLM_ROOT),
                    "--model-name",
                    model_name,
                    "--output-dir",
                    str(comments_mlm_dir),
                    "--batch-size",
                    "8",
                    "--gradient-accumulation-steps",
                    "16",
                    "--num-epochs",
                    "2",
                    "--learning-rate",
                    "2e-5",
                    "--max-length",
                    str(model_plan["comments_mlm_max_length"]),
                    "--mlm-probability",
                    "0.15",
                    "--warmup-steps",
                    "1000",
                    "--save-steps",
                    "2500",
                    "--logging-steps",
                    "100",
                    "--num-workers",
                    "4",
                    "--seed",
                    "42",
                    "--gradient-checkpointing",
                ],
                env=env,
            )

        comment_runs = []
        for run_idx, seed in enumerate(DEFAULT_SEEDS):
            run_dir = comments_root / f"run_{run_idx}"
            if not (args.skip_existing and comment_run_usable(run_dir)):
                run_cmd(
                    [
                        sys.executable,
                        str(SCRIPTS_DIR / "train_comments_fixed.py"),
                        "--model-name",
                        str(comments_mlm_final),
                        "--train-file",
                        str(comments_splits["train"]),
                        "--dev-file",
                        str(comments_splits["dev"]),
                        "--test-file",
                        str(comments_splits["test"]),
                        "--output-dir",
                        str(run_dir),
                        "--max-length",
                        str(model_plan["comments_span_max_length"]),
                        "--batch-size",
                        "8",
                        "--gradient-accumulation-steps",
                        "1",
                        "--learning-rate",
                        "2.6049078843086554e-05",
                        "--num-epochs",
                        "28",
                        "--threshold",
                        "0.35",
                        "--dropout-prob",
                        "0.2",
                        "--weight-decay",
                        "0.03",
                        "--warmup-ratio",
                        "0.1",
                        "--freeze-layers",
                        "6",
                        "--unfreeze-epoch",
                        "2",
                        "--seed",
                        str(seed),
                    ],
                    env=env,
                )
            comment_runs.append(
                {
                    "run": run_idx,
                    "seed": seed,
                    "model_dir": str(run_dir),
                    "eval_token_f1_macro": read_f1(run_dir),
                }
            )
        write_summary(model_root / "comments_summary.json", "comments", comment_runs)

        ensure_model_chunks(
            chunk_dir=chunks_dir,
            tokenizer_ref=comments_root / "run_0",
            split_dir=split_dir,
            target_min_tokens=int(model_plan["chunk_target_min_tokens"]),
            target_max_tokens=int(model_plan["chunk_target_max_tokens"]),
            chunk_max_tokens=int(model_plan["chunk_max_tokens"]),
            env=env,
            skip_existing=args.skip_existing,
        )

        transcript_init_comment_dir = comments_root / "run_0"
        transcript_mlm_dir = transcript_mlm_root / "shared"
        transcript_mlm_final = transcript_mlm_dir / "final_model"
        if not (args.skip_existing and transcript_mlm_usable(transcript_mlm_dir, expected_init_model_dir=transcript_init_comment_dir)):
            run_cmd(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "train_transcript_mlm_dapt.py"),
                    "--train-text-file",
                    str(mlm_corpus_dir / "mlm_train.txt"),
                    "--val-text-file",
                    str(mlm_corpus_dir / "mlm_val.txt"),
                    "--init-model-dir",
                    str(transcript_init_comment_dir),
                    "--output-dir",
                    str(transcript_mlm_dir),
                    "--max-length",
                    str(model_plan["transcript_mlm_max_length"]),
                    "--batch-size",
                    str(model_plan["transcript_mlm_batch_size"]),
                    "--gradient-accumulation-steps",
                    str(model_plan["transcript_mlm_grad_accum"]),
                    "--learning-rate",
                    "1e-5",
                    "--num-epochs",
                    "3",
                    "--max-steps",
                    "40000",
                    "--seed",
                    "42",
                ],
                env=env,
            )

        transcript_runs = []
        for run_idx, seed in enumerate(DEFAULT_SEEDS):
            transcript_run_dir = transcript_root / f"run_{run_idx}"

            if not (
                args.skip_existing
                and transcript_run_usable(
                    transcript_run_dir,
                    expected_init_model_dir=transcript_init_comment_dir,
                    expected_encoder_init_from_mlm=transcript_mlm_final,
                )
            ):
                run_cmd(
                    [
                        sys.executable,
                        str(SCRIPTS_DIR / "train_span_no_overlap.py"),
                        "--train-file",
                        str(chunks_dir / "train.jsonl"),
                        "--dev-file",
                        str(chunks_dir / "dev.jsonl"),
                        "--test-file",
                        str(chunks_dir / "test.jsonl"),
                        "--comments-train-file",
                        str(COMMENTS_MIXED_TRAIN),
                        "--train-mode",
                        "mixed",
                        "--mix-comment-fraction",
                        "0.5",
                        "--mixed-steps-per-epoch",
                        str(model_plan["mixed_steps_per_epoch"]),
                        "--init-model-dir",
                        str(transcript_init_comment_dir),
                        "--encoder-init-from-mlm",
                        str(transcript_mlm_final),
                        "--output-dir",
                        str(transcript_run_dir),
                        "--max-length",
                        str(model_plan["transcript_span_max_length"]),
                        "--batch-size",
                        str(model_plan["transcript_span_batch_size"]),
                        "--gradient-accumulation-steps",
                        str(model_plan["transcript_span_grad_accum"]),
                        "--weight-decay",
                        "0.03",
                        "--dropout-prob",
                        "0.2",
                        "--use-curriculum",
                        "--stage-a-epochs",
                        "3",
                        "--stage-b-epochs",
                        "16",
                        "--stage-a-learning-rate",
                        "2e-5",
                        "--stage-b-learning-rate",
                        "2e-6",
                        "--ignore-labels",
                        "intentional vagueness,reductio ad hitlerum",
                        "--threshold-recall-priority-labels",
                        "smears/doubt,loaded language,distraction",
                        "--threshold-recall-beta",
                        "2.0",
                        "--freeze-layers",
                        "6",
                        "--seed",
                        str(seed),
                    ],
                    env=env,
                )
            transcript_runs.append(
                {
                    "run": run_idx,
                    "seed": seed,
                    "model_dir": str(transcript_run_dir),
                    "eval_token_f1_macro": read_f1(transcript_run_dir),
                }
            )
        write_summary(model_root / "transcripts_summary.json", "transcripts", transcript_runs)

        pipeline_manifest["models"].append(
                {
                    "requested_name": requested_name,
                    "resolved_model_name": model_name,
                    "slug": model_slug,
                    "output_dir": str(model_root),
                    "plan": model_plan,
                    "comments_splits": {k: str(v) for k, v in comments_splits.items()},
                }
            )

    with (output_root / "pipeline_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(pipeline_manifest, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
