#!/usr/bin/env python3
import argparse
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Iterator, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic comment-domain MLM stage copied from the old YouTube DAPT procedure."
    )
    parser.add_argument(
        "--root-dir",
        default="ADD_PATH",
        help="Root directory containing comment JSONL files.",
    )
    parser.add_argument("--model-name", required=True, help="Base encoder or checkpoint to adapt.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=2500)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--include-context", dest="include_context", action="store_true")
    parser.add_argument("--no-include-context", dest="include_context", action="store_false")
    parser.set_defaults(include_context=True)
    return parser.parse_args()


def find_jsonl_files(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    jsonl_files = sorted(root.rglob("*.jsonl"))
    LOGGER.info("Found %s JSONL files under %s", len(jsonl_files), root)
    return jsonl_files


def read_comments(jsonl_files: List[Path], include_context: bool = True) -> Iterator[str]:
    for file_path in jsonl_files:
        LOGGER.info("Processing %s", file_path)
        try:
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    comment_text = data.get("CommentText", "").strip()
                    if not comment_text:
                        continue
                    if include_context and str(data.get("IsReply", "")).strip() == "True":
                        context = data.get("ConText", "").strip()
                        if context:
                            yield f"{context} [SEP] {comment_text}"
                            continue
                    yield comment_text
        except Exception as exc:
            LOGGER.warning("Skipping %s due to read error: %s", file_path, exc)


def create_dataset(root_dir: str, include_context: bool, max_samples: int) -> Dataset:
    jsonl_files = find_jsonl_files(root_dir)
    rows = []
    for idx, comment in enumerate(read_comments(jsonl_files, include_context=include_context)):
        rows.append({"text": comment})
        if max_samples > 0 and idx + 1 >= max_samples:
            break
        if (idx + 1) % 10000 == 0:
            LOGGER.info("Loaded %s comments", idx + 1)
    if not rows:
        raise RuntimeError(f"No comments loaded from {root_dir}")
    LOGGER.info("Total comments loaded: %s", len(rows))
    return Dataset.from_list(rows)


def tokenize_function(examples, tokenizer, max_length: int = 512):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_special_tokens_mask=True,
    )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    torch.set_float32_matmul_precision("high")

    LOGGER.info("Loading tokenizer/model from %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    if args.gradient_checkpointing:
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        enable_sig = inspect.signature(model.gradient_checkpointing_enable)
        if "gradient_checkpointing_kwargs" in enable_sig.parameters:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            LOGGER.info("Gradient checkpointing enabled with use_reentrant=False")
        else:
            LOGGER.info(
                "Gradient checkpointing requested, but model API does not expose "
                "gradient_checkpointing_kwargs; relying on Trainer settings."
            )

    dataset = create_dataset(
        root_dir=args.root_dir,
        include_context=args.include_context,
        max_samples=args.max_samples,
    )
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
        num_proc=max(1, args.num_workers),
    )
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )
    training_kwargs = {
        "output_dir": str(output_dir / "checkpoints"),
        "overwrite_output_dir": True,
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "eval_strategy": "steps",
        "eval_steps": args.save_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": 0.01,
        "fp16": args.fp16,
        "dataloader_num_workers": args.num_workers,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "push_to_hub": False,
        "gradient_checkpointing": args.gradient_checkpointing,
        "optim": "adamw_torch",
        "dataloader_pin_memory": False,
    }
    if "gradient_checkpointing_kwargs" in inspect.signature(TrainingArguments.__init__).parameters:
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    report = {
        "model_name": args.model_name,
        "root_dir": args.root_dir,
        "output_dir": str(output_dir),
        "final_model_dir": str(final_dir),
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "eval_metrics": eval_metrics,
        "args": vars(args),
    }
    with (output_dir / "training_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    LOGGER.info("Comment MLM complete: %s", final_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
