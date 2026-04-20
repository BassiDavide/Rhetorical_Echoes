#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    set_seed,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

TRANSCRIPT_MLM_PIPELINE_VERSION = "shared_transcript_mlm_v3"

from span_singlehead.model import TokenDeberta

try:
    import inspect
    from accelerate import Accelerator

    if "keep_torch_compile" not in inspect.signature(Accelerator.unwrap_model).parameters:
        _orig_unwrap_model = Accelerator.unwrap_model

        def _unwrap_model_compat(self, model, keep_fp32_wrapper: bool = True, keep_torch_compile: bool = True):
            return _orig_unwrap_model(self, model, keep_fp32_wrapper=keep_fp32_wrapper)

        Accelerator.unwrap_model = _unwrap_model_compat
except Exception:
    pass


class CompatAdamW(AdamW):
    def train(self):
        return self

    def eval(self):
        return self


class PackedMLMDataset(Dataset):
    def __init__(
        self,
        text_file: str,
        tokenizer,
        max_length: int,
        min_doc_tokens: int = 16,
        min_tail_fraction: float = 0.25,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_doc_tokens = min_doc_tokens
        self.min_tail_fraction = min_tail_fraction
        self.examples = self._build_examples(text_file)

    def _read_lines(self, text_file: str) -> List[str]:
        out: List[str] = []
        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(line)
        return out

    def _build_examples(self, text_file: str):
        lines = self._read_lines(text_file)
        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        block_size = self.max_length - special_tokens
        if block_size < 8:
            raise ValueError(f"max_length={self.max_length} too small for tokenizer special tokens")

        sep_id = self.tokenizer.sep_token_id
        if sep_id is None:
            sep_id = self.tokenizer.eos_token_id

        stream: List[int] = []
        docs_kept = 0
        for text in lines:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < self.min_doc_tokens:
                continue
            stream.extend(ids)
            if sep_id is not None:
                stream.append(sep_id)
            docs_kept += 1

        if not stream:
            raise RuntimeError(f"No tokenized content from: {text_file}")

        token_chunks: List[List[int]] = []
        idx = 0
        while idx + block_size <= len(stream):
            token_chunks.append(stream[idx : idx + block_size])
            idx += block_size
        tail = stream[idx:]
        min_tail = max(8, int(round(block_size * self.min_tail_fraction)))
        if len(tail) >= min_tail:
            token_chunks.append(tail)

        examples = []
        for chunk in token_chunks:
            encoded = self.tokenizer.prepare_for_model(
                chunk,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_attention_mask=True,
            )
            examples.append(
                {
                    "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
                }
            )

        if not examples:
            raise RuntimeError(f"No MLM examples created from: {text_file}")
        print(
            f"[MLM Dataset] file={text_file} docs={docs_kept} chunks={len(examples)} "
            f"max_length={self.max_length}"
        )
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Domain-adaptive MLM training on transcripts starting from token-model encoder weights."
    )
    parser.add_argument("--train-text-file", required=True)
    parser.add_argument("--val-text-file", required=True)
    parser.add_argument("--init-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def infer_num_token_labels(model_dir: Path) -> int:
    label_path = model_dir / "label_list.json"
    if label_path.exists():
        with label_path.open("r", encoding="utf-8") as f:
            return len(json.load(f))
    return 20


def _looks_like_custom_token_checkpoint(model_ref: Path) -> bool:
    label_path = model_ref / "label_list.json"
    if label_path.exists():
        return True
    config_path = model_ref / "config.json"
    if not config_path.exists():
        return False
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    archs = payload.get("architectures", [])
    return any(name in {"TokenDeberta", "MultiHeadTokenDeberta", "BinaryMultiHeadTokenDeberta"} for name in archs)


def _extract_encoder_module(model_obj, reference_state_keys=None):
    candidates: List[Tuple[str, object]] = [("model", model_obj)]
    base_prefix = getattr(model_obj, "base_model_prefix", "")
    if base_prefix and hasattr(model_obj, base_prefix):
        candidates.append((f"model.{base_prefix}", getattr(model_obj, base_prefix)))
    for attr in ["encoder", "deberta", "roberta", "bert", "model"]:
        if hasattr(model_obj, attr):
            candidates.append((f"model.{attr}", getattr(model_obj, attr)))

    best_name = None
    best_module = None
    best_score = None
    for name, cand in candidates:
        if not hasattr(cand, "state_dict"):
            continue
        keys = set(cand.state_dict().keys())
        if not keys:
            continue
        if reference_state_keys is not None:
            missing = len(reference_state_keys - keys)
            unexpected = len(keys - reference_state_keys)
            overlap = len(reference_state_keys & keys)
            score = (-(missing + unexpected), overlap, -len(keys))
        else:
            score = (len(keys),)
        if best_score is None or score > best_score:
            best_score = score
            best_name = name
            best_module = cand
    if best_module is None:
        raise ValueError(f"Could not locate encoder module inside {type(model_obj).__name__}")
    return best_name, best_module


def initialize_mlm_from_token_checkpoint(init_model_ref: str) -> Tuple[torch.nn.Module, AutoConfig]:
    init_path = Path(init_model_ref)
    cfg = AutoConfig.from_pretrained(init_model_ref)
    if init_path.exists() and _looks_like_custom_token_checkpoint(init_path):
        num_token_labels = infer_num_token_labels(init_path)
        token_model = TokenDeberta.from_pretrained(
            init_model_ref,
            config=cfg,
            num_token_labels=num_token_labels,
        )
        mlm_model = AutoModelForMaskedLM.from_config(cfg)
        source_state = token_model.encoder.state_dict()
        encoder_name, encoder_module = _extract_encoder_module(
            mlm_model,
            reference_state_keys=set(source_state.keys()),
        )
        missing, unexpected = encoder_module.load_state_dict(source_state, strict=False)
        print(
            f"[MLM Init] loaded encoder from token model into {encoder_name}: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        return mlm_model, cfg

    mlm_model = AutoModelForMaskedLM.from_pretrained(init_model_ref, config=cfg)
    print(f"[MLM Init] loaded MLM model directly from base encoder: {init_model_ref}")
    return mlm_model, cfg


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    init_model_ref = args.init_model_dir
    tokenizer = AutoTokenizer.from_pretrained(init_model_ref)

    train_dataset = PackedMLMDataset(
        text_file=args.train_text_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    val_dataset = PackedMLMDataset(
        text_file=args.val_text_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    model, _ = initialize_mlm_from_token_checkpoint(init_model_ref)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    eval_strategy = "steps" if args.eval_steps > 0 else "epoch"
    save_strategy = "steps" if args.save_steps > 0 else "epoch"

    train_args = TrainingArguments(
        output_dir=str(Path(args.output_dir) / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        eval_steps=args.eval_steps if eval_strategy == "steps" else None,
        save_steps=args.save_steps if save_strategy == "steps" else None,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=args.fp16,
        report_to="none",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    optimizer = CompatAdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        steps_per_epoch = max(
            1,
            math.ceil(len(train_dataset) / (args.batch_size * args.gradient_accumulation_steps)),
        )
        total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    final_dir = Path(args.output_dir) / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    payload = {
        "def_com_trans_pipeline_version": TRANSCRIPT_MLM_PIPELINE_VERSION,
        "init_model_dir": str(init_model_ref),
        "train_text_file": args.train_text_file,
        "val_text_file": args.val_text_file,
        "final_model_dir": str(final_dir),
        "eval_metrics": eval_metrics,
        "args": vars(args),
    }
    with (Path(args.output_dir) / "mlm_training_report.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[MLM Done] final_model={final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
