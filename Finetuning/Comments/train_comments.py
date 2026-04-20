#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    Adafactor,
    AutoConfig,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    set_seed,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

COMMENTS_ROOT = Path("PATH_TO_COMMENTS")
COMMENT_PIPELINE_VERSION = "nature_legacy_v1"


def resolve_comment_split_dir() -> Path:
    candidate_dirs = [
        COMMENTS_ROOT / "Def_splits",
        COMMENTS_ROOT / "Def_Anon_Splits",
    ]
    for split_dir in candidate_dirs:
        if all((split_dir / f"{name}.jsonl").exists() for name in ["train", "dev", "test"]):
            return split_dir
    raise FileNotFoundError(
        "Could not find comment splits. Checked: "
        + ", ".join(str(path) for path in candidate_dirs)
    )


DEFAULT_COMMENTS_SPLIT_DIR = resolve_comment_split_dir()

from scripts.labels import LABEL_LIST
from comments_legacy.data_processing import prepare_datasets
from comments_legacy.model import TokenDeberta
from comments_legacy.utils import (
    compute_pos_weight_from_dataset,
    compute_span_metrics,
    compute_token_metrics_from_logits,
    compute_token_per_label_metrics,
    print_training_summary,
)


class TokenTrainerWrapper(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_loss_components = {}
        self._last_grad_norm = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        token_labels = inputs.pop("token_labels", None)
        if token_labels is None:
            raise ValueError("Missing token_labels in batch.")
        if torch.isnan(token_labels).any():
            raise ValueError("Found NaNs in token_labels")
        device = model.device if hasattr(model, "device") else next(model.parameters()).device
        token_labels = token_labels.to(device)
        outputs = model(**inputs, token_labels=token_labels)
        loss = outputs.get("loss", None)

        if not return_outputs and loss is None:
            raise ValueError("Model must return a loss when labels are provided.")

        with torch.no_grad():
            loss_components = {}
            if loss is not None:
                loss_components["loss"] = float(loss.detach().cpu().item())
            token_loss = outputs.get("token_loss")
            if token_loss is not None:
                loss_components["token_loss"] = float(token_loss.detach().cpu().item())
            if loss_components:
                self._last_loss_components = loss_components

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if torch.isnan(param.grad).any():
                raise ValueError(f"NaN gradient detected in parameter '{name}' at step {self.state.global_step}")
            grad_norms.append(torch.norm(param.grad.detach().float(), 2))
        if grad_norms:
            stacked = torch.stack(grad_norms)
            self._last_grad_norm = float(torch.norm(stacked, 2).detach().cpu().item())
        else:
            self._last_grad_norm = None
        return loss

    def log(self, logs, start_time=None):
        merged_logs = dict(logs)
        if getattr(self, "_last_loss_components", None):
            for key, value in self._last_loss_components.items():
                merged_logs.setdefault(key, value)
        if getattr(self, "_last_grad_norm", None) is not None:
            merged_logs.setdefault("grad_norm", self._last_grad_norm)
        super().log(merged_logs, start_time)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = inputs.get("token_labels") is not None
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
            token_logits = outputs["token_logits"]
            loss = outputs.get("loss") if has_labels else None

        if prediction_loss_only:
            return (loss, None, None)

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            raise ValueError("attention_mask is required for prediction")
        logits = token_logits.detach().cpu()

        if has_labels:
            labels = (
                inputs["token_labels"].detach().cpu(),
                attention_mask.detach().cpu(),
            )
        else:
            labels = None

        return (loss, logits, labels)


class NaNGradientCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                raise ValueError(f"NaN gradient detected in parameter '{name}' at step {state.global_step}")


def _get_encoder_layer_list(model):
    candidates = []
    for attr in ["encoder", "deberta", "roberta", "bert", "model"]:
        if hasattr(model, attr):
            candidates.append(getattr(model, attr))
    for candidate in list(candidates):
        if candidate is not None and hasattr(candidate, "encoder"):
            candidates.append(candidate.encoder)

    for candidate in candidates:
        if candidate is None:
            continue
        for layers_attr in ["layer", "layers"]:
            layers = getattr(candidate, layers_attr, None)
            if layers is not None:
                return list(layers)
    return None


def freeze_bottom_layers(model, num_layers_to_freeze: int = 6):
    layers = _get_encoder_layer_list(model)
    if not layers:
        print("[LayerFreeze] Warning: could not access encoder layers. No freezing applied.")
        return
    for layer in layers[:num_layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False
    print(f"[LayerFreeze] Frozen bottom {min(num_layers_to_freeze, len(layers))} layers.")


def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True
    print("[LayerFreeze] All layers unfrozen for full fine-tuning.")


class UnfreezeCallback(TrainerCallback):
    def __init__(self, unfreeze_epoch: int = 2):
        self.unfreeze_epoch = unfreeze_epoch
        self.unfrozen = False

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if self.unfrozen or state.epoch is None or model is None:
            return
        if state.epoch >= self.unfreeze_epoch:
            unfreeze_all_layers(model)
            self.unfrozen = True
            print(f"[LayerFreeze] Layers unfrozen at epoch {self.unfreeze_epoch}.")


class DelayedEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, min_epochs: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.min_epochs = max(0, min_epochs)

    def on_evaluate(self, args, state, control, **kwargs):
        epoch = state.epoch or 0
        if epoch + 1 < self.min_epochs:
            return control
        return super().on_evaluate(args, state, control, **kwargs)


def apply_dropout_to_config(config, dropout_prob: float) -> None:
    for attr in [
        "hidden_dropout_prob",
        "attention_probs_dropout_prob",
        "hidden_dropout",
        "attention_dropout",
        "dropout",
        "classifier_dropout",
        "classifier_dropout_prob",
        "embed_dropout",
        "embedding_dropout",
    ]:
        if hasattr(config, attr):
            setattr(config, attr, dropout_prob)
    if not hasattr(config, "hidden_dropout_prob"):
        config.hidden_dropout_prob = dropout_prob
    if not hasattr(config, "attention_probs_dropout_prob"):
        config.attention_probs_dropout_prob = dropout_prob


def tune_token_threshold_on_dev(trainer, dev_dataset, base_threshold: float, grid=None) -> Tuple[float, Optional[float]]:
    if grid is None:
        grid = [round(x, 2) for x in np.arange(0.30, 0.71, 0.02)]
    try:
        preds = trainer.predict(dev_dataset)
        token_logits = preds.predictions
        label_ids = preds.label_ids
        if isinstance(label_ids, (list, tuple)) and len(label_ids) == 2:
            token_labels, attention_mask = label_ids
        else:
            token_labels = label_ids
            attention_mask = None
    except Exception as exc:
        print(f"[Threshold] Dev predictions failed, fallback to base threshold {base_threshold}: {exc}")
        return base_threshold, None

    best_thr = base_threshold
    best_f1 = -1.0
    for thr in grid:
        metrics = compute_token_metrics_from_logits(
            token_logits,
            token_labels,
            attention_mask=attention_mask,
            threshold=thr,
        )
        f1 = metrics.get("token_f1_macro", 0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed-parameter comments span training stage.")
    parser.add_argument("--model-name", required=True, help="Encoder/checkpoint path used to initialize the span model.")
    parser.add_argument(
        "--train-file",
        default=str(DEFAULT_COMMENTS_SPLIT_DIR / "train.jsonl"),
    )
    parser.add_argument(
        "--dev-file",
        default=str(DEFAULT_COMMENTS_SPLIT_DIR / "dev.jsonl"),
    )
    parser.add_argument(
        "--test-file",
        default=str(DEFAULT_COMMENTS_SPLIT_DIR / "test.jsonl"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2.6049078843086554e-05)
    parser.add_argument("--num-epochs", type=int, default=28)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--dropout-prob", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--freeze-layers", type=int, default=6)
    parser.add_argument("--unfreeze-epoch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset, dev_dataset, test_dataset, _ = prepare_datasets(
        args.train_file,
        args.dev_file,
        args.test_file,
        LABEL_LIST,
        tokenizer,
        max_length=args.max_length,
    )
    pos_weight = compute_pos_weight_from_dataset(
        train_dataset,
        len(LABEL_LIST),
        field="token_labels",
    )

    config = AutoConfig.from_pretrained(args.model_name)
    apply_dropout_to_config(config, args.dropout_prob)
    model = TokenDeberta(
        config=config,
        num_token_labels=len(LABEL_LIST),
        pretrained_model_name=args.model_name,
        dropout_prob=args.dropout_prob,
    )
    model.set_pos_weight(pos_weight)
    if model.pos_weight is not None:
        print("Token pos_weight:", model.pos_weight.detach().cpu())
    freeze_bottom_layers(model, num_layers_to_freeze=args.freeze_layers)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_dir=str(Path(args.output_dir) / "logs"),
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_token_f1_macro",
        greater_is_better=True,
        fp16=args.fp16,
        report_to="none",
        save_total_limit=1,
        max_grad_norm=0.5,
        remove_unused_columns=False,
    )
    optimizer = Adafactor(
        model.parameters(),
        lr=args.learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(
        1,
        math.ceil(len(train_dataset) / (args.batch_size * args.gradient_accumulation_steps)),
    )
    total_steps = max(1, steps_per_epoch * args.num_epochs)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    trainer = TokenTrainerWrapper(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=lambda pred: compute_span_metrics(pred, threshold=args.threshold),
        optimizers=(optimizer, scheduler),
        callbacks=[
            NaNGradientCallback(),
            DelayedEarlyStoppingCallback(
                min_epochs=max(5, args.num_epochs // 2),
                early_stopping_patience=3,
                early_stopping_threshold=0.001,
            ),
            UnfreezeCallback(unfreeze_epoch=args.unfreeze_epoch),
        ],
    )

    trainer.train()
    tuned_threshold, tuned_dev_f1 = tune_token_threshold_on_dev(
        trainer,
        dev_dataset,
        base_threshold=args.threshold,
    )
    if tuned_dev_f1 is not None:
        print(f"[Threshold] Tuned on dev: {tuned_threshold:.2f} (F1_macro={tuned_dev_f1:.3f})")
    else:
        print(f"[Threshold] Using base threshold {tuned_threshold:.2f}")
    trainer.compute_metrics = lambda pred: compute_span_metrics(pred, threshold=tuned_threshold)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with (Path(args.output_dir) / "label_list.json").open("w", encoding="utf-8") as f:
        json.dump(LABEL_LIST, f, indent=2)

    config_payload = vars(args).copy()
    config_payload["label_list"] = LABEL_LIST
    config_payload["tuned_threshold"] = float(tuned_threshold)
    config_payload["comment_pipeline_version"] = COMMENT_PIPELINE_VERSION
    with (Path(args.output_dir) / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    test_results = trainer.evaluate(test_dataset)
    print_training_summary(test_results, "Final", mode="test")
    with (Path(args.output_dir) / "test_results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {k: float(v) if isinstance(v, np.floating) else v for k, v in test_results.items()},
            f,
            indent=2,
        )

    test_predictions = trainer.predict(test_dataset)
    token_logits = test_predictions.predictions
    token_probs = torch.sigmoid(torch.tensor(token_logits)).numpy()
    token_preds = (token_probs > tuned_threshold).astype(int)
    np.save(Path(args.output_dir) / "test_token_predictions.npy", token_preds)
    np.save(Path(args.output_dir) / "test_token_probabilities.npy", token_probs)

    label_ids = test_predictions.label_ids
    if isinstance(label_ids, (list, tuple)) and len(label_ids) == 2:
        token_labels, attention_mask = label_ids
    else:
        token_labels = label_ids
        attention_mask = np.ones(token_labels.shape[:2], dtype=bool)
    per_label_metrics = compute_token_per_label_metrics(
        token_logits,
        token_labels,
        attention_mask=attention_mask,
        label_list=LABEL_LIST,
        threshold=tuned_threshold,
    )
    with (Path(args.output_dir) / "test_token_per_label.json").open("w", encoding="utf-8") as f:
        json.dump(per_label_metrics, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
