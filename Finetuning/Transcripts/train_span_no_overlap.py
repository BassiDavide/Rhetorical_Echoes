#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    get_cosine_schedule_with_warmup,
    set_seed,
)

from scripts.labels import LABEL_LIST
from span_singlehead.data_processing import SpanDataProcessor, prepare_datasets, prepare_single_dataset
from span_singlehead.model import MultiHeadTokenDeberta, TokenDeberta
from span_singlehead.utils import (
    compute_pos_weight_from_dataset,
    compute_span_metrics,
    compute_token_per_label_metrics,
    compute_token_metrics_from_logits,
)

TRANSCRIPT_SPAN_PIPELINE_VERSION = "shared_transcript_span_v3"

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


DEFAULT_INIT_MODEL = Path(
    "ADD_PATH"
)
DEFAULT_DATA_DIR = Path("ADD_PATH")
DEFAULT_OUTPUT = Path("ADD_PATH")
DEFAULT_COMMENTS_TRAIN = Path(
    "ADD_PATH"
)


class CompatAdamW(AdamW):
    """Compatibility shim for accelerate versions that call optimizer.train()/eval()."""

    def train(self):
        return self

    def eval(self):
        return self


class BalancedDomainBatchSampler(Sampler[List[int]]):
    """Yield batches with fixed comment/transcript composition."""

    def __init__(
        self,
        n_comments: int,
        n_transcripts: int,
        batch_size: int,
        comment_fraction: float,
        steps_per_epoch: int,
        seed: int = 42,
    ):
        if n_comments <= 0 or n_transcripts <= 0:
            raise ValueError("Both comment and transcript datasets must be non-empty for balanced sampling.")
        if batch_size < 2:
            raise ValueError("batch_size must be >= 2 for balanced sampling.")
        if comment_fraction <= 0.0 or comment_fraction >= 1.0:
            raise ValueError("comment_fraction must be in (0, 1).")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be > 0.")

        self.n_comments = int(n_comments)
        self.n_transcripts = int(n_transcripts)
        self.batch_size = int(batch_size)
        self.comment_fraction = float(comment_fraction)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.epoch = 0

        comment_count = int(round(self.batch_size * self.comment_fraction))
        comment_count = max(1, min(comment_count, self.batch_size - 1))
        self.comment_count = comment_count
        self.transcript_count = self.batch_size - self.comment_count

        self.comment_indices = np.arange(0, self.n_comments, dtype=np.int64)
        self.transcript_indices = np.arange(self.n_comments, self.n_comments + self.n_transcripts, dtype=np.int64)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        comment_pool = rng.permutation(self.comment_indices)
        transcript_pool = rng.permutation(self.transcript_indices)
        comment_ptr = 0
        transcript_ptr = 0

        for _ in range(self.steps_per_epoch):
            if comment_ptr + self.comment_count > len(comment_pool):
                comment_pool = rng.permutation(self.comment_indices)
                comment_ptr = 0
            if transcript_ptr + self.transcript_count > len(transcript_pool):
                transcript_pool = rng.permutation(self.transcript_indices)
                transcript_ptr = 0

            c = comment_pool[comment_ptr : comment_ptr + self.comment_count]
            t = transcript_pool[transcript_ptr : transcript_ptr + self.transcript_count]
            comment_ptr += self.comment_count
            transcript_ptr += self.transcript_count

            batch = np.concatenate([c, t]).tolist()
            rng.shuffle(batch)
            yield batch


class SamplerEpochCallback(TrainerCallback):
    def __init__(self, sampler: BalancedDomainBatchSampler):
        self.sampler = sampler

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = 0 if state.epoch is None else int(state.epoch)
        self.sampler.set_epoch(epoch)


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


def load_label_list(model_dir: Path):
    label_path = model_dir / "label_list.json"
    if label_path.exists():
        with label_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    print("[LabelList] label_list.json not found in init model; using default canonical label list")
    return list(LABEL_LIST)


def build_span_model(
    *,
    init_model_ref: str,
    config,
    num_labels: int,
    model_architecture: str,
    multi_head_dim: int,
):
    init_path = Path(init_model_ref)
    custom_checkpoint = init_path.exists() and _looks_like_custom_token_checkpoint(init_path)
    if model_architecture == "multi_head":
        if custom_checkpoint:
            return MultiHeadTokenDeberta.from_pretrained(
                init_model_ref,
                config=config,
                num_token_labels=num_labels,
                head_dim=multi_head_dim,
            )
        return MultiHeadTokenDeberta(
            config=config,
            num_token_labels=num_labels,
            head_dim=multi_head_dim,
            pretrained_model_name=init_model_ref,
        )
    if custom_checkpoint:
        return TokenDeberta.from_pretrained(
            init_model_ref,
            config=config,
            num_token_labels=num_labels,
        )
    return TokenDeberta(
        config=config,
        num_token_labels=num_labels,
        pretrained_model_name=init_model_ref,
    )


def _extract_encoder_module(model_obj):
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
        score = len(keys)
        if best_score is None or score > best_score:
            best_score = score
            best_name = name
            best_module = cand
    if best_module is None:
        raise ValueError(f"Could not locate encoder module inside {type(model_obj).__name__}")
    return best_name, best_module


def parse_ignore_labels(ignore_labels_arg: str, label_list: Sequence[str]) -> Tuple[List[str], np.ndarray]:
    requested = [x.strip() for x in (ignore_labels_arg or "").split(",") if x.strip()]
    by_cf = {label.casefold(): label for label in label_list}
    ignored: List[str] = []
    for name in requested:
        canonical = by_cf.get(name.casefold())
        if canonical is None:
            print(f"[IgnoreLabels] warning: '{name}' not found in label_list, skipping")
            continue
        if canonical not in ignored:
            ignored.append(canonical)
    active_mask = np.array([label not in set(ignored) for label in label_list], dtype=bool)
    if not np.any(active_mask):
        raise ValueError("All labels were ignored; at least one active label is required.")
    return ignored, active_mask


def parse_label_name_list(labels_arg: str, label_list: Sequence[str]) -> List[str]:
    requested = [x.strip() for x in (labels_arg or "").split(",") if x.strip()]
    by_cf = {label.casefold(): label for label in label_list}
    resolved: List[str] = []
    for name in requested:
        canonical = by_cf.get(name.casefold())
        if canonical is None:
            print(f"[LabelList] warning: '{name}' not found in label_list, skipping")
            continue
        if canonical not in resolved:
            resolved.append(canonical)
    return resolved


def compute_label_priors_from_dataset(
    dataset,
    num_labels: int,
    field: str = "token_labels",
    supervision_mask_field: str = "token_supervision_mask",
    active_label_mask=None,
    min_prior: float = 1e-4,
) -> np.ndarray:
    pos_counts = torch.zeros(num_labels, dtype=torch.float64)
    total_counts = torch.tensor(0.0, dtype=torch.float64)
    for example in dataset:
        labels = example[field]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        labels = labels.detach().to(dtype=torch.float64)
        supervision_mask = example.get(supervision_mask_field, None)
        if supervision_mask is not None:
            if not isinstance(supervision_mask, torch.Tensor):
                supervision_mask = torch.tensor(supervision_mask)
            supervision_mask = supervision_mask.detach().bool()
            labels = labels[supervision_mask]
            if labels.numel() == 0:
                continue
        pos_counts += labels.sum(dim=0)
        total_counts += float(labels.shape[0])

    # Beta(1,1) smoothing to avoid degenerate 0/1 priors.
    priors = (pos_counts + 1.0) / (total_counts + 2.0)
    floor = float(min_prior)
    if floor <= 0.0 or floor >= 0.5:
        raise ValueError(f"min_prior must be in (0, 0.5), got {floor}")
    priors = priors.clamp(min=floor, max=1.0 - floor)
    priors_np = priors.cpu().numpy().astype(np.float32)
    if active_label_mask is not None:
        active_mask = np.array(active_label_mask).astype(bool)
        priors_np[~active_mask] = 1e-4
    return priors_np


def initialize_multi_head_biases(
    *,
    model,
    mode: str,
    constant: float,
    label_prior_floor: float,
    train_dataset,
    num_labels: int,
    active_label_mask=None,
) -> None:
    if not isinstance(model, MultiHeadTokenDeberta):
        return
    if mode == "none":
        return

    with torch.no_grad():
        if mode == "constant":
            for head in model.heads:
                head.bias.fill_(float(constant))
            print(f"[MultiHeadBiasInit] mode=constant value={float(constant):.4f}")
            return

        if mode == "label_prior":
            priors = compute_label_priors_from_dataset(
                train_dataset,
                num_labels=num_labels,
                active_label_mask=active_label_mask,
                min_prior=label_prior_floor,
            )
            logits = np.log(priors / (1.0 - priors))
            for idx, head in enumerate(model.heads):
                head.bias.fill_(float(logits[idx]))
            print(
                "[MultiHeadBiasInit] mode=label_prior "
                f"prior_floor={float(label_prior_floor):.6f} "
                f"prior_min={float(priors.min()):.6f} "
                f"prior_max={float(priors.max()):.6f} "
                f"logit_min={float(logits.min()):.4f} "
                f"logit_max={float(logits.max()):.4f}"
            )
            return

    raise ValueError(f"Unsupported multi_head_init_bias_mode: {mode}")


def build_optimizer_param_groups(
    *,
    model,
    base_learning_rate: float,
    encoder_learning_rate: float,
    head_learning_rate: float,
):
    use_encoder_lr = encoder_learning_rate > 0.0
    use_head_lr = head_learning_rate > 0.0
    if not use_encoder_lr and not use_head_lr:
        return model.parameters(), {
            "enabled": False,
            "encoder_lr": None,
            "head_lr": None,
            "base_lr": float(base_learning_rate),
            "n_encoder_params": 0,
            "n_head_params": 0,
            "n_base_params": 0,
        }

    groups_by_lr: Dict[float, List[torch.nn.Parameter]] = {}
    n_encoder = 0
    n_head = 0
    n_base = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr = float(base_learning_rate)
        if use_encoder_lr and name.startswith("encoder."):
            lr = float(encoder_learning_rate)
            n_encoder += param.numel()
        elif use_head_lr and (
            name.startswith("token_classifier.")
            or name.startswith("proj.")
            or name.startswith("heads.")
        ):
            lr = float(head_learning_rate)
            n_head += param.numel()
        else:
            n_base += param.numel()
        groups_by_lr.setdefault(lr, []).append(param)

    param_groups = [
        {"params": params, "lr": lr}
        for lr, params in sorted(groups_by_lr.items(), key=lambda x: x[0])
    ]
    return param_groups, {
        "enabled": True,
        "encoder_lr": float(encoder_learning_rate) if use_encoder_lr else None,
        "head_lr": float(head_learning_rate) if use_head_lr else None,
        "base_lr": float(base_learning_rate),
        "n_encoder_params": int(n_encoder),
        "n_head_params": int(n_head),
        "n_base_params": int(n_base),
    }


def freeze_bottom_layers(model, num_layers_to_freeze: int):
    encoder = None
    layers = None
    for attr in ["encoder", "deberta", "roberta", "bert", "model"]:
        if not hasattr(model, attr):
            continue
        cand = getattr(model, attr)
        candidates = [cand]
        if hasattr(cand, "encoder"):
            candidates.append(cand.encoder)
        for candidate in candidates:
            if hasattr(candidate, "layer"):
                encoder = candidate
                layers = list(candidate.layer)
                break
            if hasattr(candidate, "layers"):
                encoder = candidate
                layers = list(candidate.layers)
                break
        if layers is not None:
            break
    if encoder is None or not layers:
        print("[LayerFreeze] skipped: encoder layers not found")
        return
    for layer in layers[:num_layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False
    print(f"[LayerFreeze] froze bottom {min(num_layers_to_freeze, len(layers))} layers")


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


def load_encoder_from_mlm_checkpoint(model, mlm_path: str) -> None:
    if not mlm_path:
        return
    mlm_dir = Path(mlm_path)
    if not mlm_dir.exists():
        raise FileNotFoundError(f"MLM checkpoint not found: {mlm_dir}")

    encoder_model = AutoModel.from_pretrained(mlm_dir)
    best_name, best_encoder = _extract_encoder_module(encoder_model)
    best_state = best_encoder.state_dict()
    target_keys = set(model.encoder.state_dict().keys())
    src_keys = set(best_state.keys())
    best_missing = len(target_keys - src_keys)
    best_unexpected = len(src_keys - target_keys)

    missing, unexpected = model.encoder.load_state_dict(best_state, strict=False)
    print(
        f"[EncoderInitFromMLM] path={mlm_dir} "
        f"source={best_name} "
        f"candidate_missing={best_missing} candidate_unexpected={best_unexpected} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


class TokenTrainerWrapper(Trainer):
    def __init__(self, *args, balanced_batch_sampler: Optional[BalancedDomainBatchSampler] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.balanced_batch_sampler = balanced_batch_sampler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        token_labels = inputs.pop("token_labels", None)
        token_supervision_mask = inputs.pop("token_supervision_mask", None)
        if token_labels is None:
            raise ValueError("Missing token_labels")
        token_labels = token_labels.to(model.device)
        if token_supervision_mask is not None:
            token_supervision_mask = token_supervision_mask.to(model.device)
        outputs = model(**inputs, token_labels=token_labels, token_supervision_mask=token_supervision_mask)
        loss = outputs.get("loss")
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        if self.balanced_batch_sampler is None:
            return super().get_train_dataloader()
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.balanced_batch_sampler,
            collate_fn=self.data_collator or default_data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

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
            raise ValueError("attention_mask is required")
        logits = token_logits.detach().cpu()

        if has_labels:
            supervision_mask = inputs.get("token_supervision_mask")
            labels = (
                inputs["token_labels"].detach().cpu(),
                attention_mask.detach().cpu(),
                supervision_mask.detach().cpu() if supervision_mask is not None else None,
            )
        else:
            labels = None
        return (loss, logits, labels)


class DelayedEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, min_epochs: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.min_epochs = max(0, min_epochs)

    def on_evaluate(self, args, state, control, **kwargs):
        epoch = state.epoch or 0
        if epoch + 1 < self.min_epochs:
            return control
        return super().on_evaluate(args, state, control, **kwargs)


class NaNGradientCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                raise ValueError(f"NaN gradient in '{name}' at step {state.global_step}")


def unpack_label_payload(label_ids):
    if isinstance(label_ids, (list, tuple)) and len(label_ids) == 3:
        token_labels, attention_mask, supervision_mask = label_ids
    elif isinstance(label_ids, (list, tuple)) and len(label_ids) == 2:
        token_labels, attention_mask = label_ids
        supervision_mask = None
    else:
        token_labels = label_ids
        attention_mask = None
        supervision_mask = None
    return token_labels, attention_mask, supervision_mask


def tune_token_threshold_on_dev(
    trainer,
    dev_dataset,
    base_threshold: float,
    grid=None,
    active_label_mask=None,
) -> Tuple[float, Optional[float]]:
    if grid is None:
        grid = [round(x, 2) for x in np.arange(0.30, 0.71, 0.02)]
    preds = trainer.predict(dev_dataset)
    token_logits = preds.predictions
    token_labels, attention_mask, supervision_mask = unpack_label_payload(preds.label_ids)

    best_thr = base_threshold
    best_f1 = -1.0
    for thr in grid:
        metrics = compute_token_metrics_from_logits(
            token_logits,
            token_labels,
            attention_mask=(attention_mask, supervision_mask),
            threshold=thr,
            active_label_mask=active_label_mask,
        )
        f1 = metrics.get("token_f1_macro", 0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


def tune_token_thresholds_per_label_on_dev(
    trainer,
    dev_dataset,
    label_list: Sequence[str],
    base_threshold: float,
    grid: Sequence[float],
    min_support_per_label: int = 1,
    active_label_mask=None,
    priority_metric_by_label: Optional[Dict[str, str]] = None,
    recall_priority_beta: float = 2.0,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    preds = trainer.predict(dev_dataset)
    token_logits = np.array(preds.predictions)
    probs = 1.0 / (1.0 + np.exp(-token_logits))

    token_labels, attention_mask, supervision_mask = unpack_label_payload(preds.label_ids)

    labels = np.array(token_labels).astype(int)
    if attention_mask is None:
        mask = np.ones(labels.shape[:2], dtype=bool)
    else:
        mask = np.array(attention_mask).astype(bool)
    if supervision_mask is not None:
        mask = mask & np.array(supervision_mask).astype(bool)

    probs_flat = probs[mask]
    labels_flat = labels[mask]

    num_labels = len(label_list)
    priority_metric_by_label = priority_metric_by_label or {}
    if active_label_mask is None:
        active_label_mask = np.ones((num_labels,), dtype=bool)
    else:
        active_label_mask = np.array(active_label_mask).astype(bool)
        if active_label_mask.shape[0] != num_labels:
            raise ValueError("active_label_mask length mismatch.")
    best_thresholds: List[float] = [float(base_threshold)] * num_labels
    summary: Dict[str, Dict[str, float]] = {}

    for idx, label_name in enumerate(label_list):
        if not active_label_mask[idx]:
            best_thresholds[idx] = 1.0
            summary[label_name] = {
                "threshold": 1.0,
                "f1": 0.0,
                "objective_name": "ignored",
                "objective_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "support": 0.0,
                "used_min_support_fallback": 0.0,
                "ignored": 1.0,
            }
            continue
        y_true = labels_flat[:, idx]
        y_prob = probs_flat[:, idx]
        support = int(y_true.sum())
        metric_name = priority_metric_by_label.get(label_name, "f1")

        if support < min_support_per_label:
            y_pred = (y_prob > base_threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            if metric_name == "fbeta_recall":
                objective_score = fbeta_score(y_true, y_pred, beta=recall_priority_beta, zero_division=0)
            else:
                objective_score = f1
            summary[label_name] = {
                "threshold": float(base_threshold),
                "f1": float(f1),
                "objective_name": metric_name,
                "objective_score": float(objective_score),
                "precision": float(p),
                "recall": float(r),
                "support": float(support),
                "used_min_support_fallback": 1.0,
                "ignored": 0.0,
            }
            continue

        best_thr = float(base_threshold)
        best_f1 = -1.0
        best_objective = -1.0
        best_p = 0.0
        best_r = 0.0
        for thr in grid:
            y_pred = (y_prob > thr).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            if metric_name == "fbeta_recall":
                objective_score = fbeta_score(y_true, y_pred, beta=recall_priority_beta, zero_division=0)
            else:
                objective_score = f1
            # Tie-break: prefer lower threshold to favor recall on ambiguous labels.
            if (objective_score > best_objective) or (
                abs(objective_score - best_objective) < 1e-12 and thr < best_thr
            ):
                best_objective = float(objective_score)
                best_f1 = float(f1)
                best_thr = float(thr)
                best_p = float(p)
                best_r = float(r)

        best_thresholds[idx] = best_thr
        summary[label_name] = {
            "threshold": best_thr,
            "f1": best_f1,
            "objective_name": metric_name,
            "objective_score": best_objective,
            "precision": best_p,
            "recall": best_r,
            "support": float(support),
            "used_min_support_fallback": 0.0,
            "ignored": 0.0,
        }

    return np.array(best_thresholds, dtype=np.float32), summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train transcript span model on no-overlap chunks.")
    parser.add_argument("--train-file", default=str(DEFAULT_DATA_DIR / "train.jsonl"))
    parser.add_argument("--dev-file", default=str(DEFAULT_DATA_DIR / "dev.jsonl"))
    parser.add_argument("--test-file", default=str(DEFAULT_DATA_DIR / "test.jsonl"))
    parser.add_argument(
        "--comments-train-file",
        default=str(DEFAULT_COMMENTS_TRAIN),
        help="Comment training split used when --train-mode mixed.",
    )
    parser.add_argument(
        "--train-mode",
        choices=["transcripts_only", "mixed"],
        default="transcripts_only",
        help="Use transcript chunks only or mixed comments+transcript training.",
    )
    parser.add_argument(
        "--mix-comment-fraction",
        type=float,
        default=0.5,
        help="Target comment fraction within each mixed batch.",
    )
    parser.add_argument(
        "--mixed-steps-per-epoch",
        type=int,
        default=0,
        help="Override mixed-mode steps per epoch (0 = auto from transcript coverage).",
    )
    parser.add_argument("--init-model-dir", default=str(DEFAULT_INIT_MODEL))
    parser.add_argument(
        "--encoder-init-from-mlm",
        default="",
        help="Optional MLM checkpoint; if set, load encoder weights from this path before span fine-tuning.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument(
        "--encoder-learning-rate",
        type=float,
        default=0.0,
        help="Optional override LR for encoder params; 0 disables differential LR.",
    )
    parser.add_argument(
        "--head-learning-rate",
        type=float,
        default=0.0,
        help="Optional override LR for task-head params; 0 disables differential LR.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--num-epochs", type=int, default=12)
    parser.add_argument(
        "--use-curriculum",
        action="store_true",
        help="Run stage A mixed training then stage B transcript-only adaptation.",
    )
    parser.add_argument("--stage-a-epochs", type=int, default=3)
    parser.add_argument("--stage-b-epochs", type=int, default=12)
    parser.add_argument("--stage-a-learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--stage-b-learning-rate", type=float, default=3.0e-6)
    parser.add_argument(
        "--ignore-labels",
        type=str,
        default="",
        help="Comma-separated canonical labels to ignore in supervision/loss/eval.",
    )
    parser.add_argument("--dropout-prob", type=float, default=0.2)
    parser.add_argument(
        "--model-architecture",
        choices=["single_head", "multi_head"],
        default="single_head",
        help="Token classification head type on top of the shared encoder.",
    )
    parser.add_argument(
        "--multi-head-dim",
        type=int,
        default=64,
        help="Shared projection dim used before per-technique heads when --model-architecture=multi_head.",
    )
    parser.add_argument(
        "--multi-head-loss-normalization",
        choices=["sum", "mean_over_active_labels"],
        default="sum",
        help=(
            "Loss reduction mode for multi_head. "
            "'sum' keeps previous behavior; "
            "'mean_over_active_labels' normalizes by active label count."
        ),
    )
    parser.add_argument(
        "--multi-head-init-bias-mode",
        choices=["none", "constant", "label_prior"],
        default="none",
        help="Optional initialization for multi-head output biases.",
    )
    parser.add_argument(
        "--multi-head-init-bias-constant",
        type=float,
        default=-2.5,
        help="Bias value used when --multi-head-init-bias-mode=constant.",
    )
    parser.add_argument(
        "--multi-head-label-prior-floor",
        type=float,
        default=0.005,
        help=(
            "Lower clamp for per-label token priors when "
            "--multi-head-init-bias-mode=label_prior. "
            "Prevents extreme negative logits on sparse labels."
        ),
    )
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=None,
        help="Optional threshold used only for dev eval during training/early stopping.",
    )
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument(
        "--threshold-mode",
        choices=["global", "per_label"],
        default="per_label",
        help="Tune one global threshold or one threshold per technique on dev.",
    )
    parser.add_argument("--threshold-grid-min", type=float, default=0.10)
    parser.add_argument("--threshold-grid-max", type=float, default=0.70)
    parser.add_argument("--threshold-grid-step", type=float, default=0.02)
    parser.add_argument("--min-support-per-label", type=int, default=1)
    parser.add_argument(
        "--threshold-recall-priority-labels",
        type=str,
        default="smears/doubt,loaded language,distraction",
        help="Comma-separated labels whose thresholds are tuned with recall-biased F-beta instead of F1.",
    )
    parser.add_argument(
        "--threshold-recall-beta",
        type=float,
        default=2.0,
        help="Beta used for recall-biased F-beta threshold tuning.",
    )
    parser.add_argument("--freeze-layers", type=int, default=6)
    parser.add_argument(
        "--no-pos-weight",
        action="store_false",
        dest="use_pos_weight",
        help="Disable class-imbalance pos_weight in BCE.",
    )
    parser.add_argument(
        "--use-pos-weight",
        action="store_true",
        dest="use_pos_weight",
        help="Enable class-imbalance pos_weight in BCE (default).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.set_defaults(use_pos_weight=True)
    return parser.parse_args()


def train_stage(
    *,
    stage_name: str,
    model,
    train_dataset,
    dev_dataset,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    fp16: bool,
    output_root: str,
    threshold_for_eval: float,
    active_label_mask=None,
    balanced_sampler: Optional[BalancedDomainBatchSampler] = None,
    encoder_learning_rate: float = 0.0,
    head_learning_rate: float = 0.0,
):
    stage_output_dir = str(Path(output_root) / stage_name)
    train_args = TrainingArguments(
        output_dir=stage_output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_token_f1_macro",
        greater_is_better=True,
        fp16=fp16,
        report_to="none",
        save_total_limit=1,
        max_grad_norm=0.5,
        remove_unused_columns=False,
    )

    optimizer_params, lr_info = build_optimizer_param_groups(
        model=model,
        base_learning_rate=learning_rate,
        encoder_learning_rate=encoder_learning_rate,
        head_learning_rate=head_learning_rate,
    )
    optimizer = CompatAdamW(
        optimizer_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    if balanced_sampler is not None:
        steps_per_epoch = max(1, math.ceil(len(balanced_sampler) / gradient_accumulation_steps))
    else:
        steps_per_epoch = max(1, math.ceil(len(train_dataset) / (batch_size * gradient_accumulation_steps)))
    total_steps = max(1, steps_per_epoch * num_epochs)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    callbacks = [
        NaNGradientCallback(),
        DelayedEarlyStoppingCallback(
            min_epochs=max(1, min(num_epochs, max(2, num_epochs // 2))),
            early_stopping_patience=2 if num_epochs <= 6 else 3,
            early_stopping_threshold=0.001,
        ),
    ]
    if balanced_sampler is not None:
        callbacks.append(SamplerEpochCallback(balanced_sampler))

    trainer = TokenTrainerWrapper(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=lambda pred: compute_span_metrics(
            pred,
            threshold=threshold_for_eval,
            active_label_mask=active_label_mask,
        ),
        optimizers=(optimizer, scheduler),
        callbacks=callbacks,
        balanced_batch_sampler=balanced_sampler,
    )

    print(
        f"[Stage:{stage_name}] train_examples={len(train_dataset)} "
        f"epochs={num_epochs} lr={learning_rate:.2e}"
    )
    if lr_info["enabled"]:
        print(
            f"[OptimizerLR] base_lr={lr_info['base_lr']:.2e} "
            f"encoder_lr={lr_info['encoder_lr'] if lr_info['encoder_lr'] is not None else 'inherit'} "
            f"head_lr={lr_info['head_lr'] if lr_info['head_lr'] is not None else 'inherit'} "
            f"n_encoder_params={lr_info['n_encoder_params']} "
            f"n_head_params={lr_info['n_head_params']} "
            f"n_base_params={lr_info['n_base_params']}"
        )
    trainer.train()
    return trainer, {
        "stage_name": stage_name,
        "epochs": int(num_epochs),
        "learning_rate": float(learning_rate),
        "train_examples": int(len(train_dataset)),
        "balanced_sampling": bool(balanced_sampler is not None),
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    init_model_ref = args.init_model_dir
    init_model_dir = Path(init_model_ref)
    label_list = load_label_list(init_model_dir)
    num_labels = len(label_list)
    ignored_labels, active_label_mask = parse_ignore_labels(args.ignore_labels, label_list)
    active_labels = [label for idx, label in enumerate(label_list) if active_label_mask[idx]]
    threshold_recall_priority_labels = parse_label_name_list(args.threshold_recall_priority_labels, label_list)
    threshold_priority_metric_by_label = {
        label: "fbeta_recall" for label in threshold_recall_priority_labels if label in active_labels
    }
    print(
        f"[Labels] total={num_labels} active={len(active_labels)} ignored={len(ignored_labels)}"
    )
    if ignored_labels:
        print(f"[Labels] ignored: {', '.join(ignored_labels)}")
    if threshold_recall_priority_labels:
        print(
            "[ThresholdObjective] recall-priority labels: "
            + ", ".join(threshold_recall_priority_labels)
            + f" (beta={args.threshold_recall_beta:.2f})"
        )

    tokenizer = AutoTokenizer.from_pretrained(init_model_ref)
    transcript_train_dataset, dev_dataset, test_dataset, processor = prepare_datasets(
        args.train_file,
        args.dev_file,
        args.test_file,
        label_list,
        tokenizer,
        max_length=args.max_length,
        ignore_canonical_labels=ignored_labels,
    )
    comments_train_dataset = None
    train_dataset = transcript_train_dataset
    balanced_sampler: Optional[BalancedDomainBatchSampler] = None
    n_comments = 0
    n_transcripts = len(transcript_train_dataset)
    mixed_steps_per_epoch = None
    stage_summaries: List[Dict[str, object]] = []
    effective_train_mode = args.train_mode

    if args.train_mode == "mixed" or args.use_curriculum:
        comments_path = Path(args.comments_train_file)
        if not comments_path.exists():
            raise FileNotFoundError(f"comments train file not found: {comments_path}")
        comments_train_dataset = prepare_single_dataset(
            str(comments_path),
            processor=processor,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        n_comments = len(comments_train_dataset)
        if n_comments == 0:
            raise ValueError(f"Comments dataset is empty: {comments_path}")
        if n_transcripts == 0 and (args.train_mode == "mixed" or args.use_curriculum):
            raise ValueError("Transcript train dataset is empty; cannot run mixed/curriculum mode.")

    if args.train_mode == "mixed" and not args.use_curriculum:
        if comments_train_dataset is None:
            raise ValueError("comments_train_dataset is required for mixed mode.")
        train_dataset = ConcatDataset([comments_train_dataset, transcript_train_dataset])
        comment_per_batch = max(1, min(args.batch_size - 1, int(round(args.batch_size * args.mix_comment_fraction))))
        transcript_per_batch = args.batch_size - comment_per_batch
        if transcript_per_batch <= 0:
            raise ValueError("mix-comment-fraction produces zero transcript items per batch.")

        if args.mixed_steps_per_epoch > 0:
            mixed_steps_per_epoch = args.mixed_steps_per_epoch
        else:
            mixed_steps_per_epoch = max(1, math.ceil(n_transcripts / transcript_per_batch))

        balanced_sampler = BalancedDomainBatchSampler(
            n_comments=n_comments,
            n_transcripts=n_transcripts,
            batch_size=args.batch_size,
            comment_fraction=args.mix_comment_fraction,
            steps_per_epoch=mixed_steps_per_epoch,
            seed=args.seed,
        )
        print(
            "[MixedTrain] "
            f"comments={n_comments}, transcripts={n_transcripts}, "
            f"batch={args.batch_size} ({balanced_sampler.comment_count} comments + "
            f"{balanced_sampler.transcript_count} transcripts), "
            f"steps/epoch={mixed_steps_per_epoch}"
        )
    elif not args.use_curriculum:
        print(f"[TrainMode] transcripts_only with {n_transcripts} transcript chunks")

    config = AutoConfig.from_pretrained(init_model_ref)
    apply_dropout_to_config(config, args.dropout_prob)
    config.multi_head_loss_normalization = args.multi_head_loss_normalization

    model = build_span_model(
        init_model_ref=init_model_ref,
        config=config,
        num_labels=num_labels,
        model_architecture=args.model_architecture,
        multi_head_dim=args.multi_head_dim,
    )
    print(
        f"[Model] architecture={args.model_architecture} "
        f"num_labels={num_labels} "
        + (
            f"head_dim={args.multi_head_dim} loss_norm={args.multi_head_loss_normalization}"
            if args.model_architecture == "multi_head"
            else ""
        )
    )
    load_encoder_from_mlm_checkpoint(model, args.encoder_init_from_mlm)
    if hasattr(model, "set_label_loss_mask"):
        model.set_label_loss_mask(torch.tensor(active_label_mask, dtype=torch.float32))
    freeze_bottom_layers(model, args.freeze_layers)

    threshold = args.threshold
    threshold_for_eval = args.eval_threshold if args.eval_threshold is not None else args.threshold
    print(f"[EvalThreshold] train_eval_threshold={threshold_for_eval:.2f} final_tuning_base={args.threshold:.2f}")
    if args.model_architecture == "multi_head" and not args.use_pos_weight:
        print(
            "[ConfigWarning] multi_head with pos_weight disabled can collapse to all-negative "
            "predictions on sparse labels."
        )
    if args.model_architecture == "multi_head" and threshold_for_eval >= 0.4:
        print(
            "[ConfigWarning] high eval threshold may hide useful early recall. "
            "Consider --eval-threshold in [0.20, 0.35]."
        )

    if args.use_curriculum:
        effective_train_mode = "curriculum_mixed_then_transcripts"
        if comments_train_dataset is None:
            raise ValueError("comments_train_dataset is required when --use-curriculum is set.")
        if args.stage_a_epochs <= 0 or args.stage_b_epochs <= 0:
            raise ValueError("stage-a-epochs and stage-b-epochs must be > 0 for curriculum training.")

        stage_a_dataset = ConcatDataset([comments_train_dataset, transcript_train_dataset])
        comment_per_batch = max(1, min(args.batch_size - 1, int(round(args.batch_size * args.mix_comment_fraction))))
        transcript_per_batch = args.batch_size - comment_per_batch
        if transcript_per_batch <= 0:
            raise ValueError("mix-comment-fraction produces zero transcript items per batch.")
        stage_a_steps = (
            args.mixed_steps_per_epoch
            if args.mixed_steps_per_epoch > 0
            else max(1, math.ceil(n_transcripts / transcript_per_batch))
        )
        mixed_steps_per_epoch = stage_a_steps
        stage_a_sampler = BalancedDomainBatchSampler(
            n_comments=n_comments,
            n_transcripts=n_transcripts,
            batch_size=args.batch_size,
            comment_fraction=args.mix_comment_fraction,
            steps_per_epoch=stage_a_steps,
            seed=args.seed,
        )
        print(
            "[Curriculum Stage A] "
            f"comments={n_comments}, transcripts={n_transcripts}, "
            f"batch={args.batch_size} ({stage_a_sampler.comment_count} comments + "
            f"{stage_a_sampler.transcript_count} transcripts), "
            f"steps/epoch={stage_a_steps}"
        )
        initialize_multi_head_biases(
            model=model,
            mode=args.multi_head_init_bias_mode,
            constant=args.multi_head_init_bias_constant,
            label_prior_floor=args.multi_head_label_prior_floor,
            train_dataset=stage_a_dataset,
            num_labels=num_labels,
            active_label_mask=active_label_mask,
        )
        if args.use_pos_weight:
            model.set_pos_weight(
                compute_pos_weight_from_dataset(
                    stage_a_dataset,
                    num_labels,
                    field="token_labels",
                    active_label_mask=active_label_mask,
                )
            )
        else:
            model.set_pos_weight(None)
            print("[PosWeight] disabled for stage_a_mixed")
        trainer, info_a = train_stage(
            stage_name="stage_a_mixed",
            model=model,
            train_dataset=stage_a_dataset,
            dev_dataset=dev_dataset,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.stage_a_learning_rate,
            weight_decay=args.weight_decay,
            num_epochs=args.stage_a_epochs,
            fp16=args.fp16,
            output_root=args.output_dir,
            threshold_for_eval=threshold_for_eval,
            active_label_mask=active_label_mask,
            balanced_sampler=stage_a_sampler,
            encoder_learning_rate=args.encoder_learning_rate,
            head_learning_rate=args.head_learning_rate,
        )
        stage_summaries.append(info_a)

        print(f"[Curriculum Stage B] transcripts_only chunks={n_transcripts}")
        if args.use_pos_weight:
            model.set_pos_weight(
                compute_pos_weight_from_dataset(
                    transcript_train_dataset,
                    num_labels,
                    field="token_labels",
                    active_label_mask=active_label_mask,
                )
            )
        else:
            model.set_pos_weight(None)
            print("[PosWeight] disabled for stage_b_transcripts")
        trainer, info_b = train_stage(
            stage_name="stage_b_transcripts",
            model=model,
            train_dataset=transcript_train_dataset,
            dev_dataset=dev_dataset,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.stage_b_learning_rate,
            weight_decay=args.weight_decay,
            num_epochs=args.stage_b_epochs,
            fp16=args.fp16,
            output_root=args.output_dir,
            threshold_for_eval=threshold_for_eval,
            active_label_mask=active_label_mask,
            balanced_sampler=None,
            encoder_learning_rate=args.encoder_learning_rate,
            head_learning_rate=args.head_learning_rate,
        )
        stage_summaries.append(info_b)
    else:
        if args.train_mode == "mixed" and comments_train_dataset is not None:
            bias_dataset = ConcatDataset([comments_train_dataset, transcript_train_dataset])
        else:
            bias_dataset = train_dataset
        initialize_multi_head_biases(
            model=model,
            mode=args.multi_head_init_bias_mode,
            constant=args.multi_head_init_bias_constant,
            label_prior_floor=args.multi_head_label_prior_floor,
            train_dataset=bias_dataset,
            num_labels=num_labels,
            active_label_mask=active_label_mask,
        )
        if args.use_pos_weight:
            model.set_pos_weight(
                compute_pos_weight_from_dataset(
                    train_dataset,
                    num_labels,
                    field="token_labels",
                    active_label_mask=active_label_mask,
                )
            )
        else:
            model.set_pos_weight(None)
            print("[PosWeight] disabled")
        trainer, single_info = train_stage(
            stage_name="single_stage",
            model=model,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            fp16=args.fp16,
            output_root=args.output_dir,
            threshold_for_eval=threshold_for_eval,
            active_label_mask=active_label_mask,
            balanced_sampler=balanced_sampler,
            encoder_learning_rate=args.encoder_learning_rate,
            head_learning_rate=args.head_learning_rate,
        )
        stage_summaries.append(single_info)

    threshold_grid = [
        round(x, 4)
        for x in np.arange(
            args.threshold_grid_min,
            args.threshold_grid_max + 1e-9,
            args.threshold_grid_step,
        )
    ]
    if args.threshold_mode == "per_label":
        threshold, per_label_tuning = tune_token_thresholds_per_label_on_dev(
            trainer=trainer,
            dev_dataset=dev_dataset,
            label_list=label_list,
            base_threshold=args.threshold,
            grid=threshold_grid,
            min_support_per_label=args.min_support_per_label,
            active_label_mask=active_label_mask,
            priority_metric_by_label=threshold_priority_metric_by_label,
            recall_priority_beta=args.threshold_recall_beta,
        )
        tuned_dev_f1 = float(
            np.mean([per_label_tuning[label_name]["f1"] for idx, label_name in enumerate(label_list) if active_label_mask[idx]])
        )
        print(f"[Threshold] mode=per_label tuned_dev_macro_f1={tuned_dev_f1:.4f}")
        for label_idx, label_name in enumerate(label_list):
            if not active_label_mask[label_idx]:
                continue
            info = per_label_tuning[label_name]
            print(
                f"  {label_name}: thr={info['threshold']:.2f} "
                f"objective={info['objective_name']}:{info['objective_score']:.4f} "
                f"f1={info['f1']:.4f} p={info['precision']:.4f} "
                f"r={info['recall']:.4f} support={int(info['support'])}"
            )
    else:
        tuned_threshold, tuned_dev_f1 = tune_token_threshold_on_dev(
            trainer,
            dev_dataset,
            base_threshold=args.threshold,
            grid=threshold_grid,
            active_label_mask=active_label_mask,
        )
        threshold = tuned_threshold
        print(f"[Threshold] mode=global tuned={tuned_threshold:.2f} dev_f1_macro={tuned_dev_f1:.4f}")
        per_label_tuning = {}

    trainer.compute_metrics = lambda pred: compute_span_metrics(
        pred,
        threshold=threshold,
        active_label_mask=active_label_mask,
    )
    test_results = trainer.evaluate(test_dataset)
    test_predictions = trainer.predict(test_dataset)
    test_token_labels, test_attention_mask, test_supervision_mask = unpack_label_payload(test_predictions.label_ids)
    test_per_label = compute_token_per_label_metrics(
        test_predictions.predictions,
        test_token_labels,
        attention_mask=test_attention_mask,
        supervision_mask=test_supervision_mask,
        label_list=label_list,
        threshold=threshold,
        active_label_mask=active_label_mask,
    )

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with (Path(args.output_dir) / "label_list.json").open("w", encoding="utf-8") as f:
        json.dump(label_list, f, indent=2, ensure_ascii=False)

    saved_threshold = (
        float(threshold)
        if np.isscalar(threshold)
        else [float(x) for x in np.array(threshold).astype(float).tolist()]
    )
    training_config = {
        "def_com_trans_pipeline_version": TRANSCRIPT_SPAN_PIPELINE_VERSION,
        "train_file": args.train_file,
        "dev_file": args.dev_file,
        "test_file": args.test_file,
        "comments_train_file": args.comments_train_file,
        "train_mode": args.train_mode,
        "effective_train_mode": effective_train_mode,
        "use_curriculum": bool(args.use_curriculum),
        "stage_a_epochs": args.stage_a_epochs,
        "stage_b_epochs": args.stage_b_epochs,
        "stage_a_learning_rate": args.stage_a_learning_rate,
        "stage_b_learning_rate": args.stage_b_learning_rate,
        "ignore_labels": ignored_labels,
        "active_labels": active_labels,
        "num_active_labels": int(np.sum(active_label_mask)),
        "stage_summaries": stage_summaries,
        "mix_comment_fraction": args.mix_comment_fraction,
        "mixed_steps_per_epoch": mixed_steps_per_epoch,
        "n_train_comments": n_comments,
        "n_train_transcript_chunks": n_transcripts,
        "init_model_dir": str(init_model_dir),
        "encoder_init_from_mlm": args.encoder_init_from_mlm,
        "model_architecture": args.model_architecture,
        "multi_head_dim": args.multi_head_dim,
        "multi_head_loss_normalization": args.multi_head_loss_normalization,
        "multi_head_init_bias_mode": args.multi_head_init_bias_mode,
        "multi_head_init_bias_constant": args.multi_head_init_bias_constant,
        "multi_head_label_prior_floor": args.multi_head_label_prior_floor,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "encoder_learning_rate": args.encoder_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "dropout_prob": args.dropout_prob,
        "eval_threshold": args.eval_threshold,
        "effective_eval_threshold": threshold_for_eval,
        "use_pos_weight": bool(args.use_pos_weight),
        "threshold_mode": args.threshold_mode,
        "threshold": saved_threshold,
        "threshold_grid_min": args.threshold_grid_min,
        "threshold_grid_max": args.threshold_grid_max,
        "threshold_grid_step": args.threshold_grid_step,
        "min_support_per_label": args.min_support_per_label,
        "threshold_recall_priority_labels": threshold_recall_priority_labels,
        "threshold_recall_beta": args.threshold_recall_beta,
        "focus_only_supervision": True,
        "seed": args.seed,
    }
    with (Path(args.output_dir) / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2, ensure_ascii=False)

    serializable_test = {
        k: float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v
        for k, v in test_results.items()
    }
    with (Path(args.output_dir) / "test_results.json").open("w", encoding="utf-8") as f:
        json.dump(serializable_test, f, indent=2, ensure_ascii=False)
    label_thresholds = {}
    if np.isscalar(threshold):
        label_thresholds = {label: float(threshold) for label in label_list}
    else:
        threshold_vec = np.array(threshold).astype(float).tolist()
        label_thresholds = {label: float(threshold_vec[idx]) for idx, label in enumerate(label_list)}
    with (Path(args.output_dir) / "label_thresholds.json").open("w", encoding="utf-8") as f:
        json.dump(label_thresholds, f, indent=2, ensure_ascii=False)
    if per_label_tuning:
        with (Path(args.output_dir) / "dev_per_label_threshold_tuning.json").open("w", encoding="utf-8") as f:
            json.dump(per_label_tuning, f, indent=2, ensure_ascii=False)
    with (Path(args.output_dir) / "test_per_label_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_per_label, f, indent=2, ensure_ascii=False)

    print("Test metrics:")
    for key in sorted(serializable_test):
        print(f"  {key}: {serializable_test[key]}")
    print("Test per-technique F1 (support>0):")
    for label_name in active_labels:
        m = test_per_label[label_name]
        if m["support"] <= 0:
            continue
        print(
            f"  {label_name}: f1={m['f1']:.4f} "
            f"p={m['precision']:.4f} r={m['recall']:.4f} support={m['support']}"
        )
    print(f"Saved model and metrics to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
