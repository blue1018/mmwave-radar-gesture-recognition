from __future__ import annotations

import gc
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# -----------------------------
# Training utilities and metrics
# -----------------------------

def cleanup_accelerator(model: nn.Module | None = None) -> None:
    # Release accelerator memory between runs to reduce MPS allocation failures.
    if model is not None:
        model.to("cpu")
        del model
    gc.collect()
    if torch.backends.mps.is_available() and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def format_seconds(seconds: float) -> str:
    # Compact elapsed-time text for progress logs.
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def should_log_batch(batch_idx: int, total_batches: int, progress_every_n_batches: int) -> bool:
    # Log the first, last, and every configured batch interval.
    interval = max(1, progress_every_n_batches)
    return batch_idx == 1 or batch_idx == total_batches or batch_idx % interval == 0


def build_progress_event(
    progress_log_rows: list[dict[str, object]],
    display_names: dict[str, str],
    model_name: str,
    phase: str,
    batch_idx: int,
    total_batches: int,
    running_loss: float,
    elapsed_seconds: float,
    model_index: int | None = None,
    total_models: int | None = None,
    model_work_offset: int = 0,
    model_work_total: int | None = None,
    epoch: int | None = None,
    total_epochs: int | None = None,
    learning_rate: float | None = None,
    print_live: bool = False,
) -> dict[str, object]:
    # Create one structured progress event for live prints and final tables.
    phase_progress_pct = 100.0 * batch_idx / max(1, total_batches)
    model_position = f"{model_index}/{total_models}" if model_index is not None and total_models is not None else ""
    model_progress_pct = None
    overall_progress_pct = None
    if model_work_total:
        completed_work_units = min(model_work_offset + batch_idx, model_work_total)
        model_progress_pct = 100.0 * completed_work_units / max(1, model_work_total)
    elif model_index is not None and total_models is not None:
        model_progress_pct = phase_progress_pct
    if model_progress_pct is not None and model_index is not None and total_models is not None:
        overall_progress_pct = 100.0 * ((model_index - 1) + model_progress_pct / 100.0) / max(1, total_models)

    return {
        "event_id": len(progress_log_rows) + 1,
        "model": display_names[model_name],
        "model_key": model_name,
        "model_index": model_index,
        "total_models": total_models,
        "model_position": model_position,
        "phase": phase,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "batch": batch_idx,
        "total_batches": total_batches,
        "progress_pct": round(phase_progress_pct, 1),
        "model_progress_pct": None if model_progress_pct is None else round(model_progress_pct, 1),
        "overall_progress_pct": None if overall_progress_pct is None else round(overall_progress_pct, 1),
        "loss": round(float(running_loss), 4),
        "lr": None if learning_rate is None else float(learning_rate),
        "elapsed_seconds": round(float(elapsed_seconds), 2),
        "elapsed": format_seconds(float(elapsed_seconds)),
        "printed_live": bool(print_live),
    }


def progress_live_message(event: dict[str, object], running_loss: float, learning_rate: float | None = None) -> str:
    # Format a compact progress message that includes overall and per-model progress.
    epoch = event.get("epoch")
    total_epochs = event.get("total_epochs")
    epoch_text = f" | epoch {epoch}/{total_epochs}" if epoch is not None and total_epochs is not None else ""
    lr_text = f" | lr {learning_rate:.2e}" if learning_rate is not None else ""
    if event.get("overall_progress_pct") is not None and event.get("model_progress_pct") is not None:
        progress_text = (
            f"[overall {event['overall_progress_pct']:.1f}% | "
            f"model {event['model_position']} {event['model_progress_pct']:.1f}%] "
        )
    elif event.get("model_position"):
        progress_text = f"[model {event['model_position']}] "
    else:
        progress_text = ""
    return (
        f"{progress_text}[{event['phase']}] {event['model']}{epoch_text} | "
        f"batch {event['batch']}/{event['total_batches']} ({event['progress_pct']:.1f}%) | "
        f"loss {running_loss:.4f}{lr_text} | elapsed {event['elapsed']}"
    )


def progress_phase_summary_frame(progress_df: pd.DataFrame) -> pd.DataFrame:
    # Summarize recorded progress events by model, phase, and epoch.
    if progress_df.empty:
        return pd.DataFrame()
    df = progress_df.copy()
    df["epoch_label"] = df["epoch"].fillna("")
    rows = []
    for (model, phase, epoch_label), group in df.groupby(["model", "phase", "epoch_label"], sort=False):
        final_event = group.iloc[-1]
        rows.append(
            {
                "model": model,
                "model_position": final_event.get("model_position", ""),
                "phase": phase,
                "epoch": epoch_label,
                "recorded_events": int(len(group)),
                "printed_live_events": int(group["printed_live"].sum()),
                "final_batch": int(final_event["batch"]),
                "total_batches": int(final_event["total_batches"]),
                "final_phase_progress_pct": float(final_event["progress_pct"]),
                "final_model_progress_pct": final_event.get("model_progress_pct", None),
                "final_overall_progress_pct": final_event.get("overall_progress_pct", None),
                "final_loss": float(final_event["loss"]),
                "elapsed": final_event["elapsed"],
                "elapsed_seconds": float(final_event["elapsed_seconds"]),
            }
        )
    return pd.DataFrame(rows)


def progress_tail_frame(progress_df: pd.DataFrame, tail_rows: int) -> pd.DataFrame:
    # Return recent progress events with the most useful monitoring columns.
    if progress_df.empty:
        return pd.DataFrame()
    columns = [
        "event_id",
        "model_position",
        "overall_progress_pct",
        "model_progress_pct",
        "model",
        "phase",
        "epoch",
        "batch",
        "total_batches",
        "progress_pct",
        "loss",
        "lr",
        "elapsed",
        "printed_live",
    ]
    return progress_df[columns].tail(tail_rows).reset_index(drop=True)


def classification_loss(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    # Compute cross-entropy on CPU when required by the PyTorch 2.0.1 MPS backend.
    if logits.device.type == "mps":
        return F.cross_entropy(logits.float().cpu(), targets.cpu(), label_smoothing=label_smoothing)
    return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)


def ensure_finite_tensor(
    tensor: torch.Tensor,
    name: str,
    model_name: str,
    phase: str,
    epoch: int | None = None,
    batch_idx: int | None = None,
) -> None:
    checked = tensor.detach()
    if checked.device.type != "cpu":
        checked = checked.float().cpu()
    if bool(torch.isfinite(checked).all().item()):
        return
    checked = checked.float()
    nan_count = int(torch.isnan(checked).sum().item())
    inf_count = int(torch.isinf(checked).sum().item())
    location = []
    if epoch is not None:
        location.append(f"epoch={epoch}")
    if batch_idx is not None:
        location.append(f"batch={batch_idx}")
    location_text = f" ({', '.join(location)})" if location else ""
    raise FloatingPointError(
        f"Non-finite {name} detected for {model_name} during {phase}{location_text}: "
        f"nan_count={nan_count}, inf_count={inf_count}"
    )


def clip_grad_norm_cpu_checked(
    model: nn.Module,
    max_norm: float,
) -> tuple[torch.Tensor, list[str]]:
    total_sq_norm = 0.0
    bad_gradients: list[str] = []
    parameters_with_grad = []

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        parameters_with_grad.append(parameter)
        grad_cpu = parameter.grad.detach().float().cpu()
        if not bool(torch.isfinite(grad_cpu).all().item()):
            bad_gradients.append(name)
            continue
        total_sq_norm += float(torch.sum(grad_cpu.double() * grad_cpu.double()).item())

    total_norm = math.sqrt(total_sq_norm)
    if not math.isfinite(total_norm):
        return torch.tensor(float("inf")), bad_gradients

    clip_coef = float(max_norm) / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for parameter in parameters_with_grad:
            parameter.grad.mul_(clip_coef)
    return torch.tensor(total_norm), bad_gradients


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    enabled: bool,
):
    # Per-batch warmup + cosine schedule for Transformer-family training.
    if not enabled or total_steps <= 1:
        return None
    warmup_steps = min(max(0, warmup_steps), max(0, total_steps - 1))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-3, float(step + 1) / float(warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def compute_metrics(
    losses: list[float],
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    num_classes: int,
) -> dict[str, float | list[float] | list[int]]:
    labels = np.arange(num_classes)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return {
        "loss": float(np.mean(losses)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "per_class_precision": [float(value) for value in precision],
        "per_class_recall": [float(value) for value in recall],
        "per_class_f1": [float(value) for value in per_class_f1],
        "per_class_support": [int(value) for value in support],
    }


def per_class_metrics_frame(metrics: dict[str, float | list[float] | list[int]], class_names) -> pd.DataFrame:
    # Return a readable per-class table for notebook display.
    return pd.DataFrame(
        {
            "label": class_names,
            "precision": metrics["per_class_precision"],
            "recall": metrics["per_class_recall"],
            "f1": metrics["per_class_f1"],
            "support": metrics["per_class_support"],
        }
    )


def plot_confusion_matrix(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names,
    display_name: str,
) -> None:
    # Show confusion matrix inline; no image file is written.
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{display_name} Test Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)

    fig.tight_layout()
    plt.show()


# -----------------------------
# Checkpoint state helpers
# -----------------------------


def clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    # Keep the best checkpoint in memory, not on disk.
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
