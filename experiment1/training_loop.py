from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from cache_manager import (
    can_load_result_cache,
    can_resume_checkpoint,
    load_checkpoint,
    load_result_cache,
    model_cache_key,
    model_cache_paths,
    save_model_cache,
)
from data_pipeline import make_loader
from experiment_config import (
    KD_STUDENT_MODELS,
    MODEL_DISPLAY_NAMES,
    PAPER_KD_TEACHER_MODEL,
    TRANSFORMER_TRAINING_MODELS,
    USE_TRANSFORMER_AUGMENTATION,
    USE_TRANSFORMER_MOTION_INPUT,
    config_summary,
    model_training_profile,
)
from model_zoo import augment_transformer_batch, build_gesture_model, count_parameters
from training_tools import (
    build_progress_event,
    build_warmup_cosine_scheduler,
    classification_loss,
    clip_grad_norm_cpu_checked,
    cleanup_accelerator,
    clone_state_dict_to_cpu,
    compute_metrics,
    ensure_finite_tensor,
    format_seconds,
    progress_live_message,
    progress_phase_summary_frame,
    progress_tail_frame,
    should_log_batch,
)

# -----------------------------
# Shared training/evaluation loop
# -----------------------------


@dataclass
class ExperimentRun:
    # Container for the outputs of one complete experiment run.
    results_df: pd.DataFrame
    histories: dict[str, pd.DataFrame]
    test_details: dict[str, dict[str, object]]
    failed_models: list[dict[str, object]]
    total_seconds: float
    progress_df: pd.DataFrame
    progress_summary_df: pd.DataFrame
    progress_tail_df: pd.DataFrame
    smoke_df: pd.DataFrame


class ProgressRecorder:
    # Record progress without relying on global state.
    def __init__(self, config):
        self.config = config
        self.rows: list[dict[str, object]] = []
        self.model_index: int | None = None
        self.total_models: int | None = None
        self.model_work_offset: int = 0
        self.model_work_total: int | None = None

    def start_model(self, model_index: int | None, total_models: int | None, work_total: int | None = None) -> None:
        self.model_index = model_index
        self.total_models = total_models
        self.model_work_offset = 0
        self.model_work_total = work_total

    def record(
        self,
        model_name: str,
        phase: str,
        batch_idx: int,
        total_batches: int,
        running_loss: float,
        elapsed_seconds: float,
        epoch: int | None = None,
        total_epochs: int | None = None,
        learning_rate: float | None = None,
        print_live: bool = False,
    ) -> None:
        event = build_progress_event(
            progress_log_rows=self.rows,
            display_names=MODEL_DISPLAY_NAMES,
            model_name=model_name,
            phase=phase,
            batch_idx=batch_idx,
            total_batches=total_batches,
            running_loss=running_loss,
            elapsed_seconds=elapsed_seconds,
            model_index=self.model_index,
            total_models=self.total_models,
            model_work_offset=self.model_work_offset,
            model_work_total=self.model_work_total,
            epoch=epoch,
            total_epochs=total_epochs,
            learning_rate=learning_rate,
            print_live=print_live,
        )
        self.rows.append(event)
        if print_live:
            print(progress_live_message(event, running_loss, learning_rate), flush=True)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def summary_frame(self) -> pd.DataFrame:
        progress_df = self.frame()
        return progress_phase_summary_frame(progress_df) if not progress_df.empty else pd.DataFrame()

    def tail_frame(self) -> pd.DataFrame:
        progress_df = self.frame()
        return progress_tail_frame(progress_df, self.config.progress_log_tail_rows) if not progress_df.empty else pd.DataFrame()


def build_model(model_name: str, config) -> nn.Module:
    # All model constructors receive the same input and class dimensions.
    return build_gesture_model(
        model_name,
        seq_len=config.seq_len,
        image_size=config.image_size,
        num_classes=config.num_classes,
        motion_input_enabled=USE_TRANSFORMER_MOTION_INPUT,
    )


def forward_batch(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    config,
    model_name: str | None = None,
    augmentation_enabled: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    x = x.to(config.device)
    y = y.to(config.device)
    augmentation_enabled = USE_TRANSFORMER_AUGMENTATION if augmentation_enabled is None else augmentation_enabled
    if model.training and torch.is_grad_enabled():
        x = augment_transformer_batch(
            x,
            model_name,
            transformer_model_names=TRANSFORMER_TRAINING_MODELS,
            enabled=augmentation_enabled,
        )
    return model(x), y


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    alpha: float,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # The paper uses teacher soft logits plus ordinary CE on hard labels.
    temperature = max(float(temperature), 1e-6)
    alpha = min(1.0, max(0.0, float(alpha)))
    if student_logits.device.type == "mps":
        student_for_loss = student_logits.float().cpu()
        teacher_for_loss = teacher_logits.float().cpu()
        targets_for_loss = targets.cpu()
    else:
        student_for_loss = student_logits
        teacher_for_loss = teacher_logits
        targets_for_loss = targets

    soft_loss = F.kl_div(
        F.log_softmax(student_for_loss / temperature, dim=1),
        F.softmax(teacher_for_loss / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)
    hard_loss = F.cross_entropy(student_for_loss, targets_for_loss, label_smoothing=label_smoothing)
    total_loss = alpha * soft_loss + (1.0 - alpha) * hard_loss
    return total_loss, hard_loss.detach(), soft_loss.detach()


def teacher_cache_for_split(
    config,
    train_files,
    train_labels,
    val_files,
    val_labels,
    test_files,
    test_labels,
    teacher_model_name: str = PAPER_KD_TEACHER_MODEL,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    teacher_profile = model_training_profile(teacher_model_name, config)
    teacher_cache_key = model_cache_key(
        config,
        teacher_model_name,
        teacher_profile,
        train_files,
        train_labels,
        val_files,
        val_labels,
        test_files,
        test_labels,
    )
    teacher_paths = model_cache_paths(config, teacher_model_name, teacher_cache_key)
    return teacher_profile, teacher_paths, {"teacher_cache_key": teacher_cache_key}


def load_distillation_teacher(
    config,
    train_files,
    train_labels,
    val_files,
    val_labels,
    test_files,
    test_labels,
    teacher_model_name: str = PAPER_KD_TEACHER_MODEL,
) -> tuple[nn.Module, dict[str, object]]:
    teacher_profile, teacher_paths, teacher_metadata = teacher_cache_for_split(
        config,
        train_files,
        train_labels,
        val_files,
        val_labels,
        test_files,
        test_labels,
        teacher_model_name=teacher_model_name,
    )
    if not teacher_paths["checkpoint"].exists():
        raise ValueError(
            f"KD teacher checkpoint not found: {teacher_paths['checkpoint']}. "
            f"Run {MODEL_DISPLAY_NAMES[teacher_model_name]} before the distilled student."
        )
    teacher = build_model(teacher_model_name, config).to(config.device)
    teacher.load_state_dict(load_checkpoint(teacher_paths))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    metadata = {
        **teacher_metadata,
        "teacher_model": teacher_model_name,
        "teacher_display_name": MODEL_DISPLAY_NAMES[teacher_model_name],
        "teacher_training_profile": teacher_profile,
        "teacher_checkpoint": str(teacher_paths["checkpoint"]),
    }
    return teacher, metadata


def add_distillation_result_fields(result: dict[str, object], training_profile: dict[str, object]) -> dict[str, object]:
    if not training_profile.get("distillation"):
        return result
    result.update(
        {
            "distilled_from": training_profile["teacher_model"],
            "kd_temperature": float(training_profile["kd_temperature"]),
            "kd_alpha": float(training_profile["kd_alpha"]),
        }
    )
    return result


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    epoch: int,
    total_epochs: int,
    config,
    recorder: ProgressRecorder,
    label_smoothing: float = 0.0,
    clip_grad_norm: float | None = None,
    scheduler=None,
    augmentation_enabled: bool | None = None,
    cpu_grad_clip: bool = False,
    skip_nonfinite_grad_steps: bool = False,
) -> dict[str, float]:
    model.train()
    losses, all_true, all_pred = [], [], []
    skipped_grad_steps = 0
    total_batches = len(loader)
    epoch_start = time.time()

    for batch_idx, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        logits, y = forward_batch(
            model,
            batch,
            config,
            model_name=model_name,
            augmentation_enabled=augmentation_enabled,
        )
        ensure_finite_tensor(logits, "logits", model_name, "train", epoch=epoch, batch_idx=batch_idx)
        loss = classification_loss(logits, y, label_smoothing=label_smoothing)
        ensure_finite_tensor(loss, "loss", model_name, "train", epoch=epoch, batch_idx=batch_idx)
        loss.backward()
        if clip_grad_norm is not None:
            if cpu_grad_clip:
                grad_norm, bad_gradients = clip_grad_norm_cpu_checked(model, max_norm=float(clip_grad_norm))
                if bad_gradients:
                    if skip_nonfinite_grad_steps:
                        optimizer.zero_grad(set_to_none=True)
                        skipped_grad_steps += 1
                        losses.append(loss.item())
                        preds = logits.argmax(dim=1)
                        all_true.extend(y.cpu().numpy().tolist())
                        all_pred.extend(preds.cpu().numpy().tolist())
                        continue
                    bad_names = ", ".join(bad_gradients[:5])
                    suffix = "" if len(bad_gradients) <= 5 else f", ... ({len(bad_gradients)} total)"
                    raise FloatingPointError(
                        f"Non-finite gradients detected for {model_name} during train "
                        f"(epoch={epoch}, batch={batch_idx}): {bad_names}{suffix}"
                    )
                if not bool(torch.isfinite(grad_norm).all().item()):
                    if skip_nonfinite_grad_steps:
                        optimizer.zero_grad(set_to_none=True)
                        skipped_grad_steps += 1
                        losses.append(loss.item())
                        preds = logits.argmax(dim=1)
                        all_true.extend(y.cpu().numpy().tolist())
                        all_pred.extend(preds.cpu().numpy().tolist())
                        continue
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad_norm))
            ensure_finite_tensor(grad_norm, "gradient norm", model_name, "train", epoch=epoch, batch_idx=batch_idx)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        preds = logits.argmax(dim=1)
        all_true.extend(y.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())

        record_event = should_log_batch(batch_idx, total_batches, config.progress_every_n_batches)
        print_event = should_log_batch(batch_idx, total_batches, config.print_progress_every_n_batches)
        if record_event or print_event:
            recorder.record(
                model_name=model_name,
                phase="train",
                batch_idx=batch_idx,
                total_batches=total_batches,
                running_loss=float(np.mean(losses)),
                elapsed_seconds=time.time() - epoch_start,
                epoch=epoch,
                total_epochs=total_epochs,
                learning_rate=optimizer.param_groups[0]["lr"],
                print_live=print_event,
            )
    metrics = compute_metrics(losses, all_true, all_pred, config.num_classes)
    metrics["skipped_grad_steps"] = skipped_grad_steps
    return metrics


def train_one_epoch_distilled(
    student: nn.Module,
    teacher: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    epoch: int,
    total_epochs: int,
    config,
    recorder: ProgressRecorder,
    temperature: float,
    alpha: float,
    label_smoothing: float = 0.0,
    clip_grad_norm: float | None = None,
    scheduler=None,
    augmentation_enabled: bool | None = None,
    cpu_grad_clip: bool = False,
    skip_nonfinite_grad_steps: bool = False,
) -> dict[str, float]:
    student.train()
    teacher.eval()
    losses, hard_losses, soft_losses, all_true, all_pred = [], [], [], [], []
    skipped_grad_steps = 0
    total_batches = len(loader)
    epoch_start = time.time()

    for batch_idx, batch in enumerate(loader, start=1):
        x, y = batch
        x = x.to(config.device)
        y = y.to(config.device)
        if torch.is_grad_enabled():
            x = augment_transformer_batch(
                 x,
                 model_name,
                 transformer_model_names=TRANSFORMER_TRAINING_MODELS,
                 enabled=USE_TRANSFORMER_AUGMENTATION if augmentation_enabled is None else augmentation_enabled,
             )

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits = teacher(x)
        ensure_finite_tensor(teacher_logits, "teacher logits", model_name, "distill-train", epoch=epoch, batch_idx=batch_idx)
        student_logits = student(x)
        ensure_finite_tensor(student_logits, "student logits", model_name, "distill-train", epoch=epoch, batch_idx=batch_idx)
        loss, hard_loss, soft_loss = distillation_loss(
            student_logits,
            teacher_logits,
            y,
            temperature=temperature,
            alpha=alpha,
            label_smoothing=label_smoothing,
        )
        ensure_finite_tensor(loss, "loss", model_name, "distill-train", epoch=epoch, batch_idx=batch_idx)
        ensure_finite_tensor(hard_loss, "hard loss", model_name, "distill-train", epoch=epoch, batch_idx=batch_idx)
        ensure_finite_tensor(soft_loss, "soft loss", model_name, "distill-train", epoch=epoch, batch_idx=batch_idx)
        loss.backward()
        if clip_grad_norm is not None:
            if cpu_grad_clip:
                grad_norm, bad_gradients = clip_grad_norm_cpu_checked(student, max_norm=float(clip_grad_norm))
                if bad_gradients or not bool(torch.isfinite(grad_norm).all().item()):
                    if skip_nonfinite_grad_steps:
                        optimizer.zero_grad(set_to_none=True)
                        skipped_grad_steps += 1
                        losses.append(float(loss.detach().cpu()))
                        hard_losses.append(float(hard_loss.cpu()))
                        soft_losses.append(float(soft_loss.cpu()))
                        preds = student_logits.argmax(dim=1)
                        all_true.extend(y.cpu().numpy().tolist())
                        all_pred.extend(preds.cpu().numpy().tolist())
                        continue
                    bad_names = ", ".join(bad_gradients[:5])
                    suffix = "" if len(bad_gradients) <= 5 else f", ... ({len(bad_gradients)} total)"
                    raise FloatingPointError(
                        f"Non-finite gradients detected for {model_name} during distill-train "
                        f"(epoch={epoch}, batch={batch_idx}): {bad_names}{suffix}"
                    )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=float(clip_grad_norm))
            ensure_finite_tensor(grad_norm, "gradient norm", model_name, "distill-train", epoch=epoch, batch_idx=batch_idx)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(float(loss.detach().cpu()))
        hard_losses.append(float(hard_loss.cpu()))
        soft_losses.append(float(soft_loss.cpu()))
        preds = student_logits.argmax(dim=1)
        all_true.extend(y.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())

        record_event = should_log_batch(batch_idx, total_batches, config.progress_every_n_batches)
        print_event = should_log_batch(batch_idx, total_batches, config.print_progress_every_n_batches)
        if record_event or print_event:
            recorder.record(
                model_name=model_name,
                phase="distill-train",
                batch_idx=batch_idx,
                total_batches=total_batches,
                running_loss=float(np.mean(losses)),
                elapsed_seconds=time.time() - epoch_start,
                epoch=epoch,
                total_epochs=total_epochs,
                learning_rate=optimizer.param_groups[0]["lr"],
                print_live=print_event,
            )

    metrics = compute_metrics(losses, all_true, all_pred, config.num_classes)
    metrics["hard_loss"] = float(np.mean(hard_losses))
    metrics["soft_loss"] = float(np.mean(soft_losses))
    metrics["skipped_grad_steps"] = skipped_grad_steps
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    split_name: str,
    model_name: str,
    config,
    recorder: ProgressRecorder,
    label_smoothing: float = 0.0,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    losses, all_true, all_pred = [], [], []
    total_batches = len(loader)
    eval_start = time.time()

    for batch_idx, batch in enumerate(loader, start=1):
        logits, y = forward_batch(model, batch, config, model_name=model_name)
        ensure_finite_tensor(logits, "logits", model_name, split_name, batch_idx=batch_idx)
        loss = classification_loss(logits, y, label_smoothing=label_smoothing)
        ensure_finite_tensor(loss, "loss", model_name, split_name, batch_idx=batch_idx)
        preds = logits.argmax(dim=1)

        losses.append(loss.item())
        all_true.extend(y.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())

        record_event = should_log_batch(batch_idx, total_batches, config.progress_every_n_batches)
        print_event = should_log_batch(batch_idx, total_batches, config.print_progress_every_n_batches)
        if record_event or print_event:
            recorder.record(
                model_name=model_name,
                phase=split_name,
                batch_idx=batch_idx,
                total_batches=total_batches,
                running_loss=float(np.mean(losses)),
                elapsed_seconds=time.time() - eval_start,
                print_live=print_event,
            )
    return compute_metrics(losses, all_true, all_pred, config.num_classes), np.array(all_true), np.array(all_pred)


def run_forward_smoke(config, train_files, train_labels) -> pd.DataFrame:
    # Validate tensor shapes for every selected model without full training.
    rows = []
    total_models = len(config.models)
    print("Smoke test: one forward pass per model", flush=True)
    for model_index, model_name in enumerate(config.models, start=1):
        display_name = MODEL_DISPLAY_NAMES[model_name]
        print(f"[{model_index}/{total_models}] {display_name}", flush=True)
        model = build_model(model_name, config).to(config.device)
        loader = make_loader(config.data_path, train_files, train_labels, config.seq_len, model_name, shuffle=False)
        sample_batch = next(iter(loader))
        with torch.no_grad():
            logits, _ = forward_batch(model, sample_batch, config, model_name=model_name)
        ensure_finite_tensor(logits, "logits", model_name, "forward-smoke")
        rows.append(
            {
                "model": model_name,
                "display_name": display_name,
                "params_m": round(count_parameters(model), 3),
                "logits_shape": tuple(logits.shape),
            }
        )
        print(f"[smoke] {display_name} ok | logits={tuple(logits.shape)}", flush=True)
        cleanup_accelerator(model)
    return pd.DataFrame(rows)


def run_single_model(
    model_name: str,
    config,
    train_files,
    train_labels,
    val_files,
    val_labels,
    test_files,
    test_labels,
    recorder: ProgressRecorder,
    model_index: int | None = None,
    total_models: int | None = None,
) -> tuple[dict[str, float | str], list[dict[str, float]], dict[str, object]]:
    # Train one model and return its summary, history, and test details.
    display_name = MODEL_DISPLAY_NAMES[model_name]
    prefix = f"[{model_index}/{total_models}] " if model_index is not None and total_models is not None else ""
    print(f"\n=== {prefix}Running {display_name} on {config.device} ===", flush=True)

    training_profile = model_training_profile(model_name, config)
    target_epochs = int(training_profile["epochs"])
    cache_key = model_cache_key(
        config,
        model_name,
        training_profile,
        train_files,
        train_labels,
        val_files,
        val_labels,
        test_files,
        test_labels,
    )
    cache_paths = model_cache_paths(config, model_name, cache_key)
    cache_metadata = {
        "model_name": model_name,
        "display_name": display_name,
        "cache_key": cache_key,
        "run_mode": config.run_mode,
        "training_profile": training_profile,
    }
    if training_profile.get("distillation"):
        cache_metadata["distillation"] = {
            "teacher_model": training_profile["teacher_model"],
            "teacher_display_name": MODEL_DISPLAY_NAMES[training_profile["teacher_model"]],
            "kd_temperature": training_profile["kd_temperature"],
            "kd_alpha": training_profile["kd_alpha"],
        }
    if can_load_result_cache(cache_paths):
        print(f"[cache] using saved result for {display_name}: {cache_paths['root']}", flush=True)
        result, history, details = load_result_cache(cache_paths)
        result["display_name"] = display_name
        if training_profile.get("distillation"):
            result = add_distillation_result_fields(result, training_profile)
        return result, history, details

    model = build_model(model_name, config).to(config.device)
    params_m = count_parameters(model)
    print(f"parameters: {params_m:.2f}M", flush=True)

    train_loader = make_loader(config.data_path, train_files, train_labels, config.seq_len, model_name, shuffle=True)
    val_loader = make_loader(config.data_path, val_files, val_labels, config.seq_len, model_name, shuffle=False)
    test_loader = make_loader(config.data_path, test_files, test_labels, config.seq_len, model_name, shuffle=False)
    recorder.start_model(
        model_index,
        total_models,
        work_total=target_epochs * (len(train_loader) + len(val_loader)) + len(val_loader) + len(test_loader),
    )
    print(
        f"batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)} | "
        f"epochs={target_epochs} | detailed log interval={config.progress_every_n_batches} | "
        f"live print interval={config.print_progress_every_n_batches}",
        flush=True,
    )
    print("training profile:", training_profile, flush=True)

    if can_resume_checkpoint(cache_paths):
        print(f"[cache] loading checkpoint and evaluating {display_name}: {cache_paths['checkpoint']}", flush=True)
        model.load_state_dict(load_checkpoint(cache_paths))
        recorder.model_work_offset = target_epochs * (len(train_loader) + len(val_loader))
        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            "validation-final",
            model_name,
            config=config,
            recorder=recorder,
            label_smoothing=float(training_profile["label_smoothing"]),
        )
        recorder.model_work_offset += len(val_loader)
        test_metrics, y_true, y_pred = evaluate(
            model,
            test_loader,
            "test",
            model_name,
            config=config,
            recorder=recorder,
            label_smoothing=float(training_profile["label_smoothing"]),
        )
        result = {
            "model": model_name,
            "display_name": display_name,
            "params_m": round(params_m, 3),
            "best_val_acc": round(val_metrics["acc"], 4),
            "best_val_bal_acc": round(val_metrics["balanced_acc"], 4),
            "best_val_macro_f1": round(val_metrics["macro_f1"], 4),
            "test_acc": round(test_metrics["acc"], 4),
            "test_bal_acc": round(test_metrics["balanced_acc"], 4),
            "test_macro_f1": round(test_metrics["macro_f1"], 4),
            "test_weighted_f1": round(test_metrics["weighted_f1"], 4),
            "test_loss": round(test_metrics["loss"], 4),
            "train_seconds": 0.0,
        }
        result = add_distillation_result_fields(result, training_profile)
        history = []
        details = {"metrics": test_metrics, "y_true": y_true, "y_pred": y_pred}
        save_model_cache(cache_paths, cache_metadata, result, history, details, best_state=None)
        cleanup_accelerator(model)
        return result, history, details

    teacher_model = None
    if model_name in KD_STUDENT_MODELS:
        teacher_model, teacher_metadata = load_distillation_teacher(
            config,
            train_files,
            train_labels,
            val_files,
            val_labels,
            test_files,
            test_labels,
            teacher_model_name=str(training_profile["teacher_model"]),
        )
        cache_metadata["distillation_teacher"] = teacher_metadata
        print(
            f"[kd] using teacher checkpoint for {display_name}: {teacher_metadata['teacher_checkpoint']}",
            flush=True,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_profile["learning_rate"]),
        weight_decay=float(training_profile["weight_decay"]),
    )
    total_steps = max(1, target_epochs * len(train_loader))
    warmup_steps = int(training_profile["warmup_epochs"]) * len(train_loader)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        enabled=bool(training_profile["use_cosine"]),
    )

    history = []
    best_val_bal_acc = -1.0
    best_state = None
    train_start = time.time()
    completed_work_units = 0

    for epoch in range(1, target_epochs + 1):
        recorder.model_work_offset = completed_work_units
        if teacher_model is None:
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                model_name,
                epoch,
                total_epochs=target_epochs,
                config=config,
                recorder=recorder,
                label_smoothing=float(training_profile["label_smoothing"]),
                clip_grad_norm=training_profile["clip_grad_norm"],
                scheduler=scheduler,
                augmentation_enabled=bool(training_profile["augmentation"]),
                cpu_grad_clip=bool(training_profile.get("cpu_grad_clip", False)),
                skip_nonfinite_grad_steps=bool(training_profile.get("skip_nonfinite_grad_steps", False)),
            )
        else:
            train_metrics = train_one_epoch_distilled(
                model,
                teacher_model,
                train_loader,
                optimizer,
                model_name,
                epoch,
                total_epochs=target_epochs,
                config=config,
                recorder=recorder,
                temperature=float(training_profile["kd_temperature"]),
                alpha=float(training_profile["kd_alpha"]),
                label_smoothing=float(training_profile["label_smoothing"]),
                clip_grad_norm=training_profile["clip_grad_norm"],
                scheduler=scheduler,
                augmentation_enabled=bool(training_profile["augmentation"]),
                cpu_grad_clip=bool(training_profile.get("cpu_grad_clip", False)),
                skip_nonfinite_grad_steps=bool(training_profile.get("skip_nonfinite_grad_steps", False)),
            )
        completed_work_units += len(train_loader)
        recorder.model_work_offset = completed_work_units
        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            "validation",
            model_name,
            config=config,
            recorder=recorder,
            label_smoothing=float(training_profile["label_smoothing"]),
        )
        completed_work_units += len(val_loader)
        history_row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_balanced_acc": train_metrics["balanced_acc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_balanced_acc": val_metrics["balanced_acc"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        if "hard_loss" in train_metrics and "soft_loss" in train_metrics:
            history_row["train_hard_loss"] = train_metrics["hard_loss"]
            history_row["train_soft_loss"] = train_metrics["soft_loss"]
            history_row["kd_temperature"] = float(training_profile["kd_temperature"])
            history_row["kd_alpha"] = float(training_profile["kd_alpha"])
        if train_metrics.get("skipped_grad_steps", 0):
            history_row["skipped_grad_steps"] = int(train_metrics["skipped_grad_steps"])
        history.append(history_row)

        if float(val_metrics["balanced_acc"]) > best_val_bal_acc:
            best_val_bal_acc = float(val_metrics["balanced_acc"])
            best_state = clone_state_dict_to_cpu(model)

        epoch_progress_text = ""
        if recorder.model_work_total and model_index is not None and total_models is not None:
            epoch_model_progress_pct = 100.0 * completed_work_units / max(1, recorder.model_work_total)
            epoch_overall_progress_pct = 100.0 * ((model_index - 1) + epoch_model_progress_pct / 100.0) / max(1, total_models)
            epoch_progress_text = f"[overall {epoch_overall_progress_pct:.1f}% | model {model_index}/{total_models} {epoch_model_progress_pct:.1f}%] "
        print(
            f"{epoch_progress_text}[epoch] {display_name} | {epoch}/{target_epochs} | "
            f"train bal={train_metrics['balanced_acc']:.4f}, train f1={train_metrics['macro_f1']:.4f} | "
            f"val bal={val_metrics['balanced_acc']:.4f}, val f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )

    train_seconds = time.time() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)

    recorder.model_work_offset = completed_work_units
    val_metrics, _, _ = evaluate(
        model,
        val_loader,
        "validation-final",
        model_name,
        config=config,
        recorder=recorder,
        label_smoothing=float(training_profile["label_smoothing"]),
    )
    completed_work_units += len(val_loader)
    recorder.model_work_offset = completed_work_units
    test_metrics, y_true, y_pred = evaluate(
        model,
        test_loader,
        "test",
        model_name,
        config=config,
        recorder=recorder,
        label_smoothing=float(training_profile["label_smoothing"]),
    )

    result = {
        "model": model_name,
        "display_name": display_name,
        "params_m": round(params_m, 3),
        "best_val_acc": round(val_metrics["acc"], 4),
        "best_val_bal_acc": round(val_metrics["balanced_acc"], 4),
        "best_val_macro_f1": round(val_metrics["macro_f1"], 4),
        "test_acc": round(test_metrics["acc"], 4),
        "test_bal_acc": round(test_metrics["balanced_acc"], 4),
        "test_macro_f1": round(test_metrics["macro_f1"], 4),
        "test_weighted_f1": round(test_metrics["weighted_f1"], 4),
        "test_loss": round(test_metrics["loss"], 4),
        "train_seconds": round(train_seconds, 1),
    }
    result = add_distillation_result_fields(result, training_profile)
    details = {"metrics": test_metrics, "y_true": y_true, "y_pred": y_pred}
    save_model_cache(cache_paths, cache_metadata, result, history, details, best_state=best_state)
    cleanup_accelerator(model)
    if teacher_model is not None:
        cleanup_accelerator(teacher_model)
    return result, history, details


def run_experiment(config, train_files, train_labels, val_files, val_labels, test_files, test_labels) -> ExperimentRun:
    # Public entry point for a complete configured experiment.
    wall_start = time.time()
    recorder = ProgressRecorder(config)

    if config.forward_only:
        smoke_df = run_forward_smoke(config, train_files, train_labels)
        total_seconds = time.time() - wall_start
        print(
            f"\nForward-only smoke run complete in {format_seconds(total_seconds)}. "
            "Set TRAIN_SMOKE=True for accuracy/plots, or RUN_MODE='full' for the full experiment."
        )
        return ExperimentRun(
            results_df=pd.DataFrame(),
            histories={},
            test_details={},
            failed_models=[],
            total_seconds=total_seconds,
            progress_df=pd.DataFrame(),
            progress_summary_df=pd.DataFrame(),
            progress_tail_df=pd.DataFrame(),
            smoke_df=smoke_df,
        )

    results = []
    histories = {}
    test_details = {}
    failed_models = []
    total_models = len(config.models)

    print(f"Training/evaluation run on {config.device}: {total_models} selected model(s)", flush=True)
    for model_index, model_name in enumerate(config.models, start=1):
        try:
            result, history, details = run_single_model(
                model_name,
                config,
                train_files,
                train_labels,
                val_files,
                val_labels,
                test_files,
                test_labels,
                recorder,
                model_index,
                total_models,
            )
        except (RuntimeError, NotImplementedError, ValueError, FloatingPointError) as exc:
            cleanup_accelerator()
            failed_models.append(
                {
                    "model": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                    "model_key": model_name,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[failed] {MODEL_DISPLAY_NAMES.get(model_name, model_name)}: {type(exc).__name__}: {exc}", flush=True)
            continue

        results.append(result)
        histories[model_name] = pd.DataFrame(history)
        test_details[model_name] = details
        print(f"[done] model {model_index}/{total_models}: {MODEL_DISPLAY_NAMES[model_name]} | test_bal_acc={result['test_bal_acc']:.4f}", flush=True)

    total_seconds = time.time() - wall_start
    results_df = pd.DataFrame(results).sort_values("test_bal_acc", ascending=False).reset_index(drop=True) if results else pd.DataFrame()
    if results_df.empty:
        print("No model completed successfully. Check FAILED_MODELS below and edit ENABLED_MODELS if needed.")
    else:
        print(f"Experiment run complete in {format_seconds(total_seconds)}. Detailed result and runtime tables are shown below.")

    progress_df = recorder.frame()
    progress_summary_df = recorder.summary_frame()
    progress_tail_df = recorder.tail_frame()
    if not progress_summary_df.empty:
        print("Progress log summary by model and phase")
    if not progress_tail_df.empty:
        print(f"Last {min(config.progress_log_tail_rows, len(progress_df))} recorded progress events")

    return ExperimentRun(
        results_df=results_df,
        histories=histories,
        test_details=test_details,
        failed_models=failed_models,
        total_seconds=total_seconds,
        progress_df=progress_df,
        progress_summary_df=progress_summary_df,
        progress_tail_df=progress_tail_df,
        smoke_df=pd.DataFrame(),
    )
