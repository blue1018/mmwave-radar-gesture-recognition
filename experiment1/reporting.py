from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import confusion_matrix

from experiment_config import (
    CONFIG,
    MODEL_DISPLAY_NAMES,
    TRANSFORMER_TRAINING_MODELS,
    model_architecture_profile,
    model_batch_size,
    model_training_profile,
)

# -----------------------------
# Result tables and figures
# -----------------------------

FAMILY_COLORS = {
    "Baseline": "#7a7f87",
    "Transformer": "#2f6db3",
    "MobileViT Family": "#6f63b7",
    "External Pretrained": "#b85c5c",
    "Other": "#8c6d31",
}

CARBON_AVG_POWER_W = 30.0
CARBON_INTENSITY_G_PER_KWH = 220.0
WH_PER_KWH = 1000.0


def carbon_estimate_from_runtime(
    train_seconds: float | int | None,
    epochs: float | int | None = None,
    avg_power_w: float = CARBON_AVG_POWER_W,
    carbon_intensity_g_per_kwh: float = CARBON_INTENSITY_G_PER_KWH,
) -> dict[str, float]:
    # Paper-aligned estimate:
    # E_Wh = avg_power_w * train_seconds / 3600
    # E_kWh = E_Wh / 1000
    # C_gCO2 = E_kWh * carbon_intensity_g_per_kwh
    if train_seconds is None or pd.isna(train_seconds):
        return {
            "avg_power_w_assumed": float(avg_power_w),
            "carbon_intensity_g_per_kwh": float(carbon_intensity_g_per_kwh),
            "energy_wh_est": np.nan,
            "energy_kwh_est": np.nan,
            "carbon_g_est": np.nan,
            "wh_per_epoch_est": np.nan,
        }

    train_seconds = max(0.0, float(train_seconds))
    energy_wh = float(avg_power_w) * train_seconds / 3600.0
    energy_kwh = energy_wh / WH_PER_KWH
    carbon_g = energy_kwh * float(carbon_intensity_g_per_kwh)
    valid_epochs = epochs is not None and not pd.isna(epochs) and float(epochs) > 0
    wh_per_epoch = energy_wh / float(epochs) if valid_epochs else np.nan
    return {
        "avg_power_w_assumed": round(float(avg_power_w), 3),
        "carbon_intensity_g_per_kwh": round(float(carbon_intensity_g_per_kwh), 3),
        "energy_wh_est": round(energy_wh, 4),
        "energy_kwh_est": round(energy_kwh, 6),
        "carbon_g_est": round(carbon_g, 4),
        "wh_per_epoch_est": round(wh_per_epoch, 4) if not pd.isna(wh_per_epoch) else np.nan,
    }


def model_epochs_for_reporting(model_name: str, model_training_profile_fn) -> float:
    if model_training_profile_fn is None:
        return np.nan
    try:
        return int(model_training_profile_fn(model_name)["epochs"])
    except Exception:
        return np.nan


def result_model_family(model_name: str, transformer_model_names: set[str]) -> str:
    # Classify models into the families used by result tables and figures.
    if model_name == "crnn":
        return "Baseline"
    if "mobilevit" in model_name:
        return "MobileViT Family"
    if model_name.startswith("external_"):
        return "External Pretrained"
    if model_name in transformer_model_names:
        return "Transformer"
    return "Other"


def split_summary_row(
    split_name: str,
    env_ids: str,
    labels: np.ndarray,
    configured_limit: int | None,
    label_encoder,
) -> dict[str, object]:
    # Summarize split size, class balance, and configured sample limit.
    decoded = pd.Series(label_encoder.inverse_transform(labels))
    counts = decoded.value_counts().reindex(label_encoder.classes_, fill_value=0)
    gesture_counts = counts.drop(labels=["n"], errors="ignore")
    min_gesture = int(gesture_counts.min()) if len(gesture_counts) else 0
    max_gesture = int(gesture_counts.max()) if len(gesture_counts) else 0
    ratio = float(counts.get("n", 0) / max(1, min_gesture)) if min_gesture else np.nan
    return {
        "split": split_name,
        "environment_ids": env_ids,
        "samples": int(len(labels)),
        "n_samples": int(counts.get("n", 0)),
        "gesture_samples": int(len(labels) - counts.get("n", 0)),
        "min_gesture_class": min_gesture,
        "max_gesture_class": max_gesture,
        "n_to_min_gesture_ratio": round(ratio, 2),
        "configured_limit": "full" if configured_limit is None else int(configured_limit),
    }


def class_distribution_table(split_labels: dict[str, np.ndarray], label_encoder) -> pd.DataFrame:
    # Return per-class sample counts for each data split.
    rows = []
    for split_name, labels in split_labels.items():
        decoded = pd.Series(label_encoder.inverse_transform(labels))
        counts = decoded.value_counts().reindex(label_encoder.classes_, fill_value=0)
        rows.append(counts.rename(split_name))
    return pd.DataFrame(rows).T.reset_index(names="class_label")


def model_setup_rows(
    model_order: list[str],
    display_names: dict[str, str],
    registry_df: pd.DataFrame,
    model_training_profile_fn,
    model_family_fn,
    batch_size_fn=model_batch_size,
    config=None,
    train_labels: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    test_labels: np.ndarray | None = None,
    make_loader_fn=None,
) -> list[dict[str, object]]:
    # Build rows describing model size, architecture, optimizer settings, and training work.
    registry_params = {}
    registry_available = {}
    registry_errors = {}
    if registry_df is not None and not registry_df.empty:
        registry_params = dict(zip(registry_df["model"], registry_df["params_m"]))
        if "available" in registry_df.columns:
            registry_available = dict(zip(registry_df["model"], registry_df["available"]))
        if "registry_error" in registry_df.columns:
            registry_errors = dict(zip(registry_df["model"], registry_df["registry_error"]))

    rows = []
    for model_name in model_order:
        profile = model_training_profile_fn(model_name)
        architecture = model_architecture_profile(model_name)
        grad_clip = profile["clip_grad_norm"]
        batch_size = batch_size_fn(model_name)
        epochs = int(profile["epochs"])
        train_samples = int(len(train_labels)) if train_labels is not None else np.nan
        val_samples = int(len(val_labels)) if val_labels is not None else np.nan
        test_samples = int(len(test_labels)) if test_labels is not None else np.nan
        train_batches = int(np.ceil(train_samples / batch_size)) if not pd.isna(train_samples) else np.nan
        val_batches = int(np.ceil(val_samples / batch_size)) if not pd.isna(val_samples) else np.nan
        test_batches = int(np.ceil(test_samples / batch_size)) if not pd.isna(test_samples) else np.nan
        total_optimizer_steps = int(epochs * train_batches) if not pd.isna(train_batches) else np.nan
        total_train_val_batches = int(epochs * (train_batches + val_batches)) if not pd.isna(train_batches) and not pd.isna(val_batches) else np.nan
        warmup_steps = int(profile["warmup_epochs"] * train_batches) if not pd.isna(train_batches) else np.nan
        scheduler = "warmup+cosine" if bool(profile["use_cosine"]) else "constant"
        loss_name = "kd_soft_logits + cross_entropy" if bool(profile.get("distillation", False)) else "cross_entropy"
        loss_device = "cpu" if config is not None and getattr(config.device, "type", None) == "mps" else str(getattr(config, "device", "unknown"))
        rows.append(
            {
                "model": display_names[model_name],
                "model_key": model_name,
                "family": model_family_fn(model_name),
                "params_M": round(float(registry_params.get(model_name, np.nan)), 3),
                "registry_available": registry_available.get(model_name, True),
                "registry_error": registry_errors.get(model_name, ""),
                "architecture": architecture["architecture"],
                "input_view": architecture["input_view"],
                "input_shape": f"{getattr(config, 'seq_len', 'n/a')}x{getattr(config, 'image_size', 'n/a')}x{getattr(config, 'image_size', 'n/a')}",
                "input_channels": architecture["input_channels"],
                "num_classes": getattr(config, "num_classes", np.nan),
                "spatial_encoder": architecture["spatial_encoder"],
                "temporal_encoder": architecture["temporal_encoder"],
                "embed_dim": architecture["embed_dim"],
                "hidden_dim": architecture["hidden_dim"],
                "transformer_layers": architecture["transformer_layers"],
                "attention_heads": architecture["attention_heads"],
                "ffn_dim": architecture["ffn_dim"],
                "patch_size": architecture["patch_size"],
                "dropout": architecture["dropout"],
                "pooling": architecture["pooling"],
                "optimizer": "AdamW",
                "loss": loss_name,
                "loss_device": loss_device,
                "batch_size": batch_size,
                "train_samples": train_samples,
                "val_samples": val_samples,
                "test_samples": test_samples,
                "train_batches_per_epoch": train_batches,
                "val_batches_per_epoch": val_batches,
                "test_batches_final": test_batches,
                "epochs": epochs,
                "total_optimizer_steps": total_optimizer_steps,
                "total_train_val_batches": total_train_val_batches,
                "learning_rate": float(profile["learning_rate"]),
                "weight_decay": float(profile["weight_decay"]),
                "label_smoothing": float(profile["label_smoothing"]),
                "grad_clip": "none" if grad_clip is None else float(grad_clip),
                "warmup_epochs": int(profile["warmup_epochs"]),
                "warmup_steps": warmup_steps,
                "scheduler": scheduler,
                "cosine_schedule": bool(profile["use_cosine"]),
                "motion_input": bool(profile["motion_input"]),
                "augmentation": bool(profile["augmentation"]),
                "run_mode": getattr(config, "run_mode", "unknown"),
                "device": str(getattr(config, "device", "unknown")),
            }
        )
    return rows


def accuracy_comparison_frame(
    results_df: pd.DataFrame,
    model_family_fn,
    selected_models: tuple[str, ...] | list[str] | None = None,
    display_names: dict[str, str] | None = None,
    registry_df: pd.DataFrame | None = None,
    failed_models: list[dict[str, object]] | None = None,
    model_training_profile_fn=None,
) -> pd.DataFrame:
    # Create the compact main comparison table. Successful models are ranked;
    # failed or unavailable selected models are shown at the bottom with no score.
    comparison_df = results_df[
        [
            "model",
            "display_name",
            "test_bal_acc",
            "test_acc",
            "test_macro_f1",
            "best_val_bal_acc",
            "params_m",
            "train_seconds",
        ]
    ].copy()
    comparison_df["family"] = comparison_df["model"].map(model_family_fn)
    comparison_df["status"] = "completed"
    comparison_df["failure_reason"] = ""
    comparison_df["epochs"] = comparison_df["model"].apply(
        lambda model_name: model_epochs_for_reporting(model_name, model_training_profile_fn)
    )
    carbon_df = comparison_df.apply(
        lambda row: pd.Series(carbon_estimate_from_runtime(row["train_seconds"], row["epochs"])),
        axis=1,
    )
    comparison_df = pd.concat([comparison_df, carbon_df], axis=1)
    comparison_df.insert(0, "rank_by_test_bal_acc", range(1, len(comparison_df) + 1))
    comparison_df = comparison_df.rename(
        columns={
            "display_name": "model_name",
            "best_val_bal_acc": "val_bal_acc",
            "params_m": "params_M",
        }
    )
    if selected_models is not None:
        display_names = MODEL_DISPLAY_NAMES if display_names is None else display_names
        completed = set(results_df["model"])
        registry_params = {}
        if registry_df is not None and not registry_df.empty and {"model", "params_m"}.issubset(registry_df.columns):
            registry_params = dict(zip(registry_df["model"], registry_df["params_m"]))
        failure_reasons = {
            str(row.get("model_key", "")): str(row.get("reason", ""))
            for row in (failed_models or [])
        }
        missing_rows = []
        for model_name in selected_models:
            if model_name in completed:
                continue
            missing_rows.append(
                {
                    "rank_by_test_bal_acc": np.nan,
                    "model": model_name,
                    "model_name": display_names.get(model_name, model_name),
                    "family": model_family_fn(model_name),
                    "test_bal_acc": np.nan,
                    "test_acc": np.nan,
                    "test_macro_f1": np.nan,
                    "val_bal_acc": np.nan,
                    "params_M": registry_params.get(model_name, np.nan),
                    "train_seconds": np.nan,
                    "epochs": model_epochs_for_reporting(model_name, model_training_profile_fn),
                    "energy_wh_est": np.nan,
                    "carbon_g_est": np.nan,
                    "wh_per_epoch_est": np.nan,
                    "status": "failed" if model_name in failure_reasons else "not completed",
                    "failure_reason": failure_reasons.get(model_name, ""),
                }
            )
        if missing_rows:
            comparison_df = pd.concat([comparison_df, pd.DataFrame(missing_rows)], ignore_index=True)
    return comparison_df[
        [
            "rank_by_test_bal_acc",
            "model_name",
            "model",
            "family",
            "status",
            "test_bal_acc",
            "test_acc",
            "test_macro_f1",
            "val_bal_acc",
            "params_M",
            "train_seconds",
            "energy_wh_est",
            "carbon_g_est",
            "wh_per_epoch_est",
            "failure_reason",
        ]
    ]


def build_runtime_tables(
    results_df: pd.DataFrame,
    selected_models: tuple[str, ...],
    run_mode: str,
    device: str,
    train_files: list[str],
    train_labels: np.ndarray,
    val_files: list[str],
    val_labels: np.ndarray,
    test_files: list[str],
    test_labels: np.ndarray,
    model_training_profile_fn,
    make_loader_fn,
    model_family_fn,
    failed_models: list[dict[str, object]] | None = None,
    experiment_total_seconds: float | None = None,
    batch_size_fn=model_batch_size,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Build total, per-model, detailed, and family-level runtime tables.
    runtime_rows = []
    for _, row in results_df.iterrows():
        model_name = row["model"]
        epochs = int(model_training_profile_fn(model_name)["epochs"])
        train_seconds = float(row["train_seconds"])
        params_m = float(row["params_m"])
        train_batches = len(make_loader_fn(train_files, train_labels, model_name, shuffle=False))
        val_batches = len(make_loader_fn(val_files, val_labels, model_name, shuffle=False))
        test_batches = len(make_loader_fn(test_files, test_labels, model_name, shuffle=False))
        total_epoch_batches = train_batches + val_batches
        carbon_metrics = carbon_estimate_from_runtime(train_seconds, epochs)
        runtime_rows.append(
            {
                "model": row["display_name"],
                "model_key": model_name,
                "family": model_family_fn(model_name),
                "params_M": round(params_m, 3),
                "batch_size": batch_size_fn(model_name),
                "epochs": epochs,
                "train_batches_per_epoch": train_batches,
                "val_batches_per_epoch": val_batches,
                "test_batches_final": test_batches,
                "epoch_train_val_batches": total_epoch_batches,
                "train_seconds": round(train_seconds, 1),
                "minutes": round(train_seconds / 60.0, 2),
                "seconds_per_epoch": round(train_seconds / max(1, epochs), 2),
                "seconds_per_train_batch": round(train_seconds / max(1, epochs * train_batches), 3),
                "seconds_per_train_val_batch": round(train_seconds / max(1, epochs * total_epoch_batches), 3),
                "samples_per_second_train_split": round((len(train_labels) * epochs) / max(train_seconds, 1e-6), 2),
                "seconds_per_param_M": round(train_seconds / max(params_m, 1e-6), 2),
                "test_bal_acc": row["test_bal_acc"],
                **carbon_metrics,
            }
        )
    runtime_df = pd.DataFrame(runtime_rows)

    total_model_train_seconds = float(runtime_df["train_seconds"].sum()) if not runtime_df.empty else 0.0
    total_energy_wh = float(runtime_df["energy_wh_est"].sum()) if "energy_wh_est" in runtime_df else 0.0
    total_carbon_g = float(runtime_df["carbon_g_est"].sum()) if "carbon_g_est" in runtime_df else 0.0
    total_seconds = float(experiment_total_seconds if experiment_total_seconds is not None else total_model_train_seconds)
    overhead_seconds = max(0.0, total_seconds - total_model_train_seconds)
    failed_count = len(failed_models or [])
    runtime_total_df = pd.DataFrame(
        [
            {
                "selected_models": len(selected_models),
                "completed_models": int(len(results_df)),
                "failed_models": int(failed_count),
                "run_mode": run_mode,
                "device": device,
                "total_wall_seconds": round(total_seconds, 1),
                "total_wall_minutes": round(total_seconds / 60.0, 2),
                "sum_model_train_seconds": round(total_model_train_seconds, 1),
                "non_train_overhead_seconds": round(overhead_seconds, 1),
                "mean_model_train_seconds": round(float(runtime_df["train_seconds"].mean()), 2),
                "median_model_train_seconds": round(float(runtime_df["train_seconds"].median()), 2),
                "avg_power_w_assumed": CARBON_AVG_POWER_W,
                "carbon_intensity_g_per_kwh": CARBON_INTENSITY_G_PER_KWH,
                "total_energy_wh_est": round(total_energy_wh, 4),
                "total_energy_kwh_est": round(total_energy_wh / 1000.0, 6),
                "total_carbon_g_est": round(total_carbon_g, 4),
                "slowest_model": runtime_df.sort_values("train_seconds", ascending=False).iloc[0]["model"],
                "fastest_model": runtime_df.sort_values("train_seconds", ascending=True).iloc[0]["model"],
            }
        ]
    )

    runtime_per_model_summary_df = runtime_df[
        [
            "model",
            "family",
            "train_seconds",
            "minutes",
            "seconds_per_epoch",
            "seconds_per_train_batch",
            "samples_per_second_train_split",
            "energy_wh_est",
            "carbon_g_est",
            "wh_per_epoch_est",
            "test_bal_acc",
        ]
    ].copy()
    runtime_per_model_summary_df = runtime_per_model_summary_df.sort_values("train_seconds", ascending=False).reset_index(drop=True)
    runtime_per_model_summary_df.insert(0, "runtime_rank", range(1, len(runtime_per_model_summary_df) + 1))

    runtime_group_df = (
        runtime_df.groupby("family", as_index=False)
        .agg(
            models=("model", "count"),
            total_train_seconds=("train_seconds", "sum"),
            total_train_minutes=("minutes", "sum"),
            mean_train_seconds=("train_seconds", "mean"),
            median_train_seconds=("train_seconds", "median"),
            slowest_model_seconds=("train_seconds", "max"),
            fastest_model_seconds=("train_seconds", "min"),
            total_energy_wh_est=("energy_wh_est", "sum"),
            total_carbon_g_est=("carbon_g_est", "sum"),
            mean_wh_per_epoch_est=("wh_per_epoch_est", "mean"),
            best_test_bal_acc=("test_bal_acc", "max"),
            mean_test_bal_acc=("test_bal_acc", "mean"),
        )
        .round(4)
    )
    return runtime_df, runtime_total_df, runtime_per_model_summary_df, runtime_group_df


def transformer_advantage_frame(results_df: pd.DataFrame, transformer_model_names: set[str]) -> pd.DataFrame:
    # Compare the best completed Transformer-family model with the CRNN baseline.
    baseline_rows = results_df[results_df["model"] == "crnn"]
    transformer_rows = results_df[results_df["model"].isin(transformer_model_names)]
    if baseline_rows.empty or transformer_rows.empty:
        return pd.DataFrame()
    baseline = baseline_rows.iloc[0]
    best_transformer = transformer_rows.sort_values("test_bal_acc", ascending=False).iloc[0]
    return pd.DataFrame(
        [
            {
                "comparison": "best Transformer vs CRNN baseline",
                "best_transformer": best_transformer["display_name"],
                "baseline": baseline["display_name"],
                "test_bal_acc_delta": round(best_transformer["test_bal_acc"] - baseline["test_bal_acc"], 4),
                "macro_f1_delta": round(best_transformer["test_macro_f1"] - baseline["test_macro_f1"], 4),
                "train_seconds_delta": round(best_transformer["train_seconds"] - baseline["train_seconds"], 1),
            }
        ]
    )




def default_result_model_family(model_name: str) -> str:
    return result_model_family(model_name, TRANSFORMER_TRAINING_MODELS)


def selected_result_plot_frame(
    results_df: pd.DataFrame,
    config=CONFIG,
    failed_models: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    # Keep figures aware of selected models that failed before test metrics existed.
    selected_models = list(getattr(config, "models", ()))
    if not selected_models and results_df is not None and not results_df.empty:
        selected_models = list(results_df["model"])
    result_by_model = {}
    if results_df is not None and not results_df.empty:
        result_by_model = {
            str(row["model"]): row
            for _, row in results_df.drop_duplicates("model", keep="first").iterrows()
        }
    failure_reasons = {
        str(row.get("model_key", "")): str(row.get("reason", ""))
        for row in (failed_models or [])
    }

    rows = []
    for model_name in selected_models:
        if model_name in result_by_model:
            row = result_by_model[model_name]
            rows.append(
                {
                    "model": model_name,
                    "display_name": row.get("display_name", MODEL_DISPLAY_NAMES.get(model_name, model_name)),
                    "test_bal_acc": row.get("test_bal_acc", np.nan),
                    "test_acc": row.get("test_acc", np.nan),
                    "test_macro_f1": row.get("test_macro_f1", np.nan),
                    "params_m": row.get("params_m", np.nan),
                    "train_seconds": row.get("train_seconds", np.nan),
                    "status": "completed",
                    "failure_reason": "",
                }
            )
        else:
            rows.append(
                {
                    "model": model_name,
                    "display_name": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                    "test_bal_acc": np.nan,
                    "test_acc": np.nan,
                    "test_macro_f1": np.nan,
                    "params_m": np.nan,
                    "train_seconds": np.nan,
                    "status": "failed" if model_name in failure_reasons else "not completed",
                    "failure_reason": failure_reasons.get(model_name, ""),
                }
            )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return plot_df
    plot_df["family"] = plot_df["model"].map(default_result_model_family)
    plot_df["color"] = plot_df["family"].map(FAMILY_COLORS).fillna(FAMILY_COLORS["Other"])
    plot_df["is_completed"] = plot_df["status"].eq("completed")
    return plot_df


def build_report_tables(
    config,
    label_encoder,
    train_files,
    train_labels,
    val_files,
    val_labels,
    test_files,
    test_labels,
    model_registry_df: pd.DataFrame,
    results_df: pd.DataFrame,
    failed_models: list[dict[str, object]] | None,
    experiment_total_seconds: float | None,
    make_loader_fn,
    progress_summary_df: pd.DataFrame | None = None,
    progress_tail_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    # Build all result tables through a single reporting interface.
    tables: dict[str, pd.DataFrame] = {}
    tables["dataset_summary"] = pd.DataFrame(
        [
            split_summary_row("train", "e2,e3,e4", train_labels, config.train_limit, label_encoder),
            split_summary_row("validation", "e1", val_labels, config.val_limit, label_encoder),
            split_summary_row("test", "e6", test_labels, config.test_limit, label_encoder),
        ]
    )
    tables["class_distribution"] = class_distribution_table(
        {"train": train_labels, "validation": val_labels, "test": test_labels},
        label_encoder,
    )

    tables["model_hyperparameters"] = pd.DataFrame(
        model_setup_rows(
            list(config.models),
            MODEL_DISPLAY_NAMES,
            model_registry_df if model_registry_df is not None else pd.DataFrame(),
            lambda model_name: model_training_profile(model_name, config),
            default_result_model_family,
            model_batch_size,
            config=config,
            train_labels=train_labels,
            val_labels=val_labels,
            test_labels=test_labels,
            make_loader_fn=make_loader_fn,
        )
    )
    if failed_models:
        tables["failed_models"] = pd.DataFrame(failed_models)

    if results_df is None or results_df.empty:
        return tables

    tables["accuracy_comparison"] = accuracy_comparison_frame(
        results_df,
        default_result_model_family,
        selected_models=config.models,
        display_names=MODEL_DISPLAY_NAMES,
        registry_df=model_registry_df,
        failed_models=failed_models or [],
        model_training_profile_fn=lambda model_name: model_training_profile(model_name, config),
    )
    (
        tables["runtime_detail"],
        tables["runtime_total"],
        tables["runtime_per_model"],
        tables["runtime_by_family"],
    ) = build_runtime_tables(
        results_df,
        selected_models=config.models,
        run_mode=config.run_mode,
        device=str(config.device),
        train_files=train_files,
        train_labels=train_labels,
        val_files=val_files,
        val_labels=val_labels,
        test_files=test_files,
        test_labels=test_labels,
        model_training_profile_fn=lambda model_name: model_training_profile(model_name, config),
        make_loader_fn=make_loader_fn,
        model_family_fn=default_result_model_family,
        failed_models=failed_models or [],
        experiment_total_seconds=experiment_total_seconds,
        batch_size_fn=model_batch_size,
    )
    tables["transformer_advantage"] = transformer_advantage_frame(results_df, TRANSFORMER_TRAINING_MODELS)
    if progress_summary_df is not None:
        tables["progress_summary"] = progress_summary_df
    if progress_tail_df is not None:
        tables["progress_tail"] = progress_tail_df
    return tables


def display_report_tables(tables: dict[str, pd.DataFrame]) -> None:
    # Keep the presentation order consistent and compact.
    display_order = [
        ("Dataset summary", "dataset_summary"),
        ("Class distribution by split", "class_distribution"),
        ("Model parameters, architecture, and training hyperparameters", "model_hyperparameters"),
        ("Failed models", "failed_models"),
        ("Main model comparison, ranked by test_bal_acc", "accuracy_comparison"),
        ("Total runtime summary", "runtime_total"),
        ("Per-model runtime summary", "runtime_per_model"),
        ("Detailed per-model runtime statistics", "runtime_detail"),
        ("Runtime summary by model family", "runtime_by_family"),
        ("Progress log summary by model and phase", "progress_summary"),
        ("Last recorded progress events", "progress_tail"),
        ("Transformer advantage summary", "transformer_advantage"),
    ]
    for title, key in display_order:
        df = tables.get(key)
        if df is not None and not df.empty:
            print(title)
            if key == "accuracy_comparison":
                print(
                    "Energy/carbon columns follow the paper estimate: "
                    "E_Wh = avg_power_w * train_seconds / 3600, "
                    "E_kWh = E_Wh / 1000, "
                    "C_gCO2 = E_kWh * carbon_intensity_g_per_kwh. "
                    f"Assumptions: {CARBON_AVG_POWER_W:g} W average Mac power and "
                    f"{CARBON_INTENSITY_G_PER_KWH:g} gCO2/kWh grid intensity."
                )
            display(df)


def plot_result_figures(
    results_df,
    test_details,
    config=CONFIG,
    label_encoder=None,
    failed_models: list[dict[str, object]] | None = None,
) -> pd.DataFrame | None:
    # Plot the main model-comparison figures.
    if results_df is None or results_df.empty:
        print("Training results are not available because the notebook is currently in forward-only smoke mode.")
        return None
    if label_encoder is None:
        raise ValueError("label_encoder is required for per-class plots.")

    selected_plot_df = selected_result_plot_frame(results_df, config=config, failed_models=failed_models)
    selected_plot_df["epochs"] = selected_plot_df["model"].apply(
        lambda model_name: model_epochs_for_reporting(model_name, lambda name: model_training_profile(name, config))
    )
    carbon_plot_columns = selected_plot_df.apply(
        lambda row: pd.Series(carbon_estimate_from_runtime(row["train_seconds"], row["epochs"])),
        axis=1,
    )
    selected_plot_df = pd.concat([selected_plot_df, carbon_plot_columns], axis=1)
    completed_plot_df = selected_plot_df[selected_plot_df["is_completed"]].copy()
    missing_plot_df = selected_plot_df[~selected_plot_df["is_completed"]].copy()
    if not missing_plot_df.empty:
        missing_names = ", ".join(missing_plot_df["display_name"].astype(str))
        print(f"Models without completed test metrics are shown as n/a in summary figures: {missing_names}")

    baseline_rows = results_df[results_df["model"] == "crnn"]
    baseline_bal_acc = float(baseline_rows.iloc[0]["test_bal_acc"]) if not baseline_rows.empty else None

    plot_df = selected_plot_df.copy()
    plot_df["plot_score"] = plot_df["test_bal_acc"].fillna(0.0)
    plot_df["sort_score"] = plot_df["test_bal_acc"].fillna(-1.0)
    plot_df = plot_df.sort_values(["is_completed", "sort_score"], ascending=[True, True]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.barh(plot_df["display_name"], plot_df["plot_score"], color=plot_df["color"], alpha=0.9)
    if baseline_bal_acc is not None:
        ax.axvline(baseline_bal_acc, color="#4a4a4a", linestyle="--", linewidth=1.2, label="CRNN baseline")
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        if row["is_completed"]:
            value = float(row["test_bal_acc"])
            label = f"{value:.3f}"
            x_pos = value + 0.01
        else:
            bar.set_hatch("//")
            bar.set_edgecolor("#8d8d8d")
            label = row["status"]
            x_pos = 0.01
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9)
    score_max = completed_plot_df["test_bal_acc"].max() if not completed_plot_df.empty else 0.0
    ax.set_xlim(0, max(1.0, float(score_max) + 0.1))
    ax.set_xlabel("Test Balanced Accuracy")
    ax.set_title("Model Ranking by test_bal_acc")
    if baseline_bal_acc is not None:
        ax.legend(loc="lower right")
    fig.tight_layout()
    plt.show()

    metric_df = selected_plot_df.copy()
    metric_df["sort_score"] = metric_df["test_bal_acc"].fillna(-1.0)
    metric_df = metric_df.sort_values(["is_completed", "sort_score"], ascending=[False, False]).reset_index(drop=True)
    x = np.arange(len(metric_df))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width, metric_df["test_bal_acc"].fillna(0.0), width, label="test_bal_acc", color="#2f6db3")
    ax.bar(x, metric_df["test_macro_f1"].fillna(0.0), width, label="test_macro_f1", color="#4f9d69")
    ax.bar(x + width, metric_df["test_acc"].fillna(0.0), width, label="test_acc", color="#b1782f")
    for x_pos, (_, row) in zip(x, metric_df.iterrows()):
        if not row["is_completed"]:
            ax.text(x_pos, 0.04, row["status"], ha="center", va="bottom", fontsize=8, rotation=90, color="#5c5c5c")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_df["display_name"], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Balanced Accuracy, Macro F1, and Ordinary Accuracy")
    ax.legend()
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 5.4))
    scatter_df = metric_df[metric_df["is_completed"] & metric_df["params_m"].notna() & (metric_df["params_m"] > 0)].copy()
    if not scatter_df.empty:
        runtime_scale = np.log1p(scatter_df["train_seconds"].fillna(0.0).clip(lower=0.0).astype(float))
        runtime_span = float(runtime_scale.max() - runtime_scale.min())
        if runtime_span > 0:
            scatter_df["bubble_size"] = 120 + ((runtime_scale - runtime_scale.min()) / runtime_span) * 620
        else:
            scatter_df["bubble_size"] = 320
        scatter_df["bubble_size"] = scatter_df["bubble_size"].clip(120, 740)
        x_log = np.log10(scatter_df["params_m"].astype(float))
        x_span = float(x_log.max() - x_log.min())
        scatter_df["x_pos_norm"] = 0.5 if x_span == 0 else (x_log - x_log.min()) / x_span
        y_span = float(scatter_df["test_bal_acc"].max() - scatter_df["test_bal_acc"].min())
        scatter_df["y_pos_norm"] = 0.5 if y_span == 0 else (scatter_df["test_bal_acc"] - scatter_df["test_bal_acc"].min()) / y_span
    for family, group in scatter_df.groupby("family"):
        ax.scatter(
            group["params_m"],
            group["test_bal_acc"],
            s=group["bubble_size"],
            color=FAMILY_COLORS.get(family, FAMILY_COLORS["Other"]),
            alpha=0.74,
            edgecolor="white",
            linewidth=1.0,
            label=family,
            clip_on=True,
            zorder=3,
        )
        for _, row in group.iterrows():
            x_offset = -7 if row["x_pos_norm"] > 0.72 else 7
            y_offset = -8 if row["y_pos_norm"] > 0.72 else 6
            ax.annotate(
                row["display_name"],
                (row["params_m"], row["test_bal_acc"]),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                fontsize=8,
                ha="right" if x_offset < 0 else "left",
                va="top" if y_offset < 0 else "bottom",
                color="#333333",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.62, "pad": 0.8},
                annotation_clip=True,
                clip_on=True,
                zorder=4,
            )
    if not missing_plot_df.empty:
        omitted = ", ".join(missing_plot_df["display_name"].astype(str))
        ax.text(
            0.02,
            0.02,
            f"Not plotted without completed metrics: {omitted}",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.85},
        )
    if not scatter_df.empty:
        ax.set_xscale("log")
    ax.set_xlabel("Parameters (M, log scale)")
    ax.set_ylabel("Test Balanced Accuracy")
    ax.set_title("Accuracy vs Model Size; Bubble Size Indicates Runtime")
    if not scatter_df.empty:
        ax.margins(x=0.18, y=0.16)
        scatter_min = float(scatter_df["test_bal_acc"].min())
        scatter_max = float(scatter_df["test_bal_acc"].max())
        ax.set_ylim(max(0.0, scatter_min - 0.08), min(1.02, max(1.0, scatter_max + 0.08)))
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.28)
    if not scatter_df.empty:
        ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()

    runtime_plot_df = selected_plot_df.copy()
    runtime_plot_df["sort_seconds"] = runtime_plot_df["train_seconds"].fillna(-1.0)
    runtime_plot_df = runtime_plot_df.sort_values(["is_completed", "sort_seconds"], ascending=[True, True]).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    runtime_colors = runtime_plot_df["family"].map(FAMILY_COLORS).fillna(FAMILY_COLORS["Other"])
    runtime_bars = axes[0].barh(runtime_plot_df["display_name"], runtime_plot_df["train_seconds"].fillna(0.0), color=runtime_colors)
    for bar, (_, row) in zip(runtime_bars, runtime_plot_df.iterrows()):
        if not row["is_completed"]:
            bar.set_hatch("//")
            bar.set_edgecolor("#8d8d8d")
            axes[0].text(0.01, bar.get_y() + bar.get_height() / 2, row["status"], va="center", fontsize=8)
    axes[0].set_xlabel("Training seconds")
    axes[0].set_title("Runtime Cost")
    accuracy_bars = axes[1].barh(runtime_plot_df["display_name"], runtime_plot_df["test_bal_acc"].fillna(0.0), color=runtime_colors)
    for bar, (_, row) in zip(accuracy_bars, runtime_plot_df.iterrows()):
        if not row["is_completed"]:
            bar.set_hatch("//")
            bar.set_edgecolor("#8d8d8d")
            axes[1].text(0.01, bar.get_y() + bar.get_height() / 2, row["status"], va="center", fontsize=8)
    if baseline_bal_acc is not None:
        axes[1].axvline(baseline_bal_acc, color="#4a4a4a", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Test Balanced Accuracy")
    axes[1].set_title("Accuracy on the Same Model Order")
    runtime_score_max = completed_plot_df["test_bal_acc"].max() if not completed_plot_df.empty else 0.0
    axes[1].set_xlim(0, max(1.0, float(runtime_score_max) + 0.1))
    fig.tight_layout()
    plt.show()

    carbon_plot_df = selected_plot_df.copy()
    carbon_plot_df["sort_carbon"] = carbon_plot_df["carbon_g_est"].fillna(-1.0)
    carbon_plot_df = carbon_plot_df.sort_values(["is_completed", "sort_carbon"], ascending=[True, True]).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    carbon_colors = carbon_plot_df["family"].map(FAMILY_COLORS).fillna(FAMILY_COLORS["Other"])
    energy_bars = axes[0].barh(carbon_plot_df["display_name"], carbon_plot_df["energy_wh_est"].fillna(0.0), color=carbon_colors)
    carbon_bars = axes[1].barh(carbon_plot_df["display_name"], carbon_plot_df["carbon_g_est"].fillna(0.0), color=carbon_colors)
    max_energy = max(float(carbon_plot_df["energy_wh_est"].fillna(0.0).max()), 1e-6)
    max_carbon = max(float(carbon_plot_df["carbon_g_est"].fillna(0.0).max()), 1e-6)
    for bar, (_, row) in zip(energy_bars, carbon_plot_df.iterrows()):
        if row["is_completed"] and not pd.isna(row["energy_wh_est"]):
            axes[0].text(
                float(row["energy_wh_est"]) + max_energy * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{float(row['energy_wh_est']):.2f}",
                va="center",
                fontsize=8,
            )
        else:
            bar.set_hatch("//")
            bar.set_edgecolor("#8d8d8d")
            axes[0].text(0.01, bar.get_y() + bar.get_height() / 2, row["status"], va="center", fontsize=8)
    for bar, (_, row) in zip(carbon_bars, carbon_plot_df.iterrows()):
        if row["is_completed"] and not pd.isna(row["carbon_g_est"]):
            axes[1].text(
                float(row["carbon_g_est"]) + max_carbon * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{float(row['carbon_g_est']):.2f}",
                va="center",
                fontsize=8,
            )
        else:
            bar.set_hatch("//")
            bar.set_edgecolor("#8d8d8d")
            axes[1].text(0.01, bar.get_y() + bar.get_height() / 2, row["status"], va="center", fontsize=8)
    axes[0].set_xlabel("Estimated energy (Wh)")
    axes[0].set_title("Energy from Training Runtime")
    axes[1].set_xlabel("Estimated carbon (gCO2)")
    axes[1].set_title("Carbon from Energy x Grid Intensity")
    fig.suptitle(
        f"Estimated Mac Energy and Carbon ({CARBON_AVG_POWER_W:g} W, {CARBON_INTENSITY_G_PER_KWH:g} gCO2/kWh)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

    recall_rows = []
    skipped_detail_models = []
    for model_name in metric_df["model"]:
        if model_name not in test_details:
            skipped_detail_models.append(MODEL_DISPLAY_NAMES.get(model_name, model_name))
            continue
        metrics = test_details[model_name]["metrics"]
        recall_rows.append(pd.Series(metrics["per_class_recall"], index=label_encoder.classes_, name=MODEL_DISPLAY_NAMES[model_name]))
    per_class_recall_df = pd.DataFrame(recall_rows)
    if skipped_detail_models:
        print("Skipped per-class recall/confusion plots for models without test details:", ", ".join(skipped_detail_models))
    if per_class_recall_df.empty:
        return per_class_recall_df
    print("Per-class recall heatmap values")
    display(per_class_recall_df.round(3))

    fig, ax = plt.subplots(figsize=(11, max(3.5, 0.55 * len(per_class_recall_df))))
    heatmap = ax.imshow(per_class_recall_df.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(per_class_recall_df.columns)))
    ax.set_xticklabels(per_class_recall_df.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(per_class_recall_df.index)))
    ax.set_yticklabels(per_class_recall_df.index)
    ax.set_title("Per-Class Recall by Model")
    for i in range(per_class_recall_df.shape[0]):
        for j in range(per_class_recall_df.shape[1]):
            value = per_class_recall_df.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value > 0.55 else "#1f1f1f")
    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Recall")
    fig.tight_layout()
    plt.show()

    detail_models = set(test_details)
    transformer_rows = results_df[
        results_df["model"].isin(TRANSFORMER_TRAINING_MODELS) & results_df["model"].isin(detail_models)
    ].sort_values("test_bal_acc", ascending=False)
    selected_models = []
    if not transformer_rows.empty:
        selected_models.append(transformer_rows.iloc[0]["model"])
    if "crnn" in set(results_df["model"]) and "crnn" in detail_models:
        selected_models.append("crnn")
    selected_models = list(dict.fromkeys(selected_models))

    if selected_models:
        fig, axes = plt.subplots(1, len(selected_models), figsize=(6 * len(selected_models), 5.5), squeeze=False)
        for ax, model_name in zip(axes[0], selected_models):
            y_true = test_details[model_name]["y_true"]
            y_pred = test_details[model_name]["y_pred"]
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(config.num_classes), normalize="true")
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            ax.set_title(f"{MODEL_DISPLAY_NAMES[model_name]}\nNormalized Test Confusion")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks(np.arange(len(label_encoder.classes_)))
            ax.set_yticks(np.arange(len(label_encoder.classes_)))
            ax.set_xticklabels(label_encoder.classes_, rotation=45, ha="right")
            ax.set_yticklabels(label_encoder.classes_)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cm[i, j] > 0.55 else "#1f1f1f")
        fig.tight_layout()
        plt.show()
    return per_class_recall_df
