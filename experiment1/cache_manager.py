from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd
import torch

from experiment_config import (
    CACHE_DIR,
    FORCE_RETRAIN,
    RESUME_FROM_CHECKPOINT,
    SAVE_CHECKPOINTS,
    SAVE_RUN_CACHE,
    SKIP_TRAIN_IF_RESULTS_EXIST,
    USE_RUN_CACHE,
)

# -----------------------------
# Result and checkpoint cache
# -----------------------------


def _jsonable(value):
    # Convert numpy/torch scalar containers into plain JSON-compatible values.
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _contains_nonfinite(value) -> bool:
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite(item) for item in value)
    if isinstance(value, np.ndarray):
        return np.issubdtype(value.dtype, np.number) and not bool(np.isfinite(value).all())
    if isinstance(value, (np.floating, float)):
        return not math.isfinite(float(value))
    return False


def _strict_json(value, artifact_name: str) -> str:
    jsonable = _jsonable(value)
    if _contains_nonfinite(jsonable):
        raise FloatingPointError(f"Refusing to save non-finite values in {artifact_name}.")
    return json.dumps(jsonable, indent=2, sort_keys=True, allow_nan=False)


def _hash_payload(payload: dict[str, object]) -> str:
    # Create a stable identifier for one model, data split, and configuration.
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _split_signature(files: list[str], labels: np.ndarray) -> dict[str, object]:
    # Hash the file order and labels without storing the whole list in metadata.
    file_hash = hashlib.sha256("\n".join(files).encode("utf-8")).hexdigest()
    label_hash = hashlib.sha256(np.asarray(labels, dtype=np.int64).tobytes()).hexdigest()
    return {
        "samples": int(len(files)),
        "file_hash": file_hash,
        "label_hash": label_hash,
        "first_file": files[0] if files else None,
        "last_file": files[-1] if files else None,
    }


def model_cache_key(
    config,
    model_name: str,
    training_profile: dict[str, object],
    train_files: list[str],
    train_labels: np.ndarray,
    val_files: list[str],
    val_labels: np.ndarray,
    test_files: list[str],
    test_labels: np.ndarray,
) -> str:
    # The key changes when data split, model settings, or training settings change.
    payload = {
        "model_name": model_name,
        "data_path": str(config.data_path),
        "run_mode": config.run_mode,
        "seq_len": config.seq_len,
        "image_size": config.image_size,
        "num_classes": config.num_classes,
        "seed": config.seed,
        "train_limit": config.train_limit,
        "val_limit": config.val_limit,
        "test_limit": config.test_limit,
        "training_profile": training_profile,
        "train": _split_signature(train_files, train_labels),
        "validation": _split_signature(val_files, val_labels),
        "test": _split_signature(test_files, test_labels),
    }
    return _hash_payload(payload)


def model_cache_dir(config, model_name: str, cache_key: str) -> Path:
    # Keep each model/config combination in its own directory.
    return Path(CACHE_DIR) / config.run_mode / model_name / cache_key[:16]


def model_cache_paths(config, model_name: str, cache_key: str) -> dict[str, Path]:
    root = model_cache_dir(config, model_name, cache_key)
    return {
        "root": root,
        "metadata": root / "metadata.json",
        "result": root / "result.json",
        "history": root / "history.csv",
        "metrics": root / "test_metrics.json",
        "predictions": root / "test_predictions.npz",
        "checkpoint": root / "best_state.pt",
    }


def can_load_result_cache(paths: dict[str, Path]) -> bool:
    # A complete result cache permits reuse without repeating training or evaluation.
    if FORCE_RETRAIN or not USE_RUN_CACHE or not SKIP_TRAIN_IF_RESULTS_EXIST:
        return False
    if not (paths["result"].exists() and paths["history"].exists() and paths["metrics"].exists() and paths["predictions"].exists()):
        return False
    try:
        result = json.loads(paths["result"].read_text())
        metrics = json.loads(paths["metrics"].read_text())
        if _contains_nonfinite(result) or _contains_nonfinite(metrics):
            return False
        try:
            history_df = pd.read_csv(paths["history"])
        except pd.errors.EmptyDataError:
            history_df = pd.DataFrame()
        numeric_history = history_df.select_dtypes(include=[np.number])
        if not numeric_history.empty and not bool(np.isfinite(numeric_history.to_numpy()).all()):
            return False
        prediction_data = np.load(paths["predictions"], allow_pickle=False)
        return bool(np.isfinite(prediction_data["y_true"]).all() and np.isfinite(prediction_data["y_pred"]).all())
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return False


def can_resume_checkpoint(paths: dict[str, Path]) -> bool:
    # A checkpoint can skip training, but validation/test still run to rebuild metrics.
    if FORCE_RETRAIN or not USE_RUN_CACHE or not RESUME_FROM_CHECKPOINT:
        return False
    if not paths["checkpoint"].exists():
        return False
    try:
        state = torch.load(paths["checkpoint"], map_location="cpu")
    except (OSError, RuntimeError, ValueError):
        return False
    return all(
        not torch.is_tensor(value)
        or not torch.is_floating_point(value)
        or bool(torch.isfinite(value).all().item())
        for value in state.values()
    )


def load_result_cache(paths: dict[str, Path]) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    # Restore one model's result row, history, metrics, and predictions.
    result = json.loads(paths["result"].read_text())
    try:
        history_df = pd.read_csv(paths["history"]) if paths["history"].exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        history_df = pd.DataFrame()
    metrics = json.loads(paths["metrics"].read_text())
    prediction_data = np.load(paths["predictions"], allow_pickle=False)
    details = {
        "metrics": metrics,
        "y_true": prediction_data["y_true"],
        "y_pred": prediction_data["y_pred"],
    }
    return result, history_df.to_dict("records"), details


def save_model_cache(
    paths: dict[str, Path],
    metadata: dict[str, object],
    result: dict[str, object],
    history: list[dict[str, object]],
    details: dict[str, object],
    best_state: dict[str, torch.Tensor] | None = None,
) -> None:
    # Persist the artefacts required to reproduce reports without retraining.
    if not SAVE_RUN_CACHE:
        return
    paths["root"].mkdir(parents=True, exist_ok=True)
    full_metadata = {
        **metadata,
        "saved_at_unix": time.time(),
        "cache_schema": 1,
    }
    metadata_json = _strict_json(full_metadata, "metadata")
    result_json = _strict_json(result, "result")
    metrics_json = _strict_json(details["metrics"], "metrics")
    history_df = pd.DataFrame(history)
    numeric_history = history_df.select_dtypes(include=[np.number])
    if not numeric_history.empty and not bool(np.isfinite(numeric_history.to_numpy()).all()):
        raise FloatingPointError("Refusing to save non-finite values in history.")

    paths["metadata"].write_text(metadata_json)
    paths["result"].write_text(result_json)
    history_df.to_csv(paths["history"], index=False)
    paths["metrics"].write_text(metrics_json)
    np.savez_compressed(
        paths["predictions"],
        y_true=np.asarray(details["y_true"], dtype=np.int64),
        y_pred=np.asarray(details["y_pred"], dtype=np.int64),
    )
    if SAVE_CHECKPOINTS and best_state is not None:
        torch.save(best_state, paths["checkpoint"])


def load_checkpoint(paths: dict[str, Path]) -> dict[str, torch.Tensor]:
    # Load CPU checkpoint state for model.load_state_dict().
    state = torch.load(paths["checkpoint"], map_location="cpu")
    bad_tensors = [
        name
        for name, value in state.items()
        if torch.is_tensor(value)
        and torch.is_floating_point(value)
        and not bool(torch.isfinite(value).all().item())
    ]
    if bad_tensors:
        shown = ", ".join(bad_tensors[:5])
        suffix = "" if len(bad_tensors) <= 5 else f", ... ({len(bad_tensors)} total)"
        raise ValueError(f"Checkpoint contains non-finite tensors: {shown}{suffix}")
    return state
