from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

# -----------------------------
# Experiment configuration
# -----------------------------
# Centralise experiment parameters to support reproducibility.

RUN_MODE = "full"          # "smoke" or "full"
TRAIN_SMOKE = True          # True runs one smoke-training epoch when RUN_MODE="smoke".
DEVICE_MODE = "mps"        # "auto", "mps", or "cpu"
PROGRESS_EVERY_N_BATCHES = 2
PRINT_PROGRESS_EVERY_N_BATCHES = 20
PROGRESS_LOG_TAIL_ROWS = 80

# Transformer-family input and augmentation switches.
USE_TRANSFORMER_MOTION_INPUT = True
USE_TRANSFORMER_AUGMENTATION = True
PAPER_KD_TEACHER_MODEL = "paper_mobilevit_teacher"
PAPER_KD_TEMPERATURE = 4.0
PAPER_KD_ALPHA = 0.5
EXTERNAL_MOBILEVIT_MODEL_ID = "apple/mobilevit-small"

# The model run list is the single source of truth for the experiment.
ENABLED_MODELS = [
    "crnn",
    "cnn_transformer",
    "trans_cnn_1d",
    # "lpvt_lite",
    # "mobilevit_lite",
    # "timesformer_lite",
    "lpvt_full",
    "mobilevit_full",
    "paper_mobilevit_teacher",
    "paper_mobilevit_student",
    "external_mobilevit_frozen",
    "timesformer_full",
    # "cnn3d",
    # "adaptive_topk_multistream_cnn",
    # "mff_cnn_iat",
    # "video_swin",
]

DATA_PATH = Path("/Users/blue/NCI/MCD-Gesture_ProcessedDataset")
SEED = 21

# Cache controls for reusing results when data and model settings are unchanged.
CACHE_DIR = Path("/Users/blue/Desktop/Practicum/code/experiment1/cache")
USE_RUN_CACHE = True
SAVE_RUN_CACHE = True
SKIP_TRAIN_IF_RESULTS_EXIST = True
RESUME_FROM_CHECKPOINT = True
SAVE_CHECKPOINTS = True
FORCE_RETRAIN = False

# Optional overrides. None keeps each model's `full_epochs`; an integer overrides every model.
EPOCHS_OVERRIDE = None
# Use None to keep the selected data limit; use 0 for the complete split.
TRAIN_LIMIT_OVERRIDE = None
VAL_LIMIT_OVERRIDE = None
TEST_LIMIT_OVERRIDE = None

SEQ_LEN = 32
IMAGE_SIZE = 32
NUM_CLASSES = 7

ACTION_NAMES = [
    "SlideRight",
    "SlideLeft",
    "Push",
    "Pull",
    "Clockwise",
    "Counterclockwise",
    "n",
]

MODEL_DISPLAY_NAMES = {
    "crnn": "CRNN Baseline",
    "cnn_transformer": "CNN + Transformer",
    "cnn3d": "3D CNN",
    "trans_cnn_1d": "TRANS-CNN-1D",
    "lpvt_lite": "LPVT-lite",
    "lpvt_full": "LPVT-Full",
    "mobilevit_lite": "MobileViT (Project Lite)",
    "mobilevit_full": "MobileViT (Project Full)",
    "paper_mobilevit_teacher": "MobileViT (Paper Teacher)",
    "paper_mobilevit_student": "MobileViT (Paper Student + KD)",
    "external_mobilevit_frozen": "MobileViT (External Frozen)",
    "adaptive_topk_multistream_cnn": "Adaptive Top-K Multi-Stream CNN",
    "timesformer_lite": "TimeSformer-lite",
    "timesformer_full": "TimeSformer-Full",
    "mff_cnn_iat": "MFF-CNN-IAT",
    "video_swin": "Video Swin",
}

# Final full-run training settings, grouped by model for direct lookup.
# `warmup_epoch_cap` is reduced automatically for smoke runs or EPOCHS_OVERRIDE.
MODEL_TRAINING_SETTINGS = {
    "crnn": {
        "batch_size": 64, "full_epochs": 20, "learning_rate": 1e-4, "weight_decay": 1e-4,
        "label_smoothing": 0.0, "clip_grad_norm": None, "warmup_epoch_cap": 0,
        "use_cosine": False, "augmentation": False, "motion_input": False,
    },
    "cnn_transformer": {
        "batch_size": 32, "full_epochs": 40, "learning_rate": 4e-4, "weight_decay": 3e-2,
        "label_smoothing": 0.03, "clip_grad_norm": 1.0, "warmup_epoch_cap": 5,
        "use_cosine": True, "augmentation": USE_TRANSFORMER_AUGMENTATION,
        "motion_input": USE_TRANSFORMER_MOTION_INPUT,
    },
    "trans_cnn_1d": {
        "batch_size": 16, "full_epochs": 40, "learning_rate": 1e-4, "weight_decay": 1e-4,
        "label_smoothing": 0.03, "clip_grad_norm": 0.25, "warmup_epoch_cap": 5,
        "use_cosine": True, "augmentation": False, "motion_input": USE_TRANSFORMER_MOTION_INPUT,
        "cpu_grad_clip": True, "skip_nonfinite_grad_steps": True,
        "normalization": "group_norm_temporal_conv", "temporal_tail": "conv_mlp_no_lstm",
        "stability_profile": "trans_cnn_1d_gn_convmlp_noaug_lr1e-4_wd1e-4_clip0.25",
    },
    "lpvt_lite": {
        "batch_size": 16, "full_epochs": 40, "learning_rate": 4e-4, "weight_decay": 3e-2,
        "label_smoothing": 0.03, "clip_grad_norm": 1.0, "warmup_epoch_cap": 5,
        "use_cosine": True, "augmentation": USE_TRANSFORMER_AUGMENTATION,
        "motion_input": USE_TRANSFORMER_MOTION_INPUT,
    },
    "lpvt_full": {
        "batch_size": 6, "full_epochs": 30, "learning_rate": 2e-4, "weight_decay": 5e-2,
        "label_smoothing": 0.05, "clip_grad_norm": 1.0, "warmup_epoch_cap": 6,
        "use_cosine": True, "augmentation": False,
        "motion_input": USE_TRANSFORMER_MOTION_INPUT,
    },
    "mobilevit_lite": {
        "batch_size": 24, "full_epochs": 40, "learning_rate": 4e-4, "weight_decay": 3e-2,
        "label_smoothing": 0.03, "clip_grad_norm": 1.0, "warmup_epoch_cap": 5,
        "use_cosine": True, "augmentation": USE_TRANSFORMER_AUGMENTATION,
        "motion_input": USE_TRANSFORMER_MOTION_INPUT,
    },
    "mobilevit_full": {
        "batch_size": 6, "full_epochs": 30, "learning_rate": 2e-4, "weight_decay": 5e-2,
        "label_smoothing": 0.05, "clip_grad_norm": 1.0, "warmup_epoch_cap": 6,
        "use_cosine": True, "augmentation": USE_TRANSFORMER_AUGMENTATION,
        "motion_input": USE_TRANSFORMER_MOTION_INPUT,
    },
    "paper_mobilevit_teacher": {
        "batch_size": 16, "full_epochs": 40, "learning_rate": 3e-4, "weight_decay": 3e-2,
        "label_smoothing": 0.03, "clip_grad_norm": 1.0, "warmup_epoch_cap": 5,
        "use_cosine": True, "augmentation": USE_TRANSFORMER_AUGMENTATION,
        "motion_input": USE_TRANSFORMER_MOTION_INPUT,
        "paper_adaptation": "DRAI plus temporal-difference branch",
    },
    "paper_mobilevit_student": {
        "batch_size": 32, "full_epochs": 40, "learning_rate": 3e-4, "weight_decay": 2e-2,
        "label_smoothing": 0.0, "clip_grad_norm": 1.0, "warmup_epoch_cap": 5,
        "use_cosine": True, "augmentation": USE_TRANSFORMER_AUGMENTATION,
        "motion_input": USE_TRANSFORMER_MOTION_INPUT, "distillation": True,
        "teacher_model": PAPER_KD_TEACHER_MODEL, "kd_temperature": PAPER_KD_TEMPERATURE,
        "kd_alpha": PAPER_KD_ALPHA, "paper_adaptation": "DRAI plus temporal-difference branch",
    },
    "external_mobilevit_frozen": {
        "batch_size": 2, "full_epochs": 20, "learning_rate": 1e-4, "weight_decay": 1e-4,
        "label_smoothing": 0.03, "clip_grad_norm": 1.0, "warmup_epoch_cap": 0,
        "use_cosine": False, "augmentation": False, "motion_input": False,
        "external_pretrained_weights": f"Hugging Face {EXTERNAL_MOBILEVIT_MODEL_ID} / ImageNet-1K",
        "freeze_pretrained_backbone": True,
    },
    "timesformer_lite": {
        "batch_size": 12, "full_epochs": 40, "learning_rate": 4e-4, "weight_decay": 3e-2,
        "label_smoothing": 0.03, "clip_grad_norm": 1.0, "warmup_epoch_cap": 5,
        "use_cosine": True, "augmentation": USE_TRANSFORMER_AUGMENTATION,
        "motion_input": USE_TRANSFORMER_MOTION_INPUT,
    },
    "timesformer_full": {
        "batch_size": 2, "full_epochs": 30, "learning_rate": 1e-4, "weight_decay": 5e-2,
        "label_smoothing": 0.05, "clip_grad_norm": 1.0, "warmup_epoch_cap": 6,
        "use_cosine": True, "augmentation": USE_TRANSFORMER_AUGMENTATION,
        "motion_input": USE_TRANSFORMER_MOTION_INPUT,
    },
    "cnn3d": {
        "batch_size": 16, "full_epochs": 20, "learning_rate": 1e-4, "weight_decay": 1e-4,
        "label_smoothing": 0.0, "clip_grad_norm": None, "warmup_epoch_cap": 0,
        "use_cosine": False, "augmentation": False, "motion_input": False,
    },
    "adaptive_topk_multistream_cnn": {
        "batch_size": 16, "full_epochs": 20, "learning_rate": 1e-4, "weight_decay": 1e-4,
        "label_smoothing": 0.0, "clip_grad_norm": None, "warmup_epoch_cap": 0,
        "use_cosine": False, "augmentation": False, "motion_input": False,
    },
    "mff_cnn_iat": {
        "batch_size": 16, "full_epochs": 40, "learning_rate": 4e-4, "weight_decay": 3e-2,
        "label_smoothing": 0.03, "clip_grad_norm": 1.0, "warmup_epoch_cap": 5,
        "use_cosine": True, "augmentation": USE_TRANSFORMER_AUGMENTATION,
        "motion_input": USE_TRANSFORMER_MOTION_INPUT,
    },
    "video_swin": {
        "batch_size": 4, "full_epochs": 40, "learning_rate": 4e-4, "weight_decay": 3e-2,
        "label_smoothing": 0.03, "clip_grad_norm": 1.0, "warmup_epoch_cap": 5,
        "use_cosine": True, "augmentation": USE_TRANSFORMER_AUGMENTATION,
        "motion_input": USE_TRANSFORMER_MOTION_INPUT,
    },
}

# Architecture metadata used by the hyperparameter summary table.
# These values mirror the constructors in model_zoo.py and keep the table readable.
MODEL_ARCHITECTURE_PROFILES = {
    "crnn": {
        "architecture": "CNN frame encoder + LSTM",
        "input_view": "DRAI sequence",
        "input_channels": 1,
        "spatial_encoder": "Conv2d 16-32-64",
        "temporal_encoder": "LSTM",
        "embed_dim": 128,
        "hidden_dim": 192,
        "transformer_layers": 0,
        "attention_heads": 0,
        "ffn_dim": "n/a",
        "patch_size": "n/a",
        "dropout": "0.3 frame",
        "pooling": "last time step",
    },
    "cnn_transformer": {
        "architecture": "CNN frame encoder + Transformer",
        "input_view": "DRAI + temporal-diff channel",
        "input_channels": 2,
        "spatial_encoder": "Conv2d 16-32-64",
        "temporal_encoder": "Transformer encoder",
        "embed_dim": 128,
        "hidden_dim": "n/a",
        "transformer_layers": 2,
        "attention_heads": 4,
        "ffn_dim": 512,
        "patch_size": "n/a",
        "dropout": 0.1,
        "pooling": "CLS token",
    },
    "trans_cnn_1d": {
        "architecture": "CNN + attention + 1D temporal CNN + Conv-MLP",
        "input_view": "DRAI + temporal-diff channel",
        "input_channels": 2,
        "spatial_encoder": "Conv2d 16-32-64",
        "temporal_encoder": "MHA + GroupNorm Conv1d + Conv-MLP",
        "embed_dim": 128,
        "hidden_dim": "n/a",
        "transformer_layers": "custom",
        "attention_heads": 4,
        "ffn_dim": "n/a",
        "patch_size": "n/a",
        "dropout": "0.1/0.15/0.2/0.25",
        "pooling": "attention + mean + max",
    },
    "lpvt_lite": {
        "architecture": "Local Pyramid Vision Transformer lite",
        "input_view": "DRAI + temporal-diff channel",
        "input_channels": 2,
        "spatial_encoder": "LPVT stages 32-64-128",
        "temporal_encoder": "Transformer encoder",
        "embed_dim": 128,
        "hidden_dim": "n/a",
        "transformer_layers": 1,
        "attention_heads": 4,
        "ffn_dim": 384,
        "patch_size": "stage conv",
        "dropout": "0.1/0.25",
        "pooling": "attention + mean + max",
    },
    "lpvt_full": {
        "architecture": "Local Pyramid Vision Transformer full",
        "input_view": "DRAI + temporal-diff channel",
        "input_channels": 2,
        "spatial_encoder": "LPVT stages 48-96-192-256",
        "temporal_encoder": "Transformer encoder",
        "embed_dim": 256,
        "hidden_dim": "n/a",
        "transformer_layers": 3,
        "attention_heads": 8,
        "ffn_dim": 1024,
        "patch_size": "stage conv",
        "dropout": "0.15/0.35",
        "pooling": "attention + mean + max",
    },
    "mobilevit_lite": {
        "architecture": "MobileViT lite",
        "input_view": "DRAI + temporal-diff channel",
        "input_channels": 2,
        "spatial_encoder": "MobileViT channels 24-48-96",
        "temporal_encoder": "Transformer encoder",
        "embed_dim": 96,
        "hidden_dim": "n/a",
        "transformer_layers": 1,
        "attention_heads": 4,
        "ffn_dim": 288,
        "patch_size": "n/a",
        "dropout": "0.1/0.2",
        "pooling": "attention + mean + max",
    },
    "mobilevit_full": {
        "architecture": "MobileViT full",
        "input_view": "DRAI + temporal-diff channel",
        "input_channels": 2,
        "spatial_encoder": "MobileViT channels 32-64-128-192",
        "temporal_encoder": "Transformer encoder",
        "embed_dim": 192,
        "hidden_dim": "n/a",
        "transformer_layers": 3,
        "attention_heads": 6,
        "ffn_dim": 768,
        "patch_size": "n/a",
        "dropout": "0.15/0.3",
        "pooling": "attention + mean + max",
    },
    "paper_mobilevit_teacher": {
        "architecture": "Paper-adapted dual-branch MobileViT teacher",
        "input_view": "DRAI branch + temporal-diff DRAI branch",
        "input_channels": "1 + 1 branches",
        "spatial_encoder": "dual Conv/MV2 branches + fused MobileViT block",
        "temporal_encoder": "attention pooling over DRAI frames",
        "embed_dim": 64,
        "hidden_dim": "n/a",
        "transformer_layers": 1,
        "attention_heads": 4,
        "ffn_dim": 256,
        "patch_size": "n/a",
        "dropout": "0.1/0.25",
        "pooling": "attention + mean + max",
    },
    "paper_mobilevit_student": {
        "architecture": "Distilled paper-adapted MobileViT student",
        "input_view": "DRAI branch + temporal-diff DRAI branch",
        "input_channels": "1 + 1 branches",
        "spatial_encoder": "compressed dual Conv/MV2 branches + fused MobileViT block",
        "temporal_encoder": "attention pooling over DRAI frames",
        "embed_dim": 32,
        "hidden_dim": "n/a",
        "transformer_layers": 1,
        "attention_heads": 2,
        "ffn_dim": 128,
        "patch_size": "n/a",
        "dropout": "0.1/0.2",
        "pooling": "attention + mean + max",
    },
    "external_mobilevit_frozen": {
        "architecture": "ImageNet-pretrained MobileViT frozen frame encoder",
        "input_view": "DRAI frames as normalized 3-channel images",
        "input_channels": 3,
        "spatial_encoder": "frozen Hugging Face MobileViT ImageNet backbone",
        "temporal_encoder": "attention pooling over frame features",
        "embed_dim": 256,
        "hidden_dim": "n/a",
        "transformer_layers": "pretrained",
        "attention_heads": "pretrained",
        "ffn_dim": "n/a",
        "patch_size": "n/a",
        "dropout": "0.2/0.25",
        "pooling": "attention + mean + max",
    },
    "timesformer_lite": {
        "architecture": "Factorized TimeSformer lite",
        "input_view": "DRAI + temporal-diff channel",
        "input_channels": 2,
        "spatial_encoder": "patch embedding",
        "temporal_encoder": "space-time attention",
        "embed_dim": 128,
        "hidden_dim": "n/a",
        "transformer_layers": 2,
        "attention_heads": 4,
        "ffn_dim": 512,
        "patch_size": 8,
        "dropout": "0.1/0.15",
        "pooling": "attention + mean + max tokens",
    },
    "timesformer_full": {
        "architecture": "Factorized TimeSformer full",
        "input_view": "DRAI + temporal-diff channel",
        "input_channels": 2,
        "spatial_encoder": "patch embedding",
        "temporal_encoder": "space-time attention",
        "embed_dim": 256,
        "hidden_dim": "n/a",
        "transformer_layers": 4,
        "attention_heads": 8,
        "ffn_dim": 1024,
        "patch_size": 8,
        "dropout": "0.15/0.3",
        "pooling": "attention + mean + max tokens",
    },
    "cnn3d": {
        "architecture": "3D CNN",
        "input_view": "DRAI sequence",
        "input_channels": 1,
        "spatial_encoder": "Conv3d 16-32-64",
        "temporal_encoder": "3D convolution",
        "embed_dim": "n/a",
        "hidden_dim": 128,
        "transformer_layers": 0,
        "attention_heads": 0,
        "ffn_dim": "n/a",
        "patch_size": "n/a",
        "dropout": 0.3,
        "pooling": "AdaptiveAvgPool3d",
    },
    "adaptive_topk_multistream_cnn": {
        "architecture": "3-stream 3D CNN",
        "input_view": "DRAI + temporal-diff + top-k DRAI",
        "input_channels": 1,
        "spatial_encoder": "three Conv3d branches",
        "temporal_encoder": "3D convolution",
        "embed_dim": "n/a",
        "hidden_dim": 128,
        "transformer_layers": 0,
        "attention_heads": 0,
        "ffn_dim": "n/a",
        "patch_size": "n/a",
        "dropout": 0.3,
        "pooling": "branch concat",
    },
    "mff_cnn_iat": {
        "architecture": "MFF-CNN-IAT proxy",
        "input_view": "DRAI + temporal-diff + frequency DRAI",
        "input_channels": 1,
        "spatial_encoder": "MobileNet frame branches",
        "temporal_encoder": "interference-aware attention",
        "embed_dim": 96,
        "hidden_dim": "n/a",
        "transformer_layers": 2,
        "attention_heads": 4,
        "ffn_dim": 384,
        "patch_size": "n/a",
        "dropout": "0.1/0.2",
        "pooling": "attention + mean + max",
    },
    "video_swin": {
        "architecture": "Video Swin",
        "input_view": "DRAI sequence repeated to 3 channels",
        "input_channels": 3,
        "spatial_encoder": "Swin3D-T",
        "temporal_encoder": "window attention",
        "embed_dim": "torchvision",
        "hidden_dim": "torchvision",
        "transformer_layers": "torchvision",
        "attention_heads": "torchvision",
        "ffn_dim": "torchvision",
        "patch_size": "torchvision",
        "dropout": "torchvision",
        "pooling": "Swin head",
    },
}

RUN_MODE_PROFILES = {
    "smoke": {
        "run_epochs": 1,
        "train_limit": 280,
        "val_limit": 140,
        "test_limit": 140,
        "forward_only": True,
    },
    "full": {
        "run_epochs": None,
        "train_limit": None,
        "val_limit": None,
        "test_limit": None,
        "forward_only": False,
    },
}

TRANSFORMER_TRAINING_MODELS = {
    "cnn_transformer",
    "trans_cnn_1d",
    "lpvt_lite",
    "lpvt_full",
    "mobilevit_lite",
    "mobilevit_full",
    "paper_mobilevit_teacher",
    "paper_mobilevit_student",
    "timesformer_lite",
    "timesformer_full",
    "mff_cnn_iat",
    "video_swin",
}
KD_STUDENT_MODELS = {"paper_mobilevit_student"}


@dataclass
class ExperimentConfig:
    data_path: Path
    run_mode: str
    seq_len: int
    image_size: int
    num_classes: int
    seed: int
    epochs: int | None
    train_limit: int | None
    val_limit: int | None
    test_limit: int | None
    models: tuple[str, ...]
    forward_only: bool
    progress_every_n_batches: int
    print_progress_every_n_batches: int
    progress_log_tail_rows: int
    device: torch.device


def resolve_limit(profile_value: int | None, override_value: int | None) -> int | None:
    # Interpret 0 as a request to use the complete split.
    if override_value is None:
        return profile_value
    return None if override_value == 0 else override_value


def select_device(device_mode: str = DEVICE_MODE) -> torch.device:
    # Centralise device selection for consistent execution.
    mps_available = torch.backends.mps.is_available()
    if device_mode not in {"auto", "mps", "cpu"}:
        raise ValueError("DEVICE_MODE must be 'auto', 'mps', or 'cpu'.")
    if device_mode == "mps" and not mps_available:
        raise RuntimeError(
            "DEVICE_MODE='mps', but the current Jupyter kernel does not expose MPS. "
            "Select the registered 'Python (pytorch real)' kernel, or use DEVICE_MODE='auto'/'cpu'."
        )
    if device_mode == "cpu":
        return torch.device("cpu")
    if device_mode == "mps":
        return torch.device("mps")
    return torch.device("mps" if mps_available else "cpu")


def selected_model_names() -> tuple[str, ...]:
    # Preserve order while removing accidental duplicates.
    known_model_keys = set(MODEL_DISPLAY_NAMES)
    unknown_models = [model_name for model_name in ENABLED_MODELS if model_name not in known_model_keys]
    if unknown_models:
        raise ValueError(f"Unknown model name(s) in ENABLED_MODELS: {unknown_models}")
    missing_settings = [model_name for model_name in ENABLED_MODELS if model_name not in MODEL_TRAINING_SETTINGS]
    if missing_settings:
        raise ValueError(f"Missing MODEL_TRAINING_SETTINGS for: {missing_settings}")
    models = tuple(dict.fromkeys(ENABLED_MODELS))
    if not models:
        raise ValueError("No model is enabled. Uncomment at least one model in ENABLED_MODELS.")
    return models


def make_config() -> ExperimentConfig:
    # Build the immutable configuration consumed by the runner and training code.
    if RUN_MODE not in RUN_MODE_PROFILES:
        raise ValueError(f"RUN_MODE must be one of {list(RUN_MODE_PROFILES)}")
    profile = RUN_MODE_PROFILES[RUN_MODE]
    return ExperimentConfig(
        data_path=DATA_PATH,
        run_mode=RUN_MODE,
        seq_len=SEQ_LEN,
        image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        seed=SEED,
        epochs=profile["run_epochs"] if EPOCHS_OVERRIDE is None else EPOCHS_OVERRIDE,
        train_limit=resolve_limit(profile["train_limit"], TRAIN_LIMIT_OVERRIDE),
        val_limit=resolve_limit(profile["val_limit"], VAL_LIMIT_OVERRIDE),
        test_limit=resolve_limit(profile["test_limit"], TEST_LIMIT_OVERRIDE),
        models=selected_model_names(),
        forward_only=bool(profile["forward_only"]) and not TRAIN_SMOKE,
        progress_every_n_batches=max(1, int(PROGRESS_EVERY_N_BATCHES)),
        print_progress_every_n_batches=max(1, int(PRINT_PROGRESS_EVERY_N_BATCHES)),
        progress_log_tail_rows=max(1, int(PROGRESS_LOG_TAIL_ROWS)),
        device=select_device(DEVICE_MODE),
    )


def model_batch_size(model_name: str) -> int:
    return int(MODEL_TRAINING_SETTINGS[model_name]["batch_size"])


def model_architecture_profile(model_name: str) -> dict[str, object]:
    # Return a readable architecture summary for reporting tables.
    return MODEL_ARCHITECTURE_PROFILES.get(
        model_name,
        {
            "architecture": "unknown",
            "input_view": "unknown",
            "input_channels": "unknown",
            "spatial_encoder": "unknown",
            "temporal_encoder": "unknown",
            "embed_dim": "unknown",
            "hidden_dim": "unknown",
            "transformer_layers": "unknown",
            "attention_heads": "unknown",
            "ffn_dim": "unknown",
            "patch_size": "unknown",
            "dropout": "unknown",
            "pooling": "unknown",
        },
    )


def model_epoch_count(model_name: str, config: ExperimentConfig | None = None) -> int:
    # Smoke mode and EPOCHS_OVERRIDE use one shared value; full mode uses the model row above.
    config = CONFIG if config is None else config
    if EPOCHS_OVERRIDE is not None:
        return int(EPOCHS_OVERRIDE)
    if config.run_mode != "full":
        return int(config.epochs)
    return int(MODEL_TRAINING_SETTINGS[model_name]["full_epochs"])


def model_training_profile(model_name: str, config: ExperimentConfig | None = None) -> dict[str, object]:
    # Convert one readable model row into the profile consumed by the training loop.
    config = CONFIG if config is None else config
    settings = MODEL_TRAINING_SETTINGS[model_name]
    epochs = model_epoch_count(model_name, config)
    warmup_cap = int(settings["warmup_epoch_cap"])
    warmup_epochs = min(warmup_cap, max(1, epochs // 5)) if warmup_cap else 0

    profile: dict[str, object] = {
        "epochs": epochs,
        "learning_rate": float(settings["learning_rate"]),
        "weight_decay": float(settings["weight_decay"]),
        "label_smoothing": float(settings["label_smoothing"]),
        "clip_grad_norm": settings["clip_grad_norm"],
        "warmup_epochs": warmup_epochs,
        "use_cosine": bool(settings["use_cosine"]),
        "augmentation": bool(settings["augmentation"]),
        "motion_input": bool(settings["motion_input"]),
    }

    internal_keys = {
        "batch_size",
        "full_epochs",
        "learning_rate",
        "weight_decay",
        "label_smoothing",
        "clip_grad_norm",
        "warmup_epoch_cap",
        "use_cosine",
        "augmentation",
        "motion_input",
    }
    profile.update({key: value for key, value in settings.items() if key not in internal_keys})
    if model_name in KD_STUDENT_MODELS:
        profile["teacher_training_profile"] = model_training_profile(str(settings["teacher_model"]), config)
    return profile


def config_summary(config: ExperimentConfig | None = None) -> dict[str, object]:
    # Return a concise configuration summary for reporting and verification.
    config = CONFIG if config is None else config
    return {
        "device": str(config.device),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "device_mode": DEVICE_MODE,
        "run_mode": config.run_mode,
        "forward_only": config.forward_only,
        "transformer_motion_input": USE_TRANSFORMER_MOTION_INPUT,
        "transformer_augmentation": USE_TRANSFORMER_AUGMENTATION,
        "model_full_epochs": {
            model_name: MODEL_TRAINING_SETTINGS[model_name]["full_epochs"]
            for model_name in config.models
        },
        "paper_kd_teacher_model": PAPER_KD_TEACHER_MODEL,
        "paper_kd_temperature": PAPER_KD_TEMPERATURE,
        "paper_kd_alpha": PAPER_KD_ALPHA,
        "loss_compute_device": "cpu" if config.device.type == "mps" else str(config.device),
        "enabled_models": list(config.models),
        "data_path": str(config.data_path),
        "cache_dir": str(CACHE_DIR),
        "use_run_cache": USE_RUN_CACHE,
        "save_run_cache": SAVE_RUN_CACHE,
        "skip_train_if_results_exist": SKIP_TRAIN_IF_RESULTS_EXIST,
        "resume_from_checkpoint": RESUME_FROM_CHECKPOINT,
        "save_checkpoints": SAVE_CHECKPOINTS,
        "force_retrain": FORCE_RETRAIN,
        "epochs_override": EPOCHS_OVERRIDE,
        "smoke_epochs": config.epochs if config.run_mode == "smoke" else None,
        "train_limit": config.train_limit,
        "val_limit": config.val_limit,
        "test_limit": config.test_limit,
        "progress_every_n_batches": config.progress_every_n_batches,
        "print_progress_every_n_batches": config.print_progress_every_n_batches,
    }


def print_config_summary(config: ExperimentConfig | None = None) -> None:
    for key, value in config_summary(config).items():
        print(f"{key}: {value}")


CONFIG = make_config()
