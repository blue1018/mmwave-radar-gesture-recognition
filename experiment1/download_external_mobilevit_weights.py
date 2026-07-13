#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_MODEL_ID = "apple/mobilevit-small"


def dependency_message() -> str:
    return (
        "Missing Hugging Face dependencies.\n"
        "Install them in the same Python environment used by the notebook, for example:\n\n"
        f"  {sys.executable} -m pip install -U transformers huggingface_hub safetensors\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify the external ImageNet-pretrained MobileViT weights."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id to download. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory. Omit to use the default ~/.cache/huggingface/hub cache.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None

    print(f"Python executable: {sys.executable}")
    print(f"Model id: {args.model_id}")
    print(f"Cache dir: {cache_dir if cache_dir else 'default Hugging Face cache'}")

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError:
        print(dependency_message(), file=sys.stderr)
        return 2

    try:
        local_path = snapshot_download(
            repo_id=args.model_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "pytorch_model.bin",
                "model.safetensors",
                "*.safetensors",
                "*.bin",
            ],
        )
    except Exception as exc:
        print(f"Download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Downloaded snapshot: {local_path}")

    try:
        from transformers import MobileViTForImageClassification
    except ModuleNotFoundError:
        print(dependency_message(), file=sys.stderr)
        return 2

    try:
        model = MobileViTForImageClassification.from_pretrained(
            args.model_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=True,
        )
    except Exception as exc:
        print(f"Local verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    image_size = getattr(model.config, "image_size", "unknown")
    num_labels = getattr(model.config, "num_labels", "unknown")
    param_count = sum(param.numel() for param in model.parameters()) / 1e6
    print("Local verification OK")
    print(f"Image size: {image_size}")
    print(f"Pretrained labels: {num_labels}")
    print(f"Parameters: {param_count:.3f}M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
