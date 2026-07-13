from __future__ import annotations

from collections import Counter
from pathlib import Path
import os
import random
import re

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

from experiment_config import model_batch_size

# -----------------------------
# Data split and loading helpers
# -----------------------------

def set_seed(seed: int) -> None:
    # Seed the supported random number generators for reproducible experiments.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_limit(profile_value: int | None, override_value: int | None) -> int | None:
    # Interpret 0 as a request to use the complete split.
    if override_value is None:
        return profile_value
    return None if override_value == 0 else override_value


def parse_label(file_name: str) -> str:
    # Extract the gesture label from the dataset file name.
    if file_name.startswith("n_"):
        return "n"
    end = file_name.find("_e")
    if end == -1:
        raise ValueError(f"Cannot parse label from file name: {file_name}")
    return file_name[2:end]


def parse_e_id(file_name: str) -> int:
    # Extract the full environment/session id after `_e`, not only one digit.
    match = re.search(r"_e(\d+)", file_name)
    if match is None:
        raise ValueError(f"Cannot parse environment id from file name: {file_name}")
    return int(match.group(1))


def collect_split(data_path: Path, keep_e_ids: set[int], label_encoder) -> tuple[list[str], np.ndarray]:
    files = sorted(
        name
        for name in os.listdir(data_path)
        if name.endswith(".npy") and parse_e_id(name) in keep_e_ids
    )
    labels = label_encoder.transform([parse_label(name) for name in files])
    return files, labels


def maybe_stratified_limit(
    files: list[str],
    labels: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> tuple[list[str], np.ndarray]:
    # Keep class proportions when a small smoke split is requested.
    if max_samples is None or max_samples >= len(files):
        return files, labels

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=seed)
    indices = np.arange(len(files))
    selected, _ = next(splitter.split(indices, labels))
    return [files[i] for i in selected], labels[selected]


def summarize_split(split_name: str, labels: np.ndarray, label_encoder) -> None:
    counts = Counter(label_encoder.inverse_transform(labels))
    print(f"{split_name}: {len(labels)} samples")
    print(dict(counts))


def load_sequence_tensor(root_path: Path, file_name: str, seq_len: int) -> torch.Tensor:
    # Load one DRAI sequence and return `[time, channel, height, width]`.
    frames = np.load(root_path / file_name).astype(np.float32)

    if len(frames) > seq_len:
        center = len(frames) // 2
        start = max(0, center - seq_len // 2)
        frames = frames[start : start + seq_len]

    if len(frames) < seq_len:
        pad = np.zeros((seq_len - len(frames), 32, 32), dtype=np.float32)
        frames = np.concatenate([frames, pad], axis=0)

    return torch.from_numpy(frames).unsqueeze(1)


class GestureSequenceDataset(Dataset):
    # Single-stream DRAI sequence dataset shared by the evaluated models.
    def __init__(self, data_path: Path, files: list[str], labels: np.ndarray, seq_len: int = 32):
        self.data_path = data_path
        self.files = files
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = load_sequence_tensor(self.data_path, self.files[index], self.seq_len)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y




def make_loader(
    data_path: Path,
    files: list[str],
    labels: np.ndarray,
    seq_len: int,
    model_name: str,
    shuffle: bool,
) -> DataLoader:
    # Batch size is owned by experiment_config.py.
    dataset = GestureSequenceDataset(data_path, files, labels, seq_len=seq_len)
    return DataLoader(
        dataset,
        batch_size=model_batch_size(model_name),
        shuffle=shuffle,
        num_workers=0,
    )


def prepare_splits(config, label_encoder):
    # Split by environment id, then optionally apply stratified smoke limits.
    train_files, train_labels = collect_split(config.data_path, {2, 3, 4}, label_encoder)
    val_files, val_labels = collect_split(config.data_path, {1}, label_encoder)
    test_files, test_labels = collect_split(config.data_path, {6}, label_encoder)

    train_files, train_labels = maybe_stratified_limit(train_files, train_labels, config.train_limit, config.seed)
    val_files, val_labels = maybe_stratified_limit(val_files, val_labels, config.val_limit, config.seed)
    test_files, test_labels = maybe_stratified_limit(test_files, test_labels, config.test_limit, config.seed)
    return train_files, train_labels, val_files, val_labels, test_files, test_labels
