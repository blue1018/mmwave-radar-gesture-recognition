# DI-Gesture  README

This repository contains the code and experimental results for the research project:

Research on the Application of Millimeter-Wave Radar Gesture Recognition in Public Scenarios: Improving with Transformer

The project investigates whether Transformer-based architectures can improve millimeter-wave radar gesture recognition under a shared data-processing and evaluation pipeline.

The project uses the DI-Gesture millimeter-wave radar dataset.

The dataset is not included in this repository.

The original dataset is provided for scientific research and requires a separate access application. Users must follow the original dataset provider's access conditions and licensing requirements.

## Overview

Touchless interaction can be useful in public environments such as transport systems, hospitals, public terminals, smart buildings, and accessibility devices.

Camera-based gesture-recognition systems may raise privacy concerns because they capture identifiable visual information. Millimeter-wave radar mainly captures motion and spatial information, making it a potentially suitable sensing technology for privacy-sensitive public environments.

This research compares convolutional, recurrent, lightweight attention-based, and Transformer-based models using Dynamic Range-Angle Image, or DRAI, sequences generated from FMCW millimeter-wave radar signals.

This project is for mmWave radar gesture recognition. It first converts raw `.npy` radar data into DRAI representations, then trains and compares multiple deep learning models in `experiment1/`.



## Directory

| Path | Purpose |
|---|---|
| `processing/` | Raw radar data inspection and DRAI generation. |
| `experiment1/` | Data loading, model definitions, training, evaluation, caching, and reporting. |
| `README.md` | Full  README. |

## Run Order

1. Convert DRAI:

```text
processing/convertdrai.ipynb
```

2. If using the external pretrained MobileViT baseline, download the weights first:

```bash
/opt/anaconda3/envs/digesture/bin/python experiment1/download_external_mobilevit_weights.py
```

3. Train, evaluate, and generate results:

```text
experiment1/di_gesture_experiment.ipynb
```

## Key Paths

| Item | Current setting |
|---|---|
| DRAI data path | `/Users/blue/NCI/MCD-Gesture_ProcessedDataset` |
| Experiment cache path | `/Users/blue/Desktop/Practicum/code/experiment1/cache` |
| Full-run result path | `experiment1/cache/full/<model>/<cache_key>/result.json` |
| Main configuration file | `experiment1/experiment_config.py` |

## Current Experiment Configuration

| Setting | Current value |
|---|---|
| `RUN_MODE` | `"full"` |
| `DEVICE_MODE` | `"mps"` |
| `SEED` | `21` |
| `SEQ_LEN` | `32` |
| `IMAGE_SIZE` | `32` |
| `NUM_CLASSES` | `7` |
| `USE_RUN_CACHE` | `True` |
| `SAVE_RUN_CACHE` | `True` |
| `SKIP_TRAIN_IF_RESULTS_EXIST` | `True` |
| `RESUME_FROM_CHECKPOINT` | `True` |
| `SAVE_CHECKPOINTS` | `True` |
| `FORCE_RETRAIN` | `False` |

By default, existing cached results are reused first. To force retraining, set `FORCE_RETRAIN` to `True`.

## Enabled Models

The current `ENABLED_MODELS` list includes:

| Model key | Display name |
|---|---|
| `crnn` | CRNN Baseline |
| `cnn_transformer` | CNN + Transformer |
| `trans_cnn_1d` | TRANS-CNN-1D |
| `lpvt_full` | LPVT-Full |
| `mobilevit_full` | MobileViT (Project Full) |
| `paper_mobilevit_teacher` | MobileViT (Paper Teacher) |
| `paper_mobilevit_student` | MobileViT (Paper Student + KD) |
| `external_mobilevit_frozen` | MobileViT (External Frozen) |
| `timesformer_full` | TimeSformer-Full |

Note: keep `paper_mobilevit_teacher` before `paper_mobilevit_student`, because the student model reads the teacher checkpoint during distillation.

## Data Split

| Split | Environment |
|---|---|
| Train | `e2`, `e3`, `e4` |
| Validation | `e1` |
| Test | `e6` |

The split separates data by environment ID to test generalization across different recording environments.

## Gesture Classes

```text
SlideRight, SlideLeft, Push, Pull, Clockwise, Counterclockwise, n
```

Here, `n` is the background/non-gesture class.

## Full-Run Result Snapshot

The current full cache contains 9 model results. Ranked by test balanced accuracy, `test_bal_acc`:

| Rank | Model | test_bal_acc | test_acc | macro_f1 |
|---:|---|---:|---:|---:|
| 1 | LPVT-Full | 0.9466 | 0.9600 | 0.9550 |
| 2 | MobileViT (Project Full) | 0.9266 | 0.9479 | 0.9367 |
| 3 | CNN + Transformer | 0.9147 | 0.9433 | 0.9257 |

This project prioritizes `test_bal_acc` because the background class `n` is much larger than the gesture classes, so plain accuracy can be biased by class imbalance.

## Minimal Dependency Information

| Package | Version |
|---|---|
| Python | 3.8.20 |
| PyTorch | 2.0.1 |
| TorchVision | 0.15.2 |
| NumPy | 1.24.3 |
| Pandas | 2.0.3 |
| Scikit-learn | 1.3.2 |
| Matplotlib | 3.7.2 |
| Transformers | 4.46.3 |


## Citation

When referencing this work, please cite:

```
@mastersthesis{he2026mmwave,
  author  = {Zhanjun He},
  title   = {Research on the Application of Millimeter-Wave Radar Gesture Recognition in Public Scenarios: Improving with Transformer},
  school  = {National College of Ireland},
  year    = {2026},
  type    = {MSc Research Practicum}
}
```

## License

The source code in this repository is released under the MIT License unless otherwise stated.

The dataset is not covered by the repository's MIT License. Dataset access, use, and redistribution remain subject to the original dataset provider's conditions.

## Author

Zhanjun He

MSc in Artificial Intelligence National College of Ireland