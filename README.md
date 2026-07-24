# README

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
| `README.md` | Full English README. |

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
| `USE_TRANSFORMER_MOTION_INPUT` | `True` |
| `USE_TRANSFORMER_AUGMENTATION` | `True` |
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

| Split | Environment | Samples | Background `n` | Gesture samples |
|---|---|---:|---:|---:|
| Train | `e2`, `e3`, `e4` | 13,300 | 7,600 | 5,700 |
| Validation | `e1` | 4,900 | 2,800 | 2,100 |
| Test | `e6` | 3,050 | 1,400 | 1,650 |

The split separates data by environment ID to test generalization across recording environments. The six gesture classes are balanced within each split, but the background class is larger, so balanced accuracy is the primary comparison metric.

## Gesture Classes

```text
SlideRight, SlideLeft, Push, Pull, Clockwise, Counterclockwise, n
```

Here, `n` is the background/non-gesture class.

## Full-Run Result Snapshot

This snapshot matches the active configuration and the latest executed notebook outputs as of 24 July 2026. The current run selects one configuration-matching result for each of the nine enabled models and ranks them by test balanced accuracy, `test_bal_acc`:

| Rank | Model | test_bal_acc | test_acc | macro_f1 |
|---:|---|---:|---:|---:|
| 1 | MobileViT (Project Full) | 0.9266 | 0.9479 | 0.9367 |
| 2 | CNN + Transformer | 0.9147 | 0.9433 | 0.9257 |
| 3 | LPVT-Full | 0.9076 | 0.9351 | 0.9201 |
| 4 | CRNN Baseline | 0.9062 | 0.9292 | 0.9082 |
| 5 | TRANS-CNN-1D | 0.8918 | 0.9246 | 0.9032 |
| 6 | MobileViT (Paper Teacher) | 0.8863 | 0.9138 | 0.8886 |
| 7 | MobileViT (Paper Student + KD) | 0.8418 | 0.8833 | 0.8485 |
| 8 | MobileViT (External Frozen) | 0.4509 | 0.6118 | 0.3917 |
| 9 | TimeSformer-Full | 0.4389 | 0.6148 | 0.4470 |

MobileViT (Project Full) improves over the CRNN baseline by 0.0204 balanced-accuracy points and 0.0285 macro-F1 points on `e6`.

Historical cache directories can coexist with the active results. In particular, the older LPVT-Full result with `test_bal_acc=0.9466` used augmentation, whereas the current LPVT-Full profile disables augmentation and yields 0.9076. Results from different cache keys or training profiles should not be mixed in one ranking.

## Secondary `e5` Stability Check

The notebook also evaluates the current checkpoints on the unseen `e5` environment without changing the primary split. `e5` contains 2,800 samples: 200 from each gesture class and 1,600 background samples.

| Rank | Model | e5_bal_acc | 95% bootstrap interval |
|---:|---|---:|---:|
| 1 | MobileViT (Project Full) | 0.9921 | [0.9876, 0.9963] |
| 2 | CNN + Transformer | 0.9895 | [0.9846, 0.9937] |
| 3 | LPVT-Full | 0.9870 | [0.9812, 0.9923] |

The intervals use 5,000 stratified bootstrap resamples. MobileViT's paired balanced-accuracy delta intervals versus CNN + Transformer and LPVT-Full include zero, so this bootstrap check does not establish separation among the top three on balanced accuracy. Exact McNemar tests on sample-wise correctness are significant for those selected comparisons, but they test a different metric-level question.

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

MSc in Artificial Intelligence, National College of Ireland
