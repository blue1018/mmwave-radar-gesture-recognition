from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / "experiment1" / "cache" / "full"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "pdf"
RAW_DATA_GB = 600
DRAI_DATA_GB = 6

FONT_CANDIDATES = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]

FAMILY_COLORS = {
    "Baseline": "#6C757D",
    "Transformer": "#2563A6",
    "MobileViT": "#2F8F63",
    "External Transfer": "#B65A50",
    "Other": "#8A6F36",
}


def load_font() -> font_manager.FontProperties:
    for font_path in FONT_CANDIDATES:
        path = Path(font_path)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=str(path))
    return font_manager.FontProperties()


def model_family(model_key: str) -> str:
    if model_key == "crnn":
        return "Baseline"
    if model_key.startswith("external_"):
        return "External Transfer"
    if "mobilevit" in model_key:
        return "MobileViT"
    if any(token in model_key for token in ("transformer", "trans_", "lpvt", "timesformer")):
        return "Transformer"
    return "Other"


def load_results(cache_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result_path in sorted(cache_dir.glob("*/*/result.json")):
        with result_path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        row["family"] = model_family(str(row.get("model", "")))
        row["cache_path"] = str(result_path.parent)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No result.json files found under {cache_dir}")
    return sorted(rows, key=lambda item: float(item.get("test_bal_acc", float("nan"))), reverse=True)


def pct(value: float | int | None) -> str:
    if value is None or math.isnan(float(value)):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def runtime_label(seconds: float | int | None) -> str:
    if seconds is None or math.isnan(float(seconds)):
        return "n/a"
    seconds = float(seconds)
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


def params_label(params_m: float | int | None) -> str:
    if params_m is None or math.isnan(float(params_m)):
        return "n/a"
    return f"{float(params_m):.3f} M"


def total_train_seconds(rows: list[dict[str, object]]) -> float:
    return sum(float(row.get("train_seconds", 0.0)) for row in rows)


def markdown_table(rows: list[dict[str, object]], lang: str = "zh") -> str:
    if lang == "en":
        lines = [
            "| Rank | Model | Bal. Acc | Acc | Macro F1 | Params | Runtime |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            "| 排名 | 模型 | 平衡准确率 | 准确率 | Macro F1 | 参数量 | 训练时间 |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {model} | {bal} | {acc} | {f1} | {params} | {runtime} |".format(
                rank=rank,
                model=row["display_name"],
                bal=pct(row["test_bal_acc"]),
                acc=pct(row["test_acc"]),
                f1=pct(row["test_macro_f1"]),
                params=params_label(row["params_m"]),
                runtime=runtime_label(row["train_seconds"]),
            )
        )
    return "\n".join(lines)


def write_summary_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fields = [
        "rank",
        "model",
        "display_name",
        "family",
        "test_bal_acc",
        "test_acc",
        "test_macro_f1",
        "params_m",
        "train_seconds",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "model": row.get("model", ""),
                    "display_name": row.get("display_name", ""),
                    "family": row.get("family", ""),
                    "test_bal_acc": row.get("test_bal_acc", ""),
                    "test_acc": row.get("test_acc", ""),
                    "test_macro_f1": row.get("test_macro_f1", ""),
                    "params_m": row.get("params_m", ""),
                    "train_seconds": row.get("train_seconds", ""),
                }
            )


def report_values(rows: list[dict[str, object]]) -> dict[str, object]:
    best = rows[0]
    baseline = next(row for row in rows if row["model"] == "crnn")
    project_mobilevit = next(row for row in rows if row["model"] == "mobilevit_full")
    cnn_transformer = next(row for row in rows if row["model"] == "cnn_transformer")
    teacher = next(row for row in rows if row["model"] == "paper_mobilevit_teacher")
    student = next(row for row in rows if row["model"] == "paper_mobilevit_student")
    external = next(row for row in rows if row["model"] == "external_mobilevit_frozen")
    student_drop = (float(teacher["test_bal_acc"]) - float(student["test_bal_acc"])) * 100
    student_param_ratio = float(student["params_m"]) / float(teacher["params_m"])
    transformer_gain = (float(cnn_transformer["test_bal_acc"]) - float(baseline["test_bal_acc"])) * 100
    data_reduction = (1 - DRAI_DATA_GB / RAW_DATA_GB) * 100

    return {
        "best": best,
        "baseline": baseline,
        "project_mobilevit": project_mobilevit,
        "cnn_transformer": cnn_transformer,
        "teacher": teacher,
        "student": student,
        "external": external,
        "student_drop": student_drop,
        "student_param_ratio": student_param_ratio,
        "transformer_gain": transformer_gain,
        "total_runtime": total_train_seconds(rows),
        "data_reduction": data_reduction,
    }


def build_report_script_zh(rows: list[dict[str, object]]) -> str:
    values = report_values(rows)
    best = values["best"]
    baseline = values["baseline"]
    project_mobilevit = values["project_mobilevit"]
    cnn_transformer = values["cnn_transformer"]
    teacher = values["teacher"]
    student = values["student"]
    external = values["external"]

    return f"""# 实验结果汇报稿

## 1. 研究概述

本实验比较了 9 个以动态距离–角度图（DRAI）为输入的毫米波雷达手势识别模型。实验流程包括雷达信号处理以及模型训练与评估。约 **{RAW_DATA_GB} GB** 的原始雷达数据经过 Range FFT、Doppler FFT、动态 Doppler 选择、Angle FFT 和 DRAI 聚合后，转换为约 **{DRAI_DATA_GB} GB** 的 DRAI 数据，数据量减少约 **{values["data_reduction"]:.0f}%**。该表示保留了与手势运动相关的距离和角度信息，同时降低了后续训练的数据读取与存储负担。

模型评估采用 full-run 设置，9 个模型的训练与测试累计耗时约 **{runtime_label(values["total_runtime"])}**。主要评价指标为测试集平衡准确率，因为数据包含样本量较大的非手势类别 `n`，普通准确率更容易受到类别分布影响。

## 2. 数据处理流程

数据处理从原始 mmWave radar `.npy` 文件开始。每一帧雷达张量先经过 Range FFT 提取距离信息，再经过 Doppler FFT 表征运动信息。随后抑制接近零 Doppler 的静态背景，并选择与手势运动相关的动态 Doppler bins。最后，Angle FFT 提取角度信息，动态响应被聚合为 `32 × 32` 的 DRAI 序列。

该流程将约 **{RAW_DATA_GB} GB** 的原始雷达数据转换为约 **{DRAI_DATA_GB} GB** 的 DRAI 数据。DRAI 不保留完整的复数雷达张量，而是保留与动态手势相关的距离–角度表示，从而减少后续 CNN、Transformer 和 MobileViT 模型的数据读取开销。

## 3. 总体结果

表现最佳的模型为 **{best["display_name"]}**，测试集平衡准确率为 **{pct(best["test_bal_acc"])}**，准确率为 **{pct(best["test_acc"])}**，macro F1 为 **{pct(best["test_macro_f1"])}**。在本实验配置下，LPVT 结构取得了最高的综合分类性能。

排名第二的是 **{project_mobilevit["display_name"]}**，平衡准确率为 **{pct(project_mobilevit["test_bal_acc"])}**。该结果表明，本项目的 MobileViT 实现也能有效建模 DRAI 序列。

## 4. 实用性对比

LPVT-Full 取得最高准确率，而 **CNN + Transformer** 和 **CRNN Baseline** 在性能与计算成本之间表现出更均衡的权衡。CNN + Transformer 的平衡准确率为 **{pct(cnn_transformer["test_bal_acc"])}**，比 CRNN 高 **{values["transformer_gain"]:.2f} 个百分点**，训练时间则明显低于 LPVT-Full 和 MobileViT-Full。

CRNN 的 balanced accuracy 为 **{pct(baseline["test_bal_acc"])}**，参数量只有 **{params_label(baseline["params_m"])}**，训练时间约 **{runtime_label(baseline["train_seconds"])}**。因此它可以作为轻量、稳定、容易复现的基础模型。

## 5. MobileViT 与知识蒸馏

MobileViT Teacher 的平衡准确率为 **{pct(teacher["test_bal_acc"])}**，Student + KD 为 **{pct(student["test_bal_acc"])}**。学生模型比教师低 **{values["student_drop"]:.2f} 个百分点**，但参数量仅为教师的 **{values["student_param_ratio"]:.0%}**。知识蒸馏实现了明显的模型压缩，但当前配置仍存在性能损失。

## 6. 外部预训练模型观察

External Frozen MobileViT 的平衡准确率为 **{pct(external["test_bal_acc"])}**，运行时间约为 **{runtime_label(external["train_seconds"])}**。结果表明，冻结的 ImageNet 特征对 DRAI 的直接迁移能力有限。后续研究可评估部分解冻、雷达域微调或雷达数据预训练。

## 7. 结论

DRAI 处理将约 600 GB 的原始雷达数据转换为约 6 GB 的结构化输入。LPVT-Full 在本实验中取得最高准确率，但训练成本较高。CNN + Transformer 和 CRNN 提供了更具效率的基线。MobileViT 与知识蒸馏实现了较小模型，但外部冻结图像特征仍需要更充分的雷达域适配。

## 结果表

{markdown_table(rows, lang="zh")}
"""


def build_report_script_en(rows: list[dict[str, object]]) -> str:
    values = report_values(rows)
    best = values["best"]
    baseline = values["baseline"]
    project_mobilevit = values["project_mobilevit"]
    cnn_transformer = values["cnn_transformer"]
    teacher = values["teacher"]
    student = values["student"]
    external = values["external"]

    return f"""# Experiment Presentation Script

## Main Presentation

Good morning everyone.

This project investigates hand gesture recognition using mmWave radar data.

The signal-processing pipeline converted approximately **{RAW_DATA_GB} GB** of raw radar data into **{DRAI_DATA_GB} GB** of DRAI data.

Each DRAI sample is a sequence of `32 × 32` range–angle maps that retain motion-related information in a compact representation.

I then trained and compared **9 models**. The complete experiment required approximately **36 hours**.

The best-performing model was **{best["display_name"]}**, which achieved **{pct(best["test_bal_acc"])}** test balanced accuracy.

The second best model was **{project_mobilevit["display_name"]}**, with **{pct(project_mobilevit["test_bal_acc"])}** balanced accuracy.

**CNN + Transformer** and **CRNN Baseline** provided stronger efficiency trade-offs because they were smaller and faster. CRNN achieved **{pct(baseline["test_bal_acc"])}** balanced accuracy.

Knowledge distillation substantially reduced model size, although balanced accuracy decreased from **{pct(teacher["test_bal_acc"])}** for the teacher to **{pct(student["test_bal_acc"])}** for the student.

The externally pretrained frozen MobileViT achieved **{pct(external["test_bal_acc"])}**, indicating limited direct transfer from ImageNet features to radar DRAI data.

In conclusion, DRAI provided a compact radar representation, LPVT-Full achieved the highest accuracy, and CNN + Transformer and CRNN offered more efficient alternatives.

Thank you.

## Condensed Presentation

This project uses mmWave radar data for hand gesture recognition.

The processing pipeline converted approximately **600 GB** of raw radar data into **6 GB** of DRAI data.

Then I compared **9 models**, and the full experiment took about **36 hours**.

The best model was **LPVT-Full**, with **94.66%** balanced accuracy.

MobileViT-Full was second, with **92.66%**.

CRNN and CNN + Transformer achieved lower accuracy but required less computation.

Overall, DRAI reduced the data volume substantially, and LPVT-Full achieved the strongest result under the evaluated configuration.

## Reference Values

- Raw data: **{RAW_DATA_GB}GB**
- DRAI data: **{DRAI_DATA_GB}GB**
- Full experiment time: **about 36 hours**
- Best model: **{best["display_name"]}**
- Best balanced accuracy: **{pct(best["test_bal_acc"])}**
- Practical baseline: **CRNN Baseline**, **{pct(baseline["test_bal_acc"])}**
"""


def wrap_text(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=False))


def add_box(ax, x: float, y: float, width: float, height: float, title: str, body: str, font: font_manager.FontProperties) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=0.9,
        edgecolor="#D0D7DE",
        facecolor="#FFFFFF",
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(box)
    ax.text(x + 0.014, y + height - 0.055, title, transform=ax.transAxes, fontproperties=font, fontsize=13, weight="bold", color="#111827")
    ax.text(x + 0.014, y + height - 0.118, body, transform=ax.transAxes, fontproperties=font, fontsize=8.2, color="#374151", va="top", linespacing=1.3)


def draw_poster(rows: list[dict[str, object]], png_path: Path, pdf_path: Path, lang: str = "zh") -> None:
    font = load_font()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    best = rows[0]
    baseline = next(row for row in rows if row["model"] == "crnn")
    teacher = next(row for row in rows if row["model"] == "paper_mobilevit_teacher")
    student = next(row for row in rows if row["model"] == "paper_mobilevit_student")
    student_drop = (float(teacher["test_bal_acc"]) - float(student["test_bal_acc"])) * 100
    total_runtime = total_train_seconds(rows)
    runtime_card = f"~{round(total_runtime / 3600):.0f} h"

    if lang == "en":
        title = "DRAI Gesture Recognition Model Comparison"
        subtitle = "Full run | 9 models | 600GB raw radar data -> 6GB DRAI | Metrics: balanced accuracy, Macro F1, size, runtime"
        metric_cards = [
            ("Raw data", "metadata", f"{RAW_DATA_GB} GB"),
            ("DRAI data", "processed", f"{DRAI_DATA_GB} GB"),
            ("Model runtime", "full run", runtime_card),
            ("Best model", str(best["display_name"]), pct(best["test_bal_acc"])),
            ("Strong baseline", "CRNN Baseline", pct(baseline["test_bal_acc"])),
        ]
        rank_title = "Model Ranking"
        rank_xlabel = "Test balanced accuracy (%)"
        scatter_title = "Accuracy - Size - Runtime"
        scatter_xlabel = "Parameters (M, log scale)"
        scatter_ylabel = "Balanced accuracy (%)"
        bottom_boxes = [
            (
                "Data Processing",
                "\n".join(
                    [
                        f"{RAW_DATA_GB}GB raw radar metadata",
                        "Range / Doppler / Angle FFT",
                        "Dynamic Doppler-bin selection",
                        f"Output: {DRAI_DATA_GB}GB DRAI sequences",
                    ]
                ),
            ),
            (
                "Model Experiment",
                "\n".join(
                    [
                        "9 models in full-run mode",
                        f"Total runtime: about {runtime_card}",
                        "Ranking uses balanced accuracy",
                        "Cached results support reproducible reporting",
                    ]
                ),
            ),
            (
                "Main Findings",
                "\n".join(
                    [
                        f"LPVT-Full leads: {pct(best['test_bal_acc'])}",
                        "MobileViT-Full is close behind",
                        "CNN+Transformer and CRNN are practical",
                        "CRNN remains a strong low-cost baseline",
                    ]
                ),
            ),
            (
                "Interpretation",
                "\n".join(
                    [
                        f"KD student drop: {student_drop:.2f} pp",
                        f"Student size: {params_label(student['params_m'])}",
                        "Frozen ImageNet transfer is weak",
                        "Next: KD tuning and radar-domain adaptation",
                    ]
                ),
            ),
        ]
    else:
        title = "DRAI 手势识别模型对比实验"
        subtitle = "Full run | 9 个模型 | 600GB 原始雷达数据 -> 6GB DRAI | 指标: 平衡准确率、Macro F1、参数量、训练时间"
        metric_cards = [
            ("原始数据", "雷达元数据", f"{RAW_DATA_GB} GB"),
            ("DRAI 数据", "处理后数据", f"{DRAI_DATA_GB} GB"),
            ("模型耗时", "完整 full run", runtime_card),
            ("最佳模型", str(best["display_name"]), pct(best["test_bal_acc"])),
            ("强基线", "CRNN Baseline", pct(baseline["test_bal_acc"])),
        ]
        rank_title = "模型排名"
        rank_xlabel = "Test balanced accuracy (%)"
        scatter_title = "精度 - 参数量 - 时间成本"
        scatter_xlabel = "Parameters (M, log scale)"
        scatter_ylabel = "Balanced accuracy (%)"
        bottom_boxes = [
            (
                "数据处理",
                "\n".join(
                    [
                        f"{RAW_DATA_GB}GB 原始雷达元数据",
                        "Range / Doppler / Angle FFT",
                        "动态 Doppler bins 选择",
                        f"输出: {DRAI_DATA_GB}GB DRAI 序列",
                    ]
                ),
            ),
            (
                "模型实验",
                "\n".join(
                    [
                        "9 个模型 full run 对比",
                        f"总耗时: 约 {runtime_card}",
                        "使用 balanced accuracy 排名",
                        "缓存结果可快速重生成报告",
                    ]
                ),
            ),
            (
                "主要发现",
                "\n".join(
                    [
                        f"LPVT-Full 最高: {pct(best['test_bal_acc'])}",
                        "MobileViT-Full 接近第一名",
                        "CNN+Transformer 与 CRNN 更实用",
                        "CRNN 是低成本强基线",
                    ]
                ),
            ),
            (
                "解释与下一步",
                "\n".join(
                    [
                        f"KD 学生下降: {student_drop:.2f} pp",
                        f"学生参数量: {params_label(student['params_m'])}",
                        "Frozen ImageNet 迁移较弱",
                        "下一步: KD 调参和雷达域适配",
                    ]
                ),
            ),
        ]

    fig = plt.figure(figsize=(16, 9), dpi=180)
    fig.patch.set_facecolor("#F6F8FA")

    title_ax = fig.add_axes([0.045, 0.84, 0.91, 0.12])
    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.72,
        title,
        fontproperties=font,
        fontsize=28 if lang == "en" else 30,
        weight="bold",
        color="#111827",
        transform=title_ax.transAxes,
    )
    title_ax.text(
        0.0,
        0.28,
        subtitle,
        fontproperties=font,
        fontsize=12.5 if lang == "en" else 13,
        color="#4B5563",
        transform=title_ax.transAxes,
    )

    metric_ax = fig.add_axes([0.045, 0.695, 0.91, 0.12])
    metric_ax.axis("off")
    card_colors = ["#E8F1FB", "#EAF7F0", "#FFF4E6", "#FDECEC", "#EEF2FF"]
    for idx, (label, value, detail) in enumerate(metric_cards):
        x = idx * 0.196
        card = FancyBboxPatch(
            (x, 0.02),
            0.178,
            0.92,
            boxstyle="round,pad=0.014,rounding_size=0.018",
            linewidth=0,
            facecolor=card_colors[idx],
            transform=metric_ax.transAxes,
        )
        metric_ax.add_patch(card)
        metric_ax.text(x + 0.014, 0.72, label, fontproperties=font, fontsize=10.2, color="#4B5563", transform=metric_ax.transAxes)
        metric_ax.text(x + 0.014, 0.43, value, fontproperties=font, fontsize=11.6, weight="bold", color="#111827", transform=metric_ax.transAxes)
        metric_ax.text(x + 0.014, 0.15, detail, fontproperties=font, fontsize=17, weight="bold", color="#111827", transform=metric_ax.transAxes)

    rank_ax = fig.add_axes([0.055, 0.37, 0.43, 0.28])
    names = [str(row["display_name"]) for row in rows]
    scores = [float(row["test_bal_acc"]) * 100 for row in rows]
    families = [str(row["family"]) for row in rows]
    y_pos = list(range(len(rows)))[::-1]
    colors = [FAMILY_COLORS.get(family, FAMILY_COLORS["Other"]) for family in families]
    rank_ax.barh(y_pos, scores[::-1], color=colors[::-1], height=0.62)
    rank_ax.set_xlim(0, 100)
    rank_ax.set_yticks(y_pos)
    rank_ax.set_yticklabels([wrap_text(name, 22) for name in names[::-1]], fontproperties=font, fontsize=8.5)
    rank_ax.set_xlabel(rank_xlabel, fontproperties=font, fontsize=10)
    rank_ax.set_title(rank_title, fontproperties=font, fontsize=15, weight="bold", loc="left", pad=10)
    rank_ax.grid(axis="x", linestyle="--", alpha=0.28)
    rank_ax.spines[["top", "right", "left"]].set_visible(False)
    for y, score in zip(y_pos, scores[::-1]):
        rank_ax.text(min(score + 1.0, 98), y, f"{score:.1f}", va="center", fontproperties=font, fontsize=8.5, color="#111827")

    scatter_ax = fig.add_axes([0.545, 0.37, 0.41, 0.28])
    params = [float(row["params_m"]) for row in rows]
    bal_acc = [float(row["test_bal_acc"]) * 100 for row in rows]
    runtime = [float(row["train_seconds"]) for row in rows]
    runtime_log = [math.log1p(value) for value in runtime]
    rt_min, rt_max = min(runtime_log), max(runtime_log)
    sizes = [180 + (value - rt_min) / max(rt_max - rt_min, 1e-9) * 760 for value in runtime_log]
    for row, x, y, size in zip(rows, params, bal_acc, sizes):
        scatter_ax.scatter(
            x,
            y,
            s=size,
            color=FAMILY_COLORS.get(str(row["family"]), FAMILY_COLORS["Other"]),
            alpha=0.72,
            edgecolor="white",
            linewidth=1.2,
        )
        label = str(row["display_name"]).replace("MobileViT ", "MViT ")
        if x > 2.0 and y > 80:
            xytext = (-8, -10)
            ha = "right"
            va = "top"
        elif y < 55:
            xytext = (6, 5)
            ha = "left"
            va = "bottom"
        else:
            xytext = (5, 5)
            ha = "left"
            va = "bottom"
        scatter_ax.annotate(
            wrap_text(label, 18),
            (x, y),
            xytext=xytext,
            textcoords="offset points",
            fontproperties=font,
            fontsize=6.8,
            color="#111827",
            ha=ha,
            va=va,
        )
    scatter_ax.set_xscale("log")
    scatter_ax.set_xlabel(scatter_xlabel, fontproperties=font, fontsize=10)
    scatter_ax.set_ylabel(scatter_ylabel, fontproperties=font, fontsize=10)
    scatter_ax.set_title(scatter_title, fontproperties=font, fontsize=15, weight="bold", loc="left", pad=10)
    scatter_ax.grid(True, linestyle="--", alpha=0.25)
    scatter_ax.spines[["top", "right"]].set_visible(False)

    scatter_ax.set_ylim(10, 99.5)

    note_ax = fig.add_axes([0.045, 0.055, 0.91, 0.25])
    note_ax.axis("off")
    for idx, (box_title, box_body) in enumerate(bottom_boxes):
        add_box(note_ax, idx * 0.252, 0.02, 0.232, 0.92, box_title, box_body, font)

    for ext_path in (png_path, pdf_path):
        ext_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Chinese and English experiment-report assets from recorded results.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Directory containing full-run model cache folders.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated report and poster assets.")
    args = parser.parse_args()

    rows = load_results(args.cache_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report_zh_path = args.output_dir / "experiment_result_report_script_zh.md"
    report_en_path = args.output_dir / "experiment_result_report_script_en.md"
    csv_path = args.output_dir / "experiment_result_summary.csv"
    png_zh_path = args.output_dir / "experiment_result_poster_zh.png"
    pdf_zh_path = args.output_dir / "experiment_result_poster_zh.pdf"
    png_en_path = args.output_dir / "experiment_result_poster_en.png"
    pdf_en_path = args.output_dir / "experiment_result_poster_en.pdf"

    report_zh_path.write_text(build_report_script_zh(rows), encoding="utf-8")
    report_en_path.write_text(build_report_script_en(rows), encoding="utf-8")
    write_summary_csv(rows, csv_path)
    draw_poster(rows, png_zh_path, pdf_zh_path, lang="zh")
    draw_poster(rows, png_en_path, pdf_en_path, lang="en")

    print(f"Wrote Chinese report script: {report_zh_path}")
    print(f"Wrote English report script: {report_en_path}")
    print(f"Wrote summary CSV: {csv_path}")
    print(f"Wrote Chinese poster PNG: {png_zh_path}")
    print(f"Wrote Chinese poster PDF: {pdf_zh_path}")
    print(f"Wrote English poster PNG: {png_en_path}")
    print(f"Wrote English poster PDF: {pdf_en_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
