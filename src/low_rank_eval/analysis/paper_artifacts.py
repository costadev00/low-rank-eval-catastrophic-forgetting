from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from low_rank_eval.analysis.aggregate_results import load_run_results, select_best_manual
from low_rank_eval.config import ExperimentConfig, load_config

MANUAL_CONFIGS = ("uniform", "early_heavy", "middle_heavy", "late_heavy")
TASK_ORDERS = (("ifeval", "math"), ("math", "ifeval"))
DISPLAY_NAMES = {
    "uniform": "Uniform",
    "early_heavy": "Early-heavy",
    "middle_heavy": "Middle-heavy",
    "late_heavy": "Late-heavy",
}
ORDER_NAMES = {
    ("ifeval", "math"): "IFEval→Math",
    ("math", "ifeval"): "Math→IFEval",
}
EXPECTED_EVAL_ROWS = {"ifeval": 541, "gsm8k": 1319}

# Embed TrueType outlines rather than Type-3 glyphs and visually harmonize the
# plots with the paper. This is a publication constraint, not a custom palette.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif"})


def _logical_cell(run: dict[str, Any]) -> tuple[str, tuple[str, str], int]:
    return run["configuration"], tuple(run["task_order"]), int(run["seed"])


def validate_manual_seed42_runs(
    results_dir: str | Path,
    *,
    config_dir: str | Path = "configs",
) -> list[dict[str, Any]]:
    """Load and strictly validate the eight runs used by the first paper.

    The complete-study paper builder remains guarded separately. This validator
    intentionally accepts only the controlled manual sweep: four allocations,
    both task orders, and seed 42.
    """

    root = Path(results_dir)
    runs = load_run_results(root, config_dir=config_dir)
    expected = {
        (configuration, order, 42)
        for configuration in MANUAL_CONFIGS
        for order in TASK_ORDERS
    }
    by_cell = {_logical_cell(run): run for run in runs if _logical_cell(run) in expected}
    missing = sorted(expected - set(by_cell))
    if missing:
        formatted = [f"{name}/{'_to_'.join(order)}/seed{seed}" for name, order, seed in missing]
        raise RuntimeError("Missing paper run(s): " + ", ".join(formatted))
    selected = [by_cell[cell] for cell in sorted(expected)]

    models = {(run["model"], run["model_revision"]) for run in selected}
    parameter_counts = {int(run["rank_manifest"]["lora_parameters"]) for run in selected}
    base_scores = {
        tuple(sorted((key, float(value)) for key, value in run["base"]["benchmarks"].items()))
        for run in selected
    }
    token_budgets = {int(run["train_token_budget_per_task"]) for run in selected}
    if len(models) != 1 or len(parameter_counts) != 1 or len(base_scores) != 1:
        raise RuntimeError("Paper runs do not share model, parameter budget, and base scores")
    if token_budgets != {1_000_000}:
        raise RuntimeError(f"Unexpected supervised-token budgets: {sorted(token_budgets)}")

    for run in selected:
        manifest = run["rank_manifest"]
        if abs(float(manifest["difference_percent"])) > 0.01:
            raise RuntimeError(f"Rank budget mismatch in {run['run_id']}")
        if len(manifest["block_ranks"]) != 36 or len(manifest["module_ranks"]) != 72:
            raise RuntimeError(f"Unexpected rank manifest size in {run['run_id']}")
        if any(stage["status"] != "complete" for stage in run["training"].values()):
            raise RuntimeError(f"Incomplete training stage in {run['run_id']}")
        result_root = Path(run["_path"]).parent
        for stage_name in ("stage_1_evaluation", "stage_2_evaluation"):
            for benchmark, expected_rows in EXPECTED_EVAL_ROWS.items():
                metric_path = result_root / stage_name / f"{benchmark}.metrics.json"
                if not metric_path.exists():
                    raise RuntimeError(f"Missing evaluation artifact: {metric_path}")
                metric = json.loads(metric_path.read_text(encoding="utf-8"))
                if int(metric["examples"]) != expected_rows:
                    raise RuntimeError(
                        f"Unexpected {benchmark} row count in {metric_path}: {metric['examples']}"
                    )
        for scores in (
            run["base"]["benchmarks"],
            run["stage1"]["benchmarks"],
            run["stage2"]["benchmarks"],
        ):
            if set(scores) != {"ifeval", "gsm8k"}:
                raise RuntimeError(f"Unexpected benchmark set in {run['run_id']}")
            if any(not 0.0 <= float(score) <= 100.0 for score in scores.values()):
                raise RuntimeError(f"Benchmark score outside [0, 100] in {run['run_id']}")
    return selected


def protocol_frame(runs: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        configuration = str(run["configuration"])
        order = tuple(run["task_order"])
        first = order[0]
        first_benchmark = "ifeval" if first == "ifeval" else "gsm8k"
        base = run["base"]["benchmarks"]
        stage1 = run["stage1"]["benchmarks"]
        stage2 = run["stage2"]["benchmarks"]
        calibration_base = run["base"]["calibration_nll"]
        calibration_final = run["stage2"]["calibration_nll"]
        calibration_net = np.mean(
            [
                100.0 * (calibration_base[task] - calibration_final[task]) / calibration_base[task]
                for task in ("ifeval", "math")
            ]
        )
        benchmark_net = np.mean(
            [stage2[benchmark] - base[benchmark] for benchmark in ("ifeval", "gsm8k")]
        )
        elapsed = sum(
            float(run["training"][stage]["elapsed_seconds"]) for stage in ("stage1", "stage2")
        )
        peaks = [
            int(value)
            for stage in ("stage1", "stage2")
            for value in run["training"][stage].get("peak_memory_bytes_by_rank", [])
        ]
        rows.append(
            {
                "run_id": run["run_id"],
                "configuration": configuration,
                "configuration_label": DISPLAY_NAMES[configuration],
                "task_order": "_to_".join(order),
                "task_order_label": ORDER_NAMES[order],
                "seed": int(run["seed"]),
                "lora_parameters": int(run["rank_manifest"]["lora_parameters"]),
                "trainable_percent": float(
                    run["rank_manifest"]["instantiated_model"]["trainable_percent"]
                ),
                "base_ifeval": float(base["ifeval"]),
                "stage1_ifeval": float(stage1["ifeval"]),
                "stage2_ifeval": float(stage2["ifeval"]),
                "base_gsm8k": float(base["gsm8k"]),
                "stage1_gsm8k": float(stage1["gsm8k"]),
                "stage2_gsm8k": float(stage2["gsm8k"]),
                "first_task": first,
                "first_task_gain": float(stage1[first_benchmark] - base[first_benchmark]),
                "first_task_bwt": float(stage2[first_benchmark] - stage1[first_benchmark]),
                "first_task_forgetting": float(stage1[first_benchmark] - stage2[first_benchmark]),
                "final_ifeval_net": float(stage2["ifeval"] - base["ifeval"]),
                "final_gsm8k_net": float(stage2["gsm8k"] - base["gsm8k"]),
                "mean_final_net": float(benchmark_net),
                "mean_calibration_nll_improvement_percent": float(calibration_net),
                "utility": float(run["continual_learning"]["utility"]),
                "training_time_seconds": elapsed,
                "peak_memory_gib": max(peaks, default=0) / (1024**3),
            }
        )
    return pd.DataFrame(rows).sort_values(["configuration", "task_order"]).reset_index(drop=True)


def benchmark_frame(runs: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        order = tuple(run["task_order"])
        first_benchmark = "ifeval" if order[0] == "ifeval" else "gsm8k"
        base = run["base"]["benchmarks"]
        stage1 = run["stage1"]["benchmarks"]
        stage2 = run["stage2"]["benchmarks"]
        for benchmark in ("ifeval", "gsm8k"):
            after_learning = (
                stage1[benchmark] if benchmark == first_benchmark else stage2[benchmark]
            )
            forgetting = (
                max(base[benchmark], stage1[benchmark], stage2[benchmark]) - stage2[benchmark]
            )
            rows.append(
                {
                    "configuration": run["configuration"],
                    "task_order": "_to_".join(order),
                    "seed": int(run["seed"]),
                    "benchmark": benchmark,
                    "base": float(base[benchmark]),
                    "stage1": float(stage1[benchmark]),
                    "stage2": float(stage2[benchmark]),
                    "gain": float(after_learning - base[benchmark]),
                    "forgetting": float(forgetting),
                    "bwt": float(stage2[benchmark] - after_learning),
                    "net": float(stage2[benchmark] - base[benchmark]),
                }
            )
    return pd.DataFrame(rows).sort_values(["configuration", "task_order", "benchmark"])


def _save_figure(fig: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def _rank_vectors(runs: Iterable[dict[str, Any]]) -> dict[str, list[int]]:
    vectors: dict[str, list[int]] = {}
    for run in runs:
        configuration = str(run["configuration"])
        if configuration in vectors:
            continue
        ranks = {
            int(block): int(rank)
            for block, rank in run["rank_manifest"]["block_ranks"].items()
        }
        vectors[configuration] = [ranks[block] for block in range(len(ranks))]
    return vectors


def plot_teaser(protocols: pd.DataFrame, rank_vectors: dict[str, list[int]], output: Path) -> None:
    """Build the only main-text figure: allocation, acquisition, and backward transfer."""

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.0, 2.05),
        gridspec_kw={"width_ratios": (1.25, 1.0, 1.0)},
        constrained_layout=True,
    )
    linestyles = ("-", "--", "-.", ":")
    for linestyle, configuration in zip(linestyles, MANUAL_CONFIGS, strict=True):
        axes[0].step(
            range(36),
            rank_vectors[configuration],
            where="mid",
            linestyle=linestyle,
            linewidth=1.5,
            label=DISPLAY_NAMES[configuration],
        )
    axes[0].set(xlabel="Transformer block", ylabel="LoRA rank", xlim=(-0.5, 35.5), ylim=(2, 35))
    axes[0].set_xticks((0, 11, 23, 35))
    axes[0].set_yticks((8, 16, 32))
    axes[0].legend(
        frameon=False,
        fontsize=5.5,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        borderaxespad=0.0,
    )
    axes[0].text(-0.16, 1.03, "a", transform=axes[0].transAxes, fontweight="bold")

    y = np.arange(len(MANUAL_CONFIGS))
    order_styles = (("IFEval→Math", "o", -0.10), ("Math→IFEval", "^", 0.10))
    for order_label, marker, offset in order_styles:
        subset = protocols.set_index(["configuration", "task_order_label"])
        acquisition = [
            float(subset.loc[(configuration, order_label), "first_task_gain"])
            for configuration in MANUAL_CONFIGS
        ]
        axes[1].scatter(acquisition, y + offset, marker=marker, s=28, label=order_label, zorder=3)
    axes[1].axvline(0, color="0.25", linewidth=0.8)
    axes[1].set(
        xlabel="Task-1 gain (points)",
        yticks=y,
        yticklabels=[DISPLAY_NAMES[name] for name in MANUAL_CONFIGS],
        xlim=(-3.65, 0.25),
    )
    axes[1].invert_yaxis()
    axes[1].set_title("○ IFEval→Math    △ Math→IFEval", fontsize=6.5, pad=3)
    axes[1].text(-0.20, 1.03, "b", transform=axes[1].transAxes, fontweight="bold")

    for order_label, marker, offset in order_styles:
        subset = protocols.set_index(["configuration", "task_order_label"])
        bwt = [
            float(subset.loc[(configuration, order_label), "first_task_bwt"])
            for configuration in MANUAL_CONFIGS
        ]
        axes[2].scatter(bwt, y + offset, marker=marker, s=28, label=order_label, zorder=3)
    axes[2].axvline(0, color="0.25", linewidth=0.8)
    axes[2].set(
        xlabel="Task-1 BWT (points)",
        yticks=y,
        yticklabels=[],
        xlim=(-3.45, 0.65),
    )
    axes[2].invert_yaxis()
    axes[2].text(-0.20, 1.03, "c", transform=axes[2].transAxes, fontweight="bold")

    for axis in axes:
        axis.grid(axis="x", linewidth=0.45, alpha=0.35)
        axis.tick_params(labelsize=6.5)
        axis.xaxis.label.set_size(7)
        axis.yaxis.label.set_size(7)
    _save_figure(fig, output)


def plot_rank_by_depth(rank_vectors: dict[str, list[int]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.8, 2.7), constrained_layout=True)
    for linestyle, configuration in zip(("-", "--", "-.", ":"), MANUAL_CONFIGS, strict=True):
        axis.step(
            range(36),
            rank_vectors[configuration],
            where="mid",
            linestyle=linestyle,
            linewidth=1.8,
            label=DISPLAY_NAMES[configuration],
        )
    axis.set(
        title="Manual rank allocations",
        xlabel="Transformer block",
        ylabel="LoRA rank (q_proj and v_proj)",
        xlim=(-0.5, 35.5),
        ylim=(2, 35),
    )
    axis.set_xticks((0, 5, 11, 12, 17, 23, 24, 29, 35))
    axis.set_yticks((8, 16, 32))
    axis.legend(frameon=False, ncol=4, loc="upper center")
    axis.grid(axis="y", linewidth=0.5, alpha=0.35)
    _save_figure(fig, output)


def plot_stage_trajectories(protocols: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True, constrained_layout=True)
    stages = ("Base\n(S0)", "After task 1\n(S1)", "After task 2\n(S2)")
    for row_index, order in enumerate(("ifeval_to_math", "math_to_ifeval")):
        for column_index, benchmark in enumerate(("ifeval", "gsm8k")):
            axis = axes[row_index, column_index]
            subset = protocols[protocols.task_order == order].set_index("configuration")
            for linestyle, configuration in zip(
                ("-", "--", "-.", ":"), MANUAL_CONFIGS, strict=True
            ):
                values = [
                    float(subset.loc[configuration, f"{stage}_{benchmark}"])
                    for stage in ("base", "stage1", "stage2")
                ]
                axis.plot(
                    stages,
                    values,
                    marker="o",
                    markersize=3,
                    linewidth=1.3,
                    linestyle=linestyle,
                    label=DISPLAY_NAMES[configuration],
                )
            order_label = "IFEval→Math" if order == "ifeval_to_math" else "Math→IFEval"
            benchmark_label = (
                "IFEval strict accuracy" if benchmark == "ifeval" else "GSM8K exact match"
            )
            axis.set_title(f"{order_label}: {benchmark_label}", fontsize=8)
            axis.set_ylabel("Score (0–100)")
            axis.grid(axis="y", linewidth=0.45, alpha=0.35)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="outside upper center")
    _save_figure(fig, output)


def plot_first_task_interference(protocols: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True, constrained_layout=True)
    for axis, order, title in zip(
        axes,
        ("ifeval_to_math", "math_to_ifeval"),
        ("IFEval learned first", "Math learned first"),
        strict=True,
    ):
        subset = protocols[protocols.task_order == order].set_index("configuration")
        values = [float(subset.loc[name, "first_task_bwt"]) for name in MANUAL_CONFIGS]
        axis.barh([DISPLAY_NAMES[name] for name in MANUAL_CONFIGS], values)
        axis.axvline(0, color="0.25", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("BWT (after task 2 − after task 1, points)")
        axis.grid(axis="x", linewidth=0.45, alpha=0.35)
        for index, value in enumerate(values):
            horizontal_alignment = "left" if value < 0 else "right"
            axis.text(
                value + (0.04 if value < 0 else -0.04),
                index,
                f"{value:+.2f}",
                va="center",
                ha=horizontal_alignment,
                fontsize=7,
            )
    axes[0].set_xlim(-3.60, 0.25)
    axes[1].set_xlim(-0.55, 0.38)
    axes[0].invert_yaxis()
    _save_figure(fig, output)


def plot_final_net_heatmap(protocols: pd.DataFrame, output: Path) -> None:
    ordered_rows = []
    labels = []
    for configuration in MANUAL_CONFIGS:
        for order in ("ifeval_to_math", "math_to_ifeval"):
            row = protocols[
                (protocols.configuration == configuration) & (protocols.task_order == order)
            ].iloc[0]
            ordered_rows.append([row.final_ifeval_net, row.final_gsm8k_net])
            arrow = "I→M" if order == "ifeval_to_math" else "M→I"
            labels.append(f"{DISPLAY_NAMES[configuration]} ({arrow})")
    matrix = np.asarray(ordered_rows, dtype=float)
    fig, axis = plt.subplots(figsize=(4.7, 4.0), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", vmin=-6.0, vmax=1.0)
    axis.set(
        title="Final performance after task 2 relative to the unadapted checkpoint",
        xticks=(0, 1),
        xticklabels=("IFEval", "GSM8K"),
        yticks=np.arange(len(labels)),
        yticklabels=labels,
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text_color = "white" if image.norm(value) < 0.32 else "black"
            axis.text(
                column,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.05, pad=0.03)
    colorbar.set_label("After task 2 minus base (points)")
    _save_figure(fig, output)


def plot_calibration_benchmark_contrast(protocols: pd.DataFrame, output: Path) -> None:
    ordered = protocols.copy()
    ordered["label"] = ordered.apply(
        lambda row: (
            f"{row.configuration_label} "
            f"({'I→M' if row.task_order == 'ifeval_to_math' else 'M→I'})"
        ),
        axis=1,
    )
    ordered = ordered.sort_values(["configuration", "task_order"])
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True, constrained_layout=True)
    axes[0].barh(y, ordered.mean_calibration_nll_improvement_percent)
    axes[0].set(
        title="Training-derived calibration",
        xlabel="Mean NLL reduction (%)",
        yticks=y,
        yticklabels=ordered.label,
    )
    axes[1].barh(y, ordered.mean_final_net)
    axes[1].axvline(0, color="0.25", linewidth=0.8)
    axes[1].set(title="Held-out benchmarks", xlabel="Mean final net score (points)")
    axes[0].invert_yaxis()
    for axis in axes:
        axis.grid(axis="x", linewidth=0.45, alpha=0.35)
    _save_figure(fig, output)


def plot_performance_vs_parameters(protocols: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(4.5, 3.2), constrained_layout=True)
    markers = {"ifeval_to_math": "o", "math_to_ifeval": "^"}
    ordered = protocols.sort_values("mean_final_net").reset_index(drop=True)
    label_positions = np.linspace(
        ordered.mean_final_net.min() - 0.05,
        ordered.mean_final_net.max() + 0.05,
        len(ordered),
    )
    for label_position, (_, row) in zip(label_positions, ordered.iterrows(), strict=True):
        axis.scatter(
            row.lora_parameters / 1e6,
            row.mean_final_net,
            marker=markers[row.task_order],
            s=34,
        )
        axis.annotate(
            f"{row.configuration_label} "
            f"({'I→M' if row.task_order == 'ifeval_to_math' else 'M→I'})",
            (row.lora_parameters / 1e6, row.mean_final_net),
            xytext=(5.905, label_position),
            textcoords="data",
            fontsize=6,
            va="center",
            arrowprops={"arrowstyle": "-", "linewidth": 0.45, "color": "0.45"},
        )
    axis.axhline(0, color="0.25", linewidth=0.8)
    axis.set(
        title="Performance at the controlled LoRA parameter budget",
        xlabel="Trainable LoRA parameters (millions)",
        ylabel="Mean final net score (points)",
        xlim=(5.885, 5.945),
    )
    axis.set_xticks((float(protocols.lora_parameters.iloc[0]) / 1e6,))
    axis.grid(linewidth=0.45, alpha=0.35)
    _save_figure(fig, output)


def _tex_name(configuration: str) -> str:
    return DISPLAY_NAMES[configuration]


def _write_tex_artifacts(
    protocols: pd.DataFrame,
    benchmark_rows: pd.DataFrame,
    runs: list[dict[str, Any]],
    selection: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    utility_means = protocols.groupby("configuration").utility.mean().to_dict()
    i_to_m = protocols[protocols.task_order == "ifeval_to_math"].set_index("configuration")
    m_to_i = protocols[protocols.task_order == "math_to_ifeval"].set_index("configuration")
    base_ifeval = float(protocols.base_ifeval.iloc[0])
    base_gsm8k = float(protocols.base_gsm8k.iloc[0])
    parameters = int(protocols.lora_parameters.iloc[0])
    trainable_percent = float(protocols.trainable_percent.iloc[0])
    negative_final_cells = int(
        (benchmark_rows[["net"]].to_numpy(dtype=float).ravel() < 0.0).sum()
    )
    uniform_if_forgetting = -float(i_to_m.loc["uniform", "first_task_bwt"])
    late_if_forgetting = -float(i_to_m.loc["late_heavy", "first_task_bwt"])
    middle_math_net = float(i_to_m.loc["middle_heavy", "final_gsm8k_net"])
    late_final_ifeval = float(m_to_i.loc["late_heavy", "stage2_ifeval"])
    macros = [
        rf"\newcommand{{\NumLoraParameters}}{{{parameters:,}}}",
        rf"\newcommand{{\TrainablePercent}}{{{trainable_percent:.3f}\%}}",
        rf"\newcommand{{\BaseIFEval}}{{{base_ifeval:.2f}}}",
        rf"\newcommand{{\BaseGSM}}{{{base_gsm8k:.2f}}}",
        rf"\newcommand{{\NegativeFinalCells}}{{{negative_final_cells}}}",
        rf"\newcommand{{\UniformIFForgetting}}{{{uniform_if_forgetting:.2f}}}",
        rf"\newcommand{{\LateIFForgetting}}{{{late_if_forgetting:.2f}}}",
        rf"\newcommand{{\MiddleMathNet}}{{{middle_math_net:+.2f}}}",
        rf"\newcommand{{\LateFinalIFEval}}{{{late_final_ifeval:.2f}}}",
        rf"\newcommand{{\EarlyMeanUtility}}{{{float(utility_means['early_heavy']):.2f}}}",
        rf"\newcommand{{\LateMeanUtility}}{{{float(utility_means['late_heavy']):.2f}}}",
        rf"\newcommand{{\SelectedManualConfig}}{{\textsc{{{_tex_name(selection['selected_configuration'])}}}}}",
    ]
    (output_dir / "results.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    table_lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        (
            r"Allocation & Order & \multicolumn{3}{c}{IFEval} & "
            r"\multicolumn{3}{c}{GSM8K} & $\mathrm{BWT}_1$ \\"
        ),
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r" & & Base & S1 & S2 & Base & S1 & S2 &  \\",
        r"\midrule",
    ]
    for configuration in MANUAL_CONFIGS:
        for order in ("ifeval_to_math", "math_to_ifeval"):
            row = protocols[
                (protocols.configuration == configuration) & (protocols.task_order == order)
            ].iloc[0]
            order_label = r"I$\rightarrow$M" if order == "ifeval_to_math" else r"M$\rightarrow$I"
            table_lines.append(
                f"{_tex_name(configuration)} & {order_label} & "
                f"{row.base_ifeval:.2f} & {row.stage1_ifeval:.2f} & {row.stage2_ifeval:.2f} & "
                f"{row.base_gsm8k:.2f} & {row.stage1_gsm8k:.2f} & {row.stage2_gsm8k:.2f} & "
                f"{row.first_task_bwt:+.2f} \\\\"
            )
    table_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "stage_results_table.tex").write_text(
        "\n".join(table_lines) + "\n", encoding="utf-8"
    )

    rank_vectors = _rank_vectors(runs)
    rank_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Allocation & Early & Middle & Late & Parameters \\",
        r"\midrule",
    ]
    for configuration in MANUAL_CONFIGS:
        vector = rank_vectors[configuration]
        rank_lines.append(
            f"{_tex_name(configuration)} & {vector[0]} & {vector[12]} & {vector[24]} & "
            f"{parameters:,} \\\\"
        )
    rank_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "rank_table.tex").write_text("\n".join(rank_lines) + "\n", encoding="utf-8")


def _load_decontamination_counts(config: ExperimentConfig) -> dict[str, Any]:
    root = config.data.processed_dir / "clean" / config.decontamination_fingerprint()
    output = {}
    for task, directory in (("ifeval_like", "ifeval_like"), ("numinamath", "numinamath")):
        path = root / directory / "audit" / "decontamination_manifest.json"
        if not path.exists():
            raise RuntimeError(f"Missing decontamination manifest: {path}")
        output[task] = json.loads(path.read_text(encoding="utf-8"))
    return output


def build_first_paper_artifacts(
    *,
    results_dir: str | Path = "results",
    config_dir: str | Path = "configs",
    paper_dir: str | Path = "paper",
) -> dict[str, Any]:
    results_root = Path(results_dir)
    paper_root = Path(paper_dir)
    config = load_config(Path(config_dir) / "base.yaml")
    runs = validate_manual_seed42_runs(results_root, config_dir=config_dir)
    protocols = protocol_frame(runs)
    benchmark_rows = benchmark_frame(runs)
    selection = select_best_manual(runs)
    decontamination = _load_decontamination_counts(config)

    data_dir = paper_root / "data"
    figures_dir = paper_root / "figures"
    generated_dir = paper_root / "generated"
    data_dir.mkdir(parents=True, exist_ok=True)
    protocols.to_csv(data_dir / "manual_seed42_protocols.csv", index=False)
    benchmark_rows.to_csv(data_dir / "manual_seed42_benchmarks.csv", index=False)

    rank_vectors = _rank_vectors(runs)
    plot_teaser(protocols, rank_vectors, figures_dir / "teaser")
    plot_rank_by_depth(rank_vectors, figures_dir / "rank_by_depth")
    plot_stage_trajectories(protocols, figures_dir / "stage_trajectories")
    plot_first_task_interference(protocols, figures_dir / "first_task_interference")
    plot_final_net_heatmap(protocols, figures_dir / "final_net_heatmap")
    plot_calibration_benchmark_contrast(
        protocols, figures_dir / "calibration_benchmark_contrast"
    )
    plot_performance_vs_parameters(protocols, figures_dir / "performance_vs_parameters")

    utility_means = protocols.groupby("configuration").utility.mean().to_dict()
    snapshot = {
        "scope": (
            "three depth-heavy allocations plus the uniform reference, "
            "both task orders, seed 42"
        ),
        "validated_protocols": len(protocols),
        "evaluation_examples_per_checkpoint": EXPECTED_EVAL_ROWS,
        "model": config.model.name,
        "model_revision": config.model.revision,
        "dataset_revisions": {
            "ifeval_train": config.data.ifeval_train.revision,
            "numina_train": config.data.numina_train.revision,
            "ifeval_eval": config.data.ifeval_eval.revision,
            "gsm8k": config.data.gsm8k_eval.revision,
        },
        "token_budget_per_task": config.training.train_token_budget_per_task,
        "calibration_token_budget_per_task": config.training.calibration_token_budget_per_task,
        "lora_parameters": int(protocols.lora_parameters.iloc[0]),
        "trainable_percent": float(protocols.trainable_percent.iloc[0]),
        "base_scores": {
            "ifeval": float(protocols.base_ifeval.iloc[0]),
            "gsm8k": float(protocols.base_gsm8k.iloc[0]),
        },
        "benchmark_mean_utility_by_configuration": utility_means,
        "manual_selection": selection,
        "decontamination": decontamination,
        "limitations": [
            "one training seed",
            "two task families",
            "4-bit NF4 backbone without a higher-precision control",
            "DDP sampler padding repeats one packed sequence per task and epoch",
        ],
    }
    (data_dir / "study_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_tex_artifacts(protocols, benchmark_rows, runs, selection, generated_dir)
    return snapshot
