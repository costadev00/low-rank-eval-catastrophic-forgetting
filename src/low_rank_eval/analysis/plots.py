from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_all(results_dir: str | Path) -> None:
    root = Path(results_dir)
    aggregate = root / "aggregates"
    figures = root / "figures"
    frame = pd.read_csv(aggregate / "all_results.csv")
    seed42 = frame[frame.seed == 42].copy()

    rank_rows = seed42.drop_duplicates(["configuration", "task_order"])
    for _, row in rank_rows.iterrows():
        ranks = {int(key): value for key, value in json.loads(row.rank_by_layer).items()}
        plt.plot(list(ranks), list(ranks.values()), marker="o", label=row.configuration)
    plt.xlabel("Transformer block")
    plt.ylabel("LoRA rank")
    plt.legend()
    _save(figures / "rank_by_depth.pdf")

    stage = seed42.melt(
        id_vars=["configuration", "task_order", "benchmark"],
        value_vars=["base_score", "stage1_score", "stage2_score"],
        var_name="stage",
        value_name="score",
    )
    sns.lineplot(
        data=stage,
        x="stage",
        y="score",
        hue="configuration",
        style="benchmark",
        markers=True,
    )
    plt.ylabel("Score (0–100)")
    _save(figures / "performance_by_stage.pdf")

    sns.barplot(data=seed42, x="configuration", y="forgetting", hue="benchmark")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Forgetting (points)")
    _save(figures / "forgetting_by_configuration.pdf")

    sns.scatterplot(
        data=seed42,
        x="trainable_parameters",
        y="stage2_score",
        hue="configuration",
        style="benchmark",
    )
    _save(figures / "performance_vs_parameters.pdf")

    heatmap = seed42.pivot_table(
        index="configuration",
        columns=["task_order", "benchmark"],
        values="stage2_score",
        aggfunc="mean",
    )
    sns.heatmap(heatmap, annot=True, fmt=".1f")
    _save(figures / "configuration_order_benchmark_heatmap.pdf")

    spectral_candidates = [
        *sorted((root / "spectral" / "full").glob("*/spectral_analysis.json")),
        *sorted((root / "spectral" / "smoke").glob("*/spectral_analysis.json")),
    ]
    if spectral_candidates:
        spectral_path = spectral_candidates[0]
        with spectral_path.open(encoding="utf-8") as handle:
            spectral = json.load(handle)
        for name, values in spectral.items():
            block = name.split(".layers.", 1)[-1].split(".", 1)[0]
            plt.plot(
                range(1, len(values["cumulative_energy"]) + 1),
                values["cumulative_energy"],
                alpha=0.35,
                label=f"block {block}",
            )
        plt.xlabel("Retained rank")
        plt.ylabel("Cumulative spectral energy")
        _save(figures / "spectral_energy_curves.pdf")

    summary = (
        seed42.groupby(["configuration", "task_order", "benchmark"], as_index=False)
        .agg(
            final_score=("stage2_score", "mean"),
            net=("net", "mean"),
            forgetting=("forgetting", "mean"),
            parameters=("trainable_parameters", "first"),
        )
        .sort_values(["configuration", "task_order", "benchmark"])
    )
    summary.to_csv(aggregate / "final_comparison_table.csv", index=False)
    with (aggregate / "final_comparison_table.tex").open("w", encoding="utf-8") as handle:
        handle.write(summary.to_latex(index=False, float_format="%.2f"))
