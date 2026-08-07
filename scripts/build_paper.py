#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _tex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def _required_runs(frame: pd.DataFrame, selected: str) -> list[str]:
    missing = []
    orders = {"ifeval_to_math", "math_to_ifeval"}
    initial = {"uniform", "early_heavy", "middle_heavy", "late_heavy", "spectral"}
    for config in initial:
        for order in orders:
            found = frame[
                (frame.configuration == config) & (frame.task_order == order) & (frame.seed == 42)
            ]
            if found.empty:
                missing.append(f"{config}/{order}/42")
    for config in {"uniform", selected, "spectral"}:
        for order in orders:
            for seed in {7, 42, 123}:
                found = frame[
                    (frame.configuration == config)
                    & (frame.task_order == order)
                    & (frame.seed == seed)
                ]
                if found.empty:
                    missing.append(f"{config}/{order}/{seed}")
    return missing


def main() -> None:
    aggregate = RESULTS / "aggregates"
    frame = pd.read_csv(aggregate / "all_results.csv")
    with (aggregate / "manual_selection.json").open(encoding="utf-8") as handle:
        selection = json.load(handle)
    selected = selection.get("selected_configuration")
    if not selected:
        raise RuntimeError("Manual selection is incomplete; the paper cannot be generated")
    missing = _required_runs(frame, selected)
    if missing:
        raise RuntimeError(
            "The paper refuses to invent results. Missing required runs: " + ", ".join(missing)
        )
    seed42 = frame[frame.seed == 42]
    utility = seed42.groupby("configuration").utility.mean().sort_values(ascending=False)
    best = utility.index[0]
    spectral = seed42[seed42.configuration == "spectral"]
    uniform = seed42[seed42.configuration == "uniform"]
    spectral_net = float(spectral.net.mean())
    uniform_net = float(uniform.net.mean())
    spectral_forgetting = float(spectral.forgetting.mean())
    uniform_forgetting = float(uniform.forgetting.mean())
    order_means = seed42.groupby(["configuration", "task_order"]).utility.mean().loc[best].to_dict()
    parameters = int(seed42.trainable_parameters.iloc[0])
    conclusion = (
        f"Under this protocol, {best} achieved the highest descriptive utility. "
        f"The spectral allocation changed mean net performance by "
        f"{spectral_net - uniform_net:+.2f} points and mean forgetting by "
        f"{spectral_forgetting - uniform_forgetting:+.2f} points relative to uniform ranks."
    )
    macros = [
        rf"\newcommand{{\NumLoraParameters}}{{{parameters:,}}}",
        rf"\newcommand{{\SelectedManualConfig}}{{\texttt{{{_tex_escape(selected)}}}}}",
        rf"\newcommand{{\GeneratedAbstractResult}}{{{_tex_escape(conclusion)}}}",
        rf"\newcommand{{\GeneratedMainResult}}{{The best seed-42 descriptive "
        rf"utility was obtained by "
        rf"\texttt{{{_tex_escape(best)}}}. Spectral minus uniform net gain was "
        rf"{spectral_net - uniform_net:+.2f} points; the corresponding forgetting difference was "
        rf"{spectral_forgetting - uniform_forgetting:+.2f} points.}}",
        rf"\newcommand{{\GeneratedOrderResult}}{{For the best configuration, utility was "
        rf"{order_means.get('ifeval_to_math', float('nan')):.2f} for instruction-to-math and "
        rf"{order_means.get('math_to_ifeval', float('nan')):.2f} for math-to-instruction.}}",
        rf"\newcommand{{\GeneratedConclusion}}{{{_tex_escape(conclusion)}}}",
    ]
    output = ROOT / "paper" / "generated" / "results.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
