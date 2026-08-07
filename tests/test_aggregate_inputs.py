import json
from pathlib import Path

from low_rank_eval.analysis.aggregate_results import load_run_results
from low_rank_eval.config import load_config


def _write_run(
    root: Path,
    run_id: str,
    *,
    scientific: bool,
    fingerprint: str,
) -> None:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    payload = {
        "run_id": run_id,
        "configuration": "uniform",
        "seed": 42,
        "task_order": ["ifeval", "math"],
    }
    (run_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "scientific": scientific,
        "config_fingerprint": fingerprint,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_only_current_scientific_results_are_loaded(tmp_path: Path) -> None:
    fingerprint = load_config("configs/uniform.yaml").fingerprint()
    valid_id = f"uniform-ifeval_to_math-seed42-{fingerprint}"
    _write_run(tmp_path, valid_id, scientific=True, fingerprint=fingerprint)
    _write_run(tmp_path, "smoke-result", scientific=False, fingerprint=fingerprint)
    _write_run(tmp_path, "stale-result", scientific=True, fingerprint="obsolete")

    runs = load_run_results(tmp_path, config_dir="configs")

    assert [run["run_id"] for run in runs] == [valid_id]
