import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_full_study.py"
SPEC = importlib.util.spec_from_file_location("run_full_study", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
run_full_study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_full_study)


def test_failed_subprocess_is_recorded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_full_study, "ROOT", tmp_path)

    with pytest.raises(subprocess.CalledProcessError):
        run_full_study._run("failing_step", [sys.executable, "-c", "raise SystemExit(3)"])

    state = json.loads((tmp_path / "results" / "full_study_state.json").read_text())
    assert state["step"] == "failing_step"
    assert state["status"] == "failed"
    assert state["error"]["type"] == "CalledProcessError"
    assert state["error"]["returncode"] == 3
