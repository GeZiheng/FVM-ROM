"""Unit tests for scripts/opinf_lambda_scan.py (synthetic study, no OpenFOAM)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scan = _load("opinf_lambda_scan", PROJECT / "scripts" / "opinf_lambda_scan.py")
fit_helper = _load(
    "opinf_scan_fit_helper", PROJECT / "tests" / "test_fit_and_eval_transient.py"
)

STUDY = fit_helper.HOPF_STUDY
TRAINING_KEYS = {
    "train_replay_error",
    "replay_max_norm_ratio",
    "forward_max_norm_ratio",
    "recheck_stable",
    "replay_diverged",
    "forward_diverged",
    "check_periods",
    "check_period",
    "check_period_estimated",
    "blowup_factor",
    "check_steps",
}
POST_HOC_KEYS = {
    "u_error_max_within_tau5",
    "u_error_at_tau",
    "frequency_relative_error",
    "cl_amplitude_relative_error",
    "cd_mean_relative_error",
    "cl_phase_tau5_deg",
    "diverged",
    "diverged_at",
    "energy_relative_change",
    "max_norm_ratio",
}


@pytest.fixture
def study_dir(tmp_path):
    return fit_helper.build_hopf_study(tmp_path)


def run_main(monkeypatch, tmp_path, *extra: str) -> None:
    argv = [
        "opinf_lambda_scan.py",
        "--study",
        STUDY,
        "--data-dir",
        str(tmp_path),
        "--train-periods",
        "3",
        "--n-modes",
        "U=3,p=9,phi=3",
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    scan.main()


def test_default_grid_contains_the_two_reference_ridges():
    grid = scan.DEFAULT_RIDGE_GRID
    assert grid.size == 13
    assert np.isclose(grid.min(), 1e-5) and np.isclose(grid.max(), 1e-2)
    assert np.any(grid == 1e-4)
    assert np.any(grid == 1e-3)
    assert scan.parse_ridge_grid(None).size == 13


def test_scan_writes_only_the_scan_json(study_dir, tmp_path, monkeypatch, capsys):
    run_main(monkeypatch, tmp_path)

    out_path = study_dir / "opinf_lambda_scan.json"
    report = json.loads(out_path.read_text(encoding="utf-8"))
    for key in (
        "problem",
        "study",
        "post_hoc_note",
        "config",
        "training_window",
        "scan",
        "summary",
    ):
        assert key in report
    assert "NOT used to choose lambda" in report["post_hoc_note"]
    assert report["config"]["ridge_grid"] == pytest.approx(scan.DEFAULT_RIDGE_GRID)
    assert len(report["scan"]) == 13

    for entry in report["scan"]:
        assert entry["status"] == "ok"
        assert set(entry) == {"ridge", "annotation", "status", "training", "post_hoc", "cost"}
        assert set(entry["training"]) == TRAINING_KEYS
        assert set(entry["post_hoc"]) == POST_HOC_KEYS
        assert set(entry["post_hoc"]["u_error_at_tau"]) == {"1", "3", "5", "10"}
        assert isinstance(entry["training"]["recheck_stable"], bool)
    labelled = {round(entry["ridge"], 12): entry["annotation"] for entry in report["scan"]}
    assert labelled[1e-4] == ["selected by hold-out"]
    assert labelled[1e-3] == ["formal opinf (after fallback)"]

    # read-only: no report and no models are written
    assert not (study_dir / "report.json").exists()
    assert not (study_dir / "models").exists()

    out = capsys.readouterr().out
    assert "training-data diagnostics" in out
    assert "post-hoc (uses the prediction window; NOT used to choose lambda)" in out
    assert "summary (data only; no tuning recommendation)" in out


def test_ridge_grid_override(study_dir, tmp_path, monkeypatch):
    run_main(monkeypatch, tmp_path, "--ridge-grid", "1e-5,1e-4,1e-2")

    report = json.loads((study_dir / "opinf_lambda_scan.json").read_text(encoding="utf-8"))
    ridges = [entry["ridge"] for entry in report["scan"]]
    assert ridges == pytest.approx([1e-5, 1e-4, 1e-2])
    annotations = {round(entry["ridge"], 12): entry["annotation"] for entry in report["scan"]}
    assert annotations[1e-4] == ["selected by hold-out"]
    # 1e-3 is not on the override grid, so it is not labelled
    assert annotations[1e-5] == []
    assert annotations[1e-2] == []


def test_replay_divergence_is_reported_not_raised(study_dir, tmp_path, monkeypatch):
    from fvmrom.rom.dynamics import OpInf

    original = OpInf._integrate

    def replay_diverges(self, coef, t0, a0, t_eval, threshold):
        values, diverged, at = original(self, coef, t0, a0, t_eval, threshold)
        replay = (
            self.ridge_ is not None
            and self.ridge_ <= 1e-4
            and np.isclose(t0, self.t_first_)
            and np.allclose(a0, self.a_first_)
        )
        if replay:
            return values, True, float(t0)
        return values, diverged, at

    monkeypatch.setattr(OpInf, "_integrate", replay_diverges)
    run_main(monkeypatch, tmp_path, "--ridge-grid", "1e-5,1e-3")

    report = json.loads((study_dir / "opinf_lambda_scan.json").read_text(encoding="utf-8"))
    entries = {round(entry["ridge"], 12): entry for entry in report["scan"]}
    small = entries[1e-5]["training"]
    assert small["recheck_stable"] is False
    assert small["replay_diverged"] is True
    assert small["forward_diverged"] is False
    assert entries[1e-3]["training"]["recheck_stable"] is True
    assert report["summary"]["recheck_stable"]["n"] == 1
