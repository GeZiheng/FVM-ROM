"""Estimators of the shedding frequency, amplitude and limit-cycle time."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fvmrom.rom.evaluate_transient import (
    cycle_amplitude,
    dominant_frequency,
    shedding_summary,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "shedding_report.py"
_spec = importlib.util.spec_from_file_location("shedding_report", SCRIPT)
shedding_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shedding_report)

F = 1.0 / 6.0
T = 6.0


def _signal(amplitude=0.3, t_end=120.0, dt=0.05):
    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    return t, amplitude * np.sin(2.0 * np.pi * F * t)


def test_dominant_frequency_of_a_sine():
    t, cl = _signal()
    assert dominant_frequency(t, cl) == pytest.approx(F, rel=1e-3)
    assert dominant_frequency(t, -cl) == pytest.approx(F, rel=1e-3)


def test_dominant_frequency_with_offset_and_harmonics():
    t, cl = _signal()
    y = 0.7 + cl + 0.1 * np.sin(2.0 * np.pi * 2.0 * F * t) + 0.02 * np.sin(2.5 * t)
    assert dominant_frequency(t, y) == pytest.approx(F, rel=1e-3)


def test_dominant_frequency_validation():
    t = np.arange(0.0, 10.0, 0.1)
    with pytest.raises(ValueError, match="uniform"):
        dominant_frequency(np.r_[t[:5], t[5] + 0.3, t[6:]], np.sin(t))
    with pytest.raises(ValueError, match="eight"):
        dominant_frequency(t[:5], np.sin(t[:5]))
    with pytest.raises(ValueError, match="values"):
        dominant_frequency(t, np.sin(t)[:-1])


def test_cycle_amplitude():
    t, cl = _signal(dt=0.01)
    assert cycle_amplitude(t, cl, T) == pytest.approx(0.3, rel=1e-3)
    assert cycle_amplitude(t, 2.0 * cl, T) == pytest.approx(0.6, rel=1e-3)
    with pytest.raises(ValueError, match="shorter"):
        cycle_amplitude(t[:20], cl[:20], T)


def test_shedding_summary_of_a_steady_sine():
    t, cl = _signal()
    out = shedding_summary(t, cl, np.full(t.size, 1.4))
    assert out["frequency"] == pytest.approx(F, rel=1e-3)
    assert out["period"] == pytest.approx(T, rel=1e-3)
    assert out["strouhal"] == pytest.approx(F, rel=1e-3)
    assert out["cl_amplitude"] == pytest.approx(0.3, rel=1e-3)
    assert out["cd_mean"] == pytest.approx(1.4)
    assert out["limit_cycle_time"] == pytest.approx(0.0, abs=1e-9)
    assert shedding_summary(t, cl)["cd_mean"] is None
    with pytest.raises(ValueError, match="cd"):
        shedding_summary(t, cl, np.ones(3))


def test_shedding_summary_with_a_growth_phase():
    t = np.arange(0.0, 120.025, 0.05)
    amplitude = 0.3 * np.minimum(t / 20.0, 1.0)
    cl = amplitude * np.sin(2.0 * np.pi * F * t)
    out = shedding_summary(t, cl, np.full(t.size, 1.4))
    assert out["strouhal"] == pytest.approx(F, rel=5e-3)
    assert out["cl_amplitude"] == pytest.approx(0.3, rel=2e-2)
    assert 18.0 <= out["limit_cycle_time"] <= 26.0


def test_limit_cycle_time_is_none_while_growing():
    t = np.arange(0.0, 120.025, 0.05)
    cl = (0.3 * t / 120.0) * np.sin(2.0 * np.pi * F * t)
    assert shedding_summary(t, cl)["limit_cycle_time"] is None


def test_shedding_report_loads_pilot_data(tmp_path):
    root = tmp_path / "cylinder_transient" / "pilot_x"
    root.mkdir(parents=True)
    t, cl = _signal()
    np.savez(root / "forces.npz", time=t, cd=np.full(t.size, 1.4), cl=cl)
    (root / "meta.json").write_text(
        '{\n  "study": "pilot_x",\n  "n_cells": 8192,\n  "max_courant": 0.31,\n'
        '  "mesh": {"n_cells": 8192},\n  "shedding": null\n}\n',
        encoding="utf-8",
    )
    row = shedding_report.load_study(tmp_path / "cylinder_transient", "pilot_x")
    assert row["study"] == "pilot_x"
    assert row["cells"] == 8192
    assert row["max_courant"] == 0.31
    assert row["summary"]["strouhal"] == pytest.approx(F, rel=1e-3)
    assert row["summary"]["cd_mean"] == pytest.approx(1.4)

    text = shedding_report.format_rows([row, dict(row, study="pilot_y")])
    assert "study" in text and "pilot_x" in text and "pilot_y" in text
    assert "St" in text and "limit-cycle" in text
    deviations = shedding_report.deviations([row, dict(row, study="pilot_y")])
    assert "pilot_y" in deviations and "0.00%" in deviations
