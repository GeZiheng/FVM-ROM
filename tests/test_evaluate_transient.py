"""Estimators of the shedding frequency, amplitude and limit-cycle time."""

import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from fvmrom.foam.fields import MeshTopology
from fvmrom.rom.evaluate_transient import (
    _max_divergence,
    cycle_amplitude,
    dominant_frequency,
    energy_drift,
    error_curves,
    evaluate_transient,
    phase_difference,
    prediction_horizon,
    shedding_summary,
    to_jsonable,
)
from fvmrom.rom.transient import TransientROM
from test_transient_rom import fitted

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


def _fields(nt=12, n_cells=4, n_phi=6, seed=0):
    rng = np.random.default_rng(seed)
    ref = {
        "U": rng.normal(size=(nt, n_cells, 3)) + 1.0,
        "p": rng.normal(size=(nt, n_cells)),
        "phi": rng.normal(size=(nt, n_phi)),
    }
    return ref, np.array([1.0, 2.0, 4.0, 8.0]), rng.uniform(0.5, 2.0, n_phi)


def test_error_curves_are_zero_for_an_exact_prediction():
    ref, volumes, phi_weights = _fields()
    mean = {"U": np.mean(ref["U"], axis=0), "p": np.mean(ref["p"], axis=0)}
    curves = error_curves(ref, ref, volumes, phi_weights, mean)
    for name, values in curves.items():
        assert np.allclose(values, 0.0), name
    assert set(curves) == {"U", "p", "phi", "U_fluct", "p_fluct"}


def test_error_curves_follow_the_weighted_definition():
    ref, volumes, phi_weights = _fields()
    nt, n_cells = ref["U"].shape[:2]
    pred = {name: values.copy() for name, values in ref.items()}
    pred["U"][:, 0, :] += 0.1
    curves = error_curves(pred, ref, volumes, phi_weights)

    flat = (pred["U"] - ref["U"]).reshape(nt, -1)
    flat_ref = ref["U"].reshape(nt, -1)
    w = np.repeat(volumes, 3)
    expected = np.linalg.norm(flat * np.sqrt(w), axis=1) / np.linalg.norm(
        flat_ref * np.sqrt(w), axis=1
    )
    assert np.allclose(curves["U"], expected)

    unit = error_curves(pred, ref, np.ones(n_cells), phi_weights)
    assert not np.allclose(curves["U"], unit["U"])


def test_error_curves_fluctuation_denominator():
    nt, n_cells, n_phi = 12, 4, 6
    ref = {
        "U": np.ones((nt, n_cells, 3)),
        "p": np.ones((nt, n_cells)),
        "phi": np.ones((nt, n_phi)),
    }
    pred = {"U": 1.01 * ref["U"], "p": ref["p"], "phi": ref["phi"]}
    mean = {"U": 0.5 * np.ones((n_cells, 3)), "p": 0.5 * np.ones(n_cells)}
    curves = error_curves(pred, ref, np.ones(n_cells), None, mean)
    assert np.allclose(curves["U"], 0.01)
    assert np.allclose(curves["U_fluct"], 0.02)
    assert np.allclose(curves["p_fluct"], 0.0)


def test_prediction_horizon_hits_the_analytic_crossing():
    t = np.arange(0.0, 10.0 + 1e-9, 0.1)
    err = t.copy()
    assert prediction_horizon(t, err, 2.5, 0.0) == pytest.approx(2.6)
    assert prediction_horizon(t + 5.0, err, 2.5, 5.0) == pytest.approx(2.6)
    assert prediction_horizon(t, err, 100.0, 0.0) is None
    with pytest.raises(ValueError, match="errors"):
        prediction_horizon(t, err[:-1], 2.5, 0.0)


def test_prediction_horizon_treats_nonfinite_errors_as_out_of_tolerance():
    t = np.arange(0.0, 10.0 + 1e-9, 0.1)
    small = np.full(t.size, 1e-3)

    nan_early = small.copy()
    nan_early[7] = np.nan
    assert prediction_horizon(t, nan_early, 1e-2, 0.0) == pytest.approx(t[7])

    nan_late = small.copy()
    nan_late[30:] = np.nan
    assert prediction_horizon(t, nan_late, 1e-2, 0.0) == pytest.approx(t[30])

    # a NaN after the first crossing does not move the horizon
    crossing_then_nan = t.copy()
    crossing_then_nan[40] = np.nan
    assert prediction_horizon(t, crossing_then_nan, 2.5, 0.0) == pytest.approx(2.6)

    infinite = small.copy()
    infinite[5] = np.inf
    assert prediction_horizon(t, infinite, 1e-2, 0.0) == pytest.approx(t[5])

    # all finite and inside the tolerance: no horizon
    assert prediction_horizon(t, np.full(t.size, 5e-3), 1e-2, 0.0) is None
    # equality with the tolerance is still inside
    assert prediction_horizon(t, np.full(t.size, 1e-2), 1e-2, 0.0) is None


def _line_topology():
    """Two cells sharing one internal face; phi holds just that face."""
    return MeshTopology(
        owner=np.array([0]), neighbour=np.array([1]), patches={}, n_cells=2
    )


def test_max_divergence_propagates_nan_rows():
    topo = _line_topology()
    phi = np.array([[1.0], [np.nan], [np.nan]])
    # the original bug: the builtin max dropped the NaN rows behind the finite first one
    assert _max_divergence(phi[:1], topo) == pytest.approx(1.0)
    assert np.isnan(_max_divergence(phi, topo))
    assert np.isnan(_max_divergence(phi[1:], topo))
    assert _max_divergence(np.array([[0.5], [2.0]]), topo) == pytest.approx(2.0)
    assert _max_divergence(np.array([2.0]), topo) == pytest.approx(2.0)


def test_evaluate_transient_reports_a_nonfinite_phi_divergence():
    data, rom = fitted()
    n_phi = data["snap_pred"]["phi"].shape[1]
    topo = MeshTopology(
        owner=np.zeros(n_phi, dtype=int),
        neighbour=np.arange(1, n_phi + 1),
        patches={},
        n_cells=n_phi + 1,
    )
    rom.readouts_["phi"].coef_[-1] = np.nan
    report = evaluate_transient(
        rom,
        data["t_pred"],
        data["snap_pred"],
        data["volumes"],
        data["phi_weights"],
        period=1.0,
        forces_test=data["forces_pred"],
        f_fom=1.0,
        topo=topo,
    )
    divergence = report["phi_divergence"]
    assert np.isnan(divergence["rom_max"])
    assert np.isfinite(divergence["fom_max"])
    assert divergence["ok"] is False
    json.dumps(to_jsonable(report))


def test_phase_difference_of_a_known_shift():
    t = np.arange(0.0, 20.0, 0.05)
    w = 2.0 * np.pi / T
    reference = 0.3 * np.cos(w * t) + 0.2
    shifted = 0.3 * np.cos(w * t - 0.3) + 0.2
    assert phase_difference(t, shifted, reference, F) == pytest.approx(0.3, abs=1e-3)
    assert phase_difference(t, reference, shifted, F) == pytest.approx(-0.3, abs=1e-3)
    wrapped = 0.3 * np.cos(w * t - (2.0 * np.pi - 0.2)) + 0.2
    assert phase_difference(t, wrapped, reference, F) == pytest.approx(-0.2, abs=1e-3)


def test_phase_difference_rejects_missing_frequency():
    t = np.arange(0.0, 20.0, 0.05)
    flat = np.ones(t.size)
    with pytest.raises(ValueError, match="component"):
        phase_difference(t, flat, flat, F)


def test_energy_drift_matches_the_analytic_growth():
    period = 2.0
    t = np.arange(0.0, 20.0 + 1e-9, 0.01)
    eps = 0.01
    gain = 1.0 + eps * t
    a = np.column_stack(
        [gain * np.sin(2.0 * np.pi * t / period), gain * np.cos(2.0 * np.pi * t / period)]
    )
    out = energy_drift(t, a, period, a_train_max=1.0)

    def continuous(start):
        return (
            (1.0 + eps * (start + period)) ** 3 - (1.0 + eps * start) ** 3
        ) / (3.0 * eps * period)

    def sampled(start):
        offsets = np.arange(int(round(period / 0.01))) * 0.01
        return float(np.mean((1.0 + eps * (start + offsets)) ** 2))

    assert out["n_cycles"] == 10
    assert out["first_cycle"] == pytest.approx(sampled(t[0]), rel=1e-12)
    assert out["last_cycle"] == pytest.approx(sampled(t[0] + 9.0 * period), rel=1e-12)
    assert out["first_cycle"] == pytest.approx(continuous(t[0]), rel=5e-4)
    assert out["last_cycle"] == pytest.approx(continuous(t[0] + 9.0 * period), rel=5e-4)
    assert out["relative_change"] == pytest.approx(
        (out["last_cycle"] - out["first_cycle"]) / out["first_cycle"]
    )
    assert out["max_norm"] == pytest.approx(gain[-1], rel=1e-12)
    assert out["max_norm_ratio"] == pytest.approx(gain[-1])
    assert out["train_max_norm"] == 1.0
    with pytest.raises(ValueError, match="shorter"):
        energy_drift(t[:50], a[:50], period)


def test_project_reconstructs_through_the_trained_pod():
    data, rom = fitted()
    snapshots = data["snap_pred"]["U"]
    rebuilt = rom.project("U", snapshots)
    assert rebuilt.shape == snapshots.shape
    error = np.linalg.norm((rebuilt - snapshots).reshape(snapshots.shape[0], -1))
    assert error / np.linalg.norm(snapshots.reshape(snapshots.shape[0], -1)) < 1e-6
    with pytest.raises(ValueError, match="no POD"):
        rom.project("k", snapshots)
    with pytest.raises(RuntimeError, match="not fitted"):
        TransientROM(dynamics="dmd").project("U", snapshots)


def test_evaluate_transient_on_an_exactly_predictable_system():
    data, rom = fitted()
    report = evaluate_transient(
        rom,
        data["t_pred"],
        data["snap_pred"],
        data["volumes"],
        data["phi_weights"],
        period=1.0,
        forces_test=data["forces_pred"],
        f_fom=1.0,
    )
    for name, curve in report["errors"].items():
        assert np.max(curve) < 1e-6, name
    assert report["projection_max_within_tau"]["U"] < 1e-6
    assert report["projection_max_within_tau"]["p"] < 1e-6
    for horizons in report["prediction_horizon"].values():
        assert horizons["0.01"] is None
        assert horizons["0.1"] is None
    assert report["errors_at_tau"]["U"]["1"] < 1e-6
    forces = report["scalars"]["forces"]
    assert forces["cd_mean"]["relative_error"] < 1e-6
    assert forces["cl_amplitude"]["relative_error"] < 1e-6
    attribution = forces["attribution"]
    assert attribution["true_latent"]["cl_amplitude"]["relative_error"] < 1e-6
    assert attribution["predicted_latent"]["cl_amplitude"]["relative_error"] < 1e-6
    assert attribution["latent_relative_error"] < 1e-6
    assert report["frequency"]["relative_error"] < 1e-6
    assert report["energy"]["rom"]["relative_change"] == pytest.approx(0.0, abs=1e-6)
    assert report["energy"]["fom"]["relative_change"] == pytest.approx(0.0, abs=1e-6)
    json.dumps(to_jsonable(report))


def test_evaluate_transient_requires_the_period_keyword():
    data, rom = fitted()
    with pytest.raises(TypeError):
        evaluate_transient(rom, data["t_pred"], data["snap_pred"], data["volumes"])
    parameter = inspect.signature(evaluate_transient).parameters["period"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def test_to_jsonable_handles_nested_arrays():
    payload = {
        "a": np.arange(3.0),
        "b": [np.float32(1.5), {"c": np.int64(2)}],
        "d": None,
        "e": np.float64("nan"),
    }
    text = json.dumps(to_jsonable(payload))
    assert json.loads(text) == {"a": [0.0, 1.0, 2.0], "b": [1.5, {"c": 2}], "d": None, "e": None}

