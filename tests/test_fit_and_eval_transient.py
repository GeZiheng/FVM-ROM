"""Unit tests for scripts/fit_and_eval_transient.py (synthetic study, no OpenFOAM)."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fvmrom.core.config import Settings
from fvmrom.foam import Cases, read_field
from fvmrom.rom import TransientROM

PROJECT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit_and_eval_transient = _load(
    "fit_and_eval_transient", PROJECT / "scripts" / "fit_and_eval_transient.py"
)
trajectory_helper = _load(
    "transient_rom_trajectory", Path(__file__).with_name("test_transient_rom.py")
)
mesh_helper = _load(
    "transient_foam_mesh", Path(__file__).with_name("test_transient_foam.py")
)

STUDY = "syn"


def build_study(root, study=STUDY, n_cells=30, n_phi=50) -> Path:
    """Write a tiny ``data/cylinder_transient/<study>/`` from the synthetic trajectory.

    ``dt = 0.1`` keeps the sampling commensurate with the unit period, and the
    tests train on 2.95 periods so that the training window holds a whole number
    of sampled periods (the synthetic latent is then exactly linear for DMD).
    ``n_cells``/``n_phi`` match the study to a synthetic FOM mesh when the ``phi``
    divergence checks have to run.
    """
    data = trajectory_helper.trajectory(
        n_train_periods=3.0, n_pred_periods=10.0, dt=0.1, n_cells=n_cells, n_phi=n_phi
    )
    times = np.concatenate([data["t_train"], data["t_pred"]])
    snaps = {
        name: np.concatenate([data["snap_train"][name], data["snap_pred"][name]])
        for name in ("U", "p", "phi")
    }
    # Cl at the fundamental frequency (as the cylinder forces are), Cd nearly steady
    latent = trajectory_helper.series(times)
    forces = np.column_stack([1.5 + 0.01 * latent[:, 1], 0.4 * latent[:, 0]])
    directory = Path(root) / "cylinder_transient" / study
    directory.mkdir(parents=True)
    np.savez(
        directory / "snapshots.npz",
        times=times,
        **snaps,
        forces=forces,
        volumes=data["volumes"],
        phi_weights=data["phi_weights"],
        re=100.0,
    )
    meta = {
        "problem": "cylinder_transient",
        "study": study,
        "config": {"re": 100.0, "snapshot_dt": 0.1},
        "dt": 0.1,
        "n_snapshots": int(times.size),
    }
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return directory


@pytest.fixture
def study_dir(tmp_path):
    return build_study(tmp_path)


@pytest.fixture
def mesh_study(tmp_path, monkeypatch):
    """The synthetic study plus a matching two-cell FOM case, so the phi checks run."""
    study_dir = build_study(tmp_path, n_cells=2, n_phi=7)
    settings = Settings(
        cases_dir=tmp_path / "cases",
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
    )
    root = Cases(settings).path(f"cylinder_transient_{STUDY}/run_0000", must_exist=False)
    mesh_helper._write_two_cell_mesh(root)
    monkeypatch.setattr(fit_and_eval_transient, "get_settings", lambda: settings)
    return study_dir


def run_main(
    monkeypatch,
    tmp_path,
    *extra: str,
    train_periods: str = "2.95",
    n_modes: str = "U=4,p=8,phi=4",
    study: str = STUDY,
) -> None:
    argv = [
        "fit_and_eval_transient.py",
        "--study",
        study,
        "--data-dir",
        str(tmp_path),
        "--train-periods",
        train_periods,
        "--n-modes",
        n_modes,
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    fit_and_eval_transient.main()


def test_report_model_and_acceptance(mesh_study, tmp_path, monkeypatch, capsys):
    # the two-cell synthetic mesh has two ``p`` cells, so ``p`` gets two modes
    run_main(
        monkeypatch,
        tmp_path,
        "--sweep-modes",
        "2,4",
        "--sweep-train-periods",
        "2,3",
        n_modes="U=4,p=2,phi=4",
    )
    study_dir = mesh_study
    report = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    assert (study_dir / "models" / "dmd.pkl").is_file()
    for key in ("problem", "study", "fom", "config", "training_window", "methods", "sweeps"):
        assert key in report

    method = report["methods"]["dmd"]
    assert method["model"] == "models/dmd.pkl"
    assert method["n_modes"] == {"U": 4, "p": 2, "phi": 4}
    for key in ("evaluation", "cost", "acceptance"):
        assert key in method
    evaluation = method["evaluation"]
    for key in (
        "errors",
        "errors_at_tau",
        "max_error_within_tau",
        "projection_max_within_tau",
        "frequency",
        "scalars",
        "energy",
        "spectrum",
    ):
        assert key in evaluation

    assert method["acceptance"]["verdict"] == "PASS"
    assert method["acceptance"]["pass"] is True
    for check in method["acceptance"]["checks"]:
        assert check["pass"] is True, check

    window = report["training_window"]
    assert window["n_train"] == 30
    assert window["t_train_end"] < window["t_test_start"]
    assert window["n_test"] == evaluation["n_test"]

    assert report["mesh"] == {"topology": True, "geometry": True}
    assert isinstance(report["sweeps"]["modes"], dict)
    assert isinstance(report["sweeps"]["train_periods"], dict)
    assert list(report["sweeps"]["modes"]) == ["dmd"]
    assert list(report["sweeps"]["train_periods"]) == ["dmd"]
    for grouped in (report["sweeps"]["modes"], report["sweeps"]["train_periods"]):
        for method, rows in grouped.items():
            for row in rows:
                assert row["method"] == method
                assert row["status"] == "ok", row
                assert row["max_u_error_within_tau5"] is not None
                assert row["cl_amplitude_relative_error"] is not None
    assert [row["n_modes_U"] for row in report["sweeps"]["modes"]["dmd"]] == [2, 4]
    out = capsys.readouterr().out
    assert "acceptance (plan 3.1)" in out
    assert "  dmd: PASS" in out
    assert "latent-dimension sweep" in out


def test_training_window_only_uses_the_prefix(study_dir, tmp_path, monkeypatch):
    run_main(monkeypatch, tmp_path)
    report = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    window = report["training_window"]
    times = np.load(study_dir / "snapshots.npz")["times"]
    cutoff = window["cutoff"]
    assert window["n_train"] == int(
        np.count_nonzero(times <= cutoff + 1e-9 * max(1.0, abs(cutoff)))
    )
    assert times[window["n_train"] - 1] == pytest.approx(window["t_train_end"])
    assert times[window["n_train"]] == pytest.approx(window["t_test_start"])
    assert window["t_train_end"] < window["t_test_start"]


def test_sweep_skips_a_window_without_five_test_periods(study_dir):
    times, forces, volumes, phi_w, fields = fit_and_eval_transient._load(
        study_dir / "snapshots.npz"
    )
    f_fom = fit_and_eval_transient.dominant_frequency(times, forces[:, 1])
    period = 1.0 / f_fom
    args = SimpleNamespace(
        methods=["dmd"],
        n_modes={"U": 4, "p": 8, "phi": 4},
        energy_tol=1e-10,
        readout_ridge=1e-10,
        sweep_train_periods=[8.5],
    )
    rows = fit_and_eval_transient._sweep_train_periods(
        args, times, fields, forces, volumes, phi_w, period, f_fom
    )
    assert rows[0]["status"] == "skipped"
    assert "needs five" in rows[0]["reason"]
    assert "max_u_error_within_tau5" not in rows[0]


def test_n_modes_parser():
    parse = fit_and_eval_transient._parse_n_modes
    assert parse("U=8") == {"U": 8, "p": 8, "phi": 8}
    assert parse("p=auto,phi=auto") == {"U": 8, "p": None, "phi": None}
    assert parse("U=4, p=12") == {"U": 4, "p": 12, "phi": 8}
    with pytest.raises(SystemExit, match="integer or 'auto'"):
        parse("U=x")
    with pytest.raises(SystemExit, match="unknown field"):
        parse("k=3")
    with pytest.raises(SystemExit, match="positive"):
        parse("U=0")
    with pytest.raises(SystemExit, match="field=count"):
        parse("U")


def test_unimplemented_method_is_rejected(monkeypatch, capsys):
    """No method is unimplemented any more, but unknown names still fail."""
    accepted = "dmd,optdmd,bopdmd,opinf,opinf_norecheck"
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit_and_eval_transient.py", "--study", "syn", "--methods", accepted],
    )
    args = fit_and_eval_transient.parse_args()
    assert args.methods == accepted.split(",")
    assert fit_and_eval_transient._method_dynamics("opinf") == ("opinf", {})
    assert fit_and_eval_transient._method_dynamics("opinf_norecheck") == (
        "opinf",
        {"recheck": False},
    )
    capsys.readouterr()

    monkeypatch.setattr(
        sys, "argv", ["fit_and_eval_transient.py", "--study", "syn", "--methods", "kriging"]
    )
    with pytest.raises(SystemExit):
        fit_and_eval_transient.parse_args()
    err = capsys.readouterr().err
    assert "unknown method" in err
    assert "opinf_norecheck" in err
    assert "not implemented" not in err


def test_dmd_family_methods_are_available(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit_and_eval_transient.py",
            "--study",
            "syn",
            "--methods",
            "dmd,optdmd,bopdmd",
        ],
    )
    args = fit_and_eval_transient.parse_args()
    assert args.methods == ["dmd", "optdmd", "bopdmd"]


def test_unknown_method_is_rejected(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["fit_and_eval_transient.py", "--study", "syn", "--methods", "kriging"]
    )
    with pytest.raises(SystemExit):
        fit_and_eval_transient.parse_args()
    assert "unknown method" in capsys.readouterr().err


def test_missing_snapshots_is_a_clear_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit_and_eval_transient.py", "--study", "absent", "--data-dir", str(tmp_path)],
    )
    with pytest.raises(SystemExit, match="missing snapshots"):
        fit_and_eval_transient.main()


# -- acceptance checks (plan 3.1) -----------------------------------------------------------


def _acceptance_result() -> dict:
    """A fully passing evaluation report (values only; no ROM involved)."""
    return {
        "max_error_within_tau": {"U": 1.0e-3},
        "frequency": {"relative_error": 1.0e-4},
        "scalars": {
            "forces": {
                "cd_mean": {"relative_error": 1.0e-3},
                "cl_amplitude": {"relative_error": 1.0e-3},
            }
        },
        "energy": {"rom": {"max_norm_ratio": 1.0, "relative_change": 0.01}},
        "phi_divergence": {
            "rom_max": 1.0e-11,
            "fom_max": 1.0e-13,
            "limit": 1.0e-11,
            "ok": True,
        },
    }


def _set_path(result: dict, path: str, value) -> None:
    target = result
    keys = path.split("__")
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


def _checks(acceptance: dict) -> dict[str, dict]:
    return {check["name"]: check for check in acceptance["checks"]}


def test_acceptance_passes_with_every_metric_finite():
    acceptance = fit_and_eval_transient._acceptance(_acceptance_result())
    assert acceptance["verdict"] == "PASS"
    assert acceptance["pass"] is True
    assert all(check["pass"] is True for check in acceptance["checks"])


def test_acceptance_boundaries_follow_the_plan_wording():
    result = _acceptance_result()
    # plan 3.1: the field error is "<= 1e-2" (equality passes), the frequency and
    # the Cl/Cd errors are "< 1%" / "< 2%" (equality fails)
    _set_path(result, "max_error_within_tau__U", 1e-2)
    acceptance = fit_and_eval_transient._acceptance(result)
    assert _checks(acceptance)["U max relative L2 error within tau<=5"]["pass"] is True

    result = _acceptance_result()
    _set_path(result, "max_error_within_tau__U", 1.0e-2 * (1.0 + 1e-9))
    assert _checks(fit_and_eval_transient._acceptance(result))[
        "U max relative L2 error within tau<=5"
    ]["pass"] is False

    result = _acceptance_result()
    _set_path(result, "frequency__relative_error", 1e-2)
    _set_path(result, "scalars__forces__cl_amplitude__relative_error", 2e-2)
    _set_path(result, "scalars__forces__cd_mean__relative_error", 2e-2)
    _set_path(result, "energy__rom__relative_change", 0.05)
    _set_path(result, "energy__rom__max_norm_ratio", 2.0)
    checks = _checks(fit_and_eval_transient._acceptance(result))
    assert checks["latent frequency relative error"]["pass"] is False
    assert checks["Cl amplitude relative error"]["pass"] is False
    assert checks["Cd mean relative error"]["pass"] is False
    assert checks["cycle-averaged energy drift (ROM, absolute)"]["pass"] is False
    assert checks["max ||a|| / training max"]["pass"] is True


ACCEPTANCE_METRICS = [
    ("max_error_within_tau__U", "U max relative L2 error within tau<=5"),
    ("frequency__relative_error", "latent frequency relative error"),
    ("scalars__forces__cl_amplitude__relative_error", "Cl amplitude relative error"),
    ("scalars__forces__cd_mean__relative_error", "Cd mean relative error"),
    ("energy__rom__max_norm_ratio", "max ||a|| / training max"),
    ("energy__rom__relative_change", "cycle-averaged energy drift (ROM, absolute)"),
]


@pytest.mark.parametrize("path, name", ACCEPTANCE_METRICS)
@pytest.mark.parametrize("bad", [None, float("nan")])
def test_missing_or_nonfinite_metrics_fail(path, name, bad):
    result = _acceptance_result()
    _set_path(result, path, bad)
    acceptance = fit_and_eval_transient._acceptance(result)
    check = _checks(acceptance)[name]
    assert check["pass"] is False, check
    assert check["reason"]
    assert acceptance["verdict"] == "FAIL"
    assert acceptance["pass"] is False


def test_acceptance_fails_when_the_forces_or_the_energy_are_missing():
    result = _acceptance_result()
    result["scalars"] = None
    result["energy"] = {"rom": None}
    acceptance = fit_and_eval_transient._acceptance(result)
    checks = _checks(acceptance)
    for name in (
        "Cl amplitude relative error",
        "Cd mean relative error",
        "max ||a|| / training max",
        "cycle-averaged energy drift (ROM, absolute)",
    ):
        assert checks[name]["pass"] is False, name
        assert checks[name]["reason"], name
    assert acceptance["verdict"] == "FAIL"


def test_acceptance_is_incomplete_without_the_mesh():
    result = _acceptance_result()
    result["phi_divergence"] = None
    acceptance = fit_and_eval_transient._acceptance(result)
    check = _checks(acceptance)["phi max cell divergence <= max(100 x FOM, 1e-10)"]
    assert check["pass"] is None
    assert "mesh" in check["reason"]
    assert acceptance["verdict"] == "INCOMPLETE"
    assert acceptance["pass"] is False


def test_acceptance_fails_a_nonfinite_phi_divergence():
    result = _acceptance_result()
    result["phi_divergence"] = {
        "rom_max": float("nan"),
        "fom_max": 1.0e-13,
        "limit": 1.0e-11,
        "ok": False,
    }
    acceptance = fit_and_eval_transient._acceptance(result)
    check = _checks(acceptance)["phi max cell divergence <= max(100 x FOM, 1e-10)"]
    assert check["pass"] is False
    assert check["reason"] == "phi contains non-finite values"
    assert acceptance["verdict"] == "FAIL"


def test_acceptance_requires_the_opinf_divergence_flag():
    acceptance = fit_and_eval_transient._acceptance(_acceptance_result(), {}, "opinf")
    check = _checks(acceptance)["OpInf did not diverge"]
    assert check["pass"] is False
    assert check["reason"]
    assert acceptance["verdict"] == "FAIL"


def test_print_acceptance_shows_the_verdict_and_the_skips(capsys):
    result = _acceptance_result()
    result["phi_divergence"] = None
    rows = [{"method": "dmd", "acceptance": fit_and_eval_transient._acceptance(result)}]
    fit_and_eval_transient._print_acceptance(rows)
    out = capsys.readouterr().out
    assert "  dmd: INCOMPLETE" in out
    assert "[SKIP] phi max cell divergence" in out
    assert "skipped: phi max cell divergence" in out


# -- --write-fields -------------------------------------------------------------------------


def test_write_fields_option(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit_and_eval_transient.py", "--study", "syn", "--write-fields", "25"],
    )
    assert fit_and_eval_transient.parse_args().write_fields == 25

    monkeypatch.setattr(
        sys,
        "argv",
        ["fit_and_eval_transient.py", "--study", "syn", "--write-fields", "0"],
    )
    with pytest.raises(SystemExit):
        fit_and_eval_transient.parse_args()
    assert "positive integer" in capsys.readouterr().err


def test_write_indices_and_time_names():
    assert fit_and_eval_transient.write_indices(90, 501, 25) == list(range(90, 501, 25))
    assert len(fit_and_eval_transient.write_indices(90, 501, 25)) == 17
    with pytest.raises(ValueError, match="positive"):
        fit_and_eval_transient.write_indices(0, 10, 0)
    with pytest.raises(ValueError, match="outside"):
        fit_and_eval_transient.write_indices(11, 10, 1)

    # OpenFOAM timeFormat general, timePrecision 6: the names of the FOM case
    names = [fit_and_eval_transient.time_name(t) for t in
             (60, 60.2, 78.19999999999999, 100.2, 160)]  # fmt: skip
    assert names == ["60", "60.2", "78.2", "100.2", "160"]


def test_field_difference_validates_shapes():
    pred = np.array([[1.0, 2.0, 3.0]])
    fom = np.array([[0.5, 1.0, 1.5]])
    assert np.allclose(
        fit_and_eval_transient.field_difference(pred, fom), [0.5, 1.0, 1.5]
    )
    with pytest.raises(ValueError, match="does not match"):
        fit_and_eval_transient.field_difference(np.zeros(2), np.zeros(3))


SYN_TIMES = np.array([9.8, 10.0, 10.2, 10.4, 10.6])


def _synthetic_fields(times=SYN_TIMES):
    t = np.asarray(times, dtype=float)
    U = np.stack([np.column_stack([t, np.ones_like(t), np.zeros_like(t)]),
                  np.column_stack([2.0 * t, np.ones_like(t), np.zeros_like(t)])], axis=1)
    p = np.column_stack([t, -t])
    phi = np.zeros((t.size, 7))
    phi[:, 0] = 0.5 * t
    phi[:, 1] = -0.5
    phi[:, 2] = 0.5
    return {"U": U, "p": p, "phi": phi}


def _write_fom_case(tmp_path, names=("10", "10.2", "10.4")):
    settings = Settings(
        cases_dir=tmp_path / "cases",
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
    )
    cases = Cases(settings)
    root = cases.path("cylinder_transient_syn/run_0000", must_exist=False)
    mesh_helper._write_two_cell_mesh(root)
    (root / "system").mkdir(parents=True, exist_ok=True)
    (root / "system" / "controlDict").write_text("application pimpleFoam;\n", encoding="utf-8")
    (root / "run_0000.foam").write_text("", encoding="utf-8")
    for name in names:
        mesh_helper._write_time(root, name, float(name))
    # give U a boundary value so the writer's template handling can be checked
    from fvmrom.foam import FoamField, write_field

    write_field(
        root,
        "10.2",
        FoamField(
            "U",
            "volVectorField",
            np.array([[10.2, 1.0, 0.0], [20.4, 1.0, 0.0]]),
            {"left": {"type": "patch", "value": np.array([9.0, 9.0, 9.0])}},
        ),
    )
    return settings, cases, root


def _prediction(fields, indices, offset=1.0):
    return {
        "U": np.asarray(fields["U"])[indices] + offset,
        "p": np.asarray(fields["p"])[indices] + offset,
        "phi": np.asarray(fields["phi"])[indices] + offset,
    }


def _rom_time_dirs(case_root):
    return sorted(
        path.name
        for path in case_root.iterdir()
        if path.is_dir() and fit_and_eval_transient.TIME_DIR_RE.match(path.name)
    )


def test_write_rom_fields_writes_rom_fom_and_errors(tmp_path, capsys):
    settings, cases, root = _write_fom_case(tmp_path)
    from fvmrom.foam import read_topology

    topo = read_topology(root)
    fields = _synthetic_fields()
    indices = fit_and_eval_transient.write_indices(2, len(SYN_TIMES), 2)
    assert indices == [2, 4]
    predictions = _prediction(fields, indices)

    info = fit_and_eval_transient._write_rom_fields(
        cases, "syn", "dmd", SYN_TIMES, fields, predictions, indices, 2, topo
    )

    dest = cases.path("cylinder_transient_syn_rom_dmd/run_0000")
    assert info["n_written"] == 1 and info["n_skipped"] == 1
    assert info["times"] == ["10.2"] and info["skipped"] == ["10.6"]
    assert (dest / "constant" / "polyMesh" / "points").is_file()
    assert (dest / "system" / "controlDict").is_file()
    assert (dest / "run_0000.foam").is_file()
    assert _rom_time_dirs(dest) == ["10.2"]

    U = read_field(dest, "10.2", "U")
    assert np.allclose(U.internal, predictions["U"][0])
    assert U.dimensions == "[0 0 0 0 0 0 0]"  # the synthetic template default
    U_fom = read_field(dest, "10.2", "U_fom")
    assert np.allclose(U_fom.internal, fields["U"][2])
    U_err = read_field(dest, "10.2", "U_err")
    assert np.allclose(U_err.internal, predictions["U"][0] - fields["U"][2])
    p_err = read_field(dest, "10.2", "p_err")
    assert np.allclose(p_err.internal, predictions["p"][0] - fields["p"][2])
    assert np.allclose(read_field(dest, "10.2", "p_fom").internal, fields["p"][2])

    phi = read_field(dest, "10.2", "phi")
    assert np.allclose(phi.internal, predictions["phi"][0][:1])
    assert np.allclose(phi.boundary["left"]["value"], predictions["phi"][0][1:2])
    # U/p boundary values stay the FOM template ones (as in the stage-1 writer)
    assert np.allclose(read_field(dest, "10.2", "U_err").boundary["left"]["value"], 9.0)

    out = capsys.readouterr().out
    assert "skipped 1 of 2" in out and "wrote 1 ROM time" in out


def test_write_rom_fields_rerun_clears_only_time_directories(tmp_path):
    settings, cases, root = _write_fom_case(tmp_path)
    from fvmrom.foam import read_topology

    topo = read_topology(root)
    fields = _synthetic_fields()
    dest = cases.path("cylinder_transient_syn_rom_dmd/run_0000", must_exist=False)

    first = _prediction(fields, [2])
    fit_and_eval_transient._write_rom_fields(
        cases, "syn", "dmd", SYN_TIMES, fields, first, [2], 2, topo
    )
    assert _rom_time_dirs(dest) == ["10.2"]
    # stale artefacts of an interrupted run: a time directory and a non-time directory
    (dest / "999").mkdir()
    (dest / "notes").mkdir()
    (dest / "notes" / "keep.txt").write_text("keep\n", encoding="utf-8")

    second = _prediction(fields, [2, 3, 4], offset=5.0)
    info = fit_and_eval_transient._write_rom_fields(
        cases, "syn", "dmd", SYN_TIMES, fields, second, [2, 3, 4], 1, topo
    )
    assert info["n_written"] == 2 and info["n_skipped"] == 1
    assert _rom_time_dirs(dest) == ["10.2", "10.4"]
    assert (dest / "notes" / "keep.txt").is_file()
    U = read_field(dest, "10.2", "U")
    assert np.allclose(U.internal, second["U"][0])  # overwritten, not accumulated


def test_write_rom_fields_needs_the_fom_case(tmp_path):
    settings, cases, root = _write_fom_case(tmp_path)
    from fvmrom.foam import read_topology

    topo = read_topology(root)
    fields = _synthetic_fields()
    with pytest.raises(SystemExit, match="FOM case .* is missing"):
        fit_and_eval_transient._write_rom_fields(
            cases, "absent", "dmd", SYN_TIMES, fields, _prediction(fields, [2]), [2], 2, topo
        )


def test_sweeps_are_kept_without_sweep_options(study_dir, tmp_path, monkeypatch, capsys):
    run_main(monkeypatch, tmp_path, "--sweep-modes", "2,4")
    first = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    assert first["sweeps"]["modes"]

    run_main(monkeypatch, tmp_path)
    second = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    assert second["sweeps"] == first["sweeps"]
    assert second["config"]["write_fields"] is None
    assert "kept the sweeps" in capsys.readouterr().out


def test_sweeps_of_two_methods_are_merged(study_dir, tmp_path, monkeypatch):
    run_main(monkeypatch, tmp_path, "--methods", "dmd", "--sweep-modes", "2,4")
    first = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    assert list(first["sweeps"]["modes"]) == ["dmd"]
    assert list(first["methods"]) == ["dmd"]

    run_main(monkeypatch, tmp_path, "--methods", "optdmd", "--sweep-modes", "2,4")
    second = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    assert list(second["sweeps"]["modes"]) == ["dmd", "optdmd"]
    assert second["sweeps"]["modes"]["dmd"] == first["sweeps"]["modes"]["dmd"]
    assert [row["n_modes_U"] for row in second["sweeps"]["modes"]["optdmd"]] == [2, 4]
    # the main evaluation still describes only the methods of this run
    assert list(second["methods"]) == ["optdmd"]
    assert second["sweeps"]["train_periods"] == {}


def test_sweep_modes_keeps_the_old_train_periods(study_dir, tmp_path, monkeypatch):
    run_main(
        monkeypatch, tmp_path, "--methods", "dmd", "--sweep-train-periods", "2,3"
    )
    first = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    assert list(first["sweeps"]["train_periods"]) == ["dmd"]
    assert first["sweeps"]["modes"] == {}

    run_main(monkeypatch, tmp_path, "--methods", "dmd", "--sweep-modes", "2,4")
    second = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    assert second["sweeps"]["train_periods"] == first["sweeps"]["train_periods"]
    assert [row["n_modes_U"] for row in second["sweeps"]["modes"]["dmd"]] == [2, 4]


def test_legacy_list_sweeps_are_migrated(study_dir, tmp_path, monkeypatch):
    legacy_modes = [
        {"method": "dmd", "n_modes_U": 3, "status": "ok"},
        {"method": "optdmd", "n_modes_U": 5, "status": "ok"},
    ]
    legacy_periods = [{"method": "bopdmd", "train_periods": 2.0, "status": "ok"}]
    (study_dir / "report.json").write_text(
        json.dumps(
            {"sweeps": {"modes": legacy_modes, "train_periods": legacy_periods}}
        ),
        encoding="utf-8",
    )

    run_main(monkeypatch, tmp_path, "--methods", "dmd", "--sweep-train-periods", "2,3")
    report = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    modes = report["sweeps"]["modes"]
    assert list(modes) == ["dmd", "optdmd"]
    assert modes["dmd"] == [legacy_modes[0]]
    assert modes["optdmd"] == [legacy_modes[1]]
    periods = report["sweeps"]["train_periods"]
    assert list(periods) == ["bopdmd", "dmd"]
    assert periods["bopdmd"] == legacy_periods
    assert [row["train_periods"] for row in periods["dmd"]] == [2.0, 3.0]


# -- fit_info -------------------------------------------------------------------------------


def test_fit_info_is_reported_per_method(study_dir, tmp_path, monkeypatch):
    run_main(monkeypatch, tmp_path, "--methods", "dmd,optdmd")
    report = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    keys = {"rank", "nfev", "success", "n_used", "n_rejected"}
    dmd = report["methods"]["dmd"]["fit_info"]
    opt = report["methods"]["optdmd"]["fit_info"]
    assert set(dmd) == keys and set(opt) == keys
    assert dmd["rank"] == 4  # one eigenvalue per retained latent mode
    assert dmd["nfev"] is None and dmd["success"] is None
    assert dmd["n_used"] is None and dmd["n_rejected"] is None
    assert opt["rank"] == 4
    assert isinstance(opt["nfev"], int) and opt["nfev"] >= 1
    assert opt["success"] is True
    assert opt["n_used"] is None and opt["n_rejected"] is None
    json.dumps(report)  # the whole report stays JSON serialisable


def test_bopdmd_fit_info_uses_the_reference_fit(study_dir, tmp_path, monkeypatch):
    run_main(monkeypatch, tmp_path, "--methods", "bopdmd")
    report = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    info = report["methods"]["bopdmd"]["fit_info"]
    assert info["rank"] == 4
    assert isinstance(info["nfev"], int) and info["nfev"] >= 1
    assert info["success"] is True
    assert info["n_used"] is not None and info["n_rejected"] is not None
    assert info["n_used"] + info["n_rejected"] == 100


def test_failed_fit_prints_a_warning(study_dir, tmp_path, monkeypatch, capsys):
    def fake_fit_info(rom):
        return {
            "rank": 4,
            "nfev": 7,
            "success": False,
            "n_used": None,
            "n_rejected": None,
        }

    monkeypatch.setattr(fit_and_eval_transient, "_fit_info", fake_fit_info)
    run_main(monkeypatch, tmp_path, "--methods", "dmd")
    report = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    assert report["methods"]["dmd"]["fit_info"]["success"] is False
    out = capsys.readouterr().out
    assert "fit convergence" in out
    assert "WARNING: the dmd fit did not converge" in out


def test_fit_convergence_table_rows():
    rows = [
        {"method": "dmd", "fit_info": {"rank": 4, "nfev": None, "success": None,
                                       "n_used": None, "n_rejected": None}},
        {"method": "bopdmd", "fit_info": {"rank": 4, "nfev": 5, "success": True,
                                          "n_used": 98, "n_rejected": 2}},
    ]
    assert fit_and_eval_transient._count(None) == "n/a"
    assert fit_and_eval_transient._count(np.int64(3)) == "3"
    fit_and_eval_transient._print_fit_convergence(rows)


# -- OpInf methods --------------------------------------------------------------------------

HOPF_STUDY = "hopf"

#: the ``fit_info`` keys that only the OpInf methods report
OPINF_FIT_KEYS = {
    "ridge_selected",
    "ridge",
    "refit_fallback",
    "diverged",
    "diverged_at",
    "recheck",
    "check_periods",
    "check_period_estimated",
    "check_T",
    "recheck_path",
    "ridge_path",
}


def build_hopf_study(root, study=HOPF_STUDY) -> Path:
    """A small synthetic study whose latent state follows the Noack Hopf model.

    ``hopf_trajectory`` supplies the snapshots of a quadratic three-state system,
    so OpInf represents the latent dynamics exactly.  The record starts at
    ``t = 60`` on the limit cycle: fitting OpInf on the growing part leaves the
    quadratic coefficients almost unconstrained.  Cl carries the fundamental of
    the limit cycle (as the cylinder forces do) and Cd is nearly steady.
    """
    data = trajectory_helper.hopf_trajectory(dt=0.1, train_end=60.0, n_pred=800)
    full_times = np.concatenate([data["t_train"], data["t_pred"]])
    keep = full_times >= 60.0 - 1e-9
    times = full_times[keep]
    snaps = {
        name: np.concatenate([data["snap_train"][name], data["snap_pred"][name]])[keep]
        for name in ("U", "p", "phi")
    }
    latent = trajectory_helper.hopf_latent(full_times)[keep]
    forces = np.column_stack([1.5 + 0.01 * latent[:, 1], 0.4 * latent[:, 0]])
    directory = Path(root) / "cylinder_transient" / study
    directory.mkdir(parents=True)
    np.savez(
        directory / "snapshots.npz",
        times=times,
        **snaps,
        forces=forces,
        volumes=data["volumes"],
        phi_weights=data["phi_weights"],
        re=100.0,
    )
    meta = {
        "problem": "cylinder_transient",
        "study": study,
        "config": {"re": 100.0, "snapshot_dt": 0.1},
        "dt": 0.1,
        "n_snapshots": int(times.size),
    }
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return directory


def test_opinf_methods_are_fitted_reported_and_accepted(tmp_path, monkeypatch, capsys):
    study_dir = build_hopf_study(tmp_path)
    run_main(
        monkeypatch,
        tmp_path,
        "--methods",
        "opinf,opinf_norecheck",
        n_modes="U=3,p=9,phi=3",
        study=HOPF_STUDY,
    )
    report = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    assert list(report["methods"]) == ["opinf", "opinf_norecheck"]
    base_keys = {"rank", "nfev", "success", "n_used", "n_rejected"}
    for name in ("opinf", "opinf_norecheck"):
        assert (study_dir / "models" / f"{name}.pkl").is_file()
        info = report["methods"][name]["fit_info"]
        assert set(info) == base_keys | OPINF_FIT_KEYS
        json.dumps(info)
        assert info["rank"] == 3
        assert info["ridge"] is not None and info["ridge_selected"] is not None
        assert isinstance(info["refit_fallback"], bool)
        assert info["diverged"] is False
        assert info["diverged_at"] is None
        assert set(info["ridge_path"]) == {"ridge_rel", "holdout_error", "stable"}

    opinf = report["methods"]["opinf"]["fit_info"]
    assert opinf["recheck"] is True
    assert opinf["check_periods"] == 5.0
    assert opinf["check_T"] is not None
    assert isinstance(opinf["check_period_estimated"], bool)
    assert set(opinf["recheck_path"]) == {
        "ridge_rel",
        "stable",
        "max_norm_ratio_replay",
        "max_norm_ratio_forward",
    }
    plain = report["methods"]["opinf_norecheck"]["fit_info"]
    assert plain["recheck"] is False
    assert plain["recheck_path"] is None
    assert plain["check_periods"] is None and plain["check_T"] is None

    acceptance = report["methods"]["opinf"]["acceptance"]
    # there is no FOM case in this synthetic study: the phi check is skipped and the
    # verdict is INCOMPLETE (a skip is never a pass)
    assert acceptance["verdict"] == "INCOMPLETE"
    assert acceptance["pass"] is False
    checks = {item["name"]: item for item in acceptance["checks"]}
    assert checks["OpInf did not diverge"]["pass"] is True
    assert checks["latent frequency relative error"]["pass"] is True
    assert checks["phi max cell divergence <= max(100 x FOM, 1e-10)"]["pass"] is None
    out = capsys.readouterr().out
    assert "fallback" in out and "diverged" in out


def test_opinf_divergence_is_reported_as_failure(study_dir, tmp_path, monkeypatch, capsys):
    from fvmrom.rom.dynamics import OpInf

    def diverging_predict(self, t, a0=None, t0=None):
        self.diverged_ = True
        self.diverged_at_ = float(self.t_last_)
        return np.full((np.size(t), self.r_), np.nan)

    monkeypatch.setattr(OpInf, "predict", diverging_predict)
    run_main(monkeypatch, tmp_path, "--methods", "opinf")
    report = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    info = report["methods"]["opinf"]["fit_info"]
    assert info["diverged"] is True
    assert info["diverged_at"] is not None

    acceptance = report["methods"]["opinf"]["acceptance"]
    assert acceptance["verdict"] == "FAIL"
    assert acceptance["pass"] is False
    checks = {item["name"]: item for item in acceptance["checks"]}
    assert checks["OpInf did not diverge"]["pass"] is False
    frequency = checks["latent frequency relative error"]
    assert frequency["pass"] is False
    assert "NaN" in frequency["reason"]
    assert checks["max ||a|| / training max"]["pass"] is False
    assert checks["cycle-averaged energy drift (ROM, absolute)"]["pass"] is False

    out = capsys.readouterr().out
    assert "WARNING" in out and "diverged" in out
    assert "nan" in out


# -- readout order-1 diagnostics -------------------------------------------------------------


def test_readout_diagnostics_are_reported(tmp_path, monkeypatch, capsys):
    study_dir = build_hopf_study(tmp_path)
    run_main(
        monkeypatch,
        tmp_path,
        "--methods",
        "dmd",
        n_modes="U=3,p=9,phi=3",
        study=HOPF_STUDY,
    )
    report = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    method = report["methods"]["dmd"]
    one = method["diagnostics"]["readout_order1"]
    assert one["readout_order"] == 1
    assert one["scalar_readout_features"] == {"order2": 10, "order1": 4}
    assert one["n_train"] == report["training_window"]["n_train"]
    for key in ("cl_amplitude", "cd_mean"):
        assert set(one[key]) == {"reference", "prediction", "relative_error"}
        assert one[key]["relative_error"] is not None
    assert set(one["cl_phase_deg"]) == {"1", "5", "10"}
    attribution = one["attribution"]
    assert set(attribution) == {"true_latent", "predicted_latent", "latent_relative_error"}
    json.dumps(report)
    assert report["config"]["diagnostics"] is True
    # the diagnostic fit must not leave a model file behind
    assert sorted(path.name for path in (study_dir / "models").iterdir()) == ["dmd.pkl"]

    out = capsys.readouterr().out
    assert "diagnostic: scalar readout order 1 (not part of acceptance)" in out
    assert "features o2/o1 / n_train" in out


def test_no_diagnostics_option_skips_them(tmp_path, monkeypatch, capsys):
    study_dir = build_hopf_study(tmp_path)
    run_main(
        monkeypatch,
        tmp_path,
        "--methods",
        "dmd",
        "--no-diagnostics",
        n_modes="U=3,p=9,phi=3",
        study=HOPF_STUDY,
    )
    report = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    assert "diagnostics" not in report["methods"]["dmd"]
    assert report["config"]["diagnostics"] is False
    out = capsys.readouterr().out
    assert "diagnostic: scalar readout order 1" not in out


# -- long-horizon stability ------------------------------------------------------------------


def test_long_horizon_stability_of_a_stable_system(tmp_path, monkeypatch, capsys):
    study_dir = build_hopf_study(tmp_path)
    run_main(
        monkeypatch,
        tmp_path,
        "--methods",
        "dmd",
        n_modes="U=3,p=9,phi=3",
        study=HOPF_STUDY,
    )
    report = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    info = report["methods"]["dmd"]["long_horizon"]
    window = report["training_window"]
    assert info["periods"] == 20.0
    assert info["n_points"] > 1
    assert info["t_start"] > window["t_train_end"]
    assert info["t_end"] == pytest.approx(
        window["t_train_end"] + 20.0 * report["fom"]["period"], rel=1e-9
    )
    assert info["diverged"] is False
    assert info["diverged_at"] is None
    assert 0.0 < info["max_norm_ratio"] < info["blowup_factor"]
    assert info["energy"] is not None
    assert info["energy"]["relative_change"] is not None
    assert info["frequency"]["rom"] is not None
    assert info["frequency"]["relative_error"] is not None
    json.dumps(report)
    out = capsys.readouterr().out
    assert "long-horizon stability (20 periods beyond the training window" in out


def test_long_periods_zero_skips_the_stability_check(tmp_path, monkeypatch, capsys):
    study_dir = build_hopf_study(tmp_path)
    run_main(
        monkeypatch,
        tmp_path,
        "--methods",
        "dmd",
        "--long-periods",
        "0",
        n_modes="U=3,p=9,phi=3",
        study=HOPF_STUDY,
    )
    report = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    assert "long_horizon" not in report["methods"]["dmd"]
    out = capsys.readouterr().out
    assert "long-horizon stability" not in out


def test_long_horizon_flags_a_diverged_trajectory(study_dir, tmp_path, monkeypatch, capsys):
    from fvmrom.rom.dynamics import OpInf

    def diverging_predict(self, t, a0=None, t0=None):
        self.diverged_ = True
        self.diverged_at_ = float(self.t_last_)
        return np.full((np.size(t), self.r_), np.nan)

    monkeypatch.setattr(OpInf, "predict", diverging_predict)
    run_main(monkeypatch, tmp_path, "--methods", "opinf", "--long-periods", "2")
    report = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    info = report["methods"]["opinf"]["long_horizon"]
    assert info["diverged"] is True
    assert info["diverged_at"] is not None
    assert info["max_norm_ratio"] is None
    assert info["energy"] is None
    assert info["frequency"]["rom"] is None
    assert info["frequency"]["relative_error"] is None
    out = capsys.readouterr().out
    assert "long-horizon stability (2 periods beyond the training window" in out
    assert "NOTE: the opinf long-horizon trajectory diverged" in out


# -- sweep columns ---------------------------------------------------------------------------


def test_sweep_rows_carry_diverged_and_ridge(tmp_path, monkeypatch, capsys):
    study_dir = build_hopf_study(tmp_path)
    run_main(
        monkeypatch,
        tmp_path,
        "--methods",
        "dmd,opinf",
        "--sweep-modes",
        "3",
        n_modes="U=3,p=9,phi=3",
        study=HOPF_STUDY,
    )
    report = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    modes = report["sweeps"]["modes"]
    dmd = modes["dmd"][0]
    opinf = modes["opinf"][0]
    assert dmd["diverged"] is None and dmd["ridge"] is None
    assert opinf["diverged"] is False
    assert isinstance(opinf["ridge"], float) and opinf["ridge"] > 0.0

    out = capsys.readouterr().out
    assert "latent-dimension sweep (U modes)" in out
    assert "diverged" in out and "ridge" in out
    assert "n/a" in out


def test_sweep_table_tolerates_legacy_rows(capsys):
    rows = [
        {
            "method": "dmd",
            "n_modes_U": 4,
            "n_train": 30,
            "t_train_end": 3.0,
            "status": "ok",
            "max_u_error_within_tau5": 1e-3,
            "u_error_at_tau5": 2e-3,
            "cl_amplitude_relative_error": 1e-2,
            "cd_mean_relative_error": 1e-2,
        },
        {
            "method": "opinf",
            "n_modes_U": 6,
            "n_train": 30,
            "t_train_end": 3.0,
            "status": "ok",
            "max_u_error_within_tau5": 1e-3,
            "u_error_at_tau5": 2e-3,
            "cl_amplitude_relative_error": 1e-2,
            "cd_mean_relative_error": 1e-2,
            "diverged": True,
            "ridge": 1e-4,
        },
    ]
    fit_and_eval_transient._print_sweep("legacy sweep", rows, "n_modes_U")
    out = capsys.readouterr().out
    assert "diverged" in out and "ridge" in out
    assert "n/a" in out and "yes" in out


def test_legacy_sweep_rows_without_new_columns_are_merged(study_dir, tmp_path, monkeypatch):
    legacy = [{"method": "dmd", "n_modes_U": 3, "status": "ok"}]
    (study_dir / "report.json").write_text(
        json.dumps({"sweeps": {"modes": legacy, "train_periods": []}}),
        encoding="utf-8",
    )
    run_main(monkeypatch, tmp_path, "--methods", "dmd", "--sweep-train-periods", "2,3")
    report = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    assert report["sweeps"]["modes"]["dmd"] == legacy
    for row in report["sweeps"]["train_periods"]["dmd"]:
        assert "diverged" in row and "ridge" in row
    json.dumps(report)


# -- no leakage of the prediction window -----------------------------------------------------


def _snapshot_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _perturb_prediction_window(study_dir: Path, t_train_end: float) -> None:
    """Replace the held-out snapshots by clearly different values; the rest is untouched."""
    arrays = _snapshot_arrays(study_dir / "snapshots.npz")
    mask = arrays["times"] > t_train_end
    assert np.any(mask)
    rng = np.random.default_rng(11)
    for name in ("U", "p", "phi"):
        values = arrays[name]
        values[mask] = 3.0 * values[mask] + rng.normal(size=values[mask].shape)
        arrays[name] = values
    # Cl only changes scale, so the whole-record frequency (which decides the split)
    # stays the same and both runs share one training window
    values = arrays["forces"]
    values[mask] = 3.0 * values[mask]
    arrays["forces"] = values
    np.savez(study_dir / "snapshots.npz", **arrays)


def _assert_same_training_products(first: TransientROM, second: TransientROM) -> None:
    assert first.n_modes_ == second.n_modes_
    assert np.array_equal(first.t_train, second.t_train)
    assert np.array_equal(first.latent_train_, second.latent_train_)
    for name in first.pods_:
        assert np.array_equal(first.pods_[name].mean_, second.pods_[name].mean_), name
        assert np.array_equal(first.pods_[name].modes_, second.pods_[name].modes_), name
    for name in first.readouts_:
        assert np.array_equal(first.readouts_[name].coef_, second.readouts_[name].coef_), name
    for attribute in (
        "mu_",
        "omega_",
        "modes_",
        "amplitudes_",
        "lambda_",
        "coef_",
        "scale_",
        "A_",
        "H_",
        "c_",
        "ridge_",
    ):
        if hasattr(first.model_, attribute):
            assert np.array_equal(
                getattr(first.model_, attribute), getattr(second.model_, attribute)
            ), attribute


def _finite_curve(values) -> np.ndarray:
    return np.array([np.nan if value is None else value for value in values], dtype=float)


def test_models_do_not_depend_on_the_prediction_window(tmp_path, monkeypatch, capsys):
    """Script-level leak check: fit twice, rewriting only the held-out snapshots.

    ``tests/test_transient_rom.py`` checks the ``fit`` interface; this one goes
    through ``fit_and_eval_transient.main`` and compares the saved ``models/*.pkl``.
    The changed prediction-window errors prove that the evaluation did read the
    changed data (the test would otherwise be a no-op).
    """
    study_dir = build_hopf_study(tmp_path)
    options = ("--methods", "dmd,opinf", "--no-diagnostics", "--long-periods", "0")
    settings = dict(n_modes="U=3,p=9,phi=3", study=HOPF_STUDY)
    methods = ("dmd", "opinf")

    run_main(monkeypatch, tmp_path, *options, **settings)
    first = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    t_train_end = first["training_window"]["t_train_end"]
    models_before = {
        name: TransientROM.load(study_dir / "models" / f"{name}.pkl") for name in methods
    }
    errors_before = {
        name: _finite_curve(first["methods"][name]["evaluation"]["errors"]["U"])
        for name in methods
    }

    _perturb_prediction_window(study_dir, t_train_end)

    run_main(monkeypatch, tmp_path, *options, **settings)
    second = json.loads((study_dir / "report.json").read_text(encoding="utf-8"))
    assert second["training_window"]["n_train"] == first["training_window"]["n_train"]
    for name in methods:
        model_after = TransientROM.load(study_dir / "models" / f"{name}.pkl")
        _assert_same_training_products(models_before[name], model_after)
        errors_after = _finite_curve(second["methods"][name]["evaluation"]["errors"]["U"])
        assert errors_after.shape == errors_before[name].shape
        assert not np.allclose(errors_after, errors_before[name], equal_nan=True), name
    capsys.readouterr()


