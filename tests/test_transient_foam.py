"""Transient runs and force-coefficient reading, without OpenFOAM (fake jobs/cases)."""

import re
from pathlib import Path

import numpy as np
import pytest

from fvmrom.core.config import Settings
from fvmrom.foam import Cases
from fvmrom.foam.transient import courant_max, read_forces, run_transient
from fvmrom.case_templates.cylinder_transient import CylinderTransientProblem, TransientConfig
from fvmrom.rom.evaluate_transient import dominant_frequency

LOG_OK = """\
Time = 0.005
Courant Number mean: 0.01 max: 0.31
End

Time = 2.005
Courant Number mean: 0.02 max: 0.28
End
"""


class FakeJobs:
    def __init__(
        self, cases, log_text=None, states=("completed",), fail_status=False, append=False
    ):
        self.cases = cases
        self.log_text = log_text
        self.states = list(states)
        self.fail_status = fail_status
        self.append = append
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.stopped_cases: list[str] = []

    def start(self, case, app, args, np=1, log_name=None):
        self.started.append(case)
        if self.log_text is not None:
            path = self.cases.path(case) / "log.pimpleFoam"
            with path.open("a" if self.append else "w", encoding="utf-8", newline="\n") as f:
                f.write(self.log_text)
        return {"job_id": "job-1"}

    def status(self, job_id, tail_lines=0):
        if self.fail_status:
            raise RuntimeError("status failed")
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return {"state": state, "job_id": job_id}

    def stop(self, job_id):
        self.stopped.append(job_id)
        return {"job_id": job_id}

    def stop_case(self, case, timeout=30.0):
        self.stopped_cases.append(case)
        return []


def _settings(tmp_path) -> Settings:
    return Settings(
        cases_dir=tmp_path / "cases",
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
    )


def _problem(t_spin=2.0, t_end=4.0) -> CylinderTransientProblem:
    return CylinderTransientProblem(
        TransientConfig(re=100.0, snapshot_dt=0.2, t_spin=t_spin, t_end=t_end)
    )


def test_courant_max_parse():
    assert courant_max(LOG_OK) == pytest.approx(0.31)
    assert courant_max("no Courant here") is None


def _coefficient(path: Path, rows, header=True, time_format=".10g"):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Force and moment coefficients"]
    if header:
        lines.append("# Time\tCd\tCd(f)\tCd(r)\tCl\tCl(f)\tCl(r)")
    for time, cd, cl in rows:
        lines.append(f"{time:{time_format}}\t{cd:.10e}\t0\t0\t{cl:.10e}\t0\t0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_read_forces_merges_restart_directories(tmp_path):
    root = tmp_path / "case"
    _coefficient(
        root / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
        [(0.01, 1.0, 0.1), (0.02, 1.0, 0.2), (2.0, 1.0, 0.5)],
    )
    _coefficient(
        root / "postProcessing" / "forceCoeffs" / "2" / "coefficient.dat",
        [(2.0, 2.0, 9.0), (2.01, 2.0, 0.6)],
    )
    _coefficient(
        root / "postProcessing" / "forceCoeffs" / "junk" / "coefficient.dat",
        [(99.0, 9.0, 9.0)],
    )
    out = read_forces(root)
    assert out["time"] == pytest.approx([0.01, 0.02, 2.0, 2.01])
    # the same time in two directories: the later directory wins
    assert out["cd"] == pytest.approx([1.0, 1.0, 2.0, 2.0])
    assert out["cl"] == pytest.approx([0.1, 0.2, 9.0, 0.6])


def test_read_forces_resume_history_is_continuous(tmp_path):
    # one run killed and resumed twice: every restart repeats the start time
    root = tmp_path / "case"
    first = [(round(0.01 * k, 2), 1.0, 0.1 * k) for k in range(1, 101)]  # 0.01 .. 1.00
    second = [(1.0, 9.0, 9.0)] + [
        (round(1.0 + 0.01 * k, 2), 1.0, 0.5) for k in range(1, 101)
    ]  # 1.00 .. 2.00
    third = [(2.0, 8.0, 8.0)] + [(round(2.0 + 0.01 * k, 2), 1.0, 0.7) for k in range(1, 4)]
    for name, rows in (("0", first), ("1", second), ("2", third)):
        _coefficient(root / "postProcessing" / "forceCoeffs" / name / "coefficient.dat", rows)

    out = read_forces(root)
    assert out["time"].size == 203
    assert np.all(np.diff(out["time"]) > 0.0)
    assert out["time"][0] == pytest.approx(0.01)
    assert out["time"][-1] == pytest.approx(2.03)
    # the later restart directory wins at the seam
    assert out["cd"][np.searchsorted(out["time"], 1.0)] == pytest.approx(9.0)
    assert out["cl"][np.searchsorted(out["time"], 2.0)] == pytest.approx(8.0)


def test_read_forces_skips_bad_rows(tmp_path):
    root = tmp_path / "case"
    path = root / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"
    path.parent.mkdir(parents=True)
    path.write_text(
        "# header\n\n1.0 1.2 0.6 0.6 0.3\nnot a number\n2.0 1.3\n3.0 1.4 0.7 0.7 0.4 0.2 0.2\n",
        encoding="utf-8",
    )
    out = read_forces(root)
    assert out["time"] == pytest.approx([1.0, 3.0])
    assert out["cd"] == pytest.approx([1.2, 1.4])
    assert out["cl"] == pytest.approx([0.3, 0.4])

    with pytest.raises(FileNotFoundError, match="forceCoeffs"):
        read_forces(tmp_path / "empty")


# -- dt snapping: the printed time column has only six significant digits ----------------


DT_FINE = 0.2 / 60  # pilot_fine step: 0.0033333..., printed as 0.003, 0.00333, ...


def _fine_rows(steps, cl=None):
    return [(k * DT_FINE, 1.4, 0.4 if cl is None else float(cl(k * DT_FINE))) for k in steps]


def test_read_forces_snaps_the_printed_times_onto_the_steps(tmp_path):
    # near t = 250 the printed time is rounded to 0.001, coarser than the step itself
    steps = np.arange(74_900, 75_001)
    root = tmp_path / "case"
    _coefficient(
        root / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
        _fine_rows(steps, lambda t: 0.4 * np.sin(2.0 * np.pi * 30.0 * t)),
        time_format=".6g",
    )
    printed = np.array([float(f"{k * DT_FINE:.6g}") for k in steps])
    raw = read_forces(root)
    assert raw["time"] == pytest.approx(printed)  # dt=None keeps the printed column as is
    with pytest.raises(ValueError, match="uniformly spaced"):
        dominant_frequency(raw["time"], raw["cl"])

    out = read_forces(root, dt=DT_FINE)
    assert out["time"] == pytest.approx(steps * DT_FINE, abs=1e-9)
    assert np.allclose(np.diff(out["time"]), DT_FINE, rtol=0.0, atol=1e-9)

    exact = 0.4 * np.sin(2.0 * np.pi * 30.0 * steps * DT_FINE)
    frequency = dominant_frequency(out["time"], out["cl"])
    assert frequency == pytest.approx(dominant_frequency(steps * DT_FINE, exact), rel=1e-6)
    assert frequency == pytest.approx(30.0, rel=1e-2)


def test_read_forces_snaps_before_merging_restarts(tmp_path):
    root = tmp_path / "case"
    first = np.arange(74_960, 74_981)
    second = np.arange(74_980, 75_001)  # starts at the last step of the first directory
    _coefficient(
        root / "postProcessing" / "forceCoeffs" / "249.87" / "coefficient.dat",
        [(k * DT_FINE, 1.4, 0.1) for k in first],
        time_format=".6g",
    )
    _coefficient(
        root / "postProcessing" / "forceCoeffs" / "249.93" / "coefficient.dat",
        [(k * DT_FINE, 1.4, 0.9) for k in second],
        time_format=".6g",
    )

    out = read_forces(root, dt=DT_FINE)
    assert out["time"].size == first.size + second.size - 1  # the seam is not duplicated
    assert np.allclose(np.diff(out["time"]), DT_FINE, rtol=0.0, atol=1e-9)
    assert out["time"][0] == pytest.approx(first[0] * DT_FINE, abs=1e-9)
    assert out["time"][-1] == pytest.approx(second[-1] * DT_FINE, abs=1e-9)
    seam = np.searchsorted(out["time"], second[0] * DT_FINE)
    assert out["cl"][seam] == pytest.approx(0.9)  # the later restart directory wins


def test_read_forces_rejects_a_dt_that_does_not_match(tmp_path):
    root = tmp_path / "case"
    steps = np.arange(74_900, 74_950)
    _coefficient(
        root / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
        _fine_rows(steps),
        time_format=".6g",
    )
    with pytest.raises(ValueError, match="does not match"):
        read_forces(root, dt=3.0 * DT_FINE)  # e.g. snapshot_dt instead of the solver step
    with pytest.raises(ValueError, match="dt must be positive"):
        read_forces(root, dt=0.0)


def test_run_transient_converged(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    jobs = FakeJobs(cases, LOG_OK, states=("running", "completed"))
    run = run_transient(_problem(), "pilot_a", cases=cases, jobs=jobs, poll_s=0.0)
    assert run.converged is True
    assert "2 phase(s) completed" in run.message
    assert run.max_courant == pytest.approx(0.31)
    assert run.case == "cylinder_transient_pilot_a/run_0000"
    assert jobs.started == [run.case]
    assert (cases.path(run.case) / "system/controlDict.spin").is_file()


def test_run_transient_needs_all_phases_to_end(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    log = LOG_OK.replace("End\n", "", 1)  # only the second phase reached 'End'
    jobs = FakeJobs(cases, log)
    run = run_transient(_problem(), "pilot_b", cases=cases, jobs=jobs, poll_s=0.0)
    assert run.converged is False
    assert "1 of 2 phase(s)" in run.message


def test_run_transient_fatal_and_fpe(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    fatal = LOG_OK + "\n--> FOAM FATAL ERROR: something\n"
    run = run_transient(
        _problem(), "fatal", cases=cases, jobs=FakeJobs(cases, fatal), poll_s=0.0
    )
    assert run.converged is False and "FOAM FATAL" in run.message

    run = run_transient(
        _problem(),
        "fpe",
        cases=cases,
        jobs=FakeJobs(cases, LOG_OK + "\nFloating point exception\n"),
        poll_s=0.0,
    )
    assert run.converged is False and "floating point" in run.message

    # the banner mentions 'Floating point exception trapping enabled'; that is no crash
    banner = "trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).\n" + LOG_OK
    run = run_transient(
        _problem(), "banner", cases=cases, jobs=FakeJobs(cases, banner), poll_s=0.0
    )
    assert run.converged is True


def test_run_transient_failed_job(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    jobs = FakeJobs(cases, LOG_OK, states=("failed",))
    run = run_transient(_problem(), "failed", cases=cases, jobs=jobs, poll_s=0.0)
    assert run.converged is False and run.message.startswith("job failed")


def test_run_transient_no_log(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    jobs = FakeJobs(cases, None)
    run = run_transient(_problem(), "nolog", cases=cases, jobs=jobs, poll_s=0.0)
    assert run.converged is False and run.message == "no log.pimpleFoam"


def test_run_transient_resume_skips_finished_case(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    first = FakeJobs(cases, LOG_OK)
    run = run_transient(_problem(), "resume", cases=cases, jobs=first, poll_s=0.0)
    assert run.converged

    (cases.path(run.case) / "4").mkdir()
    second = FakeJobs(cases, LOG_OK)
    again = run_transient(
        _problem(), "resume", cases=cases, jobs=second, resume=True, poll_s=0.0
    )
    assert again.converged and again.message.startswith("skipped:")
    assert second.started == []
    assert second.stopped_cases == []


def test_run_transient_resume_does_not_verify_a_finished_case(tmp_path):
    # pilot_base-style case: finished with an older template, only the forces are still missing
    settings = _settings(tmp_path)
    cases = Cases(settings)
    run = run_transient(_problem(), "keep", cases=cases, jobs=FakeJobs(cases, LOG_OK), poll_s=0.0)
    (cases.path(run.case) / "4").mkdir()
    cases.write(run.case, "system/controlDict.spin", "stale dictionary\n")

    jobs = FakeJobs(cases, LOG_OK)
    again = run_transient(_problem(), "keep", cases=cases, jobs=jobs, resume=True, poll_s=0.0)
    assert again.converged and again.message.startswith("skipped:")
    assert jobs.started == [] and jobs.stopped_cases == []


def test_run_transient_resume_starts_a_missing_case(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    jobs = FakeJobs(cases, LOG_OK)
    run = run_transient(_problem(), "fresh", cases=cases, jobs=jobs, resume=True, poll_s=0.0)
    assert run.converged and not run.message.startswith("skipped:")
    assert jobs.started == [run.case]
    assert jobs.stopped_cases == []
    assert (cases.path(run.case) / "system/controlDict.spin").is_file()


def test_run_transient_resumes_an_unfinished_case(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    problem = _problem()
    case = "cylinder_transient_restart/run_0000"
    problem.write_case(cases, case)
    root = cases.path(case)
    (root / "1.0").mkdir()
    (root / "1.0" / "keep.txt").write_text("checkpoint\n", encoding="utf-8")
    partial = "Time = 0.5\nCourant Number mean: 0.01 max: 0.4\n"  # interrupted: no 'End'
    (root / "log.pimpleFoam").write_text(partial, encoding="utf-8")

    jobs = FakeJobs(cases, LOG_OK, append=True)
    run = run_transient(problem, "restart", cases=cases, jobs=jobs, resume=True, poll_s=0.0)
    assert jobs.stopped_cases == [case]  # leftover processes are stopped first
    assert jobs.started == [case]
    assert run.converged, run.message
    assert run.max_courant == pytest.approx(0.4)  # the interrupted part counts too
    assert (root / "1.0" / "keep.txt").is_file()  # the case was not rebuilt
    log = (root / "log.pimpleFoam").read_text()
    assert log.startswith(partial)  # the interrupted log is appended to, not replaced
    assert len(re.findall(r"^End\s*$", log, re.M)) == 2


def test_run_transient_resume_rejects_a_changed_config(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    problem = _problem()
    case = "cylinder_transient_stale/run_0000"
    problem.write_case(cases, case)
    root = cases.path(case)
    (root / "0" / "keep.txt").write_text("keep\n", encoding="utf-8")
    (root / "log.pimpleFoam").write_text("Time = 0.5\n", encoding="utf-8")
    (root / "system/controlDict.spin").write_text("stale dictionary\n", encoding="utf-8")
    (root / "Allrun").unlink()

    jobs = FakeJobs(cases, LOG_OK, append=True)
    with pytest.raises(ValueError, match="cannot resume") as exc:
        run_transient(problem, "stale", cases=cases, jobs=jobs, resume=True, poll_s=0.0)
    assert "controlDict.spin" in str(exc.value) and "Allrun" in str(exc.value)
    assert jobs.started == []
    assert (root / "0" / "keep.txt").is_file()


def test_run_transient_resume_reports_every_stale_file(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    problem = _problem()
    case = "cylinder_transient_stale2/run_0000"
    problem.write_case(cases, case)
    root = cases.path(case)
    (root / "system/controlDict.spin").write_text("stale dictionary\n", encoding="utf-8")
    (root / "Allrun").write_text("#!/bin/bash\necho old\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot resume .*Allrun.*controlDict.spin") as exc:
        run_transient(
            problem,
            "stale2",
            cases=cases,
            jobs=FakeJobs(cases, LOG_OK, append=True),
            resume=True,
            poll_s=0.0,
        )
    assert "--resume" in str(exc.value)


def test_run_transient_cleans_an_unfinished_case(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    case = "cylinder_transient_rerun/run_0000"
    cases.create(case)
    cases.write(case, "log.pimpleFoam", "old log\n")
    jobs = FakeJobs(cases, LOG_OK)
    run = run_transient(_problem(), "rerun", cases=cases, jobs=jobs, poll_s=0.0)
    assert jobs.stopped_cases == [case]
    assert run.converged
    assert "old log" not in (cases.path(case) / "log.pimpleFoam").read_text()


def test_run_transient_stops_the_job_on_errors(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    jobs = FakeJobs(cases, LOG_OK, fail_status=True)
    with pytest.raises(RuntimeError, match="status failed"):
        run_transient(_problem(), "crash", cases=cases, jobs=jobs, poll_s=0.0)
    assert jobs.stopped == ["job-1"]


def test_run_transient_validates_study(tmp_path):
    settings = _settings(tmp_path)
    with pytest.raises(ValueError, match="invalid study name"):
        run_transient(
            _problem(), "bad/name", cases=Cases(settings), jobs=FakeJobs(None), poll_s=0.0
        )

# -- the pilot script ----------------------------------------------------------------------

import importlib.util  # noqa: E402
import sys  # noqa: E402
from types import SimpleNamespace  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_transient.py"
_spec = importlib.util.spec_from_file_location("generate_transient", SCRIPT)
generate_transient = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate_transient)


def test_geometry_presets_and_overrides():
    geometry = generate_transient.geometry_from_args(
        SimpleNamespace(geometry="wide", x_in=None, x_out=None, y_far=None, res=None)
    )
    assert (geometry.x_in, geometry.x_out, geometry.y_far, geometry.res) == (
        -15.0, 40.0, 20.0, 1.0,
    )
    geometry = generate_transient.geometry_from_args(
        SimpleNamespace(geometry="base", x_in=None, x_out=30.0, y_far=None, res=0.5)
    )
    assert geometry.x_in == -10.0 and geometry.x_out == 30.0 and geometry.res == 0.5
    assert "base" in generate_transient.GEOMETRIES and "fine" in generate_transient.GEOMETRIES


def test_pilot_argument_handling(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["generate_transient.py", "--study", "p", "--t-spin", "2", "--pilot"]
    )
    args = generate_transient.parse_args()
    assert args.t_end == 2.0 and args.pilot

    monkeypatch.setattr(
        sys, "argv", ["generate_transient.py", "--study", "p", "--t-spin", "2", "--t-end", "4"]
    )
    assert generate_transient.parse_args().t_end == 4.0

    monkeypatch.setattr(
        sys, "argv", ["generate_transient.py", "--study", "p", "--t-spin", "2"]
    )
    with pytest.raises(SystemExit):
        generate_transient.parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_transient.py", "--study", "p", "--t-spin", "2", "--t-end", "4", "--pilot"],
    )
    with pytest.raises(SystemExit):
        generate_transient.parse_args()


def test_collect_only_arguments(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_transient.py", "--study", "p", "--t-spin", "1", "--t-end", "2",
         "--collect-only", "--purge-fields", "--workers", "4"],
    )  # fmt: skip
    args = generate_transient.parse_args()
    assert args.collect_only and args.purge_fields and args.workers == 4
    assert args.t_spin == 1.0 and args.t_end == 2.0

    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_transient.py", "--study", "p", "--t-spin", "1", "--t-end", "2",
         "--workers", "0"],
    )  # fmt: skip
    with pytest.raises(SystemExit):
        generate_transient.parse_args()


# -- snapshot collection --------------------------------------------------------------------

from fvmrom.foam import FoamField, write_field  # noqa: E402
from fvmrom.foam.transient import (  # noqa: E402
    TransientRun,
    align_forces,
    collect_transient,
    inspect_run,
    select_snapshot_times,
)
from fvmrom.foam.mesh import phi_weights, read_geometry  # noqa: E402

_MESH_HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {name};
}}
"""

PHI_BOUNDARY = {
    "left": {"type": "patch", "value": np.array([-0.5])},
    "right": {"type": "patch", "value": np.array([0.5])},
    "bottom": {"type": "wall", "value": np.array([0.0, 0.0])},
    "top": {"type": "wall", "value": np.array([0.0, 0.0])},
    "frontAndBack": {"type": "empty", "value": None},
}


def _write_two_cell_mesh(root: Path) -> None:
    """A 2 x 1 x 1 hexahedral mesh: cells of volume 0.05 and 0.1, one internal face."""
    xs, ys, z1 = (0.0, 0.5, 1.5), (0.0, 1.0), 0.1
    points = [(x, y, z) for x in xs for y in ys for z in (0.0, z1)]

    def p(i, j, k):
        return (i * 2 + j) * 2 + k

    def cell(i):
        return [p(i, 0, 0), p(i + 1, 0, 0), p(i + 1, 1, 0), p(i, 1, 0),
                p(i, 0, 1), p(i + 1, 0, 1), p(i + 1, 1, 1), p(i, 1, 1)]  # fmt: skip

    c0, c1 = cell(0), cell(1)
    faces = [[c0[1], c0[2], c0[6], c0[5]]]
    owner = [0]
    neighbour = [1]
    patches = (
        ("left", "patch", [([c0[0], c0[4], c0[7], c0[3]], 0)]),
        ("right", "patch", [([c1[1], c1[2], c1[6], c1[5]], 1)]),
        (
            "bottom",
            "patch",
            [([c0[0], c0[1], c0[5], c0[4]], 0), ([c1[0], c1[1], c1[5], c1[4]], 1)],
        ),
        (
            "top",
            "patch",
            [([c0[3], c0[7], c0[6], c0[2]], 0), ([c1[3], c1[7], c1[6], c1[2]], 1)],
        ),
        (
            "frontAndBack",
            "empty",
            [
                ([c0[0], c0[3], c0[2], c0[1]], 0),
                ([c1[0], c1[3], c1[2], c1[1]], 1),
                ([c0[4], c0[5], c0[6], c0[7]], 0),
                ([c1[4], c1[5], c1[6], c1[7]], 1),
            ],
        ),
    )
    boundary = []
    for name, kind, patch_faces in patches:
        first = len(owner)
        for verts, own in patch_faces:
            faces.append(verts)
            owner.append(own)
        boundary.append((name, kind, len(patch_faces), first))

    poly = root / "constant" / "polyMesh"
    poly.mkdir(parents=True, exist_ok=True)
    (poly / "points").write_text(
        _MESH_HEADER.format(cls="vectorField", name="points")
        + f"\n{len(points)}\n(\n"
        + "\n".join(f"({x:.17g} {y:.17g} {z:.17g})" for x, y, z in points)
        + "\n)\n"
    )
    (poly / "faces").write_text(
        _MESH_HEADER.format(cls="faceList", name="faces")
        + f"\n{len(faces)}\n(\n"
        + "\n".join(f"{len(f)}({' '.join(str(v) for v in f)})" for f in faces)
        + "\n)\n"
    )
    (poly / "owner").write_text(
        _MESH_HEADER.format(cls="labelList", name="owner")
        + f"\n{len(owner)}\n(\n"
        + "\n".join(str(v) for v in owner)
        + "\n)\n"
    )
    (poly / "neighbour").write_text(
        _MESH_HEADER.format(cls="labelList", name="neighbour")
        + f"\n{len(neighbour)}\n(\n"
        + "\n".join(str(v) for v in neighbour)
        + "\n)\n"
    )
    text = _MESH_HEADER.format(cls="polyBoundaryMesh", name="boundary") + f"\n{len(boundary)}\n(\n"
    for name, kind, n, first in boundary:
        text += f"    {name}\n    {{\n        type {kind};\n"
        text += f"        nFaces {n};\n        startFace {first};\n    }}\n"
    (poly / "boundary").write_text(text + ")\n")


def _write_time(root: Path, time: str, values: float) -> None:
    """U = (t, 1, 0) per cell, p = ±t, internal phi = t/2."""
    U = np.array([[values, 1.0, 0.0], [2.0 * values, 1.0, 0.0]])
    p = np.array([values, -values])
    write_field(root, time, FoamField("U", "volVectorField", U, {}))
    write_field(root, time, FoamField("p", "volScalarField", p, {}))
    write_field(
        root,
        time,
        FoamField("phi", "surfaceScalarField", np.array([0.5 * values]), PHI_BOUNDARY),
    )


def _build_case(root: Path, times) -> None:
    _write_two_cell_mesh(root)
    write_field(root, "0", FoamField("V", "volScalarField", np.array([0.05, 0.1]), {}))
    for values in times:
        _write_time(root, f"{values:g}", values)


def _write_forces(path: Path, dt: float = 0.005) -> np.ndarray:
    """Cd = 1 + t, Cl = cos(2 pi t / 6), every solver step up to t = 2."""
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(0.0, 2.0 + 1e-9, dt)
    np.savez(path, time=time, cd=1.0 + time, cl=np.cos(2.0 * np.pi * time / 6.0))
    return time


def _m1_case(tmp_path):
    settings = _settings(tmp_path)
    config = TransientConfig(re=100.0, snapshot_dt=0.2, t_spin=1.0, t_end=2.0)
    problem = CylinderTransientProblem(config)
    case = f"{problem.name}_m1/run_0000"
    root = Cases(settings).path(case, must_exist=False)
    return settings, config, problem, case, root


def test_select_snapshot_times_sorts_and_rounds_below_t_spin_away():
    picked = select_snapshot_times(
        ["postProcessing", "60.4", "59.8", "60.2", "0.1", "60", "constant"],
        t_spin=60.0,
        snapshot_dt=0.2,
    )
    assert [value for value, _ in picked] == pytest.approx([60.0, 60.2, 60.4])
    assert [name for _, name in picked] == ["60", "60.2", "60.4"]


def test_select_snapshot_times_reports_missing_times():
    with pytest.raises(ValueError, match="missing snapshot times.*1.4"):
        select_snapshot_times(["1.0", "1.2", "1.6"], t_spin=1.0, snapshot_dt=0.2)


def test_select_snapshot_times_rejects_non_multiples_and_duplicates():
    with pytest.raises(ValueError, match="not a multiple of snapshot_dt"):
        select_snapshot_times(["1.0", "1.1"], t_spin=1.0, snapshot_dt=0.2)
    with pytest.raises(ValueError, match="name the same time"):
        select_snapshot_times(["1.0", "1.00"], t_spin=1.0, snapshot_dt=0.2)
    with pytest.raises(ValueError, match="no time directories"):
        select_snapshot_times(["constant"], t_spin=1.0, snapshot_dt=0.2)


def test_select_snapshot_times_is_empty_below_t_spin():
    with pytest.raises(ValueError, match="no time directories at or after"):
        select_snapshot_times(["0", "0.5", "1.0"], t_spin=1.5, snapshot_dt=0.2)


def test_align_forces_picks_the_exact_rows():
    time = np.arange(0.0, 2.0 + 1e-9, 0.005)
    cd = 1.0 + time
    cl = np.cos(2.0 * np.pi * time / 6.0)
    times = np.array([1.0, 1.2, 1.4])
    out = align_forces(times, time, cd, cl, tol=0.00125)
    assert out.shape == (3, 2)
    assert out[:, 0] == pytest.approx(1.0 + times)
    assert out[:, 1] == pytest.approx(np.cos(2.0 * np.pi * times / 6.0))


def test_align_forces_rejects_a_time_off_the_force_history():
    time = np.arange(0.0, 2.0 + 1e-9, 0.005)
    with pytest.raises(ValueError, match="no forceCoeffs row at snapshot time 1.0525"):
        align_forces(np.array([1.0525]), time, time, time, tol=0.00125)
    with pytest.raises(ValueError, match="no forceCoeffs rows"):
        align_forces(np.array([1.0]), np.zeros(0), np.zeros(0), np.zeros(0), tol=0.1)
    with pytest.raises(ValueError, match="differ in length"):
        align_forces(np.array([1.0]), time, time[:-1], time, tol=0.1)


def test_collect_transient_writes_snapshots(tmp_path):
    settings, config, problem, case, root = _m1_case(tmp_path)
    times = [1.0 + 0.2 * k for k in range(6)]
    _build_case(root, [0.4, 0.8] + times)  # 0.4 and 0.8 are spin-up checkpoints
    out = tmp_path / "data"
    force_time = _write_forces(out / "forces.npz")

    run = TransientRun(case, config, True, "2 phase(s) completed", 0.3)
    path = collect_transient(problem, "m1", run, out, workers=2, settings=settings)

    with np.load(path) as data:
        assert set(data.files) == {
            "times", "U", "p", "phi", "volumes", "phi_weights", "forces", "re",
        }  # fmt: skip
        assert data["times"] == pytest.approx(times)
        assert data["U"].shape == (6, 2, 3)
        assert data["p"].shape == (6, 2)
        assert data["phi"].shape == (6, 7)
        assert data["volumes"] == pytest.approx([0.05, 0.1])
        assert data["phi_weights"].shape == (7,)
        assert data["re"] == pytest.approx(100.0)
        # the first snapshot at t_spin came from the directory '1'
        assert np.allclose(data["U"][0], [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
        assert data["U"][:, 0, 0] == pytest.approx(times)
        assert data["p"][:, 1] == pytest.approx([-t for t in times])
        assert data["phi"][:, 0] == pytest.approx([0.5 * t for t in times])
        index = np.rint(np.asarray(times) / 0.005).astype(int)
        assert data["forces"][:, 0] == pytest.approx(1.0 + force_time[index])
        assert data["forces"][:, 1] == pytest.approx(
            np.cos(2.0 * np.pi * force_time[index] / 6.0)
        )


def test_collect_transient_reports_a_gap(tmp_path):
    settings, config, problem, case, root = _m1_case(tmp_path)
    _build_case(root, [1.0, 1.2, 1.6])
    out = tmp_path / "data"
    _write_forces(out / "forces.npz")
    run = TransientRun(case, config, True, "ok", None)
    with pytest.raises(ValueError, match="missing snapshot times.*1.4"):
        collect_transient(problem, "m1", run, out, workers=2, settings=settings)


def test_collect_transient_falls_back_to_the_case_forces(tmp_path):
    settings, config, problem, case, root = _m1_case(tmp_path)
    times = [1.0 + 0.2 * k for k in range(6)]
    _build_case(root, times)
    source = _write_forces(tmp_path / "forces.npz")
    lines = ["# Force and moment coefficients", "# Time\tCd\tCd(f)\tCd(r)\tCl"]
    for time, cd, cl in zip(source, 1.0 + source, np.cos(2.0 * np.pi * source / 6.0)):
        lines.append(f"{time:.10g}\t{cd:.10e}\t0\t0\t{cl:.10e}\t0\t0")
    path = root / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = tmp_path / "data"
    run = TransientRun(case, config, True, "ok", 0.3)
    result = collect_transient(problem, "m1", run, out, workers=2, settings=settings)
    assert (out / "forces.npz").is_file()
    with np.load(result) as data:
        assert data["forces"][:, 0] == pytest.approx(1.0 + np.asarray(times))
        assert data["forces"][:, 1] == pytest.approx(
            np.cos(2.0 * np.pi * np.asarray(times) / 6.0)
        )


def test_snapshot_stats_are_finite(tmp_path):
    settings, config, problem, case, root = _m1_case(tmp_path)
    _build_case(root, [1.0 + 0.2 * k for k in range(6)])
    out = tmp_path / "data"
    _write_forces(out / "forces.npz")
    run = TransientRun(case, config, True, "ok", None)
    path = collect_transient(problem, "m1", run, out, workers=2, settings=settings)

    stats = generate_transient.snapshot_stats(path, read_geometry(root))
    assert stats["n_snapshots"] == 6
    for weights in ("unit", "area"):
        values = stats["flux_consistency"][weights]
        assert np.isfinite(values["mean"]) and np.isfinite(values["max"])
        assert values["max"] >= values["mean"] >= 0.0
    assert stats["phi_div_max"] >= 0.0
    assert phi_weights(read_geometry(root)).shape == (7,)


def test_purge_fields_keeps_the_ends(tmp_path):
    case = tmp_path / "case"
    for name in ("0", "0.5", "1", "1.2", "2", "constant", "system"):
        (case / name).mkdir(parents=True)
    config = TransientConfig(snapshot_dt=0.2, t_spin=1.0, t_end=2.0)
    removed = generate_transient.purge_fields(case, config)
    assert removed == ["0.5", "1.2"]
    assert not (case / "0.5").exists() and not (case / "1.2").exists()
    for name in ("0", "1", "2", "constant", "system"):
        assert (case / name).is_dir()


def test_purge_fields_only_touches_time_directories(tmp_path):
    """``float()`` accepts "nan"/"inf"/"1_0"; only OpenFOAM time names are time directories."""
    case = tmp_path / "case"
    names = (
        "0",
        "0.5",
        "10",
        "20",
        "30",
        "nan",
        "inf",
        "1_0",
        "postProcessing",
        "uniform",
        "constant",
        "system",
    )
    for name in names:
        (case / name).mkdir(parents=True)
    config = TransientConfig(snapshot_dt=5.0, t_spin=10.0, t_end=30.0)

    removed = generate_transient.purge_fields(case, config)

    # every time directory but 0, t_spin (10) and the last (30) is removed,
    # including the intermediate snapshots
    assert removed == ["0.5", "20"]
    kept = ("0", "10", "30", "nan", "inf", "1_0", "postProcessing", "uniform",
            "constant", "system")  # fmt: skip
    for name in kept:
        assert (case / name).is_dir(), name


def test_inspect_run_reads_an_existing_case(tmp_path):
    settings = _settings(tmp_path)
    problem = _problem()
    cases = Cases(settings)
    case = f"{problem.name}_inspect/run_0000"
    problem.write_case(cases, case)
    cases.write(case, "log.pimpleFoam", LOG_OK)
    run = inspect_run(problem, "inspect", settings=settings)
    assert run.case == case and run.converged
    assert run.max_courant == pytest.approx(0.31)
    with pytest.raises(FileNotFoundError, match="Case not found"):
        inspect_run(problem, "missing", settings=settings)
