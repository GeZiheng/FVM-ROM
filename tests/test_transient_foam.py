"""Transient runs and force-coefficient reading, without OpenFOAM (fake jobs/cases)."""

from pathlib import Path

import numpy as np
import pytest

from fvmrom.core.config import Settings
from fvmrom.foam import Cases
from fvmrom.foam.transient import courant_max, read_forces, run_transient
from fvmrom.case_templates.cylinder_transient import CylinderTransientProblem, TransientConfig

LOG_OK = """\
Time = 0.005
Courant Number mean: 0.01 max: 0.31
End

Time = 2.005
Courant Number mean: 0.02 max: 0.28
End
"""


class FakeJobs:
    def __init__(self, cases, log_text=None, states=("completed",), fail_status=False):
        self.cases = cases
        self.log_text = log_text
        self.states = list(states)
        self.fail_status = fail_status
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.stopped_cases: list[str] = []

    def start(self, case, app, args, np=1, log_name=None):
        self.started.append(case)
        if self.log_text is not None:
            self.cases.write(case, "log.pimpleFoam", self.log_text)
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


def _coefficient(path: Path, rows, header=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Force and moment coefficients"]
    if header:
        lines.append("# Time\tCd\tCd(f)\tCd(r)\tCl\tCl(f)\tCl(r)")
    for time, cd, cl in rows:
        lines.append(f"{time:.10g}\t{cd:.10e}\t0\t0\t{cl:.10e}\t0\t0")
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
