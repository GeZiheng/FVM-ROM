"""Transient cylinder on the real WSL2 + OpenFOAM install (deselect with -m "not wsl")."""

import re
import time

import numpy as np
import pytest

from fvmrom.core.config import Settings
from fvmrom.foam import Cases, Jobs, read_field
from fvmrom.foam.jobs import ACTIVE_STATES
from fvmrom.foam.transient import read_forces, run_transient
from fvmrom.case_templates.cylinder import CylinderGeometry
from fvmrom.case_templates.cylinder_transient import CylinderTransientProblem, TransientConfig

pytestmark = pytest.mark.wsl


def _settings(tmp_path) -> Settings:
    settings = Settings(
        cases_dir=tmp_path / "cases",
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
    )
    settings.cases_dir.mkdir()
    return settings


def _time_dirs(case_dir) -> list[float]:
    return sorted(
        float(p.name) for p in case_dir.iterdir() if re.fullmatch(r"\d+(\.\d+)?", p.name)
    )


def _last_log_time(case_dir) -> float | None:
    log = (case_dir / "log.pimpleFoam").read_text(errors="replace")
    times = re.findall(r"^Time = ([\d.eE+-]+)\s*$", log, re.M)
    return float(times[-1]) if times else None


def _wait_for_checkpoint(case_dir, t_spin: float, timeout: float = 300.0) -> float:
    """Return once a spin-up checkpoint is written and the solver has moved past it."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        done = [t for t in _time_dirs(case_dir) if 0.0 < t < t_spin]
        if done:
            latest = _last_log_time(case_dir)
            if latest is not None and latest > done[0] + 0.05:
                return done[0]
        time.sleep(0.2)
    raise AssertionError(f"no spin-up checkpoint before t_spin={t_spin} within {timeout:.0f} s")


def _wait_until_idle(jobs: Jobs, job_id: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if jobs.status(job_id, tail_lines=0)["state"] not in ACTIVE_STATES:
            return
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} was still running {timeout:.0f} s after the stop")


def test_transient_small_case_two_phases(tmp_path):
    settings = _settings(tmp_path)
    cases, jobs = Cases(settings), Jobs(settings)
    config = TransientConfig(
        re=100.0,
        geometry=CylinderGeometry(res=0.5),
        snapshot_dt=0.2,
        t_spin=1.0,
        t_end=2.0,
    )
    problem = CylinderTransientProblem(config)
    run = run_transient(
        problem, "wsl_transient", settings=settings, cases=cases, jobs=jobs, poll_s=1.0
    )
    assert run.converged, run.message
    assert run.max_courant is not None and 0.0 < run.max_courant <= 0.6
    assert config.timestep == pytest.approx(0.01)

    case_dir = cases.path(run.case)
    log = (case_dir / "log.pimpleFoam").read_text(errors="replace")
    assert len(re.findall(r"^End\s*$", log, re.M)) == 2
    assert "Courant Number mean:" in log

    assert (case_dir / "0" / "V").is_file()
    initial = read_field(case_dir, "0", "U").internal
    assert np.count_nonzero(initial[:, 1]) > 0  # setFields broke the symmetry

    times = _time_dirs(case_dir)
    assert times[0] == 0.0 and times[-1] == pytest.approx(2.0)
    assert times == pytest.approx([0.0, 1.0] + [1.0 + 0.2 * k for k in range(1, 6)])

    forces = read_forces(case_dir)
    assert forces["time"].size > 100
    assert np.all(np.diff(forces["time"]) > 0.0)
    assert np.all(np.isfinite(forces["cd"])) and np.all(np.isfinite(forces["cl"]))
    assert np.abs(forces["cl"]).max() > 0.0
    restart_dirs = sorted(
        p.name for p in (case_dir / "postProcessing" / "forceCoeffs").iterdir()
    )
    assert restart_dirs == ["0", "1"]


def test_transient_spin_checkpoints_are_purged(tmp_path):
    settings = _settings(tmp_path)
    cases, jobs = Cases(settings), Jobs(settings)
    config = TransientConfig(
        geometry=CylinderGeometry(res=0.5),
        snapshot_dt=0.1,
        checkpoint_dt=0.4,
        t_spin=1.6,
        t_end=1.6,
    )
    assert config.checkpoint_steps == 40  # 4 snapshots = 0.4
    problem = CylinderTransientProblem(config)
    run = run_transient(
        problem, "wsl_purge", settings=settings, cases=cases, jobs=jobs, poll_s=1.0
    )
    assert run.converged, run.message

    case_dir = cases.path(run.case)
    times = _time_dirs(case_dir)
    # writes at 0.4, 0.8, 1.2, 1.6; purgeWrite 3 keeps the last three of them
    assert times == pytest.approx([0.0, 0.8, 1.2, 1.6])
    assert (case_dir / "0" / "V").is_file()
    assert (case_dir / "1.6" / "U").is_file()


def test_transient_resume_from_checkpoint(tmp_path):
    settings = _settings(tmp_path)
    cases, jobs = Cases(settings), Jobs(settings)
    config = TransientConfig(
        geometry=CylinderGeometry(res=0.5),
        snapshot_dt=0.1,
        checkpoint_dt=0.5,
        t_spin=1.0,
        t_end=1.2,
    )
    assert config.checkpoint_steps == 50  # 5 snapshots = 0.5
    problem = CylinderTransientProblem(config)
    case = f"{problem.name}_wsl_resume/run_0000"
    problem.write_case(cases, case)
    case_dir = cases.path(case)

    job = jobs.start(case, "./Allrun", [])
    jobs.wait_pid(job["job_id"])
    checkpoint = _wait_for_checkpoint(case_dir, config.t_spin)
    jobs.wait_pid(job["job_id"])
    jobs.stop(job["job_id"])
    _wait_until_idle(jobs, job["job_id"])
    assert checkpoint == pytest.approx(0.5)
    killed = _time_dirs(case_dir)
    assert killed[0] == 0.0 and killed[-1] <= config.t_spin
    set_fields_mtime = (case_dir / "log.setFields").stat().st_mtime_ns

    run = run_transient(
        problem,
        "wsl_resume",
        settings=settings,
        cases=cases,
        jobs=jobs,
        resume=True,
        poll_s=1.0,
    )
    assert run.converged, run.message

    log = (case_dir / "log.pimpleFoam").read_text(errors="replace")
    assert len(re.findall(r"^End\s*$", log, re.M)) == 2
    starts = [float(s) for s in re.findall(r"^Create mesh for time = ([\d.eE+-]+)", log, re.M)]
    # the second run resumed from a checkpoint (not from 0), the last one is the production phase
    assert len(starts) >= 2 and starts[0] == 0.0 and starts[1] > 0.0
    assert starts[-1] == pytest.approx(config.t_spin)

    # blockMesh/checkMesh/setFields/writeCellVolumes were skipped on the second Allrun
    assert (case_dir / "log.setFields").stat().st_mtime_ns == set_fields_mtime

    times = _time_dirs(case_dir)
    assert times[0] == 0.0 and times[-1] == pytest.approx(1.2)
    assert times == pytest.approx([0.0, 0.5, 1.0, 1.1, 1.2])
    mid = [t for t in times if 0.0 < t < config.t_spin]
    assert 0 < len(mid) <= 3  # checkpoints of the interrupted run were purged
    assert (case_dir / "1" / "U").is_file()  # t_spin (written as '1') feeds the production phase

    forces = read_forces(case_dir)
    assert np.all(np.diff(forces["time"]) > 0.0)
    assert forces["time"][-1] == pytest.approx(1.2)
    assert forces["time"][0] < 0.1
    restart_dirs = sorted(
        p.name for p in (case_dir / "postProcessing" / "forceCoeffs").iterdir()
    )
    assert restart_dirs == ["0", "0.5", "1"]  # the production phase restarted at t_spin
