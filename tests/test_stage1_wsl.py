"""Stage-1 end-to-end check on a small cavity (deselect with -m "not wsl")."""

import shutil
import time
import uuid

import numpy as np
import pytest

from fvmrom.config import Settings, get_settings
from fvmrom.foam import (
    Cases,
    Jobs,
    Wsl,
    latest_time,
    phi_divergence,
    phi_vector,
    read_field,
    read_topology,
)
from fvmrom.foam.logs import parse_log
from fvmrom.problems.cavity import CavityProblem

pytestmark = pytest.mark.wsl

TIMEOUT_S = 300.0


def test_small_cavity_end_to_end(tmp_path):
    settings = Settings(cases_dir=tmp_path / "cases", state_dir=tmp_path / "state")
    settings.cases_dir.mkdir()
    cases, jobs = Cases(settings), Jobs(settings)
    problem = CavityProblem(re_range=(50.0, 200.0), grid=16, max_iterations=2000)
    case = "cavity_small/run_0000"
    problem.write_case(cases, case, np.array([100.0]))
    case_dir = cases.path(case)

    job = jobs.start(case, "./Allrun", [])
    state = "starting"
    deadline = time.time() + TIMEOUT_S
    while time.time() < deadline:
        state = jobs.status(job["job_id"], tail_lines=0)["state"]
        if state not in ("starting", "running"):
            break
        time.sleep(1.0)
    log = (case_dir / "log.simpleFoam").read_text(errors="replace")
    assert state == "completed", f"job ended as {state}\n{log[-2000:]}"

    info = parse_log(case_dir / "log.simpleFoam")
    assert info["converged"], info
    assert not info["fatal_error"] and not info["floating_point_exception"]

    topo = read_topology(case_dir)
    assert topo.n_cells == 16 * 16
    time_name = latest_time(case_dir)
    U = read_field(case_dir, time_name, "U")
    p = read_field(case_dir, time_name, "p")
    phi = read_field(case_dir, time_name, "phi")
    volumes = read_field(case_dir, time_name, "V").internal
    assert U.internal.shape == (topo.n_cells, 3)
    assert p.internal.shape == (topo.n_cells,)
    assert volumes.size == topo.n_cells
    assert volumes.sum() == pytest.approx(0.1)  # 1 x 1 x 0.1
    assert np.abs(U.internal[..., 0]).max() > 0.1  # the lid drives the flow

    flux = phi_vector(phi, topo)
    divergence = phi_divergence(flux, topo)
    assert np.abs(divergence).max() < 1e-8


def test_stop_case_kills_a_long_running_job():
    settings = get_settings()
    cases, jobs, wsl = Cases(settings), Jobs(settings), Wsl(settings)
    case = f"_pytest_stop_case_{uuid.uuid4().hex[:8]}/run_0000"
    case_dir = cases.path(case, must_exist=False)
    job = None
    try:
        case_dir.mkdir(parents=True)
        (case_dir / "Allrun").write_text("#!/bin/bash\nsleep 600\n", newline="\n")
        job = jobs.start(case, "./Allrun", [])
        pid = jobs.wait_pid(job["job_id"])
        assert jobs.status(job["job_id"], tail_lines=0)["state"] in ("running", "starting")

        assert jobs.stop_case(case, timeout=60.0) == [job["job_id"]]
        assert jobs.status(job["job_id"], tail_lines=0)["state"] not in (
            "running",
            "starting",
        )

        r = wsl.run(f"ps -eo pgid= | awk '$1 == {pid}' | wc -l", timeout=30, with_env=False)
        assert r.returncode == 0, r.stderr
        assert int(r.stdout.strip()) == 0  # no process of that group is left
    finally:
        if job is not None:
            shutil.rmtree(jobs.dir / job["job_id"], ignore_errors=True)
        shutil.rmtree(case_dir.parent, ignore_errors=True)
