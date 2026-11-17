"""Transient cylinder on the real WSL2 + OpenFOAM install (deselect with -m "not wsl")."""

import re

import numpy as np
import pytest

from fvmrom.core.config import Settings
from fvmrom.foam import Cases, Jobs, read_field
from fvmrom.foam.transient import read_forces, run_transient
from fvmrom.case_templates.cylinder import CylinderGeometry
from fvmrom.case_templates.cylinder_transient import CylinderTransientProblem, TransientConfig

pytestmark = pytest.mark.wsl


def test_transient_small_case_two_phases(tmp_path):
    settings = Settings(
        cases_dir=tmp_path / "cases",
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
    )
    settings.cases_dir.mkdir()
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

    times = sorted(
        float(p.name) for p in case_dir.iterdir() if re.fullmatch(r"\d+(\.\d+)?", p.name)
    )
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
