"""End-to-end checks against the real WSL2 + OpenFOAM install (deselect with -m "not wsl")."""

import time

import pytest

from fvmrom.config import Settings
from fvmrom.foam import Cases, Jobs, Wsl
from fvmrom.foam import tutorials as tut

pytestmark = pytest.mark.wsl


@pytest.fixture
def env(tmp_path):
    s = Settings(cases_dir=tmp_path / "cases", state_dir=tmp_path / "state")
    s.cases_dir.mkdir()
    return s, Wsl(s), Cases(s), Jobs(s)


def test_env(env):
    _, wsl, _, _ = env
    r = wsl.run('echo "$WM_PROJECT_VERSION"')
    assert r.returncode == 0 and r.stdout.strip() == "v2312"


def test_cavity_workflow(env):
    from fvmrom.mcp_server import server  # noqa: F401  (import check only)

    s, wsl, cases, jobs = env
    assert any(t.endswith("icoFoam/cavity/cavity") for t in tut.list_tutorials(wsl, "cavity"))
    tut.copy_tutorial(wsl, cases, "incompressible/icoFoam/cavity/cavity", "cavity")
    assert cases.list_cases() == ["cavity"]

    from fvmrom.foam.commands import build_run_script

    body = build_run_script(cases.wsl_path("cavity"), "blockMesh", [], 1, "log.blockMesh")
    assert wsl.run(body, timeout=60).returncode == 0
    assert cases.summary("cavity")["has_mesh"]

    job = jobs.start("cavity", "icoFoam", [])
    for _ in range(60):
        st = jobs.status(job["job_id"])
        if st["state"] not in ("starting", "running"):
            break
        time.sleep(1)
    assert st["state"] == "completed", st
    assert cases.time_dirs("cavity")[-1] == "0.5"


_CONTROL_DICT = """FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application icoFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 0.1;
deltaT 0.005;
writeControl timeStep;
writeInterval 10;
"""
_BLOCKMESH = """FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
convertToMeters 1;
vertices ((0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 0.1) (1 0 0.1) (1 1 0.1) (0 1 0.1));
blocks (hex (0 1 2 3 4 5 6 7) (4 4 1) simpleGrading (1 1 1));
boundary
(
    walls { type wall; faces ((3 7 6 2) (0 4 7 3) (2 6 5 1) (1 5 4 0)); }
    frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }
);
"""


def test_custom_case_from_scratch(env):
    from fvmrom.foam.commands import build_run_script

    s, wsl, cases, _ = env
    cases.create("mine")
    cases.write("mine", "system/controlDict", _CONTROL_DICT)
    cases.write("mine", "system/blockMeshDict", _BLOCKMESH)
    body = build_run_script(cases.wsl_path("mine"), "blockMesh", [], 1, "log.blockMesh")
    r = wsl.run(body, timeout=60)
    assert r.returncode == 0, (cases.root / "mine" / "log.blockMesh").read_text()
    summary = cases.summary("mine")
    assert summary["has_mesh"] and summary["control_dict"]["application"] == "icoFoam"


def test_refuses_non_openfoam_executable(env):
    from fvmrom.foam.commands import build_run_script

    s, wsl, cases, _ = env
    (cases.root / "c" / "system").mkdir(parents=True)
    (cases.root / "c" / "system" / "controlDict").write_text("")
    r = wsl.run(build_run_script(cases.wsl_path("c"), "ls", [], 1, "log.ls"), timeout=30)
    assert r.returncode == 126
