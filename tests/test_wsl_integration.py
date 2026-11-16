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


def test_refuses_non_openfoam_executable(env):
    from fvmrom.foam.commands import build_run_script

    s, wsl, cases, _ = env
    (cases.root / "c" / "system").mkdir(parents=True)
    (cases.root / "c" / "system" / "controlDict").write_text("")
    r = wsl.run(build_run_script(cases.wsl_path("c"), "ls", [], 1, "log.ls"), timeout=30)
    assert r.returncode == 126
