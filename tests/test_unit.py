from pathlib import Path

import pytest

from fvmrom.config import Settings
from fvmrom.foam.case import Cases, safe_join
from fvmrom.foam.commands import build_run_script, validate_app
from fvmrom.foam.logs import parse_log
from fvmrom.foam.wsl import win_to_wsl, wsl_to_win


def test_path_mapping():
    assert win_to_wsl(r"D:\Projects\FVM-ROM\cases") == "/mnt/d/Projects/FVM-ROM/cases"
    assert wsl_to_win("/mnt/d/Projects/x") == Path("D:/Projects/x")
    with pytest.raises(ValueError):
        wsl_to_win("/home/user")


def test_safe_join(tmp_path):
    assert safe_join(tmp_path, "a/b") == (tmp_path / "a" / "b").resolve()
    for bad in ("../x", "a/../../x", "C:/Windows", "/etc/passwd"):
        with pytest.raises(ValueError):
            safe_join(tmp_path, bad)


@pytest.mark.parametrize("app", ["blockMesh", "./Allrun", "foamToVTK", "icoFoam"])
def test_valid_apps(app):
    assert validate_app(app) == app


@pytest.mark.parametrize("app", ["rm -rf /", "a;b", "../x", "/bin/sh", "a b", "", "$(id)"])
def test_invalid_apps(app):
    with pytest.raises(ValueError):
        validate_app(app)


def test_build_run_script_quotes_args():
    s = build_run_script("/mnt/d/c", "icoFoam", ["-case", "a b; rm x"], 4, "log.icoFoam")
    assert "mpirun -np 4" in s and "-parallel" in s
    assert "'a b; rm x'" in s


def test_build_run_script_rejects_bad_log():
    with pytest.raises(ValueError):
        build_run_script("/mnt/d/c", "icoFoam", [], 1, "../log")


def test_cases_summary(tmp_path):
    case = tmp_path / "c1"
    (case / "system").mkdir(parents=True)
    (case / "0").mkdir()
    (case / "0.1").mkdir()
    (case / "constant").mkdir()
    (case / "system" / "controlDict").write_text(
        "application icoFoam;\nendTime 0.5; // c\ndeltaT 0.005;\n"
    )
    cs = Cases(Settings(cases_dir=tmp_path))
    assert cs.list_cases() == ["c1"]
    s = cs.summary("c1")
    assert s["time_dirs"] == ["0", "0.1"]
    assert s["control_dict"]["application"] == "icoFoam"
    assert s["control_dict"]["endTime"] == "0.5"
    assert not s["has_mesh"]
    cs.write("c1", "system/test", "a\nb\n")
    assert (case / "system" / "test").read_bytes() == b"a\nb\n"  # no CRLF translation
    with pytest.raises(ValueError):
        cs.write("c1", "../evil", "x")


def test_parse_log(tmp_path):
    log = tmp_path / "log.simpleFoam"
    log.write_text(
        "Time = 1\n"
        "smoothSolver:  Solving for Ux, Initial residual = 1, Final residual = 0.05, No Iterations 2\n"
        "Time = 2\n"
        "smoothSolver:  Solving for Ux, Initial residual = 0.5, Final residual = 0.02, No Iterations 3\n"
        "End\n"
    )
    r = parse_log(log)
    assert r["last_time"] == "2" and r["n_time_steps"] == 2
    assert r["fields"]["Ux"]["last_initial"] == 0.5
    assert r["finished"] and not r["fatal_error"]
