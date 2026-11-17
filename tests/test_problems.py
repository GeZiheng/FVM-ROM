import re

import numpy as np
import pytest

from fvmrom.config import Settings
from fvmrom.foam import Cases
from fvmrom.problems import Problem, get_problem
from fvmrom.problems.cavity import CavityProblem
from fvmrom.problems.cylinder import CylinderProblem


@pytest.fixture
def cases(tmp_path):
    return Cases(Settings(cases_dir=tmp_path / "cases"))


def _write(problem, cases, case, re):
    problem.write_case(cases, case, np.array([re]))
    return cases.path(case)


def _entry(text, name):
    m = re.search(rf"^\s*{name}\s+([^;]+);", text, re.M)
    assert m, f"entry {name} not found"
    return m.group(1).strip()


def test_get_problem():
    cavity = get_problem("cavity")
    assert isinstance(cavity, CavityProblem) and isinstance(cavity, Problem)
    assert cavity.name == "cavity" and cavity.fields == ("U", "p", "phi")
    cylinder = get_problem("cylinder")
    assert isinstance(cylinder, CylinderProblem) and isinstance(cylinder, Problem)
    assert cylinder.space.names == ("Re",)
    with pytest.raises(ValueError, match="Unknown problem"):
        get_problem("airfoil")


def test_param_spaces():
    cavity = get_problem("cavity").space
    assert cavity.lower == (100.0,) and cavity.upper == (1000.0,) and cavity.log == (True,)
    cylinder = get_problem("cylinder").space
    assert cylinder.lower == (10.0,) and cylinder.upper == (40.0,) and cylinder.log == (False,)


def test_cavity_case_files(cases):
    root = _write(get_problem("cavity"), cases, "cav/run_0000", 100.0)
    for rel in (
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
        "system/blockMeshDict",
        "constant/transportProperties",
        "constant/turbulenceProperties",
        "0/U",
        "0/p",
        "Allrun",
    ):
        assert (root / rel).is_file(), rel

    transport = (root / "constant/transportProperties").read_text()
    assert "transportModel" in transport
    assert float(_entry(transport, "nu")) == pytest.approx(0.01)
    assert "simulationType  laminar" in (root / "constant/turbulenceProperties").read_text()

    control = (root / "system/controlDict").read_text()
    assert _entry(control, "application") == "simpleFoam"
    assert _entry(control, "writeFormat") == "ascii"
    assert _entry(control, "writePrecision") == "12"
    assert _entry(control, "purgeWrite") == "1"
    assert int(_entry(control, "endTime")) == 5000
    assert int(_entry(control, "writeInterval")) > 5000
    assert "functions" not in control

    solution = (root / "system/fvSolution").read_text()
    assert "consistent      yes" in solution
    assert _entry(solution, "pRefCell") == "0"
    assert _entry(solution, "pRefValue") == "0"
    assert "residualControl" in solution

    mesh = (root / "system/blockMeshDict").read_text()
    assert "hex (0 1 2 3 4 5 6 7) (64 64 1)" in mesh
    assert mesh.count("hex (") == 1
    for patch in ("movingWall", "fixedWalls", "frontAndBack"):
        assert patch in mesh

    u = (root / "0/U").read_text()
    assert "volVectorField" in u and "uniform (1 0 0)" in u
    p = (root / "0/p").read_text()
    assert "volScalarField" in p and "uniform 0" in p

    allrun = (root / "Allrun").read_text()
    assert allrun.startswith("#!/bin/bash") and "set -e" in allrun
    for app in (
        "blockMesh",
        "checkMesh > log.checkMesh",
        "simpleFoam",
        "postProcess -func writeCellVolumes -latestTime",
    ):
        assert app in allrun
    assert 'grep -q "Failed" log.checkMesh' in allrun
    assert (root / "run_0000.foam").is_file()
    assert (root / "run_0000.foam").read_text() == ""


def test_cavity_velocity_scales_with_nothing_but_re(cases):
    problem = get_problem("cavity")
    a = _write(problem, cases, "cav/run_a", 100.0)
    b = _write(problem, cases, "cav/run_b", 500.0)
    assert (a / "system/blockMeshDict").read_text() == (b / "system/blockMeshDict").read_text()
    assert float(_entry((a / "constant/transportProperties").read_text(), "nu")) == 0.01
    assert float(_entry((b / "constant/transportProperties").read_text(), "nu")) == 0.002


def _cell_count(mesh: str) -> int:
    blocks = re.findall(r"hex \(([^)]*)\) \((\d+) (\d+) (\d+)\)", mesh)
    assert blocks
    return sum(int(nx) * int(ny) * int(nz) for _, nx, ny, nz in blocks)


def test_cylinder_case(cases):
    root = _write(get_problem("cylinder"), cases, "cyl/run_0000", 20.0)
    transport = (root / "constant/transportProperties").read_text()
    assert float(_entry(transport, "nu")) == pytest.approx(0.05)

    control = (root / "system/controlDict").read_text()
    assert "functions" in control and "forceCoeffs" in control
    for entry, value in (
        ("patches", "(cylinder)"),
        ("rho", "rhoInf"),
        ("rhoInf", "1"),
        ("liftDir", "(0 1 0)"),
        ("dragDir", "(1 0 0)"),
        ("CofR", "(0 0 0)"),
        ("pitchAxis", "(0 0 1)"),
        ("magUInf", "1"),
        ("lRef", "1"),
        ("Aref", "0.1"),
    ):
        assert _entry(control, entry) == value, entry
    assert "type            forceCoeffs;" in control

    allrun = (root / "Allrun").read_text()
    assert "checkMesh > log.checkMesh" in allrun and 'grep -q "Failed"' in allrun
    assert (root / "run_0000.foam").is_file()

    mesh = (root / "system/blockMeshDict").read_text()
    assert mesh.count("hex (") == 12
    assert _cell_count(mesh) == 32768
    assert mesh.count("arc ") == 8
    assert "convertToMeters 1" in mesh
    for patch in ("inlet", "outlet", "top", "bottom", "cylinder", "frontAndBack"):
        assert patch in mesh

    u = (root / "0/U").read_text()
    assert u.count("type            slip;") == 2
    assert "noSlip" in u and "uniform (1 0 0)" in u
    p = (root / "0/p").read_text()
    assert p.count("type            empty;") == 1
    assert "fixedValue" in p and "value           uniform 0" in p


def test_writes_only_one_parameter(cases):
    with pytest.raises(ValueError, match="single parameter"):
        get_problem("cavity").write_case(cases, "bad", np.array([100.0, 200.0]))


def test_refuses_existing_case(cases):
    problem = get_problem("cavity")
    problem.write_case(cases, "taken", np.array([100.0]))
    with pytest.raises(FileExistsError):
        problem.write_case(cases, "taken", np.array([100.0]))
