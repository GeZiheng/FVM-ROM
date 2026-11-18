import re

import numpy as np
import pytest

from fvmrom.core.config import Settings
from fvmrom.foam import Cases
from fvmrom.case_templates import Problem, get_problem
from fvmrom.case_templates.cavity import CavityProblem
from fvmrom.case_templates.cylinder import CylinderGeometry, CylinderProblem, block_mesh_dict

#: block_mesh_dict() for the default geometry, before the parametrisation (regression)
DEFAULT_MESH_SHA256 = "9340567d098c2f2d416ee8713e8679fd4fe5024594b3e227cd7b56d67e5b3c09"


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

def test_default_block_mesh_dict_is_unchanged():
    import hashlib

    text = block_mesh_dict()
    assert hashlib.sha256(text.encode()).hexdigest() == DEFAULT_MESH_SHA256
    assert block_mesh_dict(CylinderGeometry()) == text


def test_cylinder_geometry_counts():
    base = CylinderGeometry()
    counts = base.counts()
    assert counts["n_theta"] == 32 and counts["n_radial"] == 40
    assert counts["n_inner"] == 32
    assert counts["n_x_in"] == 48 and counts["n_x_out"] == 144 and counts["n_y_far"] == 48
    assert counts["cells"] == 32768
    assert _cell_count(block_mesh_dict(base)) == counts["cells"]

    wide = CylinderGeometry(x_in=-15.0, x_out=40.0, y_far=20.0)
    counts = wide.counts()
    assert counts["n_x_in"] == 80 and counts["n_x_out"] == 240
    assert counts["n_y_far"] == 112 and counts["n_inner"] == 32
    assert _cell_count(block_mesh_dict(wide)) == counts["cells"]

    fine = CylinderGeometry(res=1.5)
    counts = fine.counts()
    assert counts["n_theta"] == 48 and counts["n_radial"] == 60
    assert counts["n_x_in"] == 72 and counts["n_x_out"] == 216
    assert counts["n_inner"] == 48 and counts["n_y_far"] == 72
    assert _cell_count(block_mesh_dict(fine)) == counts["cells"]

    # the far-field spacing follows 0.15625/res, the ring resolution scales with res
    assert wide.counts()["cells"] != base.counts()["cells"]
    assert fine.counts()["cells"] != base.counts()["cells"]


def test_cylinder_geometry_validation():
    with pytest.raises(ValueError, match="res"):
        CylinderGeometry(res=0.0)
    with pytest.raises(ValueError, match="x_in"):
        CylinderGeometry(x_in=-2.0)
    with pytest.raises(ValueError, match="x_out"):
        CylinderGeometry(x_out=2.0)
    with pytest.raises(ValueError, match="y_far"):
        CylinderGeometry(y_far=2.0)
    with pytest.raises(ValueError, match="half_size"):
        CylinderGeometry(half_size=0.4)


def test_cylinder_problem_geometry(cases):
    geometry = CylinderGeometry(res=0.5)
    problem = CylinderProblem(geometry=geometry)
    root = _write(problem, cases, "cyl/run_0000", 20.0)
    mesh = (root / "system/blockMeshDict").read_text()
    assert mesh == block_mesh_dict(geometry)
    assert _cell_count(mesh) == geometry.counts()["cells"]
    assert _cell_count(mesh) < 32768

# -- transient case template ---------------------------------------------------------------

from fvmrom.case_templates.cylinder_transient import (  # noqa: E402
    CylinderTransientProblem,
    TransientConfig,
)


def _transient_case(cases, config, case="cyl_transient/run_0000"):
    problem = CylinderTransientProblem(config)
    problem.write_case(cases, case)
    return cases.path(case), problem


def test_transient_case_files(cases):
    config = TransientConfig(re=100.0, snapshot_dt=0.2, t_spin=2.0, t_end=4.0)
    root, problem = _transient_case(cases, config)
    assert problem.name == "cylinder_transient" and problem.fields == ("U", "p", "phi")
    for rel in (
        "system/controlDict",
        "system/controlDict.spin",
        "system/controlDict.prod",
        "system/fvSchemes",
        "system/fvSolution",
        "system/setFieldsDict",
        "system/blockMeshDict",
        "constant/transportProperties",
        "constant/turbulenceProperties",
        "0/U",
        "0/p",
        "Allrun",
        "run_0000.foam",
    ):
        assert (root / rel).is_file(), rel

    assert float(_entry((root / "constant/transportProperties").read_text(), "nu")) == 0.01
    assert "simulationType  laminar" in (
        root / "constant/turbulenceProperties"
    ).read_text()

    spin = (root / "system/controlDict.spin").read_text()
    prod = (root / "system/controlDict.prod").read_text()
    for control in (spin, prod):
        assert _entry(control, "application") == "pimpleFoam"
        assert _entry(control, "writeFormat") == "ascii"
        assert _entry(control, "writePrecision") == "10"
        assert _entry(control, "adjustTimeStep") == "no"
        assert _entry(control, "writeControl") == "timeStep"
        assert _entry(control, "startFrom") == "latestTime"
        assert float(_entry(control, "deltaT")) == pytest.approx(0.005)
        assert "type            forceCoeffs;" in control
        assert "writeInterval   1;" in control  # every step for the force coefficients
        assert "writeControl    timeStep;" in control
    assert float(_entry(spin, "endTime")) == 2.0
    assert int(spin.split("writeInterval")[1].split(";")[0]) == 400
    assert _entry(spin, "purgeWrite") == "3"  # keep the last three checkpoints
    assert float(_entry(prod, "endTime")) == 4.0
    assert int(prod.split("writeInterval")[1].split(";")[0]) == 40
    assert _entry(prod, "purgeWrite") == "0"
    assert (root / "system/controlDict").read_text() == spin

    schemes = (root / "system/fvSchemes").read_text()
    assert "default         backward;" in schemes
    assert "div(phi,U)      Gauss linearUpwind grad(U);" in schemes
    solution = (root / "system/fvSolution").read_text()
    assert "nOuterCorrectors        1;" in solution
    assert "nCorrectors             2;" in solution
    assert "momentumPredictor       yes;" in solution
    assert '"(p|pFinal)"' in solution and '"(U|UFinal)"' in solution

    fields = (root / "system/setFieldsDict").read_text()
    assert "box (0.6 -0.5 -1) (4 0.5 1)" in fields
    assert "volVectorFieldValue U (1 0.1 0)" in fields

    mesh = (root / "system/blockMeshDict").read_text()
    assert mesh == block_mesh_dict(config.geometry)

    allrun = (root / "Allrun").read_text()
    assert allrun.startswith("#!/bin/bash") and "set -e" in allrun
    assert "if [ ! -d constant/polyMesh ]; then" in allrun
    assert "    blockMesh > log.blockMesh" in allrun
    assert "    checkMesh > log.checkMesh" in allrun
    assert 'grep -q "Failed" log.checkMesh' in allrun
    assert "    setFields > log.setFields" in allrun
    assert "    postProcess -func writeCellVolumes -time 0" in allrun
    assert "cp system/controlDict.spin system/controlDict" in allrun
    assert "cp system/controlDict.prod system/controlDict" in allrun
    assert allrun.count("pimpleFoam >> log.pimpleFoam") == 2
    assert "pimpleFoam > log.pimpleFoam" not in allrun
    assert allrun.count("foamListTimes -latestTime") == 2
    assert 'end="2"' in allrun and 'end="4"' in allrun


def test_transient_phases_degenerated(cases):
    # t_spin == 0: one production run from time 0
    config = TransientConfig(snapshot_dt=0.2, t_spin=0.0, t_end=1.0)
    assert config.phases == ("production",)
    assert config.checkpoint_steps == 0
    root, _ = _transient_case(cases, config, "cyl_prod/run_0000")
    assert not (root / "system/controlDict.spin").exists()
    prod = (root / "system/controlDict.prod").read_text()
    assert _entry(prod, "startFrom") == "latestTime"
    assert int(prod.split("writeInterval")[1].split(";")[0]) == 40
    assert (root / "system/controlDict").read_text() == prod
    allrun = (root / "Allrun").read_text()
    assert allrun.count("pimpleFoam >> log.pimpleFoam") == 1
    assert "controlDict.spin" not in allrun

    # t_spin == t_end: pilot, only the spin-up phase with its checkpoints
    config = TransientConfig(snapshot_dt=0.2, t_spin=1.0, t_end=1.0)
    assert config.phases == ("spin",)
    root, _ = _transient_case(cases, config, "cyl_pilot/run_0000")
    assert not (root / "system/controlDict.prod").exists()
    spin = (root / "system/controlDict.spin").read_text()
    assert _entry(spin, "startFrom") == "latestTime"
    assert int(spin.split("writeInterval")[1].split(";")[0]) == 200
    assert _entry(spin, "purgeWrite") == "3"
    allrun = (root / "Allrun").read_text()
    assert allrun.count("pimpleFoam >> log.pimpleFoam") == 1
    assert "controlDict.prod" not in allrun


def test_transient_checkpoint_interval():
    # t_spin / snapshot_dt = 10, divisors 1, 2, 5, 10 (snapshot_steps = 40)
    def steps(checkpoint_dt: float, **kw) -> int:
        return TransientConfig(
            snapshot_dt=0.2, checkpoint_dt=checkpoint_dt, t_spin=2.0, t_end=4.0, **kw
        ).checkpoint_steps

    assert steps(10.0) == 400  # the whole spin-up: one write at t_spin
    assert steps(100.0) == 400  # a huge checkpoint interval does not add writes
    assert steps(1.0) == 200  # 5 snapshots = 1.0
    assert steps(0.4) == 80  # 2 snapshots = 0.4
    assert steps(0.3) == 40  # no divisor of 10 between 1 and 1.5 snapshots
    assert steps(0.05) == 40  # below snapshot_dt: every snapshot
    for checkpoint_dt in (0.05, 0.3, 0.4, 1.0, 10.0):
        assert 400 % steps(checkpoint_dt) == 0  # t_spin always lands on a write

    # t_spin / snapshot_dt = 5 is prime: no intermediate checkpoint unless the limit allows 5
    prime = TransientConfig(snapshot_dt=0.4, checkpoint_dt=1.0, t_spin=2.0, t_end=2.0)
    assert prime.checkpoint_steps == 80  # 1 snapshot = 0.4, the only divisor below 1.0
    assert TransientConfig(
        snapshot_dt=0.4, checkpoint_dt=2.0, t_spin=2.0, t_end=2.0
    ).checkpoint_steps == 400

    with pytest.raises(ValueError, match="checkpoint_dt"):
        TransientConfig(checkpoint_dt=0.0, t_end=1.0)


def test_transient_timestep_rule():
    base = TransientConfig(snapshot_dt=0.2, t_end=1.0)
    assert base.timestep == pytest.approx(0.005)
    assert base.snapshot_steps == 40 and base.spin_steps == 0

    coarse = TransientConfig(geometry=CylinderGeometry(res=0.5), snapshot_dt=0.2, t_end=1.0)
    assert coarse.timestep == pytest.approx(0.01)
    assert coarse.snapshot_steps == 20

    fine = TransientConfig(geometry=CylinderGeometry(res=1.5), snapshot_dt=0.2, t_end=1.0)
    assert fine.timestep == pytest.approx(0.2 / 60.0)
    assert fine.snapshot_steps == 60

    explicit = TransientConfig(snapshot_dt=0.2, t_end=1.0, dt=0.02)
    assert explicit.timestep == 0.02 and explicit.snapshot_steps == 10
    with pytest.raises(ValueError, match="multiple"):
        TransientConfig(snapshot_dt=0.2, t_end=1.0, dt=0.03).timestep


def test_transient_validation():
    with pytest.raises(ValueError, match="t_spin"):
        TransientConfig(t_spin=2.0, t_end=1.0)
    with pytest.raises(ValueError, match="multiple"):
        TransientConfig(snapshot_dt=0.2, t_end=1.1)
    with pytest.raises(ValueError, match="t_end"):
        TransientConfig(t_end=0.0)
    with pytest.raises(ValueError, match="perturb"):
        TransientConfig(t_end=1.0, perturb=-0.1)
    with pytest.raises(ValueError, match="snapshot_dt"):
        TransientConfig(snapshot_dt=0.0, t_end=1.0)


def test_transient_perturbation_can_be_disabled(cases):
    config = TransientConfig(snapshot_dt=0.2, t_end=1.0, perturb=0.0)
    root, _ = _transient_case(cases, config)
    assert not (root / "system/setFieldsDict").exists()
    assert "setFields" not in (root / "Allrun").read_text()
