"""Transient cylinder case (pimpleFoam): spin-up plus a snapshot-writing production run."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

from ..foam import Cases
from . import FIELDS, dict_file, transport_properties, turbulence_properties
from .cylinder import (
    CylinderGeometry,
    block_mesh_dict,
    force_coeffs,
    pressure_field,
    velocity_field,
)

#: biggest time step at res = 1: maximum Courant number about 0.3 on the base grid
DT_MAX_RES1 = 0.005
#: setFields box in the wake; U there becomes (1, perturb, 0)
PERTURB_BOX = ((0.6, -0.5, -1.0), (4.0, 0.5, 1.0))

#: phase name -> suffix of the written ``system/controlDict.<suffix>``
_SUFFIX = {"spin": "spin", "production": "prod"}

_FV_SCHEMES = """
ddtSchemes
{
    default         backward;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      Gauss linearUpwind grad(U);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
"""

_FV_SOLUTION = """
solvers
{
    "(p|pFinal)"
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-9;
        relTol          0;
    }
    "(U|UFinal)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-10;
        relTol          0;
    }
}

PIMPLE
{
    nOuterCorrectors        1;
    nCorrectors             2;
    nNonOrthogonalCorrectors 1;
    momentumPredictor       yes;
}
"""


@dataclass(frozen=True)
class TransientConfig:
    re: float = 100.0
    geometry: CylinderGeometry = CylinderGeometry()
    snapshot_dt: float = 0.2
    t_spin: float = 0.0
    t_end: float = 0.0
    perturb: float = 0.1
    dt: float | None = None

    def __post_init__(self) -> None:
        if not self.re > 0.0:
            raise ValueError(f"re must be positive, got {self.re}")
        if not self.snapshot_dt > 0.0:
            raise ValueError(f"snapshot_dt must be positive, got {self.snapshot_dt}")
        if not self.t_end > 0.0:
            raise ValueError(f"t_end must be positive, got {self.t_end}")
        if not 0.0 <= self.t_spin <= self.t_end:
            raise ValueError(f"need 0 <= t_spin <= t_end, got {self.t_spin} and {self.t_end}")
        if self.perturb < 0.0:
            raise ValueError(f"perturb must be non-negative, got {self.perturb}")
        if self.dt is not None and not self.dt > 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        _steps(self.t_spin, self.snapshot_dt, "t_spin")
        _steps(self.t_end, self.snapshot_dt, "t_end")

    @property
    def timestep(self) -> float:
        """Simulation time step; ``snapshot_dt`` is always an integer multiple of it."""
        if self.dt is not None:
            _steps(self.snapshot_dt, self.dt, "snapshot_dt")
            return float(self.dt)
        dt_max = DT_MAX_RES1 / self.geometry.res
        return self.snapshot_dt / ceil(self.snapshot_dt / dt_max - 1e-9)

    @property
    def snapshot_steps(self) -> int:
        return _steps(self.snapshot_dt, self.timestep, "snapshot_dt")

    @property
    def spin_steps(self) -> int:
        return _steps(self.t_spin, self.timestep, "t_spin")

    @property
    def phases(self) -> tuple[str, ...]:
        """``spin`` keeps only the final time, ``production`` writes every snapshot."""
        if self.t_spin == 0.0:
            return ("production",)
        if self.t_spin == self.t_end:
            return ("spin",)
        return ("spin", "production")


def _steps(total: float, step: float, name: str) -> int:
    n = total / step
    if abs(n - round(n)) > 1e-9 * max(1.0, abs(n)):
        raise ValueError(f"{name} {total:g} is not a multiple of {step:g}")
    return int(round(n))


class CylinderTransientProblem:
    name = "cylinder_transient"
    fields = FIELDS

    def __init__(self, config: TransientConfig | None = None):
        self.config = config or TransientConfig()

    def write_case(self, cases: Cases, case: str) -> None:
        cases.create(case)
        cases.write(case, f"{Path(case).name}.foam", "")
        for rel, text in files(self.config).items():
            cases.write(case, rel, text)


def files(config: TransientConfig) -> dict[str, str]:
    out = {
        "system/fvSchemes": dict_file("fvSchemes", "dictionary", _FV_SCHEMES),
        "system/fvSolution": dict_file("fvSolution", "dictionary", _FV_SOLUTION),
        "system/blockMeshDict": block_mesh_dict(config.geometry),
        "constant/transportProperties": transport_properties(1.0 / config.re),
        "constant/turbulenceProperties": turbulence_properties(),
        "0/U": velocity_field(),
        "0/p": pressure_field(),
        "Allrun": allrun(config),
    }
    for phase in config.phases:
        out[f"system/controlDict.{_SUFFIX[phase]}"] = control_dict(config, phase)
    out["system/controlDict"] = control_dict(config, config.phases[0])
    if config.perturb > 0.0:
        out["system/setFieldsDict"] = set_fields_dict(config.perturb)
    return out


def control_dict(config: TransientConfig, phase: str) -> str:
    if phase == "spin":
        start_from, start_time = "startTime", 0.0
        end_time, interval = config.t_spin, config.spin_steps
    else:
        start_from = "latestTime" if config.t_spin > 0.0 else "startTime"
        start_time = 0.0
        end_time, interval = config.t_end, config.snapshot_steps
    body = f"""
application     pimpleFoam;

startFrom       {start_from};
startTime       {start_time:.12g};
stopAt          endTime;
endTime         {end_time:.12g};

deltaT          {config.timestep:.12g};
adjustTimeStep  no;

writeControl    timeStep;
writeInterval   {interval};
purgeWrite      0;

writeFormat     ascii;
writePrecision  10;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable false;
"""
    return dict_file(
        "controlDict", "dictionary", body + "\n" + force_coeffs("timeStep", 1)
    )


def set_fields_dict(perturb: float) -> str:
    (x0, y0, z0), (x1, y1, z1) = PERTURB_BOX
    body = f"""
defaultFieldValues
(
);

regions
(
    boxToCell
    {{
        box ({x0:g} {y0:g} {z0:g}) ({x1:g} {y1:g} {z1:g});
        fieldValues
        (
            volVectorFieldValue U (1 {perturb:.12g} 0)
        );
    }}
);
"""
    return dict_file("setFieldsDict", "dictionary", body)


def allrun(config: TransientConfig) -> str:
    lines = [
        "#!/bin/bash",
        "set -e",
        'cd "$(dirname "$0")"',
        "blockMesh > log.blockMesh 2>&1",
        "checkMesh > log.checkMesh 2>&1",
        "if grep -q \"Failed\" log.checkMesh; then",
        '    echo "checkMesh reported failures, see log.checkMesh" >&2',
        "    exit 1",
        "fi",
    ]
    if config.perturb > 0.0:
        lines.append("setFields > log.setFields 2>&1")
    lines.append("postProcess -func writeCellVolumes -time 0 > log.writeCellVolumes 2>&1")
    for phase in config.phases:
        suffix = _SUFFIX[phase]
        lines.append(f"cp system/controlDict.{suffix} system/controlDict")
        redirect = ">>" if phase != config.phases[0] else ">"
        lines.append(f"pimpleFoam {redirect} log.pimpleFoam 2>&1")
    return "\n".join(lines) + "\n"
