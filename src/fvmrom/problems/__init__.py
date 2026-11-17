"""Parametric steady problems: case templates and their parameter spaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..foam import Cases
from ..params import ParamSpace

#: Fields written to the snapshots, in this order.
FIELDS = ("U", "p", "phi")


@runtime_checkable
class Problem(Protocol):
    name: str
    space: ParamSpace
    fields: tuple[str, ...]

    def write_case(self, cases: Cases, case: str, mu: np.ndarray) -> None:
        """Write a complete case (dictionaries, fields, mesh input, Allrun) for one sample."""


def get_problem(name: str) -> Problem:
    from .cavity import CavityProblem
    from .cylinder import CylinderProblem

    problems = {"cavity": CavityProblem, "cylinder": CylinderProblem}
    try:
        cls = problems[name]
    except KeyError:
        raise ValueError(
            f"Unknown problem {name!r}: choose one of {sorted(problems)}"
        ) from None
    return cls()


# -- shared case template fragments -------------------------------------------------------


def dict_file(object_name: str, cls: str, body: str) -> str:
    return (
        "FoamFile\n{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {cls};\n"
        f"    object      {object_name};\n"
        "}\n\n"
        f"{body.strip()}\n"
    )


def transport_properties(nu: float) -> str:
    return dict_file(
        "transportProperties",
        "dictionary",
        f"transportModel  Newtonian;\n\nnu              {nu:.12g};\n",
    )


def turbulence_properties() -> str:
    return dict_file("turbulenceProperties", "dictionary", "simulationType  laminar;\n")


def control_dict(max_iterations: int, extra: str = "") -> str:
    # writeInterval beyond the iteration cap: only the final (converged) step is written.
    body = f"""
application     simpleFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {max_iterations};

deltaT          1;

writeControl    timeStep;
writeInterval   {max_iterations + 1};
purgeWrite      1;

writeFormat     ascii;
writePrecision  12;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable false;
"""
    if extra:
        body += "\n" + extra
    return dict_file("controlDict", "dictionary", body)


def fv_schemes() -> str:
    return dict_file(
        "fvSchemes",
        "dictionary",
        """
ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
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
""",
    )


def fv_solution(p_tol: float = 1e-5, u_tol: float = 1e-6) -> str:
    return dict_file(
        "fvSolution",
        "dictionary",
        f"""
solvers
{{
    p
    {{
        solver          GAMG;
        tolerance       1e-8;
        relTol          0.01;
        smoother        GaussSeidel;
    }}
    U
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-9;
        relTol          0.1;
    }}
}}

SIMPLE
{{
    nNonOrthogonalCorrectors 0;
    consistent      yes;

    pRefCell        0;
    pRefValue       0;

    residualControl
    {{
        p               {p_tol:g};
        U               {u_tol:g};
    }}
}}

relaxationFactors
{{
    equations
    {{
        U               0.9;
        ".*"            0.9;
    }}
}}
""",
    )


def allrun() -> str:
    return """#!/bin/bash
set -e
cd "$(dirname "$0")"
blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh 2>&1
if grep -q "Failed" log.checkMesh; then
    echo "checkMesh reported failures, see log.checkMesh" >&2
    exit 1
fi
simpleFoam > log.simpleFoam 2>&1
postProcess -func writeCellVolumes -latestTime > log.writeCellVolumes 2>&1
"""


def re_value(mu: np.ndarray) -> float:
    mu = np.asarray(mu, dtype=float).reshape(-1)
    if mu.size != 1:
        raise ValueError(f"expected a single parameter (Re), got {mu.size} values")
    return float(mu[0])
