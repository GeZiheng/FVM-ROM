"""Lid-driven cavity: 64x64 uniform mesh, steady laminar simpleFoam, parameter Re."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..foam import Cases
from ..core.params import ParamSpace
from . import (
    FIELDS,
    allrun,
    control_dict,
    dict_file,
    fv_schemes,
    fv_solution,
    re_value,
    transport_properties,
    turbulence_properties,
)

_U = dict_file(
    "U",
    "volVectorField",
    """
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    movingWall
    {
        type            fixedValue;
        value           uniform (1 0 0);
    }
    fixedWalls
    {
        type            noSlip;
    }
    frontAndBack
    {
        type            empty;
    }
}
""",
)

_P = dict_file(
    "p",
    "volScalarField",
    """
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    movingWall
    {
        type            zeroGradient;
    }
    fixedWalls
    {
        type            zeroGradient;
    }
    frontAndBack
    {
        type            empty;
    }
}
""",
)


class CavityProblem:
    name = "cavity"
    fields = FIELDS

    def __init__(
        self,
        re_range: tuple[float, float] = (100.0, 1000.0),
        grid: int = 64,
        max_iterations: int = 5000,
    ):
        self.space = ParamSpace(
            names=("Re",), lower=(re_range[0],), upper=(re_range[1],), log=(True,)
        )
        self.grid = grid
        self.max_iterations = max_iterations

    def write_case(self, cases: Cases, case: str, mu: np.ndarray) -> None:
        re = re_value(mu)
        cases.create(case)
        cases.write(case, f"{Path(case).name}.foam", "")
        for rel, text in _files(re, self.grid, self.max_iterations).items():
            cases.write(case, rel, text)


def _files(re: float, grid: int, max_iterations: int) -> dict[str, str]:
    nu = 1.0 / re
    block_mesh = dict_file(
        "blockMeshDict",
        "dictionary",
        f"""
convertToMeters 1;

vertices
(
    (0 0 0)
    (1 0 0)
    (1 1 0)
    (0 1 0)
    (0 0 0.1)
    (1 0 0.1)
    (1 1 0.1)
    (0 1 0.1)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({grid} {grid} 1) simpleGrading (1 1 1)
);

edges
();

boundary
(
    movingWall
    {{
        type wall;
        faces ((3 7 6 2));
    }}
    fixedWalls
    {{
        type wall;
        faces ((0 4 7 3) (2 6 5 1) (1 5 4 0));
    }}
    frontAndBack
    {{
        type empty;
        faces ((0 3 2 1) (4 5 6 7));
    }}
);

mergePatchPairs
();
""",
    )
    return {
        "system/controlDict": control_dict(max_iterations),
        "system/fvSchemes": fv_schemes(),
        "system/fvSolution": fv_solution(),
        "system/blockMeshDict": block_mesh,
        "constant/transportProperties": transport_properties(nu),
        "constant/turbulenceProperties": turbulence_properties(),
        "0/U": _U,
        "0/p": _P,
        "Allrun": allrun(),
    }
