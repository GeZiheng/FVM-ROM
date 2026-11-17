"""Flow past a cylinder: 2D O-grid (4 ring blocks + 8 far-field blocks), parameter Re."""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np

from ..foam import Cases
from ..params import ParamSpace
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

RADIUS = 0.5
HALF_SIZE = 2.5  # inner square [-HALF_SIZE, HALF_SIZE]^2 holding the cylinder
X_IN, X_OUT, Y_FAR = -10.0, 25.0, 10.0
Z0, Z1 = 0.0, 0.1

N_THETA = 32  # cells per 90 degree arc
N_RADIAL = 40
N_INNER = 32  # cells along one side of the inner square
N_X_IN = 48  # x cells between the inlet and the inner square
N_X_OUT = 144  # x cells between the inner square and the outlet
N_Y_FAR = 48  # y cells between the inner square and the top/bottom boundary

_A = RADIUS / sqrt(2.0)

# z0 layer: 0-11 far-field grid nodes, 12-15 inner square corners, 16-19 circle
# points at 45 degrees. The z1 layer repeats the same points at Z1 (+20).
_POINTS = [
    (X_IN, -Y_FAR), (-HALF_SIZE, -Y_FAR), (HALF_SIZE, -Y_FAR), (X_OUT, -Y_FAR),
    (X_IN, -HALF_SIZE), (X_OUT, -HALF_SIZE),
    (X_IN, HALF_SIZE), (X_OUT, HALF_SIZE),
    (X_IN, Y_FAR), (-HALF_SIZE, Y_FAR), (HALF_SIZE, Y_FAR), (X_OUT, Y_FAR),
    (-HALF_SIZE, -HALF_SIZE), (HALF_SIZE, -HALF_SIZE),
    (-HALF_SIZE, HALF_SIZE), (HALF_SIZE, HALF_SIZE),
    (_A, _A), (-_A, _A), (-_A, -_A), (_A, -_A),
]  # fmt: skip

# name -> (bottom vertices, (n_x, n_y)); every block spans one z cell.
_BLOCKS = {
    "top": ((17, 16, 15, 14), (N_THETA, N_RADIAL)),
    "right": ((16, 19, 13, 15), (N_THETA, N_RADIAL)),
    "bottom": ((19, 18, 12, 13), (N_THETA, N_RADIAL)),
    "left": ((18, 17, 14, 12), (N_THETA, N_RADIAL)),
    "bottomLeft": ((0, 1, 12, 4), (N_X_IN, N_Y_FAR)),
    "bottomMiddle": ((1, 2, 13, 12), (N_INNER, N_Y_FAR)),
    "bottomRight": ((2, 3, 5, 13), (N_X_OUT, N_Y_FAR)),
    "leftMiddle": ((4, 12, 14, 6), (N_X_IN, N_INNER)),
    "rightMiddle": ((13, 5, 7, 15), (N_X_OUT, N_INNER)),
    "topLeft": ((6, 14, 9, 8), (N_X_IN, N_Y_FAR)),
    "topMiddle": ((14, 15, 10, 9), (N_INNER, N_Y_FAR)),
    "topRight": ((15, 7, 11, 10), (N_X_OUT, N_Y_FAR)),
}

# Boundary faces: (a, b) denotes the quad (a, a+z, b+z, b), i.e. the outward side of
# the 2D edge a->b.
_INLET = [(0, 4), (4, 6), (6, 8)]
_OUTLET = [(5, 3), (7, 5), (11, 7)]
_TOP = [(8, 9), (9, 10), (10, 11)]
_BOTTOM = [(3, 2), (2, 1), (1, 0)]
_CYLINDER = [(16, 17), (19, 16), (18, 19), (17, 18)]

_U = dict_file(
    "U",
    "volVectorField",
    """
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (1 0 0);

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform (1 0 0);
    }
    outlet
    {
        type            zeroGradient;
    }
    top
    {
        type            slip;
    }
    bottom
    {
        type            slip;
    }
    cylinder
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
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    top
    {
        type            zeroGradient;
    }
    bottom
    {
        type            zeroGradient;
    }
    cylinder
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


_FORCE_COEFFS = """
functions
{
    forceCoeffs
    {
        type            forceCoeffs;
        libs            ("libforces.so");
        patches         (cylinder);
        rho             rhoInf;
        rhoInf          1;
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        CofR            (0 0 0);
        pitchAxis       (0 0 1);
        magUInf         1;
        lRef            1;
        Aref            0.1;
        writeControl    writeTime;
    }
}
"""


class CylinderProblem:
    name = "cylinder"
    fields = FIELDS

    def __init__(
        self,
        re_range: tuple[float, float] = (10.0, 40.0),
        max_iterations: int = 10000,
    ):
        self.space = ParamSpace(
            names=("Re",), lower=(re_range[0],), upper=(re_range[1],), log=(False,)
        )
        self.max_iterations = max_iterations

    def write_case(self, cases: Cases, case: str, mu: np.ndarray) -> None:
        re = re_value(mu)
        cases.create(case)
        cases.write(case, f"{Path(case).name}.foam", "")
        for rel, text in _files(re, self.max_iterations).items():
            cases.write(case, rel, text)


def _files(re: float, max_iterations: int) -> dict[str, str]:
    return {
        "system/controlDict": control_dict(max_iterations, _FORCE_COEFFS),
        "system/fvSchemes": fv_schemes(),
        "system/fvSolution": fv_solution(),
        "system/blockMeshDict": block_mesh_dict(),
        "constant/transportProperties": transport_properties(1.0 / re),
        "constant/turbulenceProperties": turbulence_properties(),
        "0/U": _U,
        "0/p": _P,
        "Allrun": allrun(),
    }


def _face(a: int, b: int) -> str:
    return f"({a} {a + 20} {b + 20} {b})"


def block_mesh_dict() -> str:
    vertices = []
    for x, y in _POINTS:
        vertices.append(f"    ({x:.12g} {y:.12g} {Z0:.12g})")
    for x, y in _POINTS:
        vertices.append(f"    ({x:.12g} {y:.12g} {Z1:.12g})")

    blocks = [
        f"    hex ({' '.join(map(str, bottom))} "
        f"{' '.join(str(v + 20) for v in bottom)}) ({nx} {ny} 1) simpleGrading (1 1 1)"
        for bottom, (nx, ny) in _BLOCKS.values()
    ]

    front_and_back = []
    for bottom, _ in _BLOCKS.values():
        v0, v1, v2, v3 = bottom
        front_and_back.append(f"            ({v0} {v3} {v2} {v1})")
        front_and_back.append(f"            ({v0 + 20} {v1 + 20} {v2 + 20} {v3 + 20})")

    def patch(name: str, kind: str, faces: list[str]) -> str:
        body = "\n".join(f"            {f}" for f in faces)
        return f"    {name}\n    {{\n        type {kind};\n        faces\n        (\n{body}\n        );\n    }}"

    patches = [
        patch("inlet", "patch", [_face(*e) for e in _INLET]),
        patch("outlet", "patch", [_face(*e) for e in _OUTLET]),
        patch("top", "wall", [_face(*e) for e in _TOP]),
        patch("bottom", "wall", [_face(*e) for e in _BOTTOM]),
        patch("cylinder", "wall", [_face(*e) for e in _CYLINDER]),
        patch("frontAndBack", "empty", front_and_back),
    ]

    body = f"""
convertToMeters 1;

vertices
(
{chr(10).join(vertices)}
);

blocks
(
{chr(10).join(blocks)}
);

edges
(
    arc 17 16 (0 {RADIUS:.12g} {Z0:.12g})
    arc 16 19 ({RADIUS:.12g} 0 {Z0:.12g})
    arc 19 18 (0 -{RADIUS:.12g} {Z0:.12g})
    arc 18 17 (-{RADIUS:.12g} 0 {Z0:.12g})
    arc 37 36 (0 {RADIUS:.12g} {Z1:.12g})
    arc 36 39 ({RADIUS:.12g} 0 {Z1:.12g})
    arc 39 38 (0 -{RADIUS:.12g} {Z1:.12g})
    arc 38 37 (-{RADIUS:.12g} 0 {Z1:.12g})
);

boundary
(
{chr(10).join(patches)}
);

mergePatchPairs
();
"""
    return dict_file("blockMeshDict", "dictionary", body)
