"""Flow past a cylinder: 2D O-grid (4 ring blocks + 8 far-field blocks), parameter Re."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
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

RADIUS = 0.5
Z0, Z1 = 0.0, 0.1
#: far-field cell size at ``res = 1`` (7.5/48 = 22.5/144 = 5/32)
FAR_CELL = 0.15625


@dataclass(frozen=True)
class CylinderGeometry:
    """Domain and grid resolution; ``res`` scales the cell count by ``res^2``."""

    x_in: float = -10.0
    x_out: float = 25.0
    y_far: float = 10.0
    half_size: float = 2.5
    res: float = 1.0

    def __post_init__(self) -> None:
        if not self.res > 0.0:
            raise ValueError(f"res must be positive, got {self.res}")
        if not self.half_size > RADIUS:
            raise ValueError(f"half_size must exceed the cylinder radius ({RADIUS})")
        if not self.x_in < -self.half_size:
            raise ValueError("x_in must lie upstream of the inner square")
        if not self.x_out > self.half_size:
            raise ValueError("x_out must lie downstream of the inner square")
        if not self.y_far > self.half_size:
            raise ValueError("y_far must exceed half_size")

    def counts(self) -> dict:
        """Cells per direction in each block family and the total cell count."""
        h = FAR_CELL / self.res
        n_theta = max(1, round(32 * self.res))
        n_radial = max(1, round(40 * self.res))
        counts = {
            "n_theta": n_theta,
            "n_radial": n_radial,
            "n_inner": _cells(2.0 * self.half_size, h),
            "n_x_in": _cells(-self.half_size - self.x_in, h),
            "n_x_out": _cells(self.x_out - self.half_size, h),
            "n_y_far": _cells(self.y_far - self.half_size, h),
        }
        cells = 4 * n_theta * n_radial
        cells += 2 * counts["n_x_in"] * counts["n_y_far"]
        cells += 2 * counts["n_inner"] * counts["n_y_far"]
        cells += 2 * counts["n_x_out"] * counts["n_y_far"]
        cells += (counts["n_x_in"] + counts["n_x_out"]) * counts["n_inner"]
        counts["cells"] = cells
        return counts


def _cells(length: float, h: float) -> int:
    return max(1, round(length / h))


# z0 layer: 0-11 far-field grid nodes, 12-15 inner square corners, 16-19 circle
# points at 45 degrees. The z1 layer repeats the same points at Z1 (+20).
def _plane_points(geometry: CylinderGeometry) -> list[tuple[float, float]]:
    x_in, x_out, y_far = geometry.x_in, geometry.x_out, geometry.y_far
    half = geometry.half_size
    a = RADIUS / sqrt(2.0)
    return [
        (x_in, -y_far), (-half, -y_far), (half, -y_far), (x_out, -y_far),
        (x_in, -half), (x_out, -half),
        (x_in, half), (x_out, half),
        (x_in, y_far), (-half, y_far), (half, y_far), (x_out, y_far),
        (-half, -half), (half, -half),
        (-half, half), (half, half),
        (a, a), (-a, a), (-a, -a), (a, -a),
    ]  # fmt: skip


# name -> (bottom vertices, (count in x, count in y)); every block spans one z cell.
_BLOCKS = {
    "top": ((17, 16, 15, 14), ("n_theta", "n_radial")),
    "right": ((16, 19, 13, 15), ("n_theta", "n_radial")),
    "bottom": ((19, 18, 12, 13), ("n_theta", "n_radial")),
    "left": ((18, 17, 14, 12), ("n_theta", "n_radial")),
    "bottomLeft": ((0, 1, 12, 4), ("n_x_in", "n_y_far")),
    "bottomMiddle": ((1, 2, 13, 12), ("n_inner", "n_y_far")),
    "bottomRight": ((2, 3, 5, 13), ("n_x_out", "n_y_far")),
    "leftMiddle": ((4, 12, 14, 6), ("n_x_in", "n_inner")),
    "rightMiddle": ((13, 5, 7, 15), ("n_x_out", "n_inner")),
    "topLeft": ((6, 14, 9, 8), ("n_x_in", "n_y_far")),
    "topMiddle": ((14, 15, 10, 9), ("n_inner", "n_y_far")),
    "topRight": ((15, 7, 11, 10), ("n_x_out", "n_y_far")),
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


def velocity_field() -> str:
    """``0/U`` of the cylinder case (shared with the transient template)."""
    return _U


def pressure_field() -> str:
    """``0/p`` of the cylinder case (shared with the transient template)."""
    return _P


def force_coeffs(write_control: str = "writeTime", write_interval: int | None = None) -> str:
    """``forceCoeffs`` function object; the interval is written for time-step control."""
    anchor = "writeControl    writeTime;"
    text = _FORCE_COEFFS.replace(anchor, f"writeControl    {write_control};")
    if write_interval is not None:
        text = text.replace(
            f"writeControl    {write_control};",
            f"writeControl    {write_control};\n        writeInterval   {write_interval};",
        )
    return text


class CylinderProblem:
    name = "cylinder"
    fields = FIELDS

    def __init__(
        self,
        re_range: tuple[float, float] = (10.0, 40.0),
        max_iterations: int = 10000,
        geometry: CylinderGeometry | None = None,
    ):
        self.space = ParamSpace(
            names=("Re",), lower=(re_range[0],), upper=(re_range[1],), log=(False,)
        )
        self.max_iterations = max_iterations
        self.geometry = geometry or CylinderGeometry()

    def write_case(self, cases: Cases, case: str, mu: np.ndarray) -> None:
        re = re_value(mu)
        cases.create(case)
        cases.write(case, f"{Path(case).name}.foam", "")
        for rel, text in _files(re, self.max_iterations, self.geometry).items():
            cases.write(case, rel, text)


def _files(re: float, max_iterations: int, geometry: CylinderGeometry | None = None) -> dict:
    return {
        "system/controlDict": control_dict(max_iterations, force_coeffs()),
        "system/fvSchemes": fv_schemes(),
        "system/fvSolution": fv_solution(),
        "system/blockMeshDict": block_mesh_dict(geometry),
        "constant/transportProperties": transport_properties(1.0 / re),
        "constant/turbulenceProperties": turbulence_properties(),
        "0/U": velocity_field(),
        "0/p": pressure_field(),
        "Allrun": allrun(),
    }


def _face(a: int, b: int) -> str:
    return f"({a} {a + 20} {b + 20} {b})"


def block_mesh_dict(geometry: CylinderGeometry | None = None) -> str:
    counts = (geometry or CylinderGeometry()).counts()
    plane = _plane_points(geometry or CylinderGeometry())

    vertices = []
    for x, y in plane:
        vertices.append(f"    ({x:.12g} {y:.12g} {Z0:.12g})")
    for x, y in plane:
        vertices.append(f"    ({x:.12g} {y:.12g} {Z1:.12g})")

    blocks = [
        f"    hex ({' '.join(map(str, bottom))} "
        f"{' '.join(str(v + 20) for v in bottom)}) "
        f"({counts[x_key]} {counts[y_key]} 1) simpleGrading (1 1 1)"
        for bottom, (x_key, y_key) in _BLOCKS.values()
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
