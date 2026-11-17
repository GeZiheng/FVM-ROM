"""Mesh geometry from ``constant/polyMesh``: volumes, areas, centres and weights."""

import numpy as np
import pytest

from fvmrom.core.config import get_settings
from fvmrom.foam import latest_time, phi_vector, read_field
from fvmrom.foam.mesh import phi_weights, read_geometry

_HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {name};
}}
"""


def structured_mesh(root, xs, ys, z1):
    """Write a 2D-extruded hexahedral mesh with the given grid lines."""
    nx, ny = len(xs) - 1, len(ys) - 1
    points = [
        (xs[i], ys[j], z) for i in range(nx + 1) for j in range(ny + 1) for z in (0.0, z1)
    ]

    def p(i, j, k):
        return (i * (ny + 1) + j) * 2 + k

    def cell(i, j):
        return [p(i, j, 0), p(i + 1, j, 0), p(i + 1, j + 1, 0), p(i, j + 1, 0),
                p(i, j, 1), p(i + 1, j, 1), p(i + 1, j + 1, 1), p(i, j + 1, 1)]  # fmt: skip

    def index(i, j):
        return j * nx + i

    faces: list[list[int]] = []
    owner: list[int] = []
    neighbour: list[int] = []
    for j in range(ny):
        for i in range(nx - 1):
            c = cell(i, j)
            faces.append([c[1], c[2], c[6], c[5]])
            owner.append(index(i, j))
            neighbour.append(index(i + 1, j))
    for j in range(ny - 1):
        for i in range(nx):
            c = cell(i, j)
            faces.append([c[3], c[7], c[6], c[2]])
            owner.append(index(i, j))
            neighbour.append(index(i, j + 1))

    patches = []
    for name, kind, sides in (
        ("left", "patch", [("xmin", 0)]),
        ("right", "patch", [("xmax", nx - 1)]),
        ("bottom", "patch", [("ymin", 0)]),
        ("top", "patch", [("ymax", ny - 1)]),
        ("frontAndBack", "empty", [("zmin", 0), ("zmax", 0)]),
    ):
        patch_faces = []
        for side, fixed in sides:
            count = nx if side in ("ymin", "ymax") else ny
            if side in ("zmin", "zmax"):
                count = nx * ny
            for k in range(count):
                i, j = (k, fixed) if side in ("ymin", "ymax") else (fixed, k)
                if side in ("zmin", "zmax"):
                    i, j = k % nx, k // nx
                c = cell(i, j)
                verts = {
                    "xmin": [c[0], c[4], c[7], c[3]],
                    "xmax": [c[1], c[2], c[6], c[5]],
                    "ymin": [c[0], c[1], c[5], c[4]],
                    "ymax": [c[3], c[7], c[6], c[2]],
                    "zmin": [c[0], c[3], c[2], c[1]],
                    "zmax": [c[4], c[5], c[6], c[7]],
                }[side]
                patch_faces.append((verts, index(i, j)))
        patches.append((name, kind, patch_faces))

    boundary = []
    for name, kind, patch_faces in patches:
        first = len(owner)
        for verts, own in patch_faces:
            faces.append(verts)
            owner.append(own)
        boundary.append((name, kind, len(patch_faces), first))

    poly = root / "constant" / "polyMesh"
    poly.mkdir(parents=True, exist_ok=True)
    (poly / "points").write_text(
        _HEADER.format(cls="vectorField", name="points")
        + f"\n{len(points)}\n(\n"
        + "\n".join(f"({x:.17g} {y:.17g} {z:.17g})" for x, y, z in points)
        + "\n)\n"
    )
    (poly / "faces").write_text(
        _HEADER.format(cls="faceList", name="faces")
        + f"\n{len(faces)}\n(\n"
        + "\n".join(f"{len(f)}({' '.join(str(v) for v in f)})" for f in faces)
        + "\n)\n"
    )
    (poly / "owner").write_text(
        _HEADER.format(cls="labelList", name="owner")
        + f"\n{len(owner)}\n(\n"
        + "\n".join(str(v) for v in owner)
        + "\n)\n"
    )
    (poly / "neighbour").write_text(
        _HEADER.format(cls="labelList", name="neighbour")
        + f"\n{len(neighbour)}\n(\n"
        + "\n".join(str(v) for v in neighbour)
        + "\n)\n"
    )
    text = _HEADER.format(cls="polyBoundaryMesh", name="boundary") + f"\n{len(boundary)}\n(\n"
    for name, kind, n, first in boundary:
        text += f"    {name}\n    {{\n        type {kind};\n"
        text += f"        nFaces {n};\n        startFace {first};\n    }}\n"
    (poly / "boundary").write_text(text + ")\n")
    return root


@pytest.fixture
def nonuniform(tmp_path):
    return structured_mesh(tmp_path / "nonuniform", [0.0, 0.5, 1.5], [0.0, 1.0], 0.1)


@pytest.fixture
def uniform(tmp_path):
    return structured_mesh(tmp_path / "uniform", [0.0, 0.5, 1.0], [0.0, 0.5], 0.1)


def test_volumes_and_centres(nonuniform):
    geom = read_geometry(nonuniform)
    assert geom.topo.n_cells == 2
    assert geom.volumes == pytest.approx([0.05, 0.1])
    assert geom.cell_centres == pytest.approx(
        np.array([[0.25, 0.5, 0.05], [1.0, 0.5, 0.05]])
    )


def test_face_areas_and_closure(nonuniform):
    geom = read_geometry(nonuniform)
    areas, centres = geom.face_areas, geom.face_centres
    # face 0 is the internal face, 1-7 the patches (left, right, bottom x2, top x2, empty x4)
    assert areas[0] == pytest.approx([0.1, 0.0, 0.0])
    assert centres[0] == pytest.approx([0.5, 0.5, 0.05])
    assert areas[1] == pytest.approx([-0.1, 0.0, 0.0])  # left, outward
    assert areas[2] == pytest.approx([0.1, 0.0, 0.0])  # right, outward

    topo = geom.topo
    n_internal = topo.neighbour.size
    net = np.zeros((topo.n_cells, 3))
    np.add.at(net, topo.owner[:n_internal], areas[:n_internal])
    np.add.at(net, topo.neighbour, -areas[:n_internal])
    pos = n_internal
    for patch_type, start, n in topo.patches.values():
        if patch_type == "empty":
            continue
        np.add.at(net, topo.owner[start : start + n], areas[start : start + n])
        pos += n
    magnitude = np.linalg.norm(areas, axis=1).sum()
    assert np.abs(net).max() / magnitude < 1e-14


def test_linear_weights(nonuniform, uniform):
    geom = read_geometry(nonuniform)
    # P at x=0.25, N at x=1.0, face at x=0.5: w = |0.5 - 0.25| / 0.75 = 2/3
    assert geom.lin_weights == pytest.approx([2.0 / 3.0])
    assert read_geometry(uniform).lin_weights == pytest.approx([0.5])


def test_phi_weights_uniform_is_constant(uniform):
    geom = read_geometry(uniform)
    weights = phi_weights(geom)
    assert weights.size == 1 + 1 + 1 + 2 + 2  # internal, left, right, bottom, top
    assert weights == pytest.approx([1.0 / 0.1] * weights.size)


def test_phi_weights_layout(nonuniform):
    geom = read_geometry(nonuniform)
    weights = phi_weights(geom)
    assert weights.size == 7
    assert weights == pytest.approx(
        [
            (0.05 + 0.1) / (2 * 0.1**2),  # internal
            0.05 / 0.1**2,  # left
            0.1 / 0.1**2,  # right
            0.05 / 0.05**2,  # bottom, cell 0
            0.1 / 0.1**2,  # bottom, cell 1
            0.05 / 0.05**2,  # top, cell 0
            0.1 / 0.1**2,  # top, cell 1
        ]
    )


def test_read_geometry_errors(nonuniform, tmp_path):
    poly = nonuniform / "constant" / "polyMesh"
    with pytest.raises(FileNotFoundError):
        read_geometry(tmp_path / "missing")
    (poly / "points").write_text(
        (poly / "points").read_text().replace("format      ascii", "format      binary")
    )
    with pytest.raises(ValueError, match="ASCII"):
        read_geometry(nonuniform)
    (poly / "points.gz").write_bytes(b"")
    with pytest.raises(ValueError, match="compressed"):
        read_geometry(nonuniform)


def _real_case(name):
    case = get_settings().cases_dir / f"{name}_re1d" / "run_0000"
    if not case.is_dir():
        pytest.skip(f"stage-1 case {case} is not available")
    return case


@pytest.mark.parametrize("name", ["cylinder", "cavity"])
def test_volumes_match_openfoam(name):
    case = _real_case(name)
    geom = read_geometry(case)
    volumes = read_field(case, latest_time(case), "V").internal.reshape(-1)
    assert volumes.size == geom.volumes.size
    error = np.max(np.abs(geom.volumes - volumes) / np.abs(volumes))
    assert error <= 1e-10, f"{name}: max relative volume error {error:.3e}"

    topo = geom.topo
    phi = phi_vector(read_field(case, latest_time(case), "phi"), topo)
    weights = phi_weights(geom)
    assert weights.shape == phi.shape
    assert np.all(np.isfinite(weights)) and np.all(weights > 0.0)

    # the outward area vectors of every closed cell sum to zero
    areas = geom.face_areas
    net = np.zeros((topo.n_cells, 3))
    n_internal = topo.neighbour.size
    np.add.at(net, topo.owner[:n_internal], areas[:n_internal])
    np.add.at(net, topo.neighbour, -areas[:n_internal])
    for patch_type, start, n in topo.patches.values():
        if patch_type != "empty":
            np.add.at(net, topo.owner[start : start + n], areas[start : start + n])
    assert np.abs(net).max() / np.linalg.norm(areas, axis=1).sum() < 1e-13
