"""Mesh geometry: face centres and areas, cell centres and volumes, ``linear`` weights.

The formulas follow OpenFOAM's ``primitiveMeshTools`` (central triangle fan for faces,
pyramid decomposition around the mean face centre for cells) so that volumes agree with
the ``V`` field to round-off.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fields import MeshTopology, _check_compressed, _read_text, _tokenize, read_topology

_ASCII_RE = re.compile(r"\bformat\s+binary\b")
_ROOTS_MALL = 1e-15


@dataclass
class MeshGeometry:
    topo: MeshTopology
    cell_centres: np.ndarray  # (n_cells, 3)
    volumes: np.ndarray  # (n_cells,)
    face_centres: np.ndarray  # (n_faces, 3), all faces including the empty patches
    face_areas: np.ndarray  # (n_faces, 3), area vectors pointing out of the owner
    lin_weights: np.ndarray  # (n_internal,), f = w*owner + (1 - w)*neighbour


def read_geometry(case_dir: Path) -> MeshGeometry:
    """Read ``constant/polyMesh`` and compute the geometry of its cells and faces."""
    poly = Path(case_dir) / "constant" / "polyMesh"
    topo = read_topology(case_dir)
    points = _read_points(poly / "points")
    faces = _read_faces(poly / "faces")
    if len(faces) != topo.owner.size:
        raise ValueError(
            f"{poly}/faces: {len(faces)} faces, but owner lists {topo.owner.size}"
        )
    centres, areas = _face_geometry(points, faces)
    cell_centres, volumes = _cell_geometry(topo, centres, areas)
    weights = _linear_weights(topo, centres, areas, cell_centres)
    return MeshGeometry(
        topo=topo,
        cell_centres=cell_centres,
        volumes=volumes,
        face_centres=centres,
        face_areas=areas,
        lin_weights=weights,
    )


def phi_weights(geom: MeshGeometry) -> np.ndarray:
    """Weights of the ``phi`` vector for an approximate cell-volume L2 norm.

    Internal faces get ``(V_P + V_N) / (2 |Sf|^2)`` and the non-empty patch faces
    ``V_P / |Sf|^2``; the layout is that of :func:`fvmrom.foam.phi_vector`.
    """
    topo = geom.topo
    n_internal = topo.neighbour.size
    areas2 = np.einsum("ij,ij->i", geom.face_areas, geom.face_areas)
    if np.any(areas2 <= 0.0):
        raise ValueError("zero-area faces carry no flux weight")
    if n_internal:
        internal = (geom.volumes[topo.owner[:n_internal]] + geom.volumes[topo.neighbour]) / (
            2.0 * areas2[:n_internal]
        )
    else:
        internal = np.zeros(0)
    parts = [internal]
    for name, (patch_type, start, n) in topo.patches.items():
        if patch_type == "empty":
            continue
        owner = topo.owner[start : start + n]
        parts.append(geom.volumes[owner] / areas2[start : start + n])
    return np.concatenate(parts)


# -- mesh file parsing ---------------------------------------------------------------------


def _tokens(path: Path, what: str) -> list[str]:
    _check_compressed(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {path}")
    text = _read_text(path)
    if _ASCII_RE.search(text[:4096]):
        raise ValueError(f"{path}: only ASCII {what} are supported (format binary)")
    return _tokenize(text)


def _list_start(tokens: list[str], path: Path, what: str) -> int:
    try:
        i = tokens.index("(")
    except ValueError:
        raise ValueError(f"{path}: no {what} list") from None
    return i


def _read_points(path: Path) -> np.ndarray:
    tokens = _tokens(path, "points")
    start = _list_start(tokens, path, "point")
    try:
        n = int(tokens[start - 1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{path}: malformed point count") from exc
    values = tokens[start + 1 :]
    if len(values) < 5 * n:
        raise ValueError(f"{path}: truncated point list (expected {n} points)")
    points = np.empty((n, 3))
    pos = 0
    for i in range(n):
        if values[pos] != "(" or values[pos + 4] != ")":
            raise ValueError(f"{path}: malformed point {i}")
        try:
            points[i] = [float(values[pos + 1]), float(values[pos + 2]), float(values[pos + 3])]
        except ValueError as exc:
            raise ValueError(f"{path}: malformed point {i} ({exc})") from exc
        pos += 5
    return points


def _read_faces(path: Path) -> list[np.ndarray]:
    tokens = _tokens(path, "faces")
    start = _list_start(tokens, path, "face")
    try:
        n = int(tokens[start - 1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{path}: malformed face count") from exc
    values = tokens[start + 1 :]
    faces: list[np.ndarray] = []
    pos = 0
    for i in range(n):
        try:
            size = int(values[pos])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"{path}: malformed face {i}") from exc
        pos += 1
        if pos >= len(values) or values[pos] != "(":
            raise ValueError(f"{path}: malformed face {i}")
        pos += 1
        if pos + size >= len(values) or values[pos + size] != ")":
            raise ValueError(f"{path}: truncated face {i}")
        try:
            faces.append(np.array(values[pos : pos + size], dtype=np.int64))
        except ValueError as exc:
            raise ValueError(f"{path}: malformed face {i} ({exc})") from exc
        pos += size + 1
    return faces


# -- geometry ------------------------------------------------------------------------------


def _face_geometry(points: np.ndarray, faces: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """OpenFOAM face centres and area vectors (triangle fan around the vertex mean)."""
    n = len(faces)
    if n == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    counts = np.fromiter((face.size for face in faces), dtype=np.int64, count=n)
    if counts.min() < 3:
        raise ValueError("faces with fewer than three points are not supported")
    n_points = points.shape[0]
    for face in faces:
        if face.min() < 0 or face.max() >= n_points:
            raise ValueError("face references a point outside the points list")
    max_n = int(counts.max())
    idx = np.zeros((n, max_n), dtype=np.int64)
    nxt = np.zeros((n, max_n), dtype=np.int64)
    for i, face in enumerate(faces):
        idx[i, : face.size] = face
        nxt[i, : face.size] = np.roll(face, -1)

    this = points[idx]
    next_ = points[nxt]
    valid = np.arange(max_n)[None, :] < counts[:, None]
    verts = np.where(valid[..., None], this, 0.0)
    vertex_mean = verts.sum(axis=1) / counts[:, None]

    normal = np.cross(next_ - this, vertex_mean[:, None, :] - this)
    area = np.linalg.norm(normal, axis=2)
    area = np.where(valid, area, 0.0)
    normal = np.where(valid[..., None], normal, 0.0)
    sum_area = area.sum(axis=1)
    sum_normal = normal.sum(axis=1)
    tri_centre_sum = this + next_ + vertex_mean[:, None, :]
    centres = (area[..., None] * tri_centre_sum).sum(axis=1) / np.maximum(
        3.0 * sum_area, _ROOTS_MALL
    )[:, None]
    areas = 0.5 * sum_normal

    # OpenFOAM special-cases triangles to avoid round-off; tiny faces fall back to the
    # vertex mean with a zero area vector.
    tri = counts == 3
    if np.any(tri):
        tri_points = this[tri][:, :3]
        centres[tri] = tri_points.mean(axis=1)
        areas[tri] = 0.5 * np.cross(
            tri_points[:, 1] - tri_points[:, 0], tri_points[:, 2] - tri_points[:, 0]
        )
    tiny = sum_area < _ROOTS_MALL
    if np.any(tiny):
        centres[tiny] = vertex_mean[tiny]
        areas[tiny] = 0.0
    return centres, areas


def _cell_geometry(
    topo: MeshTopology, face_centres: np.ndarray, face_areas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """OpenFOAM cell centres and volumes (pyramids around the mean face centre)."""
    n_cells = topo.n_cells
    if n_cells <= 0:
        raise ValueError(f"the mesh has no cells (n_cells = {n_cells})")
    own = np.asarray(topo.owner, dtype=np.int64)
    nei = np.asarray(topo.neighbour, dtype=np.int64)
    if own.size and own.max() >= n_cells:
        raise ValueError("face owners exceed the cell count")
    if nei.size and nei.max() >= n_cells:
        raise ValueError("face neighbours exceed the cell count")
    n_faces = np.bincount(own, minlength=n_cells)
    if nei.size:
        n_faces += np.bincount(nei, minlength=n_cells)
    if np.any(n_faces == 0):
        raise ValueError("some cells have no faces")

    estimate = np.empty((n_cells, 3))
    for d in range(3):
        estimate[:, d] = np.bincount(
            own, weights=face_centres[:, d], minlength=n_cells
        )
        if nei.size:
            estimate[:, d] += np.bincount(
                nei, weights=face_centres[: nei.size, d], minlength=n_cells
            )
    estimate /= n_faces[:, None]

    acc_centre = np.zeros((n_cells, 3))
    pyramid = np.einsum("ij,ij->i", face_areas, face_centres - estimate[own])
    centre = 0.75 * face_centres + 0.25 * estimate[own]
    for d in range(3):
        acc_centre[:, d] = np.bincount(own, weights=pyramid * centre[:, d], minlength=n_cells)
    acc_volume = np.bincount(own, weights=pyramid, minlength=n_cells)

    if nei.size:
        fc = face_centres[: nei.size]
        pyramid = np.einsum("ij,ij->i", face_areas[: nei.size], estimate[nei] - fc)
        centre = 0.75 * fc + 0.25 * estimate[nei]
        for d in range(3):
            acc_centre[:, d] += np.bincount(
                nei, weights=pyramid * centre[:, d], minlength=n_cells
            )
        acc_volume += np.bincount(nei, weights=pyramid, minlength=n_cells)

    good = np.abs(acc_volume) > _ROOTS_MALL
    centres = estimate.copy()
    centres[good] = acc_centre[good] / acc_volume[good, None]
    return centres, acc_volume / 3.0


def _linear_weights(
    topo: MeshTopology,
    face_centres: np.ndarray,
    face_areas: np.ndarray,
    cell_centres: np.ndarray,
) -> np.ndarray:
    """OpenFOAM ``linear`` weights: ``f = w*P + (1 - w)*N`` on the internal faces."""
    n = topo.neighbour.size
    if n == 0:
        return np.zeros(0)
    own = np.asarray(topo.owner[:n], dtype=np.int64)
    nei = np.asarray(topo.neighbour, dtype=np.int64)
    sf = face_areas[:n]
    d_own = np.abs(np.einsum("ij,ij->i", sf, face_centres[:n] - cell_centres[own]))
    d_nei = np.abs(np.einsum("ij,ij->i", sf, cell_centres[nei] - face_centres[:n]))
    total = d_own + d_nei
    weights = np.full(n, 0.5)
    good = total > _ROOTS_MALL
    weights[good] = d_nei[good] / total[good]
    return weights
