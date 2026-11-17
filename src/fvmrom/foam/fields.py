"""Reading and writing ASCII OpenFOAM fields and mesh topology (Windows-side file access)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TIME_RE = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")
_COMMENTS = re.compile(r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*', re.S)
_TOKENS = re.compile(r'"[^"]*"|[{}();]|[^\s{}();"]+')
_NCELLS_RE = re.compile(r"nCells:(\d+)")

_BANNER = [
    "/*--------------------------------*- C++ -*----------------------------------*\\",
    "\\*---------------------------------------------------------------------------*/",
]
_RULE = "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"


@dataclass
class FoamField:
    name: str
    cls: str
    internal: np.ndarray
    boundary: dict[str, dict]
    dimensions: str = "[0 0 0 0 0 0 0]"


@dataclass
class MeshTopology:
    owner: np.ndarray
    neighbour: np.ndarray
    patches: dict[str, tuple[str, int, int]]
    n_cells: int


class _Value:
    """Tokens of one dictionary entry value (a list, kept verbatim)."""

    def __init__(self, tokens: list[str]):
        self.tokens = tokens


class _Dict:
    """Minimal parser for the OpenFOAM dictionary syntax."""

    def __init__(self, tokens: list[str], pos: int = 0):
        self.tokens = tokens
        self.pos = pos

    def parse(self) -> dict[str, object]:
        out: dict[str, object] = {}
        while self.pos < len(self.tokens):
            if self.tokens[self.pos] == "}":
                break
            key = self.tokens[self.pos]
            self.pos += 1
            if key.startswith("#"):
                raise ValueError(f"Unsupported OpenFOAM directive: {key}")
            if self.pos < len(self.tokens) and self.tokens[self.pos] == "{":
                self.pos += 1
                out[key] = self.parse()
                self._expect("}")
            else:
                start = self.pos
                while self.pos < len(self.tokens) and self.tokens[self.pos] not in (";", "}"):
                    self.pos += 1
                out[key] = _Value(self.tokens[start:self.pos])
                if self.pos < len(self.tokens) and self.tokens[self.pos] == ";":
                    self.pos += 1
        return out

    def expect(self, token: str) -> None:
        self._expect(token)

    def _expect(self, token: str) -> None:
        if self.pos >= len(self.tokens) or self.tokens[self.pos] != token:
            raise ValueError(f"Malformed OpenFOAM dictionary: expected {token!r}")
        self.pos += 1


# -- fields --------------------------------------------------------------------------------


def read_field(case_dir: Path, time: str, name: str) -> FoamField:
    path = Path(case_dir) / str(time) / name
    if not path.is_file():
        raise FileNotFoundError(f"Field not found: {path}")
    tokens = _tokenize(_read_text(path))
    header, pos = _foam_file(tokens, path)

    fmt = _word(header.get("format"))
    if fmt is not None and fmt != "ascii":
        raise ValueError(f"{path}: only ASCII fields are supported (format {fmt})")
    cls = _word(header.get("class")) or ""

    entries = _Dict(tokens, pos).parse()
    if "internalField" not in entries:
        raise ValueError(f"{path}: no internalField")
    raw_internal = entries["internalField"]
    uniform = isinstance(raw_internal, _Value) and raw_internal.tokens[:1] == ["uniform"]
    n_cells = _n_cells(case_dir, path) if uniform else None
    internal = _field_value(raw_internal, n_cells, f"{path}: internalField")

    boundary: dict[str, dict] = {}
    patches = entries.get("boundaryField")
    if not isinstance(patches, dict):
        raise ValueError(f"{path}: no boundaryField")
    for patch, patch_entry in patches.items():
        if not isinstance(patch_entry, dict):
            raise ValueError(f"{path}: malformed boundary patch {patch!r}")
        value = patch_entry.get("value")
        boundary[patch] = {
            "type": _word(patch_entry.get("type")) or "patch",
            "value": None if value is None else _field_value(value, None, f"{path}: {patch}"),
        }

    dims = _dimensions(entries.get("dimensions"))
    return FoamField(name=name, cls=cls, internal=internal, boundary=boundary, dimensions=dims)


def write_field(
    case_dir: Path, time: str, field: FoamField, template: FoamField | None = None
) -> Path:
    path = Path(case_dir) / str(time) / field.name
    cls = field.cls or (template.cls if template is not None else "")
    patches: dict[str, dict] = {}
    if template is not None:
        for name, entry in template.boundary.items():
            patches[name] = {"type": entry.get("type") or "patch", "value": None}
    for name, entry in field.boundary.items():
        target = patches.setdefault(name, {"type": "patch", "value": None})
        target["type"] = entry.get("type") or target["type"]
        if entry.get("value") is not None:
            target["value"] = entry["value"]

    lines = [
        *_BANNER,
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        f"    class       {cls};",
        f'    location    "{time}";',
        f"    object      {field.name};",
        "}",
        _RULE,
        "",
        f"dimensions      {field.dimensions};",
        "",
        *_field_entry("internalField", np.asarray(field.internal, dtype=float), uniform=False),
        "",
        "boundaryField",
        "{",
    ]
    for name, entry in patches.items():
        lines.append(f"    {name}")
        lines.append("    {")
        lines.append(f"        type            {entry['type']};")
        value = entry["value"]
        if value is not None:
            arr = np.asarray(value, dtype=float)
            lines.extend(_field_entry("value", arr, uniform=arr.shape[0] == 1, indent="        "))
        lines.append("    }")
    lines.append("}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def latest_time(case_dir: Path) -> str:
    root = Path(case_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Case not found: {root}")
    times = [p.name for p in root.iterdir() if p.is_dir() and _TIME_RE.match(p.name)]
    if not times:
        raise FileNotFoundError(f"No time directories in {root}")
    return max(times, key=float)


# -- topology ------------------------------------------------------------------------------


def read_topology(case_dir: Path) -> MeshTopology:
    poly = Path(case_dir) / "constant" / "polyMesh"
    owner = _read_int_list(poly / "owner")
    neighbour = _read_int_list(poly / "neighbour")
    patches = _read_patches(poly / "boundary")
    if neighbour.size > owner.size:
        raise ValueError(f"{poly}: more internal faces than faces")
    notes = _notes(poly / "owner")
    n_cells = notes.get("nCells")
    if n_cells is None:
        n_cells = int(owner.max()) + 1 if owner.size else 0
    elif owner.size and int(owner.max()) >= n_cells:
        raise ValueError(f"{poly}/owner: face owners exceed the {n_cells} cells in the header note")
    for name, (_, start, n) in patches.items():
        if start < neighbour.size:
            raise ValueError(f"{poly}/boundary: patch {name!r} overlaps the internal faces")
        if start + n > owner.size:
            raise ValueError(f"{poly}/boundary: patch {name!r} extends past the last face")
    return MeshTopology(
        owner=owner, neighbour=neighbour, patches=patches, n_cells=n_cells
    )


def phi_vector(field: FoamField, topo: MeshTopology) -> np.ndarray:
    """Internal faces followed by the non-empty patches, in boundary-file order."""
    internal = np.asarray(field.internal, dtype=float).reshape(-1)
    if internal.size != topo.neighbour.size:
        raise ValueError(
            f"{field.name}: {internal.size} internal faces, topology has {topo.neighbour.size}"
        )
    parts = [internal]
    for name, (patch_type, _, n) in topo.patches.items():
        if patch_type == "empty":
            continue
        entry = field.boundary.get(name)
        value = None if entry is None else entry.get("value")
        if value is None:
            raise ValueError(
                f"{field.name}: patch {name!r} (type {patch_type}) has no phi value"
            )
        value = np.asarray(value, dtype=float).reshape(-1)
        if value.size == 1:
            value = np.full(n, value[0])
        elif value.size != n:
            raise ValueError(
                f"{field.name}: patch {name!r} has {value.size} values for {n} faces"
            )
        parts.append(value)
    return np.concatenate(parts)


def phi_divergence(phi: np.ndarray, topo: MeshTopology) -> np.ndarray:
    """Net outward flux per cell; phi is laid out as in :func:`phi_vector`."""
    phi = np.asarray(phi, dtype=float).reshape(-1)
    n_internal = topo.neighbour.size
    if phi.size < n_internal:
        raise ValueError(f"phi has {phi.size} values, topology has {n_internal} internal faces")
    div = np.zeros(topo.n_cells)
    if n_internal:
        np.add.at(div, topo.owner[:n_internal], phi[:n_internal])
        np.add.at(div, topo.neighbour, -phi[:n_internal])
    pos = n_internal
    # Patches follow the internal faces in boundary-file order; empty patches carry no
    # flux and are absent from the vector.
    for name, (patch_type, start, n) in topo.patches.items():
        if patch_type == "empty":
            continue
        if pos + n > phi.size:
            raise ValueError(f"phi does not match the face count of patch {name!r}")
        div[topo.owner[start:start + n]] += phi[pos:pos + n]
        pos += n
    if pos != phi.size:
        raise ValueError(f"phi has {phi.size - pos} trailing values not covered by patches")
    return div


# -- low-level parsing ---------------------------------------------------------------------


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _tokenize(text: str) -> list[str]:
    text = _COMMENTS.sub(lambda m: m.group(0) if m.group(0).startswith('"') else " ", text)
    return _TOKENS.findall(text)


def _foam_file(tokens: list[str], path: Path) -> tuple[dict[str, object], int]:
    if "FoamFile" not in tokens:
        raise ValueError(f"{path}: no FoamFile header")
    i = tokens.index("FoamFile")
    if i + 1 >= len(tokens) or tokens[i + 1] != "{":
        raise ValueError(f"{path}: malformed FoamFile header")
    parser = _Dict(tokens, i + 2)
    header = parser.parse()
    parser.expect("}")
    return header, parser.pos


def _word(value: object) -> str | None:
    if isinstance(value, _Value) and value.tokens:
        return value.tokens[0]
    return None


def _int_word(value: object, where: str) -> int:
    word = _word(value)
    if word is None:
        raise ValueError(f"{where}: missing value")
    try:
        return int(word)
    except ValueError as exc:
        raise ValueError(f"{where}: expected an integer, got {word!r}") from exc


def _dimensions(value: object) -> str:
    dims = " ".join(value.tokens) if isinstance(value, _Value) else ""
    return dims or "[0 0 0 0 0 0 0]"


def _field_value(value: object, n: int | None, where: str) -> np.ndarray:
    if not isinstance(value, _Value) or not value.tokens:
        raise ValueError(f"{where}: empty field value")
    kind, rest = value.tokens[0], value.tokens[1:]
    if kind == "uniform":
        vec = "(" in rest
        vals = _parse_numbers(rest, vec)
        if n is not None:
            if vec:
                return np.repeat(vals, n, axis=0)
            return np.zeros(n) if vals.size == 0 else np.full(n, vals[0])
        return vals
    if kind == "nonuniform":
        vals, _ = _nonuniform(rest, where)
        if n is not None and vals.shape[0] != n:
            raise ValueError(f"{where}: expected {n} values, found {vals.shape[0]}")
        return vals
    raise ValueError(f"{where}: unsupported field value starting with {kind!r}")


def _nonuniform(tokens: list[str], where: str) -> tuple[np.ndarray, bool]:
    if "(" not in tokens:
        raise ValueError(f"{where}: malformed nonuniform list")
    i = tokens.index("(")
    tail = tokens[i + 1:]
    close = len(tail) - 1 - tail[::-1].index(")") if ")" in tail else len(tail)
    inner = tail[:close]
    vec = "(" in inner
    return _parse_numbers(inner, vec), vec


def _parse_numbers(tokens: list[str], vec: bool) -> np.ndarray:
    if not vec:
        return np.array([float(t) for t in tokens if t not in ("(", ")")], dtype=float)
    rows: list[list[float]] = []
    current: list[float] | None = None
    for token in tokens:
        if token == "(":
            current = []
        elif token == ")":
            if current is None or len(current) != 3:
                raise ValueError("Malformed vector list")
            rows.append(current)
            current = None
        elif current is None:
            raise ValueError("Malformed vector list")
        else:
            current.append(float(token))
    return np.array(rows, dtype=float).reshape(-1, 3)


def _n_cells(case_dir: Path, field_path: Path) -> int:
    poly = Path(case_dir) / "constant" / "polyMesh"
    notes = _notes(poly / "owner")
    if notes.get("nCells") is None:
        _check_compressed(poly / "owner")
        raise ValueError(
            f"{field_path}: a uniform internalField needs constant/polyMesh/owner with a "
            "'nCells' note"
        )
    return notes["nCells"]


def _notes(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    head = _read_text(path)[:4096]
    notes: dict[str, int] = {}
    for key in ("nPoints", "nCells", "nFaces", "nInternalFaces"):
        m = re.search(rf"{key}:(\d+)", head)
        if m:
            notes[key] = int(m.group(1))
    return notes


def _check_compressed(path: Path) -> None:
    if path.with_name(path.name + ".gz").is_file():
        raise ValueError(f"{path}: compressed mesh files are not supported")


def _read_int_list(path: Path) -> np.ndarray:
    _check_compressed(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {path}")
    tokens = _tokenize(_read_text(path))
    if "(" not in tokens:
        raise ValueError(f"{path}: no face list")
    i = tokens.index("(")
    tail = tokens[i + 1:]
    close = len(tail) - 1 - tail[::-1].index(")") if ")" in tail else len(tail)
    try:
        return np.array([int(t) for t in tail[:close]], dtype=np.int64)
    except ValueError as exc:
        raise ValueError(f"{path}: malformed face list ({exc})") from exc


def _read_patches(path: Path) -> dict[str, tuple[str, int, int]]:
    _check_compressed(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {path}")
    tokens = _tokenize(_read_text(path))
    if "(" not in tokens:
        raise ValueError(f"{path}: no patch list")
    i = tokens.index("(")
    try:
        count = int(tokens[i - 1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{path}: malformed patch count") from exc
    patches: dict[str, tuple[str, int, int]] = {}
    pos = i + 1
    for _ in range(count):
        if pos >= len(tokens):
            raise ValueError(f"{path}: truncated patch list")
        name = tokens[pos]
        pos += 1
        if pos >= len(tokens) or tokens[pos] != "{":
            raise ValueError(f"{path}: malformed patch {name!r}")
        parser = _Dict(tokens, pos + 1)
        entry = parser.parse()
        parser.expect("}")
        pos = parser.pos
        patches[name] = (
            _word(entry.get("type")) or "patch",
            _int_word(entry.get("startFace"), f"{path}: patch {name!r} startFace"),
            _int_word(entry.get("nFaces"), f"{path}: patch {name!r} nFaces"),
        )
    if pos >= len(tokens) or tokens[pos] != ")":
        raise ValueError(f"{path}: malformed patch list")
    return patches


def _fmt(value: float) -> str:
    return f"{value:.12g}"


def _fmt_value(value: np.ndarray) -> str:
    if value.ndim == 0:
        return _fmt(float(value))
    return "(" + " ".join(_fmt(v) for v in np.ravel(value)) + ")"


def _field_entry(keyword: str, arr: np.ndarray, uniform: bool, indent: str = "") -> list[str]:
    if arr.size == 0:
        return [f"{indent}{keyword}   nonuniform 0();"]
    if uniform:
        return [f"{indent}{keyword}   uniform {_fmt_value(arr[0])};"]
    kind = "vector" if arr.ndim == 2 else "scalar"
    rows = [_fmt_value(row) for row in arr] if arr.ndim == 2 else [_fmt(v) for v in arr]
    lines = [f"{indent}{keyword}   nonuniform List<{kind}>", f"{indent}{len(rows)}", f"{indent}("]
    lines.extend(f"{indent}{row}" for row in rows)
    lines.extend([f"{indent})", f"{indent};"])
    return lines
