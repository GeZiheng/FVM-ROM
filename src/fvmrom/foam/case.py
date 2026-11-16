"""Case directory handling (Windows-side file access to the cases directory)."""

from __future__ import annotations

import re
from pathlib import Path

from ..config import Settings, get_settings
from .wsl import win_to_wsl

MAX_READ_BYTES = 200_000
_ENTRY_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s+([^;{}]+);", re.M)
_TIME_RE = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")


def safe_join(base: Path, rel: str) -> Path:
    """Resolve ``rel`` below ``base``; reject absolute paths and escapes."""
    if not rel or rel.strip() in {".", "./"}:
        return base.resolve()
    if Path(rel).is_absolute() or re.match(r"^[A-Za-z]:", rel):
        raise ValueError(f"Absolute paths are not allowed: {rel}")
    p = (base / rel).resolve()
    if not p.is_relative_to(base.resolve()):
        raise ValueError(f"Path escapes its base directory: {rel}")
    return p


class Cases:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.root = self.settings.cases_dir

    def path(self, case: str, must_exist: bool = True) -> Path:
        p = safe_join(self.root, case)
        if p == self.root.resolve():
            raise ValueError("A case name is required")
        if must_exist and not p.is_dir():
            raise FileNotFoundError(f"Case not found: {case}")
        return p

    def create(self, case: str) -> dict:
        p = self.path(case, must_exist=False)
        if p.exists():
            raise FileExistsError(f"Case already exists: {case}")
        for sub in ("system", "constant", "0"):
            (p / sub).mkdir(parents=True)
        return self.summary(case)

    def wsl_path(self, case: str) -> str:
        return win_to_wsl(self.path(case))

    def file(self, case: str, rel: str) -> Path:
        return safe_join(self.path(case), rel)

    # -- listing ---------------------------------------------------------------------------

    def list_cases(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            p.parent.parent.relative_to(self.root).as_posix()
            for p in self.root.glob("**/system/controlDict")
        )

    def list_files(self, case: str, subdir: str = ".", max_entries: int = 300) -> list[str]:
        d = self.file(case, subdir)
        if not d.is_dir():
            raise NotADirectoryError(f"Not a directory: {subdir}")
        out = []
        for p in sorted(d.iterdir()):
            out.append(p.name + ("/" if p.is_dir() else f"  ({p.stat().st_size} B)"))
            if len(out) >= max_entries:
                out.append("... (truncated)")
                break
        return out

    def time_dirs(self, case: str) -> list[str]:
        root = self.path(case)
        times = [p.name for p in root.iterdir() if p.is_dir() and _TIME_RE.match(p.name)]
        return sorted(times, key=float)

    # -- reading / writing -----------------------------------------------------------------

    def read(self, case: str, rel: str, max_bytes: int = MAX_READ_BYTES) -> dict:
        p = self.file(case, rel)
        if not p.is_file():
            raise FileNotFoundError(f"No such file in case {case}: {rel}")
        size = p.stat().st_size
        with p.open("rb") as f:
            data = f.read(max_bytes)
        return {
            "path": rel,
            "size": size,
            "truncated": size > max_bytes,
            "content": data.decode("utf-8", errors="replace"),
        }

    def write(self, case: str, rel: str, content: str, overwrite: bool = True) -> dict:
        p = self.file(case, rel)
        if p == self.path(case):
            raise ValueError("Refusing to write to the case root")
        if p.exists() and not overwrite:
            raise FileExistsError(f"File exists: {rel}")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        return {"path": rel, "bytes": p.stat().st_size}

    # -- summary ---------------------------------------------------------------------------

    def control_dict(self, case: str) -> dict[str, str]:
        p = self.path(case) / "system" / "controlDict"
        if not p.is_file():
            return {}
        text = re.sub(r"/\*.*?\*/|//[^\n]*", "", p.read_text(errors="replace"), flags=re.S)
        wanted = {
            "application", "startFrom", "startTime", "stopAt", "endTime", "deltaT",
            "writeControl", "writeInterval", "adjustTimeStep", "maxCo",
        }  # fmt: skip
        return {k: v.strip() for k, v in _ENTRY_RE.findall(text) if k in wanted}

    def summary(self, case: str) -> dict:
        root = self.path(case)
        poly = root / "constant" / "polyMesh"
        has_mesh = any((poly / n).exists() for n in ("points", "points.gz"))
        procs = sorted(p.name for p in root.glob("processor*") if p.is_dir())
        return {
            "case": case,
            "windows_path": str(root),
            "wsl_path": win_to_wsl(root),
            "has_system": (root / "system").is_dir(),
            "has_constant": (root / "constant").is_dir(),
            "has_mesh": has_mesh,
            "time_dirs": self.time_dirs(case),
            "decomposed_into": len(procs),
            "control_dict": self.control_dict(case),
            "logs": sorted(p.name for p in root.glob("log.*")),
            "system_files": sorted(p.name for p in (root / "system").glob("*") if p.is_file())
            if (root / "system").is_dir()
            else [],
        }
