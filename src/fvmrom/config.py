"""Runtime configuration, overridable through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("FVMROM_ROOT", Path(__file__).resolve().parents[2]))


@dataclass(frozen=True)
class Settings:
    #: WSL distro name; None uses the default distro.
    distro: str | None = field(default_factory=lambda: os.environ.get("FVMROM_WSL_DISTRO") or None)
    #: OpenFOAM environment file, as a path inside WSL.
    foam_bashrc: str = field(
        default_factory=lambda: os.environ.get(
            "FVMROM_FOAM_BASHRC", "/usr/lib/openfoam/openfoam2312/etc/bashrc"
        )
    )
    #: Directory (Windows path) holding all OpenFOAM cases.
    cases_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("FVMROM_CASES_DIR", PROJECT_ROOT / "cases"))
    )
    #: Directory (Windows path) for snapshots, models and reports.
    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("FVMROM_DATA_DIR", PROJECT_ROOT / "data"))
    )
    #: Directory (Windows path) for MCP runtime state (jobs, generated scripts).
    state_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("FVMROM_STATE_DIR", PROJECT_ROOT / ".fvmrom"))
    )
    #: Hard cap on foreground command runtime (seconds).
    max_foreground_timeout: int = 600


def get_settings() -> Settings:
    return Settings()
