"""OpenFOAM (WSL2) access layer, shared by the MCP server and the ROM code."""

from .case import Cases
from .fields import (
    FoamField,
    MeshTopology,
    latest_time,
    phi_divergence,
    phi_vector,
    read_field,
    read_topology,
    write_field,
)
from .jobs import Jobs
from .sweep import RunResult, collect_snapshots, run_sweep, validate_study
from .wsl import Wsl, win_to_wsl, wsl_to_win

__all__ = [
    "Cases",
    "FoamField",
    "Jobs",
    "MeshTopology",
    "RunResult",
    "Wsl",
    "collect_snapshots",
    "latest_time",
    "phi_divergence",
    "phi_vector",
    "read_field",
    "read_topology",
    "run_sweep",
    "validate_study",
    "win_to_wsl",
    "write_field",
    "wsl_to_win",
]
