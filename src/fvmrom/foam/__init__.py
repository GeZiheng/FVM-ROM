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
from .mesh import MeshGeometry, phi_weights, read_geometry
from .sweep import RunResult, collect_snapshots, run_sweep, validate_study
from .transient import (
    TransientRun,
    collect_transient,
    inspect_run,
    read_forces,
    run_transient,
)
from .wsl import Wsl, win_to_wsl, wsl_to_win

__all__ = [
    "Cases",
    "FoamField",
    "Jobs",
    "MeshGeometry",
    "MeshTopology",
    "RunResult",
    "TransientRun",
    "Wsl",
    "collect_snapshots",
    "collect_transient",
    "inspect_run",
    "latest_time",
    "phi_divergence",
    "phi_vector",
    "phi_weights",
    "read_field",
    "read_forces",
    "read_geometry",
    "read_topology",
    "run_sweep",
    "run_transient",
    "validate_study",
    "win_to_wsl",
    "write_field",
    "wsl_to_win",
]
