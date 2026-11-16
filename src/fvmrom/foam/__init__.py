"""OpenFOAM (WSL2) access layer, shared by the MCP server and the ROM code."""

from .case import Cases
from .jobs import Jobs
from .wsl import Wsl, win_to_wsl, wsl_to_win

__all__ = ["Cases", "Jobs", "Wsl", "win_to_wsl", "wsl_to_win"]
