# FVM-ROM – agent notes

Goal: reduced-order models (ROM) from OpenFOAM v2312 (ESI, apt-installed, WSL2 Ubuntu). The ROM
technique is **not decided yet** – discuss with the user before building `src/fvmrom/rom/`.

## Environment facts
- Host: Windows; Python is managed by `uv` (`uv sync --extra rom`, `uv run ...`). Do not rely on
  WSL's Python (no numpy, no venv).
- OpenFOAM: `/usr/lib/openfoam/openfoam2312` inside WSL (`source <that>/etc/bashrc`), precision
  DP, label 32, OpenMPI 4.1.2, 20 cores / 31 GB.
- Project is at `D:\Projects\FVM-ROM` = `/mnt/d/Projects/FVM-ROM` in WSL. Cases live in `cases/`.
- Shell on the host is PowerShell; avoid passing complex quoting through `wsl -e bash -c`. Use the
  MCP server / `fvmrom.foam.Wsl` (script-file based) instead.

## Conventions
- Run OpenFOAM through the `openfoam` MCP tools (or `fvmrom.foam`) – not ad-hoc `wsl` commands.
- Solver/app output goes to `<case>/log.<app>`.
- With `np>1` the case must already be decomposed and `np` must equal `numberOfSubdomains`.
- ROM data (snapshots, bases, models) goes to `data/`, never into `cases/`.
- `uv run pytest` runs unit tests plus real-WSL integration tests (`-m "not wsl"` skips the latter).
- `uv run python scripts/smoke_mcp.py` exercises the MCP server end-to-end (cavity).
