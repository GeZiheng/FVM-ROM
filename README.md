# FVM-ROM

Reduced-order modelling (ROM) built on top of a pre-compiled **OpenFOAM v2312** (ESI) running in
**WSL2 Ubuntu**. The ROM technique is not decided yet; this repository currently provides the
project skeleton and an **OpenFOAM MCP server** so an agent can set up, run and inspect cases.

## Layout

```
src/fvmrom/
  config.py          settings (env-var overridable)
  foam/              Windows -> WSL2 OpenFOAM access layer (shared by MCP and ROM code)
    wsl.py           run bash scripts in WSL, path mapping D:\ <-> /mnt/d
    commands.py      safe script generation for one OpenFOAM application
    case.py          case directories: list / read / write / summary
    jobs.py          background jobs (file-based state)
    logs.py          solver log tail and residual parsing
    tutorials.py     list / copy tutorials from $FOAM_TUTORIALS
  mcp_server/        FastMCP stdio server (entry point: fvmrom-mcp)
  rom/               ROM code (placeholder)
cases/               OpenFOAM cases (visible to Windows and WSL)
data/                ROM data: snapshots, bases, models (git-ignored)
docs/                architecture and roadmap
tests/
opencode.json        registers the MCP server with OpenCode
```

## Setup

Requirements: Windows with [uv](https://docs.astral.sh/uv/), WSL2 with OpenFOAM v2312 installed
at `/usr/lib/openfoam/openfoam2312`.

```powershell
uv sync                 # MCP server + dev tools
uv sync --extra rom     # additionally numpy/scipy for ROM work
uv run pytest           # unit tests; add `-m wsl` style selection, see tests/
```

OpenCode picks up the server from `opencode.json` (name: `openfoam`). To run it by hand:
`uv run fvmrom-mcp`.

## Configuration (environment variables)

| Variable | Default |
| --- | --- |
| `FVMROM_WSL_DISTRO` | default WSL distro |
| `FVMROM_FOAM_BASHRC` | `/usr/lib/openfoam/openfoam2312/etc/bashrc` |
| `FVMROM_CASES_DIR` | `<project>/cases` |
| `FVMROM_STATE_DIR` | `<project>/.fvmrom` (jobs, generated scripts) |

## MCP tools

Environment: `foam_env_info`, `list_tutorials`
Cases: `copy_tutorial`, `list_cases`, `case_summary`, `list_case_files`, `read_case_file`,
`write_case_file`, `foam_dict_get`, `foam_dict_set`
Execution: `run_foam_app` (foreground, optional `np` for MPI), `start_foam_job`, `job_status`,
`list_jobs`, `stop_job`
Logs: `read_log_tail`, `parse_solver_log`
