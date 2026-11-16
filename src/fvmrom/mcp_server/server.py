"""OpenFOAM MCP server (stdio).

Runs on Windows and drives the OpenFOAM installation inside WSL2. Cases live in the project's
``cases/`` directory, which is visible from both sides (``D:\\...`` and ``/mnt/d/...``).
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ..config import get_settings
from ..foam import Cases, Jobs, Wsl
from ..foam import tutorials as tut
from ..foam.commands import build_run_script
from ..foam.logs import parse_log, tail

settings = get_settings()
wsl = Wsl(settings)
cases = Cases(settings)
jobs = Jobs(settings)

mcp = FastMCP(
    "openfoam",
    instructions=(
        "Tools for OpenFOAM v2312 running in WSL2. Typical flow: copy_tutorial (or create_case + "
        "write_case_file) -> run_foam_app blockMesh/checkMesh -> start_foam_job for the solver -> "
        "job_status / parse_solver_log -> case_summary. Case names are paths relative to the "
        "cases directory. Applications run with the case as working directory and write their "
        "output to <case>/log.<app>."
    ),
)


# -- environment ---------------------------------------------------------------------------


@mcp.tool()
def foam_env_info() -> dict:
    """Report OpenFOAM version/paths, MPI, CPU and memory of the WSL2 environment."""
    r = wsl.run(
        'echo "version=$WM_PROJECT_VERSION"\n'
        'echo "project_dir=$WM_PROJECT_DIR"\n'
        'echo "precision=$WM_PRECISION_OPTION label=$WM_LABEL_SIZE mpi=$WM_MPLIB"\n'
        'echo "appbin=$FOAM_APPBIN"\n'
        'echo "tutorials=$FOAM_TUTORIALS"\n'
        'echo "cpus=$(nproc)"\n'
        'echo "mem_gb=$(free -g | awk \'/Mem:/{print $2}\')"\n'
        'echo "mpirun=$(mpirun --version 2>&1 | head -1)"\n',
        timeout=60,
    )
    info: dict = {"ok": r.returncode == 0 and "version=v" in r.stdout}
    for line in r.stdout.splitlines():
        for part in line.split(" ") if line.startswith("precision=") else [line]:
            k, _, v = part.partition("=")
            info[k] = v
    info["cases_dir"] = str(cases.root)
    info["distro"] = settings.distro or "(default)"
    if r.stderr:
        info["stderr"] = r.stderr[-500:]
    return info


@mcp.tool()
def list_tutorials(contains: str = "", limit: int = 100) -> list[str]:
    """List OpenFOAM tutorial cases (paths relative to $FOAM_TUTORIALS), filtered by substring."""
    return tut.list_tutorials(wsl, contains, limit)


# -- cases ---------------------------------------------------------------------------------


@mcp.tool()
def copy_tutorial(tutorial: str, case_name: str) -> dict:
    """Copy a tutorial (e.g. 'incompressible/icoFoam/cavity/cavity') into cases/<case_name>."""
    return tut.copy_tutorial(wsl, cases, tutorial, case_name)


@mcp.tool()
def create_case(case: str) -> dict:
    """Create an empty case cases/<case> with system/, constant/ and 0/ directories. Then fill it
    with write_case_file (controlDict, fvSchemes, fvSolution, blockMeshDict, 0/U, ...). Fails if
    the case already exists."""
    return cases.create(case)


@mcp.tool()
def list_cases() -> list[str]:
    """List the cases in the cases directory."""
    return cases.list_cases()


@mcp.tool()
def case_summary(case: str) -> dict:
    """Mesh/time-directory/decomposition status, key controlDict entries and logs of a case."""
    return cases.summary(case)


@mcp.tool()
def list_case_files(case: str, subdir: str = ".") -> list[str]:
    """List files inside a case directory (or a subdirectory such as 'system' or '0')."""
    return cases.list_files(case, subdir)


@mcp.tool()
def read_case_file(case: str, path: str) -> dict:
    """Read a text file of a case, e.g. 'system/fvSolution' or '0/U' (max ~200 kB)."""
    return cases.read(case, path)


@mcp.tool()
def write_case_file(case: str, path: str, content: str, overwrite: bool = True) -> dict:
    """Create/overwrite a text file in a case (LF line endings). Use for dictionaries and fields."""
    return cases.write(case, path, content, overwrite)


@mcp.tool()
def foam_dict_get(case: str, file: str, entry: str | None = None) -> str:
    """Read an entry (or whole dictionary) using foamDictionary. Scopes are '/'-separated,
    e.g. entry='boundaryField/inlet/type'."""
    args = [file] if entry is None else ["-entry", entry, "-value", file]
    return _run(case, "foamDictionary", args, 1, 60)["output"]


@mcp.tool()
def foam_dict_set(case: str, file: str, entry: str, value: str) -> dict:
    """Set an entry using foamDictionary, e.g. file='system/controlDict', entry='endTime',
    value='2'."""
    return _run(case, "foamDictionary", ["-entry", entry, "-set", value, file], 1, 60)


# -- running -------------------------------------------------------------------------------


def _run(case: str, app: str, args: list[str], np: int, timeout: int, log_name: str | None = None):
    timeout = max(1, min(timeout, settings.max_foreground_timeout))
    log_name = log_name or f"log.{app.removeprefix('./')}"
    body = build_run_script(cases.wsl_path(case), app, args, np, log_name)
    r = wsl.run(body, timeout=timeout)
    log_path = cases.path(case) / log_name
    return {
        "returncode": r.returncode,
        "timed_out": r.timed_out,
        "log": log_name,
        "output": tail(log_path, 60) or r.stderr[-1500:],
    }


@mcp.tool()
def run_foam_app(
    case: str,
    app: str,
    args: list[str] | None = None,
    np: int = 1,
    timeout: int = 120,
    log_name: str | None = None,
) -> dict:
    """Run an OpenFOAM application in a case and wait (e.g. blockMesh, checkMesh, decomposePar,
    reconstructPar, postProcess, foamToVTK; or a case script like './Allrun'). With np>1 it is run
    as `mpirun -np N app ... -parallel`. Full output is in <case>/<log_name> (default log.<app>);
    the last 60 lines are returned. For anything that may exceed the timeout (max 600 s) use
    start_foam_job instead."""
    return _run(case, app, args or [], np, timeout, log_name)


@mcp.tool()
def start_foam_job(
    case: str, app: str, args: list[str] | None = None, np: int = 1, log_name: str | None = None
) -> dict:
    """Start an OpenFOAM application in the background (solvers, long runs). Returns a job_id;
    follow with job_status / parse_solver_log; stop with stop_job."""
    return jobs.start(case, app, args or [], np, log_name)


@mcp.tool()
def job_status(job_id: str, tail_lines: int = 5) -> dict:
    """State (running/completed/failed/died), exit code, elapsed time and log tail of a job."""
    return jobs.status(job_id, tail_lines)


@mcp.tool()
def list_jobs(limit: int = 20) -> list[dict]:
    """List recent background jobs, newest first."""
    return jobs.list(limit)


@mcp.tool()
def stop_job(job_id: str) -> dict:
    """Send SIGTERM to a running job (its whole process group, including MPI ranks)."""
    return jobs.stop(job_id)


# -- logs ----------------------------------------------------------------------------------


@mcp.tool()
def read_log_tail(case: str, log: str, lines: int = 50) -> str:
    """Last lines of a log file in the case (e.g. 'log.simpleFoam')."""
    return tail(cases.file(case, log), lines)


@mcp.tool()
def parse_solver_log(case: str, log: str) -> dict:
    """Summarise a solver log: last time, per-field residuals, convergence/finish/fatal flags."""
    return parse_log(cases.file(case, log))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
