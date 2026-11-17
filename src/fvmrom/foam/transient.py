"""Two-phase transient runs (spin-up plus production) and force-coefficient reading."""

from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..core.config import Settings, get_settings
from .case import Cases
from .fields import latest_time
from .jobs import Jobs
from .sweep import validate_study

if TYPE_CHECKING:  # avoids a circular import: fvmrom.case_templates imports fvmrom.foam
    from ..case_templates.cylinder_transient import TransientConfig

SOLVER_LOG = "log.pimpleFoam"
_COURANT_RE = re.compile(r"Courant Number mean:\s*([\d.eE+-]+)\s+max:\s*([\d.eE+-]+)")
_END_RE = re.compile(r"^End\s*$", re.M)
_FPE_RE = re.compile(r"^Floating point exception", re.M)
ACTIVE_STATES = ("starting", "running")


@dataclass
class TransientRun:
    case: str
    config: "TransientConfig"
    converged: bool
    message: str
    max_courant: float | None


def courant_max(text: str) -> float | None:
    """Largest ``Courant Number ... max:`` printed in a solver log."""
    values = [float(match.group(2)) for match in _COURANT_RE.finditer(text)]
    return max(values) if values else None


def read_forces(case_dir: Path) -> dict[str, np.ndarray]:
    """Time, Cd and Cl from every ``postProcessing/forceCoeffs/*/coefficient.dat``.

    Restart directories are merged and sorted by time; when a time appears in more
    than one directory the later directory wins.  Columns are ``Time Cd Cd(f) Cd(r)
    Cl ...``.
    """
    root = Path(case_dir) / "postProcessing" / "forceCoeffs"
    files = []
    if root.is_dir():
        for path in root.glob("*/coefficient.dat"):
            try:
                start = float(path.parent.name)
            except ValueError:
                continue
            files.append((start, path))
    if not files:
        raise FileNotFoundError(f"No forceCoeffs output under {root}")
    files.sort(key=lambda item: item[0])

    times: list[float] = []
    cds: list[float] = []
    cls: list[float] = []
    for _, path in files:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                time_value, cd, cl = float(parts[0]), float(parts[1]), float(parts[4])
            except ValueError:
                continue
            times.append(time_value)
            cds.append(cd)
            cls.append(cl)

    time_array = np.asarray(times, dtype=float)
    order = np.argsort(time_array, kind="stable")
    time_array = time_array[order]
    cd = np.asarray(cds, dtype=float)[order]
    cl = np.asarray(cls, dtype=float)[order]
    # keep the last row of duplicated times: that is the later restart directory
    keep = np.ones(time_array.size, dtype=bool)
    keep[:-1] = time_array[:-1] != time_array[1:]
    return {"time": time_array[keep], "cd": cd[keep], "cl": cl[keep]}


def run_transient(
    problem,
    study: str,
    settings: Settings | None = None,
    resume: bool = False,
    poll_s: float = 5.0,
    cases: Cases | None = None,
    jobs: Jobs | None = None,
) -> TransientRun:
    """Run ``./Allrun`` of one transient case and check the solver log."""
    if cases is None or jobs is None:
        settings = settings or get_settings()
        cases = cases or Cases(settings)
        jobs = jobs or Jobs(settings)
    settings = settings or getattr(cases, "settings", None) or get_settings()
    config = problem.config
    study = validate_study(study, problem=problem.name, settings=settings)
    case = f"{problem.name}_{study}/run_0000"
    case_dir = cases.path(case, must_exist=False)

    if resume and _finished(cases, case, config):
        converged, message, courant = _check_log(cases, case, len(config.phases))
        return TransientRun(case, config, converged, f"skipped: {message}", courant)

    if case_dir.is_dir():
        jobs.stop_case(case)
        shutil.rmtree(case_dir)
    problem.write_case(cases, case)

    job = jobs.start(case, "./Allrun", [])
    state = "starting"
    try:
        while True:
            state = jobs.status(job["job_id"], tail_lines=0)["state"]
            if state not in ACTIVE_STATES:
                break
            time.sleep(poll_s)
    finally:
        if state in ACTIVE_STATES:
            jobs.stop(job["job_id"])

    converged, message, courant = _check_log(cases, case, len(config.phases))
    if state != "completed":
        converged = False
        message = f"job {state}: {message}"
    return TransientRun(case, config, converged, message, courant)


def _check_log(cases: Cases, case: str, phases: int) -> tuple[bool, str, float | None]:
    path = cases.file(case, SOLVER_LOG)
    if not path.is_file():
        return False, "no log.pimpleFoam", None
    text = path.read_text(encoding="utf-8", errors="replace")
    courant = courant_max(text)
    if "FOAM FATAL" in text:
        return False, "FOAM FATAL error", courant
    if _FPE_RE.search(text):
        return False, "floating point exception", courant
    ended = len(_END_RE.findall(text))
    if ended != phases:
        return False, f"{ended} of {phases} phase(s) reached 'End'", courant
    return True, f"{ended} phase(s) completed", courant


def _finished(cases: Cases, case: str, config: "TransientConfig") -> bool:
    converged, _, _ = _check_log(cases, case, len(config.phases))
    if not converged:
        return False
    try:
        latest = float(latest_time(cases.path(case)))
    except FileNotFoundError:
        return False
    return latest >= config.t_end * (1.0 - 1e-6)
