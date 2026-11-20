"""Two-phase transient runs (spin-up plus production) and force-coefficient reading."""

from __future__ import annotations

import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np

from ..core.config import Settings, get_settings
from .case import Cases
from .fields import FoamField, latest_time, phi_vector, read_field, read_topology
from .jobs import Jobs
from .mesh import phi_weights, read_geometry
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


def read_forces(case_dir: Path, dt: float | None = None) -> dict[str, np.ndarray]:
    """Time, Cd and Cl from every ``postProcessing/forceCoeffs/*/coefficient.dat``.

    Restart directories are merged and sorted by time; when a time appears in more
    than one directory the later directory wins.  Columns are ``Time Cd Cd(f) Cd(r)
    Cl ...``.  With ``dt`` the time column, which only carries six significant digits,
    is rounded onto the steps ``k * dt`` before the merge (see :func:`_snap_times`).
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
    if dt is not None:
        time_array = _snap_times(time_array, dt)
    order = np.argsort(time_array, kind="stable")
    time_array = time_array[order]
    cd = np.asarray(cds, dtype=float)[order]
    cl = np.asarray(cls, dtype=float)[order]
    # keep the last row of duplicated times: that is the later restart directory
    keep = np.ones(time_array.size, dtype=bool)
    keep[:-1] = time_array[:-1] != time_array[1:]
    return {"time": time_array[keep], "cd": cd[keep], "cl": cl[keep]}


def _snap_times(times: np.ndarray, dt: float) -> np.ndarray:
    """Round times printed with six significant digits onto the steps ``k * dt``."""
    if not dt > 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    steps = np.rint(times / dt)
    error = np.abs(times - steps * dt)
    # at t >= 100 the printed resolution is 0.001, so a dt = 1/300 step may sit exactly
    # 0.1 dt off; the relative slack only absorbs the floating-point noise of that case
    tolerance = 0.1 * dt * (1.0 + 1e-9)
    bad = np.flatnonzero(error > tolerance)
    if bad.size:
        index = int(bad[0])
        raise ValueError(
            f"time {times[index]:g} is {error[index] / dt:.2f} dt away from a multiple of "
            f"dt={dt:g}; dt does not match the forceCoeffs output"
        )
    return steps * dt


# -- snapshot collection -------------------------------------------------------------------


def select_snapshot_times(
    names: Iterable[str], t_spin: float, snapshot_dt: float
) -> list[tuple[float, str]]:
    """Time directories from ``t_spin`` on that are multiples of ``snapshot_dt``.

    The directory at ``t_spin`` ends the spin-up phase and is the first snapshot.
    Times are rounded onto ``k * snapshot_dt``; the selection must be the contiguous
    range from the first to the last time, otherwise the missing times are named.
    """
    picked: dict[int, tuple[float, str]] = {}
    start = t_spin - 1e-9 * max(1.0, abs(t_spin))
    for name in names:
        value = _time_value(name)
        if value is None or value < start:
            continue
        k = int(round(value / snapshot_dt))
        if abs(value - k * snapshot_dt) > 1e-6 * max(1.0, abs(value)):
            raise ValueError(
                f"time directory {name!r} is not a multiple of snapshot_dt={snapshot_dt:g}"
            )
        if k in picked:
            raise ValueError(
                f"time directories {picked[k][1]!r} and {name!r} name the same time"
            )
        picked[k] = (k * snapshot_dt, name)
    if not picked:
        raise ValueError(f"no time directories at or after t_spin={t_spin:g}")
    keys = sorted(picked)
    missing = sorted(set(range(keys[0], keys[-1] + 1)) - set(picked))
    if missing:
        times = ", ".join(f"{k * snapshot_dt:g}" for k in missing)
        raise ValueError(
            f"missing snapshot times between {keys[0] * snapshot_dt:g} and "
            f"{keys[-1] * snapshot_dt:g}: {times}"
        )
    out = [picked[k] for k in keys]
    steps = np.diff([value for value, _ in out])
    if steps.size and np.max(np.abs(steps - snapshot_dt)) > 1e-6 * snapshot_dt:
        raise ValueError(f"snapshot times are not uniformly spaced by {snapshot_dt:g}")
    return out


def snapshot_times(case_dir: Path, config: "TransientConfig") -> list[tuple[float, str]]:
    """Production snapshot times (with their directory names) of a case."""
    root = Path(case_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Case not found: {root}")
    return select_snapshot_times(
        (p.name for p in root.iterdir() if p.is_dir()), config.t_spin, config.snapshot_dt
    )


def align_forces(
    times: np.ndarray,
    force_time: np.ndarray,
    cd: np.ndarray,
    cl: np.ndarray,
    tol: float,
) -> np.ndarray:
    """(n, 2) Cd/Cl at exactly the given times; no interpolation.

    The nearest recorded step must be within ``tol`` of the snapshot time.
    """
    times = np.asarray(times, dtype=float).reshape(-1)
    force_time = np.asarray(force_time, dtype=float).reshape(-1)
    cd = np.asarray(cd, dtype=float).reshape(-1)
    cl = np.asarray(cl, dtype=float).reshape(-1)
    if force_time.size == 0:
        raise ValueError("no forceCoeffs rows to take the snapshot forces from")
    if cd.size != force_time.size or cl.size != force_time.size:
        raise ValueError(
            f"force columns differ in length: {force_time.size} times, "
            f"{cd.size} Cd, {cl.size} Cl"
        )
    index = np.searchsorted(force_time, times)
    left = np.clip(index - 1, 0, force_time.size - 1)
    right = np.clip(index, 0, force_time.size - 1)
    pick = np.where(
        np.abs(force_time[left] - times) <= np.abs(force_time[right] - times), left, right
    )
    bad = np.flatnonzero(np.abs(force_time[pick] - times) > tol)
    if bad.size:
        first = int(bad[0])
        raise ValueError(
            f"no forceCoeffs row at snapshot time {times[first]:g} "
            f"(nearest {force_time[pick[first]]:g}, step {force_time[1] - force_time[0]:g})"
        )
    return np.column_stack([cd[pick], cl[pick]])


def _read_snapshot(job: tuple[str, str]) -> tuple[np.ndarray, np.ndarray, FoamField]:
    """Read U, p and phi of one time directory (top-level so that the pool pickles it)."""
    case_dir, time = job
    root = Path(case_dir)
    return (
        read_field(root, time, "U").internal,
        read_field(root, time, "p").internal.reshape(-1),
        read_field(root, time, "phi"),
    )


def collect_transient(
    problem,
    study: str,
    run: TransientRun,
    out_dir: Path,
    workers: int = 8,
    settings: Settings | None = None,
) -> Path:
    """Write ``snapshots.npz`` (U, p, phi, forces, volumes) of the production phase.

    The time directories are read in parallel by :class:`ProcessPoolExecutor`; the
    snapshot arrays are filled in place, so the batch is held in memory only once.
    Forces come from ``out_dir/forces.npz`` (written there by the script) or, if it
    is missing, from the case's ``forceCoeffs`` output.
    """
    settings = settings or get_settings()
    study = validate_study(study, problem=problem.name, settings=settings)
    cases = Cases(settings)
    config = problem.config
    case_dir = cases.path(run.case)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    picked = snapshot_times(case_dir, config)
    times = np.asarray([value for value, _ in picked], dtype=float)
    names = [name for _, name in picked]
    topo = read_topology(case_dir)
    volumes = read_field(case_dir, "0", "V").internal.reshape(-1)
    if volumes.size != topo.n_cells:
        raise ValueError(f"0/V has {volumes.size} values for {topo.n_cells} cells")
    geometry = read_geometry(case_dir)
    weights = phi_weights(geometry)

    n = len(names)
    U = np.empty((n, topo.n_cells, 3))
    p = np.empty((n, topo.n_cells))
    phi = np.empty((n, weights.size))
    jobs = [(str(case_dir), name) for name in names]
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for i, (u, pi, phi_field) in enumerate(pool.map(_read_snapshot, jobs)):
            if u.shape != (topo.n_cells, 3):
                raise ValueError(f"{names[i]}: U has shape {u.shape}")
            if pi.size != topo.n_cells:
                raise ValueError(
                    f"{names[i]}: p has {pi.size} values for {topo.n_cells} cells"
                )
            value = phi_vector(phi_field, topo)
            if value.size != weights.size:
                raise ValueError(
                    f"{names[i]}: phi has {value.size} values, the geometry gives "
                    f"{weights.size}"
                )
            U[i] = u
            p[i] = pi
            phi[i] = value

    path = out_dir / "snapshots.npz"
    np.savez_compressed(
        path,
        times=times,
        U=U,
        p=p,
        phi=phi,
        volumes=volumes,
        phi_weights=weights,
        forces=_snapshot_forces(out_dir, case_dir, config, times),
        re=config.re,
    )
    return path


def _snapshot_forces(
    out_dir: Path, case_dir: Path, config: "TransientConfig", times: np.ndarray
) -> np.ndarray:
    path = Path(out_dir) / "forces.npz"
    if path.is_file():
        with np.load(path) as data:
            force_time, cd, cl = data["time"], data["cd"], data["cl"]
    else:
        forces = read_forces(case_dir, dt=config.timestep)
        np.savez(path, **forces)
        force_time, cd, cl = forces["time"], forces["cd"], forces["cl"]
    return align_forces(times, force_time, cd, cl, tol=0.25 * config.timestep)


def _time_value(name: str) -> float | None:
    try:
        return float(name)
    except ValueError:
        return None


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
        if resume:
            _verify_case(cases, case, config)
        else:
            shutil.rmtree(case_dir)
    if not case_dir.is_dir():
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


def inspect_run(
    problem,
    study: str,
    settings: Settings | None = None,
    cases: Cases | None = None,
) -> TransientRun:
    """Check the log of an existing case without running or modifying it."""
    if cases is None:
        settings = settings or get_settings()
        cases = Cases(settings)
    settings = settings or getattr(cases, "settings", None) or get_settings()
    config = problem.config
    study = validate_study(study, problem=problem.name, settings=settings)
    case = f"{problem.name}_{study}/run_0000"
    case_dir = cases.path(case, must_exist=False)
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case not found: {case_dir}")
    converged, message, courant = _check_log(cases, case, len(config.phases))
    return TransientRun(case, config, converged, message, courant)


def _verify_case(cases: Cases, case: str, config: "TransientConfig") -> None:
    """Refuse to resume a case whose dictionaries were written from another config."""
    from ..case_templates.cylinder_transient import files  # local: avoids a circular import

    # files that the run itself rewrites: setFields touches 0/U, Allrun copies the controlDicts
    mutable = {"0/U", "system/controlDict"}
    root = cases.path(case)
    stale = []
    for rel, text in files(config).items():
        if rel in mutable:
            continue
        path = root / rel
        if not path.is_file():
            stale.append(f"{rel} (missing)")
        elif path.read_text(encoding="utf-8", errors="replace") != text:
            stale.append(rel)
    if stale:
        raise ValueError(
            f"cannot resume {case}: it does not match the current config ({', '.join(stale)}); "
            "rerun without --resume to rebuild the case, or check the config"
        )


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
    if not cases.path(case, must_exist=False).is_dir():
        return False
    converged, _, _ = _check_log(cases, case, len(config.phases))
    if not converged:
        return False
    try:
        latest = float(latest_time(cases.path(case)))
    except FileNotFoundError:
        return False
    return latest >= config.t_end * (1.0 - 1e-6)
