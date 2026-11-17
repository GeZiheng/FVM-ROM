"""Batch runs of one case per parameter sample, and collection of the snapshots."""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from ..core.config import Settings, get_settings
from .case import Cases
from .fields import latest_time, phi_vector, read_field, read_topology
from .jobs import Jobs
from .logs import parse_log

if TYPE_CHECKING:  # avoids a circular import: fvmrom.case_templates imports fvmrom.foam
    from ..case_templates import Problem

SOLVER_LOG = "log.simpleFoam"
_STUDY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def validate_study(
    name: str, problem: str | None = None, settings: Settings | None = None
) -> str:
    """Reject study names that could escape the data/case directories or alias them."""
    if not _STUDY_RE.fullmatch(name) or ".." in name:
        raise ValueError(
            f"invalid study name {name!r}: use letters, digits, '_', '.', '-' "
            "(not starting with '.' or '-') and no '..'"
        )
    if name.endswith((".", " ")):
        raise ValueError(f"invalid study name {name!r}: must not end with '.' or a space")
    if name.split(".")[0].upper() in _RESERVED_NAMES:
        raise ValueError(
            f"invalid study name {name!r}: {name.split('.')[0]} is a reserved "
            "device name on Windows"
        )
    if problem is not None:
        _check_study_conflict(name, problem, settings or get_settings())
    return name


def _check_study_conflict(name: str, problem: str, settings: Settings) -> None:
    """Windows resolves names that differ only in letter case to the same directory."""
    for parent, leaf in (
        (settings.cases_dir, f"{problem}_{name}"),
        (settings.data_dir / problem, name),
    ):
        if not parent.is_dir():
            continue
        for entry in parent.iterdir():
            if entry.name != leaf and entry.name.lower() == leaf.lower():
                raise ValueError(
                    f"invalid study name {name!r}: {entry.name!r} already exists and "
                    "only the letter case differs"
                )


@dataclass
class RunResult:
    index: int
    case: str
    params: np.ndarray
    converged: bool
    job_id: str
    message: str


def run_sweep(
    problem: Problem,
    study: str,
    params: np.ndarray,
    max_parallel: int = 16,
    poll_s: float = 5.0,
    settings: Settings | None = None,
    resume: bool = False,
    progress: Callable[[RunResult], None] | None = None,
    cases: Cases | None = None,
    jobs: Jobs | None = None,
) -> list[RunResult]:
    """Run one case per row of ``params``, at most ``max_parallel`` at a time."""
    params = np.atleast_2d(np.asarray(params, dtype=float))
    if params.shape[1] != problem.space.dim:
        raise ValueError(
            f"{problem.name}: expected {problem.space.dim} parameters per sample, "
            f"got {params.shape[1]}"
        )
    if cases is None or jobs is None:
        settings = settings or get_settings()
        cases = cases or Cases(settings)
        jobs = jobs or Jobs(settings)
    settings = settings or getattr(cases, "settings", None) or get_settings()
    study = validate_study(study, problem=problem.name, settings=settings)
    root = f"{problem.name}_{study}"
    pending = list(range(params.shape[0]))
    running: dict[str, tuple[int, str]] = {}
    results: dict[int, RunResult] = {}

    def record(result: RunResult) -> None:
        results[result.index] = result
        if progress is not None:
            progress(result)

    try:
        while pending or running:
            while pending and len(running) < max(1, max_parallel):
                i = pending.pop(0)
                case = f"{root}/run_{i:04d}"
                if resume and _finished(cases, case):
                    record(
                        RunResult(i, case, params[i], True, "", "skipped: already converged")
                    )
                    continue
                case_dir = cases.path(case, must_exist=False)
                _check_case_path(cases, case, case_dir)
                if case_dir.is_dir():
                    jobs.stop_case(case)
                    shutil.rmtree(case_dir)
                problem.write_case(cases, case, params[i])
                meta = jobs.start(case, "./Allrun", [])
                running[meta["job_id"]] = (i, case)

            done = False
            for job_id, (i, case) in list(running.items()):
                state = jobs.status(job_id, tail_lines=0)["state"]
                if state not in ("completed", "failed", "died"):
                    continue
                converged, message = _check_solver_log(cases, case)
                if converged and not _finished(cases, case):
                    converged, message = False, "log is converged but V is missing"
                if state != "completed":
                    converged = False
                    message = f"job {state}: {message}"
                record(RunResult(i, case, params[i], converged, job_id, message))
                del running[job_id]
                done = True

            if running and not done:
                time.sleep(poll_s)
    finally:
        _stop_running(jobs, running)

    return [results[i] for i in sorted(results)]


def _check_case_path(cases: Cases, case: str, case_dir: Path) -> None:
    """Reject names such as ``cavity_re1d.`` that resolve onto another case directory."""
    expected = Path(case).parts
    actual = case_dir.relative_to(cases.root.resolve()).parts
    if actual != expected:
        raise ValueError(
            f"case path {case!r} resolves to {Path(*actual).as_posix()!r}: "
            "the name must match the directory exactly"
        )


def _stop_running(jobs: Jobs, running: dict[str, tuple[int, str]]) -> None:
    """Stop the jobs started by this sweep; raise if any of them cannot be stopped."""
    failures: dict[str, Exception] = {}
    for job_id in list(running):
        try:
            try:
                jobs.stop(job_id)
            except Exception:
                jobs.wait_pid(job_id)  # the wrapper script is still starting
                jobs.stop(job_id)
        except Exception as exc:
            failures[job_id] = exc
    if failures:
        detail = "; ".join(f"{job_id}: {err}" for job_id, err in failures.items())
        msg = f"failed to stop jobs {', '.join(failures)} ({detail})"
        raise RuntimeError(msg) from next(iter(failures.values()))


def collect_snapshots(
    problem: Problem,
    study: str,
    results: list[RunResult],
    out_dir: Path,
    roles: list[str],
    settings: Settings | None = None,
) -> Path:
    """Write ``snapshots.npz`` (converged samples only) and ``samples.json``."""
    if len(roles) != len(results):
        raise ValueError(f"got {len(roles)} roles for {len(results)} results")
    settings = settings or get_settings()
    study = validate_study(study, problem=problem.name, settings=settings)
    cases = Cases(settings)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kept = [(r, role) for r, role in zip(results, roles) if r.converged]
    if not kept:
        raise ValueError("no converged samples to collect")

    ref: tuple[object, np.ndarray] | None = None
    fields: dict[str, list[np.ndarray]] = {name: [] for name in problem.fields}
    kept_params, kept_roles = [], []
    for result, role in kept:
        case_dir = cases.path(result.case)
        topo = read_topology(case_dir)
        time = latest_time(case_dir)
        if ref is None:
            volumes = read_field(case_dir, time, "V").internal.reshape(-1)
            if volumes.size != topo.n_cells:
                raise ValueError(
                    f"{result.case}: V has {volumes.size} values for {topo.n_cells} cells"
                )
            ref = (topo, volumes)
        else:
            _check_topology(result.case, topo, ref[0])

        U = read_field(case_dir, time, "U").internal
        p = read_field(case_dir, time, "p").internal.reshape(-1)
        phi = phi_vector(read_field(case_dir, time, "phi"), topo)
        if U.shape != (topo.n_cells, 3):
            raise ValueError(f"{result.case}: U has shape {U.shape}")
        if p.size != topo.n_cells:
            raise ValueError(f"{result.case}: p has {p.size} values for {topo.n_cells} cells")
        fields["U"].append(U)
        fields["p"].append(p)
        fields["phi"].append(phi)
        kept_params.append(result.params)
        kept_roles.append(role)

    path = out_dir / "snapshots.npz"
    np.savez_compressed(
        path,
        params=np.asarray(kept_params, dtype=float),
        U=np.asarray(fields["U"]),
        p=np.asarray(fields["p"]),
        phi=np.asarray(fields["phi"]),
        volumes=np.asarray(ref[1], dtype=float),
        converged=np.ones(len(kept), dtype=bool),
        role=np.asarray(kept_roles),
    )
    meta = {
        "problem": problem.name,
        "study": study,
        "param_names": list(problem.space.names),
        "lower": list(problem.space.lower),
        "upper": list(problem.space.upper),
        "log": list(problem.space.log),
        "fields": list(problem.fields),
        "n_cells": int(ref[0].n_cells),
        # length of the phi vector, i.e. internal faces plus the non-empty patches
        "n_faces": int(fields["phi"][0].size),
        "n_samples": len(results),
        "n_converged": len(kept),
        "samples": [_sample_entry(problem, cases, r, role) for r, role in zip(results, roles)],
    }
    (out_dir / "samples.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def _sample_entry(problem: Problem, cases: Cases, result: RunResult, role: str) -> dict:
    entry = {
        "index": result.index,
        "case": result.case,
        "role": role,
        "params": [float(v) for v in np.ravel(result.params)],
        "converged": result.converged,
        "job_id": result.job_id,
        "message": result.message,
    }
    if problem.name == "cylinder":
        entry["cd"] = _read_cd(cases, result.case) if result.converged else None
    return entry


def _read_cd(cases: Cases, case: str) -> float | None:
    """Drag coefficient from the last row of forceCoeffs' coefficient.dat.

    The latest numeric time directory wins; non-numeric rows are skipped.
    """
    root = cases.path(case, must_exist=False) / "postProcessing" / "forceCoeffs"
    files = []
    for path in root.glob("*/coefficient.dat"):
        value = _time_value(path.parent.name)
        if value is not None:
            files.append((value, path))
    if not files:
        return None
    latest = max(files, key=lambda item: item[0])[1]
    try:
        text = latest.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            return float(parts[1])  # columns: Time Cd Cd(f) Cd(r) Cl ...
        except ValueError:
            continue
    return None


def _time_value(name: str) -> float | None:
    try:
        return float(name)
    except ValueError:
        return None


def _n_mesh_faces(topo) -> int:
    """All faces of the mesh, including the empty patches absent from the phi vector."""
    return int(topo.neighbour.size + sum(n for _, _, n in topo.patches.values()))


def _check_topology(case: str, topo, ref) -> None:
    if topo.n_cells != ref.n_cells or _n_mesh_faces(topo) != _n_mesh_faces(ref):
        raise ValueError(f"{case}: mesh differs from the first sample (cell/face count)")
    if not np.array_equal(topo.owner, ref.owner):
        raise ValueError(f"{case}: mesh differs from the first sample (owner array)")
    if not np.array_equal(topo.neighbour, ref.neighbour):
        raise ValueError(f"{case}: mesh differs from the first sample (neighbour array)")
    if topo.patches != ref.patches:
        raise ValueError(f"{case}: mesh differs from the first sample (patches)")


def _check_solver_log(cases: Cases, case: str) -> tuple[bool, str]:
    try:
        info = parse_log(cases.file(case, SOLVER_LOG))
    except FileNotFoundError:
        return False, "no log.simpleFoam"
    if info["fatal_error"]:
        return False, "FOAM FATAL error"
    if info["floating_point_exception"]:
        return False, "floating point exception"
    if not info["converged"]:
        return False, f"not converged (last time {info['last_time']})"
    return True, info["converged"]


def _finished(cases: Cases, case: str) -> bool:
    path = cases.path(case, must_exist=False)
    if not path.is_dir():
        return False
    converged, _ = _check_solver_log(cases, case)
    if not converged:
        return False
    try:
        time = latest_time(path)
    except FileNotFoundError:
        return False
    return (path / time / "V").is_file()
