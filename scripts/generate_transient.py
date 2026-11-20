"""Run a transient cylinder case; pilot mode keeps only the force coefficients.

Usage:
  uv run python scripts/generate_transient.py --study pilot_base --t-spin 250 --pilot
  uv run python scripts/generate_transient.py --study re100 --t-spin 60 --t-end 160
  uv run python scripts/generate_transient.py --study re100 --t-spin 60 --t-end 160 \\
      --collect-only --purge-fields
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from fvmrom.core.config import get_settings
from fvmrom.foam import (
    Cases,
    collect_transient,
    inspect_run,
    latest_time,
    phi_divergence,
    phi_weights,
    read_geometry,
    validate_study,
)
from fvmrom.foam.fields import _TIME_RE as TIME_DIR_RE
from fvmrom.foam.transient import read_forces, run_transient
from fvmrom.case_templates.cylinder import CylinderGeometry
from fvmrom.case_templates.cylinder_transient import CylinderTransientProblem, TransientConfig
from fvmrom.rom.evaluate import flux_consistency
from fvmrom.rom.evaluate_transient import shedding_summary

GEOMETRIES = {
    "base": CylinderGeometry(),
    "wide": CylinderGeometry(x_in=-15.0, x_out=40.0, y_far=20.0),
    "fine": CylinderGeometry(res=1.5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--geometry", choices=sorted(GEOMETRIES), default="base")
    parser.add_argument("--snapshot-dt", type=float, default=0.2)
    parser.add_argument("--checkpoint-dt", type=float, default=10.0)
    parser.add_argument("--t-spin", type=float, required=True)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--pilot", action="store_true", help="stop at t_spin (forces only)")
    parser.add_argument("--perturb", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--x-in", type=float, default=None)
    parser.add_argument("--x-out", type=float, default=None)
    parser.add_argument("--y-far", type=float, default=None)
    parser.add_argument("--res", type=float, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip a finished case; continue an unfinished one from its latest time",
    )
    parser.add_argument("--workers", type=int, default=8, help="snapshot reader processes")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="do not run OpenFOAM: collect the snapshots of an existing case",
    )
    parser.add_argument(
        "--purge-fields",
        action="store_true",
        help="after a successful collection delete every time directory except 0, "
        "t_spin and the last one; the spin-up checkpoints and the intermediate "
        "snapshots go too, so --collect-only cannot recollect them afterwards",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.pilot:
        if args.t_end is not None and args.t_end != args.t_spin:
            parser.error("--pilot needs --t-end to be equal to --t-spin")
        args.t_end = args.t_spin
    elif args.t_end is None:
        parser.error("--t-end is required unless --pilot is given")
    return args


def geometry_from_args(args: argparse.Namespace) -> CylinderGeometry:
    overrides = {}
    for name in ("x_in", "x_out", "y_far", "res"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    return replace(GEOMETRIES[args.geometry], **overrides)


def snapshot_stats(path: Path, geometry, chunk: int = 25) -> dict:
    """Flux consistency (unit and area weights) and the worst phi divergence.

    The snapshots are processed in chunks so that the interpolation of a whole
    batch never sits in memory at once.
    """
    weights = phi_weights(geometry)
    with np.load(path) as data:
        U, phi = data["U"], data["phi"]
        n = int(U.shape[0])
        unit = _chunked_flux_consistency(U, phi, geometry, None, chunk)
        area = _chunked_flux_consistency(U, phi, geometry, weights, chunk)
        divergence = max(
            float(np.abs(phi_divergence(phi[i], geometry.topo)).max()) for i in range(n)
        )
    return {
        "n_snapshots": n,
        "flux_consistency": {"unit": _stats(unit), "area": _stats(area)},
        "phi_div_max": divergence,
    }


def _chunked_flux_consistency(U, phi, geometry, weights, chunk: int) -> np.ndarray:
    parts = [
        flux_consistency(U[i : i + chunk], phi[i : i + chunk], geometry, weights)
        for i in range(0, U.shape[0], chunk)
    ]
    return np.concatenate(parts)


def _stats(values: np.ndarray) -> dict:
    return {"mean": float(np.mean(values)), "max": float(np.max(values))}


def purge_fields(case_dir: Path, config: TransientConfig) -> list[str]:
    """Delete every time directory except ``0``, ``t_spin`` and the last one.

    This includes the spin-up checkpoints (``checkpoint_dt``) and the intermediate
    snapshots of the production phase: a later ``--collect-only`` cannot recollect
    them, so run the collection once before purging.  Directories whose names are
    not OpenFOAM time names (``constant``, ``system``, ``postProcessing``, ...,
    or look-alikes such as ``nan``/``inf``/``1_0``) are left alone.
    """
    root = Path(case_dir)
    entries = [
        (float(path.name), path)
        for path in root.iterdir()
        if path.is_dir() and TIME_DIR_RE.match(path.name)
    ]
    if not entries:
        return []
    last = max(value for value, _ in entries)
    tol = 1e-9 * max(1.0, abs(last))
    removed = []
    for value, path in sorted(entries):
        if abs(value) <= tol or abs(value - config.t_spin) <= tol or value >= last - tol:
            continue
        shutil.rmtree(path)
        removed.append(path.name)
    return removed


def main() -> None:
    args = parse_args()
    settings = get_settings()
    study = validate_study(args.study, problem="cylinder_transient", settings=settings)
    config = TransientConfig(
        re=args.re,
        geometry=geometry_from_args(args),
        snapshot_dt=args.snapshot_dt,
        checkpoint_dt=args.checkpoint_dt,
        t_spin=args.t_spin,
        t_end=args.t_end,
        perturb=args.perturb,
        dt=args.dt,
    )
    problem = CylinderTransientProblem(config)
    cases = Cases(settings)
    if args.collect_only:
        try:
            run = inspect_run(problem, study, settings=settings, cases=cases)
        except FileNotFoundError as exc:
            raise SystemExit(f"--collect-only needs an existing case: {exc}") from exc
    else:
        run = run_transient(problem, study, resume=args.resume, cases=cases)
    print(
        f"{run.case}: converged={run.converged} ({run.message}), "
        f"max Courant {run.max_courant}"
    )
    if not run.converged:
        raise SystemExit(f"run did not converge: {run.message}")

    case_dir = cases.path(run.case)
    forces = read_forces(case_dir, dt=config.timestep)
    try:
        summary = shedding_summary(forces["time"], forces["cl"], forces["cd"])
    except ValueError as exc:
        print(f"WARNING: shedding summary skipped ({exc})")
        summary = None
    geometry = read_geometry(case_dir)
    data_dir = settings.data_dir / "cylinder_transient" / study
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez(data_dir / "forces.npz", **forces)
    meta = {
        "problem": problem.name,
        "study": study,
        "config": asdict(config),
        "dt": config.timestep,
        "phases": list(config.phases),
        "mesh": {
            "n_cells": int(geometry.topo.n_cells),
            "n_faces": int(geometry.face_areas.shape[0]),
        },
        "max_courant": run.max_courant,
        "converged": run.converged,
        "message": run.message,
        "mode": "pilot" if config.t_spin == config.t_end else "production",
        "latest_time": latest_time(case_dir),
        "n_snapshots": None,
        "snapshots": None,
        "shedding": summary,
        "flux_consistency": None,
        "phi_div_max": None,
    }
    print(f"forces: {forces['time'].size} rows -> {data_dir / 'forces.npz'}")
    if summary is not None:
        print(
            "shedding: "
            + ", ".join(
                f"{key} {value:.4g}" for key, value in summary.items() if value is not None
            )
        )
    if config.t_spin < config.t_end:
        path = collect_transient(
            problem, study, run, data_dir, workers=args.workers, settings=settings
        )
        stats = snapshot_stats(path, geometry)
        meta["n_snapshots"] = stats["n_snapshots"]
        meta["snapshots"] = path.name
        meta["flux_consistency"] = stats["flux_consistency"]
        meta["phi_div_max"] = stats["phi_div_max"]
        print(
            f"snapshots: {stats['n_snapshots']} times -> {path} "
            f"({path.stat().st_size / 1e6:.1f} MB)"
        )
        for name, values in stats["flux_consistency"].items():
            print(
                f"flux consistency ({name}): mean {values['mean']:.3g}, "
                f"max {values['max']:.3g}"
            )
        print(f"phi divergence: max {stats['phi_div_max']:.3g}")
        if args.purge_fields:
            removed = purge_fields(case_dir, config)
            print(f"purged fields: {len(removed)} time directories")
    (data_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
