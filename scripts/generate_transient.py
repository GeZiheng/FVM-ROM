"""Run a transient cylinder case; pilot mode keeps only the force coefficients.

Usage:
  uv run python scripts/generate_transient.py --study pilot_base --t-spin 250 --pilot
  uv run python scripts/generate_transient.py --study re100 --t-spin 100 --t-end 200
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace

import numpy as np

from fvmrom.core.config import get_settings
from fvmrom.foam import Cases, latest_time, read_geometry, validate_study
from fvmrom.foam.transient import read_forces, run_transient
from fvmrom.case_templates.cylinder import CylinderGeometry
from fvmrom.case_templates.cylinder_transient import CylinderTransientProblem, TransientConfig
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
    args = parser.parse_args()
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
    run = run_transient(problem, study, resume=args.resume, cases=cases)
    print(
        f"{run.case}: converged={run.converged} ({run.message}), "
        f"max Courant {run.max_courant}"
    )
    if not run.converged:
        raise SystemExit(f"run did not converge: {run.message}")

    case_dir = cases.path(run.case)
    forces = read_forces(case_dir)
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
        "shedding": summary,
    }
    (data_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"forces: {forces['time'].size} rows -> {data_dir / 'forces.npz'}")
    if summary is not None:
        print(
            "shedding: "
            + ", ".join(
                f"{key} {value:.4g}" for key, value in summary.items() if value is not None
            )
        )
    if config.t_spin < config.t_end:
        print("field snapshots (snapshots.npz) are collected in M1")


if __name__ == "__main__":
    main()
