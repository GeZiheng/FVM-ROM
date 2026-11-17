"""Sample the parameter space, run one steady case per sample and collect snapshots.

Usage:
  uv run python scripts/generate_snapshots.py --problem cavity --study re1d --n-train 30 --n-test 10
"""

from __future__ import annotations

import argparse
import time

from fvmrom.core.config import get_settings
from fvmrom.foam import RunResult, collect_snapshots, run_sweep, validate_study
from fvmrom.case_templates import get_problem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem", required=True, choices=["cavity", "cylinder"])
    p.add_argument("--study", default="re1d")
    p.add_argument("--n-train", type=int, default=30)
    p.add_argument("--n-test", type=int, default=10)
    p.add_argument("--max-parallel", type=int, default=16)
    p.add_argument("--poll-s", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="skip cases that already converged")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    problem = get_problem(args.problem)
    study = validate_study(args.study, problem=problem.name, settings=settings)
    total = args.n_train + args.n_test
    params = problem.space.sample(total, seed=args.seed)
    roles = ["train"] * args.n_train + ["test"] * args.n_test
    out_dir = settings.data_dir / problem.name / study

    def progress(result: RunResult) -> None:
        values = " ".join(
            f"{name}={v:g}" for name, v in zip(problem.space.names, result.params)
        )
        status = "ok" if result.converged else "FAILED"
        print(f"[{result.index + 1:3d}/{total}] {result.case} {values} {status}: {result.message}",
              flush=True)

    print(
        f"{problem.name}/{study}: {args.n_train} train + {args.n_test} test samples, "
        f"parameters {problem.space.names} in [{problem.space.lower}, {problem.space.upper}]",
        flush=True,
    )
    started = time.time()
    results = run_sweep(
        problem,
        study,
        params,
        max_parallel=args.max_parallel,
        poll_s=args.poll_s,
        settings=settings,
        resume=args.resume,
        progress=progress,
    )
    path = collect_snapshots(problem, study, results, out_dir, roles, settings=settings)

    converged = sum(r.converged for r in results)
    failed = [r.case for r in results if not r.converged]
    print(
        f"done in {time.time() - started:.1f} s: {converged}/{total} converged, "
        f"{len(results) - converged} dropped"
    )
    if failed:
        print("failed cases:", ", ".join(failed))
    print("snapshots:", path)


if __name__ == "__main__":
    main()
