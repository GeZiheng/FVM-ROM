"""Compare the shedding statistics of pilot runs: ``--studies pilot_base,pilot_wide``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fvmrom.core.config import get_settings
from fvmrom.rom.evaluate_transient import shedding_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", required=True, help="comma separated study names")
    return parser.parse_args()


def load_study(root: Path, study: str) -> dict:
    """Read ``forces.npz`` and ``meta.json`` of one study and summarise the shedding."""
    data = Path(root) / study
    with np.load(data / "forces.npz") as npz:
        time = np.asarray(npz["time"], dtype=float)
        cl = np.asarray(npz["cl"], dtype=float)
        cd = np.asarray(npz["cd"], dtype=float) if "cd" in npz else None
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    summary = shedding_summary(time, cl, cd)
    mesh = meta.get("mesh") or {}
    return {
        "study": study,
        "cells": int(mesh.get("n_cells", meta.get("n_cells", 0))),
        "max_courant": meta.get("max_courant"),
        "converged": meta.get("converged"),
        "summary": summary,
    }


def format_rows(rows: list[dict]) -> str:
    header = ["study", "cells", "max Co", "St", "period", "mean Cd", "Cl amp", "limit-cycle"]
    body = []
    for row in rows:
        summary = row["summary"]
        limit = summary["limit_cycle_time"]
        courant = row["max_courant"]
        body.append(
            [
                row["study"],
                str(row["cells"]),
                "n/a" if courant is None else f"{courant:.2f}",
                f"{summary['strouhal']:.4f}",
                f"{summary['period']:.3f}",
                "n/a" if summary["cd_mean"] is None else f"{summary['cd_mean']:.4f}",
                f"{summary['cl_amplitude']:.4f}",
                "n/a" if limit is None else f"{limit:.1f}",
            ]
        )
    widths = [
        max(len(line[i]) for line in [header, *body]) + 2 for i in range(len(header))
    ]
    lines = ["".join(c.ljust(w) for c, w in zip(header, widths)).rstrip()]
    lines += ["".join(c.ljust(w) for c, w in zip(line, widths)).rstrip() for line in body]
    return "\n".join(lines)


def deviations(rows: list[dict], st: float = 0.02, cd: float = 0.03, cl: float = 0.05) -> str:
    """Deviations of the later studies from the first one (plan 3.3 thresholds)."""
    if not rows:
        return ""
    base = rows[0]["summary"]
    lines = []
    for row in rows[1:]:
        other = row["summary"]
        cols = []
        for label, value, reference, threshold in (
            ("St", other["strouhal"], base["strouhal"], st),
            ("Cd", other["cd_mean"], base["cd_mean"], cd),
            ("Cl", other["cl_amplitude"], base["cl_amplitude"], cl),
        ):
            if value is None or reference in (None, 0.0):
                cols.append(f"{label} n/a")
                continue
            relative = (value - reference) / reference
            verdict = "ok" if abs(relative) < threshold else "OVER"
            cols.append(f"{label} {100.0 * relative:+.2f}% ({verdict})")
        lines.append(f"  {row['study']}: " + ", ".join(cols))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    studies = [name.strip() for name in args.studies.split(",") if name.strip()]
    root = get_settings().data_dir / "cylinder_transient"
    rows = [load_study(root, study) for study in studies]
    print(format_rows(rows))
    if len(rows) > 1:
        print(f"\nrelative to {rows[0]['study']} (St < 2%, Cd < 3%, Cl < 5%)")
        print(deviations(rows))


if __name__ == "__main__":
    main()
