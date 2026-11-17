"""Parsing of OpenFOAM solver logs."""

from __future__ import annotations

import re
from pathlib import Path

_SOLVE_RE = re.compile(
    r"Solving for (\w+), Initial residual = ([\d.eE+-]+), "
    r"Final residual = ([\d.eE+-]+), No Iterations (\d+)"
)
_TIME_RE = re.compile(r"^Time = (\S+)", re.M)
_CONVERGED_RE = re.compile(r"solution converged in (\S+) iterations", re.I)
# The run banner says "trapFpe: Floating point exception trapping enabled"; only a message
# at the start of a line (printed by bash when the solver is killed) is a real FPE.
_FPE_RE = re.compile(r"^Floating point exception", re.M)


def tail(path: Path, n: int = 40, max_bytes: int = 64_000) -> str:
    if n <= 0 or not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(max(0, size - max_bytes))
        data = f.read()
    return "\n".join(data.decode("utf-8", errors="replace").splitlines()[-n:])


def parse_log(path: Path) -> dict:
    """Summarise residual history, last time step and termination state of a solver log."""
    if not path.is_file():
        raise FileNotFoundError(f"Log not found: {path.name}")
    text = path.read_text(errors="replace")

    fields: dict[str, dict] = {}
    for name, init, final, iters in _SOLVE_RE.findall(text):
        f = fields.setdefault(
            name, {"solves": 0, "first_initial": float(init), "history_tail": []}
        )
        f["solves"] += 1
        f["last_initial"] = float(init)
        f["last_final"] = float(final)
        f["last_iterations"] = int(iters)
        f["history_tail"] = (f["history_tail"] + [float(init)])[-20:]

    times = _TIME_RE.findall(text)
    conv = _CONVERGED_RE.search(text)
    return {
        "log": path.name,
        "last_time": times[-1] if times else None,
        "n_time_steps": len(times),
        "fields": fields,
        "converged": conv.group(0) if conv else None,
        "finished": bool(re.search(r"^End\s*$", text, re.M)),
        "fatal_error": "FOAM FATAL" in text,
        "floating_point_exception": bool(_FPE_RE.search(text)),
    }
