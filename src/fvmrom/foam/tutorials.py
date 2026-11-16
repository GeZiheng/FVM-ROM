"""Access to the tutorials shipped with the OpenFOAM installation."""

from __future__ import annotations

import re
import shlex

from .case import Cases
from .wsl import Wsl, win_to_wsl

_REL_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./+-]*$")


def list_tutorials(wsl: Wsl, contains: str = "", limit: int = 100) -> list[str]:
    """Tutorial case paths relative to ``$FOAM_TUTORIALS`` (directories with system/controlDict)."""
    r = wsl.run(
        'cd "$FOAM_TUTORIALS" && find . -path "*/system/controlDict" -printf "%P\\n" | sort',
        timeout=60,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Could not list tutorials: {r.stderr or r.stdout}")
    cases = [line.removesuffix("/system/controlDict") for line in r.stdout.splitlines() if line]
    needle = contains.lower()
    return [c for c in cases if needle in c.lower()][:limit]


def copy_tutorial(wsl: Wsl, cases: Cases, tutorial: str, case_name: str) -> dict:
    if not _REL_RE.match(tutorial) or ".." in tutorial.split("/"):
        raise ValueError(f"Invalid tutorial path: {tutorial!r}")
    dest = cases.path(case_name, must_exist=False)
    if dest.exists():
        raise FileExistsError(f"Case already exists: {case_name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = wsl.run(
        f'src="$FOAM_TUTORIALS"/{shlex.quote(tutorial)}\n'
        f'[ -f "$src/system/controlDict" ] || {{ echo "Not a tutorial case: {tutorial}" >&2; exit 2; }}\n'
        f"cp -r --no-preserve=mode,ownership \"$src\" {shlex.quote(win_to_wsl(dest))}",
        timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Copy failed: {r.stderr or r.stdout}")
    return cases.summary(case_name)
