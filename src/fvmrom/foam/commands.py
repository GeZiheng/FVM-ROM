"""Build bash scripts that run one OpenFOAM application inside a case."""

from __future__ import annotations

import re
import shlex

_APP_RE = re.compile(r"^(\./)?[A-Za-z0-9_][A-Za-z0-9_.+-]*$")

# Executables must live under one of the OpenFOAM install/user/site bin directories. Empty
# variables are skipped explicitly: a pattern of "$EMPTY"/* would match every absolute path.
_APP_CHECK = r"""
APP={app}
case "$APP" in
  ./*) [ -f "$APP" ] || {{ echo "No such script in case: $APP" >&2; exit 127; }} ;;
  *)
    p=$(type -P "$APP")
    [ -n "$p" ] || {{ echo "Not found in the OpenFOAM environment: $APP" >&2; exit 127; }}
    ok=0
    for d in "$WM_PROJECT_DIR" "$FOAM_USER_APPBIN" "$FOAM_SITE_APPBIN"; do
      if [ -n "$d" ]; then case "$p" in "$d"/*) ok=1 ;; esac; fi
    done
    [ "$ok" = 1 ] || {{ echo "Refusing to run non-OpenFOAM executable: $p" >&2; exit 126; }}
    ;;
esac
"""


def validate_app(app: str) -> str:
    if not _APP_RE.match(app) or ".." in app:
        raise ValueError(
            f"Invalid application name {app!r}: use a bare OpenFOAM executable (e.g. 'blockMesh') "
            "or a script in the case directory (e.g. './Allrun')."
        )
    return app


def build_run_script(
    case_wsl: str,
    app: str,
    args: list[str],
    np: int,
    log_name: str,
    pid_file: str | None = None,
    exit_file: str | None = None,
) -> str:
    """Script body (after the environment preamble). Output goes to ``<case>/<log_name>``."""
    validate_app(app)
    if np < 1:
        raise ValueError("np must be >= 1")
    if "/" in log_name or "\\" in log_name or log_name.startswith("."):
        raise ValueError(f"Invalid log name: {log_name!r}")

    qargs = " ".join(shlex.quote(a) for a in args)
    if np > 1:
        cmd = f'mpirun -np {np} "$APP" {qargs} -parallel'
    else:
        cmd = f'"$APP" {qargs}'

    lines = []
    if exit_file:
        lines.append(f'trap "echo \\$? > {shlex.quote(exit_file)}" EXIT')
    if pid_file:
        lines.append(f"echo $$ > {shlex.quote(pid_file)}")
    lines.append(f"cd {shlex.quote(case_wsl)} || exit 1")
    lines.append(f"exec > {shlex.quote(log_name)} 2>&1")
    lines.append(_APP_CHECK.format(app=shlex.quote(app)))
    lines.append(cmd)
    return "\n".join(lines) + "\n"
