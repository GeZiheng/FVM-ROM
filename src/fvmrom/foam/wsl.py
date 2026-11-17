"""Thin bridge from Windows to the OpenFOAM installation inside WSL2.

Commands are never passed to ``wsl.exe`` through the command line (Windows/PowerShell/bash quoting
makes that fragile). Instead each invocation is written to a small LF-terminated script under the
state directory, which bash then executes by its ``/mnt/<drive>/...`` path.
"""

from __future__ import annotations

import re
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from ..core.config import Settings, get_settings

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def win_to_wsl(path: str | Path) -> str:
    """``D:\\a\\b`` -> ``/mnt/d/a/b``."""
    p = PureWindowsPath(Path(path).resolve())
    drive = p.drive
    if not re.fullmatch(r"[A-Za-z]:", drive):
        raise ValueError(f"Only drive-letter paths can be mapped into WSL: {path}")
    return "/mnt/" + drive[0].lower() + "/" + "/".join(p.parts[1:])


def wsl_to_win(path: str) -> Path:
    """``/mnt/d/a/b`` -> ``D:\\a\\b``."""
    m = re.fullmatch(r"/mnt/([a-zA-Z])(?:/(.*))?", path)
    if not m:
        raise ValueError(f"Not a /mnt/<drive> path: {path}")
    return Path(f"{m.group(1).upper()}:/{m.group(2) or ''}")


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class Wsl:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    # -- script plumbing -------------------------------------------------------------------

    def preamble(self) -> str:
        return f'source "{self.settings.foam_bashrc}" >/dev/null 2>&1\n'

    def _write_script(self, body: str, with_env: bool = True, name: str | None = None) -> Path:
        d = self.settings.state_dir / "scripts"
        d.mkdir(parents=True, exist_ok=True)
        path = d / (name or f"{uuid.uuid4().hex}.sh")
        text = "#!/bin/bash\n" + (self.preamble() if with_env else "") + body
        path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))
        return path

    def _argv(self, *inner: str) -> list[str]:
        argv = ["wsl.exe"]
        if self.settings.distro:
            argv += ["-d", self.settings.distro]
        return argv + ["--exec", *inner]

    # -- public API ------------------------------------------------------------------------

    def run(self, body: str, timeout: int = 60, with_env: bool = True) -> RunResult:
        """Run a bash snippet (OpenFOAM environment sourced) and wait for it."""
        script = self._write_script(body, with_env)
        try:
            argv = self._argv("timeout", "-k", "5", str(timeout), "bash", win_to_wsl(script))
            try:
                cp = subprocess.run(
                    argv,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=timeout + 15,
                    creationflags=_CREATE_NO_WINDOW,
                )
            except subprocess.TimeoutExpired as e:
                return RunResult(-1, _dec(e.stdout), _dec(e.stderr), timed_out=True)
            return RunResult(
                cp.returncode, _dec(cp.stdout), _dec(cp.stderr), timed_out=cp.returncode == 124
            )
        finally:
            script.unlink(missing_ok=True)

    def spawn(self, body: str, script_name: str) -> subprocess.Popen:
        """Start a long-running bash script in its own session; the caller keeps the handle.

        The ``wsl.exe`` process is kept alive for the job's whole duration so that WSL never sees
        the distro as idle. Job state lives in files, so losing this handle is harmless.
        """
        script = self._write_script(body, True, script_name)
        argv = self._argv("setsid", "-w", "bash", win_to_wsl(script))
        return subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_CREATE_NO_WINDOW,
        )


def _dec(b: bytes | None) -> str:
    return (b or b"").decode("utf-8", errors="replace").replace("\r\n", "\n")
