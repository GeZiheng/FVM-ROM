"""Background OpenFOAM jobs. State is kept in files so it survives MCP server restarts."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path

from ..config import Settings, get_settings
from .case import Cases
from .commands import build_run_script
from .logs import tail
from .wsl import Wsl, win_to_wsl


class Jobs:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.cases = Cases(self.settings)
        self.wsl = Wsl(self.settings)
        self.dir = self.settings.state_dir / "jobs"
        self._procs: dict[str, subprocess.Popen] = {}

    def _jdir(self, job_id: str) -> Path:
        if not job_id.replace("-", "").isalnum():
            raise ValueError(f"Invalid job id: {job_id!r}")
        d = self.dir / job_id
        if not d.is_dir():
            raise FileNotFoundError(f"Unknown job: {job_id}")
        return d

    def start(
        self, case: str, app: str, args: list[str], np: int = 1, log_name: str | None = None
    ) -> dict:
        case_dir = self.cases.path(case)
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        d = self.dir / job_id
        d.mkdir(parents=True)
        log_name = log_name or f"log.{app.removeprefix('./')}"
        pid_file, exit_file = d / "pid", d / "exit_code"
        body = build_run_script(
            win_to_wsl(case_dir), app, args, np, log_name, win_to_wsl(pid_file), win_to_wsl(exit_file)
        )
        meta = {
            "job_id": job_id,
            "case": case,
            "app": app,
            "args": args,
            "np": np,
            "log": log_name,
            "started": time.time(),
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        self._procs[job_id] = self.wsl.spawn(body, f"job-{job_id}.sh")
        return meta

    def status(self, job_id: str, tail_lines: int = 5) -> dict:
        d = self._jdir(job_id)
        meta = json.loads((d / "meta.json").read_text())
        exit_file, pid_file = d / "exit_code", d / "pid"
        if exit_file.is_file() and exit_file.read_text().strip():
            rc = int(exit_file.read_text().strip())
            state = "completed" if rc == 0 else "failed"
            end_time = exit_file.stat().st_mtime
        else:
            rc, end_time = None, time.time()
            proc = self._procs.get(job_id)
            if proc is not None and proc.poll() is None:
                state = "running"
            elif pid_file.is_file() and self._pid_alive(pid_file.read_text().strip()):
                state = "running"
            elif proc is None and not pid_file.is_file() and time.time() - meta["started"] < 15:
                state = "starting"
            else:
                state = "died"  # no exit code recorded: killed externally or WSL restarted
        log = self.cases.path(meta["case"]) / meta["log"]
        return {
            **meta,
            "state": state,
            "exit_code": rc,
            "elapsed_s": round(end_time - meta["started"], 1),
            "log_tail": tail(log, tail_lines),
        }

    def _pid_alive(self, pid: str) -> bool:
        if not pid.isdigit():
            return False
        r = self.wsl.run(f"kill -0 {pid} 2>/dev/null", timeout=15, with_env=False)
        return r.returncode == 0

    def stop(self, job_id: str) -> dict:
        d = self._jdir(job_id)
        pid_file = d / "pid"
        if not pid_file.is_file():
            raise RuntimeError("Job has no recorded pid yet (still starting?)")
        pid = pid_file.read_text().strip()
        if not pid.isdigit():
            raise RuntimeError(f"Bad pid file for job {job_id}")
        self.wsl.run(f"kill -TERM -- -{pid} 2>/dev/null || kill -TERM {pid}", 15, with_env=False)
        return {"job_id": job_id, "signal": "TERM", "pid": int(pid)}

    def list(self, limit: int = 20) -> list[dict]:
        if not self.dir.is_dir():
            return []
        ids = sorted((p.name for p in self.dir.iterdir() if p.is_dir()), reverse=True)[:limit]
        out = []
        for i in ids:
            s = self.status(i, tail_lines=0)
            out.append({k: s[k] for k in ("job_id", "case", "app", "np", "state", "elapsed_s")})
        return out
