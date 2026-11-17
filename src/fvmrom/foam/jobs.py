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

ACTIVE_STATES = ("running", "starting")
PID_WAIT_S = 20.0
STOP_POLL_S = 0.5


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

    def wait_pid(self, job_id: str, timeout: float = PID_WAIT_S) -> str:
        """Block until the wrapper script has written its pid (required before ``stop``)."""
        pid_file = self._jdir(job_id) / "pid"
        deadline = time.time() + timeout
        while not pid_file.is_file():
            if time.time() >= deadline:
                raise RuntimeError(f"Job {job_id} wrote no pid file within {timeout:.0f} s")
            time.sleep(STOP_POLL_S)
        pid = pid_file.read_text().strip()
        if not pid.isdigit():
            raise RuntimeError(f"Bad pid file for job {job_id}")
        return pid

    def stop_case(self, case: str, timeout: float = 30.0) -> list[str]:
        """Stop every job of ``case`` that is still running and wait until it is gone."""
        targets = [
            job_id
            for job_id in self._case_job_ids(case)
            if self.status(job_id, tail_lines=0)["state"] in ACTIVE_STATES
        ]
        if not targets:
            return []
        for job_id in targets:
            try:
                self.stop(job_id)
            except RuntimeError:
                self.wait_pid(job_id)  # still starting: its pid file is not written yet
                self.stop(job_id)
        deadline = time.time() + timeout
        while True:
            alive = [
                job_id
                for job_id in targets
                if self.status(job_id, tail_lines=0)["state"] in ACTIVE_STATES
            ]
            if not alive:
                return targets
            if time.time() >= deadline:
                raise RuntimeError(
                    f"jobs of {case} still running after {timeout:.0f} s: {', '.join(alive)}"
                )
            time.sleep(STOP_POLL_S)

    def _case_job_ids(self, case: str) -> list[str]:
        if not self.dir.is_dir():
            return []
        out = []
        for d in sorted(self.dir.iterdir()):
            meta_file = d / "meta.json"
            if not d.is_dir() or not meta_file.is_file():
                continue
            try:
                meta = json.loads(meta_file.read_text())
            except (OSError, ValueError):
                continue
            if meta.get("case") == case:
                out.append(d.name)
        return out

    def list(self, limit: int = 20) -> list[dict]:
        if not self.dir.is_dir():
            return []
        ids = sorted((p.name for p in self.dir.iterdir() if p.is_dir()), reverse=True)[:limit]
        out = []
        for i in ids:
            s = self.status(i, tail_lines=0)
            out.append({k: s[k] for k in ("job_id", "case", "app", "np", "state", "elapsed_s")})
        return out
