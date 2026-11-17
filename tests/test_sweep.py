"""run_sweep / collect_snapshots with fake cases and jobs (no OpenFOAM)."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from fvmrom.core.config import Settings
from fvmrom.foam import (
    Cases,
    FoamField,
    Jobs,
    RunResult,
    collect_snapshots,
    run_sweep,
    validate_study,
    write_field,
)
from fvmrom.foam.sweep import _check_case_path, _finished, _read_cd
from fvmrom.core.params import ParamSpace

SPACE = ParamSpace(names=("a",), lower=(1.0,), upper=(2.0,), log=(False,))
SOLVER_LOG = "log.simpleFoam"
DEFAULT_OWNER = "0 0 1 0 1 0 1 0 1"


def _mesh_files(
    root: Path,
    n_cells: int = 2,
    owner: str = DEFAULT_OWNER,
    neighbour: str = "1",
) -> None:
    poly = root / "constant" / "polyMesh"
    poly.mkdir(parents=True, exist_ok=True)
    (poly / "owner").write_text(
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        f'    note        "nPoints:6 nCells:{n_cells} nFaces:9 nInternalFaces:1";\n'
        "    class       labelList;\n    object      owner;\n}\n\n"
        f"9\n(\n{owner}\n)\n"
    )
    (poly / "neighbour").write_text(
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        "    class       labelList;\n    object      neighbour;\n}\n\n"
        f"1\n(\n{neighbour}\n)\n"
    )
    (poly / "boundary").write_text(
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        "    class       polyBoundaryMesh;\n    object      boundary;\n}\n\n"
        "5\n(\n"
        "    left\n    {\n        type            patch;\n        nFaces          1;\n"
        "        startFace       1;\n    }\n"
        "    right\n    {\n        type            patch;\n        nFaces          1;\n"
        "        startFace       2;\n    }\n"
        "    bottom\n    {\n        type            wall;\n        nFaces          2;\n"
        "        startFace       3;\n    }\n"
        "    top\n    {\n        type            wall;\n        nFaces          2;\n"
        "        startFace       5;\n    }\n"
        "    frontAndBack\n    {\n        type            empty;\n        nFaces          2;\n"
        "        startFace       7;\n    }\n"
        ")\n"
    )


def _write_case_dir(root: Path, converged: bool = True, n_cells: int = 2, with_v: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "system").mkdir(exist_ok=True)
    _mesh_files(root, n_cells=n_cells)

    U = np.zeros((n_cells, 3))
    U[:, 0] = np.linspace(0.5, 0.25, n_cells)
    p = np.linspace(1.0, 0.5, n_cells)
    volumes = np.full(n_cells, 1.0 / n_cells)
    phi = FoamField(
        "phi",
        "surfaceScalarField",
        np.array([0.5]),
        {
            "left": {"type": "patch", "value": np.array([-0.5])},
            "right": {"type": "patch", "value": np.array([0.5])},
            "bottom": {"type": "wall", "value": np.array([0.0, 0.0])},
            "top": {"type": "wall", "value": np.array([0.0, 0.0])},
            "frontAndBack": {"type": "empty", "value": None},
        },
    )
    write_field(root, "1", FoamField("U", "volVectorField", U, {}))
    write_field(root, "1", FoamField("p", "volScalarField", p, {}))
    if with_v:
        write_field(root, "1", FoamField("V", "volScalarField", volumes, {}))
    write_field(root, "1", phi)

    text = "Time = 1\n"
    if converged:
        text += "simpleFoam: SIMPLE solution converged in 10 iterations\n"
    text += "End\n"
    (root / SOLVER_LOG).write_text(text)


class FakeProblem:
    name = "fake"
    fields = ("U", "p", "phi")
    space = SPACE

    def __init__(self, converged: bool = True, with_v: bool = True):
        self.written: list[str] = []
        self.converged = converged
        self.with_v = with_v

    def write_case(self, cases: Cases, case: str, mu: np.ndarray) -> None:
        self.written.append(case)
        _write_case_dir(
            cases.path(case, must_exist=False), converged=self.converged, with_v=self.with_v
        )


class FakeJobs:
    def __init__(
        self,
        states=None,
        default=("completed",),
        exc=None,
        fail_on_call=None,
        stop_exc=None,
        pending_pid=(),
    ):
        self.states = {key: list(value) for key, value in (states or {}).items()}
        self.default = list(default)
        self.exc = exc
        self.fail_on_call = fail_on_call
        self.stop_exc = dict(stop_exc or {})
        self.pending_pid = set(pending_pid)
        self.calls = 0
        self.counts: dict[str, int] = {}
        self.started: list[tuple[str, str]] = []
        self.stopped: list[str] = []
        self.stopped_cases: list[str] = []
        self.waited: list[str] = []
        self.running: set[str] = set()
        self.max_running = 0

    def start(self, case: str, app: str, args: list[str], np: int = 1) -> dict:
        job_id = f"job{len(self.started)}"
        self.started.append((case, job_id))
        self.running.add(job_id)
        self.max_running = max(self.max_running, len(self.running))
        return {"job_id": job_id}

    def status(self, job_id: str, tail_lines: int = 0) -> dict:
        self.calls += 1
        if self.fail_on_call is not None and self.calls >= self.fail_on_call:
            raise (self.exc or RuntimeError)("interrupted")
        n = self.counts.get(job_id, 0)
        self.counts[job_id] = n + 1
        plan = self.states.get(job_id, self.default)
        state = plan[min(n, len(plan) - 1)]
        if state in ("completed", "failed", "died"):
            self.running.discard(job_id)
        return {"job_id": job_id, "state": state}

    def stop(self, job_id: str) -> dict:
        if job_id in self.pending_pid:
            raise RuntimeError("Job has no recorded pid yet (still starting?)")
        if job_id in self.stop_exc:
            raise self.stop_exc[job_id]
        self.stopped.append(job_id)
        self.running.discard(job_id)
        return {"job_id": job_id}

    def wait_pid(self, job_id: str, timeout: float = 20.0) -> str:
        self.waited.append(job_id)
        self.pending_pid.discard(job_id)
        return "4242"

    def stop_case(self, case: str, timeout: float = 30.0) -> list[str]:
        self.stopped_cases.append(case)
        ids = [j for c, j in self.started if c == case and j in self.running]
        for job_id in ids:
            self.stop(job_id)
        return ids


def _settings(tmp_path) -> Settings:
    return Settings(cases_dir=tmp_path / "cases", state_dir=tmp_path / "state")


def test_validate_study_accepts_valid_names():
    for name in ("re1d", "Re-1.0", "a_b", "0abc", "A"):
        assert validate_study(name) == name


@pytest.mark.parametrize(
    "name", ["", ".", "..", "a..b", "a/b", "..\\x", "-x", "x y", "a\\b", "re1d\n"]
)
def test_validate_study_rejects_bad_names(name):
    with pytest.raises(ValueError):
        validate_study(name)


@pytest.mark.parametrize(
    "name", ["re1d.", "re1d ", "CON", "nul", "COM1", "CON.txt", "lpt9", "Aux.tar.gz"]
)
def test_validate_study_rejects_trailing_dot_and_device_names(name):
    with pytest.raises(ValueError):
        validate_study(name)


def _study_settings(tmp_path, *, case_dirs=(), data_dirs=()) -> Settings:
    settings = Settings(
        cases_dir=tmp_path / "cases",
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
    )
    for d in case_dirs:
        (settings.cases_dir / d).mkdir(parents=True, exist_ok=True)
    for d in data_dirs:
        (settings.data_dir / d).mkdir(parents=True, exist_ok=True)
    return settings


def test_validate_study_rejects_case_only_conflict(tmp_path):
    settings = _study_settings(tmp_path, data_dirs=["fake/re1d"])
    for name in ("RE1D", "Re1d"):
        with pytest.raises(ValueError, match="letter case differs"):
            validate_study(name, problem="fake", settings=settings)
    assert validate_study("re1d", problem="fake", settings=settings) == "re1d"
    assert validate_study("re1d2", problem="fake", settings=settings) == "re1d2"


def test_validate_study_rejects_case_conflict_in_cases_dir(tmp_path):
    settings = _study_settings(tmp_path, case_dirs=["fake_X"])
    with pytest.raises(ValueError, match="letter case differs"):
        validate_study("x", problem="fake", settings=settings)


def test_validate_study_conflict_needs_the_same_problem(tmp_path):
    settings = _study_settings(tmp_path, case_dirs=["other_abc"], data_dirs=["other/abc"])
    assert validate_study("ABC", problem="fake", settings=settings) == "ABC"


def test_run_sweep_rejects_a_case_path_resolved_elsewhere(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    root = cases.path("fake_s/run_0000", must_exist=False)
    root.mkdir(parents=True)
    if not (tmp_path / "cases" / "FAKE_S").is_dir():
        pytest.skip("needs a case-insensitive filesystem")

    _check_case_path(cases, "fake_s/run_0000", root)
    with pytest.raises(ValueError, match="resolves to"):
        _check_case_path(cases, "FAKE_S/run_0000", root)
    with pytest.raises(ValueError, match="resolves to"):
        _check_case_path(cases, "fake_s./run_0000", root)


def test_resume_skips_converged_cases(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    root = cases.path("fake_s/run_0000", must_exist=False)
    _write_case_dir(root)

    problem = FakeProblem()
    jobs = FakeJobs()
    results = run_sweep(
        problem, "s", np.array([[1.5]]), settings=settings, resume=True,
        cases=cases, jobs=jobs,
    )  # fmt: skip

    assert len(results) == 1 and results[0].converged
    assert "skipped" in results[0].message
    assert problem.written == [] and jobs.started == []


def test_resume_cleans_and_reruns_unfinished_cases(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    root = cases.path("fake_s/run_0000", must_exist=False)
    root.mkdir(parents=True)
    (root / "stale.txt").write_text("old run")
    (root / SOLVER_LOG).write_text("Time = 1\nEnd\n")

    problem = FakeProblem()
    jobs = FakeJobs()
    results = run_sweep(
        problem, "s", np.array([[1.5]]), settings=settings, resume=True,
        cases=cases, jobs=jobs,
    )  # fmt: skip

    assert not (root / "stale.txt").exists()
    assert problem.written == ["fake_s/run_0000"]
    assert len(jobs.started) == 1
    assert results[0].converged


def test_default_mode_reruns_converged_cases(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    root = cases.path("fake_s/run_0000", must_exist=False)
    _write_case_dir(root)
    (root / "stale.txt").write_text("old run")

    problem = FakeProblem()
    jobs = FakeJobs()
    results = run_sweep(
        problem, "s", np.array([[1.5]]), settings=settings, cases=cases, jobs=jobs
    )

    assert not (root / "stale.txt").exists()
    assert jobs.stopped_cases == ["fake_s/run_0000"]
    assert problem.written == ["fake_s/run_0000"]
    assert len(jobs.started) == 1
    assert results[0].converged


def test_run_sweep_validates_the_study_name(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    with pytest.raises(ValueError, match="invalid study name"):
        run_sweep(
            FakeProblem(),
            "../escape",
            np.array([[1.5]]),
            settings=settings,
            cases=cases,
            jobs=FakeJobs(),
        )


def test_finished_requires_the_volume_field(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    case = "fake_s/run_0000"
    _write_case_dir(cases.path(case, must_exist=False), with_v=False)
    assert not _finished(cases, case)
    _write_case_dir(cases.path(case))
    assert _finished(cases, case)


def test_converged_log_without_V_is_not_converged(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    results = run_sweep(
        FakeProblem(with_v=False),
        "s",
        np.array([[1.5]]),
        settings=settings,
        cases=cases,
        jobs=FakeJobs(),
    )
    assert not results[0].converged
    assert "V is missing" in results[0].message


def test_failed_job_is_not_converged_and_is_excluded(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    problem = FakeProblem()
    jobs = FakeJobs(states={"job0": ["failed"], "job1": ["completed"]})
    params = np.array([[1.5], [1.9]])
    results = run_sweep(
        problem, "s", params, settings=settings, cases=cases, jobs=jobs, poll_s=0.0
    )

    assert [r.converged for r in results] == [False, True]
    assert "job failed" in results[0].message

    out_dir = tmp_path / "data"
    path = collect_snapshots(
        problem, "s", results, out_dir, ["train", "test"], settings=settings
    )
    data = np.load(path)
    assert data["params"].shape == (1, 1)
    assert np.allclose(data["params"][0], params[1])
    assert data["U"].shape == (1, 2, 3) and data["p"].shape == (1, 2)
    assert data["phi"].shape == (1, 7)

    meta = json.loads((out_dir / "samples.json").read_text(encoding="utf-8"))
    assert [s["converged"] for s in meta["samples"]] == [False, True]
    assert meta["n_converged"] == 1
    assert meta["n_faces"] == 7


def test_collect_snapshots_reports_the_phi_length_as_n_faces(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    problem = FakeProblem()
    case = "fake_s/run_0000"
    _write_case_dir(cases.path(case, must_exist=False))
    results = [RunResult(0, case, np.array([1.5]), True, "", "ok")]

    out_dir = tmp_path / "data"
    collect_snapshots(problem, "s", results, out_dir, ["train"], settings=settings)
    meta = json.loads((out_dir / "samples.json").read_text(encoding="utf-8"))
    # 1 internal face + 6 faces on non-empty patches; the 2 empty-patch faces are excluded
    assert meta["n_faces"] == 7
    with np.load(out_dir / "snapshots.npz") as data:
        assert data["phi"].shape == (1, meta["n_faces"])


def test_collect_snapshots_rejects_inconsistent_topology(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    problem = FakeProblem()
    results = []
    for i, n_cells in enumerate((2, 3)):
        case = f"fake_s/run_{i:04d}"
        _write_case_dir(cases.path(case, must_exist=False), n_cells=n_cells)
        results.append(RunResult(i, case, np.array([1.5]), True, "", "ok"))

    with pytest.raises(ValueError, match="differs from the first sample"):
        collect_snapshots(
            problem, "s", results, tmp_path / "data", ["train", "train"], settings=settings
        )


@pytest.mark.parametrize(
    "name,value,match",
    [
        ("owner", "1 1 0 1 0 1 0 1 0", "owner array"),
        ("neighbour", "0", "neighbour array"),
    ],
)
def test_collect_snapshots_rejects_different_face_arrays(tmp_path, name, value, match):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    problem = FakeProblem()
    results = []
    for i in range(2):
        case = f"fake_s/run_{i:04d}"
        root = cases.path(case, must_exist=False)
        _write_case_dir(root)
        if i == 1:  # same face counts, different connectivity
            _mesh_files(root, **{name: value})
        results.append(RunResult(i, case, np.array([1.5]), True, "", "ok"))

    with pytest.raises(ValueError, match=match):
        collect_snapshots(
            problem, "s", results, tmp_path / "data", ["train", "train"], settings=settings
        )


def test_run_sweep_respects_max_parallel(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    problem = FakeProblem()
    jobs = FakeJobs(default=("running", "running", "completed"))
    params = np.linspace(1.1, 1.5, 5).reshape(-1, 1)
    results = run_sweep(
        problem,
        "s",
        params,
        max_parallel=2,
        poll_s=0.0,
        settings=settings,
        cases=cases,
        jobs=jobs,
    )

    assert len(results) == 5 and all(r.converged for r in results)
    assert len(jobs.started) == 5
    assert jobs.max_running == 2


@pytest.mark.parametrize("exc", [RuntimeError, KeyboardInterrupt])
def test_sweep_stops_running_jobs_on_interrupt(tmp_path, exc):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    problem = FakeProblem()
    jobs = FakeJobs(exc=exc, fail_on_call=1)
    with pytest.raises(exc):
        run_sweep(
            problem,
            "s",
            np.array([[1.5], [1.9]]),
            max_parallel=2,
            poll_s=0.0,
            settings=settings,
            cases=cases,
            jobs=jobs,
        )
    assert len(jobs.started) == 2
    assert set(jobs.stopped) == {job_id for _, job_id in jobs.started}


def test_sweep_stops_stale_jobs_before_cleaning_a_case(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    root = cases.path("fake_s/run_0000", must_exist=False)
    root.mkdir(parents=True)
    (root / "stale.txt").write_text("old run")

    problem = FakeProblem()
    jobs = FakeJobs()
    run_sweep(
        problem, "s", np.array([[1.5]]), settings=settings, cases=cases, jobs=jobs
    )

    assert jobs.stopped_cases == ["fake_s/run_0000"]
    assert problem.written == ["fake_s/run_0000"]


def test_sweep_does_not_touch_absent_cases(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    jobs = FakeJobs()
    run_sweep(
        FakeProblem(), "s", np.array([[1.5]]), settings=settings, cases=cases, jobs=jobs
    )

    assert jobs.stopped_cases == []


def test_sweep_raises_when_a_job_cannot_be_stopped(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    jobs = FakeJobs(
        exc=RuntimeError, fail_on_call=1, stop_exc={"job0": RuntimeError("kill failed")}
    )

    with pytest.raises(RuntimeError, match="job0") as info:
        run_sweep(
            FakeProblem(),
            "s",
            np.array([[1.5], [1.9]]),
            max_parallel=2,
            poll_s=0.0,
            settings=settings,
            cases=cases,
            jobs=jobs,
        )

    assert isinstance(info.value.__cause__, RuntimeError)
    assert str(info.value.__cause__) == "kill failed"
    assert isinstance(info.value.__context__, RuntimeError)  # the original error is kept


def test_sweep_waits_for_pid_of_starting_jobs(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    jobs = FakeJobs(
        default=("running",), exc=RuntimeError, fail_on_call=2, pending_pid={"job0"}
    )

    with pytest.raises(RuntimeError, match="interrupted"):
        run_sweep(
            FakeProblem(), "s", np.array([[1.5]]), poll_s=0.0, settings=settings,
            cases=cases, jobs=jobs,
        )  # fmt: skip

    assert jobs.waited == ["job0"]
    assert jobs.stopped == ["job0"]


def _fake_job(jobs: Jobs, job_id: str, case: str, pid: str | None = "1234") -> None:
    d = jobs.dir / job_id
    d.mkdir(parents=True)
    (d / "meta.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "case": case,
                "app": "./Allrun",
                "args": [],
                "np": 1,
                "log": "log.Allrun",
                "started": 0.0,
            }
        )
    )
    if pid is not None:
        (d / "pid").write_text(pid)


def test_stop_case_stops_running_jobs_and_waits(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    jobs = Jobs(settings)
    _fake_job(jobs, "job-run", "fake_s/run_0000")
    _fake_job(jobs, "job-done", "fake_s/run_0000")
    _fake_job(jobs, "job-other", "fake_s/run_0001")
    plan = {"job-run": ["running", "running", "died"], "job-done": ["completed"],
            "job-other": ["running"]}  # fmt: skip
    counts: dict[str, int] = {}
    stopped: list[str] = []

    def status(job_id, tail_lines=0):
        n = counts.get(job_id, 0)
        counts[job_id] = n + 1
        return {"job_id": job_id, "state": plan[job_id][min(n, len(plan[job_id]) - 1)]}

    monkeypatch.setattr(jobs, "status", status)
    monkeypatch.setattr(jobs, "stop", lambda job_id: stopped.append(job_id))
    monkeypatch.setattr("fvmrom.foam.jobs.STOP_POLL_S", 0.0)

    assert jobs.stop_case("fake_s/run_0000") == ["job-run"]
    assert stopped == ["job-run"]
    assert counts["job-run"] >= 3  # polled until the process was gone
    assert counts.get("job-other", 0) == 0


def test_stop_case_raises_on_timeout(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    jobs = Jobs(settings)
    _fake_job(jobs, "job-run", "fake_s/run_0000")
    monkeypatch.setattr(jobs, "status", lambda job_id, tail_lines=0: {"state": "running"})
    monkeypatch.setattr(jobs, "stop", lambda job_id: {"job_id": job_id})

    with pytest.raises(RuntimeError, match="still running after"):
        jobs.stop_case("fake_s/run_0000", timeout=0.0)


def test_stop_case_returns_without_jobs(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    jobs = Jobs(settings)
    _fake_job(jobs, "job-done", "fake_s/run_0000")
    monkeypatch.setattr(jobs, "status", lambda job_id, tail_lines=0: {"state": "completed"})
    monkeypatch.setattr(
        jobs, "stop", lambda job_id: pytest.fail("stop must not be called")
    )

    assert jobs.stop_case("fake_s/run_0000") == []
    assert jobs.stop_case("nothing/here") == []


def test_stop_case_waits_for_pid_of_a_starting_job(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    jobs = Jobs(settings)
    _fake_job(jobs, "job-start", "fake_s/run_0000", pid=None)
    calls: list[str] = []
    state = {"value": "starting"}

    def status(job_id, tail_lines=0):
        return {"state": state["value"]}

    def stop(job_id):
        calls.append("stop")
        if state["value"] == "starting":
            raise RuntimeError("Job has no recorded pid yet (still starting?)")
        state["value"] = "died"
        return {"job_id": job_id}

    def wait_pid(job_id, timeout=20.0):
        calls.append("wait_pid")
        state["value"] = "running"
        return "4242"

    monkeypatch.setattr(jobs, "status", status)
    monkeypatch.setattr(jobs, "stop", stop)
    monkeypatch.setattr(jobs, "wait_pid", wait_pid)

    assert jobs.stop_case("fake_s/run_0000") == ["job-start"]
    assert calls == ["stop", "wait_pid", "stop"]


COEFFICIENT_HEADER = (
    "# Force and moment coefficients\n"
    "#\n"
    "# Time\tCd\tCd(f)\tCd(r)\tCl\tCl(f)\tCl(r)\n"
)


def _write_cd(root: Path, value: float, time: str = "0") -> None:
    path = root / "postProcessing" / "forceCoeffs" / time / "coefficient.dat"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        COEFFICIENT_HEADER
        + f"{time}\t{value:.12e}\t{value / 2:.12e}\t{value / 2:.12e}\t0\t0\t0\n"
    )


class CylinderFakeProblem(FakeProblem):
    name = "cylinder"


def test_collect_snapshots_records_cd_for_cylinder(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    problem = CylinderFakeProblem()
    results = []
    for i in range(2):
        case = f"cylinder_s/run_{i:04d}"
        _write_case_dir(cases.path(case, must_exist=False))
        results.append(RunResult(i, case, np.array([1.5 + i]), True, "", "ok"))
    # the second sample has no forceCoeffs output: cd is null, not an error
    _write_cd(cases.path(results[0].case), 1.62)

    out_dir = tmp_path / "data"
    collect_snapshots(problem, "s", results, out_dir, ["train", "test"], settings=settings)
    meta = json.loads((out_dir / "samples.json").read_text(encoding="utf-8"))
    assert [s["cd"] for s in meta["samples"]] == [pytest.approx(1.62), None]


def test_collect_snapshots_has_no_cd_for_other_problems(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    problem = FakeProblem()
    case = "fake_s/run_0000"
    _write_case_dir(cases.path(case, must_exist=False))
    results = [RunResult(0, case, np.array([1.5]), True, "", "ok")]
    out_dir = tmp_path / "data"
    collect_snapshots(problem, "s", results, out_dir, ["train"], settings=settings)
    meta = json.loads((out_dir / "samples.json").read_text(encoding="utf-8"))
    assert "cd" not in meta["samples"][0]


def test_read_cd_picks_the_latest_numeric_time_directory(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    case = "cylinder_s/run_0000"
    root = cases.path(case, must_exist=False)
    _write_case_dir(root)
    _write_cd(root, 1.0, time="0")
    _write_cd(root, 1.5, time="10")
    _write_cd(root, 9.9, time="latest")  # a non-numeric directory name is ignored
    # the numeric directory name decides, not the modification time
    old = root / "postProcessing" / "forceCoeffs" / "10" / "coefficient.dat"
    os.utime(old, (1.0, 1.0))

    assert _read_cd(cases, case) == pytest.approx(1.5)


def test_read_cd_skips_non_numeric_rows(tmp_path):
    settings = _settings(tmp_path)
    cases = Cases(settings)
    case = "cylinder_s/run_0000"
    root = cases.path(case, must_exist=False)
    _write_case_dir(root)
    _write_cd(root, 1.62, time="0")
    path = root / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"
    path.write_text(path.read_text(encoding="utf-8") + "End\n", encoding="utf-8")

    assert _read_cd(cases, case) == pytest.approx(1.62)
