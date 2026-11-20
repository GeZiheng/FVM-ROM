"""Scan the OpInf ridge on the transient study and report the sensitivity (read-only).

Every ridge of the grid is fitted explicitly on the training window only
(``TransientROM(dynamics="opinf", dynamics_kw={"ridge": ridge})``), so neither the
ridge selection nor its post-refit recheck runs.  Two groups of diagnostics are
reported:

- training-data diagnostics (they may be used to choose a ridge): the replay of
  the whole training window from its first state, the forward integration from
  its last state for ``--check-periods`` periods, and the recheck-style stability
  verdict (the same procedure as the ``OpInf`` refit recheck);
- post-hoc diagnostics of the prediction window (they use held-out data, they are
  **not** used to choose a ridge and they do not replace the formal acceptance):
  the U error within ``tau <= 5``, the U error at ``tau = 1/3/5/10``, the
  frequency, Cl amplitude, Cd mean and Cl phase errors, divergence and the energy
  drift.

Nothing is written except ``data/cylinder_transient/<study>/opinf_lambda_scan.json``;
``report.json`` and ``models/`` are untouched.

Usage:
  uv run python scripts/opinf_lambda_scan.py --study re100 --train-periods 3 \\
      --n-modes U=8,p=8,phi=8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from fvmrom.core.config import PROJECT_ROOT, get_settings
from fvmrom.foam import validate_study
from fvmrom.rom import TransientROM
from fvmrom.rom.dynamics import _dominant_period, _max_norm_ratio
from fvmrom.rom.evaluate_transient import evaluate_transient, to_jsonable


def _load_shared_script():
    path = Path(__file__).resolve().with_name("fit_and_eval_transient.py")
    spec = importlib.util.spec_from_file_location("fit_and_eval_transient_shared", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit_and_eval = _load_shared_script()

PROBLEM = fit_and_eval.PROBLEM
DEFAULT_N_MODES_TEXT = fit_and_eval.DEFAULT_N_MODES_TEXT

#: default ridge grid: four values per decade from 1e-5 to 1e-2, 13 values;
#: it contains 1e-4 (selected by the holdout) and 1e-3 (the formal value after
#: the recheck fallback)
DEFAULT_RIDGE_GRID = np.logspace(-5.0, -2.0, 13)

#: ridge values annotated in the output when the grid contains them
HOLDOUT_RIDGE = 1e-4
FORMAL_RIDGE = 1e-3
ANNOTATIONS = {
    HOLDOUT_RIDGE: "selected by hold-out",
    FORMAL_RIDGE: "formal opinf (after fallback)",
}

DEFAULT_CHECK_PERIODS = 5.0
DEFAULT_BLOWUP_FACTOR = 5.0

#: marker of the post-hoc group in the terminal and JSON output
POST_HOC_NOTE = "post-hoc (uses the prediction window; NOT used to choose lambda)"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", default="re100")
    p.add_argument("--train-periods", type=float, default=3.0)
    p.add_argument(
        "--n-modes",
        default=DEFAULT_N_MODES_TEXT,
        metavar="FIELD=INT",
        help="per-field modes, e.g. U=8,p=8,phi=8; 'auto' uses --energy-tol",
    )
    p.add_argument(
        "--ridge-grid",
        default=None,
        metavar="R1,R2,...",
        help="comma separated ridge values (default: logspace(-5, -2, 13))",
    )
    p.add_argument("--energy-tol", type=float, default=1e-10)
    p.add_argument("--readout-ridge", type=float, default=1e-10)
    p.add_argument(
        "--check-periods",
        type=float,
        default=DEFAULT_CHECK_PERIODS,
        help="periods of the forward stability integration (training data only)",
    )
    p.add_argument(
        "--blowup-factor",
        type=float,
        default=DEFAULT_BLOWUP_FACTOR,
        help="norm ratio over the largest training norm counted as unstable",
    )
    p.add_argument("--data-dir", type=Path, default=None, help="data root (default: settings)")
    args = p.parse_args(argv)

    if not args.train_periods > 0.0:
        p.error("--train-periods must be positive")
    if not args.check_periods > 0.0:
        p.error("--check-periods must be positive")
    if not args.blowup_factor > 0.0:
        p.error("--blowup-factor must be positive")
    args.n_modes = fit_and_eval._parse_n_modes(args.n_modes)
    args.ridge_grid = parse_ridge_grid(args.ridge_grid)
    return args


def parse_ridge_grid(text: str | None) -> np.ndarray:
    if text is None:
        return DEFAULT_RIDGE_GRID.copy()
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError:
            raise SystemExit(
                f"bad --ridge-grid entry {item!r} (expected a positive number)"
            ) from None
        if not (np.isfinite(value) and value >= 0.0):
            raise SystemExit(
                f"bad --ridge-grid entry {item!r} (must be a finite non-negative number)"
            )
        values.append(value)
    if not values:
        raise SystemExit("--ridge-grid got no values")
    return np.asarray(values, dtype=float)


def annotation(ridge: float) -> list[str]:
    """The labels of a ridge value when it is exactly on the grid (none otherwise)."""
    for value, label in ANNOTATIONS.items():
        if np.isclose(ridge, value, rtol=1e-12, atol=0.0):
            return [label]
    return []


def main() -> None:
    args = parse_args()
    settings = get_settings()
    root = settings.data_dir if args.data_dir is None else Path(args.data_dir)
    study = validate_study(args.study, problem=PROBLEM, settings=settings)
    data_dir = root / PROBLEM / study
    npz_path = data_dir / "snapshots.npz"
    if not npz_path.is_file():
        raise SystemExit(
            f"missing snapshots in {data_dir}: run scripts/generate_transient.py first"
        )
    out_path = data_dir / "opinf_lambda_scan.json"

    times, forces, volumes, phi_w, fields_all = fit_and_eval._load(npz_path)
    try:
        f_fom = fit_and_eval.dominant_frequency(times, forces[:, 1])
    except ValueError as exc:
        raise SystemExit(f"cannot estimate the FOM shedding frequency: {exc}") from exc
    period = 1.0 / f_fom

    n_train, window = fit_and_eval._split(times, args.train_periods, period)
    if window["n_test"] == 0:
        raise SystemExit("the training window covers every snapshot; reduce --train-periods")
    if n_train < 5:
        raise SystemExit(f"OpInf needs at least five training snapshots, got {n_train}")

    t_train, t_test = times[:n_train], times[n_train:]
    snap_train = {name: values[:n_train] for name, values in fields_all.items()}
    snap_test = {name: values[n_train:] for name, values in fields_all.items()}
    mean_train = {
        name: np.asarray(values[:n_train].mean(axis=0), dtype=float)
        for name, values in fields_all.items()
        if name in ("U", "p")
    }
    forces_train = forces[:n_train]
    forces_test = forces[n_train:]

    dt = float(times[1] - times[0]) if times.size > 1 else None
    print(
        f"{PROBLEM}/{study}: {times.size} snapshots t=[{times[0]:g}, {times[-1]:g}], "
        f"dt={dt:g}, FOM period {period:.4f}"
    )
    print(
        f"training window: {window['n_train']} snapshots t<={window['cutoff']:.4f}, "
        f"t_train_end={window['t_train_end']:.4f} "
        f"({window['train_periods']:.2f} periods); prediction window "
        f"{window['n_test']} snapshots from t={window['t_test_start']:.4f} "
        f"({window['test_periods']:.2f} periods)"
    )
    print(
        f"n_modes {fit_and_eval._n_modes_text(args.n_modes)}, "
        f"energy_tol {args.energy_tol:g}, readout ridge {args.readout_ridge:g}"
    )
    present = [
        f"{value:g} = {label}"
        for value, label in ANNOTATIONS.items()
        if any(np.isclose(ridge, value, rtol=1e-12, atol=0.0) for ridge in args.ridge_grid)
    ]
    print(
        f"ridge grid: {args.ridge_grid.size} values in "
        f"[{args.ridge_grid.min():g}, {args.ridge_grid.max():g}]"
        + (f"; on the grid: {'; '.join(present)}" if present else "")
    )
    print(f"{POST_HOC_NOTE}\n", flush=True)

    entries = []
    for ridge in args.ridge_grid:
        entry = _scan_ridge(
            float(ridge),
            args.n_modes,
            t_train,
            t_test,
            snap_train,
            snap_test,
            forces_train,
            forces_test,
            volumes,
            phi_w,
            mean_train,
            period=period,
            f_fom=f_fom,
            energy_tol=args.energy_tol,
            readout_ridge=args.readout_ridge,
            check_periods=args.check_periods,
            blowup_factor=args.blowup_factor,
        )
        entries.append(entry)
        if entry["status"] == "ok":
            training = entry["training"]
            post = entry["post_hoc"] or {}
            print(
                f"ridge {ridge:.2e}: stable={'yes' if training['recheck_stable'] else 'no'}, "
                f"replay err {_sci(training['train_replay_error'])}, "
                f"U max tau<=5 {_sci(post.get('u_error_max_within_tau5'))}, "
                f"diverged={'yes' if post.get('diverged') else 'no'} "
                f"({entry['cost']['fit_seconds']:.1f} s)",
                flush=True,
            )
        else:
            print(f"ridge {ridge:.2e}: {entry['status']}: {entry['error']}", flush=True)

    summary = _summary(entries)
    _print_training_table(entries)
    _print_post_hoc_table(entries)
    _print_notes(entries)
    _print_summary(summary)

    report = {
        "problem": PROBLEM,
        "study": study,
        "snapshots": _label(npz_path),
        "read_only": True,
        "post_hoc_note": POST_HOC_NOTE,
        "config": {
            "train_periods": args.train_periods,
            "n_modes": args.n_modes,
            "energy_tol": args.energy_tol,
            "readout_ridge": args.readout_ridge,
            "ridge_grid": args.ridge_grid,
            "check_periods": args.check_periods,
            "blowup_factor": args.blowup_factor,
            "fom_period": period,
        },
        "training_window": window,
        "scan": entries,
        "summary": summary,
    }
    out_path.write_text(
        json.dumps(to_jsonable(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nscan: {out_path}")


def _label(path: Path) -> str:
    if path.is_relative_to(PROJECT_ROOT):
        return path.relative_to(PROJECT_ROOT).as_posix()
    return str(path)


def _scan_ridge(
    ridge: float,
    n_modes: dict[str, int | None],
    t_train: np.ndarray,
    t_test: np.ndarray,
    snap_train: dict[str, np.ndarray],
    snap_test: dict[str, np.ndarray],
    forces_train: np.ndarray,
    forces_test: np.ndarray,
    volumes: np.ndarray,
    phi_w: np.ndarray,
    mean_train: dict[str, np.ndarray],
    *,
    period: float,
    f_fom: float,
    energy_tol: float,
    readout_ridge: float,
    check_periods: float,
    blowup_factor: float,
) -> dict:
    """Fit one explicit ridge on the training window and diagnose it."""
    entry = {
        "ridge": float(ridge),
        "annotation": annotation(ridge),
        "status": "ok",
        "training": None,
        "post_hoc": None,
    }
    rom = TransientROM(
        dynamics="opinf",
        n_modes=n_modes,
        energy_tol=energy_tol,
        readout_ridge=readout_ridge,
        dynamics_kw={"ridge": float(ridge)},
    )
    started = time.perf_counter()
    try:
        rom.fit(
            t_train,
            snap_train,
            volumes,
            phi_weights=phi_w,
            scalars={"forces": forces_train},
        )
    except (ValueError, RuntimeError) as exc:
        entry["status"] = "failed"
        entry["error"] = str(exc)
        entry["cost"] = {"fit_seconds": time.perf_counter() - started}
        return entry
    entry["cost"] = {"fit_seconds": time.perf_counter() - started}

    entry["training"] = _training_diagnostics(
        rom, check_periods=check_periods, blowup_factor=blowup_factor
    )

    started = time.perf_counter()
    try:
        result = evaluate_transient(
            rom,
            t_test,
            snap_test,
            volumes,
            phi_w,
            period=period,
            forces_test=forces_test,
            mean_train=mean_train,
            f_fom=f_fom,
            topo=None,
            geom=None,
        )
    except (ValueError, RuntimeError) as exc:
        entry["cost"]["evaluate_seconds"] = time.perf_counter() - started
        entry["post_hoc_error"] = str(exc)
        return entry
    entry["cost"]["evaluate_seconds"] = time.perf_counter() - started
    entry["post_hoc"] = _post_hoc(result, rom)
    return entry


def _training_diagnostics(rom, *, check_periods: float, blowup_factor: float) -> dict:
    """Replay and forward stability check of the fitted model (training data only)."""
    model = rom.model_
    t = np.asarray(rom.t_train, dtype=float)
    a = np.asarray(rom.latent_train_, dtype=float)
    period = _dominant_period(t, a)
    estimated = bool(np.isfinite(period) and period > 0.0)
    if not estimated:
        period = float(t[-1] - t[0])
    threshold = model._threshold(blowup_factor)

    replay, diverged_replay, _ = model._integrate(
        model.coef_, float(t[0]), a[0], t, threshold
    )
    span = check_periods * period
    steps = max(1, int(round(span / model.dt_)))
    forward_times = float(t[-1]) + np.linspace(span / steps, span, steps)
    forward, diverged_forward, _ = model._integrate(
        model.coef_, float(t[-1]), a[-1], forward_times, threshold
    )
    return {
        "train_replay_error": _finite(_relative_trajectory_error(replay, a)),
        "replay_max_norm_ratio": _finite(_max_norm_ratio(replay, model.a_max_norm_)),
        "forward_max_norm_ratio": _finite(_max_norm_ratio(forward, model.a_max_norm_)),
        "recheck_stable": bool(not (diverged_replay or diverged_forward)),
        "replay_diverged": bool(diverged_replay),
        "forward_diverged": bool(diverged_forward),
        "check_periods": float(check_periods),
        "check_period": float(period),
        "check_period_estimated": estimated,
        "blowup_factor": float(blowup_factor),
        "check_steps": int(steps),
    }


def _relative_trajectory_error(values: np.ndarray, target: np.ndarray) -> float:
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(target))):
        return float("nan")
    denominator = float(np.linalg.norm(target))
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm(values - target) / denominator)


def _post_hoc(result: dict, rom) -> dict:
    """Prediction-window metrics (held-out data; not used to choose the ridge)."""
    forces_report = fit_and_eval._forces(result)
    energy = fit_and_eval._energy(result, "rom")
    model = rom.model_
    return {
        "u_error_max_within_tau5": _finite(result["max_error_within_tau"].get("U")),
        "u_error_at_tau": {
            key: _finite(value)
            for key, value in result["errors_at_tau"].get("U", {}).items()
        },
        "frequency_relative_error": _finite(result["frequency"]["relative_error"]),
        "cl_amplitude_relative_error": _finite(
            None if forces_report is None
            else forces_report["cl_amplitude"]["relative_error"]
        ),
        "cd_mean_relative_error": _finite(
            None if forces_report is None else forces_report["cd_mean"]["relative_error"]
        ),
        "cl_phase_tau5_deg": _finite(
            None if forces_report is None
            else forces_report["cl_phase_deg"].get("5")
        ),
        "diverged": None if model.diverged_ is None else bool(model.diverged_),
        "diverged_at": _finite(model.diverged_at_),
        "energy_relative_change": _finite(
            None if energy is None else energy["relative_change"]
        ),
        "max_norm_ratio": _finite(None if energy is None else energy["max_norm_ratio"]),
    }


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _sci(value) -> str:
    return fit_and_eval._sci(value)


def _print_training_table(entries: list[dict]) -> None:
    print("\ntraining-data diagnostics (may be used to choose a ridge)")
    body = []
    for entry in entries:
        info = entry["training"] or {}
        body.append(
            [
                f"{entry['ridge']:.2e}",
                _sci(info.get("train_replay_error")),
                fit_and_eval._ratio(info.get("replay_max_norm_ratio")),
                fit_and_eval._ratio(info.get("forward_max_norm_ratio")),
                _yes_no(info.get("recheck_stable")),
            ]
        )
    fit_and_eval._table(
        [
            "ridge",
            "train replay err",
            "replay max|a|/train",
            "forward max|a|/train",
            "recheck stable",
        ],
        body,
    )


def _print_post_hoc_table(entries: list[dict]) -> None:
    print(f"\n{POST_HOC_NOTE}")
    body = []
    for entry in entries:
        info = entry["post_hoc"] or {}
        errors = info.get("u_error_at_tau") or {}
        body.append(
            [
                f"{entry['ridge']:.2e}",
                _sci(info.get("u_error_max_within_tau5")),
                _sci(errors.get("5")),
                _sci(info.get("frequency_relative_error")),
                _sci(info.get("cl_amplitude_relative_error")),
                _sci(info.get("cd_mean_relative_error")),
                fit_and_eval._deg(info.get("cl_phase_tau5_deg")),
                _diverged_text(info),
            ]
        )
    fit_and_eval._table(
        [
            "ridge",
            "U max tau<=5",
            "U tau5",
            "freq err",
            "Cl amp err",
            "Cd mean err",
            "Cl phase tau5 deg",
            "diverged",
        ],
        body,
    )


def _diverged_text(info: dict) -> str:
    if info.get("diverged") is None:
        return "n/a"
    if not info["diverged"]:
        return "no"
    at = info.get("diverged_at")
    return "yes" if at is None else f"yes (t={at:g})"


def _yes_no(value) -> str:
    return "n/a" if value is None else ("yes" if value else "no")


def _print_notes(entries: list[dict]) -> None:
    lines = [
        f"  ridge {entry['ridge']:.2e}: {', '.join(entry['annotation'])}"
        for entry in entries
        if entry["annotation"]
    ]
    if lines:
        print("\nridge labels (values present on the ridge grid):")
        print("\n".join(lines))
    failed = [entry for entry in entries if entry["status"] != "ok"]
    for entry in failed:
        print(f"NOTE: ridge {entry['ridge']:.2e} could not be fitted: {entry['error']}")
    for entry in entries:
        if entry.get("post_hoc_error"):
            print(
                f"NOTE: ridge {entry['ridge']:.2e} has no post-hoc diagnostics: "
                f"{entry['post_hoc_error']}"
            )


def _summary(entries: list[dict]) -> dict:
    """Facts of the scan: minima of the post-hoc U error, stability counts."""
    ok = [entry for entry in entries if entry["status"] == "ok" and entry["post_hoc"]]
    stable = [entry for entry in ok if entry["training"]["recheck_stable"]]
    summary = {}
    for name, group in (("all", ok), ("recheck_stable", stable)):
        values = [
            (entry["post_hoc"]["u_error_max_within_tau5"], entry["ridge"])
            for entry in group
            if entry["post_hoc"]["u_error_max_within_tau5"] is not None
        ]
        best, ridge = min(values) if values else (None, None)
        summary[name] = {
            "n": len(group),
            "min_post_hoc_u_error_within_tau5": None if best is None else float(best),
            "at_ridge": None if ridge is None else float(ridge),
            "any_below_1e-2": bool(values) and any(value < 1e-2 for value, _ in values),
            "n_diverged": sum(1 for entry in group if entry["post_hoc"]["diverged"]),
        }
    return summary


def _print_summary(summary: dict) -> None:
    print("\nsummary (data only; no tuning recommendation)")
    for name, label in (
        ("all", "all fitted ridges"),
        ("recheck_stable", "ridges the training-data recheck calls stable"),
    ):
        info = summary.get(name) or {}
        minimum = info.get("min_post_hoc_u_error_within_tau5")
        if minimum is None:
            print(f"  {label}: {info.get('n', 0)} ridges, no post-hoc U error available")
            continue
        print(
            f"  {label}: {info['n']} ridges; smallest post-hoc U max error within "
            f"tau<=5 is {minimum:.2e} at ridge {info['at_ridge']:.2e}; "
            f"{'some' if info['any_below_1e-2'] else 'none'} below 1e-2; "
            f"{info['n_diverged']} diverged"
        )


if __name__ == "__main__":
    main()
