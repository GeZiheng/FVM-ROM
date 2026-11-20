"""Fit a transient ROM on the collected snapshots, evaluate it and write a JSON report.

Usage:
  uv run python scripts/fit_and_eval_transient.py --study re100 --methods dmd \\
      --train-periods 3 --n-modes U=8,p=8,phi=8 --sweep-modes 4,6,8,10,12
  uv run python scripts/fit_and_eval_transient.py --study re100 --n-modes U=8,p=auto
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from fvmrom.core.config import PROJECT_ROOT, get_settings
from fvmrom.foam import (
    Cases,
    FoamField,
    read_field,
    read_geometry,
    read_topology,
    validate_study,
    write_field,
)
from fvmrom.foam.sweep import _check_case_path
from fvmrom.rom import TransientROM
from fvmrom.rom.evaluate_transient import (
    dominant_frequency,
    energy_drift,
    evaluate_transient,
    shedding_summary,
    to_jsonable,
)

PROBLEM = "cylinder_transient"
FIELDS = ("U", "p", "phi")

#: directory names of OpenFOAM time directories (negative times included)
TIME_DIR_RE = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")

#: method name -> (dynamics kind, :class:`TransientROM` dynamics arguments)
#:
#: ``opinf_norecheck`` is the same OpInf fit without the stability recheck after
#: the refit (the plain select-then-refit flow of plan 3.7), kept as a control.
METHOD_DYNAMICS = {
    "dmd": ("dmd", {}),
    "optdmd": ("optdmd", {}),
    "bopdmd": ("bopdmd", {}),
    "opinf": ("opinf", {}),
    "opinf_norecheck": ("opinf", {"recheck": False}),
}
METHODS = tuple(METHOD_DYNAMICS)
IMPLEMENTED = tuple(METHOD_DYNAMICS)

DEFAULT_N_MODES = {"U": 8, "p": 8, "phi": 8}
DEFAULT_N_MODES_TEXT = "U=8,p=8,phi=8"

#: acceptance thresholds of plan 3.1 (never relaxed; failures are reported)
U_ERROR_TOL = 1e-2
FREQUENCY_TOL = 1e-2
FORCE_TOL = 2e-2
ENERGY_DRIFT_TOL = 0.05
NORM_RATIO_TOL = 2.0

#: latent norm ratio the extrapolated long-horizon check counts as divergence
LONG_HORIZON_BLOWUP = 10.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--methods", default="dmd", help="comma separated, e.g. dmd")
    p.add_argument("--train-periods", type=float, default=3.0)
    p.add_argument(
        "--n-modes",
        default=DEFAULT_N_MODES_TEXT,
        metavar="FIELD=INT",
        help="per-field modes, e.g. U=8,p=8,phi=8; 'auto' uses --energy-tol",
    )
    p.add_argument("--energy-tol", type=float, default=1e-10)
    p.add_argument("--readout-ridge", type=float, default=1e-10)
    p.add_argument("--sweep-modes", default=None, help="sweep the U latent dimension")
    p.add_argument("--sweep-train-periods", default=None, help="sweep the training window")
    p.add_argument(
        "--write-fields",
        type=int,
        default=None,
        metavar="N",
        help="write the ROM fields of every N-th prediction snapshot to OpenFOAM "
        "time directories under cases/<problem>_<study>_rom_<method>/",
    )
    p.add_argument("--data-dir", type=Path, default=None, help="data root (default: settings)")
    p.add_argument(
        "--long-periods",
        type=float,
        default=20.0,
        metavar="N",
        help="periods beyond the training window used by the extrapolated latent "
        "stability check (0 skips it)",
    )
    p.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="skip the scalar readout order-1 diagnostics (one extra fit per method)",
    )
    args = p.parse_args()

    methods = _list(args.methods)
    if not methods:
        p.error("--methods must name at least one of " + ", ".join(METHODS))
    unknown = [name for name in methods if name.lower() not in METHODS]
    if unknown:
        p.error(
            f"unknown method(s) {', '.join(unknown)} in --methods "
            f"(choose from {', '.join(METHODS)})"
        )
    not_ready = [name for name in methods if name.lower() not in IMPLEMENTED]
    if not_ready:
        p.error(
            f"method(s) {', '.join(not_ready)} are not implemented yet "
            f"(milestone M3/M4); only {', '.join(IMPLEMENTED)} is available"
        )
    if not args.train_periods > 0.0:
        p.error("--train-periods must be positive")
    if args.write_fields is not None and args.write_fields < 1:
        p.error("--write-fields must be a positive integer")
    if not args.long_periods >= 0.0:
        p.error("--long-periods must be non-negative (0 skips the check)")
    args.diagnostics = not args.no_diagnostics
    args.methods = [name.lower() for name in methods]
    args.n_modes = _parse_n_modes(args.n_modes)
    args.sweep_modes = _parse_ints(args.sweep_modes, "--sweep-modes")
    args.sweep_train_periods = _parse_floats(
        args.sweep_train_periods, "--sweep-train-periods"
    )
    return args


def main() -> None:
    args = parse_args()
    settings = get_settings()
    root = settings.data_dir if args.data_dir is None else Path(args.data_dir)
    study = validate_study(args.study, problem=PROBLEM, settings=settings)
    data_dir = root / PROBLEM / study
    npz_path = data_dir / "snapshots.npz"
    meta_path = data_dir / "meta.json"
    report_path = data_dir / "report.json"
    if not npz_path.is_file():
        raise SystemExit(
            f"missing snapshots in {data_dir}: run scripts/generate_transient.py first"
        )
    meta = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    times, forces, volumes, phi_w, fields_all = _load(npz_path)
    re_value = meta.get("config", {}).get("re", meta.get("re"))

    try:
        f_fom = dominant_frequency(times, forces[:, 1])
    except ValueError as exc:
        raise SystemExit(f"cannot estimate the FOM shedding frequency: {exc}") from exc
    period = 1.0 / f_fom
    try:
        summary = shedding_summary(times, forces[:, 1], forces[:, 0])
    except ValueError as exc:
        print(f"WARNING: shedding summary skipped ({exc})")
        summary = None

    dt = float(times[1] - times[0]) if times.size > 1 else None
    print(
        f"{PROBLEM}/{study}: {times.size} snapshots t=[{times[0]:g}, {times[-1]:g}], "
        f"dt={dt:g}, Re={re_value}"
    )
    if summary is not None:
        limit = summary["limit_cycle_time"]
        print(
            "FOM shedding: St {:.4f}, period {:.4f}, mean Cd {:.4f}, "
            "Cl amplitude {:.4f}, limit cycle {}".format(
                summary["strouhal"],
                summary["period"],
                summary["cd_mean"],
                summary["cl_amplitude"],
                "n/a" if limit is None else f"{limit:.1f}",
            )
        )
    print(
        f"energy_tol {args.energy_tol:g}, n_modes {_n_modes_text(args.n_modes)}, "
        f"readout ridge {args.readout_ridge:g}"
    )

    n_train, window = _split(times, args.train_periods, period)
    if window["n_test"] == 0:
        raise SystemExit("the training window covers every snapshot; reduce --train-periods")
    print(
        f"training window: {window['n_train']} snapshots t<={window['cutoff']:.4f}, "
        f"t_train_end={window['t_train_end']:.4f} "
        f"({window['train_periods']:.2f} periods); prediction window "
        f"{window['n_test']} snapshots from t={window['t_test_start']:.4f} "
        f"({window['test_periods']:.2f} periods)"
    )

    topo, geom = _mesh(Cases(settings), f"{PROBLEM}_{study}")

    rows = []
    models = {}
    for method in args.methods:
        rom, result, cost = _fit_and_evaluate(
            method,
            args.n_modes,
            times,
            fields_all,
            n_train,
            forces,
            volumes,
            phi_w,
            period=period,
            f_fom=f_fom,
            topo=topo,
            geom=geom,
            energy_tol=args.energy_tol,
            readout_ridge=args.readout_ridge,
        )
        rom.save(data_dir / "models" / f"{method}.pkl")
        models[method] = rom
        fit_info = _fit_info(rom)
        row = {
            "method": method,
            "result": result,
            "cost": cost,
            "n_modes": dict(rom.n_modes_),
            "acceptance": _acceptance(result, fit_info, method),
            "fit_info": fit_info,
        }
        if args.diagnostics:
            row["diagnostics"] = {
                "readout_order1": _readout_diagnostic(
                    method,
                    args.n_modes,
                    times,
                    fields_all,
                    n_train,
                    forces,
                    volumes,
                    phi_w,
                    period=period,
                    f_fom=f_fom,
                    energy_tol=args.energy_tol,
                    readout_ridge=args.readout_ridge,
                )
            }
        if args.long_periods > 0.0 and dt is not None:
            row["long_horizon"] = _long_horizon(
                rom, float(times[n_train - 1]), dt, period, f_fom, args.long_periods
            )
        if args.write_fields is not None:
            if topo is None:
                raise SystemExit(
                    "--write-fields needs the FOM mesh in "
                    f"cases/{PROBLEM}_{study}/run_0000"
                )
            indices = write_indices(n_train, times.size, args.write_fields)
            predicted = rom.predict(times[indices])
            row["fields"] = _write_rom_fields(
                Cases(settings),
                study,
                method,
                times,
                fields_all,
                {name: predicted[name] for name in FIELDS},
                indices,
                args.write_fields,
                topo,
            )
        rows.append(row)

        print(
            f"fitted {method}: {cost['fit_seconds']:.2f} s, "
            f"predict {cost['predict_seconds']:.2f} s, "
            f"evaluate {cost['evaluate_seconds']:.2f} s",
            flush=True,
        )

    _print_metrics(rows)
    _print_fit_convergence(rows)
    _print_diagnostics(rows)
    _print_long_horizon(rows, args.long_periods)
    _print_acceptance(rows)

    previous = _previous_sweeps(report_path)
    sweeps = {"modes": {}, "train_periods": {}}
    if previous is not None:
        sweeps = {kind: dict(grouped) for kind, grouped in previous.items()}
    rerun: dict[str, list[str]] = {}
    if args.sweep_modes:
        rows_modes = _sweep_modes(
            args, times, fields_all, forces, volumes, phi_w, period, f_fom, n_train
        )
        sweeps["modes"].update(_group_by_method(rows_modes))
        rerun["modes"] = list(args.methods)
        _print_sweep("latent-dimension sweep (U modes)", rows_modes, "n_modes_U")
    if args.sweep_train_periods:
        rows_periods = _sweep_train_periods(
            args, times, fields_all, forces, volumes, phi_w, period, f_fom
        )
        sweeps["train_periods"].update(_group_by_method(rows_periods))
        rerun["train_periods"] = list(args.methods)
        _print_sweep(
            "training-window sweep (periods)", rows_periods, "train_periods"
        )
    note = _sweep_note(previous, rerun)
    if note:
        print(f"\n{note}")

    label = (
        npz_path.relative_to(PROJECT_ROOT).as_posix()
        if npz_path.is_relative_to(PROJECT_ROOT)
        else str(npz_path)
    )
    report = {
        "problem": PROBLEM,
        "study": study,
        "snapshots": label,
        "fom": {
            "frequency": f_fom,
            "period": period,
            "shedding": summary,
            "meta_shedding": meta.get("shedding"),
            "re": re_value,
            "dt": meta.get("dt", dt),
            "n_snapshots": int(times.size),
            "max_courant": meta.get("max_courant"),
        },
        "config": {
            "methods": list(args.methods),
            "train_periods": args.train_periods,
            "n_modes": args.n_modes,
            "energy_tol": args.energy_tol,
            "readout_ridge": args.readout_ridge,
            "sweep_modes": args.sweep_modes,
            "sweep_train_periods": args.sweep_train_periods,
            "write_fields": args.write_fields,
            "long_periods": args.long_periods,
            "diagnostics": args.diagnostics,
        },
        "training_window": window,
        "mesh": {"topology": topo is not None, "geometry": geom is not None},
        "methods": {row["method"]: _method_entry(row) for row in rows},
        "sweeps": sweeps,
    }
    report_path = data_dir / "report.json"
    report_path.write_text(
        json.dumps(to_jsonable(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nreport: {report_path}")
    print(f"models: {data_dir / 'models'}")


# -- fitting and evaluation -----------------------------------------------------------------


def _method_dynamics(method: str) -> tuple[str, dict]:
    """Dynamics kind and ``dynamics_kw`` of a ``--methods`` name."""
    try:
        kind, kwargs = METHOD_DYNAMICS[method]
    except KeyError:
        raise ValueError(
            f"unknown method {method!r} (choose from {', '.join(METHODS)})"
        ) from None
    return kind, dict(kwargs)


def _fit_and_evaluate(
    method: str,
    n_modes: dict[str, int | None],
    times: np.ndarray,
    fields_all: dict[str, np.ndarray],
    n_train: int,
    forces: np.ndarray,
    volumes: np.ndarray,
    phi_w: np.ndarray,
    *,
    period: float,
    f_fom: float,
    topo,
    geom,
    energy_tol: float,
    readout_ridge: float,
    eval_fields: tuple[str, ...] | None = None,
    readout_order: dict[str, int] | None = None,
) -> tuple[TransientROM, dict, dict]:
    """Fit on the prefix of ``n_train`` snapshots and evaluate the rest.

    ``eval_fields`` restricts the reference fields (and the report) to a subset;
    the sweeps use it to skip the expensive ``p``/``phi`` comparisons.
    """
    t_train, t_test = times[:n_train], times[n_train:]
    snap_train = {name: values[:n_train] for name, values in fields_all.items()}
    snap_test = {
        name: values[n_train:]
        for name, values in fields_all.items()
        if eval_fields is None or name in eval_fields
    }
    mean_train = {
        name: np.asarray(values[:n_train].mean(axis=0), dtype=float)
        for name, values in fields_all.items()
        if name in ("U", "p")
    }
    dynamics, dynamics_kw = _method_dynamics(method)
    rom = TransientROM(
        dynamics=dynamics,
        n_modes=n_modes,
        energy_tol=energy_tol,
        readout_order=readout_order,
        readout_ridge=readout_ridge,
        dynamics_kw=dynamics_kw,
    )
    started = time.perf_counter()
    rom.fit(
        t_train,
        snap_train,
        volumes,
        phi_weights=phi_w,
        scalars={"forces": forces[:n_train]},
    )
    fit_s = time.perf_counter() - started

    # timed separately from the evaluation, which predicts internally as well
    started = time.perf_counter()
    predicted = rom.predict(t_test)
    predict_s = time.perf_counter() - started
    del predicted

    started = time.perf_counter()
    result = evaluate_transient(
        rom,
        t_test,
        snap_test,
        volumes,
        phi_w,
        period=period,
        forces_test=forces[n_train:],
        mean_train=mean_train,
        f_fom=f_fom,
        topo=topo,
        geom=geom,
    )
    evaluate_s = time.perf_counter() - started
    return rom, result, {
        "fit_seconds": fit_s,
        "predict_seconds": predict_s,
        "evaluate_seconds": evaluate_s,
    }


def _scalar_readout_features(rank: int, order: int) -> int:
    """Columns of the scalar readout design matrix (bias, linear, quadratic)."""
    features = 1 + int(rank)
    if order == 2:
        features += rank * (rank + 1) // 2
    return features


def _readout_diagnostic(
    method: str,
    n_modes: dict[str, int | None],
    times: np.ndarray,
    fields_all: dict[str, np.ndarray],
    n_train: int,
    forces: np.ndarray,
    volumes: np.ndarray,
    phi_w: np.ndarray,
    *,
    period: float,
    f_fom: float,
    energy_tol: float,
    readout_ridge: float,
) -> dict:
    """Fit the same model with an order-1 scalar readout and compare the forces.

    The latent dynamics and the POD bases are those of the formal fit; only the
    scalar readout drops from order 2 (the default) to order 1.  That separates
    the readout error from the dynamics error of Cl/Cd (the quadratic readout
    amplifies small latent deviations).  Only the forces part of the evaluation
    is kept, and no model file is written.
    """
    rom, result, _ = _fit_and_evaluate(
        method,
        n_modes,
        times,
        fields_all,
        n_train,
        forces,
        volumes,
        phi_w,
        period=period,
        f_fom=f_fom,
        topo=None,
        geom=None,
        energy_tol=energy_tol,
        readout_ridge=readout_ridge,
        eval_fields=("U",),
        readout_order={"scalars": 1},
    )
    rank = int(rom.n_modes_["U"])
    scalars = _forces(result) or {}
    return {
        "readout_order": 1,
        "scalar_readout_features": {
            "order2": _scalar_readout_features(rank, 2),
            "order1": _scalar_readout_features(rank, 1),
        },
        "n_train": int(n_train),
        "cl_amplitude": scalars.get("cl_amplitude"),
        "cd_mean": scalars.get("cd_mean"),
        "cl_phase_deg": scalars.get("cl_phase_deg"),
        "attribution": scalars.get("attribution"),
    }


def _long_horizon(
    rom,
    t_train_end: float,
    dt: float,
    period: float,
    f_fom: float,
    periods: float,
    blowup: float = LONG_HORIZON_BLOWUP,
) -> dict:
    """Latent stability of a ROM extrapolated ``periods`` periods past the data.

    The times run from ``t_train_end + dt`` to ``t_train_end + periods * period``
    with the snapshot step ``dt`` (uniform, so the frequency estimate is
    meaningful); there is no FOM reference there, so only the latent norm, the
    cycle-averaged energy and the dominant frequency are reported.  ``diverged``
    covers non-finite coefficients and a norm above ``blowup`` times the largest
    training norm; ``diverged_at`` is the first such time.
    """
    n = max(2, int(round(periods * period / dt)))
    times = np.linspace(t_train_end + dt, t_train_end + periods * period, n)
    a = np.asarray(rom.latent(times), dtype=float)
    train_max = float(np.max(np.linalg.norm(rom.latent_train_, axis=1)))
    norms = np.linalg.norm(a, axis=1)
    finite = np.isfinite(norms)
    ratio = (
        float(np.max(norms[finite]) / train_max) if finite.any() and train_max > 0.0 else None
    )
    bad = ~finite
    if train_max > 0.0:
        bad = bad | (norms > blowup * train_max)
    diverged = bool(np.any(bad))
    diverged_at = float(times[int(np.argmax(bad))]) if diverged else None

    energy = None
    if np.all(np.isfinite(a)):
        try:
            drift = energy_drift(times, a, period, train_max)
        except ValueError:
            drift = None
        if drift is not None:
            energy = {
                "n_cycles": int(drift["n_cycles"]),
                "first_cycle": _finite_or_none(drift["first_cycle"]),
                "last_cycle": _finite_or_none(drift["last_cycle"]),
                "relative_change": _finite_or_none(drift["relative_change"]),
                "max_norm": _finite_or_none(drift["max_norm"]),
            }

    f_rom = _latent_frequency_safe(times, a)
    return {
        "periods": float(periods),
        "period": float(period),
        "t_start": float(times[0]),
        "t_end": float(times[-1]),
        "n_points": int(times.size),
        "blowup_factor": float(blowup),
        "train_max_norm": train_max,
        "max_norm_ratio": ratio,
        "diverged": diverged,
        "diverged_at": diverged_at,
        "energy": energy,
        "frequency": {
            "fom": None if f_fom is None else float(f_fom),
            "rom": f_rom,
            "relative_error": (
                None
                if f_fom is None or f_rom is None
                else abs(f_rom - f_fom) / abs(f_fom)
            ),
        },
    }


def _latent_frequency_safe(t: np.ndarray, a: np.ndarray) -> float | None:
    """Dominant frequency of the most energetic latent column (``None`` on NaN)."""
    if not np.all(np.isfinite(a)):
        return None
    centred = a - np.mean(a, axis=0)
    norms = np.linalg.norm(centred, axis=0)
    if not np.any(norms > 0.0):
        return None
    try:
        return dominant_frequency(t, a[:, int(np.argmax(norms))])
    except ValueError:
        return None


def _finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


# -- writing the ROM fields ----------------------------------------------------------------


def write_indices(n_train: int, n_total: int, every: int) -> list[int]:
    """Snapshot indices of the prediction window, one every ``every`` snapshots."""
    if every < 1:
        raise ValueError(f"every must be a positive integer, got {every}")
    if not 0 <= n_train <= n_total:
        raise ValueError(f"training window {n_train} is outside {n_total} snapshots")
    return list(range(n_train, n_total, every))


def time_name(value: float) -> str:
    """OpenFOAM time-directory name (``timeFormat general``, ``timePrecision 6``)."""
    return f"{float(value):.6g}"


def field_difference(pred: np.ndarray, fom: np.ndarray) -> np.ndarray:
    """Cell-wise ROM - FOM difference; the shapes must agree."""
    pred = np.asarray(pred, dtype=float)
    fom = np.asarray(fom, dtype=float)
    if pred.shape != fom.shape:
        raise ValueError(
            f"prediction shape {pred.shape} does not match the FOM shape {fom.shape}"
        )
    return pred - fom


def _write_rom_fields(
    cases: Cases,
    study: str,
    method: str,
    times: np.ndarray,
    fields_all: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    indices: list[int],
    every: int,
    topo,
) -> dict:
    """Write the predicted snapshots of one method as OpenFOAM time directories.

    The FOM case provides ``constant/``, ``system/`` and the field templates; the FOM
    truth of the difference fields comes from ``snapshots.npz`` (``fields_all``), not
    from the case.
    """
    case = f"{PROBLEM}_{study}_rom_{method}/run_0000"
    dest = cases.path(case, must_exist=False)
    _check_case_path(cases, case, dest)
    fom_case_name = f"{PROBLEM}_{study}/run_0000"
    try:
        fom_case = cases.path(fom_case_name)
    except FileNotFoundError:
        raise SystemExit(
            f"cannot write ROM fields: the FOM case {fom_case_name} is missing; "
            "--write-fields copies its constant/, system/ and .foam file and reads the "
            "field templates from its time directories"
        ) from None

    _prepare_rom_case(cases, case, dest, fom_case)
    written, skipped = [], []
    seen = set()
    for k, index in enumerate(indices):
        name = time_name(times[index])
        if name in seen:
            raise ValueError(
                f"snapshot times {times[index]:.12g} and another selected time both "
                f"format to the time directory {name!r}"
            )
        seen.add(name)
        if not (fom_case / name).is_dir():
            skipped.append(name)
            continue
        templates = {field: read_field(fom_case, name, field) for field in FIELDS}
        pred = {field: predictions[field][k] for field in FIELDS}
        fom = {field: fields_all[field][index] for field in ("U", "p")}
        write_rom_time(dest, name, pred, fom, templates, topo)
        written.append(name)

    if skipped:
        shown = ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else "")
        print(
            f"WARNING: skipped {len(skipped)} of {len(indices)} time directories "
            f"without a FOM counterpart ({shown})"
        )
    if written:
        print(
            f"wrote {len(written)} ROM time directories to {case} "
            f"(every {every} snapshots, t=[{written[0]}, {written[-1]}])"
        )
    else:
        print(
            f"WARNING: no ROM time directories written to {case}: none of the selected "
            f"times has a FOM time directory in {fom_case_name}"
        )
    return {
        "case": case,
        "every": every,
        "n_selected": len(indices),
        "n_written": len(written),
        "n_skipped": len(skipped),
        "times": written,
        "skipped": skipped,
    }


def _prepare_rom_case(cases: Cases, case: str, dest: Path, fom_case: Path) -> None:
    """Clear stale time directories, then copy the mesh, dictionaries and .foam file."""
    if dest.is_dir():
        _clear_time_dirs(dest)
    else:
        dest.mkdir(parents=True)
    for sub in ("constant", "system"):
        shutil.copytree(fom_case / sub, dest / sub, dirs_exist_ok=True)
    foam_name = f"{Path(case).name}.foam"
    if (fom_case / foam_name).is_file():
        shutil.copy2(fom_case / foam_name, dest / foam_name)
    elif not (dest / foam_name).is_file():
        cases.write(case, foam_name, "")


def _clear_time_dirs(case_dir: Path) -> list[str]:
    """Remove only the time directories (numeric names) inside a ROM case."""
    removed = []
    for path in sorted(case_dir.iterdir()):
        if path.is_dir() and TIME_DIR_RE.match(path.name):
            shutil.rmtree(path)
            removed.append(path.name)
    return removed


def write_rom_time(case_dir: Path, name: str, pred: dict, fom: dict, templates: dict, topo):
    """Write the ROM fields, the FOM truth and the differences of one snapshot."""
    for field in FIELDS:
        template = templates[field]
        values = np.asarray(pred[field], dtype=float)
        if field == "phi":
            foam_field = _split_phi(values, topo, template)
        else:
            foam_field = FoamField(
                field,
                template.cls,
                values,
                _boundary(template),
                template.dimensions,
            )
        write_field(case_dir, name, foam_field, template=template)

    for field in ("U", "p"):
        template = templates[field]
        boundary = _boundary(template)
        values = (
            ("fom", np.asarray(fom[field], dtype=float)),
            ("err", field_difference(pred[field], fom[field])),
        )
        for suffix, data in values:
            write_field(
                case_dir,
                name,
                FoamField(
                    f"{field}_{suffix}",
                    template.cls,
                    data,
                    boundary,
                    template.dimensions,
                ),
                template=template,
            )


def _boundary(template: FoamField) -> dict[str, dict]:
    return {
        name: {"type": entry.get("type") or "patch", "value": entry.get("value")}
        for name, entry in template.boundary.items()
    }


def _split_phi(phi: np.ndarray, topo, template: FoamField) -> FoamField:
    """Undo :func:`phi_vector`: internal faces first, then the patches with a value."""
    phi = np.asarray(phi, dtype=float).reshape(-1)
    pos = topo.neighbour.size
    boundary = {}
    for name, (_, _, n) in topo.patches.items():
        entry = template.boundary.get(name)
        value = None if entry is None else entry.get("value")
        if value is None or np.size(value) == 0 or n == 0:
            continue
        boundary[name] = {"type": entry["type"], "value": np.array(phi[pos : pos + n])}
        pos += n
    if pos != phi.size:
        raise ValueError(f"phi has {phi.size - pos} values not covered by the patches")
    return FoamField(
        "phi", template.cls, phi[: topo.neighbour.size], boundary, template.dimensions
    )


# -- sweeps ---------------------------------------------------------------------------------


def _sweep_modes(args, times, fields_all, forces, volumes, phi_w, period, f_fom, n_train):
    rows = []
    for method in args.methods:
        for n_u in args.sweep_modes:
            window = _window_info(times, n_train, period)
            row = {
                "method": method,
                "n_modes_U": n_u,
                "n_modes": {**args.n_modes, "U": n_u},
                "n_train": window["n_train"],
                "t_train_end": window["t_train_end"],
                "n_test": window["n_test"],
            }
            _run_sweep_row(
                row,
                method,
                {**args.n_modes, "U": n_u},
                times,
                fields_all,
                forces,
                volumes,
                phi_w,
                n_train,
                window,
                args,
                period,
                f_fom,
            )
            rows.append(row)
            print(f"sweep U modes {n_u} ({method}): {row['status']}", flush=True)
    return rows


def _sweep_train_periods(args, times, fields_all, forces, volumes, phi_w, period, f_fom):
    rows = []
    for method in args.methods:
        for n_periods in args.sweep_train_periods:
            n_train, window = _split(times, n_periods, period)
            row = {
                "method": method,
                "train_periods": n_periods,
                "n_train": window["n_train"],
                "t_train_end": window["t_train_end"],
                "n_test": window["n_test"],
            }
            _run_sweep_row(
                row,
                method,
                args.n_modes,
                times,
                fields_all,
                forces,
                volumes,
                phi_w,
                n_train,
                window,
                args,
                period,
                f_fom,
            )
            rows.append(row)
            print(f"sweep train periods {n_periods:g} ({method}): {row['status']}", flush=True)
    return rows


def _run_sweep_row(
    row, method, n_modes, times, fields_all, forces, volumes, phi_w, n_train, window,
    args, period, f_fom,
):
    if window["n_train"] < 3:
        row["status"] = "skipped"
        row["reason"] = f"only {window['n_train']} training snapshots (DMD needs three)"
        return
    if window["test_periods"] is None or window["test_periods"] < 5.0 - 1e-9:
        row["status"] = "skipped"
        row["reason"] = (
            f"only {window['test_periods'] or 0.0:.2f} periods after the training window "
            "(needs five)"
        )
        return
    try:
        rom, result, cost = _fit_and_evaluate(
            method,
            n_modes,
            times,
            fields_all,
            n_train,
            forces,
            volumes,
            phi_w,
            period=period,
            f_fom=f_fom,
            topo=None,
            geom=None,
            energy_tol=args.energy_tol,
            readout_ridge=args.readout_ridge,
            eval_fields=("U",),
        )
    except ValueError as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
        return
    forces_report = _forces(result)
    info = _fit_info(rom)
    row.update(
        {
            "status": "ok",
            "max_u_error_within_tau5": result["max_error_within_tau"]["U"],
            "u_error_at_tau5": _tau_value(result, "U", 5.0),
            "cl_amplitude_relative_error": (
                None if forces_report is None
                else forces_report["cl_amplitude"]["relative_error"]
            ),
            "cd_mean_relative_error": (
                None if forces_report is None
                else forces_report["cd_mean"]["relative_error"]
            ),
            "frequency_relative_error": result["frequency"]["relative_error"],
            "n_modes_effective": dict(rom.n_modes_),
            "diverged": info.get("diverged"),
            "ridge": info.get("ridge"),
            "cost": cost,
        }
    )
    del rom


# -- terminal output ------------------------------------------------------------------------


def _print_metrics(rows: list[dict]) -> None:
    print("\nprediction accuracy (relative L2 unless noted)")
    body = []
    for row in rows:
        result = row["result"]
        forces_report = _forces(result)
        body.append(
            [
                row["method"],
                _sci(_tau_value(result, "U", 1.0)),
                _sci(_tau_value(result, "U", 3.0)),
                _sci(_tau_value(result, "U", 5.0)),
                _sci(_tau_value(result, "U", 10.0)),
                _sci(result["max_error_within_tau"].get("U")),
                _sci(result["max_error_within_tau"].get("U_fluct")),
                _sci(result["projection_max_within_tau"].get("U")),
                _sci(result["frequency"]["relative_error"]),
                _sci(None if forces_report is None
                     else forces_report["cl_amplitude"]["relative_error"]),
                _sci(None if forces_report is None
                     else forces_report["cd_mean"]["relative_error"]),
                _deg(_phase(result, 1.0)),
                _deg(_phase(result, 5.0)),
                _deg(_phase(result, 10.0)),
            ]
        )
    _table(
        [
            "method",
            "U tau1",
            "U tau3",
            "U tau5",
            "U tau10",
            "U max tau<=5",
            "U fluct max tau<=5",
            "proj max tau<=5",
            "freq err",
            "Cl amp err",
            "Cd mean err",
            "phase1 deg",
            "phase5 deg",
            "phase10 deg",
        ],
        body,
    )

    print("\nstability and cost")
    body = []
    for row in rows:
        result = row["result"]
        rom_energy = _energy(result, "rom")
        fom_energy = _energy(result, "fom")
        divergence = result.get("phi_divergence")
        body.append(
            [
                row["method"],
                _pct(None if rom_energy is None else rom_energy["relative_change"]),
                _pct(None if fom_energy is None else fom_energy["relative_change"]),
                _ratio(None if rom_energy is None else rom_energy["max_norm_ratio"]),
                "n/a" if divergence is None
                else ("ok" if divergence["ok"] else "FAILED"),
                f"{row['cost']['fit_seconds']:.2f}",
                f"{row['cost']['predict_seconds']:.2f}",
                f"{row['cost']['evaluate_seconds']:.2f}",
            ]
        )
    _table(
        [
            "method",
            "E drift ROM",
            "E drift FOM",
            "max|a|/train",
            "phi div",
            "fit s",
            "predict s",
            "eval s",
        ],
        body,
    )


def _print_fit_convergence(rows: list[dict]) -> None:
    print("\nfit convergence (latent dynamics)")
    body = []
    for row in rows:
        info = row.get("fit_info") or {}
        success = info.get("success")
        body.append(
            [
                row["method"],
                _count(info.get("rank")),
                _count(info.get("nfev")),
                "n/a" if success is None else ("yes" if success else "no"),
                _count(info.get("n_used")),
                _count(info.get("n_rejected")),
                "n/a" if info.get("ridge") is None else _sci(info["ridge"]),
                _yes_no(info.get("refit_fallback")),
                _diverged_text(info),
            ]
        )
    _table(
        [
            "method",
            "rank",
            "nfev",
            "success",
            "n used",
            "n rejected",
            "ridge",
            "fallback",
            "diverged",
        ],
        body,
    )
    for row in rows:
        info = row.get("fit_info") or {}
        if info.get("success") is False:
            print(
                f"WARNING: the {row['method']} fit did not converge "
                f"(success=False, nfev={_count(info.get('nfev'))}); "
                "its frequencies and amplitudes may be unreliable"
            )
        if info.get("diverged"):
            at = info.get("diverged_at")
            where = "" if at is None else f" at t={at:g}"
            print(
                f"WARNING: the {row['method']} OpInf trajectory diverged{where}; "
                "its latent state is NaN from there on, so the accuracy metrics "
                "and the acceptance checks fail"
            )


def _yes_no(value) -> str:
    return "n/a" if value is None else ("yes" if value else "no")


def _diverged_text(info: dict) -> str:
    if info.get("diverged") is None:
        return "n/a"
    if not info["diverged"]:
        return "no"
    at = info.get("diverged_at")
    return "yes" if at is None else f"yes (t={at:g})"


def _print_acceptance(rows: list[dict]) -> None:
    print("\nacceptance (plan 3.1)")
    for row in rows:
        acceptance = row["acceptance"]
        checks = acceptance["checks"]
        print(f"  {row['method']}: {acceptance['verdict']}")
        for check in checks:
            status = {True: "PASS", False: "FAIL", None: "SKIP"}[check["pass"]]
            limit = "" if check["threshold"] is None else f" (limit {_sci(check['threshold'])})"
            reason = "" if check.get("reason") is None else f" - {check['reason']}"
            print(
                f"    [{status}] {check['name']}: "
                f"{_check_text(check['value'])}{limit}{reason}"
            )
        skipped = [check for check in checks if check["pass"] is None]
        if skipped:
            print(
                "    skipped: "
                + "; ".join(
                    check["name"]
                    + ("" if check.get("reason") is None else f" - {check['reason']}")
                    for check in skipped
                )
            )


def _print_sweep(title: str, rows: list[dict], column: str) -> None:
    print(f"\n{title}")
    body = []
    for row in rows:
        status = row["status"]
        if status != "ok":
            status = f"{status}: {row.get('reason', row.get('error', ''))}"
        body.append(
            [
                row["method"],
                f"{row[column]:g}",
                str(row["n_train"]),
                f"{row['t_train_end']:.2f}",
                _sci(row.get("max_u_error_within_tau5")),
                _sci(row.get("u_error_at_tau5")),
                _sci(row.get("cl_amplitude_relative_error")),
                _sci(row.get("cd_mean_relative_error")),
                _sweep_diverged_text(row.get("diverged")),
                _sci(row.get("ridge")),
                status,
            ]
        )
    _table(
        [
            "method",
            column,
            "n train",
            "t train end",
            "max U err tau<=5",
            "U err tau5",
            "Cl amp err",
            "Cd mean err",
            "diverged",
            "ridge",
            "status",
        ],
        body,
    )


def _sweep_diverged_text(value) -> str:
    return "n/a" if value is None else ("yes" if value else "no")


def _print_diagnostics(rows: list[dict]) -> None:
    present = [row for row in rows if row.get("diagnostics")]
    if not present:
        return
    print("\ndiagnostic: scalar readout order 1 (not part of acceptance)")
    body = []
    for row in present:
        diag = (row["diagnostics"].get("readout_order1") or {})
        formal = _forces(row["result"])
        features = diag.get("scalar_readout_features") or {}
        n_train = diag.get("n_train")
        body.append(
            [
                row["method"],
                _both(_cl_error(formal), _cl_error(diag)),
                _both(_cd_error(formal), _cd_error(diag)),
                _both_deg(_phase_error(formal), _phase_error(diag)),
                "n/a"
                if not features or n_train is None
                else f"{features['order2']}/{features['order1']} / {int(n_train)}",
            ]
        )
    _table(
        [
            "method",
            "Cl amp err (ord2 / ord1)",
            "Cd mean err (ord2 / ord1)",
            "Cl phase tau5 deg (ord2 / ord1)",
            "features o2/o1 / n_train",
        ],
        body,
    )


def _print_long_horizon(rows: list[dict], periods: float) -> None:
    present = [row for row in rows if row.get("long_horizon")]
    if not present:
        return
    print(
        f"\nlong-horizon stability ({periods:g} periods beyond the training window; "
        "extrapolated beyond the data, no FOM reference)"
    )
    body = []
    for row in present:
        info = row["long_horizon"]
        energy = info.get("energy") or {}
        frequency = info.get("frequency") or {}
        body.append(
            [
                row["method"],
                _ratio(info.get("max_norm_ratio")),
                _horizon_diverged_text(info),
                _pct(energy.get("relative_change")),
                _sci(frequency.get("relative_error")),
            ]
        )
    _table(
        ["method", "max|a|/train", "diverged", "E drift first->last", "latent freq err"],
        body,
    )
    for row in present:
        info = row["long_horizon"]
        if info.get("diverged"):
            at = info.get("diverged_at")
            where = "" if at is None else f" at t={at:g}"
            print(
                f"NOTE: the {row['method']} long-horizon trajectory diverged{where} "
                "(extrapolated beyond the data, no FOM reference)"
            )


def _horizon_diverged_text(info: dict) -> str:
    if not info.get("diverged"):
        return "no"
    at = info.get("diverged_at")
    return "yes" if at is None else f"yes (t={at:g})"


def _cl_error(report: dict | None):
    return None if not report else (report.get("cl_amplitude") or {}).get("relative_error")


def _cd_error(report: dict | None):
    return None if not report else (report.get("cd_mean") or {}).get("relative_error")


def _phase_error(report: dict | None):
    return None if not report else (report.get("cl_phase_deg") or {}).get("5")


def _both(first, second) -> str:
    return f"{_sci(first)} / {_sci(second)}"


def _both_deg(first, second) -> str:
    return f"{_deg(first)} / {_deg(second)}"


# -- report pieces --------------------------------------------------------------------------


def _method_entry(row: dict) -> dict:
    """One ``methods.<name>`` block; the optional extras appear only when computed."""
    entry = {
        "model": f"models/{row['method']}.pkl",
        "n_modes": row["n_modes"],
        "cost": row["cost"],
        "fit_info": row["fit_info"],
        "acceptance": row["acceptance"],
        "evaluation": row["result"],
        "fields": row.get("fields"),
    }
    for key in ("diagnostics", "long_horizon"):
        if key in row:
            entry[key] = row[key]
    return entry


def _previous_sweeps(report_path: Path) -> dict | None:
    """The ``sweeps`` of an existing report, grouped by scan type and method.

    Older reports kept one flat list of rows per scan type (each row naming its
    ``method``); those lists are regrouped per method, so a run that sweeps only
    some methods keeps the rows of the others.
    """
    if not report_path.is_file():
        return None
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    sweeps = data.get("sweeps")
    if not isinstance(sweeps, dict):
        return None
    grouped = {"modes": {}, "train_periods": {}}
    for kind in grouped:
        value = sweeps.get(kind)
        if isinstance(value, list):
            grouped[kind] = _group_by_method(value)
        elif isinstance(value, dict):
            grouped[kind] = {
                str(method): list(rows)
                for method, rows in value.items()
                if isinstance(rows, list) and rows
            }
    if not (grouped["modes"] or grouped["train_periods"]):
        return None
    return grouped


def _group_by_method(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        method = row.get("method")
        if method is None:
            continue
        grouped.setdefault(str(method), []).append(row)
    return grouped


def _sweep_note(previous: dict | None, rerun: dict[str, list[str]]) -> str | None:
    """What was carried over from the old report: scans/methods not rerun now."""
    if previous is None:
        return None
    kept = {
        kind: sorted(
            method for method in grouped if method not in rerun.get(kind, ())
        )
        for kind, grouped in previous.items()
    }
    parts = [
        f"{kind}: {', '.join(methods)}"
        for kind, methods in kept.items()
        if methods
    ]
    if not parts:
        return None
    return "kept the sweeps of the previous report for " + "; ".join(parts)


def _fit_info(rom) -> dict:
    """Convergence and regularisation data of the latent dynamics.

    DMD is a direct linear solve, so it only reports the effective rank
    (number of retained eigenvalues); the bagged BOP-DMD reports the reference
    full-data fit's optimiser counters.  OpInf rows add the ridge selection, the
    post-refit stability check and the divergence flag; attributes that a fit
    does not have (e.g. ``recheck_path_`` with ``recheck=False``) stay ``None``.
    """
    info = dict.fromkeys(("rank", "nfev", "success", "n_used", "n_rejected"))
    model = getattr(rom, "model_", None)
    if model is None:
        return info
    eigenvalues = getattr(model, "lambda_", None)
    if eigenvalues is None:
        eigenvalues = getattr(model, "omega_", None)
    info["rank"] = None if eigenvalues is None else int(np.size(eigenvalues))
    if hasattr(model, "recheck"):
        # OpInf: the rank is the latent dimension, not an eigenvalue count
        info["rank"] = _as_int(getattr(model, "r_", None))
    info["nfev"] = _as_int(_first_not_none(model, "nfev_", "reference_nfev_"))
    info["success"] = _as_bool(_first_not_none(model, "success_", "reference_success_"))
    info["n_used"] = _as_int(getattr(model, "n_used_", None))
    info["n_rejected"] = _as_int(getattr(model, "n_rejected_", None))
    if hasattr(model, "recheck"):
        info.update(_opinf_fit_info(model))
    return info


def _opinf_fit_info(model) -> dict:
    check = getattr(model, "check_info_", None) or {}
    return {
        "ridge_selected": _as_float(getattr(model, "ridge_selected_", None)),
        "ridge": _as_float(getattr(model, "ridge_", None)),
        "refit_fallback": _as_bool(getattr(model, "refit_fallback_", None)),
        "diverged": _as_bool(getattr(model, "diverged_", None)),
        "diverged_at": _as_float(getattr(model, "diverged_at_", None)),
        "recheck": _as_bool(getattr(model, "recheck", None)),
        "check_periods": _as_float(check.get("check_periods")),
        "check_period_estimated": _as_bool(check.get("period_estimated")),
        "check_T": _as_float(check.get("period")),
        "recheck_path": _path_info(
            getattr(model, "recheck_path_", None),
            ("ridge_rel", "stable", "max_norm_ratio_replay", "max_norm_ratio_forward"),
        ),
        "ridge_path": _path_info(
            getattr(model, "ridge_path_", None),
            ("ridge_rel", "holdout_error", "stable"),
        ),
    }


def _path_info(path, keys: tuple[str, ...]) -> dict | None:
    """One of the OpInf ``*_path_`` dicts as JSON-friendly lists (``None`` if absent)."""
    if path is None:
        return None
    return {key: to_jsonable(path[key]) for key in keys}


def _first_not_none(model, *names: str):
    for name in names:
        value = getattr(model, name, None)
        if value is not None:
            return value
    return None


def _as_int(value) -> int | None:
    return None if value is None else int(value)


def _as_bool(value) -> bool | None:
    return None if value is None else bool(value)


def _as_float(value) -> float | None:
    return None if value is None else float(value)


def _acceptance(result: dict, fit_info: dict | None = None, method: str | None = None) -> dict:
    """The acceptance checks of plan 3.1 for one evaluated method.

    A missing (``None``) or non-finite metric is never a pass: every such check
    fails with a reason.  The only check that may be skipped is the ``phi``
    divergence, which needs the FOM mesh; its ``pass`` is then ``None`` and the
    method's verdict becomes ``INCOMPLETE`` instead of ``PASS``.
    """
    fit_info = fit_info or {}
    checks = []

    def add(name: str, value, threshold, ok, reason: str | None = None) -> None:
        entry = {
            "name": name,
            "value": _check_input(value),
            "threshold": _check_input(threshold),
            "pass": None if ok is None else bool(ok),
        }
        if reason is not None:
            entry["reason"] = reason
        checks.append(entry)

    def add_upper(
        name: str, value, threshold: float, strict: bool, missing: str, nonfinite: str
    ) -> None:
        """A metric that has to stay at or below ``threshold`` (strictly below if ``strict``)."""
        if value is None:
            add(name, None, threshold, False, missing)
            return
        value = float(value)
        if not np.isfinite(value):
            add(name, value, threshold, False, nonfinite)
            return
        add(name, value, threshold, value < threshold if strict else value <= threshold)

    max_u = result["max_error_within_tau"].get("U")
    add_upper(
        "U max relative L2 error within tau<=5",
        max_u,
        U_ERROR_TOL,
        strict=False,
        missing="the U error within tau<=5 could not be evaluated",
        nonfinite="U contains non-finite values (diverged)",
    )
    frequency = result["frequency"].get("relative_error")
    add_upper(
        "latent frequency relative error",
        frequency,
        FREQUENCY_TOL,
        strict=True,
        missing=(
            "latent trajectory contains NaN (diverged)"
            if fit_info.get("diverged")
            else "the latent frequency could not be estimated"
        ),
        nonfinite="latent trajectory contains NaN (diverged)",
    )
    forces_report = _forces(result)
    cl_error = None if forces_report is None else forces_report["cl_amplitude"]["relative_error"]
    add_upper(
        "Cl amplitude relative error",
        cl_error,
        FORCE_TOL,
        strict=True,
        missing="the Cl amplitude could not be evaluated (no force readout)",
        nonfinite="Cl contains non-finite values (diverged)",
    )
    cd_error = None if forces_report is None else forces_report["cd_mean"]["relative_error"]
    add_upper(
        "Cd mean relative error",
        cd_error,
        FORCE_TOL,
        strict=True,
        missing="the Cd mean could not be evaluated (no force readout)",
        nonfinite="Cd contains non-finite values (diverged)",
    )
    rom_energy = _energy(result, "rom")
    ratio = None if rom_energy is None else rom_energy["max_norm_ratio"]
    add_upper(
        "max ||a|| / training max",
        ratio,
        NORM_RATIO_TOL,
        strict=False,
        missing="the latent norm could not be evaluated",
        nonfinite="latent trajectory contains NaN (diverged)",
    )
    drift = None if rom_energy is None else rom_energy["relative_change"]
    drift = None if drift is None else abs(drift)
    add_upper(
        "cycle-averaged energy drift (ROM, absolute)",
        drift,
        ENERGY_DRIFT_TOL,
        strict=True,
        missing="the cycle-averaged energy could not be evaluated",
        nonfinite="latent trajectory contains NaN (diverged)",
    )
    divergence = result.get("phi_divergence")
    phi_name = "phi max cell divergence <= max(100 x FOM, 1e-10)"
    if divergence is None:
        add(
            phi_name,
            None,
            None,
            None,
            "the FOM mesh is unavailable (no cases/<problem>_<study> directory), "
            "so the phi divergence cannot be computed",
        )
    else:
        rom_max, limit = divergence["rom_max"], divergence["limit"]
        if rom_max is None:
            add(phi_name, None, limit, False, "the phi divergence could not be evaluated")
        elif not np.isfinite(rom_max):
            add(phi_name, rom_max, limit, False, "phi contains non-finite values")
        else:
            add(phi_name, rom_max, limit, bool(np.isfinite(limit) and divergence["ok"]))
    if method is not None and method.lower().startswith("opinf"):
        diverged = fit_info.get("diverged")
        if diverged is None:
            add("OpInf did not diverge", None, None, False,
                "the OpInf divergence flag is unavailable")
        else:
            add("OpInf did not diverge", not diverged, None, not diverged)
    verdict = _verdict(checks)
    return {"checks": checks, "verdict": verdict, "pass": verdict == "PASS"}


def _verdict(checks: list[dict]) -> str:
    """``PASS`` only with no failure and no skip; ``FAIL`` beats ``INCOMPLETE``."""
    if any(check["pass"] is False for check in checks):
        return "FAIL"
    if any(check["pass"] is None for check in checks):
        return "INCOMPLETE"
    return "PASS"


def _check_input(value):
    """Keep booleans and ``None``; store numbers as plain floats."""
    if value is None or isinstance(value, bool):
        return value
    return float(value)


def _check_text(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return _sci(value)


def _forces(result: dict) -> dict | None:
    scalars = result.get("scalars")
    return None if scalars is None else scalars.get("forces")


def _energy(result: dict, which: str) -> dict | None:
    energies = result.get("energy")
    return None if energies is None else energies.get(which)


def _phase(result: dict, level: float) -> float | None:
    forces_report = _forces(result)
    if forces_report is None:
        return None
    return forces_report["cl_phase_deg"].get(f"{level:g}")


def _tau_value(result: dict, field: str, level: float):
    values = result.get("errors_at_tau", {}).get(field, {})
    return values.get(f"{level:g}")


# -- input helpers --------------------------------------------------------------------------


def _load(npz_path: Path):
    with np.load(npz_path) as npz:
        times = np.asarray(npz["times"], dtype=float).reshape(-1)
        names = [name for name in FIELDS if name in npz.files]
        if "U" not in names:
            raise SystemExit(f"{npz_path} has no 'U' snapshots")
        fields = {name: np.asarray(npz[name], dtype=float) for name in names}
        forces = np.asarray(npz["forces"], dtype=float)
        volumes = np.asarray(npz["volumes"], dtype=float).reshape(-1)
        phi_w = np.asarray(npz["phi_weights"], dtype=float).reshape(-1)
    if times.size < 2:
        raise SystemExit(f"{npz_path}: need at least two snapshot times")
    if not np.all(np.diff(times) > 0.0):
        raise SystemExit(f"{npz_path}: snapshot times must strictly increase")
    if forces.shape != (times.size, 2):
        raise SystemExit(
            f"{npz_path}: forces has shape {forces.shape}, expected ({times.size}, 2)"
        )
    for name, values in fields.items():
        if values.shape[0] != times.size:
            raise SystemExit(
                f"{npz_path}: {name} has {values.shape[0]} snapshots for {times.size} times"
            )
    return times, forces, volumes, phi_w, fields


def _mesh(cases: Cases, study_root: str):
    case = f"{study_root}/run_0000"
    try:
        path = cases.path(case)
    except FileNotFoundError:
        print(
            f"WARNING: no mesh in {case}; phi divergence and U-phi consistency are skipped"
        )
        return None, None
    return read_topology(path), read_geometry(path)


def _split(times: np.ndarray, n_periods: float, period: float) -> tuple[int, dict]:
    """First ``n_periods`` of the record form the training window (prefix)."""
    cutoff = float(times[0]) + float(n_periods) * period
    tol = 1e-9 * max(1.0, abs(cutoff))
    n_train = int(np.count_nonzero(times <= cutoff + tol))
    return n_train, _window_info(times, n_train, period, cutoff)


def _window_info(
    times: np.ndarray, n_train: int, period: float, cutoff: float | None = None
) -> dict:
    t_end = float(times[n_train - 1]) if n_train else None
    return {
        "n_train": int(n_train),
        "n_test": int(times.size - n_train),
        "t_start": float(times[0]),
        "cutoff": None if cutoff is None else float(cutoff),
        "t_train_end": t_end,
        "t_test_start": float(times[n_train]) if n_train < times.size else None,
        "train_periods": None if t_end is None else (t_end - times[0]) / period,
        "test_periods": None if t_end is None else (float(times[-1]) - t_end) / period,
        "period": float(period),
    }


def _list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_n_modes(text: str) -> dict[str, int | None]:
    out = dict(DEFAULT_N_MODES)
    for item in _list(text):
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep:
            raise SystemExit(
                f"bad --n-modes entry {item!r} (expected field=count or field=auto)"
            )
        if key not in FIELDS:
            raise SystemExit(
                f"unknown field {key!r} in --n-modes (choose from {', '.join(FIELDS)})"
            )
        value = value.strip().lower()
        if value == "auto":
            out[key] = None
            continue
        try:
            count = int(value)
        except ValueError:
            raise SystemExit(
                f"bad --n-modes entry {item!r} (expected an integer or 'auto')"
            ) from None
        if count < 1:
            raise SystemExit(f"bad --n-modes entry {item!r} (the count must be positive)")
        out[key] = count
    return out


def _parse_ints(text: str | None, option: str) -> list[int] | None:
    if text is None:
        return None
    out = []
    for item in _list(text):
        try:
            value = int(item)
        except ValueError:
            raise SystemExit(f"bad {option} entry {item!r} (expected an integer)") from None
        if value < 1:
            raise SystemExit(f"bad {option} entry {item!r} (must be positive)")
        out.append(value)
    if not out:
        raise SystemExit(f"{option} got no values")
    return out


def _parse_floats(text: str | None, option: str) -> list[float] | None:
    if text is None:
        return None
    out = []
    for item in _list(text):
        try:
            value = float(item)
        except ValueError:
            raise SystemExit(f"bad {option} entry {item!r} (expected a number)") from None
        if not value > 0.0:
            raise SystemExit(f"bad {option} entry {item!r} (must be positive)")
        out.append(value)
    if not out:
        raise SystemExit(f"{option} got no values")
    return out


def _n_modes_text(n_modes: dict[str, int | None]) -> str:
    return ", ".join(
        f"{name}={value if value is not None else 'auto'}" for name, value in n_modes.items()
    )


# -- terminal formatting --------------------------------------------------------------------


def _sci(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2e}"


def _count(value) -> str:
    return "n/a" if value is None else str(int(value))


def _deg(value) -> str:
    return "n/a" if value is None else f"{value:+.1f}"


def _pct(value) -> str:
    return "n/a" if value is None else f"{100.0 * value:+.2f}%"


def _ratio(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _table(header: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(str(row[i])) for row in [header, *rows]) + 2 for i in range(len(header))
    ]
    print("".join(str(cell).ljust(w) for cell, w in zip(header, widths)).rstrip())
    for row in rows:
        print("".join(str(cell).ljust(w) for cell, w in zip(row, widths)).rstrip())


if __name__ == "__main__":
    main()
