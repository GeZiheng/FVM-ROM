"""Fit a steady ROM on the collected snapshots, evaluate it and write a JSON report.

Usage:
  uv run python scripts/fit_and_eval.py --problem cavity --study re1d \
      --regressors rbf,gpr,mlp --learning-curve --write-fields 2
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from fvmrom.config import PROJECT_ROOT, get_settings
from fvmrom.foam import (
    Cases,
    FoamField,
    latest_time,
    read_field,
    read_topology,
    validate_study,
    write_field,
)
from fvmrom.params import ParamSpace
from fvmrom.problems import get_problem
from fvmrom.rom import SteadyROM
from fvmrom.rom.evaluate import (
    evaluate_on_test,
    learning_curve,
    loo_cv,
    projection_error,
    retained_energy,
    std_error_correlation,
)
from fvmrom.rom.regressors import REGRESSOR_KINDS

ROM_TIME = "1"

# The GPR kernel optimum often sits on a bound of the length-scale/noise bounds; that is
# expected for these small snapshot sets and would otherwise flood the report.
warnings.filterwarnings("ignore", category=ConvergenceWarning, module=r"sklearn")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem", required=True, choices=["cavity", "cylinder"])
    p.add_argument("--study", default="re1d")
    p.add_argument("--regressors", default="rbf,gpr,mlp")
    p.add_argument("--energy-tol", type=float, default=1e-10)
    p.add_argument("--n-modes", default=None, help="per-field caps, e.g. U=10,p=10,phi=10")
    p.add_argument("--write-fields", type=int, default=0, metavar="N")
    p.add_argument("--learning-curve", action="store_true")
    p.add_argument("--sizes", default="5,8,12,16,20,30")
    p.add_argument("--mlp-seeds", type=int, default=5)
    args = p.parse_args()
    kinds = _list(args.regressors)
    if not kinds:
        p.error("--regressors must name at least one of rbf, gpr, mlp")
    unknown = [kind for kind in kinds if kind.lower() not in REGRESSOR_KINDS]
    if unknown:
        p.error(
            f"unknown regressor(s) {', '.join(unknown)} in --regressors "
            f"(choose from {', '.join(sorted(REGRESSOR_KINDS))})"
        )
    return args


def main() -> None:
    args = parse_args()
    settings = get_settings()
    study = validate_study(args.study, problem=args.problem, settings=settings)
    data_dir = settings.data_dir / args.problem / study
    npz_path = data_dir / "snapshots.npz"
    meta_path = data_dir / "samples.json"
    if not npz_path.is_file() or not meta_path.is_file():
        raise SystemExit(
            f"missing snapshots in {data_dir}: run scripts/generate_snapshots.py first"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(npz_path) as npz:
        names = list(meta["fields"])
        snaps = {name: np.asarray(npz[name], dtype=float) for name in names}
        mu = np.asarray(npz["params"], dtype=float)
        volumes = np.asarray(npz["volumes"], dtype=float)
        roles = npz["role"].astype(str)

    space = ParamSpace(
        names=tuple(meta["param_names"]),
        lower=tuple(meta["lower"]),
        upper=tuple(meta["upper"]),
        log=tuple(meta["log"]),
    )
    train, test = roles == "train", roles == "test"
    n_train, n_test = int(train.sum()), int(test.sum())
    if not n_train or not n_test:
        raise SystemExit("snapshots need samples with both roles 'train' and 'test'")

    kinds = _list(args.regressors)
    n_modes = _n_modes(args.n_modes)
    cases = Cases(settings)
    topo = _topology(cases, f"{args.problem}_{study}")
    test_cases = _test_cases(meta)

    print(
        f"{args.problem}/{study}: {n_train} train + {n_test} test samples, "
        f"{space.names} in [{space.lower}, {space.upper}], log {space.log}"
    )
    print(f"energy_tol {args.energy_tol:g}, mode caps {n_modes or 'none (energy only)'}")

    fitted: dict[str, SteadyROM] = {}
    rows = []
    for kind in kinds:
        rom = SteadyROM(space, kind, n_modes=n_modes, energy_tol=args.energy_tol)
        started = time.perf_counter()
        rom.fit(
            mu[train], {name: values[train] for name, values in snaps.items()}, volumes
        )
        fit_s = time.perf_counter() - started
        rom.save(data_dir / "models" / f"{kind}.pkl")
        started = time.perf_counter()
        rom.predict(mu[test])
        predict_s = (time.perf_counter() - started) / n_test
        test_eval = evaluate_on_test(
            rom,
            mu[test],
            {name: values[test] for name, values in snaps.items()},
            volumes,
            topo,
        )
        started = time.perf_counter()
        loo = loo_cv(
            lambda kind=kind: SteadyROM(
                space, kind, n_modes=n_modes, energy_tol=args.energy_tol
            ),
            mu[train],
            {name: values[train] for name, values in snaps.items()},
            volumes,
        )
        loo_s = time.perf_counter() - started
        fitted[kind] = rom
        rows.append(
            {
                "kind": kind,
                "fit_seconds": fit_s,
                "predict_seconds_per_sample": predict_s,
                "loo_seconds": loo_s,
                "test": test_eval,
                "loo": loo,
            }
        )
        print(f"fitted {kind}: {fit_s:.2f} s, LOO {loo_s:.1f} s", flush=True)

    pod_report = _pod_report(fitted[kinds[0]], snaps, train, test)

    print("\nPOD (shared by all regressors)")
    _table(
        ["field", "modes", "discarded energy", "projection mean", "projection max"],
        [
            [
                name,
                str(info["n_modes"]),
                _sci(1.0 - info["retained_energy"]),
                _sci(info["projection_error"]["mean"]),
                _sci(info["projection_error"]["max"]),
            ]
            for name, info in pod_report.items()
        ],
    )

    print("\ntest-set and LOO-CV relative L2 error")
    _table(
        [
            "regressor",
            "field",
            "test mean",
            "test max",
            "LOO mean",
            "LOO max",
            "fit s",
            "predict ms",
        ],
        [
            [
                row["kind"],
                name,
                _sci(row["test"]["fields"][name]["mean"]),
                _sci(row["test"]["fields"][name]["max"]),
                _sci(row["loo"]["fields"][name]["mean"]),
                _sci(row["loo"]["fields"][name]["max"]),
                f"{row['fit_seconds']:.2f}",
                f"{1e3 * row['predict_seconds_per_sample']:.2f}",
            ]
            for row in rows
            for name in names
        ],
    )

    divergence = _divergence_report(rows, topo)
    if divergence is not None:
        print(
            f"\nphi divergence (max |div phi| per cell): FOM {_sci(divergence['fom_max'])}, "
            f"limit {_sci(divergence['limit'])}"
        )
        for kind, values in divergence["rom"].items():
            print(
                f"  {kind:<4} ROM {_sci(values['max'])}  "
                f"{'ok' if values['ok'] else 'FAILED'}"
            )
    else:
        print("\nphi divergence: not evaluated (no mesh topology found)")

    comparison = None
    if args.learning_curve:
        comparison = _compare(
            args, rows, fitted, space, mu, snaps, train, test, volumes, names,
            pod_report, n_modes,
        )

    _print_acceptance(rows, pod_report, divergence, names)

    snapshots_label = (
        npz_path.relative_to(PROJECT_ROOT).as_posix()
        if npz_path.is_relative_to(PROJECT_ROOT)
        else str(npz_path)
    )
    report = {
        "problem": args.problem,
        "study": study,
        "snapshots": snapshots_label,
        "n_train": n_train,
        "n_test": n_test,
        "param_names": list(space.names),
        "lower": list(space.lower),
        "upper": list(space.upper),
        "log": list(space.log),
        "energy_tol": args.energy_tol,
        "n_modes": n_modes,
        "pod": pod_report,
        "regressors": {
            row["kind"]: {
                "model": f"models/{row['kind']}.pkl",
                "test": row["test"],
                "loo": row["loo"],
                "cost": {
                    "fit_seconds": row["fit_seconds"],
                    "predict_seconds_per_sample": row["predict_seconds_per_sample"],
                    "loo_seconds": row["loo_seconds"],
                },
            }
            for row in rows
        },
        "phi_divergence": divergence,
        "comparison": comparison,
    }
    report_path = data_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"\nreport: {report_path}")
    print(f"models: {data_dir / 'models'}")

    if args.write_fields > 0:
        _write_rom_cases(args, cases, topo, fitted, mu[test], names, test_cases)


# -- comparison (plan 3.6) ------------------------------------------------------------------


def _compare(
    args, rows, fitted, space, mu, snaps, train, test, volumes, names, pod_report, n_modes
) -> dict:
    sizes = [int(s) for s in _list(args.sizes)]
    curve = learning_curve(
        list(fitted),
        sizes,
        mu[train],
        {name: values[train] for name, values in snaps.items()},
        mu[test],
        {name: values[test] for name, values in snaps.items()},
        volumes,
        mlp_seeds=tuple(range(args.mlp_seeds)),
        progress=lambda kind, n: print(f"  learning curve {kind} n={n}", flush=True),
        space=space,
        energy_tol=args.energy_tol,
        n_modes=n_modes,
    )

    print("\nlearning curve: test-set mean relative L2 error")
    header = ["n"] + [f"{kind}/{name}" for kind in fitted for name in names]
    body = []
    for i, n in enumerate(curve["sizes"]):
        body.append(
            [str(n)]
            + [_sci(curve["kinds"][kind][i]["fields"][name]["mean"]) for kind in fitted for name in names]
        )
    _table(header, body)

    if curve["mlp"] is not None:
        print(f"\nMLP stability: mean +- std over seeds {curve['mlp']['seeds']}")
        body = []
        for row in curve["mlp"]["rows"]:
            body.append(
                [str(row["n"])]
                + [
                    f"{_sci(row['per_seed'][name]['mean'])} +- {_sci(row['per_seed'][name]['std'])}"
                    for name in names
                ]
            )
        _table(["n"] + names, body)

    uncertainty = None
    if "gpr" in fitted:
        uncertainty = std_error_correlation(
            fitted["gpr"],
            mu[test],
            {name: values[test] for name, values in snaps.items()},
            volumes,
        )
        print("\nGPR uncertainty: Spearman rho of the predicted std vs the actual error")
        _table(
            ["field", "sample level", "mode level"],
            [
                [name, _rho(info["sample_rho"]), _rho(info["mode_rho"])]
                for name, info in uncertainty["fields"].items()
            ],
        )

    return {
        # 1. error against the number of training samples
        "learning_curve": {"sizes": curve["sizes"], "curve": curve["kinds"]},
        # 2. accuracy: projection error (shared lower bound), test and LOO errors
        "accuracy": {
            "projection_error": {
                name: pod_report[name]["projection_error"] for name in names
            },
            "test": {row["kind"]: row["test"]["fields"] for row in rows},
            "loo": {row["kind"]: row["loo"]["fields"] for row in rows},
        },
        # 3. MLP stability over seeds
        "mlp_stability": curve["mlp"],
        # 4. cost
        "cost": {
            "fit": {
                row["kind"]: {
                    "train_seconds": row["fit_seconds"],
                    "predict_seconds_per_sample": row["predict_seconds_per_sample"],
                }
                for row in rows
            },
            "learning_curve": {
                kind: [
                    {
                        "n": row["n"],
                        "train_seconds": row["train_seconds"],
                        "predict_seconds_per_sample": row["predict_seconds_per_sample"],
                    }
                    for row in curve["kinds"][kind]
                ]
                for kind in curve["kinds"]
            },
            "loo": {
                row["kind"]: {"folds": row["loo"]["folds"], "seconds": row["loo_seconds"]}
                for row in rows
            },
        },
        # 5. uncertainty (GPR only)
        "uncertainty": uncertainty,
        # MLP vs the classical regressors, required by the plan
        "mlp_gap": _mlp_gap(rows, names),
    }


def _mlp_gap(rows, names) -> dict | None:
    by_kind = {row["kind"]: row for row in rows}
    if "mlp" not in by_kind:
        return None
    baselines = [kind for kind in ("rbf", "gpr") if kind in by_kind]
    if not baselines:
        return None
    out = {}
    for name in names:
        mlp = by_kind["mlp"]["test"]["fields"][name]["mean"]
        best = min(by_kind[kind]["test"]["fields"][name]["mean"] for kind in baselines)
        out[name] = {
            "mlp": mlp,
            "best_baseline": best,
            "best_baseline_kind": min(
                baselines, key=lambda kind: by_kind[kind]["test"]["fields"][name]["mean"]
            ),
            "ratio": float(mlp / best) if best > 0 else None,
        }
    return out


# -- report pieces --------------------------------------------------------------------------


def _pod_report(rom: SteadyROM, snaps, train, test) -> dict:
    out = {}
    for name, values in snaps.items():
        info = _pod_field(rom, name, snaps, test)
        k = int(train.sum())
        info["retained_energy"] = retained_energy(rom.pods_[name], values[train].reshape(k, -1))
        out[name] = info
    return out


def _pod_field(rom: SteadyROM, name: str, snaps, test) -> dict:
    pod = rom.pods_[name]
    errors = projection_error(pod, snaps[name][test])
    return {
        "n_modes": int(pod.modes_.shape[0]),
        "projection_error": {
            "mean": float(np.mean(errors)),
            "max": float(np.max(errors)),
            "errors": [float(v) for v in errors],
        },
    }


def _divergence_report(rows, topo) -> dict | None:
    if topo is None:
        return None
    first = rows[0]["test"]["phi_divergence"]
    if first is None:
        return None
    return {
        "fom_max": first["fom_max"],
        "limit": first["limit"],
        "rom": {
            row["kind"]: {
                "max": row["test"]["phi_divergence"]["rom_max"],
                "ok": row["test"]["phi_divergence"]["ok"],
            }
            for row in rows
        },
    }


def _print_acceptance(rows, pod_report, divergence, names) -> None:
    by_kind = {row["kind"]: row for row in rows}
    print("\nacceptance")
    candidates = [kind for kind in ("rbf", "gpr") if kind in by_kind]
    if "U" in names and candidates:
        values = {kind: by_kind[kind]["test"]["fields"]["U"]["mean"] for kind in candidates}
        best = min(values, key=values.get)
        print(
            f"  U mean relative L2 error < 2% (rbf or gpr): "
            f"{', '.join(f'{k} {_sci(v)}' for k, v in values.items())} -> "
            f"{'PASS' if values[best] < 2e-2 else 'FAILED'}"
        )
    checks = []
    for name in names:
        proj = pod_report[name]["projection_error"]["mean"]
        rom_min = min(row["test"]["fields"][name]["mean"] for row in rows)
        checks.append(f"{name} {_sci(proj)} <= {_sci(rom_min)} {'ok' if proj <= rom_min else 'FAILED'}")
    print("  POD projection error <= ROM prediction error: " + "; ".join(checks))
    if divergence is not None:
        failed = [kind for kind, values in divergence["rom"].items() if not values["ok"]]
        print(
            f"  phi max cell divergence <= {_sci(divergence['limit'])}: "
            + ", ".join(f"{kind} {_sci(v['max'])}" for kind, v in divergence["rom"].items())
            + (" -> PASS" if not failed else f" -> FAILED ({', '.join(failed)})")
        )
    else:
        print("  phi divergence check SKIPPED (no mesh topology)")
    if "mlp" in by_kind:
        gap = _mlp_gap(rows, names)
        if gap:
            print(
                "  MLP vs best of rbf/gpr (test mean, ratio): "
                + ", ".join(
                    f"{name} {_sci(info['mlp'])} vs {_sci(info['best_baseline'])}"
                    f" ({info['ratio']:.1f}x)" if info["ratio"] else f"{name} {_sci(info['mlp'])}"
                    for name, info in gap.items()
                )
            )


# -- writing the reconstructed fields -------------------------------------------------------


def _write_rom_cases(args, cases, topo, fitted, mu_test, names, test_cases):
    if topo is None:
        raise SystemExit("--write-fields needs the mesh topology (cases/<problem>_<study>/run_0000)")
    n_write = min(args.write_fields, mu_test.shape[0])
    templates = [
        {name: read_field(cases.path(test_cases[k]), latest_time(cases.path(test_cases[k])), name)
         for name in names}
        for k in range(n_write)
    ]  # fmt: skip
    for kind, rom in fitted.items():
        pred = rom.predict(mu_test[:n_write])
        root = f"{args.problem}_{args.study}_rom_{kind}"
        for k in range(n_write):
            case = f"{root}/run_{k:04d}"
            dest = cases.path(case, must_exist=False)
            shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True)
            for sub in ("constant", "system"):
                shutil.copytree(cases.path(test_cases[k]) / sub, dest / sub)
            cases.write(case, f"{Path(case).name}.foam", "")
            write_rom_case(
                dest,
                {name: np.asarray(pred[name][k]) for name in names},
                templates[k],
                topo,
            )
            print(f"wrote ROM fields: {case} (test sample {k}, Re={mu_test[k, 0]:g})")


def write_rom_case(dest, pred: dict, templates: dict, topo, time: str = ROM_TIME):
    """Write the ROM fields and the ROM-FOM differences of one sample into ``dest``.

    ``pred``/``templates`` map field names to a single-sample prediction and the
    matching FOM field; the difference fields (``U_err``, ``p_err``) are ROM - FOM
    per cell, with the FOM boundary kept as the template.
    """
    for name, values in pred.items():
        template = templates[name]
        if name == "phi":
            field = _split_phi(np.asarray(values), topo, template)
        else:
            field = FoamField(
                name,
                template.cls,
                np.asarray(values),
                _boundary(template),
                template.dimensions,
            )
        write_field(dest, time, field, template=template)

    for name in ("U", "p"):
        if name not in pred or name not in templates:
            continue
        template = templates[name]
        fom = np.asarray(template.internal, dtype=float)
        diff = np.asarray(pred[name], dtype=float) - fom
        for suffix, values in (("fom", fom), ("err", diff)):
            write_field(
                dest,
                time,
                FoamField(
                    f"{name}_{suffix}",
                    template.cls,
                    values,
                    _boundary(template),
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
        # same rule as phi_vector: patches without a value (empty) carry no flux
        if value is None or np.size(value) == 0 or n == 0:
            continue
        boundary[name] = {"type": entry["type"], "value": np.array(phi[pos : pos + n])}
        pos += n
    if pos != phi.size:
        raise ValueError(f"phi has {phi.size - pos} values not covered by the patches")
    return FoamField(
        "phi", template.cls, phi[: topo.neighbour.size], boundary, template.dimensions
    )


# -- input helpers --------------------------------------------------------------------------


def _topology(cases: Cases, study_root: str):
    case = f"{study_root}/run_0000"
    try:
        return read_topology(cases.path(case))
    except FileNotFoundError:
        print(f"WARNING: no mesh in {case}; phi divergence is not evaluated")
        return None


def _test_cases(meta: dict) -> list[str]:
    return [s["case"] for s in meta["samples"] if s["converged"] and s["role"] == "test"]


def _list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _n_modes(text: str | None) -> dict[str, int] | None:
    if not text:
        return None
    out = {}
    for item in _list(text):
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"bad --n-modes entry {item!r} (expected field=count)")
        out[key.strip()] = int(value)
    return out


# -- terminal formatting --------------------------------------------------------------------


def _sci(value: float) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2e}"


def _rho(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}"


def _table(header: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(str(row[i])) for row in [header, *rows]) + 2 for i in range(len(header))
    ]
    print("".join(str(cell).ljust(w) for cell, w in zip(header, widths)).rstrip())
    for row in rows:
        print("".join(str(cell).ljust(w) for cell, w in zip(row, widths)).rstrip())


if __name__ == "__main__":
    main()
