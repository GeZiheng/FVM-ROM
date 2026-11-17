"""Evaluation of a steady ROM: errors, projection error, LOO-CV and comparisons.

Snapshot arrays keep the sample axis first, e.g. ``U`` of shape (n, n_cells, 3).
Weights follow the POD convention: cell volumes for cell-centred fields, unit
weights for face fluxes.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np
from scipy.stats import spearmanr

from ..foam.fields import MeshTopology, phi_divergence
from ..foam.mesh import MeshGeometry
from .pod import WeightedPOD
from .regressors import GPRRegressor
from .steady import VOLUME_WEIGHTED, SteadyROM


def rel_l2_error(
    pred: np.ndarray, ref: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Per-snapshot relative L2 error ``||pred - ref||_W / ||ref||_W``.

    ``pred`` and ``ref`` have the same shape, sample axis first.  ``weights`` is
    over the flattened trailing shape; for a vector field the cell volumes may be
    given per cell and are repeated for the three components.
    """
    p = np.asarray(pred, dtype=float)
    r = np.asarray(ref, dtype=float)
    if p.shape != r.shape:
        raise ValueError(f"prediction shape {p.shape} differs from reference {r.shape}")
    if r.ndim < 1 or r.shape[0] == 0:
        raise ValueError(f"expected at least one snapshot, got shape {r.shape}")
    n = r.shape[0]
    pf, rf = p.reshape(n, -1), r.reshape(n, -1)
    w = _expand_weights(weights, r.shape[1:])
    scale = np.sqrt(w)
    return np.linalg.norm((pf - rf) * scale, axis=1) / np.linalg.norm(rf * scale, axis=1)


def projection_error(pod: WeightedPOD, X: np.ndarray) -> np.ndarray:
    """Relative L2 error of the POD projection of ``X`` (the ROM error lower bound)."""
    flat = _as_flat(X)
    return rel_l2_error(pod.reconstruct(pod.project(flat)), flat, pod.weights)


def flux_consistency(
    U: np.ndarray,
    phi: np.ndarray,
    geom: MeshGeometry,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Per-snapshot ``||phi - I(U).Sf||_W / ||I(U).Sf||_W`` on the internal faces.

    ``U`` is (n, n_cells, 3), ``phi`` (n, n_phi) and ``I(U)`` the OpenFOAM ``linear``
    interpolation (:attr:`MeshGeometry.lin_weights`).  The boundary patches are left
    out because their ``U`` values are not part of the snapshots.  ``weights`` follows
    the ``phi`` layout (or covers the internal faces only).
    """
    topo = geom.topo
    n_faces = topo.neighbour.size
    U = np.asarray(U, dtype=float)
    if U.ndim == 2:
        U = U[None]
    if U.ndim != 3 or U.shape[1:] != (topo.n_cells, 3):
        raise ValueError(f"U: expected (n, {topo.n_cells}, 3), got {U.shape}")
    phi = np.asarray(phi, dtype=float)
    if phi.ndim == 1:
        phi = phi[None]
    if phi.ndim != 2 or phi.shape[0] != U.shape[0]:
        raise ValueError(f"phi: expected (n, n_phi), got {phi.shape}")
    if phi.shape[1] < n_faces:
        raise ValueError(
            f"phi: expected at least {n_faces} values per snapshot, got {phi.shape[1]}"
        )
    if n_faces == 0:
        return np.zeros(phi.shape[0])
    w = np.asarray(geom.lin_weights, dtype=float).reshape(-1)
    if w.size != n_faces:
        raise ValueError(f"{n_faces} internal faces but {w.size} interpolation weights")
    own = np.asarray(topo.owner[:n_faces], dtype=np.int64)
    nei = np.asarray(topo.neighbour, dtype=np.int64)
    interp = w[:, None] * U[:, own] + (1.0 - w)[:, None] * U[:, nei]
    flux = np.einsum("nfj,fj->nf", interp, geom.face_areas[:n_faces])
    residual = phi[:, :n_faces] - flux
    weight = _flux_weights(weights, n_faces, phi.shape[1])
    scale = np.sqrt(weight)
    numerator = np.linalg.norm(residual * scale, axis=1)
    denominator = np.linalg.norm(flux * scale, axis=1)
    out = np.full(numerator.shape, np.inf)
    good = denominator > 0.0
    out[good] = numerator[good] / denominator[good]
    out[~good & (numerator == 0.0)] = 0.0
    return out


def retained_energy(pod: WeightedPOD, X: np.ndarray) -> float:
    """Fraction of the training energy (weighted, mean removed) kept by the POD."""
    flat = _as_flat(X)
    if pod.mean_ is None:
        raise RuntimeError("WeightedPOD is not fitted")
    weights = pod.weights
    if weights is None:
        weights = np.ones(flat.shape[1])
    elif weights.size != flat.shape[1]:
        raise ValueError(f"expected {flat.shape[1]} weights, got {weights.size}")
    centred = flat - pod.mean_
    total = float(np.sum(centred**2 * weights))
    if total <= 0.0:
        return 1.0
    residual = flat - pod.reconstruct(pod.project(flat))
    return 1.0 - float(np.sum(residual**2 * weights)) / total


def loo_cv(
    make_rom: Callable[[], SteadyROM],
    mu: np.ndarray,
    snapshots: dict[str, np.ndarray],
    volumes: np.ndarray | None = None,
    phi_weights: np.ndarray | None = None,
) -> dict:
    """Leave-one-out CV: refit POD and regressor without sample i, predict sample i.

    ``make_rom`` returns a fresh, unfitted ROM; it is called once per fold.
    """
    mu = np.asarray(mu, dtype=float)
    if mu.ndim == 1:  # a 1-D batch is a single parameter (as in SteadyROM)
        mu = mu.reshape(-1, 1)
    if mu.ndim != 2:
        raise ValueError(f"expected parameters of shape (n, d), got {mu.shape}")
    n = mu.shape[0]
    if n < 2:
        raise ValueError("leave-one-out cross-validation needs at least two samples")
    names = list(snapshots)
    values = {name: _as_snapshots(snapshots[name], n, name) for name in names}
    errors = {name: [] for name in names}
    extra = {} if phi_weights is None else {"phi_weights": phi_weights}
    for i in range(n):
        keep = np.arange(n) != i
        rom = make_rom().fit(
            mu[keep], {name: values[name][keep] for name in names}, volumes, **extra
        )
        pred = rom.predict(mu[i : i + 1])
        for name in names:
            ref = values[name][i : i + 1]
            weights = _field_weights(name, ref.shape[1:], volumes, phi_weights)
            errors[name].append(float(rel_l2_error(pred[name], ref, weights)[0]))
    return {
        "n": n,
        "folds": n,
        "fields": {name: _stats(errors[name]) for name in names},
    }


def evaluate_on_test(
    rom: SteadyROM,
    mu_test: np.ndarray,
    snap_test: dict[str, np.ndarray],
    volumes: np.ndarray | None = None,
    topo: MeshTopology | None = None,
    geom: MeshGeometry | None = None,
    phi_weights: np.ndarray | None = None,
) -> dict:
    """Test-set errors per field, plus divergence and U-``phi`` consistency.

    ``topo`` adds the ``phi`` divergence residual, ``geom`` the U-``phi`` flux
    consistency of the ROM and of the FOM baseline.
    """
    mu_test = _as_batch(mu_test, rom.space.dim, "mu_test")
    n = mu_test.shape[0]
    pred = rom.predict(mu_test)
    out: dict = {
        "n_test": n,
        "fields": {},
        "phi_divergence": None,
        "flux_consistency": None,
    }
    for name, ref in snap_test.items():
        ref = _as_snapshots(ref, n, name)
        errors = rel_l2_error(
            pred[name], ref, _field_weights(name, ref.shape[1:], volumes, phi_weights)
        )
        out["fields"][name] = _stats(errors)

    if topo is not None and "phi" in snap_test:
        ref_phi = _as_snapshots(snap_test["phi"], n, "phi")
        rom_max = _max_divergence(pred["phi"], topo)
        fom_max = _max_divergence(ref_phi, topo)
        limit = max(100.0 * fom_max, 1e-10)
        out["phi_divergence"] = {
            "rom_max": rom_max,
            "fom_max": fom_max,
            "limit": limit,
            "ok": bool(rom_max <= limit),
        }

    if geom is not None and {"U", "phi"} <= set(snap_test):
        fom = flux_consistency(
            _as_snapshots(snap_test["U"], n, "U"),
            _as_snapshots(snap_test["phi"], n, "phi"),
            geom,
            phi_weights,
        )
        rom = flux_consistency(pred["U"], pred["phi"], geom, phi_weights)
        out["flux_consistency"] = {
            "fom": _stats(fom),
            "rom": _stats(rom),
            "ratio": float(np.mean(rom) / np.mean(fom)) if np.mean(fom) > 0.0 else None,
        }
    return out


def std_error_correlation(
    rom: SteadyROM,
    mu_test: np.ndarray,
    snap_test: dict[str, np.ndarray],
    volumes: np.ndarray | None = None,
    phi_weights: np.ndarray | None = None,
) -> dict:
    """Spearman correlation between the GPR prediction std and the actual error.

    Reported per field at two levels: over test samples (field error vs RMS std
    over modes) and over all (sample, mode) pairs (coefficient error vs std).
    """
    if rom.regressor != "gpr":
        raise ValueError("std_error_correlation needs a ROM fitted with the GPR regressor")
    mu_test = _as_batch(mu_test, rom.space.dim, "mu_test")
    n = mu_test.shape[0]
    z = rom.space.to_unit(mu_test)
    pred = rom.predict(mu_test)
    out = {"method": "spearman", "n_samples": n, "fields": {}}
    for name, ref in snap_test.items():
        model = rom.models_.get(name)
        if not isinstance(model, GPRRegressor):
            raise ValueError(f"{name}: the fitted model is not a GPRRegressor")
        ref = _as_snapshots(ref, n, name)
        mean, std = model.predict(z, return_std=True)
        coeff = rom.pods_[name].project(ref.reshape(n, -1))
        errors = rel_l2_error(
            pred[name], ref, _field_weights(name, ref.shape[1:], volumes, phi_weights)
        )
        sample_std = np.sqrt(np.mean(std**2, axis=1))
        out["fields"][name] = {
            "n_modes": int(std.shape[1]),
            "sample_rho": _rho(sample_std, errors),
            "mode_rho": _rho(std.ravel(), np.abs(mean - coeff).ravel()),
        }
    return out


def learning_curve(
    kinds: Sequence[str],
    sizes: Sequence[int],
    mu_train: np.ndarray,
    snap_train: dict[str, np.ndarray],
    mu_test: np.ndarray,
    snap_test: dict[str, np.ndarray],
    volumes: np.ndarray | None = None,
    mlp_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    progress: Callable[[str, int], None] | None = None,
    phi_weights: np.ndarray | None = None,
    **rom_kw,
) -> dict:
    """Error against the number of training samples, for each regressor.

    Sizes are prefixes of the training set (Sobol order makes them nested); each
    point refits the POD and the regressor.  The MLP curve entry is a single
    network (seed 0); ``mlp`` additionally reports the spread over single seeds.
    ``rom_kw`` is passed to :class:`SteadyROM` and must contain ``space``.
    """
    if "space" not in rom_kw:
        raise ValueError("learning_curve needs the parameter space: pass space=... in rom_kw")
    space = rom_kw["space"]
    mu_train = _as_batch(mu_train, space.dim, "mu_train")
    mu_test = _as_batch(mu_test, space.dim, "mu_test")
    n_train = mu_train.shape[0]
    sizes = sorted({int(s) for s in sizes if 0 < int(s) <= n_train})
    if not sizes:
        raise ValueError(f"no learning-curve size fits the {n_train} training samples")
    if sizes[0] < 2:
        raise ValueError("a learning-curve point needs at least two training samples")
    names = list(snap_train)
    train_values = {name: _as_snapshots(snap_train[name], n_train, name) for name in names}
    test_values = {
        name: _as_snapshots(snap_test[name], mu_test.shape[0], name) for name in names
    }
    base_kw = dict(rom_kw.pop("regressor_kw", None) or {})
    seeds = tuple(int(s) for s in mlp_seeds)

    out: dict = {"sizes": sizes, "fields": names, "kinds": {}, "mlp": None}
    for kind in kinds:
        rows = []
        for n in sizes:
            kw = dict(base_kw)
            if kind == "mlp":
                kw["seeds"] = (0,)
            _rom, pred, train_s, predict_s = _fit_and_time(
                kind,
                {**rom_kw, "regressor_kw": kw},
                mu_train[:n],
                {name: train_values[name][:n] for name in names},
                volumes,
                mu_test,
                phi_weights,
            )
            rows.append(
                {
                    "n": n,
                    "train_seconds": train_s,
                    "predict_seconds_per_sample": predict_s,
                    "fields": _field_stats(pred, test_values, volumes, phi_weights),
                }
            )
            if progress is not None:
                progress(kind, n)
        out["kinds"][kind] = rows

    if "mlp" in kinds and len(seeds) >= 2:
        out["mlp"] = _mlp_stability(
            sizes,
            seeds,
            {**rom_kw, "regressor_kw": base_kw},
            mu_train,
            train_values,
            mu_test,
            test_values,
            volumes,
            phi_weights,
        )
    return out


def _mlp_stability(
    sizes: Sequence[int],
    seeds: Sequence[int],
    rom_kw: dict,
    mu_train: np.ndarray,
    train_values: dict[str, np.ndarray],
    mu_test: np.ndarray,
    test_values: dict[str, np.ndarray],
    volumes: np.ndarray | None,
    phi_weights: np.ndarray | None = None,
) -> dict:
    base_kw = dict(rom_kw.get("regressor_kw") or {})
    rows = []
    for n in sizes:
        per_seed = {name: [] for name in train_values}
        for seed in seeds:
            rom = SteadyROM(
                regressor="mlp", **{**rom_kw, "regressor_kw": {**base_kw, "seeds": (seed,)}}
            ).fit(
                mu_train[:n],
                {name: train_values[name][:n] for name in train_values},
                volumes,
                phi_weights,
            )
            stats = _field_stats(rom.predict(mu_test), test_values, volumes, phi_weights)
            for name in per_seed:
                per_seed[name].append(stats[name]["mean"])
        rows.append(
            {
                "n": n,
                "per_seed": {
                    name: {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)),
                        "values": [float(v) for v in values],
                    }
                    for name, values in per_seed.items()
                },
            }
        )
    return {"seeds": [int(s) for s in seeds], "rows": rows}


def _fit_and_time(
    kind: str,
    rom_kw: dict,
    mu_train: np.ndarray,
    snap_train: dict[str, np.ndarray],
    volumes: np.ndarray | None,
    mu_test: np.ndarray,
    phi_weights: np.ndarray | None = None,
) -> tuple[SteadyROM, dict, float, float]:
    started = time.perf_counter()
    rom = SteadyROM(regressor=kind, **rom_kw).fit(
        mu_train, snap_train, volumes, phi_weights
    )
    train_s = time.perf_counter() - started
    started = time.perf_counter()
    pred = rom.predict(mu_test)
    predict_s = (time.perf_counter() - started) / mu_test.shape[0]
    return rom, pred, train_s, predict_s


def _field_stats(
    pred: dict,
    snapshots: dict,
    volumes: np.ndarray | None,
    phi_weights: np.ndarray | None = None,
) -> dict:
    out = {}
    for name, ref in snapshots.items():
        errors = rel_l2_error(
            pred[name], ref, _field_weights(name, ref.shape[1:], volumes, phi_weights)
        )
        out[name] = {"mean": float(np.mean(errors)), "max": float(np.max(errors))}
    return out


def _max_divergence(phi: np.ndarray, topo: MeshTopology) -> float:
    phi = np.asarray(phi, dtype=float)
    if phi.ndim == 1:
        phi = phi[None]
    return float(max(np.abs(phi_divergence(v, topo)).max() for v in phi))


def _stats(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "max": float(np.max(a)),
        "errors": [float(v) for v in a],
    }


def _rho(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3:
        return None
    rho = spearmanr(a, b).statistic
    return float(rho) if np.isfinite(rho) else None


def _as_flat(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim < 2:
        raise ValueError(f"expected a snapshot matrix (n, m), got shape {X.shape}")
    return X.reshape(X.shape[0], -1)


def _as_snapshots(X: np.ndarray, n: int, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim < 1 or X.shape[0] != n:
        raise ValueError(f"{name}: expected {n} snapshots, got shape {X.shape}")
    return X


def _as_batch(mu: np.ndarray, dim: int, name: str) -> np.ndarray:
    a = np.asarray(mu, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, dim)
    if a.ndim != 2 or a.shape[1] != dim:
        raise ValueError(f"{name}: expected shape (n, {dim}), got {a.shape}")
    return a


def _field_weights(
    name: str,
    shape: tuple[int, ...],
    volumes: np.ndarray | None,
    phi_weights: np.ndarray | None = None,
) -> np.ndarray | None:
    if name == "phi" and phi_weights is not None:
        w = np.asarray(phi_weights, dtype=float).reshape(-1)
        if not shape or w.size != shape[0]:
            raise ValueError(f"{name}: cannot weight {shape} values with {w.size} phi weights")
        return w
    if volumes is None or name not in VOLUME_WEIGHTED:
        return None
    volumes = np.asarray(volumes, dtype=float).reshape(-1)
    if not shape or volumes.size != shape[0]:
        raise ValueError(f"{name}: cannot weight {shape} values with {volumes.size} volumes")
    if len(shape) == 2 and shape[1] == 3:
        return np.repeat(volumes, 3)
    if len(shape) != 1:
        raise ValueError(f"{name}: unexpected snapshot shape {shape}")
    return volumes


def _flux_weights(
    weights: np.ndarray | None, n_internal: int, n_phi: int
) -> np.ndarray:
    if weights is None:
        return np.ones(n_internal)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size == n_phi:
        w = w[:n_internal]
    if w.size != n_internal:
        raise ValueError(
            f"expected {n_internal} internal-face weights (or {n_phi} for phi), got {w.size}"
        )
    return w


def _expand_weights(weights: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    size = int(np.prod(shape)) if shape else 1
    if weights is None:
        return np.ones(size)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size == size:
        return w
    if len(shape) == 2 and shape[1] == 3 and w.size == shape[0]:
        return np.repeat(w, 3)
    raise ValueError(f"expected {size} weights for snapshots of shape {shape}, got {w.size}")
