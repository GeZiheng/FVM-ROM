"""Metrics of a transient (shedding) solution.

Frequency, cycle amplitude and limit-cycle summary, plus the error curves,
prediction horizon, phase and energy metrics and the full report of
:func:`evaluate_transient`.  Snapshot arrays keep the time axis first.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..foam.fields import MeshTopology, phi_divergence
from ..foam.mesh import MeshGeometry
from .evaluate import _field_weights, flux_consistency, rel_l2_error


def dominant_frequency(t: np.ndarray, x: np.ndarray) -> float:
    """Frequency of the dominant component of ``x(t)``, in cycles per unit time.

    The mean is removed, a Hann window is applied and the magnitude spectrum is
    searched for its largest non-DC peak; the peak is refined by a parabola through
    the log magnitudes of the peak and its two neighbours.  ``t`` must be uniform.
    """
    t, x = _pair(t, x)
    dt = _uniform_step(t)
    y = x - np.mean(x)
    spectrum = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    freqs = np.fft.rfftfreq(y.size, d=dt)
    if spectrum.size < 3:
        raise ValueError("need at least three frequency bins")
    k = int(np.argmax(spectrum[1:])) + 1
    delta = 0.0
    if 1 <= k < spectrum.size - 1:
        a, b, c = np.log(np.maximum(spectrum[k - 1 : k + 2], np.finfo(float).tiny))
        denominator = a - 2.0 * b + c
        if denominator != 0.0:
            delta = 0.5 * (a - c) / denominator
    return float((k + delta) * freqs[1])


def cycle_amplitude(t: np.ndarray, x: np.ndarray, T: float) -> float:
    """Mean over the complete periods in the window of ``(max - min) / 2``."""
    cycles = _cycle_amplitudes(t, x, T)
    if not cycles:
        raise ValueError(f"the window is shorter than one period ({T:g})")
    return float(np.mean([amplitude for _, amplitude in cycles]))


def shedding_summary(
    t: np.ndarray,
    cl: np.ndarray,
    cd: np.ndarray | None = None,
    diameter: float = 1.0,
    velocity: float = 1.0,
    tail_periods: int = 10,
) -> dict:
    """Strouhal number, period, mean Cd, Cl amplitude and limit-cycle time.

    The frequency comes from :func:`dominant_frequency` over the whole window; the
    Cl amplitude and the mean Cd use the last ``tail_periods`` periods.  The
    limit-cycle time is the start of the first period whose Cl amplitude is within
    1% of the mean amplitude of the final periods and whose later periods stay
    within that band; it is ``None`` if the amplitude never settles.
    """
    t, cl = _pair(t, cl)
    frequency = dominant_frequency(t, cl)
    period = 1.0 / frequency
    if tail_periods < 1:
        raise ValueError("tail_periods must be positive")
    mask = t >= t[-1] - tail_periods * period
    if np.count_nonzero(mask) < 3:
        raise ValueError("fewer than three samples in the tail window")
    cd_values = None if cd is None else np.asarray(cd, dtype=float).reshape(-1)
    if cd_values is not None and cd_values.size != t.size:
        raise ValueError(f"cd has {cd_values.size} values for {t.size} times")
    return {
        "frequency": frequency,
        "period": period,
        "strouhal": frequency * diameter / velocity,
        "cl_amplitude": cycle_amplitude(t[mask], cl[mask], period),
        "cd_mean": None if cd_values is None else float(np.mean(cd_values[mask])),
        "limit_cycle_time": _limit_cycle_time(t, cl, period),
    }


def _limit_cycle_time(t: np.ndarray, x: np.ndarray, T: float) -> float | None:
    cycles = _cycle_amplitudes(t, x, T)
    if len(cycles) < 2:
        return None
    final = float(np.mean([amplitude for _, amplitude in cycles[-min(10, len(cycles)) :]]))
    if final <= 0.0:
        return None
    settled = None
    for index, (_, amplitude) in enumerate(cycles):
        if abs(amplitude - final) / final >= 0.01:
            settled = index + 1
    if settled is None:
        settled = 0
    if settled >= len(cycles):
        return None
    return float(cycles[settled][0])


def _cycle_amplitudes(t: np.ndarray, x: np.ndarray, T: float) -> list[tuple[float, float]]:
    """``(start time, (max - min)/2)`` for every complete period in the window."""
    t, x = _pair(t, x)
    if not T > 0.0:
        raise ValueError(f"the period must be positive, got {T}")
    n_cycles = int(np.floor((t[-1] - t[0]) / T + 1e-9))
    out = []
    for k in range(n_cycles):
        start = t[0] + k * T
        stop = start + T
        tol = 1e-12 * max(1.0, abs(stop))
        mask = (t >= start - tol) & (t < stop - tol)
        if k == n_cycles - 1:
            mask = (t >= start - tol) & (t <= stop + tol)
        if np.count_nonzero(mask) < 2:
            continue
        out.append((float(start), 0.5 * float(np.ptp(x[mask]))))
    return out


def _pair(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    if t.size != x.size:
        raise ValueError(f"{x.size} values for {t.size} times")
    if t.size < 8:
        raise ValueError(f"need at least eight samples, got {t.size}")
    return t, x


def _uniform_step(t: np.ndarray) -> float:
    steps = np.diff(t)
    dt = float(np.mean(steps))
    if not dt > 0.0:
        raise ValueError("times must increase")
    spread = float(np.max(np.abs(steps - dt))) / dt
    if spread > 1e-6:
        raise ValueError(f"times must be uniformly spaced (spread {spread:.1e})")
    return dt


def error_curves(
    pred: dict[str, np.ndarray],
    ref: dict[str, np.ndarray],
    volumes: np.ndarray,
    phi_weights: np.ndarray | None = None,
    mean_train: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Per-snapshot relative L2 errors of ``U``, ``p`` and ``phi``.

    The full-field errors follow the steady-ROM convention: cell volumes weight
    ``U`` and ``p``, ``phi_weights`` (or unit weights) weight ``phi``.  When
    ``mean_train`` gives the training means of ``U`` and/or ``p``, the fluctuation
    errors ``||pred - ref|| / ||ref - mean_train||`` are added as ``U_fluct`` and
    ``p_fluct``.  Every returned array has one value per snapshot.
    """
    out: dict[str, np.ndarray] = {}
    for name, reference in ref.items():
        if name not in pred:
            raise ValueError(f"the prediction has no field {name!r}")
        r = np.asarray(reference, dtype=float)
        p = np.asarray(pred[name], dtype=float)
        if p.shape != r.shape:
            raise ValueError(f"{name}: prediction shape {p.shape} differs from {r.shape}")
        weights = _field_weights(name, r.shape[1:], volumes, phi_weights)
        out[name] = rel_l2_error(p, r, weights)
        if name in ("U", "p") and mean_train is not None and name in mean_train:
            out[f"{name}_fluct"] = _fluctuation_error(p, r, mean_train[name], weights)
    return out


def prediction_horizon(
    times: np.ndarray, err: np.ndarray, tol: float, t_start: float
) -> float | None:
    """Time at which ``err`` first leaves ``tol``, relative to ``t_start``.

    A point counts as outside the tolerance when it is above ``tol`` (equality is
    inside) or when it is not finite, so a diverged trajectory has a finite
    horizon.  Returns ``None`` only when every error is finite and stays at or
    below ``tol`` in the window, which means the horizon is at least the window
    length.
    """
    t = np.asarray(times, dtype=float).reshape(-1)
    e = np.asarray(err, dtype=float).reshape(-1)
    if t.size != e.size:
        raise ValueError(f"{e.size} errors for {t.size} times")
    crossing = np.flatnonzero(~(e <= tol))
    if crossing.size == 0:
        return None
    return float(t[crossing[0]] - t_start)


def phase_difference(
    t: np.ndarray, x_pred: np.ndarray, x_ref: np.ndarray, f: float
) -> float:
    """Phase of ``x_pred`` minus the phase of ``x_ref`` at frequency ``f``, in radians.

    Both signals are least-squares fitted with ``a cos(2 pi f t) + b sin(2 pi f t) + c``
    over the whole window (no period truncation, so the window length is free);
    the phase of ``R cos(2 pi f t - phi)`` is ``phi = atan2(b, a)``.  The result is
    wrapped to ``(-pi, pi]``.  Signals without a component at ``f`` are rejected.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    xp = np.asarray(x_pred, dtype=float).reshape(-1)
    xr = np.asarray(x_ref, dtype=float).reshape(-1)
    if not (t.size == xp.size == xr.size):
        raise ValueError(f"need equal lengths, got {t.size}, {xp.size} and {xr.size}")
    if t.size < 4:
        raise ValueError(f"need at least four samples, got {t.size}")
    if not f > 0.0:
        raise ValueError(f"the frequency must be positive, got {f}")
    design = np.column_stack(
        [np.cos(2.0 * np.pi * f * t), np.sin(2.0 * np.pi * f * t), np.ones(t.size)]
    )
    phases = []
    for x in (xp, xr):
        a, b, _ = np.linalg.lstsq(design, x, rcond=None)[0]
        if np.hypot(a, b) <= 1e-12 * max(1.0, float(np.max(np.abs(x)))):
            raise ValueError(f"the signal has no component at {f:g}")
        phases.append(float(np.arctan2(b, a)))
    return _wrap_angle(phases[0] - phases[1])


def energy_drift(
    times: np.ndarray,
    a: np.ndarray,
    T: float,
    a_train_max: float | None = None,
) -> dict:
    """Cycle-averaged fluctuation energy of the latent coefficients ``a``.

    ``a`` is ``(nt, r)``.  Because the ``U`` POD modes are orthonormal in the
    cell-volume inner product, ``||a(t)||^2`` is the fluctuation energy measure
    (twice the kinetic energy ``E(t) = ||a||^2 / 2``); ``E_k`` below is the mean
    of ``||a||^2`` over one complete period.  Returns the first and last complete
    prediction-cycle values, their relative change, the largest ``||a(t)||`` and,
    when ``a_train_max`` is given, its ratio to the training maximum.
    """
    t = np.asarray(times, dtype=float).reshape(-1)
    A = np.asarray(a, dtype=float)
    if A.ndim != 2 or A.shape[0] != t.size:
        raise ValueError(f"expected coefficients of shape ({t.size}, r), got {A.shape}")
    if not T > 0.0:
        raise ValueError(f"the period must be positive, got {T}")
    values = np.sum(A**2, axis=1)
    cycles = _cycle_means(t, values, T)
    if not cycles:
        raise ValueError(f"the window is shorter than one period ({T:g})")
    first, last = cycles[0], cycles[-1]
    max_norm = float(np.max(np.linalg.norm(A, axis=1)))
    return {
        "n_cycles": len(cycles),
        "first_cycle": float(first),
        "last_cycle": float(last),
        "relative_change": float((last - first) / first) if first > 0.0 else None,
        "max_norm": max_norm,
        "train_max_norm": None if a_train_max is None else float(a_train_max),
        "max_norm_ratio": None if a_train_max is None else max_norm / float(a_train_max),
    }


def evaluate_transient(
    rom,
    times_test: np.ndarray,
    snap_test: dict[str, np.ndarray],
    volumes: np.ndarray,
    phi_weights: np.ndarray | None = None,
    *,
    period: float,
    forces_test: np.ndarray | None = None,
    mean_train: dict[str, np.ndarray] | None = None,
    f_fom: float | None = None,
    topo: MeshTopology | None = None,
    geom: MeshGeometry | None = None,
    tau_levels: Sequence[float] = (1.0, 3.0, 5.0, 10.0),
    max_tau: float = 5.0,
    horizon_tols: Sequence[float] = (1e-2, 1e-1),
    phase_taus: Sequence[float] = (1.0, 5.0, 10.0),
) -> dict:
    """Report of a fitted :class:`~fvmrom.rom.transient.TransientROM` on a held-out window.

    ``period`` is the FOM shedding period and is required: the prediction horizon
    is expressed in periods ``tau = (t - t_train_end) / period`` with
    ``t_train_end = rom.t_train[-1]``.  ``f_fom`` (the FOM Cl frequency) and
    ``forces_test`` (``(nt, 2)`` with columns Cd, Cl) are optional.  ``mean_train``
    enables the fluctuation error curves, ``topo`` the ``phi`` divergence check and
    ``geom`` the U-``phi`` flux consistency.  The function reads no files: every
    array comes from the arguments, and the returned dict holds ndarrays that
    :func:`to_jsonable` turns into plain JSON types.  Values at ``tau_levels``
    (and the phase windows) use the nearest snapshots to the requested ``tau``.
    """
    if not period > 0.0:
        raise ValueError(f"the period must be positive, got {period}")
    t = np.asarray(times_test, dtype=float).reshape(-1)
    n = t.size
    if n == 0:
        raise ValueError("times_test must not be empty")
    if rom.t_train is None:
        raise RuntimeError("the ROM is not fitted")
    t_end = float(rom.t_train[-1])
    tau = (t - t_end) / period

    ref: dict[str, np.ndarray] = {}
    for name in ("U", "p", "phi"):
        if name in snap_test:
            data = np.asarray(snap_test[name], dtype=float)
            if data.ndim == 0 or data.shape[0] != n:
                raise ValueError(f"{name}: expected {n} snapshots, got shape {data.shape}")
            ref[name] = data
    if "U" not in ref:
        raise ValueError("snap_test must contain 'U'")
    fields = [name for name in ("U", "p", "phi") if name in ref and name in rom.pods_]
    predicted = rom.predict(t)
    pred = {name: predicted[name] for name in fields}

    curves = error_curves(
        pred, {name: ref[name] for name in fields}, volumes, phi_weights, mean_train
    )
    errors_at_tau = {
        key: {_tau_key(level): _value_at_tau(tau, curve, level) for level in tau_levels}
        for key, curve in curves.items()
    }
    within = tau <= max_tau + 1e-12
    max_within = {
        key: (float(np.max(curve[within])) if np.any(within) else None)
        for key, curve in curves.items()
    }
    horizons = {
        key: {
            f"{tol:g}": prediction_horizon(t, curve, tol, t_end) for tol in horizon_tols
        }
        for key, curve in curves.items()
    }

    projection: dict[str, np.ndarray] = {}
    projection_max: dict[str, float | None] = {}
    for name in fields:
        rebuilt = rom.project(name, ref[name])
        weights = _field_weights(name, ref[name].shape[1:], volumes, phi_weights)
        projection[name] = rel_l2_error(rebuilt, ref[name], weights)
        projection_max[name] = (
            float(np.max(projection[name][within])) if np.any(within) else None
        )

    latent_pred = np.asarray(rom.latent(t), dtype=float)
    latent_true = rom.pods_["U"].project(ref["U"].reshape(n, -1))
    f_rom = _latent_frequency(t, latent_pred)
    frequency = {
        "fom": None if f_fom is None else float(f_fom),
        "rom": f_rom,
        "relative_error": (
            None if f_fom is None or f_rom is None else abs(f_rom - f_fom) / abs(f_fom)
        ),
    }

    train_max = None
    if rom.latent_train_ is not None:
        train_max = float(np.max(np.linalg.norm(rom.latent_train_, axis=1)))
    energy = {}
    for name, coefficients in (("rom", latent_pred), ("fom", latent_true)):
        try:
            energy[name] = energy_drift(t, coefficients, period, train_max)
        except ValueError:
            energy[name] = None

    scalars = None
    if forces_test is not None:
        if "forces" not in rom.scalar_shapes_:
            raise ValueError("forces_test given but the ROM has no 'forces' readout")
        forces = np.asarray(forces_test, dtype=float)
        if forces.ndim != 2 or forces.shape != (n, 2):
            raise ValueError(f"forces_test: expected ({n}, 2), got {forces.shape}")
        readout = rom.readouts_["forces"]
        on_pred = readout.predict(latent_pred)
        on_true = readout.predict(latent_true)
        scalars = {
            "forces": {
                **_forces_metrics(t, t_end, period, forces, on_pred, phase_taus),
                "attribution": {
                    "true_latent": _forces_metrics(
                        t, t_end, period, forces, on_true, phase_taus
                    ),
                    "predicted_latent": _forces_metrics(
                        t, t_end, period, forces, on_pred, phase_taus
                    ),
                    "latent_relative_error": _relative_norm(latent_pred, latent_true),
                },
            }
        }

    divergence = None
    if topo is not None and "phi" in fields:
        rom_max = _max_divergence(pred["phi"], topo)
        fom_max = _max_divergence(ref["phi"], topo)
        limit = max(100.0 * fom_max, 1e-10)
        divergence = {
            "rom_max": rom_max,
            "fom_max": fom_max,
            "limit": limit,
            "ok": bool(np.isfinite(rom_max) and rom_max <= limit),
        }

    consistency = None
    if geom is not None and {"U", "phi"} <= set(fields):
        fom_flux = flux_consistency(ref["U"], ref["phi"], geom, phi_weights)
        rom_flux = flux_consistency(pred["U"], pred["phi"], geom, phi_weights)
        mean_fom = float(np.mean(fom_flux))
        consistency = {
            "fom": {"mean": mean_fom, "max": float(np.max(fom_flux))},
            "rom": {"mean": float(np.mean(rom_flux)), "max": float(np.max(rom_flux))},
            "ratio": float(np.mean(rom_flux) / mean_fom) if mean_fom > 0.0 else None,
        }

    return {
        "n_test": int(n),
        "t_train_end": t_end,
        "period": float(period),
        "tau": tau,
        "errors": curves,
        "errors_at_tau": errors_at_tau,
        "max_error_within_tau": max_within,
        "prediction_horizon": horizons,
        "projection_error": projection,
        "projection_max_within_tau": projection_max,
        "frequency": frequency,
        "spectrum": rom.spectrum(),
        "scalars": scalars,
        "energy": energy,
        "phi_divergence": divergence,
        "flux_consistency": consistency,
    }


def to_jsonable(obj):
    """Convert ``obj`` to plain JSON types (ndarray -> list, numpy scalars -> Python).

    Non-finite floats become ``None`` so that the result is valid JSON.
    """
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (bool, int, str)) or obj is None:
        return obj
    raise TypeError(f"cannot convert {type(obj).__name__} to JSON")


def _fluctuation_error(
    pred: np.ndarray,
    ref: np.ndarray,
    mean: np.ndarray,
    weights: np.ndarray | None,
) -> np.ndarray:
    n = ref.shape[0]
    pf, rf = pred.reshape(n, -1), ref.reshape(n, -1)
    centre = np.asarray(mean, dtype=float).reshape(-1)
    if centre.size != rf.shape[1]:
        raise ValueError(f"expected {rf.shape[1]} mean values, got {centre.size}")
    scale = np.ones(rf.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    scale = np.sqrt(scale.reshape(-1))
    numerator = np.linalg.norm((pf - rf) * scale, axis=1)
    denominator = np.linalg.norm((rf - centre) * scale, axis=1)
    out = np.full(n, np.inf)
    good = denominator > 0.0
    out[good] = numerator[good] / denominator[good]
    out[~good & (numerator == 0.0)] = 0.0
    return out


def _cycle_means(t: np.ndarray, values: np.ndarray, T: float) -> list[float]:
    n_cycles = int(np.floor((t[-1] - t[0]) / T + 1e-9))
    out = []
    for k in range(n_cycles):
        start = t[0] + k * T
        stop = start + T
        tol = 1e-12 * max(1.0, abs(stop))
        mask = (t >= start - tol) & (t < stop - tol)
        if np.count_nonzero(mask) < 2:
            continue
        out.append(float(np.mean(values[mask])))
    return out


def _value_at_tau(tau: np.ndarray, values: np.ndarray, level: float) -> float | None:
    if tau.size == 0 or level < tau[0] or level > tau[-1]:
        return None
    index = int(np.argmin(np.abs(tau - level)))
    return float(values[index])


def _tau_key(level: float) -> str:
    return f"{level:g}"


def _wrap_angle(angle: float) -> float:
    wrapped = (angle + np.pi) % (2.0 * np.pi) - np.pi
    if wrapped <= -np.pi:
        wrapped += 2.0 * np.pi
    return float(wrapped)


def _latent_frequency(t: np.ndarray, a: np.ndarray) -> float | None:
    centred = a - np.mean(a, axis=0)
    energy = np.linalg.norm(centred, axis=0)
    if not np.any(energy > 0.0):
        return None
    column = int(np.argmax(energy))
    try:
        return dominant_frequency(t, a[:, column])
    except ValueError:
        return None


def _phase_at(
    t: np.ndarray,
    t_end: float,
    period: float,
    level: float,
    x_pred: np.ndarray,
    x_ref: np.ndarray,
) -> float | None:
    """Phase difference in degrees over the one-period window centred on ``level``."""
    centre = t_end + level * period
    mask = np.abs(t - centre) <= 0.5 * period * (1.0 + 1e-9)
    tp = t[mask]
    if tp.size < 4 or tp[-1] - tp[0] < 0.5 * period:
        return None
    try:
        angle = phase_difference(tp, x_pred[mask], x_ref[mask], 1.0 / period)
    except ValueError:
        return None
    return float(np.degrees(angle))


def _forces_metrics(
    t: np.ndarray,
    t_end: float,
    period: float,
    ref: np.ndarray,
    pred: np.ndarray,
    phase_taus: Sequence[float],
) -> dict:
    reference_cd, reference_cl = ref[:, 0], ref[:, 1]
    predicted_cd, predicted_cl = pred[:, 0], pred[:, 1]
    return {
        "cd_mean": _relative_pair(float(np.mean(reference_cd)), float(np.mean(predicted_cd))),
        "cl_amplitude": _relative_pair(
            _safe_amplitude(t, reference_cl, period),
            _safe_amplitude(t, predicted_cl, period),
        ),
        "cl_phase_deg": {
            _tau_key(level): _phase_at(t, t_end, period, level, predicted_cl, reference_cl)
            for level in phase_taus
        },
    }


def _relative_pair(reference: float | None, prediction: float | None) -> dict:
    error = None
    if reference is not None and prediction is not None and reference != 0.0:
        error = abs(prediction - reference) / abs(reference)
    return {"reference": reference, "prediction": prediction, "relative_error": error}


def _safe_amplitude(t: np.ndarray, x: np.ndarray, period: float) -> float | None:
    try:
        return cycle_amplitude(t, x, period)
    except ValueError:
        return None


def _relative_norm(pred: np.ndarray, ref: np.ndarray) -> float:
    denominator = float(np.linalg.norm(ref))
    if denominator == 0.0:
        return 0.0 if np.linalg.norm(pred) == 0.0 else float("inf")
    return float(np.linalg.norm(pred - ref) / denominator)


def _max_divergence(phi: np.ndarray, topo: MeshTopology) -> float:
    """Largest cell divergence over the snapshots; NaN in any snapshot gives NaN.

    The per-snapshot maxima are collected first and reduced with ``np.max``, whose
    NaN propagation keeps a diverged (NaN-filled) trajectory visible; the builtin
    ``max`` would silently skip the NaN rows.
    """
    values = np.asarray(phi, dtype=float)
    if values.ndim == 1:
        values = values[None]
    rows = [np.abs(phi_divergence(v, topo)).max() for v in values]
    return float(np.max(rows))
