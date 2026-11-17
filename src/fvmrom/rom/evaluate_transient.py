"""Metrics of a transient (shedding) solution: frequency, cycle amplitude, summary."""

from __future__ import annotations

import numpy as np


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
