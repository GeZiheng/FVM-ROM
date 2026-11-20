"""Latent-space dynamics for the transient ROM.

Every model shares the interface ``fit(t, A)`` / ``predict(t, ...)`` /
``spectrum()``: ``A`` holds one latent coefficient vector per row and ``t`` the
corresponding (uniformly spaced) sample times.  DMD-family models fit the whole
trajectory at once and evaluate it analytically, so they ignore the optional
initial state of ``predict``.  :class:`OpInf` is an ODE model instead: it is
integrated from a given initial state.
"""

from __future__ import annotations

import warnings
from typing import Any, Protocol

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, linear_sum_assignment


class Dynamics(Protocol):
    """Fit-and-predict interface for latent dynamics."""

    def fit(self, t: np.ndarray, A: np.ndarray) -> "Dynamics": ...

    def predict(
        self, t: np.ndarray, a0: np.ndarray | None = None, t0: float | None = None
    ) -> np.ndarray: ...

    def spectrum(self) -> dict | None: ...


class DMD:
    """Exact DMD of the coefficient matrix ``A`` (one snapshot per row).

    ``A`` is factorised as ``A ≈ Re Σ_j b_j φ_j exp(ω_j (t - t_first))``;
    ``predict`` evaluates that expansion, so it ignores ``a0``/``t0`` and accepts
    any times after ``t_first`` (earlier ones are evaluated as well, but the
    expansion is only meaningful inside and after the training window).
    Eigenvalues are sorted by decreasing amplitude.

    The DMD modes are scaled to unit norm and the scale is folded into the
    amplitudes, so ``spectrum()["amplitude"]`` is ``|b_j|`` for unit-norm spatial
    modes; the prediction is unchanged.  :class:`OptDMD`/``BOPDMD`` normalise the
    temporal exponentials instead and report ``‖b_j‖`` of unit-norm columns,
    which is larger by about a factor ``sqrt(nt)``: the two amplitudes are not
    comparable across the two conventions (only within OptDMD/BOPDMD).
    """

    def __init__(self, svd_rank: int | None = None):
        if svd_rank is not None and svd_rank < 1:
            raise ValueError("svd_rank must be a positive integer")
        self.svd_rank = svd_rank
        self.dt_: float | None = None
        self.t_first_: float | None = None
        self.mu_: np.ndarray | None = None
        self.omega_: np.ndarray | None = None
        self.modes_: np.ndarray | None = None
        self.amplitudes_: np.ndarray | None = None

    def fit(self, t: np.ndarray, A: np.ndarray) -> "DMD":
        t = np.asarray(t, dtype=float).reshape(-1)
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != t.size:
            raise ValueError(
                f"expected coefficients of shape ({t.size}, r), got {A.shape}"
            )
        if t.size < 3:
            raise ValueError("DMD needs at least three snapshots")
        self.dt_ = _uniform_dt(t)
        x1, x2 = A[:-1].T, A[1:].T

        u, s, vt = np.linalg.svd(x1, full_matrices=False)
        k = int(np.count_nonzero(s > s[0] * 1e-12)) if s.size and s[0] > 0.0 else 0
        if self.svd_rank is not None:
            k = min(k, self.svd_rank)
        if k == 0:
            raise ValueError("the coefficient matrix is rank deficient")
        u, s, vt = u[:, :k], s[:k], vt[:k]

        atilde = (u.T @ x2 @ vt.T) / s
        mu, w = np.linalg.eig(atilde)
        modes = x2 @ (vt.T / s) @ w
        amplitudes = np.linalg.lstsq(modes, A[0], rcond=None)[0]

        norms = np.linalg.norm(modes, axis=0)
        norms[norms == 0.0] = 1.0
        modes = modes / norms
        amplitudes = amplitudes * norms

        order = np.argsort(-np.abs(amplitudes), kind="stable")
        self.modes_ = modes[:, order]
        self.amplitudes_ = amplitudes[order]
        self.mu_ = mu[order]
        self.omega_ = np.log(self.mu_) / self.dt_
        self.t_first_ = float(t[0])
        return self

    def predict(
        self, t: np.ndarray, a0: np.ndarray | None = None, t0: float | None = None
    ) -> np.ndarray:
        self._check_fitted()
        t = np.asarray(t, dtype=float).reshape(-1)
        growth = np.exp(np.outer(t - self.t_first_, self.omega_))
        return np.real(growth @ (self.amplitudes_ * self.modes_).T)

    def spectrum(self) -> dict:
        """Return ``freq``, ``growth``, ``amplitude`` and ``eigenvalues``.

        ``eigenvalues`` are returned as ``(k, 2)`` arrays of real and imaginary
        parts so that the whole dict is JSON serialisable.
        """
        self._check_fitted()
        return {
            "freq": np.imag(self.omega_) / (2.0 * np.pi),
            "growth": np.real(self.omega_),
            "amplitude": np.abs(self.amplitudes_),
            "eigenvalues": np.column_stack([self.mu_.real, self.mu_.imag]),
        }

    def _check_fitted(self) -> None:
        if self.modes_ is None:
            raise RuntimeError("DMD is not fitted")


class OptDMD:
    """Optimized DMD of ``A`` (Askham & Kutz 2018, variable projection).

    The model is ``A ≈ Re Σ_j b_j exp(λ_j (t - t_first))`` with complex
    eigenvalues ``λ_j`` and latent amplitude vectors ``b_j``; the "spatial"
    modes of :class:`DMD` become the scalar exponentials here.  For a trial set
    of eigenvalues, ``Φ_ij = exp(λ_j (t_i - t_first))`` and the amplitudes
    follow from ``B = lstsq(Φ, A)``; ``scipy.optimize.least_squares`` (real and
    imaginary parts of the residual stacked, numerical Jacobian) minimises the
    residual ``‖A - ΦB‖_F`` over the real and imaginary parts of ``λ``.

    The optimiser is unconstrained; the result is made conjugate symmetric
    afterwards, i.e. the two members of each conjugate pair are replaced by
    their mean real part and mean absolute imaginary part (a leftover
    eigenvalue becomes real).  Real data therefore yields conjugate pairs and
    ``predict`` returns real values.

    ``rank`` fixes the number of eigenvalues taken from the initial standard
    DMD spectrum (``None`` keeps its effective rank).  Sample times may be
    non-uniform: for the initial guess DMD is then run on a uniform
    interpolation of ``A``.  ``spectrum()["eigenvalues"]`` uses the mean sample
    spacing as ``Δt``, and ``spectrum()["amplitude"]`` is ``‖b_j‖`` after every
    column of ``Φ`` is scaled to unit norm.  This amplitude is about ``sqrt(nt)``
    larger than the ``|b_j|`` of :class:`DMD` (unit-norm spatial modes), so the
    two are not comparable; OptDMD and :class:`BOPDMD` share this convention.
    """

    def __init__(self, rank: int | None = None):
        if rank is not None and rank < 1:
            raise ValueError("rank must be a positive integer")
        self.rank = rank
        self.dt_: float | None = None
        self.t_first_: float | None = None
        self.lambda_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.norms_: np.ndarray | None = None
        self.amplitudes_: np.ndarray | None = None
        self.nfev_: int | None = None
        self.success_: bool | None = None

    def fit(self, t: np.ndarray, A: np.ndarray) -> "OptDMD":
        t = np.asarray(t, dtype=float).reshape(-1)
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != t.size:
            raise ValueError(
                f"expected coefficients of shape ({t.size}, r), got {A.shape}"
            )
        if t.size < 3:
            raise ValueError("OptDMD needs at least three snapshots")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("sample times must be strictly increasing")
        self.t_first_ = float(t[0])
        self.dt_ = float((t[-1] - t[0]) / (t.size - 1))

        guess = _dmd_initial_spectrum(t, A)
        lam0 = guess.omega_ if self.rank is None else guess.omega_[: self.rank]
        lam, solution = _fit_eigenvalues(t - self.t_first_, A, lam0)
        lam = _conjugate_symmetric(lam)

        phi = np.exp(np.outer(t - self.t_first_, lam))
        amplitudes = np.linalg.lstsq(phi, A, rcond=None)[0]
        norms = np.linalg.norm(phi, axis=0)
        norms[norms == 0.0] = 1.0
        amplitudes = amplitudes * norms[:, None]

        order = np.argsort(-np.linalg.norm(amplitudes, axis=1), kind="stable")
        self.lambda_ = lam[order]
        self.eigenvalues_ = np.exp(self.lambda_ * self.dt_)
        self.norms_ = norms[order]
        self.amplitudes_ = amplitudes[order]
        self.nfev_ = int(solution.nfev)
        self.success_ = bool(solution.success)
        return self

    def predict(
        self, t: np.ndarray, a0: np.ndarray | None = None, t0: float | None = None
    ) -> np.ndarray:
        self._check_fitted()
        t = np.asarray(t, dtype=float).reshape(-1)
        growth = np.exp(np.outer(t - self.t_first_, self.lambda_)) / self.norms_
        return np.real(growth @ self.amplitudes_)

    def spectrum(self) -> dict:
        """Return ``freq``, ``growth``, ``amplitude`` and ``eigenvalues``.

        Frequencies and growth rates are those of ``λ``; ``eigenvalues`` are
        ``exp(λ Δt)`` with the mean sample spacing ``Δt``.  ``amplitude`` is
        ``‖b_j‖`` for unit-norm exponential columns.
        """
        self._check_fitted()
        return {
            "freq": np.imag(self.lambda_) / (2.0 * np.pi),
            "growth": np.real(self.lambda_),
            "amplitude": np.linalg.norm(self.amplitudes_, axis=1),
            "eigenvalues": np.column_stack(
                [self.eigenvalues_.real, self.eigenvalues_.imag]
            ),
        }

    def _check_fitted(self) -> None:
        if self.lambda_ is None:
            raise RuntimeError("OptDMD is not fitted")


class BOPDMD:
    """Bagging Optimized DMD (Sashidhar & Kutz 2022).

    First a full :class:`OptDMD` fit of every training sample provides the
    reference eigenvalues ``λ`` and coefficients ``B`` (the rows of ``B`` are the
    complex amplitude vectors ``b_j`` of the model ``A ≈ Re Σ_j exp(λ_j τ) b_j``,
    ``τ = t - t_first``).  Then ``n_trials`` subsets of ``trial_fraction`` of the
    samples (drawn without replacement, indices sorted) are fitted with the same
    variable projection, starting from the reference eigenvalues.  The result is
    the mean of ``λ`` and ``B`` over the trials, with their standard deviations.

    Before averaging, every trial is aligned to the reference:

    - the eigenvalues are matched one-to-one to the reference ones by nearest
      distance (linear assignment); a trial whose count differs or whose largest
      matching distance exceeds ``5%`` of the largest reference ``|λ|`` is
      discarded;
    - each amplitude vector is rotated so that its inner product with the
      reference vector is real and positive.  A trial's coefficients ``b`` are
      only meaningful together with its own ``λ``; the phase alignment makes the
      average of ``b`` coherent, and the mean ``λ`` defines the exponentials of
      the averaged model.

    A trial is also discarded when ``least_squares`` reports no success, so that
    ``n_used`` (reported by :meth:`spectrum`) can be smaller than ``n_trials``;
    the reference fit is always kept and used for the initial guess only.  A
    ``RuntimeError`` is raised if no trial survives.

    ``predict`` uses the mean ``λ`` and the mean ``B`` exactly like
    :class:`OptDMD` and ignores ``a0``/``t0``.  ``spectrum()`` has the
    :class:`OptDMD` keys plus ``freq_std``, ``growth_std``, ``amplitude_std`` and
    ``n_used``; ``amplitude`` and its standard deviation are taken over the
    trials (unit-norm exponential columns on the full training times, the
    OptDMD convention).  ``reference_nfev_``/``reference_success_`` are the
    optimiser counters of the full-data reference fit.  ``rank``, like in
    :class:`OptDMD`, limits the number of eigenvalues; sample times may be
    non-uniform.  Fitting twice with the same ``seed`` gives bit-identical
    results.
    """

    def __init__(
        self,
        rank: int | None = None,
        n_trials: int = 100,
        trial_fraction: float = 0.8,
        seed: int = 0,
    ):
        if rank is not None and rank < 1:
            raise ValueError("rank must be a positive integer")
        if isinstance(n_trials, bool) or not isinstance(n_trials, (int, np.integer)):
            raise ValueError("n_trials must be a positive integer")
        if n_trials < 1:
            raise ValueError("n_trials must be a positive integer")
        if not 0.0 < float(trial_fraction) <= 1.0:
            raise ValueError("trial_fraction must lie in (0, 1]")
        self.rank = rank
        self.n_trials = int(n_trials)
        self.trial_fraction = float(trial_fraction)
        self.seed = seed
        self.dt_: float | None = None
        self.t_first_: float | None = None
        self.lambda_: np.ndarray | None = None
        self.lambda_std_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.norms_: np.ndarray | None = None
        self.amplitudes_: np.ndarray | None = None
        self.amplitude_: np.ndarray | None = None
        self.freq_std_: np.ndarray | None = None
        self.growth_std_: np.ndarray | None = None
        self.amplitude_std_: np.ndarray | None = None
        self.n_used_: int | None = None
        self.n_rejected_: int | None = None
        self.reference_nfev_: int | None = None
        self.reference_success_: bool | None = None

    def fit(self, t: np.ndarray, A: np.ndarray) -> "BOPDMD":
        t = np.asarray(t, dtype=float).reshape(-1)
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != t.size:
            raise ValueError(
                f"expected coefficients of shape ({t.size}, r), got {A.shape}"
            )
        if t.size < 3:
            raise ValueError("BOPDMD needs at least three snapshots")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("sample times must be strictly increasing")
        self.t_first_ = float(t[0])
        self.dt_ = float((t[-1] - t[0]) / (t.size - 1))
        tau = t - self.t_first_

        reference = OptDMD(rank=self.rank).fit(t, A)
        lam_ref = np.array(reference.lambda_, dtype=complex)
        coef_ref = reference.amplitudes_ / reference.norms_[:, None]
        self.reference_nfev_ = reference.nfev_
        self.reference_success_ = reference.success_
        k = lam_ref.size
        size = int(round(self.trial_fraction * t.size))
        if size < max(3, k):
            raise ValueError(
                f"a trial of {self.trial_fraction:g} of {t.size} samples gives {size} "
                f"samples, fewer than the {max(3, k)} needed"
            )
        tol = _bop_match_tolerance(lam_ref)
        rng = np.random.default_rng(self.seed)

        lam_trials, coef_trials = [], []
        rejected = 0
        for _ in range(self.n_trials):
            indices = np.sort(rng.choice(t.size, size=size, replace=False))
            tau_sub, target = tau[indices], A[indices]
            lam, solution = _fit_eigenvalues(tau_sub, target, lam_ref)
            if not solution.success:
                rejected += 1
                continue
            lam = _conjugate_symmetric(lam)
            coef = np.linalg.lstsq(np.exp(np.outer(tau_sub, lam)), target, rcond=None)[0]
            matched = _match_eigenvalues(lam, lam_ref, tol)
            if matched is None:
                rejected += 1
                continue
            lam_trials.append(lam[matched])
            coef_trials.append(_align_amplitudes(coef[matched], coef_ref))
        if not lam_trials:
            raise RuntimeError(
                f"all {self.n_trials} BOP-DMD trials were rejected "
                f"(success or reference-alignment check failed)"
            )

        lam_all = np.array(lam_trials)
        coef_all = np.array(coef_trials)
        self.lambda_ = lam_all.mean(axis=0)
        self.lambda_std_ = lam_all.std(axis=0)
        self.norms_ = np.linalg.norm(np.exp(np.outer(tau, self.lambda_)), axis=0)
        self.norms_[self.norms_ == 0.0] = 1.0
        self.amplitudes_ = coef_all.mean(axis=0) * self.norms_[:, None]
        self.eigenvalues_ = np.exp(self.lambda_ * self.dt_)

        column_norms = np.exp(np.outer(tau, (2.0 * lam_all.real).ravel()))
        column_norms = column_norms.reshape(tau.size, *lam_all.shape).sum(axis=0)
        trial_amplitudes = np.linalg.norm(coef_all, axis=2) * np.sqrt(column_norms)
        self.amplitude_ = trial_amplitudes.mean(axis=0)
        self.amplitude_std_ = trial_amplitudes.std(axis=0)
        self.freq_std_ = np.std(lam_all.imag / (2.0 * np.pi), axis=0)
        self.growth_std_ = np.std(lam_all.real, axis=0)
        self.n_used_ = int(lam_all.shape[0])
        self.n_rejected_ = int(rejected)
        return self

    def predict(
        self, t: np.ndarray, a0: np.ndarray | None = None, t0: float | None = None
    ) -> np.ndarray:
        self._check_fitted()
        t = np.asarray(t, dtype=float).reshape(-1)
        growth = np.exp(np.outer(t - self.t_first_, self.lambda_)) / self.norms_
        return np.real(growth @ self.amplitudes_)

    def spectrum(self) -> dict:
        """Return the OptDMD keys plus the trial standard deviations and ``n_used``.

        ``freq``, ``growth`` and ``eigenvalues`` are those of the mean ``λ``;
        ``amplitude`` is the mean of the per-trial amplitudes (``‖b_j‖`` for
        unit-norm exponential columns on the full training times) and
        ``amplitude_std`` its standard deviation.  ``n_used`` is the number of
        trials that survived the alignment/convergence checks.
        """
        self._check_fitted()
        return {
            "freq": np.imag(self.lambda_) / (2.0 * np.pi),
            "growth": np.real(self.lambda_),
            "amplitude": self.amplitude_,
            "eigenvalues": np.column_stack(
                [self.eigenvalues_.real, self.eigenvalues_.imag]
            ),
            "freq_std": self.freq_std_,
            "growth_std": self.growth_std_,
            "amplitude_std": self.amplitude_std_,
            "n_used": self.n_used_,
        }

    def _check_fitted(self) -> None:
        if self.lambda_ is None:
            raise RuntimeError("BOPDMD is not fitted")


#: tolerances of the OpInf trajectory integration
_INTEGRATION_RTOL = 1e-9
_INTEGRATION_ATOL = 1e-11

#: floor of the training norm used as the divergence threshold base
_NORM_FLOOR = 1e-12

#: minimum number of samples of the period estimate used by the OpInf recheck
_MIN_PERIOD_SAMPLES = 8


def _dominant_period(t: np.ndarray, A: np.ndarray) -> float:
    """Dominant period of the first coefficient column, ``NaN`` if not estimable.

    Hann-windowed FFT peak with parabolic refinement in log magnitude (the same
    peak search as the frequency metric of the evaluation module, which is not
    imported to avoid a module cycle).  ``NaN`` is returned when the window has
    fewer than eight samples, when the column is constant, or when the peak lies
    in the first non-DC bin (fewer than two cycles in the window).
    """
    span = float(t[-1] - t[0])
    if t.size < _MIN_PERIOD_SAMPLES:
        return float("nan")
    x = np.asarray(A[:, 0], dtype=float)
    x = x - np.mean(x)
    if not np.any(x != 0.0):
        return float("nan")
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    if spectrum.size < 3:
        return float("nan")
    k = int(np.argmax(spectrum[1:])) + 1
    if k < 2:
        return float("nan")
    delta = 0.0
    if k < spectrum.size - 1:
        a, b, c = np.log(np.maximum(spectrum[k - 1 : k + 2], np.finfo(float).tiny))
        denominator = a - 2.0 * b + c
        if denominator != 0.0:
            delta = 0.5 * (a - c) / denominator
    dt = float(t[1] - t[0])
    frequency = float((k + delta) * np.fft.rfftfreq(x.size, d=dt)[1])
    if not np.isfinite(frequency) or frequency <= 0.0:
        return float("nan")
    period = 1.0 / frequency
    if period > 2.0 * span:
        return float("nan")
    return float(period)


def _max_norm_ratio(values: np.ndarray, reference: float) -> float:
    """Largest row norm of ``values`` relative to ``reference`` (``NaN`` if empty)."""
    norms = np.linalg.norm(values, axis=1)
    norms = norms[np.isfinite(norms)]
    if norms.size == 0:
        return float("nan")
    return float(norms.max() / max(float(reference), _NORM_FLOOR))


class OpInf:
    """Operator Inference of a quadratic latent model (Peherstorfer & Willcox).

    The model is ``da/dt = c + A a + H (a ⊗ a)``, where ``a ⊗ a`` keeps the
    upper triangle only (``a_i a_j`` with ``i <= j``, the order of
    ``np.triu_indices(r)``), so ``H`` is ``(r, r(r+1)/2)``: row ``k`` of ``H``
    holds the coefficients of ``d a_k / dt`` for the quadratic monomials in that
    order.  ``a`` are the centred POD coefficients, so ``c`` absorbs the offset
    of the mean; a non-zero ``c`` is not an error.

    Derivatives come from the snapshots only: fourth-order central differences
    inside the training window and fourth-order one-sided differences at both
    ends (:func:`_fourth_order_derivative`), so at least five snapshots are
    required and the times must be uniformly spaced.  The fit solves

        ``min ‖D O - Ȧ‖² + λ‖O‖²``,  ``D = [1, a, a ⊗ a]``,  ``O = [c; Aᵀ; Hᵀ]``

    on columns scaled to unit Euclidean norm (``scale_``); the coefficients are
    mapped back to the original scale, so the model is unaffected by the column
    scaling.

    ``ridge`` is a **dimensionless** ridge coefficient: the design columns are
    scaled to unit Euclidean norm and the penalty of the scaled problem is
    ``ridge × ‖D_scaled‖²_F / n_cols`` (the mean squared column norm, which is
    ``1`` for unit-norm columns), so with the default scaling ``ridge_`` equals
    ``ridge`` and the grid is directly the penalty on unit columns.
    ``ridge=None`` (default) selects it from ``ridge_grid`` (default
    ``logspace(-8, 2, 11)``, the same dimensionless parametrisation) by forward
    holdout validation on the training window only: the first 75% of the
    samples (at least five) are fitted, the model is integrated from the true
    state at the start of the holdout segment up to the end of the window, and
    the relative error of the latent trajectory is compared.  A candidate whose
    trajectory exceeds ``blowup_factor`` (default 5) times the largest training
    norm, or whose integration fails, is rejected; the best surviving candidate
    is used and the whole window is refitted with it.  When every candidate is
    rejected the largest ridge is used with a ``RuntimeWarning``.  The selected
    dimensionless value is ``ridge_rel_``, the absolute penalty used with the
    full window is ``ridge_``, and ``ridge_path_`` holds the candidates
    (``ridge_rel``, ``ridge``, ``holdout_error`` and a ``stable`` mask; the
    error is ``NaN`` for a diverged candidate).  Passing ``ridge`` skips the
    selection and sets ``ridge_path_ = None``.

    The refit on the whole window is checked with training data only
    (``recheck=True``, the default; ``recheck=False`` keeps the plain
    selection): the model is replayed from the first training state over the
    whole training window, then integrated forward from the last training
    state for ``check_periods`` (default 5) periods.  The period is the
    dominant one of the first coefficient column (``_dominant_period``); when it
    cannot be estimated, e.g. on a very short window, the training window length
    is used instead, and ``check_info_["period_estimated"]`` is ``False``.  A
    trajectory whose norm exceeds ``check_blowup_factor`` (default: the
    selection ``blowup_factor``) times the largest training norm, or an
    integration failure, marks the candidate as unstable.  The ridge is then
    raised step by step to the next larger values of ``ridge_grid`` (refitting
    and rechecking each) until one is stable; when even the largest ridge is
    unstable it is used with a ``RuntimeWarning``.  ``ridge_selected_`` is the
    holdout value before any fallback, ``refit_fallback_`` says whether the
    final ``ridge_rel_``/``ridge_`` differ from it, and ``recheck_path_``
    records every attempt (``ridge_rel``, ``ridge``, a ``stable`` mask,
    ``max_norm_ratio_replay``, ``max_norm_ratio_forward``).  ``check_info_``
    holds ``check_periods``, the period ``T``, ``period_estimated`` and the
    ``blowup_factor`` used by the check.  The check roughly adds one replay plus
    one forward integration per attempted ridge to the fit time.  Passing
    ``ridge`` disables the selection **and** the check, so the user's value is
    used as is (``recheck_path_`` and ``check_info_`` are then ``None``).

    ``predict`` integrates the model with ``solve_ivp`` (DOP853,
    ``rtol=1e-9``, ``atol=1e-11``) from ``(t0, a0)``; the defaults are the last
    training time and the last training coefficient, i.e. the ODE starts from
    the true end state of the training window.  The requested times must not
    decrease and must satisfy ``t >= t0``.  Divergence (``‖a‖`` above
    ``predict_blowup_factor`` (default 10) times the largest training norm, or
    a failed integration) is not an exception: the result keeps the shape
    ``(len(t), r)`` with ``NaN`` from the failure point on, and ``diverged_`` /
    ``diverged_at_`` report it.  ``spectrum`` returns ``None`` because this is
    not an eigenvalue model; ``linear_eigenvalues`` gives the eigenvalues of the
    linear part ``A`` for information only.
    """

    def __init__(
        self,
        ridge: float | None = None,
        ridge_grid: np.ndarray | None = None,
        blowup_factor: float = 5.0,
        predict_blowup_factor: float = 10.0,
        recheck: bool = True,
        check_periods: float = 5.0,
        check_blowup_factor: float | None = None,
    ):
        if ridge is not None and ridge < 0.0:
            raise ValueError("ridge must be non-negative or None")
        if blowup_factor <= 0.0:
            raise ValueError("blowup_factor must be positive")
        if predict_blowup_factor <= 0.0:
            raise ValueError("predict_blowup_factor must be positive")
        if check_periods <= 0.0:
            raise ValueError("check_periods must be positive")
        if check_blowup_factor is not None and check_blowup_factor <= 0.0:
            raise ValueError("check_blowup_factor must be positive")
        if ridge_grid is None:
            grid = np.logspace(-8.0, 2.0, 11)
        else:
            grid = np.asarray(ridge_grid, dtype=float).reshape(-1)
            if (
                grid.size == 0
                or not np.all(np.isfinite(grid))
                or np.any(grid <= 0.0)
            ):
                raise ValueError("ridge_grid must contain positive finite values")
        self.ridge = None if ridge is None else float(ridge)
        self.ridge_grid = grid
        self.blowup_factor = float(blowup_factor)
        self.predict_blowup_factor = float(predict_blowup_factor)
        self.recheck = bool(recheck)
        self.check_periods = float(check_periods)
        self.check_blowup_factor = (
            None if check_blowup_factor is None else float(check_blowup_factor)
        )
        self.dt_: float | None = None
        self.t_first_: float | None = None
        self.t_last_: float | None = None
        self.a_first_: np.ndarray | None = None
        self.a_last_: np.ndarray | None = None
        self.a_max_norm_: float | None = None
        self.r_: int = 0
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.c_: np.ndarray | None = None
        self.A_: np.ndarray | None = None
        self.H_: np.ndarray | None = None
        self.ridge_rel_: float | None = None
        self.ridge_: float | None = None
        self.ridge_path_: dict | None = None
        self.ridge_selected_: float | None = None
        self.refit_fallback_: bool = False
        self.recheck_path_: dict | None = None
        self.check_info_: dict | None = None
        self.diverged_: bool | None = None
        self.diverged_at_: float | None = None

    def fit(self, t: np.ndarray, A: np.ndarray) -> "OpInf":
        t = np.asarray(t, dtype=float).reshape(-1)
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != t.size:
            raise ValueError(
                f"expected coefficients of shape ({t.size}, r), got {A.shape}"
            )
        if t.size < 5:
            raise ValueError(
                "OpInf needs at least five snapshots for fourth-order differences"
            )
        if not np.all(np.isfinite(A)):
            raise ValueError("coefficients contain non-finite values")
        self.dt_ = _uniform_dt(t)
        self.t_first_ = float(t[0])
        self.t_last_ = float(t[-1])
        self.a_first_ = A[0].copy()
        self.a_last_ = A[-1].copy()
        self.r_ = A.shape[1]
        self.a_max_norm_ = float(np.max(np.linalg.norm(A, axis=1)))
        self.diverged_ = False
        self.diverged_at_ = None
        self.ridge_selected_ = None
        self.refit_fallback_ = False
        self.recheck_path_ = None
        self.check_info_ = None

        if self.ridge is None:
            coef = self._fit_with_selection(t, A)
            if self.recheck:
                coef = self._recheck_and_fallback(t, A, coef)
        else:
            coef, self.scale_, self.ridge_ = _ridge_solve(
                _design_matrix(A), _fourth_order_derivative(A, self.dt_), self.ridge
            )
            self.ridge_rel_ = self.ridge
            self.ridge_path_ = None

        self.coef_ = coef
        self.c_ = coef[0].copy()
        self.A_ = coef[1 : 1 + self.r_].T.copy()
        self.H_ = coef[1 + self.r_ :].T.copy()
        return self

    def predict(
        self, t: np.ndarray, a0: np.ndarray | None = None, t0: float | None = None
    ) -> np.ndarray:
        self._check_fitted()
        t = np.asarray(t, dtype=float).reshape(-1)
        if t.size == 0:
            raise ValueError("times must not be empty")
        if not np.all(np.isfinite(t)):
            raise ValueError("times must be finite")
        if np.any(np.diff(t) < 0.0):
            raise ValueError("times must be non-decreasing")
        start = self.t_last_ if t0 is None else float(t0)
        state = self.a_last_ if a0 is None else np.asarray(a0, dtype=float).reshape(-1)
        if state.size != self.r_:
            raise ValueError(f"expected a0 of length {self.r_}, got {state.size}")
        if not np.all(np.isfinite(state)):
            raise ValueError("a0 must be finite")
        if np.any(t < start):
            raise ValueError(f"times must not precede t0={start:g}")

        unique, inverse = np.unique(t, return_inverse=True)
        values, diverged, diverged_at = self._integrate(
            self.coef_, start, state, unique, self._threshold(self.predict_blowup_factor)
        )
        self.diverged_ = bool(diverged)
        self.diverged_at_ = diverged_at
        return values[inverse]

    def spectrum(self) -> None:
        """Return ``None``: OpInf is an ODE model, not an eigenvalue model."""
        self._check_fitted()
        return None

    def linear_eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the linear part ``A`` (auxiliary information only)."""
        self._check_fitted()
        return np.linalg.eigvals(self.A_)

    def _fit_with_selection(self, t: np.ndarray, A: np.ndarray) -> np.ndarray:
        n_fit = int(round(0.75 * t.size))
        if n_fit < 5 or n_fit >= t.size:
            raise ValueError(
                f"automatic ridge selection needs at least eight snapshots "
                f"(got {t.size}); pass ridge=... to fix it explicitly"
            )
        design = _design_matrix(A[:n_fit])
        derivative = _fourth_order_derivative(A[:n_fit], self.dt_)
        threshold = self._threshold(self.blowup_factor)
        errors = np.full(self.ridge_grid.size, np.nan)
        stable = np.zeros(self.ridge_grid.size, dtype=bool)
        penalties = np.empty(self.ridge_grid.size)
        for k, rel in enumerate(self.ridge_grid):
            coef, _, penalties[k] = _ridge_solve(design, derivative, rel)
            errors[k], stable[k] = self._holdout_result(
                t, A, n_fit, coef, threshold
            )

        if stable.any():
            errors_stable = errors[stable]
            rel = float(self.ridge_grid[np.flatnonzero(stable)[np.argmin(errors_stable)]])
        else:
            rel = float(self.ridge_grid[-1])
            warnings.warn(
                "all ridge candidates diverged on the holdout window; "
                f"using the largest ridge {rel:g}",
                RuntimeWarning,
                stacklevel=3,
            )

        coef, self.scale_, self.ridge_ = _ridge_solve(
            _design_matrix(A), _fourth_order_derivative(A, self.dt_), rel
        )
        self.ridge_rel_ = rel
        self.ridge_selected_ = rel
        self.ridge_path_ = {
            "ridge_rel": self.ridge_grid,
            "ridge": penalties,
            "holdout_error": errors,
            "stable": stable,
        }
        return coef

    def _recheck_and_fallback(
        self, t: np.ndarray, A: np.ndarray, coef: np.ndarray
    ) -> np.ndarray:
        """Recheck the refitted model and raise the ridge until it is stable."""
        period = _dominant_period(t, A)
        estimated = bool(np.isfinite(period) and period > 0.0)
        if not estimated:
            period = float(t[-1] - t[0])
        factor = (
            self.blowup_factor
            if self.check_blowup_factor is None
            else self.check_blowup_factor
        )
        self.check_info_ = {
            "check_periods": self.check_periods,
            "period": float(period),
            "period_estimated": estimated,
            "blowup_factor": float(factor),
        }

        records = {
            "ridge_rel": [],
            "ridge": [],
            "stable": [],
            "max_norm_ratio_replay": [],
            "max_norm_ratio_forward": [],
        }

        def attempt(rel: float, penalty: float, candidate: np.ndarray) -> bool:
            stable, replay, forward = self._recheck(t, A, candidate, factor)
            records["ridge_rel"].append(rel)
            records["ridge"].append(penalty)
            records["stable"].append(stable)
            records["max_norm_ratio_replay"].append(replay)
            records["max_norm_ratio_forward"].append(forward)
            return stable

        selected = float(self.ridge_selected_)
        stable = attempt(selected, float(self.ridge_), coef)
        if not stable:
            design = _design_matrix(A)
            derivative = _fourth_order_derivative(A, self.dt_)
            candidates = np.sort(self.ridge_grid[self.ridge_grid > selected])
            for rel in candidates:
                candidate, scale, penalty = _ridge_solve(design, derivative, rel)
                if attempt(float(rel), penalty, candidate):
                    coef, self.scale_, self.ridge_ = candidate, scale, penalty
                    self.ridge_rel_ = float(rel)
                    self.refit_fallback_ = True
                    break
            else:
                largest = float(np.max(self.ridge_grid))
                if candidates.size:
                    coef, self.scale_, self.ridge_ = _ridge_solve(
                        design, derivative, largest
                    )
                    self.ridge_rel_ = largest
                    self.refit_fallback_ = True
                    warnings.warn(
                        "OpInf stayed unstable when refitted with the selected "
                        f"ridge {selected:g} and with every larger ridge; "
                        f"using the largest ridge {largest:g}",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                else:
                    warnings.warn(
                        "OpInf stayed unstable when refitted with the selected "
                        f"ridge {selected:g}, which is already the largest "
                        "candidate; no larger ridge is available",
                        RuntimeWarning,
                        stacklevel=3,
                    )

        self.recheck_path_ = {
            "ridge_rel": np.asarray(records["ridge_rel"], dtype=float),
            "ridge": np.asarray(records["ridge"], dtype=float),
            "stable": np.asarray(records["stable"], dtype=bool),
            "max_norm_ratio_replay": np.asarray(
                records["max_norm_ratio_replay"], dtype=float
            ),
            "max_norm_ratio_forward": np.asarray(
                records["max_norm_ratio_forward"], dtype=float
            ),
        }
        return coef

    def _recheck(
        self,
        t: np.ndarray,
        A: np.ndarray,
        coef: np.ndarray,
        factor: float,
    ) -> tuple[bool, float, float]:
        """Replay the training window and integrate forward; report stability."""
        threshold = self._threshold(factor)
        replay, diverged_replay, _ = self._integrate(
            coef, float(t[0]), A[0], t, threshold
        )
        span = self.check_periods * self.check_info_["period"]
        steps = max(1, int(round(span / self.dt_)))
        forward_times = self.t_last_ + np.linspace(span / steps, span, steps)
        forward, diverged_forward, _ = self._integrate(
            coef, float(t[-1]), A[-1], forward_times, threshold
        )
        return (
            not (diverged_replay or diverged_forward),
            _max_norm_ratio(replay, self.a_max_norm_),
            _max_norm_ratio(forward, self.a_max_norm_),
        )

    def _holdout_result(
        self,
        t: np.ndarray,
        A: np.ndarray,
        n_fit: int,
        coef: np.ndarray,
        threshold: float,
    ) -> tuple[float, bool]:
        """Relative holdout error of one candidate and whether it stayed stable."""
        target = A[n_fit:]
        values, diverged, _ = self._integrate(
            coef, float(t[n_fit - 1]), A[n_fit - 1], t[n_fit:], threshold
        )
        if diverged:
            return float("nan"), False
        scale = float(np.linalg.norm(target))
        if scale == 0.0:
            return float("nan"), True
        return float(np.linalg.norm(values - target) / scale), True

    def _integrate(
        self,
        coef: np.ndarray,
        t0: float,
        a0: np.ndarray,
        t_eval: np.ndarray,
        threshold: float,
    ) -> tuple[np.ndarray, bool, float | None]:
        """Integrate one trajectory; ``NaN`` after the failure point."""
        r = self.r_
        values = np.full((t_eval.size, r), np.nan)
        if t_eval.size == 0:
            return values, False, None
        if np.linalg.norm(a0) > threshold:
            return values, True, float(t0)
        if t_eval[-1] <= t0:
            values[:] = a0
            return values, False, None

        constant = coef[0]
        linear = coef[1 : 1 + r].T
        quadratic = coef[1 + r :].T

        def rhs(_t: float, a: np.ndarray) -> np.ndarray:
            return constant + linear @ a + quadratic @ _upper_kron(a)

        def blowup(_t: float, a: np.ndarray) -> float:
            return threshold - np.linalg.norm(a)

        blowup.terminal = True
        blowup.direction = -1
        solution = solve_ivp(
            rhs,
            (t0, float(t_eval[-1])),
            a0,
            method="DOP853",
            t_eval=t_eval,
            rtol=_INTEGRATION_RTOL,
            atol=_INTEGRATION_ATOL,
            events=blowup,
        )
        stored = int(np.size(solution.t))
        if stored:
            values[:stored] = np.asarray(solution.y, dtype=float).T
        if solution.t_events and solution.t_events[0].size:
            return values, True, float(solution.t_events[0][0])
        if not solution.success:
            last = float(np.ravel(solution.t)[-1]) if stored else float(t0)
            return values, True, last
        return values, False, None

    def _threshold(self, factor: float) -> float:
        return factor * max(float(self.a_max_norm_), _NORM_FLOOR)

    def _check_fitted(self) -> None:
        if self.coef_ is None:
            raise RuntimeError("OpInf is not fitted")


#: dynamics names accepted by :func:`make_dynamics`
DYNAMICS_KINDS = {"dmd": DMD, "optdmd": OptDMD, "bopdmd": BOPDMD, "opinf": OpInf}


def make_dynamics(kind: str, **kw) -> Dynamics:
    try:
        cls = DYNAMICS_KINDS[kind.lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"unknown dynamics {kind!r} (choose from {sorted(DYNAMICS_KINDS)})"
        ) from None
    return cls(**kw)


def _upper_kron(a: np.ndarray) -> np.ndarray:
    """Upper triangle of ``a ⊗ a``, in ``np.triu_indices`` order."""
    a = np.asarray(a, dtype=float)
    i, j = np.triu_indices(a.shape[-1])
    return a[..., i] * a[..., j]


def _design_matrix(a: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((a.shape[0], 1)), a, _upper_kron(a)])


def _fourth_order_derivative(A: np.ndarray, dt: float) -> np.ndarray:
    """Time derivative of every row of ``A`` (uniform spacing ``dt``).

    Fourth-order central differences in the interior and fourth-order one-sided
    differences at the two ends, so ``A`` needs at least five rows.
    """
    if A.shape[0] < 5:
        raise ValueError("fourth-order differences need at least five rows")
    d = np.empty_like(A)
    d[2:-2] = (-A[4:] + 8.0 * A[3:-1] - 8.0 * A[1:-3] + A[:-4]) / (12.0 * dt)
    d[0] = (-25.0 * A[0] + 48.0 * A[1] - 36.0 * A[2] + 16.0 * A[3] - 3.0 * A[4]) / (
        12.0 * dt
    )
    d[1] = (-3.0 * A[0] - 10.0 * A[1] + 18.0 * A[2] - 6.0 * A[3] + A[4]) / (12.0 * dt)
    d[-2] = (3.0 * A[-1] + 10.0 * A[-2] - 18.0 * A[-3] + 6.0 * A[-4] - A[-5]) / (
        12.0 * dt
    )
    d[-1] = (
        25.0 * A[-1] - 48.0 * A[-2] + 36.0 * A[-3] - 16.0 * A[-4] + 3.0 * A[-5]
    ) / (12.0 * dt)
    return d


def _ridge_solve(
    design: np.ndarray, derivative: np.ndarray, ridge_rel: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Ridge solve on unit-norm columns; returns coefficients, scales and ``λ``."""
    n_cols = design.shape[1]
    scale = np.linalg.norm(design, axis=0)
    scale[scale == 0.0] = 1.0
    scaled = design / scale
    ridge_abs = float(ridge_rel) * float(np.sum(scaled**2) / n_cols)
    if ridge_abs > 0.0:
        scaled = np.vstack([scaled, np.sqrt(ridge_abs) * np.eye(n_cols)])
        derivative = np.vstack([derivative, np.zeros((n_cols, derivative.shape[1]))])
    coef = np.linalg.lstsq(scaled, derivative, rcond=None)[0] / scale[:, None]
    return coef, scale, ridge_abs


def _uniform_dt(t: np.ndarray) -> float:
    dt = t[1] - t[0]
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("sample times must be strictly increasing")
    if np.max(np.abs(np.diff(t) - dt)) > 1e-6 * abs(dt):
        raise ValueError("sample times must be uniformly spaced")
    return float(dt)


def _dmd_initial_spectrum(t: np.ndarray, A: np.ndarray) -> DMD:
    """Standard DMD spectrum of ``A``, the starting point of Optimized DMD.

    DMD needs uniform samples; non-uniform times are interpolated onto a
    uniform grid with the mean spacing first (initial guess only).
    """
    steps = np.diff(t)
    if np.max(np.abs(steps - steps[0])) <= 1e-6 * abs(steps[0]):
        return DMD().fit(t, A)
    uniform = t[0] + np.arange(t.size) * (t[-1] - t[0]) / (t.size - 1)
    interpolated = np.column_stack(
        [np.interp(uniform, t, A[:, j]) for j in range(A.shape[1])]
    )
    return DMD().fit(uniform, interpolated)


def _fit_eigenvalues(
    tau: np.ndarray, A: np.ndarray, lam0: np.ndarray
) -> tuple[np.ndarray, Any]:
    """Variable-projection fit of the eigenvalues on ``tau = t - t_first``."""
    k = lam0.size
    if k == 0:
        raise ValueError("no eigenvalues to optimise")
    target = A

    def residuals(x: np.ndarray) -> np.ndarray:
        lam = x[:k] + 1j * x[k:]
        phi = np.exp(np.outer(tau, lam))
        coef = np.linalg.lstsq(phi, target, rcond=None)[0]
        res = target - phi @ coef
        return np.concatenate([res.real.ravel(), res.imag.ravel()])

    solution = least_squares(
        residuals, np.concatenate([lam0.real, lam0.imag]), method="trf", x_scale="jac"
    )
    return solution.x[:k] + 1j * solution.x[k:], solution


def _conjugate_symmetric(lam: np.ndarray) -> np.ndarray:
    """Pair eigenvalues into conjugate pairs (or real ones) after the fit."""
    lam = np.asarray(lam, dtype=complex)
    tol = 1e-8 * max(1.0, float(np.max(np.abs(lam))))
    out = np.empty_like(lam)
    used = np.zeros(lam.size, dtype=bool)
    for i in range(lam.size):
        if used[i]:
            continue
        used[i] = True
        if abs(lam[i].imag) <= tol:
            out[i] = complex(lam[i].real, 0.0)
            continue
        candidates = np.flatnonzero(
            ~used & (np.sign(lam.imag) == -np.sign(lam[i].imag))
        )
        if candidates.size == 0:
            out[i] = complex(lam[i].real, 0.0)
            continue
        j = candidates[np.argmin(np.abs(lam[candidates] - np.conj(lam[i])))]
        used[j] = True
        sigma = 0.5 * (lam[i].real + lam[j].real)
        omega = 0.5 * (abs(lam[i].imag) + abs(lam[j].imag))
        out[i] = sigma + 1j * omega
        out[j] = sigma - 1j * omega
    return out


def _bop_match_tolerance(lam_ref: np.ndarray) -> float:
    """Largest accepted distance between a trial and a reference eigenvalue."""
    return 0.05 * max(1.0, float(np.max(np.abs(lam_ref))))


def _match_eigenvalues(
    lam: np.ndarray, lam_ref: np.ndarray, tol: float
) -> np.ndarray | None:
    """Permutation that matches ``lam`` to ``lam_ref``, or ``None`` on failure."""
    if lam.size != lam_ref.size:
        return None
    cost = np.abs(lam[:, None] - lam_ref[None, :])
    rows, cols = linear_sum_assignment(cost)
    if np.max(cost[rows, cols]) > tol:
        return None
    perm = np.empty(lam.size, dtype=int)
    perm[cols] = rows
    return perm


def _align_amplitudes(coef: np.ndarray, coef_ref: np.ndarray) -> np.ndarray:
    """Rotate each row of ``coef`` so that it is in phase with ``coef_ref``."""
    out = np.array(coef, dtype=complex)
    for j in range(out.shape[0]):
        ref, value = coef_ref[j], coef[j]
        norm_ref, norm_value = np.linalg.norm(ref), np.linalg.norm(value)
        if norm_ref == 0.0 or norm_value == 0.0:
            continue
        gamma = np.vdot(ref, value)
        if abs(gamma) <= 1e-12 * norm_ref * norm_value:
            continue
        out[j] = value * (np.conj(gamma) / abs(gamma))
    return out
