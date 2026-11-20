import json
import pickle

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from fvmrom.rom.dynamics import (
    BOPDMD,
    DMD,
    OpInf,
    OptDMD,
    _dominant_period,
    _fourth_order_derivative,
    make_dynamics,
)
from fvmrom.rom.evaluate_transient import to_jsonable

DT = 0.05
T = np.arange(0.0, 10.0 + 1e-9, DT)
FREQS = (0.5, 1.3)
GROWTHS = (0.01, -0.02)


def modal_data(t=T, n=4, seed=0):
    """Two conjugate mode pairs (real data) built from complex spatial modes."""
    rng = np.random.default_rng(seed)
    modes = rng.normal(size=(2, n)) + 1j * rng.normal(size=(2, n))
    amplitudes = np.array([1.0 + 0.3j, 0.4 - 0.6j])
    data = np.zeros((t.size, n))
    for k, (freq, growth) in enumerate(zip(FREQS, GROWTHS)):
        omega = growth + 2j * np.pi * freq
        data += 2.0 * np.real(np.exp(omega * t)[:, None] * (amplitudes[k] * modes[k]))
    return data


def test_recovers_eigenvalues_and_growth():
    data = modal_data()
    dmd = DMD().fit(T, data)
    spec = dmd.spectrum()

    assert np.allclose(np.sort(np.abs(spec["freq"])), np.sort(np.repeat(FREQS, 2)), atol=1e-8)
    assert np.allclose(np.sort(spec["growth"]), np.sort(np.repeat(GROWTHS, 2)), atol=1e-8)
    assert spec["amplitude"].size == 4
    assert np.all(spec["amplitude"] > 0.0)


def test_prediction_replays_and_extrapolates():
    data = modal_data()
    dmd = DMD().fit(T, data)

    replay = dmd.predict(T)
    assert replay.shape == data.shape
    assert np.allclose(replay, data, atol=1e-8)

    # ten periods of the slower mode after the end of the window
    t_far = T[-1] + np.arange(1, 401) * DT
    truth = modal_data(t_far)
    assert np.allclose(dmd.predict(t_far), truth, atol=1e-8)


def test_rejects_non_uniform_times():
    data = modal_data()
    t = T.copy()
    t[5] += 0.25 * DT
    with pytest.raises(ValueError, match="uniform"):
        DMD().fit(t, data)


def test_svd_rank_truncates():
    data = modal_data(n=6)
    full = DMD().fit(T, data)
    assert full.spectrum()["freq"].size == 4

    truncated = DMD(svd_rank=2).fit(T, data)
    spec = truncated.spectrum()
    assert spec["freq"].size == 2
    assert spec["growth"].size == spec["amplitude"].size == 2
    assert spec["eigenvalues"].shape == (2, 2)


def test_spectrum_keys():
    data = modal_data()
    spec = DMD().fit(T, data).spectrum()
    assert set(spec) == {"freq", "growth", "amplitude", "eigenvalues"}
    assert spec["eigenvalues"].ndim == 2 and spec["eigenvalues"].shape[1] == 2


def test_make_dynamics():
    assert isinstance(make_dynamics("dmd"), DMD)
    assert isinstance(make_dynamics("DMD", svd_rank=3), DMD)
    assert make_dynamics("dmd", svd_rank=3).svd_rank == 3
    assert isinstance(make_dynamics("optdmd"), OptDMD)
    assert isinstance(make_dynamics("OPTDMD", rank=2), OptDMD)
    assert make_dynamics("optdmd", rank=2).rank == 2
    assert isinstance(make_dynamics("bopdmd"), BOPDMD)
    assert isinstance(make_dynamics("BOPDMD", rank=2, n_trials=5), BOPDMD)
    dynamics = make_dynamics("bopdmd", rank=2, n_trials=5, trial_fraction=0.5, seed=3)
    assert (dynamics.rank, dynamics.n_trials, dynamics.seed) == (2, 5, 3)
    assert dynamics.trial_fraction == 0.5
    assert isinstance(make_dynamics("opinf"), OpInf)
    assert isinstance(make_dynamics("OPINF", ridge=1e-4), OpInf)
    assert make_dynamics("opinf", ridge=1e-4).ridge == 1e-4
    unchecked = make_dynamics("opinf", recheck=False, check_periods=2.0)
    assert unchecked.recheck is False and unchecked.check_periods == 2.0
    with pytest.raises(ValueError, match="unknown dynamics"):
        make_dynamics("lstm")


def test_pickle_round_trip():
    data = modal_data()
    dmd = DMD().fit(T, data)
    clone = pickle.loads(pickle.dumps(dmd))
    t = T[-1] + np.arange(1, 51) * DT
    assert np.allclose(clone.predict(t), dmd.predict(t))
    for key, value in dmd.spectrum().items():
        assert np.allclose(clone.spectrum()[key], value)


# -- Optimized DMD -------------------------------------------------------------------------


def single_pair_data(t, amp=0.85 * np.exp(0.3j), freq=0.7, growth=0.0):
    """One conjugate pair with a unit-norm complex mode in two entries."""
    phi = np.array([1.0, 1.0j]) / np.sqrt(2.0)
    omega = growth + 2j * np.pi * freq
    return 2.0 * np.real(np.exp(omega * t)[:, None] * (amp * phi))


def growth_bias(spec, truths=GROWTHS):
    """Largest distance from an estimated growth rate to the nearest true one."""
    growth = spec["growth"]
    return max(float(np.min(np.abs(growth - truth))) for truth in truths)


def test_dmd_amplitude_uses_unit_norm_modes():
    amp = 0.85 * np.exp(0.3j)
    data = single_pair_data(T, amp=amp)
    dmd = DMD().fit(T, data)
    assert np.allclose(np.linalg.norm(dmd.modes_, axis=0), 1.0)
    assert np.allclose(dmd.spectrum()["amplitude"], abs(amp), atol=1e-8)
    assert np.allclose(dmd.predict(T), data, atol=1e-8)


def test_optdmd_recovers_eigenvalues():
    data = modal_data()
    spec = OptDMD().fit(T, data).spectrum()
    assert np.allclose(
        np.sort(np.abs(spec["freq"])), np.sort(np.repeat(FREQS, 2)), atol=1e-8
    )
    assert np.allclose(
        np.sort(spec["growth"]), np.sort(np.repeat(GROWTHS, 2)), atol=1e-8
    )


def test_optdmd_accepts_non_uniform_times():
    rng = np.random.default_rng(3)
    t = T + rng.uniform(-0.3, 0.3, T.size) * DT
    t[0] = T[0]
    data = modal_data(t)
    with pytest.raises(ValueError, match="uniform"):
        DMD().fit(t, data)

    spec = OptDMD().fit(t, data).spectrum()
    assert np.allclose(
        np.sort(np.abs(spec["freq"])), np.sort(np.repeat(FREQS, 2)), atol=1e-6
    )
    assert np.allclose(
        np.sort(spec["growth"]), np.sort(np.repeat(GROWTHS, 2)), atol=1e-6
    )


def test_optdmd_reduces_the_noise_bias_of_dmd():
    # white noise of 2% of the signal standard deviation: standard DMD is
    # noticeably biased, the variable projection is much less so
    data = modal_data()
    rng = np.random.default_rng(7)
    noisy = data + 0.02 * data.std() * rng.standard_normal(data.shape)
    bias_dmd = growth_bias(DMD().fit(T, noisy).spectrum())
    bias_opt = growth_bias(OptDMD().fit(T, noisy).spectrum())
    assert bias_dmd > bias_opt


def test_optdmd_rank_controls_the_number_of_eigenvalues():
    data = modal_data(n=6)
    assert OptDMD().fit(T, data).spectrum()["freq"].size == 4

    spec = OptDMD(rank=2).fit(T, data).spectrum()
    assert spec["freq"].size == spec["growth"].size == spec["amplitude"].size == 2
    assert spec["eigenvalues"].shape == (2, 2)
    with pytest.raises(ValueError, match="rank"):
        OptDMD(rank=0)


def test_optdmd_spectrum_is_paired_and_real():
    amp = 0.85 * np.exp(0.3j)
    data = single_pair_data(T, amp=amp)
    opt = OptDMD().fit(T, data)
    spec = opt.spectrum()

    assert set(spec) == {"freq", "growth", "amplitude", "eigenvalues"}
    assert spec["eigenvalues"].shape == (2, 2)
    for value in opt.lambda_:
        assert np.min(np.abs(opt.lambda_ - np.conj(value))) < 1e-10
    # unit-norm exponential columns: |b| = |a| sqrt(n) for a pure oscillation
    assert np.allclose(spec["amplitude"], abs(amp) * np.sqrt(T.size), rtol=1e-6)
    assert np.allclose(spec["growth"], 0.0, atol=1e-8)

    pred = opt.predict(T)
    assert np.isrealobj(pred)
    assert np.allclose(pred, data, atol=1e-8)


def test_optdmd_pickle_round_trip():
    data = modal_data()
    opt = OptDMD().fit(T, data)
    clone = pickle.loads(pickle.dumps(opt))
    t = T[-1] + np.arange(1, 51) * DT
    assert np.allclose(clone.predict(t), opt.predict(t))
    for key, value in opt.spectrum().items():
        assert np.allclose(clone.spectrum()[key], value)


# -- BOP-DMD -------------------------------------------------------------------------------


def test_bopdmd_defaults():
    model = BOPDMD()
    assert model.rank is None
    assert model.n_trials == 100
    assert model.trial_fraction == 0.8
    assert isinstance(model.seed, int)
    assert model.reference_nfev_ is None
    assert model.reference_success_ is None

    t = np.arange(0.0, 2.0 + 1e-9, DT)
    spec = model.fit(t, modal_data(t)).spectrum()
    assert 1 <= spec["n_used"] <= model.n_trials
    assert isinstance(model.reference_nfev_, int) and model.reference_nfev_ >= 1
    assert model.reference_success_ is True


def test_bopdmd_recovers_eigenvalues():
    data = modal_data()
    spec = BOPDMD(n_trials=10, seed=1).fit(T, data).spectrum()
    assert np.allclose(
        np.sort(np.abs(spec["freq"])), np.sort(np.repeat(FREQS, 2)), atol=1e-6
    )
    assert np.allclose(
        np.sort(spec["growth"]), np.sort(np.repeat(GROWTHS, 2)), atol=1e-6
    )
    assert np.allclose(spec["freq_std"], 0.0, atol=1e-8)
    assert np.allclose(spec["growth_std"], 0.0, atol=1e-8)


def test_bopdmd_reduces_the_noise_bias_of_dmd():
    data = modal_data()
    rng = np.random.default_rng(7)
    noisy = data + 0.02 * data.std() * rng.standard_normal(data.shape)
    bias_dmd = growth_bias(DMD().fit(T, noisy).spectrum())
    spec = BOPDMD(n_trials=20, seed=11).fit(T, noisy).spectrum()
    assert growth_bias(spec) <= bias_dmd
    assert np.all(spec["growth_std"] > 0.0)
    assert np.all(spec["freq_std"] > 0.0)
    assert np.all(spec["amplitude_std"] > 0.0)
    assert 1 <= spec["n_used"] <= 20


def test_bopdmd_is_reproducible_with_a_fixed_seed():
    data = modal_data()
    rng = np.random.default_rng(5)
    noisy = data + 0.02 * data.std() * rng.standard_normal(data.shape)
    t = T[-1] + np.arange(1, 51) * DT

    first = BOPDMD(n_trials=8, seed=4).fit(T, noisy)
    second = BOPDMD(n_trials=8, seed=4).fit(T, noisy)
    other = BOPDMD(n_trials=8, seed=5).fit(T, noisy)

    assert np.array_equal(first.predict(t), second.predict(t))
    for key, value in first.spectrum().items():
        assert np.array_equal(second.spectrum()[key], value)
    assert not np.array_equal(first.predict(t), other.predict(t))


def test_bopdmd_parameter_validation():
    with pytest.raises(ValueError, match="n_trials"):
        BOPDMD(n_trials=0)
    with pytest.raises(ValueError, match="n_trials"):
        BOPDMD(n_trials=-3)
    with pytest.raises(ValueError, match="trial_fraction"):
        BOPDMD(trial_fraction=0.0)
    with pytest.raises(ValueError, match="trial_fraction"):
        BOPDMD(trial_fraction=1.5)
    with pytest.raises(ValueError, match="rank"):
        BOPDMD(rank=0)
    with pytest.raises(ValueError, match="trial"):
        BOPDMD(n_trials=2, trial_fraction=0.01).fit(T, modal_data())


def test_bopdmd_spectrum_keys_are_jsonable():
    spec = BOPDMD(n_trials=4, seed=2).fit(T, modal_data()).spectrum()
    assert set(spec) == {
        "freq",
        "growth",
        "amplitude",
        "eigenvalues",
        "freq_std",
        "growth_std",
        "amplitude_std",
        "n_used",
    }
    assert spec["eigenvalues"].shape == (4, 2)
    assert isinstance(spec["n_used"], int)
    json.dumps(to_jsonable(spec))


def test_bopdmd_predict_is_real_and_pickle_round_trip():
    data = modal_data()
    model = BOPDMD(n_trials=6, seed=3).fit(T, data)
    t = T[-1] + np.arange(1, 51) * DT
    pred = model.predict(t)
    assert np.isrealobj(pred)
    assert np.allclose(model.predict(T), data, atol=1e-6)

    clone = pickle.loads(pickle.dumps(model))
    assert np.array_equal(clone.predict(t), pred)
    for key, value in model.spectrum().items():
        assert np.array_equal(clone.spectrum()[key], value)


def test_bopdmd_accepts_non_uniform_times():
    rng = np.random.default_rng(3)
    t = T + rng.uniform(-0.3, 0.3, T.size) * DT
    t[0] = T[0]
    data = modal_data(t)
    spec = BOPDMD(n_trials=6, seed=1).fit(t, data).spectrum()
    assert np.allclose(
        np.sort(np.abs(spec["freq"])), np.sort(np.repeat(FREQS, 2)), atol=1e-6
    )
    assert np.allclose(
        np.sort(spec["growth"]), np.sort(np.repeat(GROWTHS, 2)), atol=1e-6
    )


# -- Operator Inference --------------------------------------------------------------------


def hopf_rhs(t, a):
    """Noack-type three-mode Hopf model with a stable unit limit cycle."""
    a1, a2, a3 = a
    return np.array(
        [
            0.1 * a1 - 1.0 * a2 - 0.1 * a1 * a3,
            1.0 * a1 + 0.1 * a2 - 0.1 * a2 * a3,
            -0.1 * (a3 - a1 * a1 - a2 * a2),
        ]
    )


def hopf(t, a0=(0.05, 0.0, 0.0), t0=None):
    """Reference trajectory of the Hopf model, integrated from ``t0`` (default ``t[0]``)."""
    start = float(t[0]) if t0 is None else float(t0)
    solution = solve_ivp(
        hopf_rhs,
        (start, float(t[-1])),
        a0,
        t_eval=t,
        method="DOP853",
        rtol=1e-13,
        atol=1e-15,
    )
    return solution.y.T


#: training window of the Hopf model: growth from rest plus the limit cycle
HOPF_TRAIN = np.arange(0.0, 60.0 + 1e-9, 0.2)
HOPF_A = np.array([[0.1, -1.0, 0.0], [1.0, 0.1, 0.0], [0.0, 0.0, -0.1]])
HOPF_H = np.zeros((3, 6))  # columns in np.triu_indices(3) order
HOPF_H[0, 2] = -0.1  # a1 a3
HOPF_H[1, 4] = -0.1  # a2 a3
HOPF_H[2, 0] = 0.1  # a1²
HOPF_H[2, 3] = 0.1  # a2²


def test_hopf_reference_has_a_stable_limit_cycle():
    t = np.arange(0.0, 400.0 + 4.0 * np.pi, 0.2)
    first = hopf(t)[-314:]
    second = hopf(t, (0.9, 0.2, 0.5))[-314:]
    # both ends of a transient reach the same periodic orbit, up to the phase
    assert np.allclose(first.min(axis=0), second.min(axis=0), atol=1e-3)
    assert np.allclose(first.max(axis=0), second.max(axis=0), atol=1e-3)
    assert np.allclose(first.max(axis=0), [1.0, 1.0, 1.0], atol=2e-3)
    assert np.allclose(first.min(axis=0), [-1.0, -1.0, 1.0], atol=2e-3)


def test_opinf_recovers_hopf_coefficients():
    model = OpInf().fit(HOPF_TRAIN, hopf(HOPF_TRAIN))
    assert model.H_.shape == (3, 6)
    assert np.allclose(model.c_, 0.0, atol=1e-3)
    assert np.allclose(model.A_, HOPF_A, atol=1e-3)
    assert np.allclose(model.H_, HOPF_H, atol=1e-3)


def test_opinf_predicts_ten_limit_cycles():
    data = hopf(HOPF_TRAIN)
    model = OpInf().fit(HOPF_TRAIN, data)
    t_pred = HOPF_TRAIN[-1] + np.arange(1, 315) * 0.2  # ten periods
    truth = hopf(t_pred, data[-1], HOPF_TRAIN[-1])
    pred = model.predict(t_pred)

    assert not model.diverged_ and model.diverged_at_ is None
    assert pred.shape == truth.shape
    relative = np.linalg.norm(pred - truth, axis=1) / np.linalg.norm(truth, axis=1)
    assert relative.max() < 1e-2
    radius2 = pred[:, 0] ** 2 + pred[:, 1] ** 2
    assert np.allclose(radius2.mean(), 1.0, atol=5e-2)


def test_fourth_order_derivative_accuracy():
    errors = {}
    for dt in (0.2, 0.1):
        t = np.arange(0.0, 10.0, dt)
        derivative = _fourth_order_derivative(np.sin(t)[:, None], dt)[:, 0]
        error = np.abs(derivative - np.cos(t))
        assert error[2:-2].max() < 1e-4  # interior, central stencil
        assert max(error[0], error[1], error[-2], error[-1]) < 1e-3  # one-sided ends
        errors[dt] = error.max()
    ratio = errors[0.2] / errors[0.1]
    assert 12.0 < ratio < 20.0  # fourth order: halving dt divides the error by 16


def test_opinf_ridge_path_records_the_candidates():
    model = OpInf().fit(HOPF_TRAIN, hopf(HOPF_TRAIN))
    path = model.ridge_path_
    grid = model.ridge_grid
    assert np.array_equal(path["ridge_rel"], grid)
    assert path["holdout_error"].shape == grid.shape
    assert path["stable"].dtype == bool
    assert model.ridge_rel_ in grid
    assert model.ridge_ > 0.0
    stable = path["stable"]
    assert stable.any()
    assert np.all(np.isfinite(path["holdout_error"][stable]))
    assert np.all(np.isnan(path["holdout_error"][~stable]))
    best = np.argmin(np.where(stable, path["holdout_error"], np.inf))
    assert model.ridge_rel_ == grid[best]


def test_opinf_manual_ridge_skips_selection():
    model = OpInf(ridge=1e-12).fit(HOPF_TRAIN, hopf(HOPF_TRAIN))
    assert model.ridge_path_ is None
    assert model.ridge_rel_ == 1e-12
    assert 0.0 < model.ridge_ <= 1.000001e-12
    # a user-set ridge is used as is: no selection and no recheck
    assert model.ridge_selected_ is None
    assert model.refit_fallback_ is False
    assert model.recheck_path_ is None
    assert model.check_info_ is None


def test_opinf_rejects_unstable_candidates_on_the_holdout():
    data = hopf(HOPF_TRAIN)
    model = OpInf().fit(HOPF_TRAIN, data)
    n_fit = int(round(0.75 * HOPF_TRAIN.size))
    threshold = model._threshold(model.blowup_factor)
    coefficients = np.vstack([np.zeros((1, 3)), HOPF_A.T, HOPF_H.T])

    error, stable = model._holdout_result(HOPF_TRAIN, data, n_fit, coefficients, threshold)
    assert stable
    assert np.isfinite(error) and error < 1e-2

    unstable = coefficients.copy()
    unstable[1:4] = 5.0 * HOPF_A.T
    unstable[4:] = 0.0  # no quadratic saturation: linear growth 0.5 per time unit
    error, stable = model._holdout_result(HOPF_TRAIN, data, n_fit, unstable, threshold)
    assert not stable
    assert np.isnan(error)


def test_opinf_warns_when_every_candidate_diverges():
    model = OpInf(blowup_factor=1e-6)
    with pytest.warns(RuntimeWarning, match="ridge"):
        model.fit(HOPF_TRAIN, hopf(HOPF_TRAIN))
    assert model.ridge_rel_ == model.ridge_grid[-1]
    assert not model.ridge_path_["stable"].any()
    assert np.all(np.isnan(model.ridge_path_["holdout_error"]))


def test_opinf_divergence_returns_nan_rows():
    # a' = a² blows up in finite time; the prediction must stop instead of raising
    t = np.arange(0.0, 0.5 + 1e-9, 0.02)
    model = OpInf(ridge=0.0).fit(t, (1.0 / (1.0 - t))[:, None])
    assert np.allclose(model.A_, 0.0, atol=1e-3)
    assert np.allclose(model.H_, 1.0, atol=1e-3)

    t_pred = t[-1] + np.arange(1, 51) * 0.02
    pred = model.predict(t_pred)
    assert pred.shape == (50, 1)
    assert model.diverged_ is True
    assert t_pred[0] < model.diverged_at_ <= 1.0
    late = t_pred > model.diverged_at_
    assert late.any() and not late.all()
    assert np.all(np.isnan(pred[late]))
    assert np.all(np.isfinite(pred[~late]))


def test_opinf_predict_defaults_and_replay():
    data = hopf(HOPF_TRAIN)
    model = OpInf().fit(HOPF_TRAIN, data)

    # default t0/a0: the last training sample
    at_end = model.predict(np.array([model.t_last_]))
    assert np.allclose(at_end[0], data[-1])
    step = np.array([model.t_last_ + 0.5])
    assert np.allclose(model.predict(step), hopf(step, data[-1], model.t_last_), atol=1e-3)

    # explicit t0/a0 replay the training window
    replay = model.predict(HOPF_TRAIN, a0=data[0], t0=float(HOPF_TRAIN[0]))
    assert replay.shape == data.shape
    assert np.linalg.norm(replay - data) / np.linalg.norm(data) < 1e-2

    with pytest.raises(ValueError, match="precede"):
        model.predict(np.array([model.t_last_ - 0.2]))
    with pytest.raises(ValueError, match="non-decreasing"):
        model.predict(np.array([model.t_last_, model.t_last_ - 0.1]))
    with pytest.raises(ValueError, match="length"):
        model.predict(step, a0=np.zeros(2))


def test_opinf_spectrum_and_linear_eigenvalues():
    model = OpInf().fit(HOPF_TRAIN, hopf(HOPF_TRAIN))
    assert model.spectrum() is None
    eigenvalues = model.linear_eigenvalues()
    for expected in (0.1 + 1.0j, 0.1 - 1.0j, -0.1):
        assert np.min(np.abs(eigenvalues - expected)) < 1e-3


def test_opinf_pickle_round_trip():
    data = hopf(HOPF_TRAIN)
    model = OpInf().fit(HOPF_TRAIN, data)
    clone = pickle.loads(pickle.dumps(model))
    t = model.t_last_ + np.arange(1, 51) * 0.2
    assert np.array_equal(clone.predict(t), model.predict(t))
    assert np.array_equal(clone.A_, model.A_)
    assert np.array_equal(clone.H_, model.H_)
    assert clone.ridge_rel_ == model.ridge_rel_
    json.dumps(to_jsonable(model.ridge_path_))


def test_opinf_validation():
    with pytest.raises(ValueError, match="ridge must"):
        OpInf(ridge=-1.0)
    with pytest.raises(ValueError, match="ridge_grid"):
        OpInf(ridge_grid=[0.0, 1.0])
    with pytest.raises(ValueError, match="blowup_factor"):
        OpInf(blowup_factor=0.0)
    with pytest.raises(ValueError, match="predict_blowup_factor"):
        OpInf(predict_blowup_factor=-1.0)
    with pytest.raises(ValueError, match="check_periods"):
        OpInf(check_periods=0.0)
    with pytest.raises(ValueError, match="check_blowup_factor"):
        OpInf(check_blowup_factor=-1.0)

    short = HOPF_TRAIN[:4]
    with pytest.raises(ValueError, match="five snapshots"):
        OpInf().fit(short, hopf(short))
    with pytest.raises(ValueError, match="eight snapshots"):
        OpInf().fit(HOPF_TRAIN[:6], hopf(HOPF_TRAIN[:6]))
    uneven = HOPF_TRAIN.copy()
    uneven[10] += 1e-3
    with pytest.raises(ValueError, match="uniform"):
        OpInf().fit(uneven, hopf(HOPF_TRAIN))

    model = OpInf(ridge=0.0)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(np.array([0.0]))
    with pytest.raises(RuntimeError, match="not fitted"):
        model.spectrum()


# -- OpInf refit recheck -------------------------------------------------------------------

#: an undamped oscillator (plus its first harmonic) with period 2π
OSC_TRAIN = np.arange(0.0, 60.0 + 1e-9, 0.2)


def oscillator(t=None):
    """Undamped two-mode oscillator with a first harmonic."""
    t = OSC_TRAIN if t is None else np.asarray(t, dtype=float)
    return np.column_stack(
        [np.cos(t + 0.3), np.sin(t + 0.3), 0.5 * np.cos(2.0 * t + 0.3)]
    )


def noisy_oscillator(seed=1, scale=1e-3):
    """``oscillator`` plus noise: a tiny ridge refits into a growing model."""
    return oscillator() + np.random.default_rng(seed).normal(
        scale=scale, size=(OSC_TRAIN.size, 3)
    )


def test_opinf_recheck_falls_back_to_a_larger_ridge(monkeypatch):
    # Force the selection to the smallest candidate.  On 1e-3 noise the refit at
    # 1e-8 turns the undamped oscillator into a growing model that leaves the
    # 5x training-norm band during the training-window replay, while a larger
    # ridge stays bounded (deterministic with seed 1).
    monkeypatch.setattr(OpInf, "_holdout_result", lambda *args, **kwargs: (1.0, True))
    data = noisy_oscillator()
    model = OpInf().fit(OSC_TRAIN, data)

    assert model.refit_fallback_ is True
    assert model.ridge_selected_ == model.ridge_grid[0]
    assert model.ridge_rel_ > model.ridge_selected_

    path = model.recheck_path_
    assert set(path) == {
        "ridge_rel",
        "ridge",
        "stable",
        "max_norm_ratio_replay",
        "max_norm_ratio_forward",
    }
    assert np.array_equal(path["ridge_rel"], model.ridge_grid[: path["ridge_rel"].size])
    assert path["stable"].dtype == bool
    assert not path["stable"][:-1].any()
    assert path["stable"][-1]
    assert path["max_norm_ratio_replay"].shape == path["stable"].shape
    assert np.all(np.isfinite(path["max_norm_ratio_replay"]))
    assert path["max_norm_ratio_replay"][-1] <= model.blowup_factor
    assert model.ridge_ == path["ridge"][-1]

    info = model.check_info_
    assert info["check_periods"] == 5.0
    assert info["blowup_factor"] == 5.0
    assert info["period_estimated"] is True
    assert info["period"] == pytest.approx(2.0 * np.pi, rel=1e-2)

    replay = model.predict(OSC_TRAIN, a0=data[0], t0=float(OSC_TRAIN[0]))
    assert np.all(np.isfinite(replay))
    norms = np.linalg.norm(replay, axis=1)
    assert np.all(norms <= model.blowup_factor * model.a_max_norm_)

    span = info["check_periods"] * info["period"]
    steps = max(1, int(round(span / model.dt_)))
    forward = model.predict(model.t_last_ + np.linspace(span / steps, span, steps))
    assert np.all(np.isfinite(forward))
    assert not model.diverged_


def test_opinf_recheck_false_keeps_the_plain_selection(monkeypatch):
    monkeypatch.setattr(OpInf, "_holdout_result", lambda *args, **kwargs: (1.0, True))
    data = noisy_oscillator()
    model = OpInf(recheck=False).fit(OSC_TRAIN, data)

    assert model.refit_fallback_ is False
    assert model.recheck_path_ is None
    assert model.check_info_ is None
    assert model.ridge_rel_ == model.ridge_selected_ == model.ridge_grid[0]

    replay = model.predict(OSC_TRAIN, a0=data[0], t0=float(OSC_TRAIN[0]))
    assert model.diverged_ is True
    assert np.isnan(replay).any()


def test_opinf_clean_data_needs_no_fallback():
    data = hopf(HOPF_TRAIN)
    checked = OpInf().fit(HOPF_TRAIN, data)
    plain = OpInf(recheck=False).fit(HOPF_TRAIN, data)

    assert checked.refit_fallback_ is False
    assert checked.ridge_rel_ == checked.ridge_selected_
    assert checked.ridge_ == pytest.approx(checked.ridge_selected_, rel=1e-12)
    assert np.array_equal(checked.coef_, plain.coef_)
    assert np.array_equal(checked.scale_, plain.scale_)
    assert np.array_equal(checked.ridge_path_["stable"], plain.ridge_path_["stable"])
    assert checked.recheck_path_ is not None
    assert checked.check_info_ is not None


def test_opinf_recheck_uses_the_window_length_when_the_period_is_unknown():
    # half a cycle of a slow oscillator: no dominant period, the window length
    # (20) stands in for it
    t = np.arange(0.0, 20.0 + 1e-9, 0.2)
    data = np.column_stack(
        [
            np.sin(2.0 * np.pi * t / 40.0),
            np.cos(2.0 * np.pi * t / 40.0),
            0.5 * np.sin(2.0 * np.pi * t / 20.0),
        ]
    )
    model = OpInf().fit(t, data)
    info = model.check_info_
    assert info["period_estimated"] is False
    assert info["period"] == pytest.approx(float(t[-1] - t[0]))
    assert model.refit_fallback_ is False


def test_opinf_warns_when_the_recheck_never_succeeds(monkeypatch):
    monkeypatch.setattr(
        OpInf, "_recheck", lambda *args, **kwargs: (False, float("nan"), float("nan"))
    )
    model = OpInf()
    with pytest.warns(RuntimeWarning, match="largest"):
        model.fit(HOPF_TRAIN, hopf(HOPF_TRAIN))
    path = model.recheck_path_
    assert model.refit_fallback_ is True
    assert model.ridge_rel_ == model.ridge_grid[-1]
    assert model.ridge_ == path["ridge"][-1]
    assert not path["stable"].any()
    assert path["stable"].size == model.ridge_grid.size
    assert np.all(np.isnan(path["max_norm_ratio_replay"]))


def test_dominant_period_of_clean_and_noisy_sines():
    t = np.arange(0.0, 20.0, 0.05)
    sine = np.sin(2.0 * np.pi * t / 2.0)
    A = np.column_stack([sine, np.zeros_like(sine)])
    assert _dominant_period(t, A) == pytest.approx(2.0, rel=1e-3)

    noisy = A + np.random.default_rng(0).normal(scale=0.05, size=A.shape)
    assert _dominant_period(t, noisy) == pytest.approx(2.0, rel=1e-2)


def test_dominant_period_falls_back_to_nan():
    t = np.arange(0.0, 20.0, 0.05)
    assert np.isnan(_dominant_period(t[:5], np.ones((5, 2))))
    assert np.isnan(_dominant_period(t, np.ones((t.size, 2))))
    # fewer than two cycles of a slow sine in the window
    slow = np.column_stack([np.sin(2.0 * np.pi * t / 40.0), np.zeros_like(t)])
    assert np.isnan(_dominant_period(t, slow))


def test_opinf_recheck_attributes_pickle_round_trip():
    model = OpInf().fit(HOPF_TRAIN, hopf(HOPF_TRAIN))
    clone = pickle.loads(pickle.dumps(model))
    assert clone.refit_fallback_ == model.refit_fallback_
    assert clone.ridge_selected_ == model.ridge_selected_
    assert clone.ridge_ == model.ridge_
    assert clone.check_info_ == model.check_info_
    assert set(clone.recheck_path_) == set(model.recheck_path_)
    for key, value in model.recheck_path_.items():
        assert np.array_equal(clone.recheck_path_[key], value)
    json.dumps(to_jsonable(model.recheck_path_))
    json.dumps(to_jsonable(model.check_info_))

