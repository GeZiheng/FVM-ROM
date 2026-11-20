import pickle
import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from fvmrom.rom.transient import Readout, TransientROM


def design(A, order):
    columns = [np.ones((A.shape[0], 1)), A]
    if order == 2:
        i, j = np.triu_indices(A.shape[1])
        columns.append(A[:, i] * A[:, j])
    return np.hstack(columns)


def latent(n=60, r=4, seed=0):
    return np.random.default_rng(seed).normal(size=(n, r))


def targets(A, order, m=2, seed=1):
    rng = np.random.default_rng(seed)
    return design(A, order) @ rng.normal(size=(design(A, order).shape[1], m))


def test_order_one_exact():
    A = latent()
    Y = targets(A, 1)
    readout = Readout(order=1).fit(A, Y)
    assert np.allclose(readout.predict(A), Y, atol=1e-8)
    A_new = latent(seed=3)
    Y_new = design(A_new, 1) @ np.linalg.lstsq(design(A, 1), Y, rcond=None)[0]
    assert np.allclose(readout.predict(A_new), Y_new, atol=1e-8)


def test_order_two_exact():
    A = latent()
    Y = targets(A, 2)
    readout = Readout(order=2).fit(A, Y)
    assert np.allclose(readout.predict(A), Y, atol=1e-8)


def test_order_one_cannot_fit_quadratic_data():
    A = latent()
    Y = targets(A, 2)
    err1 = np.max(np.abs(Readout(order=1).fit(A, Y).predict(A) - Y))
    err2 = np.max(np.abs(Readout(order=2).fit(A, Y).predict(A) - Y))
    assert err1 > 1e-2
    assert err2 < 1e-8
    assert err2 < 1e-6 * err1


@pytest.mark.parametrize("factor", [1e-3, 1e3])
def test_ridge_is_relative_to_feature_scale(factor):
    A = factor * latent()
    Y = targets(A / factor, 2)
    readout = Readout(order=2).fit(A, Y)
    assert np.allclose(readout.predict(A), Y, atol=1e-8)


def test_bias_only_readout():
    Y = np.tile([[1.5, -2.0]], (10, 1))
    readout = Readout(order=2).fit(np.zeros((10, 3)), Y)
    assert np.allclose(readout.predict(np.zeros((5, 3)) + 0.3), Y[0], atol=1e-8)


def test_validation():
    with pytest.raises(ValueError, match="order"):
        Readout(order=3)
    with pytest.raises(ValueError, match="ridge"):
        Readout(ridge=-1.0)
    with pytest.raises(RuntimeError, match="not fitted"):
        Readout().predict(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="rows"):
        Readout().fit(np.zeros((4, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="latent"):
        Readout().fit(np.zeros((4, 2)), np.zeros((4, 2))).predict(np.zeros((4, 3)))


def test_pickle_round_trip():
    A = latent()
    Y = targets(A, 2)
    readout = Readout(order=2).fit(A, Y)
    clone = pickle.loads(pickle.dumps(readout))
    assert np.allclose(clone.predict(A), readout.predict(A))
    assert clone.order == 2 and clone.ridge == readout.ridge


N_LATENT = 4
PERIOD = 1.0
OMEGA = 2.0 * np.pi / PERIOD


def series(t):
    """Latent state: a fundamental and its double frequency, sin/cos pair each."""
    w = OMEGA
    return np.column_stack(
        [
            np.sin(w * t),
            np.cos(w * t),
            0.5 * np.sin(2.0 * w * t + 0.3),
            0.5 * np.cos(2.0 * w * t + 0.3),
        ]
    )


def trajectory(n_train_periods=3.0, n_pred_periods=10.0, dt=0.05, n_cells=30, n_phi=50, seed=0):
    """Synthetic transient: U in a 4-dim linear subspace, quadratic p and forces, linear phi."""
    rng = np.random.default_rng(seed)
    n_train = int(round(n_train_periods / dt))
    t_train = np.arange(n_train) * dt
    t_pred = t_train[-1] + np.arange(1, int(round(n_pred_periods / dt)) + 1) * dt
    n_quad = design(np.zeros((1, N_LATENT)), 2).shape[1]
    modes = rng.normal(size=(N_LATENT, n_cells, 3))
    mean_u = 0.5 * rng.normal(size=(n_cells, 3))
    volumes = rng.uniform(0.5, 2.0, n_cells)
    phi_weights = rng.uniform(0.5, 2.0, n_phi)
    c_p = rng.normal(size=(n_quad, n_cells))
    c_phi = rng.normal(size=(N_LATENT, n_phi))
    c_s = rng.normal(size=(n_quad, 2))

    def build(t):
        a = series(t)
        quad = design(a, 2)
        snaps = {
            "U": mean_u + np.einsum("tr,rci->tci", a, modes),
            "p": quad @ c_p,
            "phi": a @ c_phi,
        }
        return snaps, quad @ c_s

    snaps_train, forces_train = build(t_train)
    snaps_pred, forces_pred = build(t_pred)
    return {
        "t_train": t_train,
        "t_pred": t_pred,
        "snap_train": snaps_train,
        "snap_pred": snaps_pred,
        "forces_train": forces_train,
        "forces_pred": forces_pred,
        "volumes": volumes,
        "phi_weights": phi_weights,
        "modes": modes,
    }


def hopf_latent(t, a0=(0.05, 0.0, 0.0)):
    """Noack-type Hopf trajectory: growth from rest onto a stable unit limit cycle."""

    def rhs(_t, a):
        a1, a2, a3 = a
        return np.array(
            [
                0.1 * a1 - a2 - 0.1 * a1 * a3,
                a1 + 0.1 * a2 - 0.1 * a2 * a3,
                -0.1 * (a3 - a1 * a1 - a2 * a2),
            ]
        )

    solution = solve_ivp(
        rhs, (float(t[0]), float(t[-1])), a0, t_eval=t, method="DOP853", rtol=1e-12, atol=1e-14
    )
    return solution.y.T


def hopf_trajectory(dt=0.2, train_end=60.0, n_pred=315, n_cells=30, n_phi=50, seed=0):
    """Synthetic transient whose latent state is the Hopf trajectory (quadratic dynamics)."""
    rng = np.random.default_rng(seed)
    t_train = np.arange(0.0, train_end + 1e-9, dt)
    t_pred = train_end + np.arange(1, n_pred) * dt
    latent = hopf_latent(np.concatenate([t_train, t_pred]))
    n = latent.shape[1]
    modes = rng.normal(size=(n, n_cells, 3))
    mean_u = 0.5 * rng.normal(size=(n_cells, 3))
    volumes = rng.uniform(0.5, 2.0, n_cells)
    phi_weights = rng.uniform(0.5, 2.0, n_phi)
    n_quad = design(np.zeros((1, n)), 2).shape[1]
    c_p = rng.normal(size=(n_quad, n_cells))
    c_phi = rng.normal(size=(n, n_phi))
    c_s = rng.normal(size=(n_quad, 2))

    def build(rows):
        a = latent[rows]
        quad = design(a, 2)
        return (
            {
                "U": mean_u + np.einsum("tr,rci->tci", a, modes),
                "p": quad @ c_p,
                "phi": a @ c_phi,
            },
            quad @ c_s,
        )

    snaps_train, forces_train = build(slice(0, t_train.size))
    snaps_pred, forces_pred = build(slice(t_train.size, None))
    return {
        "t_train": t_train,
        "t_pred": t_pred,
        "snap_train": snaps_train,
        "snap_pred": snaps_pred,
        "forces_train": forces_train,
        "forces_pred": forces_pred,
        "volumes": volumes,
        "phi_weights": phi_weights,
        "modes": modes,
    }


def rel_l2(pred, ref, weights=None):
    diff = np.asarray(pred, dtype=float) - np.asarray(ref, dtype=float)
    ref = np.asarray(ref, dtype=float)
    n = diff.shape[0]
    diff, ref = diff.reshape(n, -1), ref.reshape(n, -1)
    w = np.ones(diff.shape[1]) if weights is None else np.asarray(weights)
    return np.linalg.norm(diff * np.sqrt(w)) / np.linalg.norm(ref * np.sqrt(w))


def fitted(dynamics="dmd", **kwargs):
    data = trajectory(**kwargs)
    rom = TransientROM(dynamics=dynamics, n_modes={"U": 4, "p": 20, "phi": 4})
    rom.fit(
        data["t_train"],
        data["snap_train"],
        data["volumes"],
        phi_weights=data["phi_weights"],
        scalars={"forces": data["forces_train"]},
    )
    return data, rom


def test_prediction_of_all_fields_and_scalars():
    data, rom = fitted()
    pred = rom.predict(data["t_pred"])
    ref = data["snap_pred"]
    weights = {
        "U": np.repeat(data["volumes"], 3),
        "p": data["volumes"],
        "phi": data["phi_weights"],
        "forces": None,
    }
    assert pred["U"].shape == ref["U"].shape
    assert pred["p"].shape == ref["p"].shape
    assert pred["phi"].shape == ref["phi"].shape
    assert pred["forces"].shape == data["forces_pred"].shape
    for name, values in ref.items():
        err = rel_l2(pred[name], values, weights[name])
        assert err < 1e-6, f"{name}: {err:.3e}"
    assert rel_l2(pred["forces"], data["forces_pred"]) < 1e-6
    assert rom.n_modes_["U"] == 4
    assert rom.t_first == 0.0
    assert np.array_equal(rom.t_train, data["t_train"])
    assert rom.latent(data["t_pred"]).shape == (data["t_pred"].size, 4)
    assert np.allclose(rom.latent(data["t_train"]), rom.pods_["U"].project(
        data["snap_train"]["U"].reshape(data["t_train"].size, -1)
    ))
    assert rom.spectrum()["freq"].size == rom.n_modes_["U"]


def test_replays_the_training_window():
    data, rom = fitted()
    replay = rom.predict(data["t_train"])
    for name, values in data["snap_train"].items():
        assert rel_l2(replay[name], values) < 1e-8


def test_mode_count_follows_energy_tol_when_unspecified():
    data = trajectory()
    rom = TransientROM(dynamics="dmd", n_modes={"U": None, "p": None, "phi": None})
    rom.fit(data["t_train"], data["snap_train"], data["volumes"], phi_weights=data["phi_weights"])
    assert rom.n_modes_["U"] == 4
    assert rom.n_modes_["phi"] == 4
    # the quadratic data of ``p`` only spans 8 dimensions here
    assert rom.n_modes_["p"] == 8


def test_phi_reconstruction_keeps_linear_constraints():
    data = trajectory(seed=2)
    rng = np.random.default_rng(5)
    n_phi = data["phi_weights"].size
    d = rng.normal(size=(6, n_phi))
    _, _, vt = np.linalg.svd(d)
    basis = vt[6:].T
    phi = rng.normal(size=(data["t_train"].size, basis.shape[1])) @ basis.T
    rom = TransientROM(dynamics="dmd", n_modes={"U": 4, "phi": 20})
    rom.fit(
        data["t_train"],
        {"U": data["snap_train"]["U"], "phi": phi},
        data["volumes"],
        phi_weights=data["phi_weights"],
    )
    pred = rom.predict(data["t_pred"])
    assert np.max(np.abs(phi @ d.T)) < 1e-10
    assert np.max(np.abs(pred["phi"] @ d.T)) < 1e-10


def test_fit_only_uses_the_given_window():
    """The ``fit`` interface ignores the held-out window.

    This only covers ``TransientROM.fit``: the held-out arrays are never passed to
    it here, so the test cannot catch a caller (the script) that leaks them.  The
    script-level leak check is
    ``tests/test_fit_and_eval_transient.py::test_models_do_not_depend_on_the_prediction_window``,
    which rewrites the held-out part of ``snapshots.npz`` between two runs of
    ``fit_and_eval_transient.main`` and compares the saved models.
    """
    data = trajectory(seed=4)
    config = dict(
        dynamics="dmd",
        n_modes={"U": 4, "p": 20, "phi": 4},
    )

    def fit():
        return TransientROM(**config).fit(
            data["t_train"],
            data["snap_train"],
            data["volumes"],
            phi_weights=data["phi_weights"],
            scalars={"forces": data["forces_train"]},
        )

    first = fit()
    # the held-out window is never passed to ``fit``; changing it must change nothing
    data["snap_pred"]["U"] += 1e3
    data["snap_pred"]["p"] -= 1e3
    second = fit()
    for name in first.pods_:
        assert np.array_equal(first.pods_[name].mean_, second.pods_[name].mean_)
        assert np.array_equal(first.pods_[name].modes_, second.pods_[name].modes_)
    for name in first.readouts_:
        assert np.array_equal(first.readouts_[name].coef_, second.readouts_[name].coef_)
    assert np.array_equal(first.model_.mu_, second.model_.mu_)
    assert np.array_equal(first.model_.omega_, second.model_.omega_)
    assert np.array_equal(first.model_.amplitudes_, second.model_.amplitudes_)

    t_all = np.concatenate([data["t_train"], data["t_pred"]])
    snaps = {}
    for name, values in data["snap_train"].items():
        tail = np.full((data["t_pred"].size,) + values.shape[1:], np.nan)
        snaps[name] = np.concatenate([values, tail])
    with pytest.raises(ValueError, match="finite"):
        TransientROM(**config).fit(t_all, snaps, data["volumes"])


def test_save_load_roundtrip(tmp_path):
    data, rom = fitted()
    path = tmp_path / "models" / "dmd.pkl"
    rom.save(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = TransientROM.load(path)
    assert loaded.dynamics == "dmd"
    assert loaded.n_modes_ == rom.n_modes_
    assert loaded.t_first == rom.t_first
    assert np.array_equal(loaded.t_train, rom.t_train)
    assert np.allclose(loaded.phi_weights, rom.phi_weights)
    before, after = rom.predict(data["t_pred"]), loaded.predict(data["t_pred"])
    for name in before:
        assert np.array_equal(before[name], after[name]), name


def test_load_rejects_other_formats(tmp_path):
    _, rom = fitted()
    path = tmp_path / "dmd.pkl"
    rom.save(path)
    payload = pickle.loads(path.read_bytes())
    payload["version"] = 99
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="unsupported ROM format"):
        TransientROM.load(path)


def test_load_warns_on_dependency_mismatch(tmp_path):
    _, rom = fitted()
    path = tmp_path / "dmd.pkl"
    rom.save(path)
    payload = pickle.loads(path.read_bytes())
    payload["versions"]["numpy"] = "0.0.0"
    path.write_bytes(pickle.dumps(payload))
    with pytest.warns(RuntimeWarning, match="dependency versions"):
        TransientROM.load(path)


def test_default_dynamics_is_opinf():
    assert TransientROM().dynamics == "opinf"
    data = hopf_trajectory()
    rom = TransientROM(n_modes={"U": 3, "p": 10, "phi": 3})
    rom.fit(
        data["t_train"],
        data["snap_train"],
        data["volumes"],
        phi_weights=data["phi_weights"],
        scalars={"forces": data["forces_train"]},
    )
    assert rom.n_modes_["U"] == 3
    assert rom.spectrum() is None
    pred = rom.predict(data["t_pred"])
    assert pred["U"].shape == data["snap_pred"]["U"].shape
    assert rel_l2(pred["U"], data["snap_pred"]["U"]) < 1e-2
    assert rel_l2(pred["forces"], data["forces_pred"]) < 1e-2


def test_opinf_save_load_roundtrip(tmp_path):
    data = hopf_trajectory(seed=3)
    rom = TransientROM(n_modes={"U": 3, "p": 10, "phi": 3}, dynamics_kw={"ridge": 1e-10})
    rom.fit(
        data["t_train"],
        data["snap_train"],
        data["volumes"],
        phi_weights=data["phi_weights"],
        scalars={"forces": data["forces_train"]},
    )
    assert rom.model_.ridge_path_ is None  # manual ridge: no selection
    path = tmp_path / "models" / "opinf.pkl"
    rom.save(path)
    loaded = TransientROM.load(path)
    assert loaded.dynamics == "opinf"
    assert loaded.model_.ridge_rel_ == rom.model_.ridge_rel_
    before, after = rom.predict(data["t_pred"]), loaded.predict(data["t_pred"])
    for name in before:
        assert np.array_equal(before[name], after[name]), name


def test_fit_validation():
    data = trajectory()
    t, snaps, volumes = data["t_train"], data["snap_train"], data["volumes"]
    rom = TransientROM(dynamics="dmd", n_modes={"U": 4, "p": 20, "phi": 4})

    uneven = t.copy()
    uneven[10] += 1e-3
    with pytest.raises(ValueError, match="uniformly spaced"):
        rom.fit(uneven, snaps, volumes)
    with pytest.raises(ValueError, match="no snapshots"):
        rom.fit(t, {}, volumes)
    with pytest.raises(ValueError, match="'U'"):
        rom.fit(t, {"p": snaps["p"]}, volumes)
    with pytest.raises(ValueError, match="unsupported snapshot fields"):
        rom.fit(t, {**snaps, "k": snaps["p"]}, volumes)
    with pytest.raises(ValueError, match="n_modes 1000"):
        TransientROM(dynamics="dmd", n_modes={"U": 1000}).fit(t, snaps, volumes)
    with pytest.raises(ValueError, match="expected 57 snapshots"):
        rom.fit(t[:-3], snaps, volumes)
    with pytest.raises(ValueError, match="clashes"):
        rom.fit(t, snaps, volumes, scalars={"p": data["forces_train"]})
    with pytest.raises(ValueError, match="60 scalar rows"):
        rom.fit(t, snaps, volumes, scalars={"forces": data["forces_train"][:-1]})
    with pytest.raises(ValueError, match="finite"):
        rom.fit(t, {**snaps, "phi": np.full_like(snaps["phi"], np.nan)}, volumes)
    with pytest.raises(ValueError, match="finite"):
        rom.fit(t, snaps, volumes, scalars={"forces": np.full((t.size, 2), np.inf)})
    with pytest.raises(RuntimeError, match="not fitted"):
        TransientROM(dynamics="dmd").predict(t)


def test_constructor_validation():
    with pytest.raises(ValueError, match="energy_tol"):
        TransientROM(energy_tol=1.0)
    with pytest.raises(ValueError, match="readout_ridge"):
        TransientROM(readout_ridge=-1.0)
    with pytest.raises(ValueError, match="n_modes"):
        TransientROM(n_modes={"U": -1})
    with pytest.raises(ValueError, match="readout_order"):
        TransientROM(readout_order={"p": 3})
    rom = TransientROM(dynamics="dmd", readout_order={"phi": 2})
    assert rom.readout_order == {"p": 2, "phi": 2, "scalars": 2}

