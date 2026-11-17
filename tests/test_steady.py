import pickle
import warnings

import numpy as np
import pytest

from fvmrom.params import ParamSpace
from fvmrom.rom.steady import SteadyROM

SPACE = ParamSpace(names=("mu",), lower=(1.0,), upper=(10.0,), log=(False,))


def synthetic(n=20, seed=0, n_cells=40, n_faces=120):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_cells)
    mu = rng.uniform(1.0, 10.0, n)
    c1 = np.sin(mu)[:, None]
    c2 = np.exp(-mu / 5.0)[:, None]
    u = np.zeros((n, n_cells, 3))
    u[:, :, 0] = c1 * np.sin(np.pi * x)[None, :] + c2 * x[None, :] ** 2
    u[:, :, 1] = 0.5 * c2 * x[None, :]
    p = c1 * np.cos(np.pi * x)[None, :]
    xf = np.linspace(0.0, 1.0, n_faces)
    phi = c1 * np.ones(n_faces)[None, :] + c2 * np.sin(np.pi * xf)[None, :]
    volumes = np.linspace(0.5, 2.0, n_cells)
    return mu, {"U": u, "p": p, "phi": phi}, volumes


def rel_l2(pred, ref, weights=None):
    diff = np.asarray(pred, dtype=float) - np.asarray(ref, dtype=float)
    ref = np.asarray(ref, dtype=float)
    n = diff.shape[0]
    diff, ref = diff.reshape(n, -1), ref.reshape(n, -1)
    w = np.ones(diff.shape[1]) if weights is None else np.asarray(weights)
    return np.linalg.norm(diff * np.sqrt(w)) / np.linalg.norm(ref * np.sqrt(w))


def test_field_weights():
    mu, snaps, volumes = synthetic()
    rom = SteadyROM(SPACE, "rbf").fit(mu, snaps, volumes)
    assert np.allclose(rom.pods_["p"].weights, volumes)
    assert np.allclose(rom.pods_["U"].weights, np.repeat(volumes, 3))
    assert rom.pods_["phi"].weights is None


def test_prediction_shapes():
    mu, snaps, volumes = synthetic()
    rom = SteadyROM(SPACE, "rbf").fit(mu, snaps, volumes)

    single = rom.predict(np.array([5.0]))
    assert single["U"].shape == (40, 3)
    assert single["p"].shape == (40,)
    assert single["phi"].shape == (120,)

    batch = rom.predict(mu[:4])
    assert batch["U"].shape == (4, 40, 3)
    assert batch["p"].shape == (4, 40)
    assert batch["phi"].shape == (4, 120)


@pytest.mark.parametrize("kind,tol", [("rbf", 1e-3), ("gpr", 1e-3)])
def test_prediction_error_on_held_out_samples(kind, tol):
    mu, snaps, volumes = synthetic(n=40)
    rom = SteadyROM(SPACE, kind).fit(mu, snaps, volumes)
    mu_test, snaps_test, _ = synthetic(n=10, seed=1)
    pred = rom.predict(mu_test)
    for name, weights in (("U", np.repeat(volumes, 3)), ("p", volumes), ("phi", None)):
        err = rel_l2(pred[name], snaps_test[name], weights)
        assert err < tol, f"{kind}/{name}: relative error {err:.3e}"


def test_mlp_prediction_error():
    pytest.importorskip("torch")
    mu, snaps, volumes = synthetic(n=40)
    rom = SteadyROM(SPACE, "mlp", regressor_kw={"adam_steps": 300, "lbfgs_steps": 100})
    rom.fit(mu, snaps, volumes)
    mu_test, snaps_test, _ = synthetic(n=10, seed=1)
    pred = rom.predict(mu_test)
    for name, weights in (("U", np.repeat(volumes, 3)), ("p", volumes), ("phi", None)):
        assert rel_l2(pred[name], snaps_test[name], weights) < 5e-2


def test_save_load_roundtrip(tmp_path):
    mu, snaps, volumes = synthetic()
    rom = SteadyROM(SPACE, "gpr").fit(mu, snaps, volumes)
    path = tmp_path / "models" / "gpr.pkl"
    rom.save(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = SteadyROM.load(path)
    assert loaded.space == SPACE and loaded.regressor == "gpr"
    before, after = rom.predict(mu[:5]), loaded.predict(mu[:5])
    for name in snaps:
        assert np.array_equal(before[name], after[name])


def test_save_records_dependency_versions(tmp_path):
    mu, snaps, volumes = synthetic()
    path = tmp_path / "rbf.pkl"
    SteadyROM(SPACE, "rbf").fit(mu, snaps, volumes).save(path)
    payload = pickle.loads(path.read_bytes())
    versions = payload["versions"]
    for key in ("numpy", "scipy", "sklearn", "torch", "fvmrom"):
        assert key in versions
    assert versions["numpy"] == np.__version__


def test_load_warns_on_version_mismatch(tmp_path):
    mu, snaps, volumes = synthetic()
    path = tmp_path / "rbf.pkl"
    SteadyROM(SPACE, "rbf").fit(mu, snaps, volumes).save(path)
    payload = pickle.loads(path.read_bytes())
    payload["versions"]["numpy"] = "0.0.0"
    path.write_bytes(pickle.dumps(payload))
    with pytest.warns(RuntimeWarning, match="dependency versions"):
        SteadyROM.load(path)


def test_load_pickle_without_versions(tmp_path):
    mu, snaps, volumes = synthetic()
    path = tmp_path / "rbf.pkl"
    SteadyROM(SPACE, "rbf").fit(mu, snaps, volumes).save(path)
    payload = pickle.loads(path.read_bytes())
    payload.pop("versions")
    path.write_bytes(pickle.dumps(payload))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = SteadyROM.load(path)
    assert loaded.regressor == "rbf"


def test_load_old_model_file(tmp_path):
    mu, snaps, volumes = synthetic()
    path = tmp_path / "rbf.pkl"
    SteadyROM(SPACE, "rbf").fit(mu, snaps, volumes).save(path)

    # emulate a pre-M5 pickle: no dependency versions, PODs store ``weights`` directly
    payload = pickle.loads(path.read_bytes())
    payload.pop("versions")
    for pod in payload["pods"].values():
        pod.__dict__["weights"] = pod.__dict__.pop("_weights")
    path.write_bytes(pickle.dumps(payload))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rom = SteadyROM.load(path)
    assert rom.regressor == "rbf" and "U" in rom.pods_
    assert np.allclose(rom.pods_["U"].weights, np.repeat(volumes, 3))
    assert rom.predict(np.array([5.0]))["U"].shape == (40, 3)


def test_n_modes_per_field():
    mu, snaps, volumes = synthetic()
    rom = SteadyROM(
        SPACE, "rbf", n_modes={"U": 2, "p": 1, "phi": 2}
    ).fit(mu, snaps, volumes)
    assert rom.pods_["U"].modes_.shape[0] == 2
    assert rom.pods_["p"].modes_.shape[0] == 1
    assert rom.pods_["phi"].modes_.shape[0] == 2

    full = SteadyROM(SPACE, "rbf").fit(mu, snaps, volumes)
    # the synthetic family spans two coefficients (sin(mu), exp(-mu/5))
    assert full.pods_["U"].modes_.shape[0] == 2
    assert full.pods_["p"].modes_.shape[0] == 1


def test_validation_errors():
    mu, snaps, volumes = synthetic()
    with pytest.raises(RuntimeError):
        SteadyROM(SPACE, "rbf").predict(mu[:2])
    with pytest.raises(ValueError):
        SteadyROM(SPACE, "rbf").fit(np.c_[mu, mu], snaps, volumes)  # wrong dim
    bad = dict(snaps, p=snaps["p"][:-1])
    with pytest.raises(ValueError):
        SteadyROM(SPACE, "rbf").fit(mu, bad, volumes)
    with pytest.raises(ValueError):
        SteadyROM(SPACE, "rbf").fit(mu, {}, volumes)
    with pytest.raises(ValueError):
        SteadyROM(SPACE, "rbf").fit(mu, snaps, volumes[:-1])
    with pytest.raises(ValueError):
        SteadyROM(SPACE, "kriging").fit(mu, snaps, volumes)
