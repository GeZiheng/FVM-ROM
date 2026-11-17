import numpy as np
import pytest

from fvmrom.rom.regressors import GPRRegressor, MLPRegressor, RBFRegressor, make_regressor


def analytic(z):
    return np.column_stack([np.sin(2 * np.pi * z[:, 0]), np.cos(np.pi * z[:, 0])])


def train_set(n=60):
    z = np.linspace(0.0, 1.0, n)[:, None]
    return z, analytic(z)


def held_out_set():
    # the spline and the GP overshoot slightly at the very edge of the domain
    z = np.linspace(0.02, 0.98, 150)[:, None]
    return z, analytic(z)


@pytest.fixture(scope="module")
def mlp():
    pytest.importorskip("torch")
    return MLPRegressor


def test_rbf_interpolates_analytic_function():
    z, a = train_set()
    zt, at = held_out_set()
    model = RBFRegressor().fit(z, a)
    err = np.abs(model.predict(zt) - at)
    assert err.shape == at.shape
    assert err.max() < 1e-3


def test_gpr_interpolates_analytic_function():
    z, a = train_set()
    zt, at = held_out_set()
    model = GPRRegressor().fit(z, a)
    err = np.abs(model.predict(zt) - at)
    assert err.max() < 1e-3


def test_gpr_uncertainty_grows_away_from_data():
    z = np.linspace(0.0, 0.4, 20)[:, None]
    a = analytic(z)
    model = GPRRegressor().fit(z, a)
    zt = np.array([[0.2], [1.5]])
    mean, std = model.predict(zt, return_std=True)
    assert mean.shape == (2, 2) and std.shape == (2, 2)
    assert np.all(std[0] < 1e-2)
    assert np.all(std[1] > 5.0 * std[0])


def test_gpr_constant_target():
    z, _ = train_set(20)
    a = np.full((20, 1), 3.5)
    model = GPRRegressor().fit(z, a)
    assert np.allclose(model.predict(np.array([[0.5]])), 3.5, atol=1e-4)


def test_gpr_anisotropic_kernel_has_one_length_scale_per_parameter():
    kernel = GPRRegressor(length_scale=0.3).kernel(3)
    assert kernel.k1.k2.length_scale.shape == (3,)
    assert np.allclose(kernel.k1.k2.length_scale, 0.3)
    assert kernel.k2.noise_level is not None


def test_gpr_two_parameters():
    rng = np.random.default_rng(0)
    z = rng.random((30, 2))
    a = z[:, :1] + 0.5 * z[:, 1:]
    model = GPRRegressor().fit(z, a)
    zt = rng.random((10, 2))
    at = zt[:, :1] + 0.5 * zt[:, 1:]
    assert np.abs(model.predict(zt) - at).max() < 1e-3


def test_predict_before_fit_raises():
    z = np.zeros((3, 1))
    for model in (RBFRegressor(), GPRRegressor()):
        with pytest.raises(RuntimeError):
            model.predict(z)


def test_rbf_rejects_return_std():
    z, a = train_set(20)
    model = RBFRegressor().fit(z, a)
    with pytest.raises(ValueError):
        model.predict(z, return_std=True)


def test_empty_targets():
    z, _ = train_set(10)
    empty = np.zeros((10, 0))
    for model in (RBFRegressor(), GPRRegressor()):
        assert model.fit(z, empty).predict(z).shape == (10, 0)


def test_mlp_accuracy(mlp):
    z, a = train_set()
    zt, at = held_out_set()
    model = mlp(seeds=(0,), adam_steps=800, lbfgs_steps=100).fit(z, a)
    err = np.abs(model.predict(zt) - at)
    assert err.max() < 1e-2


def test_mlp_is_reproducible(mlp):
    z, a = train_set(30)
    zt = np.linspace(0.0, 1.0, 25)[:, None]
    with pytest.raises(RuntimeError):
        mlp().predict(zt)
    kw = dict(seeds=(3,), adam_steps=200, lbfgs_steps=50)
    first = mlp(**kw).fit(z, a).predict(zt)
    second = mlp(**kw).fit(z, a).predict(zt)
    assert np.array_equal(first, second)


def test_mlp_ensemble_std(mlp):
    z, a = train_set(30)
    zt = np.linspace(0.0, 1.0, 25)[:, None]
    model = mlp(seeds=(0, 1), adam_steps=200, lbfgs_steps=50).fit(z, a)
    mean, std = model.predict(zt, return_std=True)
    assert mean.shape == std.shape == (25, 2)
    assert np.all(std > 0.0)
    single = mlp(seeds=(0,), adam_steps=200, lbfgs_steps=50).fit(z, a)
    with pytest.raises(ValueError):
        single.predict(zt, return_std=True)


def test_make_regressor():
    assert isinstance(make_regressor("rbf"), RBFRegressor)
    assert isinstance(make_regressor("GPR"), GPRRegressor)
    with pytest.raises(ValueError):
        make_regressor("kriging")
    assert make_regressor("rbf", smoothing=0.5).smoothing == 0.5


def test_mlp_without_torch_gives_a_clear_error(monkeypatch):
    import importlib
    import sys

    from fvmrom.rom import regressors

    # None in sys.modules makes ``import torch`` fail, like a missing extra
    monkeypatch.setitem(sys.modules, "torch", None)
    try:
        module = importlib.reload(regressors)
        assert module.torch is None
        with pytest.raises(ImportError, match="torch"):
            module.MLPRegressor()
    finally:
        monkeypatch.undo()
        importlib.reload(regressors)


def test_mlp_does_not_change_the_global_rng(mlp):
    import torch

    z, a = train_set(20)
    torch.manual_seed(123)
    before = torch.random.get_rng_state()
    mlp(seeds=(0,), adam_steps=20, lbfgs_steps=5).fit(z, a)
    assert torch.equal(before, torch.random.get_rng_state())


def test_shape_validation():
    z, a = train_set(10)
    with pytest.raises(ValueError):
        RBFRegressor().fit(z[:, 0], a)
    with pytest.raises(ValueError):
        RBFRegressor().fit(z, a[:, 0])
    with pytest.raises(ValueError):
        RBFRegressor().fit(z, a[:-1])
    model = RBFRegressor().fit(z, a)
    with pytest.raises(ValueError):
        model.predict(z[:, 0])
