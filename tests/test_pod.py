import numpy as np
import pytest

from fvmrom.rom.pod import WeightedPOD


def low_rank_data(n=12, m=40, rank=3, seed=0):
    rng = np.random.default_rng(seed)
    mean = rng.normal(size=m)
    a = rng.normal(size=(n, rank))
    b = rng.normal(size=(rank, m))
    return mean + a @ b, mean


def test_full_rank_reconstruction_with_weights():
    w = np.linspace(0.1, 3.0, 40)
    x, _ = low_rank_data()
    pod = WeightedPOD(w, n_modes=12, energy_tol=0.0).fit(x)
    rebuilt = pod.reconstruct(pod.project(x))
    assert np.allclose(rebuilt, x, atol=1e-10)


def test_full_rank_reconstruction_unit_weights():
    x, _ = low_rank_data()
    pod = WeightedPOD(None, n_modes=x.shape[0]).fit(x)
    assert np.allclose(pod.reconstruct(pod.project(x)), x, atol=1e-10)
    assert np.allclose(pod.mean_, x.mean(axis=0))


def test_weights_do_not_change_the_mean():
    w = np.ones(40) * 2.0
    x, _ = low_rank_data()
    pod = WeightedPOD(w).fit(x)
    assert np.allclose(pod.mean_, x.mean(axis=0))


def test_singular_values_decrease():
    x, _ = low_rank_data(rank=4)
    pod = WeightedPOD(None).fit(x)
    s = pod.singular_values_
    assert np.all(np.diff(s) <= 1e-12)
    assert s[0] > 0.0


def test_modes_orthonormal_in_weighted_inner_product():
    w = np.linspace(0.5, 2.0, 40)
    x, _ = low_rank_data(n=10, rank=4)
    pod = WeightedPOD(w, n_modes=4).fit(x)
    assert pod.modes_.shape == (4, 40)
    gram = pod.modes_ @ (w[:, None] * pod.modes_.T)
    assert np.allclose(gram, np.eye(4), atol=1e-10)


def test_projection_coefficients_orthogonal():
    w = np.linspace(0.5, 2.0, 40)
    x, _ = low_rank_data(n=10, rank=4)
    pod = WeightedPOD(w, n_modes=4).fit(x)
    a = pod.project(x)
    assert a.shape == (10, 4)
    assert np.allclose(a.T @ a, np.diag(pod.singular_values_**2), atol=1e-8)


def test_energy_tol_truncation():
    n, m, rank = 20, 30, 4
    rng = np.random.default_rng(1)
    u = np.linalg.qr(rng.normal(size=(n, rank)))[0]
    v = np.linalg.qr(rng.normal(size=(m, rank)))[0].T
    # squared singular values 1, 1e-2, 1e-6, 1e-16
    x = (u * np.array([1.0, 0.1, 1e-3, 1e-8])) @ v

    # 99.9 % energy keeps the first two modes
    pod = WeightedPOD(None, energy_tol=1e-3).fit(x)
    assert pod.modes_.shape[0] == 2
    total = np.sum(np.linalg.svd(x - x.mean(axis=0), compute_uv=False) ** 2)
    assert pod.singular_values_ @ pod.singular_values_ >= (1.0 - 1e-3) * total

    # the default tolerance drops the negligible fourth mode only
    assert WeightedPOD(None).fit(x).modes_.shape[0] == 3


def test_n_modes_caps_energy_rule():
    x, _ = low_rank_data(rank=3)
    assert WeightedPOD(None, n_modes=2).fit(x).modes_.shape[0] == 2
    assert WeightedPOD(None, n_modes=99).fit(x).modes_.shape[0] == 3


def test_zero_coefficients_reconstruct_the_mean():
    x, _ = low_rank_data(rank=2)
    pod = WeightedPOD(None, n_modes=2).fit(x)
    assert np.allclose(pod.reconstruct(np.zeros((1, 2)))[0], x.mean(axis=0), atol=1e-10)


def test_constant_snapshots_have_no_modes():
    x = np.ones((5, 7)) * 2.5
    pod = WeightedPOD(None).fit(x)
    assert pod.modes_.shape == (0, 7)
    assert np.allclose(pod.reconstruct(np.zeros((1, 0)))[0], 2.5)


def test_weights_property_is_read_only():
    w = np.linspace(0.5, 2.0, 40)
    pod = WeightedPOD(w)
    assert pod.weights is not None and np.allclose(pod.weights, w)
    with pytest.raises(AttributeError):
        pod.weights = np.ones(40)
    assert WeightedPOD(None).weights is None


def test_legacy_pickle_state_is_migrated():
    x, _ = low_rank_data()
    w = np.linspace(0.5, 2.0, 40)
    pod = WeightedPOD(w).fit(x)
    state = dict(pod.__dict__)
    legacy = {"weights": state.pop("_weights"), **state}
    restored = WeightedPOD.__new__(WeightedPOD)
    restored.__setstate__(legacy)
    assert np.allclose(restored.weights, w)
    assert np.allclose(restored.reconstruct(restored.project(x)), pod.reconstruct(pod.project(x)))


def test_validation_errors():
    x, _ = low_rank_data()
    with pytest.raises(ValueError):
        WeightedPOD(np.ones(3)).fit(x)
    with pytest.raises(ValueError):
        WeightedPOD(-np.ones(40)).fit(x)
    with pytest.raises(ValueError):
        WeightedPOD(None, n_modes=-1)
    with pytest.raises(ValueError):
        WeightedPOD(None, energy_tol=1.0)
    pod = WeightedPOD(None)
    with pytest.raises(RuntimeError):
        pod.project(x)
    with pytest.raises(RuntimeError):
        pod.reconstruct(np.zeros((1, 1)))
    pod.fit(x)
    with pytest.raises(ValueError):
        pod.project(x[:, :10])
    with pytest.raises(ValueError):
        pod.reconstruct(np.zeros((1, 2)))
