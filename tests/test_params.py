import warnings

import numpy as np
import pytest

from fvmrom.params import ParamSpace

def re_space() -> ParamSpace:
    return ParamSpace(names=("Re",), lower=(100.0,), upper=(1000.0,), log=(True,))


def linear_space() -> ParamSpace:
    return ParamSpace(names=("a", "b"), lower=(0.0, 1.0), upper=(2.0, 3.0), log=(False, False))


def test_invalid_spaces():
    with pytest.raises(ValueError):
        ParamSpace(names=("a", "b"), lower=(0.0,), upper=(1.0, 2.0), log=(False, False))
    with pytest.raises(ValueError):
        ParamSpace(names=(), lower=(), upper=(), log=())
    with pytest.raises(ValueError):
        ParamSpace(names=("a",), lower=(1.0,), upper=(1.0,), log=(False,))
    with pytest.raises(ValueError):
        ParamSpace(names=("a",), lower=(0.0,), upper=(1.0,), log=(True,))


def test_linear_roundtrip():
    space = linear_space()
    mu = np.array([[0.0, 1.0], [1.0, 2.5], [2.0, 3.0]])
    z = space.to_unit(mu)
    assert z.shape == mu.shape
    assert np.allclose(z[0], [0.0, 0.0])
    assert np.allclose(z[-1], [1.0, 1.0])
    assert np.allclose(z[1], [0.5, 0.75])
    assert np.allclose(space.from_unit(z), mu)


def test_log_mapping():
    space = re_space()
    z = space.to_unit(np.array([[100.0], [1000.0], [np.sqrt(100.0 * 1000.0)]]))
    assert np.allclose(z[:, 0], [0.0, 1.0, 0.5])
    mu = space.from_unit(np.array([[0.0], [1.0], [0.5]]))
    assert np.allclose(mu, np.array([[100.0], [1000.0], [100.0 * np.sqrt(10.0)]]))


def test_single_sample_shapes():
    space = re_space()
    assert space.to_unit(np.array([100.0])).shape == (1,)
    assert space.from_unit(np.array([0.0])).shape == (1,)
    with pytest.raises(ValueError):
        space.to_unit(np.array([100.0, 200.0]))
    with pytest.raises(ValueError):
        space.to_unit(np.zeros((3, 2)))


@pytest.mark.parametrize("method", ["sobol", "lhs"])
def test_sample_within_bounds(method):
    space = re_space()
    mu = space.sample(16, method=method, seed=7)
    assert mu.shape == (16, 1)
    assert np.all(mu >= 100.0) and np.all(mu <= 1000.0)
    z = space.to_unit(mu)
    assert np.all(z >= 0.0) and np.all(z <= 1.0)

    lin = linear_space().sample(16, method=method, seed=7)
    assert lin.shape == (16, 2)
    assert np.all(lin[:, 0] >= 0.0) and np.all(lin[:, 0] <= 2.0)
    assert np.all(lin[:, 1] >= 1.0) and np.all(lin[:, 1] <= 3.0)


@pytest.mark.parametrize("method", ["sobol", "lhs"])
def test_sample_reproducible(method):
    space = re_space()
    a = space.sample(8, method=method, seed=3)
    b = space.sample(8, method=method, seed=3)
    c = space.sample(8, method=method, seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sample_log_is_not_linear():
    space = re_space()
    mu = space.sample(256, seed=1)[:, 0]
    # A log-uniform sample over [100, 1000] has its median near the geometric mean.
    assert np.median(mu) < 500.0


def test_sample_errors():
    space = re_space()
    with pytest.raises(ValueError):
        space.sample(0)
    with pytest.raises(ValueError):
        space.sample(4, method="halton")


def test_log_parameters_reject_non_positive_values():
    space = re_space()
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="positive"):
            space.to_unit(np.array([[bad]]))
    # linear parameters may be zero or negative
    assert np.allclose(linear_space().to_unit(np.array([[-1.0, 1.0]])), [-0.5, 0.0])


def test_sobol_sample_silent_for_non_power_of_two():
    # the default study sizes (30 train, 10 test) are not powers of two
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mu = re_space().sample(30, seed=0)
    assert mu.shape == (30, 1)
