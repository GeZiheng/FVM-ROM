import numpy as np
import pytest

from fvmrom.foam import MeshTopology
from fvmrom.params import ParamSpace
from fvmrom.rom.evaluate import (
    evaluate_on_test,
    learning_curve,
    loo_cv,
    projection_error,
    rel_l2_error,
    retained_energy,
    std_error_correlation,
)
from fvmrom.rom.pod import WeightedPOD
from fvmrom.rom.steady import SteadyROM

SPACE = ParamSpace(names=("Re",), lower=(100.0,), upper=(1000.0,), log=(True,))
N_CELLS = 2
# owner [0, 0, 1] with one internal face; both patches carry one face. The flux below
# has zero net divergence per cell.
TOPO = MeshTopology(
    owner=np.array([0, 0, 1]),
    neighbour=np.array([1]),
    patches={"left": ("patch", 1, 1), "right": ("patch", 2, 1)},
    n_cells=N_CELLS,
)
PHI_BASE = np.array([1.0, -1.0, 1.0])


def synthetic(n=12, seed=0, n_cells=N_CELLS):
    """Smooth family in Re: two coefficients, a divergence-free flux, non-uniform volumes."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(100.0, 1000.0, n)
    c1 = np.exp(-mu / 400.0)[:, None]
    c2 = np.sin(mu / 700.0)[:, None]
    x = np.linspace(0.0, 1.0, n_cells)
    U = np.zeros((n, n_cells, 3))
    U[:, :, 0] = c1 + c2 * x**2
    U[:, :, 1] = 0.5 * c2 + 0.0 * x
    p = c1 * (1.0 - x) + c2 * x**2
    phi = (c1 + 0.5 * c2) * PHI_BASE
    volumes = np.linspace(0.5, 1.5, n_cells)
    return mu, {"U": U, "p": p, "phi": phi}, volumes


def test_rel_l2_error():
    ref = np.array([[1.0, 2.0, 3.0], [4.0, 0.0, 0.0]])
    assert np.allclose(rel_l2_error(ref, ref), 0.0)
    assert np.allclose(rel_l2_error(2.0 * ref, ref), 1.0)

    weights = np.array([1.0, 2.0, 4.0])
    pred = ref.copy()
    pred[:, 0] = 2.0 * ref[:, 0]
    err = rel_l2_error(pred, ref, weights)
    assert err.shape == (2,)
    assert np.allclose(err, [np.sqrt(1.0 / 45.0), 1.0])

    # per-cell volumes are repeated for the three components of a vector field
    u_ref = np.arange(12.0).reshape(1, 4, 3)
    u_pred = u_ref + 1.0
    vols = np.array([0.5, 1.0, 2.0, 4.0])
    assert np.allclose(
        rel_l2_error(u_pred, u_ref, vols), rel_l2_error(u_pred, u_ref, np.repeat(vols, 3))
    )

    with pytest.raises(ValueError, match="shape"):
        rel_l2_error(ref[:, :2], ref)
    with pytest.raises(ValueError, match="weights"):
        rel_l2_error(u_pred, u_ref, np.ones(2))


def test_projection_error_and_retained_energy():
    _, snaps, volumes = synthetic()
    p = snaps["p"]
    pod = WeightedPOD(volumes).fit(p)
    assert pod.modes_.shape[0] == 2
    assert projection_error(pod, p).max() < 1e-12
    assert retained_energy(pod, p) > 1.0 - 1e-12

    truncated = WeightedPOD(volumes, n_modes=1).fit(p)
    assert projection_error(truncated, p).max() > 1e-3
    energy = retained_energy(truncated, p)
    assert 0.0 < energy < 1.0

    with pytest.raises(RuntimeError):
        retained_energy(WeightedPOD(volumes), p)


def test_evaluate_on_test():
    mu, snaps, volumes = synthetic(n=16)
    rom = SteadyROM(SPACE, "rbf").fit(mu, snaps, volumes)
    mu_test, snaps_test, _ = synthetic(n=6, seed=1)

    out = evaluate_on_test(rom, mu_test, snaps_test, volumes, TOPO)
    assert out["n_test"] == 6
    assert set(out["fields"]) == {"U", "p", "phi"}
    for name, stats in out["fields"].items():
        assert 0.0 < stats["mean"] <= stats["max"]
        assert 0.0 < stats["mean"] < 1e-2, f"{name}: {stats['mean']:.3e}"
        assert len(stats["errors"]) == 6
        assert stats["max"] == pytest.approx(max(stats["errors"]))

    div = out["phi_divergence"]
    assert div["fom_max"] == 0.0
    assert div["limit"] == 1e-10
    assert div["rom_max"] < 1e-10
    assert div["ok"] is True


def test_evaluate_on_test_without_topology():
    mu, snaps, volumes = synthetic(n=12)
    rom = SteadyROM(SPACE, "rbf").fit(mu, snaps, volumes)
    mu_test, snaps_test, _ = synthetic(n=4, seed=2)
    out = evaluate_on_test(rom, mu_test, snaps_test, volumes)
    assert out["phi_divergence"] is None

    with pytest.raises(ValueError, match="expected shape"):
        evaluate_on_test(rom, np.zeros((4, 2)), snaps_test, volumes)


def test_loo_cv():
    mu, snaps, volumes = synthetic(n=10)
    made, fitted_mu = [], []

    def make_rom():
        rom = SteadyROM(SPACE, "rbf")
        made.append(rom)
        fit = rom.fit

        def recording_fit(train_mu, train_snaps, train_volumes):
            fitted_mu.append(np.array(train_mu, dtype=float, copy=True))
            return fit(train_mu, train_snaps, train_volumes)

        rom.fit = recording_fit
        return rom

    out = loo_cv(make_rom, mu, snaps, volumes)
    assert len(made) == 10
    assert out["n"] == 10 and out["folds"] == 10
    for name in ("U", "p", "phi"):
        stats = out["fields"][name]
        assert len(stats["errors"]) == 10
        assert stats["mean"] <= stats["max"]
        assert stats["mean"] < 5e-2, f"{name}: {stats['mean']:.3e}"

    # every fold trains without the sample it scores
    assert len(fitted_mu) == 10
    for i, train_mu in enumerate(fitted_mu):
        assert train_mu.shape == (9, 1)
        assert not np.any(np.isclose(train_mu[:, 0], mu[i]))

    with pytest.raises(ValueError, match="at least two"):
        loo_cv(make_rom, mu[:1], {name: v[:1] for name, v in snaps.items()}, volumes)


def test_learning_curve_structure():
    mu, snaps, volumes = synthetic(n=12)
    mu_test, snaps_test, _ = synthetic(n=5, seed=1)

    out = learning_curve(
        ["rbf", "gpr"], (4, 8, 40), mu, snaps, mu_test, snaps_test, volumes, space=SPACE
    )
    assert out["sizes"] == [4, 8]  # 40 exceeds the training set
    assert out["fields"] == ["U", "p", "phi"]
    assert out["mlp"] is None
    for kind in ("rbf", "gpr"):
        rows = out["kinds"][kind]
        assert [row["n"] for row in rows] == [4, 8]
        for row in rows:
            assert row["train_seconds"] >= 0.0
            assert row["predict_seconds_per_sample"] >= 0.0
            assert set(row["fields"]) == {"U", "p", "phi"}
            for stats in row["fields"].values():
                assert np.isfinite(stats["mean"]) and stats["max"] >= stats["mean"]

    with pytest.raises(ValueError, match="space"):
        learning_curve(["rbf"], (4,), mu, snaps, mu_test, snaps_test, volumes)
    with pytest.raises(ValueError, match="no learning-curve size"):
        learning_curve(["rbf"], (50,), mu, snaps, mu_test, snaps_test, volumes, space=SPACE)


def test_std_error_correlation():
    mu, snaps, volumes = synthetic(n=16)
    mu_test, snaps_test, _ = synthetic(n=8, seed=1)
    rom = SteadyROM(SPACE, "gpr").fit(mu, snaps, volumes)

    out = std_error_correlation(rom, mu_test, snaps_test, volumes)
    assert out["method"] == "spearman" and out["n_samples"] == 8
    assert set(out["fields"]) == {"U", "p", "phi"}
    for name, info in out["fields"].items():
        assert info["n_modes"] >= 1
        for key in ("sample_rho", "mode_rho"):
            assert info[key] is None or -1.0 <= info[key] <= 1.0
        assert info["mode_rho"] is not None

    with pytest.raises(ValueError, match="GPR"):
        std_error_correlation(
            SteadyROM(SPACE, "rbf").fit(mu, snaps, volumes), mu_test, snaps_test, volumes
        )


def test_learning_curve_mlp_stability():
    pytest.importorskip("torch")
    mu, snaps, volumes = synthetic(n=10)
    mu_test, snaps_test, _ = synthetic(n=4, seed=1)

    out = learning_curve(
        ["mlp"],
        (6, 8),
        mu,
        snaps,
        mu_test,
        snaps_test,
        volumes,
        mlp_seeds=(0, 1),
        space=SPACE,
        regressor_kw={"adam_steps": 200, "lbfgs_steps": 50},
    )
    assert len(out["kinds"]["mlp"]) == 2
    assert out["mlp"]["seeds"] == [0, 1]
    for row, curve_row in zip(out["mlp"]["rows"], out["kinds"]["mlp"]):
        assert row["n"] == curve_row["n"]
        for name in ("U", "p", "phi"):
            assert len(row["per_seed"][name]["values"]) == 2
            assert row["per_seed"][name]["std"] >= 0.0
            # the learning-curve row is the single seed 0 network
            assert curve_row["fields"][name]["mean"] == pytest.approx(
                row["per_seed"][name]["values"][0]
            )
