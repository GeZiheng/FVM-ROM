"""Regressors mapping unit-cube parameters to POD coefficients.

All regressors share the same interface: ``fit(Z, A)`` with unit-cube parameters
``Z`` of shape (n, d) and coefficients ``A`` of shape (n, r), and
``predict(Z, return_std=False)`` returning either (n, r) or the pair
(mean, std).  One output column is treated as one independent scalar target.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

try:  # PyTorch is the optional "nn" extra
    import torch
except ImportError:  # pragma: no cover - exercised only without the extra
    torch = None

_TORCH_HINT = (
    "MLPRegressor requires PyTorch; install it with `uv sync --extra rom --extra nn` "
    "(or `uv pip install torch`)"
)


class Regressor(Protocol):
    def fit(self, Z: np.ndarray, A: np.ndarray) -> "Regressor": ...

    def predict(
        self, Z: np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]: ...


class RBFRegressor:
    """Thin-plate-spline radial basis function interpolation (SciPy)."""

    def __init__(
        self,
        kernel: str = "thin_plate_spline",
        smoothing: float = 0.0,
        degree: int | None = None,
    ):
        self.kernel = kernel
        self.smoothing = float(smoothing)
        self.degree = degree
        self.model_: RBFInterpolator | None = None
        self.fitted_ = False

    def fit(self, Z: np.ndarray, A: np.ndarray) -> "RBFRegressor":
        Z = _as_2d(Z)
        A = _as_2d(A, n_rows=Z.shape[0])
        self.fitted_ = True
        if A.shape[1] == 0:
            self.model_ = None
            return self
        self.model_ = RBFInterpolator(
            Z, A, kernel=self.kernel, smoothing=self.smoothing, degree=self.degree
        )
        return self

    def predict(self, Z: np.ndarray, return_std: bool = False) -> np.ndarray:
        if return_std:
            raise ValueError("RBFRegressor does not provide prediction uncertainties")
        _check_fitted(self.fitted_, "RBFRegressor")
        if self.model_ is None:
            return np.zeros((_as_2d(Z).shape[0], 0))
        return np.asarray(self.model_(_as_2d(Z)), dtype=float)


class GPRRegressor:
    """Independent Gaussian processes, one per coefficient, with a Matern kernel."""

    def __init__(
        self,
        nu: float = 2.5,
        length_scale: float | np.ndarray = 1.0,
        length_scale_bounds: tuple[float, float] = (1e-2, 1e2),
        noise_level: float = 1e-6,
        noise_level_bounds: tuple[float, float] = (1e-10, 1e0),
        alpha: float = 1e-10,
        n_restarts: int = 0,
        random_state: int | None = 0,
        normalize_y: bool = True,
    ):
        self.nu = float(nu)
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.noise_level = float(noise_level)
        self.noise_level_bounds = noise_level_bounds
        self.alpha = float(alpha)
        self.n_restarts = int(n_restarts)
        self.random_state = random_state
        self.normalize_y = normalize_y
        self.models_: list[GaussianProcessRegressor] = []
        self.fitted_ = False

    def kernel(self, dim: int):
        # anisotropic Matern: one length scale per parameter
        length_scale = (
            np.full(dim, self.length_scale)
            if np.isscalar(self.length_scale)
            else np.asarray(self.length_scale, dtype=float)
        )
        return ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=length_scale,
            length_scale_bounds=self.length_scale_bounds,
            nu=self.nu,
        ) + WhiteKernel(
            noise_level=self.noise_level, noise_level_bounds=self.noise_level_bounds
        )

    def fit(self, Z: np.ndarray, A: np.ndarray) -> "GPRRegressor":
        Z = _as_2d(Z)
        A = _as_2d(A, n_rows=Z.shape[0])
        self.fitted_ = True
        self.models_ = [
            GaussianProcessRegressor(
                kernel=self.kernel(Z.shape[1]),
                alpha=self.alpha,
                normalize_y=self.normalize_y,
                n_restarts_optimizer=self.n_restarts,
                random_state=self.random_state,
            ).fit(Z, A[:, j])
            for j in range(A.shape[1])
        ]
        return self

    def predict(
        self, Z: np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        Z = _as_2d(Z)
        _check_fitted(self.fitted_, "GPRRegressor")
        if not self.models_:
            empty = np.zeros((Z.shape[0], 0))
            return (empty, empty) if return_std else empty
        if not return_std:
            return np.column_stack([m.predict(Z) for m in self.models_])
        means, stds = zip(*(m.predict(Z, return_std=True) for m in self.models_))
        return np.column_stack(means), np.column_stack(stds)


class MLPRegressor:
    """Fully connected network (SiLU), Adam then L-BFGS, with optional seeds.

    Inputs and outputs are standardised with training statistics.  ``seeds``
    gives one network per entry; predictions are averaged over them and
    ``predict(..., return_std=True)`` reports the ensemble spread.
    """

    def __init__(
        self,
        hidden: int = 64,
        n_layers: int = 2,
        seeds: tuple[int, ...] = (0,),
        adam_steps: int = 1000,
        adam_lr: float = 1e-2,
        lbfgs_steps: int = 200,
        weight_decay: float = 0.0,
        device: str = "cpu",
    ):
        _require_torch()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)
        self.seeds = tuple(int(s) for s in seeds)
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        self.adam_steps = int(adam_steps)
        self.adam_lr = float(adam_lr)
        self.lbfgs_steps = int(lbfgs_steps)
        self.weight_decay = float(weight_decay)
        self.device = device
        self.models_: list = []
        self.z_mean_ = np.zeros(1)
        self.z_std_ = np.ones(1)
        self.a_mean_ = np.zeros(0)
        self.a_std_ = np.ones(0)
        self.fitted_ = False

    def fit(self, Z: np.ndarray, A: np.ndarray) -> "MLPRegressor":
        Z = _as_2d(Z)
        A = _as_2d(A, n_rows=Z.shape[0])
        self.fitted_ = True
        self.z_mean_ = Z.mean(axis=0)
        self.z_std_ = _safe_std(Z)
        self.a_mean_ = A.mean(axis=0) if A.shape[1] else np.zeros(0)
        self.a_std_ = _safe_std(A) if A.shape[1] else np.ones(0)
        self.models_ = []
        if A.shape[1] == 0:
            return self
        x = torch.tensor((Z - self.z_mean_) / self.z_std_, dtype=torch.float64)
        y = torch.tensor((A - self.a_mean_) / self.a_std_, dtype=torch.float64)
        for seed in self.seeds:
            # fork_rng keeps the global torch RNG untouched by the seeded training
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                net = self._build(Z.shape[1], A.shape[1])
                _train(
                    net, x, y, self.adam_steps, self.adam_lr, self.lbfgs_steps, self.weight_decay
                )
            net.eval()
            self.models_.append(net)
        return self

    def predict(
        self, Z: np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        Z = _as_2d(Z)
        _check_fitted(self.fitted_, "MLPRegressor")
        if not self.models_:
            empty = np.zeros((Z.shape[0], 0))
            return (empty, empty) if return_std else empty
        if return_std and len(self.models_) < 2:
            raise ValueError("return_std requires an ensemble (at least two seeds)")
        x = torch.tensor((Z - self.z_mean_) / self.z_std_, dtype=torch.float64)
        with torch.no_grad():
            preds = np.stack([net(x).numpy() for net in self.models_])
        mean = preds.mean(axis=0) * self.a_std_ + self.a_mean_
        if not return_std:
            return mean
        return mean, preds.std(axis=0) * self.a_std_

    def _build(self, n_in: int, n_out: int):
        layers: list = []
        width = n_in
        for _ in range(self.n_layers):
            layers += [torch.nn.Linear(width, self.hidden), torch.nn.SiLU()]
            width = self.hidden
        layers.append(torch.nn.Linear(width, n_out))
        return torch.nn.Sequential(*layers).to(device=self.device, dtype=torch.float64)


#: regressor names accepted by :func:`make_regressor`
REGRESSOR_KINDS = {"rbf": RBFRegressor, "gpr": GPRRegressor, "mlp": MLPRegressor}


def make_regressor(kind: str, **kw) -> Regressor:
    try:
        cls = REGRESSOR_KINDS[kind.lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"unknown regressor {kind!r} (choose from {sorted(REGRESSOR_KINDS)})"
        ) from None
    return cls(**kw)


def _train(net, x, y, adam_steps: int, adam_lr: float, lbfgs_steps: int, weight_decay: float) -> None:
    loss_fn = torch.nn.MSELoss()
    adam = torch.optim.Adam(net.parameters(), lr=adam_lr, weight_decay=weight_decay)
    for _ in range(adam_steps):
        adam.zero_grad()
        loss_fn(net(x), y).backward()
        adam.step()
    lbfgs = torch.optim.LBFGS(
        net.parameters(), max_iter=lbfgs_steps, line_search_fn="strong_wolfe"
    )

    def closure():
        lbfgs.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        return loss

    lbfgs.step(closure)


def _require_torch() -> None:
    if torch is None:
        raise ImportError(_TORCH_HINT)


def _check_fitted(fitted: bool, name: str) -> None:
    if not fitted:
        raise RuntimeError(f"{name} is not fitted")


def _safe_std(a: np.ndarray) -> np.ndarray:
    std = a.std(axis=0)
    std[std == 0.0] = 1.0
    return std


def _as_2d(a: np.ndarray, n_rows: int | None = None) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"expected an array of shape (n, m), got {a.shape}")
    if n_rows is not None and a.shape[0] != n_rows:
        raise ValueError(f"expected {n_rows} rows, got {a.shape[0]}")
    return a
