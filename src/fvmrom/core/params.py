"""Physical parameter space and sampling (Sobol / Latin hypercube)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class ParamSpace:
    names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    log: tuple[bool, ...]

    def __post_init__(self):
        d = len(self.names)
        if d == 0 or len(self.lower) != d or len(self.upper) != d or len(self.log) != d:
            raise ValueError("names, lower, upper and log must have the same non-zero length")
        lo = np.asarray(self.lower, dtype=float)
        up = np.asarray(self.upper, dtype=float)
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(up)):
            raise ValueError("parameter bounds must be finite")
        if np.any(up <= lo):
            raise ValueError("every upper bound must be greater than its lower bound")
        if np.any(np.asarray(self.log, dtype=bool) & (lo <= 0)):
            raise ValueError("log-scaled parameters must have positive bounds")

    @property
    def dim(self) -> int:
        return len(self.names)

    def to_unit(self, mu: np.ndarray) -> np.ndarray:
        mu, single = _as_2d(mu, self.dim)
        lo = np.asarray(self.lower, dtype=float)
        up = np.asarray(self.upper, dtype=float)
        log = np.asarray(self.log, dtype=bool)
        if log.any() and np.any(~(mu[:, log] > 0.0)):
            raise ValueError("log-scaled parameters must be positive")
        z = np.empty_like(mu)
        lin = ~log
        z[:, lin] = (mu[:, lin] - lo[lin]) / (up[lin] - lo[lin])
        if log.any():
            z[:, log] = (np.log(mu[:, log]) - np.log(lo[log])) / (
                np.log(up[log]) - np.log(lo[log])
            )
        return z[0] if single else z

    def from_unit(self, z: np.ndarray) -> np.ndarray:
        z, single = _as_2d(z, self.dim)
        lo = np.asarray(self.lower, dtype=float)
        up = np.asarray(self.upper, dtype=float)
        log = np.asarray(self.log, dtype=bool)
        mu = np.empty_like(z)
        lin = ~log
        mu[:, lin] = lo[lin] + z[:, lin] * (up[lin] - lo[lin])
        if log.any():
            mu[:, log] = lo[log] * (up[log] / lo[log]) ** z[:, log]
        return mu[0] if single else mu

    def sample(self, n: int, method: str = "sobol", seed: int = 0) -> np.ndarray:
        if n <= 0:
            raise ValueError("n must be positive")
        if method == "sobol":
            # a power-of-two net avoids SciPy's balance warning; prefixes stay nested
            bits = max(1, int(np.ceil(np.log2(n))))
            z = qmc.Sobol(self.dim, scramble=True, seed=seed).random_base2(bits)[:n]
        elif method == "lhs":
            z = qmc.LatinHypercube(self.dim, seed=seed).random(n)
        else:
            raise ValueError(f"Unknown sampling method: {method!r} (use 'sobol' or 'lhs')")
        return self.from_unit(z)


def _as_2d(x: np.ndarray, dim: int) -> tuple[np.ndarray, bool]:
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        if a.size != dim:
            raise ValueError(f"expected {dim} values per sample, got {a.size}")
        return a.reshape(1, dim), True
    if a.ndim == 2 and a.shape[1] == dim:
        return a, False
    raise ValueError(f"expected an array of shape (n, {dim}), got {a.shape}")
