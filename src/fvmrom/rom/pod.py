"""Weighted proper orthogonal decomposition of a snapshot matrix."""

from __future__ import annotations

import numpy as np


class WeightedPOD:
    """POD of snapshots ``X`` (one snapshot per row) in a weighted inner product.

    ``weights`` gives the per-column weight (e.g. cell volumes for a cell-centred
    field); ``None`` means the unit weight.  The subtracted mean is the arithmetic
    snapshot mean: with per-component weights it also minimises the weighted residual.
    """

    def __init__(
        self,
        weights: np.ndarray | None = None,
        n_modes: int | None = None,
        energy_tol: float = 1e-10,
    ):
        if n_modes is not None and n_modes < 0:
            raise ValueError("n_modes must be non-negative")
        if not 0.0 <= energy_tol < 1.0:
            raise ValueError("energy_tol must lie in [0, 1)")
        self._weights = None if weights is None else np.asarray(weights, dtype=float)
        self.n_modes = n_modes
        self.energy_tol = float(energy_tol)
        self.mean_: np.ndarray | None = None
        self.modes_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None

    @property
    def weights(self) -> np.ndarray | None:
        """Per-column weights of the inner product (``None`` means unit weights)."""
        return self._weights

    def __setstate__(self, state: dict) -> None:
        # pickles written before ``weights`` became read-only store it as plain state
        if "weights" in state and "_weights" not in state:
            state["_weights"] = state.pop("weights")
        self.__dict__.update(state)

    def fit(self, X: np.ndarray) -> "WeightedPOD":
        X = _as_snapshots(X)
        w = self._weight_vector(X.shape[1])
        self.mean_ = X.mean(axis=0)
        centred = X - self.mean_
        _, s, vt = np.linalg.svd(centred * np.sqrt(w), full_matrices=False)
        r = self._rank(s)
        # the SVD is computed in sqrt(w)-scaled coordinates
        self.modes_ = vt[:r] / np.sqrt(w)
        self.singular_values_ = s[:r]
        return self

    def project(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = _as_snapshots(X, self.mean_.size)
        w = self._weight_vector(self.mean_.size)
        return (X - self.mean_) @ (w[:, None] * self.modes_.T)

    def reconstruct(self, A: np.ndarray) -> np.ndarray:
        self._check_fitted()
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[1] != self.modes_.shape[0]:
            raise ValueError(f"expected coefficients of shape (n, {self.modes_.shape[0]}), got {A.shape}")
        return A @ self.modes_ + self.mean_

    def _rank(self, s: np.ndarray) -> int:
        if s.size == 0 or s[0] <= 0.0:
            return 0
        energy = np.cumsum(s**2) / np.sum(s**2)
        r = int(np.searchsorted(energy, 1.0 - self.energy_tol, side="left")) + 1
        r = min(r, int(np.count_nonzero(s > s[0] * 1e-14)))
        if self.n_modes is not None:
            r = min(r, self.n_modes)
        return r

    def _weight_vector(self, n: int) -> np.ndarray:
        if self._weights is None:
            return np.ones(n)
        if self._weights.size != n:
            raise ValueError(f"expected {n} weights, got {self._weights.size}")
        if not np.all(np.isfinite(self._weights)) or np.any(self._weights <= 0.0):
            raise ValueError("weights must be positive and finite")
        return self._weights

    def _check_fitted(self) -> None:
        if self.modes_ is None:
            raise RuntimeError("WeightedPOD is not fitted")


def _as_snapshots(X: np.ndarray, n_cols: int | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"expected a snapshot matrix of shape (n, m), got {X.shape}")
    if n_cols is not None and X.shape[1] != n_cols:
        raise ValueError(f"expected {n_cols} values per snapshot, got {X.shape[1]}")
    return X
