"""Steady parametric ROM: one POD and one coefficient regressor per field."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np

from ..params import ParamSpace
from .pod import WeightedPOD
from .regressors import make_regressor

#: fields whose POD uses cell-volume weights; every other field uses unit weights
VOLUME_WEIGHTED = frozenset({"U", "p"})

_FORMAT_VERSION = 1


class SteadyROM:
    """Field-wise POD + regression over a shared parameter space.

    ``fit`` takes snapshots shaped (n, ...) per field; ``predict`` returns the
    same layouts, with the sample axis dropped for a single parameter vector.

    Models are stored as pickles: (:meth:`load`) only load local files you trust.
    """

    def __init__(
        self,
        space: ParamSpace,
        regressor: str = "gpr",
        n_modes: dict[str, int] | None = None,
        energy_tol: float = 1e-10,
        regressor_kw: dict | None = None,
    ):
        self.space = space
        self.regressor = regressor
        self.n_modes = dict(n_modes or {})
        self.energy_tol = energy_tol
        self.regressor_kw = dict(regressor_kw or {})
        self.pods_: dict[str, WeightedPOD] = {}
        self.models_: dict = {}
        self.shapes_: dict[str, tuple[int, ...]] = {}

    def fit(
        self,
        mu: np.ndarray,
        snapshots: dict[str, np.ndarray],
        volumes: np.ndarray | None,
    ) -> "SteadyROM":
        mu, _ = _as_samples(mu, self.space)
        if not snapshots:
            raise ValueError("no snapshots given")
        z = self.space.to_unit(mu)
        volumes = None if volumes is None else np.asarray(volumes, dtype=float).reshape(-1)
        n = mu.shape[0]

        for name, snap in snapshots.items():
            data = np.asarray(snap, dtype=float)
            if data.ndim == 1:
                data = data[:, None]
            if data.shape[0] != n:
                raise ValueError(
                    f"{name}: expected {n} snapshots, got {data.shape[0]}"
                )
            shape = data.shape[1:]
            flat = data.reshape(n, -1)
            weights = _weights_for(name, flat.shape[1], volumes)
            pod = WeightedPOD(
                weights, n_modes=self.n_modes.get(name), energy_tol=self.energy_tol
            ).fit(flat)
            model = make_regressor(self.regressor, **self.regressor_kw).fit(z, pod.project(flat))
            self.pods_[name] = pod
            self.models_[name] = model
            self.shapes_[name] = shape
        return self

    def predict(self, mu: np.ndarray) -> dict[str, np.ndarray]:
        if not self.pods_:
            raise RuntimeError("SteadyROM is not fitted")
        mu, single = _as_samples(mu, self.space)
        z = self.space.to_unit(mu)
        out = {}
        for name, pod in self.pods_.items():
            coeff = np.atleast_2d(np.asarray(self.models_[name].predict(z), dtype=float))
            values = pod.reconstruct(coeff).reshape((coeff.shape[0],) + self.shapes_[name])
            out[name] = values[0] if single else values
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _FORMAT_VERSION,
            "versions": _dependency_versions(),
            "space": self.space,
            "regressor": self.regressor,
            "n_modes": self.n_modes,
            "energy_tol": self.energy_tol,
            "regressor_kw": self.regressor_kw,
            "pods": self.pods_,
            "models": self.models_,
            "shapes": self.shapes_,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "SteadyROM":
        """Read a model written by :meth:`save`; only load local files you trust."""
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        if payload.get("version") != _FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported ROM format {payload.get('version')!r}")
        _check_versions(path, payload.get("versions"))
        rom = cls(
            payload["space"],
            regressor=payload["regressor"],
            n_modes=payload["n_modes"],
            energy_tol=payload["energy_tol"],
            regressor_kw=payload["regressor_kw"],
        )
        rom.pods_ = payload["pods"]
        rom.models_ = payload["models"]
        rom.shapes_ = payload["shapes"]
        return rom


def _dependency_versions() -> dict[str, str | None]:
    import scipy
    import sklearn

    from .. import __version__ as fvmrom_version
    from .regressors import torch

    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": None if torch is None else str(torch.__version__),
        "fvmrom": fvmrom_version,
    }


def _check_versions(path: Path, saved: dict | None) -> None:
    # Models written before M5 carry no version record; they load without a check.
    if not saved:
        return
    current = _dependency_versions()
    changed = [
        f"{key} {value} -> {current[key]}"
        for key, value in saved.items()
        if key in current and value != current[key]
    ]
    if changed:
        warnings.warn(
            f"{path}: model saved with different dependency versions: "
            + ", ".join(changed),
            RuntimeWarning,
            stacklevel=3,
        )


def _as_samples(mu: np.ndarray, space: ParamSpace) -> tuple[np.ndarray, bool]:
    """Return ``(n, d)`` parameters and whether the input was a single sample.

    A 1-D array is a single sample unless the space is one-dimensional, where it
    is read as a batch (a batch of one costs a reshape at most).
    """
    a = np.asarray(mu, dtype=float)
    if a.ndim == 1:
        if space.dim == 1 and a.size != 1:
            return a.reshape(-1, 1), False
        return a.reshape(1, -1), True
    if a.ndim == 2 and a.shape[1] == space.dim:
        return a, False
    raise ValueError(f"expected parameters of shape (n, {space.dim}), got {a.shape}")


def _weights_for(
    name: str, n_values: int, volumes: np.ndarray | None
) -> np.ndarray | None:
    if volumes is None or name not in VOLUME_WEIGHTED:
        return None
    if n_values == volumes.size:
        return volumes
    # U is stored as (n_cells, 3); all three components share the cell volume
    if n_values == 3 * volumes.size:
        return np.repeat(volumes, 3)
    raise ValueError(
        f"{name}: {n_values} values per snapshot cannot be weighted by {volumes.size} volumes"
    )
