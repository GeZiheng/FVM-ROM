"""Transient ROM: latent dynamics plus readout maps for the other fields."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .dynamics import make_dynamics
from .pod import WeightedPOD
from .steady import _check_versions, _dependency_versions, _weights_for

_FORMAT_VERSION = 1

#: effective number of modes per POD when ``n_modes`` gives none
_DEFAULT_N_MODES = {"U": 8, "p": 8, "phi": 8}

#: fields reconstructed from the latent state by a readout instead of dynamics
_READOUT_FIELDS = ("p", "phi")

#: default readout orders; ``"scalars"`` covers every scalar output
_DEFAULT_READOUT_ORDER = {"p": 2, "phi": 1, "scalars": 2}


class Readout:
    """Polynomial map from latent coefficients to a target field or scalar.

    ``order=1`` gives ``y = c + G a``; ``order=2`` adds the quadratic terms
    ``K (a ⊗ a)`` with the upper triangle of ``a ⊗ a`` only.  The design matrix
    columns are normalised by their norms before the ridge solve, so ``ridge``
    (default ``1e-10``) is relative to unit columns and the quadratic terms do
    not dominate the conditioning.
    """

    def __init__(self, order: int = 1, ridge: float = 1e-10):
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        if ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        self.order = int(order)
        self.ridge = float(ridge)
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.n_inputs_: int = 0
        self.n_outputs_: int = 0

    def fit(self, A: np.ndarray, Y: np.ndarray) -> "Readout":
        A = _as_2d(A, "A")
        Y = _as_2d(Y, "Y", n_rows=A.shape[0])
        self.n_inputs_ = A.shape[1]
        self.n_outputs_ = Y.shape[1]
        design = self._design(A)
        self.scale_ = np.linalg.norm(design, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        scaled = design / self.scale_
        if self.ridge > 0.0:
            n_cols = scaled.shape[1]
            scaled = np.vstack([scaled, np.sqrt(self.ridge) * np.eye(n_cols)])
            Y = np.vstack([Y, np.zeros((n_cols, Y.shape[1]))])
        coef = np.linalg.lstsq(scaled, Y, rcond=None)[0]
        self.coef_ = coef / self.scale_[:, None]
        return self

    def predict(self, A: np.ndarray) -> np.ndarray:
        self._check_fitted()
        A = _as_2d(A, "A")
        if A.shape[1] != self.n_inputs_:
            raise ValueError(
                f"expected {self.n_inputs_} latent coefficients, got {A.shape[1]}"
            )
        return self._design(A) @ self.coef_

    def _design(self, A: np.ndarray) -> np.ndarray:
        columns = [np.ones((A.shape[0], 1)), A]
        if self.order == 2:
            i, j = np.triu_indices(A.shape[1])
            columns.append(A[:, i] * A[:, j])
        return np.hstack(columns)

    def _check_fitted(self) -> None:
        if self.coef_ is None:
            raise RuntimeError("Readout is not fitted")


class TransientROM:
    """POD + latent dynamics for ``U``, readout maps for the other outputs.

    The latent state is the centred POD coefficient of ``U`` (cell-volume
    weights).  ``p`` and ``phi`` get their own POD; their coefficients are
    predicted from the latent state by a polynomial readout, and scalar
    outputs (e.g. force coefficients) are predicted directly by a readout.
    Every basis, readout and the dynamics are fitted on the given training
    window only.

    Effective defaults: 8 modes per field (``U``, ``p``, ``phi``; ``None`` for
    a field leaves the count to ``energy_tol``) and readout orders
    ``{"p": 2, "phi": 1, "scalars": 2}``; entries in ``readout_order``
    override the defaults per field or per scalar name.

    Models are stored as pickles: :meth:`load` only loads local files you trust.
    """

    def __init__(
        self,
        dynamics: str = "opinf",
        n_modes: dict[str, int | None] | None = None,
        energy_tol: float = 1e-10,
        readout_order: dict[str, int] | None = None,
        readout_ridge: float = 1e-10,
        dynamics_kw: dict | None = None,
    ):
        if not 0.0 <= energy_tol < 1.0:
            raise ValueError("energy_tol must lie in [0, 1)")
        if readout_ridge < 0.0:
            raise ValueError("readout_ridge must be non-negative")
        self.dynamics = str(dynamics)
        self.n_modes = dict(n_modes or {})
        for name, r in self.n_modes.items():
            if r is not None and (not isinstance(r, (int, np.integer)) or r < 0):
                raise ValueError(f"n_modes[{name!r}] must be a non-negative integer or None")
        self.energy_tol = float(energy_tol)
        self.readout_order = {**_DEFAULT_READOUT_ORDER, **dict(readout_order or {})}
        for name, order in self.readout_order.items():
            if order not in (1, 2):
                raise ValueError(f"readout_order[{name!r}] must be 1 or 2")
        self.readout_ridge = float(readout_ridge)
        self.dynamics_kw = dict(dynamics_kw or {})
        self.pods_: dict[str, WeightedPOD] = {}
        self.readouts_: dict[str, Readout] = {}
        self.shapes_: dict[str, tuple[int, ...]] = {}
        self.scalar_shapes_: dict[str, tuple[int, ...]] = {}
        self.n_modes_: dict[str, int] = {}
        self.model_ = None
        self.t_train: np.ndarray | None = None
        self.t_first: float | None = None
        self.latent_train_: np.ndarray | None = None
        self.volumes: np.ndarray | None = None
        self.phi_weights: np.ndarray | None = None

    def fit(
        self,
        times: np.ndarray,
        snapshots: dict[str, np.ndarray],
        volumes: np.ndarray,
        phi_weights: np.ndarray | None = None,
        scalars: dict[str, np.ndarray] | None = None,
    ) -> "TransientROM":
        """Fit on one uniform training window; predictions for other fields follow from ``U``."""
        t = _as_times(times, uniform=True)
        if not snapshots:
            raise ValueError("no snapshots given")
        if "U" not in snapshots:
            raise ValueError("snapshots must contain 'U' (the latent dynamics state)")
        unknown = sorted(set(snapshots) - {"U", *_READOUT_FIELDS})
        if unknown:
            raise ValueError(f"unsupported snapshot fields: {', '.join(unknown)}")
        nt = t.size
        self.volumes = np.asarray(volumes, dtype=float).reshape(-1)
        self.phi_weights = (
            None if phi_weights is None else np.asarray(phi_weights, dtype=float).reshape(-1)
        )
        self.pods_, self.readouts_ = {}, {}
        self.shapes_, self.scalar_shapes_, self.n_modes_ = {}, {}, {}

        flats: dict[str, np.ndarray] = {}
        for name, snap in snapshots.items():
            data = np.asarray(snap, dtype=float)
            if data.ndim == 0 or data.shape[0] != nt:
                raise ValueError(f"{name}: expected {nt} snapshots, got shape {data.shape}")
            if not np.all(np.isfinite(data)):
                raise ValueError(f"{name}: snapshots contain non-finite values")
            flat = data.reshape(nt, -1)
            requested = self.n_modes.get(name, _DEFAULT_N_MODES.get(name))
            if requested is not None and requested > nt:
                raise ValueError(
                    f"{name}: n_modes {requested} exceeds the {nt} training snapshots"
                )
            pod = WeightedPOD(
                _weights_for(name, flat.shape[1], self.volumes, self.phi_weights),
                n_modes=requested,
                energy_tol=self.energy_tol,
            ).fit(flat)
            self.pods_[name] = pod
            self.shapes_[name] = data.shape[1:]
            self.n_modes_[name] = pod.modes_.shape[0]
            flats[name] = flat

        a = self.pods_["U"].project(flats["U"])
        for name in _READOUT_FIELDS:
            if name in self.pods_:
                target = self.pods_[name].project(flats[name])
                self.readouts_[name] = self._readout(name).fit(a, target)
        for name, values in (scalars or {}).items():
            if name in self.shapes_ or name in ("U", *_READOUT_FIELDS):
                raise ValueError(f"scalar output {name!r} clashes with a snapshot field")
            data = np.asarray(values, dtype=float)
            if data.ndim == 1:
                data = data[:, None]
            if data.ndim != 2 or data.shape[0] != nt:
                raise ValueError(f"{name}: expected {nt} scalar rows, got shape {data.shape}")
            if not np.all(np.isfinite(data)):
                raise ValueError(f"{name}: scalar output contains non-finite values")
            self.scalar_shapes_[name] = data.shape[1:]
            self.readouts_[name] = self._readout(name, group="scalars").fit(a, data)

        self.model_ = make_dynamics(self.dynamics, **self.dynamics_kw).fit(t, a)
        self.t_train = t
        self.t_first = float(t[0])
        self.latent_train_ = a
        return self

    def latent(self, times: np.ndarray) -> np.ndarray:
        """Latent coefficients of ``U`` at ``times``, shape ``(len(times), r)``."""
        self._check_fitted()
        return np.asarray(self.model_.predict(_as_times(times)), dtype=float)

    def project(self, field: str, snapshots: np.ndarray) -> np.ndarray:
        """Reconstruct snapshots of ``field`` through its trained POD.

        The result keeps the input shape and is the projection limit of the ROM:
        the error of this reconstruction is a lower bound for the prediction error.
        """
        self._check_fitted()
        if field not in self.pods_:
            raise ValueError(f"the ROM has no POD for {field!r}")
        data = np.asarray(snapshots, dtype=float)
        if data.ndim == 0:
            raise ValueError("expected at least one snapshot")
        flat = data.reshape(data.shape[0], -1)
        return self.pods_[field].reconstruct(self.pods_[field].project(flat)).reshape(
            data.shape
        )

    def predict(self, times: np.ndarray) -> dict[str, np.ndarray]:
        """Reconstruct every trained field and scalar output, time axis first."""
        self._check_fitted()
        t = _as_times(times)
        a = np.asarray(self.model_.predict(t), dtype=float)
        out = {"U": self._reconstruct("U", a)}
        for name in _READOUT_FIELDS:
            if name in self.readouts_:
                out[name] = self._reconstruct(name, self.readouts_[name].predict(a))
        for name, shape in self.scalar_shapes_.items():
            values = self.readouts_[name].predict(a)
            out[name] = values.reshape((t.size,) + shape)
        return out

    def spectrum(self) -> dict | None:
        """Latent dynamics spectrum (``freq``, ``growth``, ``amplitude``), if any."""
        self._check_fitted()
        return self.model_.spectrum()

    def save(self, path: Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _FORMAT_VERSION,
            "versions": _dependency_versions(),
            "dynamics": self.dynamics,
            "n_modes": self.n_modes,
            "energy_tol": self.energy_tol,
            "readout_order": self.readout_order,
            "readout_ridge": self.readout_ridge,
            "dynamics_kw": self.dynamics_kw,
            "pods": self.pods_,
            "readouts": self.readouts_,
            "shapes": self.shapes_,
            "scalar_shapes": self.scalar_shapes_,
            "n_modes_": self.n_modes_,
            "model": self.model_,
            "t_train": self.t_train,
            "latent_train": self.latent_train_,
            "volumes": self.volumes,
            "phi_weights": self.phi_weights,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "TransientROM":
        """Read a model written by :meth:`save`; only load local files you trust."""
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        if payload.get("version") != _FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported ROM format {payload.get('version')!r}")
        _check_versions(path, payload.get("versions"))
        rom = cls(
            dynamics=payload["dynamics"],
            n_modes=payload["n_modes"],
            energy_tol=payload["energy_tol"],
            readout_order=payload["readout_order"],
            readout_ridge=payload["readout_ridge"],
            dynamics_kw=payload["dynamics_kw"],
        )
        rom.pods_ = payload["pods"]
        rom.readouts_ = payload["readouts"]
        rom.shapes_ = payload["shapes"]
        rom.scalar_shapes_ = payload["scalar_shapes"]
        rom.n_modes_ = payload["n_modes_"]
        rom.model_ = payload["model"]
        rom.t_train = payload["t_train"]
        rom.t_first = None if rom.t_train is None else float(rom.t_train[0])
        rom.latent_train_ = payload["latent_train"]
        rom.volumes = payload["volumes"]
        rom.phi_weights = payload["phi_weights"]
        return rom

    def _readout(self, name: str, group: str | None = None) -> Readout:
        key = name if name in self.readout_order else (group or name)
        return Readout(order=self.readout_order[key], ridge=self.readout_ridge)

    def _reconstruct(self, name: str, coefficients: np.ndarray) -> np.ndarray:
        pod = self.pods_[name]
        values = coefficients @ pod.modes_
        values += pod.mean_  # in place: no second copy of the (n, n_dof) array
        return values.reshape((coefficients.shape[0],) + self.shapes_[name])

    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError("TransientROM is not fitted")


def _as_2d(a: np.ndarray, name: str, n_rows: int | None = None) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"expected {name} of shape (n, m), got {a.shape}")
    if n_rows is not None and a.shape[0] != n_rows:
        raise ValueError(f"expected {n_rows} rows in {name}, got {a.shape[0]}")
    return a


def _as_times(times: np.ndarray, uniform: bool = False) -> np.ndarray:
    t = np.asarray(times, dtype=float).reshape(-1)
    if t.size == 0:
        raise ValueError("times must not be empty")
    if not np.all(np.isfinite(t)):
        raise ValueError("times must be finite")
    if t.size > 1:
        dt = t[1] - t[0]
        if not dt > 0.0:
            raise ValueError("times must be strictly increasing")
        if uniform and np.max(np.abs(np.diff(t) - dt)) > 1e-6 * abs(dt):
            raise ValueError("training times must be uniformly spaced")
    return t
