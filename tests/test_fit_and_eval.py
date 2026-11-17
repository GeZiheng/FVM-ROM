"""Unit tests for the helpers of scripts/fit_and_eval.py (no OpenFOAM)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from fvmrom.foam import FoamField, MeshTopology, read_field

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fit_and_eval.py"
_spec = importlib.util.spec_from_file_location("fit_and_eval", SCRIPT)
fit_and_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit_and_eval)

TOPO = MeshTopology(
    owner=np.array([0, 0, 1]),
    neighbour=np.array([1]),
    patches={"left": ("patch", 1, 1), "right": ("patch", 2, 1)},
    n_cells=2,
)


def _templates():
    boundary = {"left": {"type": "patch", "value": None}, "right": {"type": "patch", "value": None}}
    U = FoamField(
        "U", "volVectorField", np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        dict(boundary), "[0 1 -1 0 0 0 0]",
    )  # fmt: skip
    p = FoamField("p", "volScalarField", np.array([0.1, 0.2]), dict(boundary), "[0 2 -2 0 0 0 0]")
    phi = FoamField(
        "phi",
        "surfaceScalarField",
        np.array([0.5]),
        {
            "left": {"type": "patch", "value": np.array([-0.5])},
            "right": {"type": "patch", "value": np.array([0.5])},
        },
        "[0 3 -1 0 0 0 0]",
    )
    return {"U": U, "p": p, "phi": phi}


def test_write_rom_case_writes_rom_fom_and_differences(tmp_path):
    templates = _templates()
    pred = {
        "U": np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]),
        "p": np.array([0.15, 0.25]),
        "phi": np.array([0.6, -0.6, 0.6]),
    }
    fit_and_eval.write_rom_case(tmp_path, pred, templates, TOPO)

    U = read_field(tmp_path, "1", "U")
    assert U.cls == "volVectorField"
    assert np.allclose(U.internal, pred["U"])

    U_fom = read_field(tmp_path, "1", "U_fom")
    assert U_fom.cls == "volVectorField"
    assert np.allclose(U_fom.internal, templates["U"].internal)

    U_err = read_field(tmp_path, "1", "U_err")
    assert U_err.cls == "volVectorField"
    assert np.allclose(U_err.internal, pred["U"] - templates["U"].internal)
    assert U_err.dimensions == "[0 1 -1 0 0 0 0]"
    assert U_err.boundary["left"]["type"] == "patch"

    p_err = read_field(tmp_path, "1", "p_err")
    assert p_err.cls == "volScalarField"
    assert np.allclose(p_err.internal, pred["p"] - templates["p"].internal)
    assert np.allclose(read_field(tmp_path, "1", "p_fom").internal, templates["p"].internal)

    phi = read_field(tmp_path, "1", "phi")
    assert np.allclose(phi.internal, pred["phi"][:1])
    assert np.allclose(phi.boundary["left"]["value"], pred["phi"][1:2])

    header = (tmp_path / "1" / "U_err").read_text(encoding="utf-8")
    assert "object      U_err" in header and "class       volVectorField" in header


def test_empty_regressor_list_is_an_argparse_error(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["fit_and_eval.py", "--problem", "cavity", "--regressors", ""]
    )
    with pytest.raises(SystemExit) as exc:
        fit_and_eval.parse_args()
    assert exc.value.code == 2


def test_unknown_regressor_is_an_argparse_error(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["fit_and_eval.py", "--problem", "cavity", "--regressors", "rbf,kriging"]
    )
    with pytest.raises(SystemExit) as exc:
        fit_and_eval.parse_args()
    assert exc.value.code == 2
