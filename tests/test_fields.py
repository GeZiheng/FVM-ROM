import numpy as np
import pytest

from fvmrom.foam.fields import (
    FoamField,
    latest_time,
    phi_divergence,
    phi_vector,
    read_field,
    read_topology,
    write_field,
)

OWNER = r"""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    note        "nPoints:6 nCells:2 nFaces:9 nInternalFaces:1";
    class       labelList;
    location    "constant/polyMesh";
    object      owner;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


9
(
0
0
1
0
1
0
1
0
1
)
"""

NEIGHBOUR = r"""FoamFile
{
    version     2.0;
    format      ascii;
    note        "nPoints:6 nCells:2 nFaces:9 nInternalFaces:1";
    class       labelList;
    object      neighbour;
}

1
(
1
)
"""

BOUNDARY = r"""FoamFile
{
    version     2.0;
    format      ascii;
    class       polyBoundaryMesh;
    object      boundary;
}

5
(
    left
    {
        type            patch;
        inGroups        1(patch);
        nFaces          1;
        startFace       1;
    }
    right
    {
        type            patch;
        nFaces          1;
        startFace       2;
    }
    bottom
    {
        type            wall;
        nFaces          2;
        startFace       3;
    }
    top
    {
        type            wall;
        nFaces          2;
        startFace       5;
    }
    frontAndBack
    {
        type            empty;
        nFaces          2;
        startFace       7;
    }
)
"""

U = r"""/*--------------------------------*- C++ -*----------------------------------*\
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    location    "0";
    object      U;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   nonuniform List<vector>
2
(
(0.5 0 0)
(0.25 0.125 0)
)
;

boundaryField
{
    left
    {
        type            noSlip;
    }
    right
    {
        type            fixedValue;
        value           uniform (1 0 0);
    }
    bottom
    {
        type            noSlip;
    }
    top
    {
        type            fixedValue;
        value           nonuniform List<vector> 2 ((1 0 0) (0.5 0 0));
    }
    frontAndBack
    {
        type            empty;
    }
}
"""

P = r"""FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      p;
}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    left
    {
        type            zeroGradient;
    }
    right
    {
        type            fixedValue;
        value           uniform 0;
    }
    bottom
    {
        type            zeroGradient;
    }
    top
    {
        type            zeroGradient;
    }
    frontAndBack
    {
        type            empty;
    }
}
"""

PHI = r"""FoamFile
{
    version     2.0;
    format      ascii;
    class       surfaceScalarField;
    location    "0";
    object      phi;
}

dimensions      [0 3 -1 0 0 0 0];

internalField   nonuniform List<scalar> 1 (0.5);

boundaryField
{
    left
    {
        type            fixedValue;
        value           uniform -0.5;
    }
    right
    {
        type            fixedValue;
        value           uniform 0.5;
    }
    bottom
    {
        type            fixedValue;
        value           nonuniform List<scalar> 2 (0 0);
    }
    top
    {
        type            fixedValue;
        value           uniform 0;
    }
    frontAndBack
    {
        type            empty;
        value           nonuniform List<scalar> 0();
    }
}
"""


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "case"
    poly = root / "constant" / "polyMesh"
    poly.mkdir(parents=True)
    (poly / "owner").write_text(OWNER)
    (poly / "neighbour").write_text(NEIGHBOUR)
    (poly / "boundary").write_text(BOUNDARY)
    (root / "0").mkdir()
    (root / "0" / "U").write_text(U)
    (root / "0" / "p").write_text(P)
    (root / "0" / "phi").write_text(PHI)
    (root / "system").mkdir()
    return root


def test_read_vector_field(case):
    field = read_field(case, "0", "U")
    assert field.cls == "volVectorField"
    assert field.dimensions == "[0 1 -1 0 0 0 0]"
    assert field.internal.shape == (2, 3)
    assert np.allclose(field.internal, [[0.5, 0.0, 0.0], [0.25, 0.125, 0.0]])
    assert field.boundary["left"]["type"] == "noSlip"
    assert field.boundary["left"]["value"] is None
    assert field.boundary["right"]["type"] == "fixedValue"
    assert field.boundary["right"]["value"].shape == (1, 3)
    assert np.allclose(field.boundary["right"]["value"], [[1.0, 0.0, 0.0]])
    assert field.boundary["top"]["value"].shape == (2, 3)
    assert np.allclose(field.boundary["top"]["value"], [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert field.boundary["frontAndBack"] == {"type": "empty", "value": None}


def test_read_uniform_internal_field(case):
    field = read_field(case, "0", "p")
    assert field.cls == "volScalarField"
    assert field.internal.shape == (2,)
    assert np.allclose(field.internal, 0.0)
    assert np.allclose(field.boundary["right"]["value"], [0.0])


def test_write_roundtrip(case, tmp_path):
    for name in ("U", "p", "phi"):
        field = read_field(case, "0", name)
        path = write_field(tmp_path / "out", "1", field)
        assert path == tmp_path / "out" / "1" / name
        assert path.read_text().count("format      ascii;") == 1

        back = read_field(tmp_path / "out", "1", name)
        assert back.cls == field.cls
        assert back.dimensions == field.dimensions
        assert np.allclose(back.internal, field.internal)
        assert list(back.boundary) == list(field.boundary)
        for patch in field.boundary:
            assert back.boundary[patch]["type"] == field.boundary[patch]["type"]
            if field.boundary[patch]["value"] is None:
                assert back.boundary[patch]["value"] is None
            else:
                assert np.allclose(back.boundary[patch]["value"], field.boundary[patch]["value"])


def test_write_with_template(case):
    template = read_field(case, "0", "U")
    predicted = FoamField(
        name="U",
        cls=template.cls,
        internal=np.zeros((2, 3)),
        boundary={
            "left": {"type": "noSlip", "value": None},
            "right": {"type": "fixedValue", "value": np.array([[2.0, 0.0, 0.0]])},
        },
        dimensions=template.dimensions,
    )
    write_field(case, "1", predicted, template=template)

    back = read_field(case, "1", "U")
    assert list(back.boundary) == list(template.boundary)
    assert np.allclose(back.internal, 0.0)
    assert np.allclose(back.boundary["right"]["value"], [[2.0, 0.0, 0.0]])
    assert back.boundary["top"]["type"] == "fixedValue"
    assert back.boundary["top"]["value"] is None
    assert back.boundary["frontAndBack"]["type"] == "empty"


def test_binary_field_rejected(tmp_path):
    path = tmp_path / "0" / "U"
    path.parent.mkdir()
    path.write_text(U.replace("format      ascii;", "format      binary;"))
    with pytest.raises(ValueError, match="binary"):
        read_field(tmp_path, "0", "U")


def test_nonuniform_empty_list(tmp_path):
    path = tmp_path / "0" / "zero"
    path.parent.mkdir()
    path.write_text(
        "FoamFile\n{\n    class volScalarField;\n    object zero;\n}\n"
        "internalField   nonuniform 0();\nboundaryField\n{\n}\n"
    )
    field = read_field(tmp_path, "0", "zero")
    assert field.internal.size == 0
    assert field.boundary == {}


def test_missing_field(case, tmp_path):
    with pytest.raises(FileNotFoundError):
        read_field(case, "0", "T")
    with pytest.raises(FileNotFoundError):
        read_field(tmp_path, "0", "U")


def test_latest_time(case):
    for name in ("0.5", "2", "10"):
        (case / name).mkdir()
    assert latest_time(case) == "10"
    empty = case / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        latest_time(empty)


def test_read_topology(case):
    topo = read_topology(case)
    assert topo.n_cells == 2
    assert topo.neighbour.tolist() == [1]
    assert topo.owner.tolist() == [0, 0, 1, 0, 1, 0, 1, 0, 1]
    assert topo.patches["left"] == ("patch", 1, 1)
    assert topo.patches["bottom"] == ("wall", 3, 2)
    assert topo.patches["frontAndBack"] == ("empty", 7, 2)
    assert list(topo.patches) == ["left", "right", "bottom", "top", "frontAndBack"]


def test_read_topology_bad_patch_range(case):
    boundary = case / "constant" / "polyMesh" / "boundary"
    boundary.write_text(BOUNDARY.replace("startFace       7;", "startFace       8;"))
    with pytest.raises(ValueError, match="past the last face"):
        read_topology(case)


def test_phi_vector_skips_empty_patches(case):
    topo = read_topology(case)
    phi = phi_vector(read_field(case, "0", "phi"), topo)
    assert phi.shape == (7,)
    assert np.allclose(phi, [0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.0])


def test_phi_divergence_conservative(case):
    topo = read_topology(case)
    phi = phi_vector(read_field(case, "0", "phi"), topo)
    assert np.allclose(phi_divergence(phi, topo), 0.0)


def test_phi_divergence_detects_imbalance(case):
    topo = read_topology(case)
    field = read_field(case, "0", "phi")
    field.boundary["left"]["value"] = np.array([-0.6])
    div = phi_divergence(phi_vector(field, topo), topo)
    assert np.allclose(div, [-0.1, 0.0])


def test_phi_divergence_rejects_short_vector(case):
    topo = read_topology(case)
    with pytest.raises(ValueError, match="internal faces"):
        phi_divergence(np.zeros(0), topo)
    with pytest.raises(ValueError, match="face count"):
        phi_divergence(np.zeros(4), topo)
    with pytest.raises(ValueError, match="trailing"):
        phi_divergence(np.zeros(10), topo)


MIDDLE_OWNER = r"""FoamFile
{
    version     2.0;
    format      ascii;
    note        "nPoints:6 nCells:2 nFaces:5 nInternalFaces:1";
    class       labelList;
    object      owner;
}

5
(
0
0
0
1
1
)
"""

MIDDLE_NEIGHBOUR = r"""FoamFile
{
    version     2.0;
    format      ascii;
    class       labelList;
    object      neighbour;
}

1
(
1
)
"""

MIDDLE_BOUNDARY = r"""FoamFile
{
    version     2.0;
    format      ascii;
    class       polyBoundaryMesh;
    object      boundary;
}

3
(
    left
    {
        type            patch;
        nFaces          1;
        startFace       1;
    }
    frontAndBack
    {
        type            empty;
        nFaces          2;
        startFace       2;
    }
    right
    {
        type            patch;
        nFaces          1;
        startFace       4;
    }
)
"""


@pytest.fixture
def middle_case(tmp_path):
    root = tmp_path / "middle"
    poly = root / "constant" / "polyMesh"
    poly.mkdir(parents=True)
    (poly / "owner").write_text(MIDDLE_OWNER)
    (poly / "neighbour").write_text(MIDDLE_NEIGHBOUR)
    (poly / "boundary").write_text(MIDDLE_BOUNDARY)
    return root


def test_phi_vector_skips_empty_patch_in_the_middle(middle_case):
    topo = read_topology(middle_case)
    field = FoamField(
        "phi",
        "surfaceScalarField",
        np.array([0.5]),
        {
            "left": {"type": "patch", "value": np.array([-0.5])},
            # an empty patch is skipped by type even if it carries stray values
            "frontAndBack": {"type": "empty", "value": np.array([9.0, 9.0])},
            "right": {"type": "patch", "value": np.array([0.5])},
        },
    )
    phi = phi_vector(field, topo)
    assert np.allclose(phi, [0.5, -0.5, 0.5])
    assert np.allclose(phi_divergence(phi, topo), 0.0)


def test_phi_vector_requires_values_on_non_empty_patches(case):
    topo = read_topology(case)
    field = read_field(case, "0", "phi")
    del field.boundary["left"]
    with pytest.raises(ValueError, match="no phi value"):
        phi_vector(field, topo)

    field = read_field(case, "0", "phi")
    field.boundary["bottom"]["value"] = None
    with pytest.raises(ValueError, match="no phi value"):
        phi_vector(field, topo)
