import pytest
import operator
import json
import numpy as np
from molSimplify.Classes.mol2D import Mol2D
from molSimplify.Informatics.graph_racs import atom_centered_AC, multi_centered_AC, octahedral_racs


@pytest.fixture
def furan():
    mol = Mol2D()
    mol.add_nodes_from(
        [
            (0, {"symbol": "O"}),
            (1, {"symbol": "C"}),
            (2, {"symbol": "C"}),
            (3, {"symbol": "C"}),
            (4, {"symbol": "C"}),
            (5, {"symbol": "H"}),
            (6, {"symbol": "H"}),
            (7, {"symbol": "H"}),
            (8, {"symbol": "H"}),
        ]
    )
    mol.add_edges_from(
        [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 5), (2, 6), (4, 7), (3, 8)]
    )
    return mol


def test_atom_centered_AC(furan):
    descriptors = atom_centered_AC(furan, 0, depth=3)
    # properties: Z, chi, T,  I, S
    ref = [
        [64.0, 11.8336, 4.0, 1.0, 0.5329],
        [96.0, 17.544, 12.0, 2.0, 1.1242],
        [112.0, 32.68, 16.0, 4.0, 1.6644],
        [16.0, 15.136, 4.0, 2.0, 0.5402],
    ]
    np.testing.assert_allclose(descriptors, ref)


def test_atom_centered_AC_diff(furan):
    descriptors = atom_centered_AC(furan, 0, depth=3, operation=operator.sub)
    # properties: Z, chi, T,  I, S
    ref = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [4.0, 1.78, -2.0, 0.0, -0.08],
        [18.0, 4.26, 0.0, 0.0, 0.64],
        [14.0, 2.48, 2.0, 0.0, 0.72],
    ]
    np.testing.assert_allclose(descriptors, ref)


def test_multi_centered_AC(furan):
    descriptors = multi_centered_AC(furan, depth=3)
    # properties: Z, chi, T,  I, S
    ref = [
        [212.0, 57.2036, 44.0, 9.0, 3.4521],
        [456.0, 118.983, 102.0, 18.0, 8.0850],
        [512.0, 171.695, 122.0, 26.0, 10.3050],
        [110.0, 126.632, 50.0, 22.0, 5.3206],
    ]
    np.testing.assert_allclose(descriptors, ref)


@pytest.mark.parametrize(
    "mol2_path, ref_path, eq_atoms",
    [
        ("fe_carbonyl_6.mol2", "racs_Fe_carbonyl_6.json", None),
        (
            "mn_furan_water_ammonia_furan_water_ammonia.mol2",
            "racs_Mn_furan_water_ammonia_furan_water_ammonia.json",
            [10, 1, 17, 13],  # From ligand_assign_consistent
        ),
        (
            "co_acac_en_water_hydrogensulfide.mol2",
            "racs_Co_acac_en_water_hydrogensulfide.json",
            [15, 16, 1, 6],  # From ligand_assign_consistent
        ),
    ],
)
def test_octahedral_racs(
    resource_path_root, mol2_path, ref_path, eq_atoms, atol=1e-4
):

    mol = Mol2D.from_mol2_file(resource_path_root / "inputs" / "informatics" / mol2_path)

    with open(resource_path_root / "refs" / "informatics" / ref_path, "r") as fin:
        ref_dict = json.load(fin)

    depth = 3
    properties = ["Z", "chi", "T", "I", "S"]
    descriptors = octahedral_racs(
        mol,
        depth=depth,
        equatorial_connecting_atoms=eq_atoms,
    )

    # Dictionary encoded the order of the descriptors in the numpy array
    start_scopes = {
        0: ("f", "all"),
        1: ("mc", "all"),
        2: ("lc", "ax"),
        3: ("lc", "eq"),
        4: ("f", "ax"),
        5: ("f", "eq"),
        6: ("D_mc", "all"),
        7: ("D_lc", "ax"),
        8: ("D_lc", "eq"),
    }

    for s, (start, scope) in start_scopes.items():
        for d in range(depth + 1):
            for p, prop in enumerate(properties):
                print(
                    start,
                    scope,
                    d,
                    prop,
                    descriptors[s, d, p],
                    ref_dict[f"{start}-{prop}-{d}-{scope}"],
                )
                assert (
                    abs(descriptors[s, d, p] - ref_dict[f"{start}-{prop}-{d}-{scope}"])
                    < atol
                )
