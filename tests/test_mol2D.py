import pytest
from molSimplify.Classes.mol2D import Mol2D
from molSimplify.Classes.mol3D import mol3D as Mol3D


def water_Mol2D():
    mol = Mol2D()
    mol.add_nodes_from(
        [(0, {"symbol": "O"}), (1, {"symbol": "H"}), (2, {"symbol": "H"})]
    )
    mol.add_edges_from([(0, 1), (0, 2)])
    return mol


def furan_Mol2D():
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
        [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 5), (2, 6), (3, 7), (4, 8)]
    )
    return mol


def acac_Mol2D():
    mol = Mol2D()
    mol.add_nodes_from(
        [
            (0, {"symbol": "O"}),
            (1, {"symbol": "C"}),
            (2, {"symbol": "C"}),
            (3, {"symbol": "C"}),
            (4, {"symbol": "C"}),
            (5, {"symbol": "O"}),
            (6, {"symbol": "C"}),
            (7, {"symbol": "H"}),
            (8, {"symbol": "H"}),
            (9, {"symbol": "H"}),
            (10, {"symbol": "H"}),
            (11, {"symbol": "H"}),
            (12, {"symbol": "H"}),
            (13, {"symbol": "H"}),
        ]
    )
    mol.add_edges_from(
        [(0, 1), (1, 2), (1, 3), (2, 7), (2, 8), (2, 9), (3, 4),
         (3, 10), (4, 5), (4, 6), (6, 11), (6, 12), (6, 13)]
    )
    return mol


@pytest.mark.parametrize(
    "mol, ref",
    [
        (water_Mol2D(), "Mol2D(O1H2)"),
        (furan_Mol2D(), "Mol2D(O1C4H4)"),
        (acac_Mol2D(), "Mol2D(O2C5H7)"),
    ])
def test_Mol2D_repr(mol, ref):
    assert mol.__repr__() == ref


@pytest.mark.parametrize(
    "name, smiles, mol_ref",
    [
        ("water", "O", water_Mol2D()),
        ("furan", "o1cccc1", furan_Mol2D()),
        ("acac", "[O-]-C(C)=CC(=O)C", acac_Mol2D()),
    ]
)
def test_Mol2D_constructors(resource_path_root, name, smiles, mol_ref):
    # From mol file
    mol = Mol2D.from_mol_file(resource_path_root / "inputs" / "io" / f"{name}.mol")

    assert mol.nodes == mol_ref.nodes
    assert mol.edges == mol_ref.edges

    # From mol2 file
    mol = Mol2D.from_mol2_file(resource_path_root / "inputs" / "io" / f"{name}.mol2")

    assert mol.nodes == mol_ref.nodes
    assert mol.edges == mol_ref.edges

    # From smiles
    mol = Mol2D.from_smiles(smiles)

    assert mol.nodes == mol_ref.nodes
    assert mol.edges == mol_ref.edges

    # from mol3D
    mol3d = Mol3D()
    mol3d.readfrommol2(resource_path_root / "inputs" / "io" / f"{name}.mol2")
    mol = Mol2D.from_mol3d(mol3d)

    assert mol.nodes == mol_ref.nodes
    assert mol.edges == mol_ref.edges


@pytest.mark.parametrize(
    "filename, node_hash_ref, edge_hash_ref, graph_det_ref",
    [
        ("water.mol2", "b25a2b34d8df66723302a66f53bc3773", "945f74e35e6192f56d3f77b55c702178", "-507.9380086"),
        ("formaldehyde.mol2", "b3ea4d0c5c30f5e22313dddede5d3dba", "0c0c56a6fc234e24a9e3a9e8faf2dbcc", "-42043.85450"),
        ("furan.mol2", "8366132e88f24330fedbbf24367877f7", "b9aa3fc505879a7a2a9a1789aee922f5", "-19404698740"),
        ("acac.mol2", "5bd7b0f59fd27a0c40cd64ed4c74707d", "686dc4f275c6129ce9ebdad99309cf17", "-2.822252628e+16"),
        ("ADUYUV.mol2", "0aab1e9fce84758186a7c82c36c7f74f", "6d95e7193be8800d6b204d31f9731d26", "2.7503745443e+80"),
    ]
)
def test_Mol2D_graph_hash(resource_path_root, filename, node_hash_ref, edge_hash_ref, graph_det_ref):
    # From mol file
    mol = Mol2D.from_mol2_file(resource_path_root / "inputs" / "io" / filename)

    node_hash = mol.graph_hash()
    edge_hash = mol.graph_hash_edge_attr()
    graph_det = mol.graph_determinant()

    assert node_hash == node_hash_ref
    assert edge_hash == edge_hash_ref
    assert graph_det == graph_det_ref


def test_find_metal(resource_path_root):
    mol = Mol2D.from_mol2_file(resource_path_root / "inputs" / "io" / "ADUYUV.mol2")
    metals = mol.find_metal()

    assert metals == [35]
