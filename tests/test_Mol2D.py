import pytest
from molSimplify.Classes.mol2D import Mol2D


@pytest.fixture
def water_Mol2D():
    mol = Mol2D()
    mol.add_nodes_from(
        [(0, {"symbol": "O"}), (1, {"symbol": "H"}), (2, {"symbol": "H"})]
    )
    mol.add_edges_from([(0, 1), (0, 2)])
    return mol


@pytest.fixture
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


@pytest.fixture
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
         (3, 13), (4, 5), (4, 6), (6, 10), (6, 11), (6, 12)]
    )
    return mol


@pytest.mark.parametrize(
    "name, smiles, mol_ref",
    [
        ("water", "O", "water_Mol2D"),
        ("furan", "o1cccc1", "furan_Mol2D"),
        ("acac", None, "acac_Mol2D"),  # TODO: find the correct SMILES string for acac
    ]
)
def test_Mol2D_constructors(resource_path_root, request, name, smiles, mol_ref):
    # Use pytests request functionality to actually construct the reference:
    mol_ref = request.getfixturevalue(mol_ref)

    # From mol file
    mol = Mol2D.from_mol_file(resource_path_root / "inputs" / "io" / f"{name}.mol")

    assert mol.nodes == mol_ref.nodes
    assert mol.edges == mol_ref.edges

    # From mol2 file
    mol = Mol2D.from_mol2_file(resource_path_root / "inputs" / "io" / f"{name}.mol2")

    assert mol.nodes == mol_ref.nodes
    assert mol.edges == mol_ref.edges

    # From smiles
    if smiles is not None:
        mol = Mol2D.from_smiles(smiles)

        assert mol.nodes == mol_ref.nodes
        assert mol.edges == mol_ref.edges


@pytest.mark.parametrize(
    "filename, node_hash_ref, edge_hash_ref",
    [
        ("water.mol2", "2c96eb4288d63b2a9437ddd4172e67f3", "e5206dd18f7e829cf77fae1a48e7b0b9"),
        ("furan.mol2", "f6090276d7369c0c0a535113ec1d97cf", "a9b6fbc7b5f53613546d5e91a7544ed6"),
        ("acac.mol2", "0dfc6836e022406f1be7b5627451d590", "d4879258a3a6c832b55b2e9d7387f96d"),
        ("ADUYUV.mol2", "e134585068e342d12364850112ec7609", "9bc68a69bfeceabffe878084da72f542"),
    ]
)
def test_Mol2D_graph_hash(resource_path_root, filename, node_hash_ref, edge_hash_ref):
    # From mol file
    mol = Mol2D.from_mol2_file(resource_path_root / "inputs" / "io" / filename)

    node_hash = mol.graph_hash()
    edge_hash = mol.graph_hash_edge_attr()

    assert node_hash == node_hash_ref
    assert edge_hash == edge_hash_ref
