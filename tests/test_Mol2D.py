from molSimplify.Classes.mol2D import Mol2D


def test_Mol2D_water(resource_path_root):
    mol_ref = Mol2D()
    mol_ref.add_nodes_from(
        [(0, {"symbol": "O"}), (1, {"symbol": "H"}), (2, {"symbol": "H"})]
    )
    mol_ref.add_edges_from([(0, 1), (0, 2)])

    # From mol file
    mol = Mol2D.from_mol_file(resource_path_root / "inputs" / "io" / "water.mol")

    assert mol.nodes == mol_ref.nodes
    assert mol.edges == mol_ref.edges

    # From mol2 file
    mol = Mol2D.from_mol2_file(resource_path_root / "inputs" / "io" / "water.mol2")

    assert mol.nodes == mol_ref.nodes
    assert mol.edges == mol_ref.edges

    # From smiles
    mol = Mol2D.from_smiles("O")

    assert mol.nodes == mol_ref.nodes
    assert mol.edges == mol_ref.edges


def test_Mol2D_furan(resource_path_root):
    mol_ref = Mol2D()
    mol_ref.add_nodes_from(
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
    mol_ref.add_edges_from(
        [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 5), (2, 6), (3, 7), (4, 8)]
    )

    # From mol file
    mol = Mol2D.from_mol_file(resource_path_root / "inputs" / "io" / "furan.mol")

    assert mol.nodes == mol_ref.nodes
    assert mol.edges == mol_ref.edges

    # From mol2 file
    mol = Mol2D.from_mol2_file(resource_path_root / "inputs" / "io" / "furan.mol2")

    assert mol.nodes == mol_ref.nodes
    assert mol.edges == mol_ref.edges

    # From smiles
    mol = Mol2D.from_smiles("o1cccc1")

    assert mol.nodes == mol_ref.nodes
    assert mol.edges == mol_ref.edges


def test_Mol2D_acac(resource_path_root):
    mol_ref = Mol2D()
    mol_ref.add_nodes_from(
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
    mol_ref.add_edges_from(
        [(0, 1), (1, 2), (1, 3), (2, 7), (2, 8), (2, 9), (3, 4),
         (3, 13), (4, 5), (4, 6), (6, 10), (6, 11), (6, 12)]
    )

    # From mol file
    mol = Mol2D.from_mol_file(resource_path_root / "inputs" / "io" / "acac.mol")

    assert mol.nodes == mol_ref.nodes
    assert mol.edges == mol_ref.edges

    # From mol2 file
    mol = Mol2D.from_mol2_file(resource_path_root / "inputs" / "io" / "acac.mol2")

    assert mol.nodes == mol_ref.nodes
    assert mol.edges == mol_ref.edges

    # TODO: this does not work as "AddHydrogens()" does not take into account charges
    # From smiles
    # mol = Mol2D.from_smiles("O=C(C)CC(=O)C")

    # assert mol.nodes == mol_ref.nodes
    # assert mol.edges == mol_ref.edges
