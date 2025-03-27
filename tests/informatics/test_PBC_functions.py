from molSimplify.Informatics.MOF.PBC_functions import (
    cell_to_cellpar,
    compute_adj_matrix,
    compute_distance_matrix,
    compute_image_flag,
    findPaths,
    frac_coord,
    fractional2cart,
    get_closed_subgraph,
    make_supercell,
    mkcell,
    overlap_removal,
    readcif,
    solvent_removal,
    writeXYZandGraph,
    write_cif,
    XYZ_connected,
    )
import filecmp
import json
import networkx as nx
import numpy as np
import pytest
from scipy import sparse

@pytest.mark.parametrize(
    "cpar, reference_cell",
    [
        (np.array([5, 10, 15, 90, 90, 90]),
            np.array([[5,0,0],[0,10,0],[0,0,15]])),
        (np.array([13.7029, 13.7029, 25.8838, 45, 45, 45]),
            np.array([[13.7029,0,0],[9.6894,9.6894,0],[18.3026,7.5812,16.6587]])),
        (np.array([6.3708, 7.6685, 9.1363, 101.055, 91.366, 99.9670]),
            np.array([[6.3708,0,0],[-1.3273,7.5528,0],[-0.2178,-1.8170,8.9511]])),
    ])
def test_mkcell(cpar, reference_cell):
    cell = mkcell(cpar)
    assert np.allclose(cell, reference_cell, atol=1e-4)

@pytest.mark.parametrize(
    "cell, reference_cpar",
    [
        (np.array([[5,0,0],[0,10,0],[0,0,15]]),
            np.array([5, 10, 15, 90, 90, 90])),
        (np.array([[13.7029,0,0],[9.6894,9.6894,0],[18.3026,7.5812,16.6587]]),
            np.array([13.7029, 13.7029, 25.8838, 45, 45, 45])),
        (np.array([[6.3708,0,0],[-1.3273,7.5528,0],[-0.2178,-1.8170,8.9511]]),
            np.array([6.3708, 7.6685, 9.1363, 101.055, 91.366, 99.9670])),
    ])
def test_cell_to_cellpar(cell, reference_cpar):
    cpar = cell_to_cellpar(cell)
    assert np.allclose(cpar, reference_cpar, atol=1e-4)

@pytest.mark.parametrize(
    "fcoords, cell, reference_cart_coord",
    [
        (np.array([[0.5,0.5,0.5],[0.7,-0.2,0.1],[0.8,0.8,1.0]]),
            np.array([[10,0,0],[0,5,0],[0,0,15]]),
            np.array([[5,2.5,7.5],[7,-1,1.5],[8,4,15]])),
        (np.array([[0.3,0.9,0.65],[0.1,0.01,-0.3],[1.1,0.8,0.7]]),
            np.array([[10,0,0],[0,5,10],[7,8,9]]),
            np.array([[7.55,9.7,14.85],[-1.1,-2.35,-2.6],[15.9,9.6,14.3]])),
    ])
def test_fractional2cart(fcoords, cell, reference_cart_coord):
    cart_coord = fractional2cart(fcoords, cell)
    assert np.allclose(cart_coord, reference_cart_coord)

@pytest.mark.parametrize(
    "coord, cell, reference_fcoords",
    [
        (np.array([[5,2.5,7.5],[7,-1,1.5],[8,4,15]]),
            np.array([[10,0,0],[0,5,0],[0,0,15]]),
            np.array([[0.5,0.5,0.5],[0.7,-0.2,0.1],[0.8,0.8,1.0]])),
        (np.array([[7.55,9.7,14.85],[-1.1,-2.35,-2.6],[15.9,9.6,14.3]]),
            np.array([[10,0,0],[0,5,10],[7,8,9]]),
            np.array([[0.3,0.9,0.65],[0.1,0.01,-0.3],[1.1,0.8,0.7]])),
    ])
def test_frac_coord(coord, cell, reference_fcoords):
    fcoords = frac_coord(coord, cell)
    assert np.allclose(fcoords, reference_fcoords)

@pytest.mark.parametrize(
    "cell, fcoord1, fcoord2, reference_shift",
    [
        (np.array([[10,0,0],[0,10,0],[0,0,10]]),
            np.array([0.2,0.2,0.8]),
            np.array([0.1,0.3,0]),
            np.array([0,0,1])),
        (np.array([[10,0,0],[0,10,8],[4,15,10]]),
            np.array([0.5,0.3,0.9]),
            np.array([0.1,0.8,0.1]),
            np.array([0,-1,1])),
        (np.array([[10,0,0],[0,10,8],[4,15,10]]),
            np.array([0.5,0.5,0.5]),
            np.array([0.5,0.5,0.5]),
            np.array([0,0,0])),
    ])
def test_compute_image_flag(cell, fcoord1, fcoord2, reference_shift):
    shift = compute_image_flag(cell, fcoord1, fcoord2)
    assert np.array_equal(shift, reference_shift)

def test_writeXYZandGraph(resource_path_root, tmp_path):
    filename = str(tmp_path / 'test_writeXYZandGraph.xyz')
    atoms = ['Cu', 'O', 'C', 'H', 'H', 'H']
    cell = np.array([[10,0,0],[0,10,0],[0,0,10]])
    fcoords = np.array([
        [1,0,0],
        [0.85,0,0],
        [0.7,0,0],
        [0.65,-0.1,0],
        [0.65,0,0.1],
        [0.65,0.05,-0.1],
        ])
    mol_graph = np.array([
        [0,1,0,0,0,0],
        [1,0,1,0,0,0],
        [0,1,0,1,1,1],
        [0,0,1,0,0,0],
        [0,0,1,0,0,0],
        [0,0,1,0,0,0],
        ])
    writeXYZandGraph(filename, atoms, cell, fcoords, mol_graph)

    reference_xyz_path = str(resource_path_root / "refs" / "informatics" / "mof" / "net" / "test_writeXYZandGraph.xyz")
    reference_net_path = str(resource_path_root / "refs" / "informatics" / "mof" / "net" / "test_writeXYZandGraph.net")

    assert filecmp.cmp(filename, reference_xyz_path)
    assert filecmp.cmp(filename.replace('.xyz','.net'), reference_net_path)

@pytest.mark.parametrize(
    "name",
    [
        "FOKYIP_clean",
        "SETDUS_clean",
        "UXUPEK_clean",
        "NEXXIZ_clean",
        "YICDAR_clean",
        "VONBIK_clean",
    ])
def test_readcif(resource_path_root, name):
    cpar, all_atom_types, fcoords = readcif(str(resource_path_root / "inputs" / "cif_files" / f"{name}.cif"))

    reference_cpar = np.loadtxt(str(resource_path_root / "refs" / "informatics" / "mof" / "txt" / f"{name}_cpar.txt"))
    reference_all_atom_types = str(resource_path_root / "refs" / "informatics" / "mof" / "json" / f"{name}_all_atom_types.json")
    reference_fcoords = np.loadtxt(str(resource_path_root / "refs" / "informatics" / "mof" / "txt" / f"{name}_fcoords.txt"))

    with open(reference_all_atom_types, 'r') as f:
        reference_all_atom_types = json.load(f)

    assert np.array_equal(cpar, reference_cpar)
    assert all_atom_types == reference_all_atom_types
    assert np.array_equal(fcoords, reference_fcoords)

@pytest.mark.parametrize(
    "name",
    [
        "FOKYIP_clean",
        "SETDUS_clean",
        "UXUPEK_clean",
        "NEXXIZ_clean",
        "YICDAR_clean",
        "VONBIK_clean",
    ])
def test_compute_distance_matrix(resource_path_root, name):
    cpar, all_atom_types, fcoords = readcif(str(resource_path_root / "inputs" / "cif_files" / f"{name}.cif"))
    cell_v = mkcell(cpar)
    cart_coords = fractional2cart(fcoords, cell_v)
    distance_mat = compute_distance_matrix(cell_v, cart_coords)

    reference_mat = np.loadtxt(str(resource_path_root / "refs" / "informatics" / "mof" / "txt" / f"{name}_distance_mat.txt"))
    assert np.allclose(distance_mat, reference_mat)

@pytest.mark.parametrize(
    "name",
    [
        "FOKYIP_clean",
        "SETDUS_clean",
        "UXUPEK_clean",
        "NEXXIZ_clean",
        "YICDAR_clean",
        "VONBIK_clean",
    ])
def test_compute_adj_matrix(resource_path_root, name):
    cpar, all_atom_types, fcoords = readcif(str(resource_path_root / "inputs" / "cif_files" / f"{name}.cif"))
    distance_mat = np.loadtxt(str(resource_path_root / "refs" / "informatics" / "mof" / "txt" / f"{name}_distance_mat.txt"))

    adj_mat, _ = compute_adj_matrix(distance_mat, all_atom_types)
    adj_mat = adj_mat.todense()

    reference_mat = np.loadtxt(str(resource_path_root / "refs" / "informatics" / "mof" / "txt" / f"{name}_adj_mat.txt"))
    assert np.array_equal(adj_mat, reference_mat)

@pytest.mark.parametrize(
    "name",
    [
        "Co_MOF",
        "Zn_MOF",
    ])
def test_solvent_removal(resource_path_root, tmp_path, name):
    input_geo = str(resource_path_root / "inputs" / "cif_files" / f"{name}_with_solvent.cif")
    output_path = str(tmp_path / f"{name}.cif")
    solvent_removal(input_geo, output_path)

    # Comparing two CIF files for equality
    reference_cif_path = str(resource_path_root / "refs" / "informatics" / "mof" / "cif" / f"{name}_solvent_removed.cif")
    cpar1, all_atom_types1, fcoords1 = readcif(output_path)
    cpar2, all_atom_types2, fcoords2 = readcif(reference_cif_path)

    assert np.array_equal(cpar1, cpar2)
    assert all_atom_types1 == all_atom_types2
    assert np.array_equal(fcoords1, fcoords2)

@pytest.mark.parametrize(
    "name, case",
    [
        ("Co_MOF", "duplicate"),
        ("Co_MOF", "overlap"),
        ("Zn_MOF", "duplicate"),
        ("Zn_MOF", "overlap"),
    ])
def test_overlap_removal(resource_path_root, tmp_path, name, case):
    # Duplicate atoms can sometimes occur when enforcing P1 symmetry with Vesta.
    # In this case, introduced artificially.
    # Duplicate: Atoms in exact same position.
    # Overlap: Atoms not in exact same position, but too close.
    input_geo = str(resource_path_root / "inputs" / "cif_files" / f"{name}_with_{case}.cif")
    output_path = str(tmp_path / f"{name}.cif")
    overlap_removal(input_geo, output_path)

    # Comparing two CIF files for equality
    reference_cif_path = str(resource_path_root / "refs" / "informatics" / "mof" / "cif" / f"{name}_{case}_fixed.cif")
    cpar1, all_atom_types1, fcoords1 = readcif(output_path)
    cpar2, all_atom_types2, fcoords2 = readcif(reference_cif_path)

    assert np.array_equal(cpar1, cpar2)
    assert all_atom_types1 == all_atom_types2
    assert np.array_equal(fcoords1, fcoords2)

@pytest.mark.parametrize(
    "anchor_idx, path_bf, correct_answer", # bf: between functionalizations
    [
        (9, 2, [[9, 1, 0], [9, 1, 5], [9, 1, 13]]),
        (14, 3, [[14, 13, 1, 0], [14, 13, 1, 5], [14, 13, 1, 9], [14, 13, 15, 16], [14, 13, 15, 33], [14, 18, 17, 16], [14, 18, 17, 19]]),
        (32, 4, [[32, 16, 15, 13, 1], [32, 16, 15, 13, 14], [32, 16, 17, 18, 14], [32, 16, 17, 18, 34], [32, 16, 17, 19, 20], [32, 16, 17, 19, 24], [32, 16, 17, 19, 28]]),
        (0, 5, [[0, 1, 13, 14, 18, 17], [0, 1, 13, 14, 18, 34], [0, 1, 13, 15, 16, 17], [0, 1, 13, 15, 16, 32]]),
    ]
    )
def test_findPaths(resource_path_root, anchor_idx, path_bf, correct_answer):
    # Adjacency matrix for benzene with two tert-butyl groups.
    adj_mat_path = str(resource_path_root / "refs" / "informatics" / "mof" / "json" / "test_findPaths.json")
    with open(adj_mat_path, 'r') as f:
        adj_mat = json.load(f)

    # 1 indicates a bond. 0 indicates no bond.
    rows, cols = np.where(np.array(adj_mat) == 1)
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.Graph()
    G.add_edges_from(edges)
    paths = findPaths(G, anchor_idx, path_bf)
    assert paths == correct_answer

def quick_load(file_list):
    result = []
    for i in file_list:
        with open(i, 'r') as f:
            result.append(json.load(f))
    return result

def test_get_closed_subgraph(resource_path_root):
    # Loading the reference files.
    # These correspond to HKUST-1.
    # rp: reference path
    rp1 = resource_path_root / "refs" / "informatics" / "mof" / "json" /  "test_get_closed_subgraph_linkers.json"
    rp2 = resource_path_root / "refs" / "informatics" / "mof" / "json" /  "test_get_closed_subgraph_remove_list.json"
    rp3 = resource_path_root / "refs" / "informatics" / "mof" / "json" /  "test_get_closed_subgraph_adj_matrix.json"
    rp4 = resource_path_root / "refs" / "informatics" / "mof" / "json" /  "test_get_closed_subgraph_linker_list.json"
    rp5 = resource_path_root / "refs" / "informatics" / "mof" / "json" /  "test_get_closed_subgraph_linker_subgraphlist.json"
    linkers, SBU_list, adj_matrix, ref_linker_list, ref_linker_subgraphlist = quick_load(
        [rp1, rp2, rp3, rp4, rp5])

    # json saves lists, so need to convert.
    linkers, SBU_list = set(linkers), set(SBU_list)
    adj_matrix = sparse.csr_matrix(np.array(adj_matrix))

    linker_list, linker_subgraphlist = get_closed_subgraph(linkers, SBU_list, adj_matrix)
    linker_subgraphlist = [i.todense().tolist() for i in linker_subgraphlist]

    assert linker_list == ref_linker_list
    assert linker_subgraphlist == ref_linker_subgraphlist

def test_XYZ_connected():
    cell = np.array([[20,0,0], [0, 20, 0], [0, 0, 20]])
    # Carboxylate junction, with half of the molecule shifted -20 in x coordinate.
    cart_coords = np.array([
        [-21.20951, 0.71729, 0.01405],
        [-3.76586, 0.45520, 0.03116],
        [-21.32576, 1.90732, -0.23185],
        [-19.97571, 0.19305, 0.13934],
        [-4.76387, -0.23556, 0.15861],
        [-3.91171, 1.76703, -0.24017],
        [-2.40654, -0.18880, 0.18284],
        [-2.35111, -0.63995, 1.19649],
        [-2.32578, -1.00235, -0.56930],
    ])

    adj_mat = np.array([
        [0., 0., 1., 1., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1., 1., 1., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0.],
    ])

    ref_fcoords = np.array([
        [-1.0604755e+00,  3.5864500e-02,  7.0250000e-04],
        [-1.1882930e+00,  2.2760000e-02,  1.5580000e-03],
        [-1.0662880e+00,  9.5366000e-02, -1.1592500e-02],
        [-9.9878550e-01,  9.6525000e-03,  6.9670000e-03],
        [-1.2381935e+00, -1.1778000e-02,  7.9305000e-03],
        [-1.1955855e+00,  8.8351500e-02, -1.2008500e-02],
        [-1.1203270e+00, -9.4400000e-03,  9.1420000e-03],
        [-1.1175555e+00, -3.1997500e-02,  5.9824500e-02],
        [-1.1162890e+00, -5.0117500e-02, -2.8465000e-02],
    ])

    fcoords = XYZ_connected(cell, cart_coords, adj_mat)
    assert np.allclose(fcoords, ref_fcoords)

def test_write_cif(resource_path_root, tmp_path):
    filename = str(tmp_path / 'test_write_cif.cif')
    atoms = ['Cu', 'O', 'C', 'H', 'H', 'H']
    cell = np.array([10, 10, 10, 90, 90, 90])
    fcoords = np.array([
        [1,0,0],
        [0.85,0,0],
        [0.7,0,0],
        [0.65,-0.1,0],
        [0.65,0,0.1],
        [0.65,0.05,-0.1],
        ])
    write_cif(filename, cell, fcoords, atoms)

    with open(filename, 'r') as f:
        contents1 = f.readlines()

    reference_cif_path = str(resource_path_root / "refs" / "informatics" / "mof" / "cif" / "test_write_cif.cif")
    with open(reference_cif_path, 'r') as f:
        contents2 = f.readlines()

    # Remove the _chemical_name_common line.
    contents1.pop(1)
    contents2.pop(1)

    assert contents1 == contents2
