import pytest
from molSimplify.Informatics.MOF.PBC_functions import (
    cell_to_cellpar,
    compute_adj_matrix,
    compute_distance_matrix,
    compute_image_flag,
    frac_coord,
    fractional2cart,
    make_supercell,
    mkcell,
    overlap_removal,
    readcif,
    returnXYZandGraph,
    solvent_removal,
    writeXYZandGraph,
    )
import numpy as np
import json

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

# def test_compute_image_flag():
#     assert False

# def test_make_supercell():
#     assert False

# def test_writeXYZandGraph():
#     assert False

# def test_returnXYZandGraph():
#     assert False

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
    cpar, allatomtypes, fcoords = readcif(str(resource_path_root / "inputs" / "cif_files" / f"{name}.cif"))

    reference_cpar = np.loadtxt(str(resource_path_root / "refs" / "informatics" / "mof" / "txt" / f"{name}_cpar.txt"))
    reference_allatomtypes = str(resource_path_root / "refs" / "informatics" / "mof" / "json" / f"{name}_allatomtypes.json")
    reference_fcoords = np.loadtxt(str(resource_path_root / "refs" / "informatics" / "mof" / "txt" / f"{name}_fcoords.txt"))

    with open(reference_allatomtypes, 'r') as f:
        reference_allatomtypes = json.load(f)

    assert np.array_equal(cpar, reference_cpar)
    assert allatomtypes == reference_allatomtypes
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
    cpar, allatomtypes, fcoords = readcif(str(resource_path_root / "inputs" / "cif_files" / f"{name}.cif"))
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
    cpar, allatomtypes, fcoords = readcif(str(resource_path_root / "inputs" / "cif_files" / f"{name}.cif"))
    distance_mat = np.loadtxt(str(resource_path_root / "refs" / "informatics" / "mof" / "txt" / f"{name}_distance_mat.txt"))

    adj_mat, _ = compute_adj_matrix(distance_mat, allatomtypes)
    adj_mat = adj_mat.todense()

    reference_mat = np.loadtxt(str(resource_path_root / "refs" / "informatics" / "mof" / "txt" / f"{name}_adj_mat.txt"))
    assert np.array_equal(adj_mat, reference_mat)

@pytest.mark.parametrize(
    "name",
    [
        "Zn_MOF",
        "Co_MOF",
    ])
def test_solvent_removal(resource_path_root, tmp_path, name):
    input_geo = str(resource_path_root / "inputs" / "cif_files" / f"{name}_with_solvent.cif")
    output_path = str(tmp_path / f"{name}.cif")
    solvent_removal(input_geo, output_path)

    # Comparing two CIF files for equality
    reference_cif_path = str(resource_path_root / "refs" / "informatics" / "mof" / "cif" / f"{name}.cif")
    cpar1, allatomtypes1, fcoords1 = readcif(output_path)
    cpar2, allatomtypes2, fcoords2 = readcif(reference_cif_path)

    assert np.array_equal(cpar1, cpar2)
    assert allatomtypes1 == allatomtypes2
    assert np.array_equal(fcoords1, fcoords2)

# def test_overlap_removal():
#     assert False
