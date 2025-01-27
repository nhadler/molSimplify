from molSimplify.Informatics.MOF.PBC_functions import (
    compute_adj_matrix,
    compute_distance_matrix3,
    fractional2cart,
    mkcell,
    readcif,
    solvent_removal,
    )
import numpy as np
import filecmp
import json

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
def test_cif_reading(resource_path_root, tmpdir, name):
    cpar, allatomtypes, fcoords = readcif(str(resource_path_root / "inputs" / "cif_files" / f"{name}.cif"))

    reference_cpar = np.loadtxt(str(resource_path_root / "refs" / "informatics" / "mof" / f"{name}_cpar.txt"))
    reference_allatomtypes = str(resource_path_root / "refs" / "informatics" / "mof" / f"{name}_allatomtypes.json")
    reference_fcoords = np.loadtxt(str(resource_path_root / "refs" / "informatics" / "mof" / f"{name}_fcoords.txt"))

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
def test_pairwise_distance_calc(resource_path_root, tmpdir, name):
    cpar, allatomtypes, fcoords = readcif(str(resource_path_root / "inputs" / "cif_files" / f"{name}.cif"))
    cell_v = mkcell(cpar)
    cart_coords = fractional2cart(fcoords, cell_v)
    distance_mat = compute_distance_matrix3(cell_v, cart_coords)

    reference_mat = np.loadtxt(str(resource_path_root / "refs" / "informatics" / "mof" / f"{name}_distance_mat.txt"))
    assert np.array_equal(distance_mat, reference_mat)


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
def test_adjacency_matrix_calc(resource_path_root, tmpdir, name):
    cpar, allatomtypes, fcoords = readcif(str(resource_path_root / "inputs" / "cif_files" / f"{name}.cif"))
    distance_mat = np.loadtxt(str(resource_path_root / "refs" / "informatics" / "mof" / f"{name}_distance_mat.txt"))

    adj_mat = compute_adj_matrix(distance_mat, allatomtypes)
    adj_mat = adj_mat.todense()

    reference_mat = np.loadtxt(str(resource_path_root / "refs" / "informatics" / "mof" / f"{name}_adj_mat.txt"))
    assert np.array_equal(adj_mat, reference_mat)

@pytest.mark.parametrize(
    "name",
    [
        "Zn_MOF",
        "Co_MOF",
    ])
def test_solvent_removal(resource_path_root, tmpdir, name):
    input_geo = str(resource_path_root / "inputs" / "cif_files" / f"{name}_with_solvent.cif")
    output_path = str(tmpdir / f"{name}.cif")
    solvent_removal(input_geo, output_path)

    # Comparing two CIF files for equality
    reference_cif_path = str(resource_path_root / "refs" / "informatics" / "mof" / f"{name}.cif")
    assert filecmp.cmp(output_path, reference_cif_path)
