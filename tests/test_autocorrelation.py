import json
import pytest
import numpy as np
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Informatics.autocorrelation import (
    construct_property_vector,
    autocorrelation,
    deltametric,
    full_autocorrelation,
    atom_only_autocorrelation,
    atom_only_deltametric,
    metal_only_autocorrelation,
    metal_only_deltametric,
    generate_atomonly_autocorrelations,
    generate_atomonly_deltametrics,
    generate_metal_autocorrelations,
    generate_metal_deltametrics,
    generate_full_complex_autocorrelations,
    )
from molSimplify.Informatics.lacRACAssemble import get_descriptor_vector

@pytest.fixture(scope="module")
def load_complex1(resource_path_root):
    xyz_file = resource_path_root / "inputs" / "xyz_files" / f"ni_porphyrin_complex.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)
    return mol


@pytest.mark.parametrize(
    "prop",
    [
    'electronegativity',
    'nuclear_charge',
    'ident',
    'topology',
    'size',
    'group_number',
    'polarizability'
    ])
def test_construct_property_vector(resource_path_root, load_complex1, prop):
    w = construct_property_vector(load_complex1, prop)
    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "construct_property_vector" / f"{prop}.json"
    with open(reference_path, 'r') as f:
        ref_w = json.load(f)

    # For saving np arrays to json, need to cast to list.
    # Convert back for comparison.
    ref_w = np.array(ref_w)
    assert np.array_equal(w, ref_w)


@ pytest.mark.parametrize(
    "orig, d, oct_flag, use_dist, size_normalize",
    [
    (0, 3, True, False, False),
    (0, 2, True, False, False),
    (5, 3, True, False, False),
    (5, 3, False, False, False),
    (5, 3, False, True, False),
    (5, 3, False, True, True),
    ])
def test_autocorrelation(resource_path_root, load_complex1, orig, d, oct_flag, use_dist, size_normalize):
    # Will focus on electronegativity for this test.
    prop = 'electronegativity'

    w = construct_property_vector(load_complex1, prop)
    v = autocorrelation(load_complex1, w, orig, d, oct=oct_flag, use_dist=use_dist, size_normalize=size_normalize)
    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "autocorrelation" / f"{orig}_{d}_{oct_flag}_{use_dist}_{size_normalize}.json"
    with open(reference_path, 'r') as f:
        ref_v = json.load(f)

    # For saving np arrays to json, need to cast to list.
    # Convert back for comparison.
    ref_v = np.array(ref_v)
    assert np.array_equal(v, ref_v)


@ pytest.mark.parametrize(
    "orig, d, oct_flag, use_dist, size_normalize",
    [
    (0, 3, True, False, False),
    (0, 2, True, False, False),
    (5, 3, True, False, False),
    (5, 3, False, False, False),
    (5, 3, False, True, False),
    (5, 3, False, True, True),
    ])
def test_deltametric(resource_path_root, load_complex1, orig, d, oct_flag, use_dist, size_normalize):
    # Will focus on nuclear charge for this test.
    prop = 'nuclear_charge'

    w = construct_property_vector(load_complex1, prop)
    v = deltametric(load_complex1, w, orig, d, oct=oct_flag, use_dist=use_dist, size_normalize=size_normalize)
    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "deltametric" / f"{orig}_{d}_{oct_flag}_{use_dist}_{size_normalize}.json"
    with open(reference_path, 'r') as f:
        ref_v = json.load(f)

    # For saving np arrays to json, need to cast to list.
    # Convert back for comparison.
    ref_v = np.array(ref_v)
    assert np.array_equal(v, ref_v)


def test_full_autocorrelation():
    pass


def test_atom_only_autocorrelation():
    pass


def test_atom_only_deltametric():
    pass


def test_metal_only_autocorrelation():
    pass


def test_metal_only_deltametric():
    pass


def test_generate_atomonly_autocorrelations():
    pass


def test_generate_atomonly_deltametrics():
    pass


def test_generate_metal_autocorrelations():
    pass


def test_generate_metal_deltametrics():
    pass


def test_generate_full_complex_autocorrelations():
    pass


def test_get_descriptor_vector():
    pass
