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


# Don't want to use anything more than function scope,
# since graph attribute of mol3D class can get set
# by createMolecularGraph when autocorrelation is called.
@pytest.fixture
def load_complex1(resource_path_root):
    # Monometallic TMC.
    xyz_file = resource_path_root / "inputs" / "xyz_files" / "ni_porphyrin_complex.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)
    return mol


@pytest.fixture
def load_complex2(resource_path_root):
    # Multimetal cluster.
    xyz_file = resource_path_root / "inputs" / "xyz_files" / "UiO-66_sbu.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)
    return mol


@pytest.fixture
def load_complex3(resource_path_root):
    # Non TM metal complex.
    xyz_file = resource_path_root / "inputs" / "xyz_files" / "in_complex.xyz"
    mol = mol3D()
    mol.readfromxyz(xyz_file)
    return mol


def get_ref(ref_path, np_array=True):
    with open(ref_path, 'r') as f:
        ref = json.load(f)

    if np_array:
        # For saving np arrays to json, need to cast to list.
        # Convert back for comparison.
        ref = np.array(ref)
    else:
        # In this case, need to change the 'results' entry
        # of a dictionary.
        for idx, val in enumerate(ref['results']):
            ref['results'][idx] = np.array(val)

    return ref


def get_atomIdx_str(atomIdx):
    if type(atomIdx) is int:
        atomIdx_str = str(atomIdx)
    elif type(atomIdx) is list:
        mod_atomIdx = [str(i) for i in atomIdx]
        atomIdx_str = '-'.join(mod_atomIdx)
    else:
        raise ValueError()

    return atomIdx_str


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
    ref_w = get_ref(reference_path)
    assert np.array_equal(w, ref_w)


@pytest.mark.parametrize(
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
    ref_v = get_ref(reference_path)
    assert np.array_equal(v, ref_v)


@pytest.mark.parametrize(
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
    ref_v = get_ref(reference_path)
    assert np.array_equal(v, ref_v)


@pytest.mark.parametrize(
    "prop, d, oct_flag, use_dist, size_normalize",
    [
    ("ident", 3, True, False, False),
    ("topology", 2, True, False, False),
    ("size", 3, True, False, False),
    ("group_number", 3, False, False, False),
    ("polarizability", 3, False, True, False),
    ("nuclear_charge", 3, False, True, True),
    ])
def test_full_autocorrelation(resource_path_root, load_complex1, prop, d, oct_flag, use_dist, size_normalize):
    v = full_autocorrelation(load_complex1, prop, d, oct=oct_flag, use_dist=use_dist, size_normalize=size_normalize)
    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "full_autocorrelation" / f"{prop}_{d}_{oct_flag}_{use_dist}_{size_normalize}.json"
    ref_v = get_ref(reference_path)
    assert np.array_equal(v, ref_v)


@pytest.mark.parametrize(
    "atomIdx, d, oct_flag, use_dist, size_normalize",
    [
    (0, 3, True, False, False),
    (0, 2, True, False, False),
    ([0, 5, 10, 15], 3, True, False, False),
    (5, 3, True, False, False),
    (5, 3, False, False, False),
    (5, 3, False, True, False),
    ([0, 5, 10, 15], 3, False, True, False),
    (5, 3, False, True, True),
    ])
def test_atom_only_autocorrelation(resource_path_root, load_complex1, atomIdx, d, oct_flag, use_dist, size_normalize):
    # Will focus on topology for this test.
    prop = 'topology'

    v = atom_only_autocorrelation(load_complex1, prop, d, atomIdx, oct=oct_flag, use_dist=use_dist, size_normalize=size_normalize)
    atomIdx_str = get_atomIdx_str(atomIdx)

    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "atom_only_autocorrelation" / f"{atomIdx_str}_{d}_{oct_flag}_{use_dist}_{size_normalize}.json"
    ref_v = get_ref(reference_path)
    assert np.array_equal(v, ref_v)


@pytest.mark.parametrize(
    "atomIdx, d, oct_flag, use_dist, size_normalize",
    [
    (0, 3, True, False, False),
    (0, 2, True, False, False),
    ([0, 5, 10, 15], 3, True, False, False),
    (5, 3, True, False, False),
    (5, 3, False, False, False),
    (5, 3, False, True, False),
    ([0, 5, 10, 15], 3, False, True, False),
    (5, 3, False, True, True),
    ])
def test_atom_only_deltametric(resource_path_root, load_complex1, atomIdx, d, oct_flag, use_dist, size_normalize):
    # Will focus on size (covalent radius) for this test.
    prop = 'size'

    v = atom_only_deltametric(load_complex1, prop, d, atomIdx, oct=oct_flag, use_dist=use_dist, size_normalize=size_normalize)
    atomIdx_str = get_atomIdx_str(atomIdx)

    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "atom_only_deltametric" / f"{atomIdx_str}_{d}_{oct_flag}_{use_dist}_{size_normalize}.json"
    ref_v = get_ref(reference_path)
    assert np.array_equal(v, ref_v)


@pytest.mark.parametrize(
    "prop, d, oct_flag, use_dist, size_normalize",
    [
    ("ident", 3, True, False, False),
    ("topology", 2, True, False, False),
    ("size", 3, True, False, False),
    ("group_number", 3, False, False, False),
    ("polarizability", 3, False, True, False),
    ("nuclear_charge", 3, False, True, True),
    ])
def test_metal_only_autocorrelation_1(resource_path_root, load_complex1, prop, d, oct_flag, use_dist, size_normalize):
    v = metal_only_autocorrelation(load_complex1, prop, d, oct=oct_flag, use_dist=use_dist, size_normalize=size_normalize)
    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "metal_only_autocorrelation_1" / f"{prop}_{d}_{oct_flag}_{use_dist}_{size_normalize}.json"
    ref_v = get_ref(reference_path)
    assert np.array_equal(v, ref_v)


@pytest.mark.parametrize(
    "prop, d, oct_flag, use_dist, size_normalize",
    [
    ("ident", 3, True, False, False),
    ("topology", 2, True, False, False),
    ("size", 3, True, False, False),
    ("group_number", 3, False, False, False),
    ("polarizability", 3, False, True, False),
    ("nuclear_charge", 3, False, True, True),
    ])
def test_metal_only_autocorrelation_2(resource_path_root, load_complex2, prop, d, oct_flag, use_dist, size_normalize):
    v = metal_only_autocorrelation(load_complex2, prop, d, oct=oct_flag, use_dist=use_dist, size_normalize=size_normalize)
    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "metal_only_autocorrelation_2" / f"{prop}_{d}_{oct_flag}_{use_dist}_{size_normalize}.json"
    ref_v = get_ref(reference_path)
    assert np.array_equal(v, ref_v)


@pytest.mark.parametrize(
    "prop, d, oct_flag, use_dist, size_normalize",
    [
    ("ident", 3, True, False, False),
    ("topology", 2, True, False, False),
    ("size", 3, True, False, False),
    ("group_number", 3, False, False, False),
    ("polarizability", 3, False, True, False),
    ("nuclear_charge", 3, False, True, True),
    ])
def test_metal_only_autocorrelation_3(resource_path_root, load_complex3, prop, d, oct_flag, use_dist, size_normalize):
    v = metal_only_autocorrelation(load_complex3, prop, d,
        oct=oct_flag, use_dist=use_dist, size_normalize=size_normalize,
        transition_metals_only=False)
    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "metal_only_autocorrelation_3" / f"{prop}_{d}_{oct_flag}_{use_dist}_{size_normalize}.json"
    ref_v = get_ref(reference_path)
    assert np.array_equal(v, ref_v)


def test_metal_only_autocorrelation_4(load_complex3):
    # This should throw an exception,
    # since complex3 has no transition metals.
    with pytest.raises(Exception):
        metal_only_autocorrelation(load_complex3, "ident", 3)


@pytest.mark.parametrize(
    "prop, d, oct_flag, use_dist, size_normalize",
    [
    ("ident", 3, True, False, False),
    ("topology", 2, True, False, False),
    ("size", 3, True, False, False),
    ("group_number", 3, False, False, False),
    ("polarizability", 3, False, True, False),
    ("nuclear_charge", 3, False, True, True),
    ])
def test_metal_only_deltametric_1(resource_path_root, load_complex1, prop, d, oct_flag, use_dist, size_normalize):
    v = metal_only_deltametric(load_complex1, prop, d, oct=oct_flag, use_dist=use_dist, size_normalize=size_normalize)
    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "metal_only_deltametric_1" / f"{prop}_{d}_{oct_flag}_{use_dist}_{size_normalize}.json"
    ref_v = get_ref(reference_path)
    assert np.array_equal(v, ref_v)


@pytest.mark.parametrize(
    "prop, d, oct_flag, use_dist, size_normalize",
    [
    ("ident", 3, True, False, False),
    ("topology", 2, True, False, False),
    ("size", 3, True, False, False),
    ("group_number", 3, False, False, False),
    ("polarizability", 3, False, True, False),
    ("nuclear_charge", 3, False, True, True),
    ])
def test_metal_only_deltametric_2(resource_path_root, load_complex2, prop, d, oct_flag, use_dist, size_normalize):
    v = metal_only_deltametric(load_complex2, prop, d, oct=oct_flag, use_dist=use_dist, size_normalize=size_normalize)
    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "metal_only_deltametric_2" / f"{prop}_{d}_{oct_flag}_{use_dist}_{size_normalize}.json"
    ref_v = get_ref(reference_path)
    assert np.array_equal(v, ref_v)


@pytest.mark.parametrize(
    "prop, d, oct_flag, use_dist, size_normalize",
    [
    ("ident", 3, True, False, False),
    ("topology", 2, True, False, False),
    ("size", 3, True, False, False),
    ("group_number", 3, False, False, False),
    ("polarizability", 3, False, True, False),
    ("nuclear_charge", 3, False, True, True),
    ])
def test_metal_only_deltametric_3(resource_path_root, load_complex3, prop, d, oct_flag, use_dist, size_normalize):
    v = metal_only_deltametric(load_complex3, prop, d,
        oct=oct_flag, use_dist=use_dist, size_normalize=size_normalize,
        transition_metals_only=False)
    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "metal_only_deltametric_3" / f"{prop}_{d}_{oct_flag}_{use_dist}_{size_normalize}.json"
    ref_v = get_ref(reference_path)
    assert np.array_equal(v, ref_v)


def test_metal_only_deltametric_4(load_complex3):
    # This should throw an exception,
    # since complex3 has no transition metals.
    with pytest.raises(Exception):
        metal_only_deltametric(load_complex3, "ident", 3)


@pytest.mark.parametrize(
    "atomIdx, depth, oct_flag, Gval, NumB, polarizability",
    [
    (0, 3, True, False, False, False),
    (0, 2, True, False, False, False),
    ([0, 5, 10, 15], 3, True, False, False, False),
    (5, 3, True, False, False, False),
    (5, 3, False, False, False, False),
    (5, 3, False, True, False, False),
    ([0, 5, 10, 15], 3, False, False, True, False),
    (5, 3, False, True, True, True),
    ])
def test_generate_atomonly_autocorrelations(resource_path_root, load_complex1, atomIdx, depth, oct_flag, Gval, NumB, polarizability):
    d = generate_atomonly_autocorrelations(load_complex1, atomIdx, depth=depth, oct=oct_flag,
        NumB=NumB, Gval=Gval, polarizability=polarizability)
    atomIdx_str = get_atomIdx_str(atomIdx)

    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "generate_atomonly_autocorrelations" / f"{atomIdx_str}_{depth}_{oct_flag}_{Gval}_{NumB}_{polarizability}.json"
    ref_d = get_ref(reference_path, np_array=False)
    assert d.keys() == ref_d.keys()
    assert d['colnames'] == ref_d['colnames']
    assert np.array_equal(d['results'], ref_d['results'])


@pytest.mark.parametrize(
    "atomIdx, depth, oct_flag, Gval, NumB, polarizability",
    [
    (0, 3, True, False, False, False),
    (0, 2, True, False, False, False),
    ([0, 5, 10, 15], 3, True, False, False, False),
    (5, 3, True, False, False, False),
    (5, 3, False, False, False, False),
    (5, 3, False, True, False, False),
    ([0, 5, 10, 15], 3, False, False, True, False),
    (5, 3, False, True, True, True),
    ])
def test_generate_atomonly_deltametrics(resource_path_root, load_complex1, atomIdx, depth, oct_flag, Gval, NumB, polarizability):
    d = generate_atomonly_deltametrics(load_complex1, atomIdx, depth=depth, oct=oct_flag,
        NumB=NumB, Gval=Gval, polarizability=polarizability)
    atomIdx_str = get_atomIdx_str(atomIdx)

    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "generate_atomonly_deltametrics" / f"{atomIdx_str}_{depth}_{oct_flag}_{Gval}_{NumB}_{polarizability}.json"
    ref_d = get_ref(reference_path, np_array=False)
    assert d.keys() == ref_d.keys()
    assert d['colnames'] == ref_d['colnames']
    assert np.array_equal(d['results'], ref_d['results'])


@pytest.mark.parametrize(
    "depth, oct_flag, use_dist, size_normalize, Gval, NumB, polarizability",
    [
    (3, True, False, False, False, False, False),
    (2, True, False, False, False, False, False),
    (3, True, False, False, False, False, False),
    (3, False, False, False, False, False, False),
    (3, False, True, False, False, False, False),
    (3, False, True, True, False, False, False),
    (3, True, False, False, True, False, False),
    (3, True, False, False, False, True, False),
    (3, True, False, False, True, True, True),
    ])
def test_generate_metal_autocorrelations_1(resource_path_root, load_complex1, depth, oct_flag, use_dist, size_normalize, Gval, NumB, polarizability):
    d = generate_metal_autocorrelations(load_complex1, depth=depth, oct=oct_flag,
        use_dist=use_dist, size_normalize=size_normalize,
        Gval=Gval, NumB=NumB, polarizability=polarizability)

    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "generate_metal_autocorrelations_1" / f"{depth}_{oct_flag}_{use_dist}_{size_normalize}_{Gval}_{NumB}_{polarizability}.json"
    ref_d = get_ref(reference_path, np_array=False)
    assert d.keys() == ref_d.keys()
    assert d['colnames'] == ref_d['colnames']
    assert np.array_equal(d['results'], ref_d['results'])


@pytest.mark.parametrize(
    "depth, oct_flag, use_dist, size_normalize, Gval, NumB, polarizability",
    [
    (3, True, False, False, False, False, False),
    (2, True, False, False, False, False, False),
    (3, True, False, False, False, False, False),
    (3, False, False, False, False, False, False),
    (3, False, True, False, False, False, False),
    (3, False, True, True, False, False, False),
    (3, True, False, False, True, False, False),
    (3, True, False, False, False, True, False),
    (3, True, False, False, True, True, True),
    ])
def test_generate_metal_autocorrelations_2(resource_path_root, load_complex2, depth, oct_flag, use_dist, size_normalize, Gval, NumB, polarizability):
    d = generate_metal_autocorrelations(load_complex2, depth=depth, oct=oct_flag,
        use_dist=use_dist, size_normalize=size_normalize,
        Gval=Gval, NumB=NumB, polarizability=polarizability)

    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "generate_metal_autocorrelations_2" / f"{depth}_{oct_flag}_{use_dist}_{size_normalize}_{Gval}_{NumB}_{polarizability}.json"
    ref_d = get_ref(reference_path, np_array=False)
    assert d.keys() == ref_d.keys()
    assert d['colnames'] == ref_d['colnames']
    assert np.array_equal(d['results'], ref_d['results'])


@pytest.mark.parametrize(
    "depth, oct_flag, use_dist, size_normalize, Gval, NumB, polarizability",
    [
    (3, True, False, False, False, False, False),
    (2, True, False, False, False, False, False),
    (3, True, False, False, False, False, False),
    (3, False, False, False, False, False, False),
    (3, False, True, False, False, False, False),
    (3, False, True, True, False, False, False),
    (3, True, False, False, True, False, False),
    (3, True, False, False, False, True, False),
    (3, True, False, False, True, True, True),
    ])
def test_generate_metal_autocorrelations_3(resource_path_root, load_complex3, depth, oct_flag, use_dist, size_normalize, Gval, NumB, polarizability):
    d = generate_metal_autocorrelations(load_complex3, depth=depth, oct=oct_flag,
        use_dist=use_dist, size_normalize=size_normalize,
        Gval=Gval, NumB=NumB, polarizability=polarizability,
        transition_metals_only=False)

    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "generate_metal_autocorrelations_3" / f"{depth}_{oct_flag}_{use_dist}_{size_normalize}_{Gval}_{NumB}_{polarizability}.json"
    ref_d = get_ref(reference_path, np_array=False)
    assert d.keys() == ref_d.keys()
    assert d['colnames'] == ref_d['colnames']
    assert np.array_equal(d['results'], ref_d['results'])


def test_generate_metal_autocorrelations_4(load_complex3):
    # This should throw an exception,
    # since complex3 has no transition metals.
    with pytest.raises(Exception):
        generate_metal_autocorrelations(load_complex3)


@pytest.mark.parametrize(
    "depth, oct_flag, use_dist, size_normalize, Gval, NumB, polarizability",
    [
    (3, True, False, False, False, False, False),
    (2, True, False, False, False, False, False),
    (3, True, False, False, False, False, False),
    (3, False, False, False, False, False, False),
    (3, False, True, False, False, False, False),
    (3, False, True, True, False, False, False),
    (3, True, False, False, True, False, False),
    (3, True, False, False, False, True, False),
    (3, True, False, False, True, True, True),
    ])
def test_generate_metal_deltametrics_1(resource_path_root, load_complex1, depth, oct_flag, use_dist, size_normalize, Gval, NumB, polarizability):
    d = generate_metal_deltametrics(load_complex1, depth=depth, oct=oct_flag,
        use_dist=use_dist, size_normalize=size_normalize,
        Gval=Gval, NumB=NumB, polarizability=polarizability)

    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "generate_metal_deltametrics_1" / f"{depth}_{oct_flag}_{use_dist}_{size_normalize}_{Gval}_{NumB}_{polarizability}.json"
    ref_d = get_ref(reference_path, np_array=False)
    assert d.keys() == ref_d.keys()
    assert d['colnames'] == ref_d['colnames']
    assert np.array_equal(d['results'], ref_d['results'])


@pytest.mark.parametrize(
    "depth, oct_flag, use_dist, size_normalize, Gval, NumB, polarizability",
    [
    (3, True, False, False, False, False, False),
    (2, True, False, False, False, False, False),
    (3, True, False, False, False, False, False),
    (3, False, False, False, False, False, False),
    (3, False, True, False, False, False, False),
    (3, False, True, True, False, False, False),
    (3, True, False, False, True, False, False),
    (3, True, False, False, False, True, False),
    (3, True, False, False, True, True, True),
    ])
def test_generate_metal_deltametrics_2(resource_path_root, load_complex2, depth, oct_flag, use_dist, size_normalize, Gval, NumB, polarizability):
    d = generate_metal_deltametrics(load_complex2, depth=depth, oct=oct_flag,
        use_dist=use_dist, size_normalize=size_normalize,
        Gval=Gval, NumB=NumB, polarizability=polarizability)

    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "generate_metal_deltametrics_2" / f"{depth}_{oct_flag}_{use_dist}_{size_normalize}_{Gval}_{NumB}_{polarizability}.json"
    ref_d = get_ref(reference_path, np_array=False)
    assert d.keys() == ref_d.keys()
    assert d['colnames'] == ref_d['colnames']
    assert np.array_equal(d['results'], ref_d['results'])


@pytest.mark.parametrize(
    "depth, oct_flag, use_dist, size_normalize, Gval, NumB, polarizability",
    [
    (3, True, False, False, False, False, False),
    (2, True, False, False, False, False, False),
    (3, True, False, False, False, False, False),
    (3, False, False, False, False, False, False),
    (3, False, True, False, False, False, False),
    (3, False, True, True, False, False, False),
    (3, True, False, False, True, False, False),
    (3, True, False, False, False, True, False),
    (3, True, False, False, True, True, True),
    ])
def test_generate_metal_deltametrics_3(resource_path_root, load_complex3, depth, oct_flag, use_dist, size_normalize, Gval, NumB, polarizability):
    d = generate_metal_deltametrics(load_complex3, depth=depth, oct=oct_flag,
        use_dist=use_dist, size_normalize=size_normalize,
        Gval=Gval, NumB=NumB, polarizability=polarizability,
        transition_metals_only=False)

    reference_path = resource_path_root / "refs" / "json" / "test_autocorrelation" / "generate_metal_deltametrics_3" / f"{depth}_{oct_flag}_{use_dist}_{size_normalize}_{Gval}_{NumB}_{polarizability}.json"
    ref_d = get_ref(reference_path, np_array=False)
    assert d.keys() == ref_d.keys()
    assert d['colnames'] == ref_d['colnames']
    assert np.array_equal(d['results'], ref_d['results'])


def test_generate_metal_deltametrics_4(load_complex3):
    # This should throw an exception,
    # since complex3 has no transition metals.
    with pytest.raises(Exception):
        generate_metal_deltametrics(load_complex3)


def test_generate_full_complex_autocorrelations():
    pass


def test_get_descriptor_vector():
    pass
