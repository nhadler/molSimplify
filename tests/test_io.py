import numpy as np
from molSimplify.Scripts.io import lig_load
from pkg_resources import resource_filename, Requirement


def test_lig_load():
    lig_file = resource_filename(
        Requirement.parse('molSimplify'),
        'tests/inputs/io/acac.mol2')
    mol, emsg = lig_load(lig_file)
    # Assert that the error message is empty
    assert not emsg
    # Convert to mol3D after loading the OBmol
    mol.convert2mol3D()
    # Load the reference from the ligand folder
    ref, _ = lig_load('acac')

    assert mol.natoms == ref.natoms
    assert all(mol.symvect() == ref.symvect())
    print(mol.coordsvect())
    print(ref.coordsvect())
    np.testing.assert_allclose(mol.coordsvect(), ref.coordsvect())
    assert mol.charge == ref.charge
