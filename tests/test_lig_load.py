import helperFuncs as hp
from molSimplify.Scripts.io import lig_load
import pytest


@pytest.mark.parametrize(
    "lig_name",
    [
		"pentacyanopentadienide", # Ligand in ligands.dict.
		"CC(C)O", 				  # SMILES for isopropyl alcohol.
    ])
def test_ligand_loading(tmp_path, resource_path_root, lig_name):
	my_mol, _ = lig_load(lig_name)
	lig_path = str(tmp_path / f"{lig_name}.xyz")
	my_mol.writexyz(lig_path)

	comparison_path = str(resource_path_root / "refs" / "xyz_files" / f"{lig_name}.xyz")

	passNumAtoms = hp.compareNumAtoms(lig_path, comparison_path)
	assert passNumAtoms

	threshold = 0.0001
	fuzzyEqual = hp.fuzzy_compare_xyz(lig_path, comparison_path, threshold)
	assert fuzzyEqual
