import helperFuncs as hp
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Informatics.decoration_manager import decorate_molecule
import numpy as np


def test_molecule_dec(tmp_path, resource_path_root):
	my_mol = mol3D()
	my_mol.readfromxyz(str(resource_path_root / "inputs" / "xyz_files" / "benzene.xyz"))
	decorated_mol = decorate_molecule(my_mol, ['Cl', 'ammonia'], [8, 9])
	decorated_path = str(tmp_path / "decorated_benzene.xyz")
	decorated_mol.writexyz(decorated_path)

	comparison_path = str(resource_path_root / "refs" / "decorated_xyz" / "mod_benzene.xyz")

	passNumAtoms = hp.compareNumAtoms(decorated_path, comparison_path)
	assert passNumAtoms

	threshold = 0.0001
	fuzzyEqual = hp.fuzzy_compare_xyz(decorated_path, comparison_path, threshold)
	assert fuzzyEqual
