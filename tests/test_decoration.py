from molSimplify.Classes.mol3D import mol3D
from molSimplify.Informatics.decoration_manager import decorate_molecule
import numpy as np


def test_molecule_dec(tmp_path, resource_path_root):
	my_mol = mol3D()
	my_mol.readfromxyz(str(resource_path_root / "inputs" / "xyz_files" / "benzene.xyz"))
	decorated_mol = decorate_molecule(my_mol, ['Cl', 'ammonia'], [8, 9])
	# decorated_mol.writexyz(str(tmp_path / 'mod_benzene.xyz'))

	comparison_mol = mol3D()
	comparison_mol.readfromxyz(str(resource_path_root / "refs" / "decorated_xyz" / "mod_benzene.xyz"))

	d_atoms = decorated_mol.getAtoms()
	c_atoms = comparison_mol.getAtoms()

	# Compare atoms one by one.
	for d_atom, c_atom in zip(d_atoms, c_atoms):
		assert np.allclose(d_atom.coords(), c_atom.coords(), atol=1e-6)
		assert d_atom.symbol() == c_atom.symbol()
