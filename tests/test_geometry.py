import numpy as np
from molSimplify.Scripts.geometry import (norm,
                                          normalize,
                                          checkplanar,
                                          dihedral,
                                          reflect_through_plane,
                                          )
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.atom3D import atom3D


def test_norm():
    v = [1.2, -.2, 0.8]

    assert abs(norm(v) - np.linalg.norm(v)) < 1e-6


def test_normalize():
    v = [1.8, 0.6, -1.8]
    v_norm = normalize(v)

    np.testing.assert_allclose(v_norm, np.array(v)/np.linalg.norm(v), atol=1e-6)


def test_checkplanar():
    a1 = [0.0, 0.0, 0.0]
    a2 = [1.2, 0.6, 1.6]
    a3 = [-1.1, 0.3, 0.8]
    a4 = [0.4, -1.2, -0.3]

    assert not checkplanar(a1, a2, a3, a4)
    # Construct a set four point in plane with the first 3
    a4 = [0.1, 0.9, 2.4]
    assert checkplanar(a1, a2, a3, a4)


def test_dihedral():
    mol = mol3D()
    mol.addAtom(atom3D(Sym='X', xyz=[0.5, 0.0, 1.2]))
    mol.addAtom(atom3D(Sym='X', xyz=[0.0, 0.0, 0.0]))
    mol.addAtom(atom3D(Sym='X', xyz=[0.0, 0.0, 1.0]))
    mol.addAtom(atom3D(Sym='X', xyz=[0.5, 0.5, -0.2]))

    d = dihedral(mol, 0, 1, 2, 3)
    assert abs(d - 45.0) < 1e-6


def test_reflect_through_plane():
    mol = mol3D()
    symbols = ['O','H','H']
    coords = [
    [-3.73751, 1.29708, 0.00050],
    [-2.74818, 1.33993, -0.00035],
    [-4.02687, 2.24392, -0.01831],
    ]
    for sym, coord in zip(symbols, coords):
        mol.addAtom(atom3D(Sym=sym, xyz=coord))

    rmol = reflect_through_plane(mol, [0,0,1], [0,0,-1])
    ref_coords = np.array([
        [-3.73751, 1.29708, -2.0005],
        [-2.74818, 1.33993, -1.99965],
        [-4.02687, 2.24392, -1.98169],
        ])

    coords = rmol.get_coordinate_array()
    assert np.allclose(coords, ref_coords, atol=1e-5)
