import numpy as np
from molSimplify.Scripts.geometry import (norm,
                                          normalize,
                                          checkplanar,
                                          dihedral,
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
