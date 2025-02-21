# Written by JP Janet for HJK Group
# Dpt of Chemical Engineering, MIT

from molSimplify.Classes.mol3D import mol3D
from molSimplify.Scripts.geometry import (checkcolinear,
                                          distance,
                                          norm,
                                          rotate_around_axis,
                                          rotation_params,
                                          vecangle,
                                          vecdiff)
from molSimplify.Scripts.io import getlicores, lig_load


def decorate_molecule(mol: mol3D, decoration, decoration_index,
                    debug: bool = False) -> mol3D:
    """
    This function is useful for functionalization.
    Adding functional groups to a base molecule.

    Parameters
    ----------
        mol : mol3D
            Molecule to add functional groups to.
        decoration : list
            List of SMILES or ligands defined in molSimplify.
        decoration_index : list
            List of ligand atoms to replace.
        debug: bool
            Debugging flag for additional printed information.

    Returns
    -------
        merged_mol: mol3D
            Built molecule with functional groups added.
    """

    # structgen depends on decoration_manager, and decoration_manager depends on structgen.ffopt.
    # Thus, this import needs to be placed here to avoid a circular dependence.
    from molSimplify.Scripts.structgen import ffopt

    mol.bo_dict = False # To avoid errors.

    # Reorder to ensure highest atom index is removed first.
    sort_order = [i[0] for i in sorted(enumerate(decoration_index), key=lambda x:x[1])]
    sort_order = sort_order[::-1]  # Reverse the list.

    decoration_index = [decoration_index[i] for i in sort_order]
    decoration = [decoration[i] for i in sort_order]
    if debug:
        print(f'decoration_index is {decoration_index}')
    licores = getlicores()
    if not isinstance(mol, mol3D):
        mol, emsg = lig_load(mol, licores)
    else:
        mol.convert2OBMol()
        mol.charge = mol.OBMol.GetTotalCharge()
    mol.convert2mol3D()  # Convert to mol3D.

    # Create the new molecule.
    merged_mol = mol3D()
    merged_mol.copymol3D(mol)
    for i, dec in enumerate(decoration):
        print(f'** decoration number {i} attaching {dec} at site {decoration_index[i]} **\n')
        dec, emsg = lig_load(dec, licores)
        dec.convert2mol3D()  # Convert to mol3D.
        if debug:
            print(i)
            print(decoration_index)

            print(merged_mol.getAtom(decoration_index[i]).symbol())
            print(merged_mol.getAtom(decoration_index[i]).coords())
            merged_mol.writexyz('basic.xyz')
        Hs = dec.getHsbyIndex(0)
        if len(Hs) > 0 and (not len(dec.cat)):
            dec.deleteatom(Hs[0])
            dec.charge = dec.charge - 1

        if len(dec.cat) > 0:
            decind = dec.cat[0]
        else:
            decind = 0
        dec.alignmol(dec.getAtom(decind), merged_mol.getAtom(decoration_index[i]))
        r1 = dec.getAtom(decind).coords()
        r2 = dec.centermass()
        rrot = r1
        decb = mol3D()
        decb.copymol3D(dec)
        ####################################
        # Center of mass of local environment (to avoid bad placement of bulky ligands).
        auxmol = mol3D()
        for at in dec.getBondedAtoms(decind):
            auxmol.addAtom(dec.getAtom(at))
        if auxmol.natoms > 0:
            r2 = auxmol.centermass()  # Overwrite global with local centermass.
            ####################################
            # Rotate around axis and get both images.
            theta, u = rotation_params(merged_mol.centermass(), r1, r2)
            dec = rotate_around_axis(dec, rrot, u, theta)
            if debug:
                dec.writexyz('dec_ARA' + str(i) + '.xyz')
            decb = rotate_around_axis(decb, rrot, u, theta-180)
            if debug:
                decb.writexyz('dec_ARB' + str(i) + '.xyz')
            d1 = distance(dec.centermass(), merged_mol.centermass())
            d2 = distance(decb.centermass(), merged_mol.centermass())
            dec = dec if (d2 < d1) else decb  # Pick best rotated mol3D.
        #####################################
        # Check for linear molecule.
        auxm = mol3D()
        for at in dec.getBondedAtoms(decind):
            auxm.addAtom(dec.getAtom(at))
        if auxm.natoms > 1:
            r0 = dec.getAtom(decind).coords()
            r1 = auxm.getAtom(0).coords()
            r2 = auxm.getAtom(1).coords()
            if checkcolinear(r1, r0, r2):
                theta, urot = rotation_params(r1, merged_mol.getAtom(decoration_index[i]).coords(), r2)
                theta = vecangle(vecdiff(r0, merged_mol.getAtom(decoration_index[i]).coords()), urot)
                dec = rotate_around_axis(dec, r0, urot, theta)

        # Get the default distance between atoms in question.
        connection_neighbours = merged_mol.getAtom(merged_mol.getBondedAtomsnotH(decoration_index[i])[0])
        new_atom = dec.getAtom(decind)
        target_distance = connection_neighbours.rad + new_atom.rad
        position_to_place = vecdiff(new_atom.coords(), connection_neighbours.coords())
        old_dist = norm(position_to_place)
        missing = (target_distance - old_dist)/2
        dec.translate([missing*position_to_place[j] for j in [0, 1, 2]])

        r1 = dec.getAtom(decind).coords()
        u = vecdiff(r1, merged_mol.getAtom(decoration_index[i]).coords())
        dtheta = 2
        optmax = -9999
        totiters = 0
        decb = mol3D()
        decb.copymol3D(dec)
        # Check for minimum distance between atoms and center of mass distance.
        while totiters < 180:
            dec = rotate_around_axis(dec, r1, u, dtheta)
            d0 = dec.mindist(merged_mol)      # Try to maximize minimum atoms distance.
            d0cm = dec.distance(merged_mol)   # Try to maximize center of mass distance.
            iteropt = d0cm+d0       # Optimization function.
            if (iteropt > optmax):  # If better conformation, keep it.
                decb = mol3D()
                decb.copymol3D(dec)
                optmax = iteropt
            totiters += 1
        dec = decb
        if debug:
            dec.writexyz(f'dec_aligned {i}.xyz')
            print(f'natoms before delete {merged_mol.natoms}')
            print(f'obmol before delete at {decoration_index[i]} is {merged_mol.OBMol.NumAtoms()}')
        # Store connectivity for deleted H.
        BO_mat = merged_mol.populateBOMatrix()
        row_deleted = BO_mat[decoration_index[i]]
        bonds_to_add = []

        # Find where to put the new bonds ->>> Issue here.
        for j, els in enumerate(row_deleted):
            if els > 0:
                # If there is a bond with an atom number
                # before the deleted atom, all is fine.
                # Else, we subtract one as the row will be be removed.
                if j < decoration_index[i]:
                    bond_partner = j
                else:
                    bond_partner = j - 1
                if len(dec.cat) > 0:
                    bonds_to_add.append((bond_partner, (merged_mol.natoms-1)+dec.cat[0], els))
                else:
                    bonds_to_add.append((bond_partner, merged_mol.natoms-1, els))

        # Perform deletion.
        merged_mol.deleteatom(decoration_index[i])

        merged_mol.convert2OBMol()
        if debug:
            merged_mol.writexyz(f'merged del {i}.xyz')
        # Merge and bond.
        merged_mol.combine(dec, bond_to_add=bonds_to_add)
        merged_mol.convert2OBMol()

        if debug:
            merged_mol.writexyz(f'merged {i}.xyz')
            merged_mol.printxyz()
            print('************')

    merged_mol.convert2OBMol()
    merged_mol, _ = ffopt('MMFF94', merged_mol, [], 0, [], False, [], 100)
    BO_mat = merged_mol.populateBOMatrix()
    if debug:
        merged_mol.writexyz('merged_relaxed.xyz')
        print(BO_mat)
    return merged_mol
