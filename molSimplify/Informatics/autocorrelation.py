import numpy as np
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.ligand import (
    ligand_assign_original,
    ligand_breakdown,
    )
from molSimplify.Informatics.lacRACAssemble import (
    atom_only_autocorrelation,
    atom_only_autocorrelation_derivative,
    atom_only_deltametric,
    atom_only_deltametric_derivative,
    autocorrelation,
    autocorrelation_derivative,
    construct_property_vector,
    deltametric,
    deltametric_derivative,
    full_autocorrelation,
    full_autocorrelation_derivative,
    generate_full_complex_autocorrelation_derivatives,
    generate_full_complex_autocorrelations,
    generate_metal_autocorrelation_derivatives,
    generate_metal_autocorrelations,
    generate_metal_deltametric_derivatives,
    generate_metal_deltametrics,
    generate_metal_ox_autocorrelation_derivatives,
    generate_metal_ox_autocorrelations,
    generate_metal_ox_deltametric_derivatives,
    generate_metal_ox_deltametrics,
    get_metal_index,
    metal_only_autocorrelation,
    metal_only_autocorrelation_derivative,
    metal_only_deltametric,
    metal_only_deltametric_derivative,
    )
from molSimplify.Scripts.geometry import distance
from molSimplify.Classes.globalvars import globalvars

# ########## UNIT CONVERSION
HF_to_Kcal_mol = 627.503


def ratiometric(mol, prop_vec_num, prop_vec_den, orig, d, oct=True):
    """This function returns the ratiometrics for one atom.

    Parameters
    ----------
        mol : mol3D class
        prop_vec : vector, property of atoms in mol in order of index
        orig : int, zero-indexed starting atom
        d : int, number of hops to travel
        oct : bool, if complex is octahedral, will use better bond checks

    Returns
    -------
        result_vector : vector of prop_vec_num / prop_vec_den

    """
    result_vector = np.zeros(d + 1)
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    result_vector[hopped] = prop_vec_num[orig] / prop_vec_den[orig]
    while hopped < (d):
        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # Prepare all atoms attached to this connection.
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        for inds in new_active_set:
            result_vector[hopped] += prop_vec_num[orig] / prop_vec_den[inds]
            historical_set.update(active_set)
        active_set = new_active_set
    return (result_vector)


def summetric(mol, prop_vec, orig, d, oct=True):
    """This function returns the summetrics for one atom.

    Parameters
    ----------
        mol : mol3D class
        prop_vec : vector, property of atoms in mol in order of index
        orig : int, zero-indexed starting atom
        d : int, number of hops to travel
        oct : bool, if complex is octahedral, will use better bond checks

    Returns
    -------
        result_vector : vector of prop_vec_num / prop_vec_den

    """
    result_vector = np.zeros(d + 1)
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    result_vector[hopped] = prop_vec[orig] + prop_vec[orig]
    while hopped < (d):
        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # Prepare all atoms attached to this connection.
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        for inds in new_active_set:
            result_vector[hopped] += prop_vec[orig] + prop_vec[inds]
            historical_set.update(active_set)
        active_set = new_active_set
    return (result_vector)


def autocorrelation_catoms(mol, prop_vec, orig, d, oct=True, catoms=None):
    # Calculate the autocorrelation for the orig to certain connecting atoms.
    result_vector = np.zeros(d + 1)
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    result_vector[hopped] = prop_vec[orig] * prop_vec[orig]
    while hopped < (d):
        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # Prepare all atoms attached to this connection.
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            if this_atom == orig and (catoms is not None):
                this_atoms_neighbors = catoms
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        for inds in new_active_set:
            result_vector[hopped] += prop_vec[orig] * prop_vec[inds]
            historical_set.update(active_set)
        active_set = new_active_set
    return (result_vector)


def deltametric_catoms(mol, prop_vec, orig, d, oct=True, catoms=None):
    # Calculate the deltametrics for the orig to certain connecting atoms.
    result_vector = np.zeros(d + 1)
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    result_vector[hopped] = 0.00
    while hopped < (d):
        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # Prepare all atoms attached to this connection.
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            if this_atom == orig and (catoms is not None):
                this_atoms_neighbors = catoms
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        for inds in new_active_set:
            result_vector[hopped] += prop_vec[orig] - prop_vec[inds]
            historical_set.update(active_set)
        active_set = new_active_set
    return (result_vector)


def multimetal_only_autocorrelation(mol, prop, d, oct=True,
                                    func=autocorrelation, modifier=False,
                                    transition_metals_only=True):
    """
    Calculate metal-centered autocorrelation, averaged over all metals.

    Parameters
    ----------
        mol : TODO
            TODO
        prop : TODO
            TODO
        d : TODO
            TODO
        oct : TODO
            TODO
        func : TODO
            TODO
        modifier : TODO
            TODO
        transition_metals_only : TODO
            TODO

    Returns
    -------
        autocorrelation_vector : TODO
            TODO

    """
    autocorrelation_vector = np.zeros(d + 1)
    n_met = len(mol.findMetal())
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
    for metal_ind in mol.findMetal(transition_metals_only=transition_metals_only):
        autocorrelation_vector += func(mol, w, metal_ind, d, oct=oct)
    autocorrelation_vector = np.divide(autocorrelation_vector, n_met)
    return (autocorrelation_vector)


def multiatom_only_autocorrelation(mol, prop, d, oct=True,
                                   func=autocorrelation, modifier=False,
                                   additional_elements=False):
    autocorrelation_vector = np.zeros(d + 1)
    metal_list = mol.findMetal()
    if additional_elements:
        for element in additional_elements:
            metal_list += mol.findAtomsbySymbol(element)
    n_met = len(metal_list)
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
    for metal_ind in metal_list:
        autocorrelation_vector += func(mol, w, metal_ind, d, oct=oct)
    autocorrelation_vector = np.divide(autocorrelation_vector, n_met)
    return (autocorrelation_vector)


def atom_only_ratiometric(mol, prop_num, prop_den, d, atomIdx, oct=True):
    # atomIdx must b either a list of indices
    # or a single index.
    w_num = construct_property_vector(mol, prop_num, oct)
    w_den = construct_property_vector(mol, prop_den, oct)
    autocorrelation_vector = np.zeros(d + 1)
    if hasattr(atomIdx, "__len__"):
        for elements in atomIdx:
            autocorrelation_vector += ratiometric(mol, w_num, w_den, elements, d, oct=oct)
        autocorrelation_vector = np.divide(autocorrelation_vector, len(atomIdx))
    else:
        autocorrelation_vector += ratiometric(mol, w_num, w_den, atomIdx, d, oct=oct)
    return (autocorrelation_vector)


def atom_only_summetric(mol, prop, d, atomIdx, oct=True):
    # atomIdx must b either a list of indices
    # or a single index.
    w = construct_property_vector(mol, prop, oct)
    autocorrelation_vector = np.zeros(d + 1)
    if hasattr(atomIdx, "__len__"):
        for elements in atomIdx:
            autocorrelation_vector += summetric(mol, w, elements, d, oct=oct)
        autocorrelation_vector = np.divide(autocorrelation_vector, len(atomIdx))
    else:
        autocorrelation_vector += summetric(mol, w, atomIdx, d, oct=oct)
    return (autocorrelation_vector)


def multimetal_only_deltametric(mol, prop, d, oct=True,
                                func=deltametric, modifier=False,
                                transition_metals_only=True):
    """
    TODO

    Parameters
    ----------
        mol : TODO
            TODO
        prop : TODO
            TODO
        d : TODO
            TODO
        oct : TODO
            TODO
        func : TODO
            TODO
        modifier : TODO
            TODO
        transition_metals_only : TODO
            TODO

    Returns
    -------
        deltametric_vector : TODO
            TODO

    """
    deltametric_vector = np.zeros(d + 1)
    metal_idxs = mol.findMetal(transition_metals_only=transition_metals_only)
    n_met = len(metal_idxs)

    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
    for metal_ind in metal_idxs:
        deltametric_vector += func(mol, w, metal_ind, d, oct=oct)
    deltametric_vector = np.divide(deltametric_vector, n_met)
    return (deltametric_vector)


def multiatom_only_deltametric(mol, prop, d, oct=True,
                               func=deltametric, modifier=False,
                               additional_elements=False):
    deltametric_vector = np.zeros(d + 1)
    metal_list = mol.findMetal()
    if additional_elements:
        for element in additional_elements:
            metal_list += mol.findAtomsbySymbol(element)
    n_met = len(metal_list)
    w = construct_property_vector(mol, prop, oct=oct, modifier=modifier)
    for metal_ind in mol.findMetal():
        deltametric_vector += func(mol, w, metal_ind, d, oct=oct)
    deltametric_vector = np.divide(deltametric_vector, n_met)
    return (deltametric_vector)


def metal_only_layer_density(mol, prop, d, oct=True):
    try:
        metal_ind = get_metal_index(mol)
        print(('metal_index is: %d' % metal_ind))
        w = construct_property_vector(mol, prop, oct=oct)
        density_vector = layer_density_in_3D(mol, w, metal_ind, d, oct=oct)
    except IndexError:
        print('Error, no metal found in mol object!')
        return False
    return density_vector


def layer_density_in_3D(mol, prop_vec, orig, d, oct=True):
    # # This function returns the density (prop^3/(d+1)^3)
    # # for one atom.
    # Inputs:
    #    mol - mol3D class
    #    prop_vec - vector, property of atoms in mol in order of index
    #    orig -  int, zero-indexed starting atom
    #    d - int, number of hops to travel
    #    oct - bool, if complex is octahedral, will use better bond checks
    result_vector = np.zeros(d + 1)
    hopped = 0
    active_set = set([orig])
    historical_set = set()
    result_vector[hopped] = prop_vec[orig] ** 3 / (hopped + 1) ** 3
    while hopped < (d):
        hopped += 1
        new_active_set = set()
        for this_atom in active_set:
            # Prepare all atoms attached to this connection.
            this_atoms_neighbors = mol.getBondedAtomsSmart(this_atom, oct=oct)
            for bound_atoms in this_atoms_neighbors:
                if (bound_atoms not in historical_set) and (bound_atoms not in active_set):
                    new_active_set.add(bound_atoms)
        for inds in new_active_set:
            result_vector[hopped] += prop_vec[inds] ** 3 / (hopped + 1) ** 3
            historical_set.update(active_set)
        active_set = new_active_set
    return result_vector


def find_ligand_autocorrelations_oct(mol, prop, loud, depth, name=False,
                                     oct=True, custom_ligand_dict=False):
    # # This function takes a
    # # symmetric (axial == axial,
    # # equatorial == equatorial)
    # # octahedral complex
    # # and returns autocorrelations for
    # # the axial and equatorial ligands.
    # # custom_ligand_dict allows the user to skip the breakdown
    # # in cases where 3D geo is not correct/formed
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=oct)
        (ax_ligand_list, eq_ligand_list, ax_natoms_list,
         eq_natoms_list, ax_con_int_list, eq_con_int_list,
         ax_con_list, eq_con_list, built_ligand_list) = ligand_assign_original(
            mol, liglist, ligdents, ligcons, loud, name=False)
    else:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    # Count ligands.
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)
    # Get full ligand AC.
    ax_ligand_ac_full = []
    eq_ligand_ac_full = []
    for i in range(0, n_ax):
        if not list(ax_ligand_ac_full):
            ax_ligand_ac_full = full_autocorrelation(ax_ligand_list[i].mol, prop, depth)
        else:
            ax_ligand_ac_full += full_autocorrelation(ax_ligand_list[i].mol, prop, depth)
    ax_ligand_ac_full = np.divide(ax_ligand_ac_full, n_ax)
    for i in range(0, n_eq):
        if not list(eq_ligand_ac_full):
            eq_ligand_ac_full = full_autocorrelation(eq_ligand_list[i].mol, prop, depth)
        else:
            eq_ligand_ac_full += full_autocorrelation(eq_ligand_list[i].mol, prop, depth)
    eq_ligand_ac_full = np.divide(eq_ligand_ac_full, n_eq)

    # Get partial ligand AC.
    ax_ligand_ac_con = []
    eq_ligand_ac_con = []

    for i in range(0, n_ax):
        if not list(ax_ligand_ac_con):
            ax_ligand_ac_con = atom_only_autocorrelation(ax_ligand_list[i].mol, prop, depth, ax_con_int_list[i])
        else:
            ax_ligand_ac_con += atom_only_autocorrelation(ax_ligand_list[i].mol, prop, depth, ax_con_int_list[i])
    ax_ligand_ac_con = np.divide(ax_ligand_ac_con, n_ax)
    for i in range(0, n_eq):
        if not list(eq_ligand_ac_con):
            eq_ligand_ac_con = atom_only_autocorrelation(eq_ligand_list[i].mol, prop, depth, eq_con_int_list[i])
        else:
            eq_ligand_ac_con += atom_only_autocorrelation(eq_ligand_list[i].mol, prop, depth, eq_con_int_list[i])
    eq_ligand_ac_con = np.divide(eq_ligand_ac_con, n_eq)

    return ax_ligand_ac_full, eq_ligand_ac_full, ax_ligand_ac_con, eq_ligand_ac_con


def find_ligand_autocorrelation_derivatives_oct(mol, prop, loud, depth, name=False,
                                                oct=True, custom_ligand_dict=False):
    # # This function takes a
    # # symmetric (axial == axial,
    # # equatorial == equatorial)
    # # octahedral complex
    # # and returns autocorrelations for
    # # the axial and equatorial ligands.
    # # custom_ligand_dict allows the user to skip the breakdown
    # # in cases where 3D geo is not correct/formed
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=oct)
        (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list,
         eq_con_int_list, ax_con_list, eq_con_list, built_ligand_list) = ligand_assign_original(
            mol, liglist, ligdents, ligcons, loud, name=False)
    else:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    # Count ligands.
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)
    # Get full ligand AC.
    ax_ligand_ac_full_derivative = None
    eq_ligand_eq_full_derivative = None

    # Allocate the full Jacobian matrix.
    ax_full_j = np.zeros([depth + 1, mol.natoms])
    eq_full_j = np.zeros([depth + 1, mol.natoms])
    ax_con_j = np.zeros([depth + 1, mol.natoms])
    eq_con_j = np.zeros([depth + 1, mol.natoms])

    # Full ligand ACs
    for i in range(0, n_ax):  # For each ax ligand
        ax_ligand_ac_full_derivative = full_autocorrelation_derivative(ax_ligand_list[i].mol, prop, depth)
        # Now we need to map back to full positions.
        for ii, row in enumerate(ax_ligand_ac_full_derivative):
            for original_ids in list(ax_ligand_list[i].ext_int_dict.keys()):
                ax_full_j[ii, original_ids] += np.divide(row[ax_ligand_list[i].ext_int_dict[original_ids]], n_ax)

    for i in range(0, n_eq):  # For each eq ligand
        # Now we need to map back to full positions.
        eq_ligand_eq_full_derivative = full_autocorrelation_derivative(eq_ligand_list[i].mol, prop, depth)
        for ii, row in enumerate(eq_ligand_eq_full_derivative):
            for original_ids in list(eq_ligand_list[i].ext_int_dict.keys()):
                eq_full_j[ii, original_ids] += np.divide(row[eq_ligand_list[i].ext_int_dict[original_ids]], n_eq)

    # Ligand connection ACs
    for i in range(0, n_ax):
        ax_ligand_ac_con_derivative = atom_only_autocorrelation_derivative(ax_ligand_list[i].mol, prop, depth,
                                                                           ax_con_int_list[i])
        # Now we need to map back to full positions.
        for ii, row in enumerate(ax_ligand_ac_con_derivative):
            for original_ids in list(ax_ligand_list[i].ext_int_dict.keys()):
                ax_con_j[ii, original_ids] += np.divide(row[ax_ligand_list[i].ext_int_dict[original_ids]], n_ax)

    for i in range(0, n_eq):
        eq_ligand_ac_con_derivative = atom_only_autocorrelation_derivative(eq_ligand_list[i].mol, prop, depth,
                                                                           eq_con_int_list[i])
        # Now we need to map back to full positions.
        for ii, row in enumerate(eq_ligand_ac_con_derivative):
            for original_ids in list(eq_ligand_list[i].ext_int_dict.keys()):
                eq_con_j[ii, original_ids] += np.divide(row[eq_ligand_list[i].ext_int_dict[original_ids]], n_eq)

    return ax_full_j, eq_full_j, ax_con_j, eq_con_j


def find_ligand_autocorrs_and_deltametrics_oct_dimers(mol, prop, depth, name=False,
                                                      oct=True, custom_ligand_dict=False):
    # # This function takes a
    # # symmetric (axial == axial,
    # # equatorial == equatorial)
    # # octahedral complex
    # # and returns autocorrelations for
    # # the axial and equatorial ligands.
    # # custom_ligand_dict allows the user to skip the breakdown
    # # in cases where 3D geo is not correct/formed
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    if not custom_ligand_dict:
        raise ValueError('No custom ligand dict provided!')
    else:
        ax1_ligand_list = custom_ligand_dict["ax1_ligand_list"]
        ax2_ligand_list = custom_ligand_dict["ax2_ligand_list"]
        ax3_ligand_list = custom_ligand_dict["ax3_ligand_list"]
        ax1_con_int_list = custom_ligand_dict["ax1_con_int_list"]
        ax2_con_int_list = custom_ligand_dict["ax2_con_int_list"]
        ax3_con_int_list = custom_ligand_dict["ax3_con_int_list"]
        axligs = [ax1_ligand_list, ax2_ligand_list, ax3_ligand_list]
        axcons = [ax1_con_int_list, ax2_con_int_list, ax3_con_int_list]
        n_axs = [len(i) for i in axligs]

    # Get full ligand AC.
    ax_ligand_ac_fulls = [False, False, False]

    for axnum in range(3):
        ax_ligand_ac_full = list()
        for i in range(0, n_axs[axnum]):
            if not list(ax_ligand_ac_full):
                ax_ligand_ac_full = full_autocorrelation(axligs[axnum][i].mol, prop, depth)
            else:
                ax_ligand_ac_full += full_autocorrelation(axligs[axnum][i].mol, prop, depth)
        ax_ligand_ac_full = np.divide(ax_ligand_ac_full, n_axs[axnum])
        ax_ligand_ac_fulls[axnum] = ax_ligand_ac_full

    # Get partial ligand AC.
    ax_ligand_ac_cons = [False, False, False]

    for axnum in range(3):
        ax_ligand_ac_con = list()
        for i in range(0, n_axs[axnum]):
            if not list(ax_ligand_ac_con):
                ax_ligand_ac_con = atom_only_autocorrelation(axligs[axnum][i].mol, prop, depth, axcons[axnum][i])
            else:
                ax_ligand_ac_con += atom_only_autocorrelation(axligs[axnum][i].mol, prop, depth, axcons[axnum][i])
        ax_ligand_ac_con = np.divide(ax_ligand_ac_con, n_axs[axnum])
        ax_ligand_ac_cons[axnum] = ax_ligand_ac_con

    # Get deltametrics.
    ax_delta_cons = [False, False, False]

    for axnum in range(3):
        ax_delta_con = list()
        for i in range(0, n_axs[axnum]):
            if not list(ax_delta_con):
                ax_delta_con = atom_only_deltametric(axligs[axnum][i].mol, prop, depth, axcons[axnum][i])
            else:
                ax_delta_con += atom_only_deltametric(axligs[axnum][i].mol, prop, depth, axcons[axnum][i])
        ax_delta_con = np.divide(ax_delta_con, n_axs[axnum])
        ax_delta_cons[axnum] = ax_delta_con

    return ax_ligand_ac_fulls + ax_ligand_ac_cons + ax_delta_cons


def find_ligand_deltametrics_oct(mol, prop, loud, depth, name=False, oct=True, custom_ligand_dict=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    # # This function takes a
    # # octahedral complex
    # # and returns deltametrics for
    # # the axial and equatorial ligands.
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=oct)
        (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list,
         eq_con_int_list, ax_con_list, eq_con_list, built_ligand_list) = ligand_assign_original(
            mol, liglist, ligdents, ligcons, loud, name=False)
    else:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]
    # Count ligands.
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)

    # Get partial ligand AC.
    ax_ligand_ac_con = []
    eq_ligand_ac_con = []

    for i in range(0, n_ax):
        if not list(ax_ligand_ac_con):
            ax_ligand_ac_con = atom_only_deltametric(ax_ligand_list[i].mol, prop, depth, ax_con_int_list[i])
        else:
            ax_ligand_ac_con += atom_only_deltametric(ax_ligand_list[i].mol, prop, depth, ax_con_int_list[i])
    ax_ligand_ac_con = np.divide(ax_ligand_ac_con, n_ax)
    for i in range(0, n_eq):
        if not list(eq_ligand_ac_con):
            eq_ligand_ac_con = atom_only_deltametric(eq_ligand_list[i].mol, prop, depth, eq_con_int_list[i])
        else:
            eq_ligand_ac_con += atom_only_deltametric(eq_ligand_list[i].mol, prop, depth, eq_con_int_list[i])
    eq_ligand_ac_con = np.divide(eq_ligand_ac_con, n_eq)

    return ax_ligand_ac_con, eq_ligand_ac_con


def find_ligand_deltametric_derivatives_oct(mol, prop, loud, depth, name=False, oct=True, custom_ligand_dict=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    # # This function takes a
    # # octahedral complex
    # # and returns deltametrics for
    # # the axial and equatorial ligands.
    if not custom_ligand_dict:
        liglist, ligdents, ligcons = ligand_breakdown(mol, BondedOct=oct)
        (ax_ligand_list, eq_ligand_list, ax_natoms_list, eq_natoms_list, ax_con_int_list,
         eq_con_int_list, ax_con_list, eq_con_list, built_ligand_list) = ligand_assign_original(
            mol, liglist, ligdents, ligcons, loud, name=False)
    else:
        ax_ligand_list = custom_ligand_dict["ax_ligand_list"]
        eq_ligand_list = custom_ligand_dict["eq_ligand_list"]
        ax_con_int_list = custom_ligand_dict["ax_con_int_list"]
        eq_con_int_list = custom_ligand_dict["eq_con_int_list"]

    # Count ligands.
    n_ax = len(ax_ligand_list)
    n_eq = len(eq_ligand_list)

    # Allocate the full Jacobian matrix.
    ax_con_j = np.zeros([depth + 1, mol.natoms])
    eq_con_j = np.zeros([depth + 1, mol.natoms])

    for i in range(0, n_ax):
        ax_ligand_ac_con_derivative = atom_only_deltametric_derivative(ax_ligand_list[i].mol, prop, depth,
                                                                       ax_con_int_list[i])
        # Now we need to map back to full positions.
        for ii, row in enumerate(ax_ligand_ac_con_derivative):
            for original_ids in list(ax_ligand_list[i].ext_int_dict.keys()):
                ax_con_j[ii, original_ids] += np.divide(row[ax_ligand_list[i].ext_int_dict[original_ids]], n_ax)

    for i in range(0, n_eq):
        eq_ligand_ac_con_derivative = atom_only_deltametric_derivative(eq_ligand_list[i].mol, prop, depth,
                                                                       eq_con_int_list[i])
        for ii, row in enumerate(eq_ligand_ac_con_derivative):
            for original_ids in list(eq_ligand_list[i].ext_int_dict.keys()):
                eq_con_j[ii, original_ids] += np.divide(row[eq_ligand_list[i].ext_int_dict[original_ids]], n_eq)

    return ax_con_j, eq_con_j


def generate_all_ligand_autocorrelations(mol, loud, depth=4, name=False, flag_name=False,
                                         custom_ligand_dict=False, NumB=False, Gval=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    result_ax_full = list()
    result_eq_full = list()
    result_ax_con = list()
    result_eq_con = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        (ax_ligand_ac_full,
         eq_ligand_ac_full,
         ax_ligand_ac_con,
         eq_ligand_ac_con) = find_ligand_autocorrelations_oct(
             mol,
             properties,
             loud=loud,
             depth=depth,
             name=name,
             oct=True,
             custom_ligand_dict=custom_ligand_dict)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result_ax_full.append(ax_ligand_ac_full)
        result_eq_full.append(eq_ligand_ac_full)
        result_ax_con.append(ax_ligand_ac_con)
        result_eq_con.append(eq_ligand_ac_con)
    if flag_name:
        results_dictionary = {'colnames': colnames,
                              'result_ax_full_ac': result_ax_full,
                              'result_eq_full_ac': result_eq_full,
                              'result_ax_con_ac': result_ax_con,
                              'result_eq_con_ac': result_eq_con}
    else:
        results_dictionary = {'colnames': colnames,
                              'result_ax_full': result_ax_full,
                              'result_eq_full': result_eq_full,
                              'result_ax_con': result_ax_con,
                              'result_eq_con': result_eq_con}
    return results_dictionary


def generate_all_ligand_autocorrelation_derivatives(mol, loud, depth=4, name=False, flag_name=False,
                                                    custom_ligand_dict=False, NumB=False, Gval=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    result_ax_full = None
    result_eq_full = None
    result_ax_con = None
    result_eq_con = None

    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        ax_ligand_ac_full, eq_ligand_ac_full, ax_ligand_ac_con, eq_ligand_ac_con = find_ligand_autocorrelation_derivatives_oct(
            mol,
            properties,
            loud=loud,
            depth=depth,
            name=name,
            oct=True,
            custom_ligand_dict=custom_ligand_dict)
        for i in range(0, depth + 1):
            colnames.append(['d' + labels_strings[ii] + '-' + str(i) + '/d' + labels_strings[ii] + str(j) for j in
                             range(0, mol.natoms)])
        if result_ax_full is None:
            result_ax_full = ax_ligand_ac_full
        else:
            result_ax_full = np.row_stack([result_ax_full, ax_ligand_ac_full])

        if result_eq_full is None:
            result_eq_full = eq_ligand_ac_full
        else:
            result_eq_full = np.row_stack([result_eq_full, eq_ligand_ac_full])

        if result_ax_con is None:
            result_ax_con = ax_ligand_ac_con
        else:
            result_ax_con = np.row_stack([result_ax_con, ax_ligand_ac_con])

        if result_eq_con is None:
            result_eq_con = eq_ligand_ac_con
        else:
            result_eq_con = np.row_stack([result_eq_con, eq_ligand_ac_con])

    if flag_name:
        results_dictionary = {'colnames': colnames, 'result_ax_full_ac': result_ax_full,
                              'result_eq_full_ac': result_eq_full,
                              'result_ax_con_ac': result_ax_con, 'result_eq_con_ac': result_eq_con}
    else:
        results_dictionary = {'colnames': colnames, 'result_ax_full': result_ax_full, 'result_eq_full': result_eq_full,
                              'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}
    return results_dictionary


def generate_all_ligand_autocorrs_and_deltametrics_dimers(mol, loud, depth=4, name=False, flag_name=False,
                                                          custom_ligand_dict=False, NumB=False, Gval=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]
    result_ax1_full = list()
    result_ax2_full = list()
    result_ax3_full = list()
    result_ax1_con = list()
    result_ax2_con = list()
    result_ax3_con = list()
    result_delta_ax1_con = list()
    result_delta_ax2_con = list()
    result_delta_ax3_con = list()

    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        # lig_autocorrs is a list of length 6 (ax{i}_ligand_ac_fulls, ax{i}_ligand_ac_cons).
        lig_autocorrs = find_ligand_autocorrs_and_deltametrics_oct_dimers(mol,
                                                                          properties,
                                                                          loud=loud,
                                                                          depth=depth,
                                                                          name=name,
                                                                          oct=True,
                                                                          custom_ligand_dict=custom_ligand_dict)
        this_colnames = []
        assert all([len(i) > 0 for i in lig_autocorrs]), 'Some ligand autocorrelations are empty! %s' % lig_autocorrs
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result_ax1_full.append(lig_autocorrs[0])
        result_ax2_full.append(lig_autocorrs[1])
        result_ax3_full.append(lig_autocorrs[2])
        result_ax1_con.append(lig_autocorrs[3])
        result_ax2_con.append(lig_autocorrs[4])
        result_ax3_con.append(lig_autocorrs[5])
        result_delta_ax1_con.append(lig_autocorrs[6])
        result_delta_ax2_con.append(lig_autocorrs[7])
        result_delta_ax3_con.append(lig_autocorrs[8])

    results_dictionary = {'colnames': colnames,
                          'result_ax1_full': result_ax1_full,
                          'result_ax2_full': result_ax2_full,
                          'result_ax3_full': result_ax3_full,
                          'result_ax1_con': result_ax1_con,
                          'result_ax2_con': result_ax2_con,
                          'result_ax3_con': result_ax3_con,
                          'result_delta_ax1_con': result_delta_ax1_con,
                          'result_delta_ax2_con': result_delta_ax2_con,
                          'result_delta_ax3_con': result_delta_ax3_con}
    return results_dictionary


def generate_all_ligand_deltametrics(mol, loud, depth=4, name=False, flag_name=False,
                                     custom_ligand_dict=False, NumB=False, Gval=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]

    result_ax_con = list()
    result_eq_con = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        ax_ligand_ac_con, eq_ligand_ac_con = find_ligand_deltametrics_oct(mol, properties, loud, depth, name, oct=True,
                                                                          custom_ligand_dict=custom_ligand_dict)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result_ax_con.append(ax_ligand_ac_con)
        result_eq_con.append(eq_ligand_ac_con)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'result_ax_con_del': result_ax_con,
                              'result_eq_con_del': result_eq_con}
    else:
        results_dictionary = {'colnames': colnames, 'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}
    return results_dictionary


def generate_all_ligand_deltametric_derivatives(mol, loud, depth=4, name=False, flag_name=False,
                                                custom_ligand_dict=False, NumB=False, Gval=False):
    # # custom_ligand_dict.keys() must be eq_ligands_list, ax_ligand_list
    # #                                    ax_con_int_list ,eq_con_int_list
    # # with types: eq/ax_ligand_list list of mol3D
    # #             eq/ax_con_int_list list of list/tuple of int e.g,  [[1,2] [1,2]]

    result_ax_con = None
    result_eq_con = None
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        ax_ligand_ac_con, eq_ligand_ac_con = find_ligand_deltametric_derivatives_oct(mol, properties, loud, depth, name,
                                                                                     oct=True,
                                                                                     custom_ligand_dict=custom_ligand_dict)

        for i in range(0, depth + 1):
            colnames.append(['d' + labels_strings[ii] + '-' + str(i) + '/d' + labels_strings[ii] + str(j) for j in
                             range(0, mol.natoms)])
        if result_ax_con is None:
            result_ax_con = ax_ligand_ac_con
        else:
            result_ax_con = np.row_stack([result_ax_con, ax_ligand_ac_con])
        if result_eq_con is None:
            result_eq_con = eq_ligand_ac_con
        else:
            result_eq_con = np.row_stack([result_eq_con, eq_ligand_ac_con])
    if flag_name:
        results_dictionary = {'colnames': colnames, 'result_ax_con_del': result_ax_con,
                              'result_eq_con_del': result_eq_con}
    else:
        results_dictionary = {'colnames': colnames, 'result_ax_con': result_ax_con, 'result_eq_con': result_eq_con}
    return results_dictionary


def generate_multimetal_autocorrelations(
    mol, depth=4, oct=True, flag_name=False,
    polarizability=False, Gval=False,
    transition_metals_only=True):
    """
    TODO

    Parameters
    ----------
        mol : TODO
            TODO
        depth : TODO
            TODO
        oct : TODO
            TODO
        flag_name : TODO
            TODO
        polarizability : TODO
            TODO
        Gval : TODO
            TODO
        transition_metals_only : TODO
            TODO

    Returns
    -------
        results_dictionary : dict
            TODO

    """
    # oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if polarizability:
        allowed_strings += ['polarizability']
        labels_strings += ['alpha']
    for ii, properties in enumerate(allowed_strings):
        metal_ac = multimetal_only_autocorrelation(mol, properties, depth, oct=oct, transition_metals_only=transition_metals_only)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(metal_ac)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_mc_ac': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_multiatom_autocorrelations(mol, depth=4, oct=True, flag_name=False, additional_elements=False):
    # oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    for ii, properties in enumerate(allowed_strings):
        metal_ac = multiatom_only_autocorrelation(mol, properties, depth, oct=oct, additional_elements=additional_elements)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(metal_ac)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_mc_ac': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_ox_eff_autocorrelations(oxmodifier, mol, depth=4, oct=True, flag_name=False, transition_metals_only=True):
    # # oxmodifier - dict, used to modify prop vector (e.g. for adding
    # #             ONLY used with  ox_nuclear_charge    ox or charge)
    # #              {"Fe":2, "Co": 3} etc, normally only 1 metal...
    #   oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    metal_ox_ac = metal_only_autocorrelation(mol, 'group_number', depth, oct=oct, modifier=oxmodifier, transition_metals_only=transition_metals_only)
    this_colnames = []
    for i in range(0, depth + 1):
        this_colnames.append('Gval' + '-' + str(i))
    colnames.append(this_colnames)
    result.append(metal_ox_ac)
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_metal_ox_eff_deltametrics(oxmodifier, mol, depth=4, oct=True, flag_name=False, transition_metals_only=True):
    # # oxmodifier - dict, used to modify prop vector (e.g. for adding
    # #             ONLY used with  ox_nuclear_charge    ox or charge)
    # #              {"Fe":2, "Co": 3} etc, normally only 1 metal...
    #   oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    metal_ox_ac = metal_only_deltametric(mol, 'group_number', depth, oct=oct, modifier=oxmodifier, transition_metals_only=transition_metals_only)
    this_colnames = []
    for i in range(0, depth + 1):
        this_colnames.append('Gval' + '-' + str(i))
    colnames.append(this_colnames)
    result.append(metal_ox_ac)
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_multimetal_deltametrics(
    mol, depth=4, oct=True, flag_name=False,
    polarizability=False, Gval=False,
    transition_metals_only=True):
    """
    TODO

    Parameters
    ----------
        mol : TODO
            TODO
        depth : TODO
            TODO
        oct : TODO
            TODO
        flag_name : TODO
            TODO
        polarizability : TODO
            TODO
        Gval : TODO
            TODO
        transition_metals_only : TODO
            TODO

    Returns
    -------
        results_dictionary : dict
            TODO

    """
    #   oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if polarizability:
        allowed_strings += ['polarizability']
        labels_strings += ['alpha']
    for ii, properties in enumerate(allowed_strings):
        metal_ac = multimetal_only_deltametric(mol, properties, depth, oct=oct, transition_metals_only=transition_metals_only)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(metal_ac)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_mc_del': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_multiatom_deltametrics(mol, depth=4, oct=True, flag_name=False, additional_elements=False):
    #   oct - bool, if complex is octahedral, will use better bond checks
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    for ii, properties in enumerate(allowed_strings):
        metal_ac = multiatom_only_deltametric(mol, properties, depth, oct=oct, additional_elements=additional_elements)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(metal_ac)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_mc_del': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_full_complex_coulomb_autocorrelations(mol,
                                                   depth=3, oct=True,
                                                   flag_name=False, modifier=False,
                                                   use_dist=False, transition_metals_only=True):
    result = list()
    colnames = []
    allowed_strings = ['ident', 'topology', 'group_number', "num_bonds"]
    labels_strings = ['I', 'T', 'Gval', "NumB"]
    for ii, properties in enumerate(allowed_strings):
        metal_ac = full_autocorrelation(mol, properties, depth,
                                        oct=oct, modifier=modifier,
                                        use_dist=use_dist,
                                        transition_metals_only=transition_metals_only)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(metal_ac)
    if flag_name:
        results_dictionary = {'colnames': colnames, 'results_f_all': result}
    else:
        results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_atomonly_autocorrelations(mol, atomIdx, depth=4, oct=True, NumB=False, Gval=False, polarizability=False):
    """
    TODO

    Parameters
    ----------
        mol : TODO
            TODO
        atomIdx : TODO
            TODO
        depth : TODO
            TODO
        oct : TODO
            TODO
        NumB : TODO
            TODO
        Gval : TODO
            TODO
        polarizability : TODO
            TODO

    Returns
    -------
        results_dictionary : dict
            TODO

    """
    # # This function gets autocorrelations for a molecule starting
    # # in one single atom only.
    # Inputs:
    #       mol - mol3D class
    #       atomIdx - int, index of atom3D class; or list of indices
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    if polarizability:
        allowed_strings += ['polarizability']
        labels_strings += ['alpha']
    for ii, properties in enumerate(allowed_strings):
        atom_only_ac = atom_only_autocorrelation(mol, properties, depth, atomIdx, oct=oct)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(atom_only_ac)
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_atomonly_autocorrelation_derivatives(mol, atomIdx, depth=4, oct=True, NumB=False, Gval=False):
    # # This function gets the d/dx for autocorrelations for a molecule starting
    # # in one single atom only.
    # Inputs:
    #       mol - mol3D class
    #       atomIdx - int, index of atom3D class
    result = None
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        atom_only_ac = atom_only_autocorrelation_derivative(mol, properties, depth, atomIdx, oct=oct)
        for i in range(0, depth + 1):
            colnames.append(['d' + labels_strings[ii] + '-' + str(i) + '/d' + labels_strings[ii] + str(j) for j in
                             range(0, mol.natoms)])
        if result is None:
            result = atom_only_ac
        else:
            result = np.row_stack([result, atom_only_ac])
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_atomonly_deltametrics(mol, atomIdx, depth=4, oct=True, NumB=False, Gval=False, polarizability=False):
    """
    This function gets deltametrics for a molecule starting in one single atom only.

    Parameters
    ----------
        mol : mol3D
            mol3D molecule to analyze.
        atomIdx : int
            index of atom3D class.
        depth : TODO
            TODO
        oct : TODO
            TODO
        NumB : TODO
            TODO
        Gval : TODO
            TODO
        polarizability : TODO
            TODO

    Returns
    -------
        results_dictionary : dict
            TODO

    """
    result = list()
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    if polarizability:
        allowed_strings += ["polarizability"]
        labels_strings += ["alpha"]
    for ii, properties in enumerate(allowed_strings):
        atom_only_ac = atom_only_deltametric(mol, properties, depth, atomIdx, oct=oct)
        this_colnames = []
        for i in range(0, depth + 1):
            this_colnames.append(labels_strings[ii] + '-' + str(i))
        colnames.append(this_colnames)
        result.append(atom_only_ac)
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary


def generate_atomonly_deltametric_derivatives(mol, atomIdx, depth=4, oct=True, NumB=False, Gval=False):
    # # This function gets deltametrics for a molecule starting
    # # in one single atom only.
    # Inputs:
    #       mol - mol3D class
    #       atomIdx - int, index of atom3D class
    result = None
    colnames = []
    allowed_strings = ['electronegativity', 'nuclear_charge', 'ident', 'topology', 'size']
    labels_strings = ['chi', 'Z', 'I', 'T', 'S']
    if Gval:
        allowed_strings += ['group_number']
        labels_strings += ['Gval']
    if NumB:
        allowed_strings += ["num_bonds"]
        labels_strings += ["NumB"]
    for ii, properties in enumerate(allowed_strings):
        atom_only_ac_der = atom_only_deltametric_derivative(mol, properties, depth, atomIdx, oct=oct)
        for i in range(0, depth + 1):
            colnames.append(['d' + labels_strings[ii] + '-' + str(i) + '/d' + labels_strings[ii] + str(j) for j in
                             range(0, mol.natoms)])
        if result is None:
            result = atom_only_ac_der
        else:
            result = np.row_stack([result, atom_only_ac_der])
    results_dictionary = {'colnames': colnames, 'results': result}
    return results_dictionary
