import numpy as np
import networkx as nx
import operator
from typing import Callable, List, Optional, Tuple
from molSimplify.Classes.mol2D import Mol2D
from molSimplify.Classes.globalvars import amassdict, endict


def racs_property_vector(mol: Mol2D, node: int) -> np.ndarray:
    """Calculates the property vector for a given node (atom) in a molecular graph.

    Parameters
    ----------
    mol : Mol2D
        molecular graph
    node : int
        index of the node

    Returns
    -------
    np.ndarray
        property vector of the node
    """
    output = np.zeros(5)
    symbol = mol.nodes[node]["symbol"]
    Z = amassdict[symbol][1]
    # property (i): nuclear charge Z
    output[0] = Z
    # property (ii): Pauling electronegativity chi
    output[1] = endict[symbol]
    # property (iii): topology T, coordination number
    output[2] = len(list(mol.neighbors(node)))
    # property (iv): identity
    output[3] = 1.0
    # property (v): covalent radius S
    output[4] = amassdict[symbol][2]
    return output


def atom_centered_AC(
    mol: Mol2D,
    starting_node: int,
    depth: int = 3,
    operation: Callable[[np.ndarray, np.ndarray], np.ndarray] = operator.mul,
    property_fun: Callable[[Mol2D, int], np.ndarray] = racs_property_vector,
) -> np.ndarray:
    """Calculates atom-centered autocorrelation vectors for a given node in a molecular graph.

    Parameters
    ----------
    mol : Mol2D
        molecular graph
    starting_node : int
        index of the starting node
    depth : int, optional
        maximum depth of the RACs, by default 3
    operation : Callable[[np.ndarray, np.ndarray], np.ndarray], optional
        correlation operation, e.g. operator.mul or operator.sub for product and difference RACs,
        by default operator.mul
    property_fun : Callable[[Mol2D, int], np.ndarray], optional
        function that calculates the atomic property vector, by default racs_property_vector

    Returns
    -------
    np.ndarray
        atom-centered autocorrelation vector
    """
    # Generate all paths from the starting node to all possible nodes
    lengths = nx.single_source_shortest_path_length(
        mol, source=starting_node, cutoff=depth
    )
    p_i = property_fun(mol, starting_node)
    output = np.zeros((len(p_i), depth + 1))
    for node, d_ij in lengths.items():
        p_j = property_fun(mol, node)
        output[:, d_ij] += operation(p_i, p_j)
    return output


def multi_centered_AC(
    mol: Mol2D,
    depth: int = 3,
    operation: Callable[[np.ndarray, np.ndarray], np.ndarray] = operator.mul,
    property_fun: Callable[[Mol2D, int], np.ndarray] = racs_property_vector
) -> np.ndarray:
    """Calculates the full-scope autocorrelation vector on molecular graph.

    Parameters
    ----------
    mol : Mol2D
        molecular graph
    depth : int, optional
        maximum depth of the RACs, by default 3
    operation : Callable[[np.ndarray, np.ndarray], np.ndarray], optional
        correlation operation, e.g. operator.mul or operator.sub for product and difference RACs,
        by default operator.mul
    property_fun : Callable[[Mol2D, int], np.ndarray], optional
        function that calculates the atomic property vector, by default racs_property_vector

    Returns
    -------
    np.ndarray
        full-scope autocorrelation vector
    """
    n_props = len(property_fun(mol, list(mol.nodes.keys())[0]))
    output = np.zeros((n_props, depth + 1))
    # Generate all pairwise path lengths
    lengths = nx.all_pairs_shortest_path_length(mol, cutoff=depth)
    for node_i, lengths_i in lengths:
        p_i = property_fun(mol, node_i)
        for node_j, d_ij in lengths_i.items():
            p_j = property_fun(mol, node_j)
            output[:, d_ij] += operation(p_i, p_j)
    return output


def octahedral_racs(
    mol: Mol2D,
    depth: int = 3,
    equatorial_connecting_atoms: Optional[List[int]] = None,
    property_fun: Callable[[Mol2D, int], np.ndarray] = racs_property_vector,
) -> np.ndarray:
    # Following J. Phys. Chem. A 2017, 121, 8939 there are 6 start/scope
    # combinations for product ACs and 3 for difference ACs.
    n_props = len(property_fun(mol, list(mol.nodes.keys())[0]))
    output = np.zeros((6 + 3, n_props, depth + 1))

    # start = f, scope = all, product
    output[0] = multi_centered_AC(mol, depth=depth, property_fun=property_fun)
    # start = mc, scope = all, product
    output[1] = atom_centered_AC(mol, 0, depth=depth, property_fun=property_fun)

    # For the other scopes the graph has to be subdivided into individual
    # ligand graphs. Make these changes on a copy of the graph:
    subgraphs = mol.copy()
    # First find all connecting atoms of the first metal:
    metals = mol.find_metal()
    if len(metals) != 1:
        raise ValueError("Currently only supports mononuclear TMCs.")
    metal = metals[0]
    connecting_atoms = sorted(list(subgraphs.neighbors(metal)))
    # Assert that we are removing 6 edges
    if len(connecting_atoms) != 6:
        raise ValueError(
            "First metal in the graph does not have 6 neighbors "
            "as expected for an octahedral complex."
        )
    # Then cut the graph by removing all connections to the metal atom
    subgraphs.remove_edges_from([(metal, c) for c in connecting_atoms])

    if equatorial_connecting_atoms is None:
        # Assume the first 4 connecting atoms belong to the equatorial ligands
        # and the other two are axial.
        equatorial_connecting_atoms = connecting_atoms[:4]
        axial_connecting_atoms = connecting_atoms[4:]
    else:
        axial_connecting_atoms = [
            c for c in connecting_atoms if c not in equatorial_connecting_atoms
        ]
        if len(equatorial_connecting_atoms) != 4 or len(axial_connecting_atoms) != 2:
            raise ValueError(
                "The provided equatorial connecting atoms "
                f"{equatorial_connecting_atoms} are not "
                "consistent with the neighbors of the first "
                f"metal in the graph {connecting_atoms}"
            )

    # Build lists of connecting atom and ligand
    # subgraph tuples by first finding set of nodes for the component that the
    # connecting atom c comes from (using nx.node_conncted_component()) and
    # then constructing a subgraph using this node set.
    axial_ligands = [
        (c, subgraphs.subgraph(nx.node_connected_component(subgraphs, c)))
        for c in axial_connecting_atoms
    ]
    equatorial_ligands = [
        (c, subgraphs.subgraph(nx.node_connected_component(subgraphs, c)))
        for c in equatorial_connecting_atoms
    ]

    # Note that the ligand centered RACs are averaged over the involved
    # ligands.
    # start = lc, scope = ax, product
    output[2] = np.mean(
        [
            atom_centered_AC(g, c, depth=depth, property_fun=property_fun)
            for (c, g) in axial_ligands
        ],
        axis=0,
    )

    # start = lc, scope = eq, product
    output[3] = np.mean(
        [
            atom_centered_AC(g, c, depth=depth, property_fun=property_fun)
            for (c, g) in equatorial_ligands
        ],
        axis=0,
    )

    # start = f, scope = ax, product
    output[4] = np.mean(
        [
            multi_centered_AC(g, depth=depth, property_fun=property_fun)
            for (_, g) in axial_ligands
        ],
        axis=0,
    )
    # start = f, scope = ax, product
    output[5] = np.mean(
        [
            multi_centered_AC(g, depth=depth, property_fun=property_fun)
            for (_, g) in equatorial_ligands
        ],
        axis=0,
    )

    # Finally calculate the difference ACs the same way:
    # start = mc, scope = all, difference
    output[6] = atom_centered_AC(
        mol, 0, depth=depth, operation=operator.sub, property_fun=property_fun
    )
    # start = lc, scope = ax, difference
    output[7] = np.mean(
        [
            atom_centered_AC(
                g, c, depth=depth, operation=operator.sub, property_fun=property_fun
            )
            for (c, g) in axial_ligands
        ],
        axis=0,
    )

    # start = lc, scope = eq, difference
    output[8] = np.mean(
        [
            atom_centered_AC(
                g, c, depth=depth, operation=operator.sub, property_fun=property_fun
            )
            for (c, g) in equatorial_ligands
        ],
        axis=0,
    )

    return output


def octahedral_racs_names(depth=3, properties=None) -> List[str]:
    if properties is None:
        properties = ["Z", "chi", "T", "I", "S"]

    start_scope = [
        ("f", "all"),
        ("mc", "all"),
        ("lc", "ax"),
        ("lc", "eq"),
        ("f", "ax"),
        ("f", "eq"),
        ("D_mc", "all"),
        ("D_lc", "ax"),
        ("D_lc", "eq"),
    ]
    return [
        f"{start}-{prop}-{d}-{scope}"
        for start, scope in start_scope
        for prop in properties
        for d in range(0, depth + 1)
    ]


def ligand_racs(
    mol: Mol2D,
    depth: int = 3,
    full_scope: bool = True,
    property_fun: Callable[[Mol2D, int], np.ndarray] = racs_property_vector,
) -> np.ndarray:

    # First cut the molecular graph into subgraphs for the metal and the ligands
    subgraphs = mol.copy()
    # First find all connecting atoms of the first metal:
    metals = mol.find_metal()
    if len(metals) != 1:
        raise ValueError("Currently only supports mononuclear TMCs.")
    metal = metals[0]
    connecting_atoms = sorted(list(subgraphs.neighbors(metal)))

    n_ligands = len(connecting_atoms)
    n_props = len(property_fun(mol, list(mol.nodes.keys())[0]))
    n_scopes = 3 if full_scope else 2
    output = np.zeros((n_ligands, n_scopes, n_props, depth + 1))

    # Then cut the graph by removing all connections to the metal atom
    subgraphs.remove_edges_from([(metal, c) for c in connecting_atoms])
    # Build list of tuples for all connecting atoms and corresponding ligand subgraphs
    # TODO: I am sure there is a better way of doing this than looping over all subgraphs
    ligand_graphs: List[Tuple[int, Mol2D]] = []
    for c in connecting_atoms:
        for gi in nx.connected_components(subgraphs):
            if c in gi:
                ligand_graphs.append((c, subgraphs.subgraph(gi)))
                break

    # Actual calculation of the RACs
    for i, (c, g) in enumerate(ligand_graphs):
        # starting with connecting atom centered product RACs
        output[i, 0] = atom_centered_AC(g, c, depth=depth, operation=operator.mul, property_fun=property_fun)
        output[i, 1] = atom_centered_AC(g, c, depth=depth, operation=operator.sub, property_fun=property_fun)
        # Add full scope RACs if requested
        if full_scope:
            output[i, 2] = multi_centered_AC(g, depth=depth, operation=operator.mul, property_fun=property_fun)

    return output


def ligand_racs_names(depth: int = 3, properties=None, full_scope: bool = True) -> List[str]:
    if properties is None:
        properties = ["Z", "chi", "T", "I", "S"]

    starts = [
        "lc",
        "D_lc",
    ]
    if full_scope:
        starts += ["f"]
    return [
        f"{start}-{prop}-{d}"
        for start in starts
        for prop in properties
        for d in range(0, depth + 1)
    ]
