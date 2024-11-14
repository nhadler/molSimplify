# @file nx_helpers.py
#  Input/output functions
#
#  Written by Roland St. Michel for HJK Group
#
#  Dpt of Chemical Engineering, MIT
import networkx as nx
from molSimplify.Classes.mol3D import mol3D

def mol3D_to_networkx(mol,get_symbols:bool=True,get_bond_order:bool=True,get_bond_distance:bool=False):
      """mol3D networkx graph object conversion
      Parameters
      ----------
      mol : mol3D object
      get_symbols : boolean
          Toggles whether symbols are inputted as node attributes in the networkx graph
      get_bond_order  : boolean
          Toggles whether bond orders are inputted as edge attributes in the networkx graph
      get_bond_distance  : boolean
          Toggles whether bond distances are inputted as edge attributes in the networkx graph
      Returns
      -------
      g : networkx graph
          Filled with information mined from the mol3D.
      """
      g = nx.Graph()

      # get every index of atoms in mol3D object
      for atom_ind in range(0,mol.natoms):
            # set each atom as a node in the graph, and add symbol information if wanted
            data={}
            if get_symbols:
                  data['symbol']=mol.getAtom(atom_ind).symbol()
            g.add_node(atom_ind,**data)
      # get every bond in mol3D object
      bond_info=mol.bo_dict
      for bond in bond_info:
            # set each bond as an edge in the graph, and add symbol information if wanted
            data={}
            if get_bond_order:
                  data['bond_order']=bond_info[bond]
            if get_bond_distance:
                  distance=mol.getAtom(bond[0]).distance(mol.getAtom(bond[1]))
                  data['bond_distance']=distance
            g.add_edge(bond[0],bond[1],**data)
      return g
