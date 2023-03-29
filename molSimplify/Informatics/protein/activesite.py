from molSimplify.Classes.protein3D import *
from molSimplify.Classes.mol3D import *
from math import *
import pickle
from scipy.spatial import ConvexHull
import os
import urllib.request as urllib
import requests
from bs4 import BeautifulSoup
import pandas as pd

# filename.txt = txt file with a list of PDB codes
# chosen_confs = folder with pickle files of protein3D objects
# the string "metal" should be replaced with the periodic table symbol of the desired metal
# pdbs = folder of pdb files; can be replaced with a urllib query
# activesites = folder of metal active sites

with open("filename.txt", 'r') as f:
    text = f.read()
    text = text.split('\n')
    for pdbid in text:
        p = pickle.load(open('chosen_confs/' + pdbid + '.pkl', 'rb'))
        for metal in p.findAtom("metal", False):
            pdb = 'pdbs/' + pdbid
            pdbfile=open(pdb+'.pdb','r').readlines()
            activesite = mol3D()
            ids = [] # atom IDs
            metal_aa3ds = p.getBoundMols(metal, True)
            if metal_aa3ds == None:
                continue
            metal_all = p.getBoundMols(metal)
            metal_aas = []
            for aa3d in metal_aa3ds:
                metal_aas.append(aa3d.three_lc)
            coords = []
            f = open('activesites/' + pdbid + "_" + str(metal) + '.pdb', "a")
            f.write("HEADER " + pdbid + "_" + str(metal) + "\n")
            ids.append(metal)
            activesite.addAtom(p.atoms[metal], metal)
            f.write(p.atoms[metal].line)
            coords.append(p.atoms[metal].coords())
            for m in metal_all:
                if type(m) == AA3D:
                    for (a_id, a) in m.atoms:
                        if a.coords() not in coords:
                            ids.append(a_id)
                            activesite.addAtom(a, a_id)
                            f.write(a.line)
                            coords.append(a.coords())
                else:
                    for a in m.atoms:
                        if a.coords() not in coords:
                            ids.append(p.getIndex(a))
                            activesite.addAtom(a, p.getIndex(a))
                            f.write(a.line)
                            coords.append(a.coords())
            for lines in range(0,len(pdbfile)):
                if "CONECT" in pdbfile[lines] and int(pdbfile[lines][6:11]) in ids:
                    f.write(pdbfile[lines])
            f.close()
