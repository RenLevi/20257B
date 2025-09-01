from ase.io import read
from MLP_opt.JmolNN import bond
from ase import atom
from rdkit import Chem
import numpy as np
import re
def check_NON_metal_atoms(atom):
    non_metal_list =[1,2,5,6,7,8,9,10,14,15,16,17,18,33,34,35,36,52,53,54,85,86,117,118]
    if atom.number in non_metal_list:
        return True
    else:
        return False
class N_atom:
    def __init__(self, coord, element,number,index):
        self.xyz = coord
        self.id = index
        self.elesymbol = element
        self.number = number
        self.bonddict = {}
        self.bondtype = {}
        self.charge = 0
class checkBonds():
    def __init__(self):
        self.atoms = []
        self.poscar = atom
        self.ads_metal = []
    def input(self,filename):
        self.poscar = read(filename)
    def AddAtoms(self):
        atoms= self.poscar
        atoms_info = []
        for i, atom in enumerate(atoms):
            atominfo = N_atom(atom.position,atom.symbol,atom.number,i)
            atoms_info.append(atominfo)
        self.atoms = atoms_info

    def CheckPBC(self):
        atoms = self.poscar
        if atoms.pbc.all() == True:
            print('PBC is open')
            return True
        else:
            print('PBC is not open')
            return False
        
    def min_dis(self,atomID1,atomID2):
        distance = self.poscar.get_distance(atomID1,atomID2, mic=True)
        return distance
    def CheckBondwith2Atoms(self,main_atomID,sub_atomID):
        dis = self.min_dis(main_atomID,sub_atomID)
        main_atom  = self.atoms[main_atomID] 
        sub_atom = self.atoms[sub_atomID]
        if check_NON_metal_atoms(main_atom) == True or check_NON_metal_atoms(sub_atom) == True:
            if check_NON_metal_atoms(main_atom) == True and check_NON_metal_atoms(sub_atom) == True:
                if bond(main_atom.elesymbol,sub_atom.elesymbol,dis).judge_bondorder() == 1:
                    print(f'there is a bond with {main_atom.elesymbol}:{main_atomID} and {sub_atom.elesymbol}:{sub_atomID}.')
                    main_atom.bonddict[sub_atom] = sub_atom.number
                    sub_atom.bonddict[main_atom] = main_atom.number
                else:
                    print(f"there isn't a bond with {main_atom.elesymbol}:{main_atomID} and {sub_atom.elesymbol}:{sub_atomID}.")    
            else:
                if bond(main_atom.elesymbol,sub_atom.elesymbol,dis).judge_bondorder() == 1:
                    print(f'there is adsorption with {main_atom.elesymbol}:{main_atomID} and {sub_atom.elesymbol}:{sub_atomID}.')
                    if check_NON_metal_atoms(main_atom) == False:
                        self.ads_metal.append(main_atom)
                    else:
                        self.ads_metal.append(sub_atom)
        else:
            pass
    def CheckAllBonds(self):
        atoms = self.poscar
        for i, atom_i in enumerate(atoms):
            for j, atom_j in enumerate(atoms):
                if j > i:
                    self.CheckBondwith2Atoms(i,j)
                else:
                    pass
        print('finish checking ALL bonds')
    def cal_vector(self):
        a=[]
        m=[]
        if self.ads_metal == []:
            print('NO ads')
        else:
            for Natom in self.atoms:
                if check_NON_metal_atoms(Natom) == True:
                    a.append(Natom)
                else:
                    pass
            for i in self.ads_metal:
                m.append(i)
                



        



if (__name__ == "__main__"):
    CB = checkBonds()
    CB.input('[H]C/1/nequipOpt.traj')
    if CB.CheckPBC() == True:
        CB.AddAtoms()
        CB.CheckAllBonds()
    else:
        pass