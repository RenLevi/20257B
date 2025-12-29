from ase.io import read
from build_ISFS.JmolNN import bond
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
def subHH(STR):
    result = re.sub(r'\[HH\]', '[H]', STR)
    return result
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
        self.adsorption = []
        self.bondsetlist=[]
        self.adsorptAtom = []
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
    def CheckAllBonds(self):
        neighbors_info_list,neighbors_idx_list = bond(self.poscar).judge_bondorder()
        for i in range(len(neighbors_idx_list)):
            ith_atom = self.atoms[i]
            if check_NON_metal_atoms(ith_atom) == True:
                for j in neighbors_idx_list[i]:
                    jth_atom = self.atoms[j]
                    if check_NON_metal_atoms(jth_atom)==True:
                        #print(f'there is a bond with {ith_atom.elesymbol}:{i} and {jth_atom.elesymbol}:{j}.')
                        ith_atom.bonddict[jth_atom]=jth_atom.number
                        jth_atom.bonddict[ith_atom]=ith_atom.number
                        if (ith_atom.id,jth_atom.id) not in self.bondsetlist and (jth_atom.id,ith_atom.id) not in self.bondsetlist:
                            self.bondsetlist.append((ith_atom.id,jth_atom.id))
                    else:
                        #print(f'there is adsorption with {ith_atom.elesymbol}:{i} and {jth_atom.elesymbol}:{j}.')
                        self.adsorptAtom.append(ith_atom)
                        self.adsorption.append(jth_atom)
            else:pass
class BuildMol2Smiles():
    def __init__(self,CB:checkBonds):
        self.metal = 0
        self.cb = CB
        self.smiles = ''
    def count(self):
        CB = self.cb
        atoms=CB.atoms 
        dount = 0
        for atom in atoms:
            if check_NON_metal_atoms(atom) == False:
                dount += 1
        self.metal = dount
    def build(self):
        self.count()
        CB = self.cb
        atoms=CB.atoms
        mol = Chem.RWMol()
        for atom in atoms:
            if check_NON_metal_atoms(atom) == True:
                mol.AddAtom(Chem.Atom(atom.elesymbol))
        for atom in atoms:
            bondatoms = atom.bonddict
            for bondatom in bondatoms:
                if not mol.GetBondBetweenAtoms(atom.id-self.metal,bondatom.id-self.metal):#poscar顺序格式满足金属-非金属
                    mol.AddBond(atom.id-self.metal,bondatom.id-self.metal,Chem.BondType.SINGLE)
        smiles = Chem.MolToSmiles(mol)
        self.smiles = subHH(smiles)
        self.mol = mol
        self.ads = CB.adsorptAtom

if (__name__ == "__main__"):
    CB = checkBonds()
    CB.input('[H]C/1/nequipOpt.traj')
    if CB.CheckPBC() == True:
        CB.AddAtoms()
        CB.CheckAllBonds()
    else:
        pass
    BM2S = BuildMol2Smiles(CB)
    BM2S.build()
    print(f'OUTPUT:{BM2S.smiles}')
    if BM2S.ads != []:
        print('ads')