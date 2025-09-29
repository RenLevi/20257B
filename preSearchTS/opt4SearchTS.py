from ase.optimize import BFGS, FIRE
from ase.io import read, write
from nequip.ase import NequIPCalculator
from ase.neb import NEB
from ase.constraints import FixAtoms
from nequip.ase import NequIPCalculator
import copy
import json
from rdkit import Chem
from ase import atom
from rdkit import Chem
import numpy as np
import re
import os
from ase.data import covalent_radii, atomic_numbers
'''------------------------------------------------------'''
class bond():
    def __init__(self, ele1,ele2,dis):
        self.ele1 = ele1
        self.ele2 = ele2
        self.length = dis
    def judge_bondorder(self):
        # 获取元素的原子序数
        z_i = atomic_numbers[self.ele1]
        z_j = atomic_numbers[self.ele2]
        r_i = covalent_radii[z_i]
        r_j = covalent_radii[z_j]
        delta = 0.45
        if self.length <= r_i+r_j+delta:
            return 1
        else:
            return 0
'''------------------------------------------------------'''
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
                    if check_NON_metal_atoms(main_atom) == True:
                        self.adsorption.append(main_atom)
                    else:
                        self.adsorption.append(sub_atom)
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
        self.ads = CB.adsorption
'''------------------------------------------------------'''

model_path = '/work/home/ac877eihwp/renyq/prototypeModel.pth'
calc = NequIPCalculator.from_deployed_model(model_path, device='cpu')

def read_data(file_name):
    Atoms = read(file_name)
    Atoms.calc = calc
    FIRE(Atoms).run(fmax=0.1,steps=10000)#
    BFGS(Atoms,maxstep=0.05).run(fmax=0.01,steps=10000)#
    ''''''
    return Atoms
def checkISFS(Atoms,model:str,answerlist):
    Reaction = answerlist[0]
    bond_smi = answerlist[-1]
    #broken_smi = answerlist[4]
    def warp(A,answer):#[Reaction,atoms(),mainBodyIdx,subBodyIdx,bonded smiles,broken smiles]
        atoms = copy.deepcopy(A)
        test = checkBonds()
        test.poscar = Atoms
        test.AddAtoms()
        test.CheckAllBonds()
        bm2s = BuildMol2Smiles(test)
        bm2s.build()
        out_t = [bool(bm2s.smiles == answer[-1]),None,None]#ttt can pass
        for i in range(1,3):
            model = copy.deepcopy(atoms)
            indices_to_remove = answer[i]
            mask = np.ones(len(model), dtype=bool)
            mask[indices_to_remove] = False
            model = model[mask]
            test = checkBonds()
            test.poscar = model
            test.AddAtoms()
            test.CheckAllBonds()
            bm2s = BuildMol2Smiles(test)
            bm2s.build()
            if bm2s.ads != []:
                out_t[i] = True
            else:
                out_t[i] = False
        return out_t
    if 'Add' in Reaction:
        if model == 'FS':
            test = checkBonds()
            test.poscar = Atoms
            test.AddAtoms()
            test.CheckAllBonds()
            bm2s = BuildMol2Smiles(test)
            bm2s.build()
            return [bool(bm2s.smiles == bond_smi),bool(bm2s.ads != [])]#ttt can pass
        elif model == 'IS':
            tl = warp(Atoms,answerlist)
            return tl
        else:
            ValueError('wrong input')
    elif 'Remove' in Reaction:
        if model == 'IS':
            test = checkBonds()
            test.poscar = Atoms
            test.AddAtoms()
            test.CheckAllBonds()
            bm2s = BuildMol2Smiles(test)
            bm2s.build()
            return [bool(bm2s.smiles == bond_smi),bool(bm2s.ads != [])]#ttt can pass
        elif model == 'FS':
            tl = warp(Atoms,answerlist)
            return tl
        else:
            ValueError('wrong input')
    else:
        ValueError('wrong input Reaction')
def json_r_w(name,data):
    with open(name, 'r') as f:
        file = f.read()
        if len(file)>0:
            ne = 'ne'
        else:
            ne = 'e'
    if ne == 'ne':
        with open (name,'r') as f:
            old_data = json.load(f)
    else:
        old_data ={}
    old_data.update(data)
    with open(name, 'w') as f:
        json.dump(old_data,f,indent=2)
'''--------------------------------'''
with open('feedback.json','w') as j:
    pass
with open('config.json','r') as j:
    data = json.load(j)
path = data['path']
folderpath=data['folderpath']
for name in folderpath:
    p0 = f'{path}/{name}'
    with open (f'{path}/foldername.json','r') as j:
        datadict4check = json.load(j)
    answerlist = datadict4check[name]#File name ：[Reaction,bond changed atoms(bid,eid),mainBodyIdx,subBodyIdx,bonded smiles,broken smiles]
    print(answerlist[0])#
    cp_m_l=[]
    tmpl=[]
    for a in range(3):
        tmp_model = read_data(f'{path}/{name}/ISs/{a*5}.vasp')
        tmp_m_tl = checkISFS(tmp_model,'IS',answerlist)
        tmpl.append(tmp_m_tl)
        if len(tmp_m_tl) == 3:
            if all(tmp_m_tl) == True:
                cp_m_l.append(tmp_model)
            else:
                pass
        else:
            ValueError('Error:1')
    if cp_m_l == []:
        json_r_w('feedback.json',{answerlist[0]:tmpl})
    else:
        cp_m_E = []
        for cp_m in cp_m_l:
            cp_m_energy = cp_m.get_potential_energy()
            cp_m_E.append(cp_m_energy)
        Emin = min(cp_m_E)
        IS, FS = cp_m_l[cp_m_E.index(Emin)], read_data(f'{path}/{name}/FS.vasp')
        if not os.path.exists(f'{p0}/IntermediateProcess'):
            os.makedirs(f'{p0}/IntermediateProcess')
            os.makedirs(f'{p0}/IntermediateProcess/step1',exist_ok=True)
            os.makedirs(f'{p0}/IntermediateProcess/step2',exist_ok=True)
            os.makedirs(f'{p0}/IntermediateProcess/step3',exist_ok=True)
            os.makedirs(f'{p0}/IntermediateProcess/results',exist_ok=True)
            os.makedirs(f'{p0}/IntermediateProcess/optimized_IS_FS',exist_ok=True) 
        p1 = f'{p0}/IntermediateProcess/optimized_IS_FS/'
        write(f'{p1}/IS_opt.vasp',IS)
        write(f'{p1}/FS_opt.vasp',FS)

                        

                        



            
            
