import ase.io
from ase import atom
from nequip.ase import NequIPCalculator
from ase.optimize import BFGS,FIRE
import ase.io
from ase.io import read
import numpy as np
import json
from ase.data import covalent_radii, atomic_numbers
from rdkit import Chem
import re
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
def get_fmax_from_traj(traj_file):
    # 读取轨迹文件的最后一个结构（单个Atoms对象）
    atoms = read(traj_file, index=-1)
    # 直接从Atoms对象获取力
    forces = atoms.get_forces()
    force_magnitudes = np.linalg.norm(forces, axis=1)
    fmax = np.max(force_magnitudes)
    return fmax
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
class mlpopt4system():
    def __init__(self,path,folder):
        self.p = f'{path}/species/{folder}'
        self.f = folder
        self.wf=[]
        self.wp=[]
        self.wb=[]
        self.wna=[]
        self.waH=[]
        self.cp = []
    def start(self,random_number):
        sys_path = self.p
        foldername = self.f
        nequipModel=NequIPCalculator.from_deployed_model(model_path='/work/home/ac877eihwp/renyq/sella/LUNIX_all/mlp_opt/prototypeModel.pth',device='cpu')
        w_fmax = []
        w_bond = []
        w_no_ads = []
        w_ads_H = []
        checkpass = []
        for i in range(1,random_number+1):
            print(f'{foldername} struct_{i} start')
            struct=read(f'{sys_path}/{i}/POSCAR')
            struct.set_calculator(nequipModel)
            print(f' Starting optmization by NequIP model:')
            FIRE(struct).run(fmax=0.1,steps=500)
            BFGS(struct, trajectory=f'{sys_path}/{i}/nequipOpt.traj').run(fmax=0.05,steps=1000)
            fmax = get_fmax_from_traj(f'{sys_path}/{i}/nequipOpt.traj')
            if fmax > 0.05:
                w_fmax.append(i)
                print(f'struct_{i} fmax up to limit')
            else:
                CB = checkBonds()
                CB.poscar = struct
                CB.AddAtoms()
                CB.CheckAllBonds()
                BM2S = BuildMol2Smiles(CB)
                BM2S.build()
                setads = set()
                for atom in BM2S.ads:
                    setads.add(atom.elesymbol)
                if BM2S.smiles == foldername:
                    if BM2S.ads !=[]:
                        if setads != {'H'}:
                            checkpass.append(i)
                            print(f'struct_{i} ckeck pass')
                        else:
                            if foldername == '[H]' or foldername =='[H][H]':
                                checkpass.append(i)
                                print(f'struct_{i} ckeck pass')
                            else:
                                w_ads_H.append(i)
                                print(f'struct_{i} adsorp with H')
                    else:
                        w_no_ads.append(i)
                        print(f'struct_{i} no adsorption')
                else:
                    print('BM2S.smiles:',BM2S.smiles)
                    print('foldername:',foldername)
                    w_bond.append(i)
                    print(f'struct_{i} bond(s) broken')          
                '''if BM2S.smiles == foldername and BM2S.ads != [] and setads != {'H'}:
                    checkpass.append(i)
                    print(f'struct_{i} ckeck pass')
                else:
                    if BM2S.smiles != foldername:
                        print('BM2S.smiles:',BM2S.smiles)
                        print('foldername:',foldername)
                        w_bond.append(i)
                        print(f'struct_{i} bond(s) broken')
                    else:
                        if BM2S.ads == []:
                            w_no_ads.append(i)
                            print(f'struct_{i} no adsorption')
                        else:
                            if setads == {'H'}:
                                w_ads_H.append(i)
                                print(f'struct_{i} adsorp with H')'''
            print(f'{foldername} struct_{i} complete')
        self.cp = checkpass
        self.wf = w_fmax
        self.wb = w_bond
        self.wna = w_no_ads
        self.waH = w_ads_H
        return checkpass
'''---------------------------------------'''
with open('config.json','r') as j:
    data = json.load(j)
path = data['path']
record = 'record.json'
folderpath=data['folderpath']
rm = data['random_number']
with open (record,'w') as file:
        pass
for name in folderpath:
    MLP4SYS = mlpopt4system(path,name)
    MLP4SYS.start(random_number=rm)
    data={name:[MLP4SYS.cp,MLP4SYS.wf,MLP4SYS.wb,MLP4SYS.wna,MLP4SYS.waH]}
    with open(record, 'r') as f:
        file = f.read()
        if len(file)>0:
            ne = 'ne'
        else:
            ne = 'e'
    if ne == 'ne':
        with open (record,'r') as f:
            old_data = json.load(f)
    else:
        old_data ={}
    old_data.update(data)
    with open(record, 'w') as f:
        json.dump(old_data,f,indent=2)
'''---------------------------------------'''