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
from pymatgen.analysis.local_env import CrystalNN,VoronoiNN,JmolNN
from pymatgen.io.ase import AseAtomsAdaptor
'''------------------------------------------------------'''
class bond():
    def __init__(self,atoms):
        self.structure = AseAtomsAdaptor.get_structure(atoms)
    def judge_bondorder(self):
        structure = self.structure
        # 获取元素的原子序数
        jnn = JmolNN()
        neighbors_info_list = []
        neighbors_idx_list = []
        for i in range(0,len(structure)):
            neighbors_info = jnn.get_nn_info(structure, i)
            neighbors_info_list.append(neighbors_info)
            neighbors_idx = []
            for dict_i in neighbors_info:
                neighbors_idx.append(dict_i['site_index'])
            neighbors_idx_list.append(neighbors_idx)
                
        return neighbors_info_list,neighbors_idx_list
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
        

    def CheckAllBonds(self):
        neighbors_info_list,neighbors_idx_list = bond(self.poscar).judge_bondorder()
        for i in range(len(neighbors_idx_list)):
            ith_atom = self.atoms[i]
            if check_NON_metal_atoms(ith_atom) == True:
                for j in neighbors_idx_list[i]:
                    jth_atom = self.atoms[j]
                    if check_NON_metal_atoms(jth_atom)==True:
                        print(f'there is a bond with {ith_atom.elesymbol}:{i} and {jth_atom.elesymbol}:{j}.')
                        ith_atom.bonddict[jth_atom]=jth_atom.number
                        jth_atom.bonddict[ith_atom]=ith_atom.number
                    else:
                        print(f'there is adsorption with {ith_atom.elesymbol}:{i} and {jth_atom.elesymbol}:{j}.')
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
        self.ads = CB.adsorption
'''------------------------------------------------------'''

model_path = '/public/home/ac877eihwp/renyq/prototypeModel.pth'
calc = NequIPCalculator.from_deployed_model(model_path, device='cpu')

def read_data(file_name,model,answerlist,p):
    def warp(A,answer,L):#[Reaction,bondatom,mainBodyIdx,subBodyIdx,bonded smiles,broken smiles]
        atoms = copy.deepcopy(A)
        alltest = checkBonds()
        alltest.poscar = atoms
        alltest.AddAtoms()
        alltest.CheckAllBonds()
        allbm2s = BuildMol2Smiles(alltest)
        allbm2s.build()
        frags = Chem.GetMolFrags(allbm2s.mol)
        out_t = [bool(allbm2s.smiles == answer[L]),None,None]#ttt can pass
        out_smi = [allbm2s.smiles,None,None]
        for i in range(2,4):
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
            out_smi[i-1]=bm2s.smiles
            if bm2s.ads != []:
                out_t[i-1] = True
            else:
                out_t[i-1] = False
        temp=out_smi[-1]
        out_smi[-1]=out_smi[-2]
        out_smi[-2]=temp
        temp=out_t[-1]
        out_t[-1]=out_t[-2]
        out_t[-2]=temp
        return out_t,out_smi,len(frags)
    [Reaction,bondatom,mainBodyIdx,subBodyIdx,bonded_smiles,broken_smiles] = answerlist
    if model == 'FS':
        Atoms = read(file_name)
        Atoms.calc = calc
        opt = BFGS(Atoms,maxstep=0.05,trajectory=f'{p}FSopt.traj').run(fmax=0.01,steps=2000)
        if opt == True :
            print('Optimization converged successfully.')
        else: 
            return ValueError('Optimization did not converge within the maximum number of steps.')
    elif model == 'IS':
        z1,z2,v3=0,0,0
        Atoms = read(file_name)
        bbb,sss,lf = warp(Atoms,answerlist,-1)
        Atoms.calc = calc
        BFGS(Atoms,trajectory='000.traj').run(steps=50)
        bbb_t,sss_t,lf = warp(Atoms,answerlist,-1)
        while bbb_t[0] == False:
            print(f'{z1,z2,v3,sss,sss_t,lf}')#
            c1,c2,c3 = bool(sss[1]==sss_t[1]), bool(sss[2]==sss_t[2]),bool(lf==1)
            if c1 == False:
                z1=1+z1
            if c2 == False:
                z2=1+z2
            if c3 == True:
                v3=1+v3
            Atoms = read(file_name)
            Atoms.calc = calc
            centerMAIN=np.array([0,0,0])
            centerSUB=np.array([0,0,0])
            for mid in mainBodyIdx:
                centerMAIN = centerMAIN+Atoms.positions[mid]
            for sid in subBodyIdx:
                centerSUB=centerSUB+Atoms.positions[sid]
            CM = centerMAIN/len(mainBodyIdx)
            CS = centerSUB/len(subBodyIdx)
            M2S=CM-CS
            M2S_new = M2S*(np.linalg.norm(M2S)+v3)/np.linalg.norm(M2S)
            CS_new = CM-M2S_new
            sv = CS_new-CS
            for mid in mainBodyIdx:
                Atoms.positions[mid] = Atoms.positions[mid]+np.array([0,0,z1])
            for sid in subBodyIdx:
                Atoms.positions[sid] = Atoms.positions[sid]+np.array([0,0,z2])+sv

            BFGS(Atoms,trajectory=f'{z1}{z2}{v3}.traj').run(steps=50)
            bbb_t,sss_t,lf= warp(Atoms,answerlist,-1)
            if z1 >= 5 or z2 >=5 or v3 >= 5:
                return ValueError(f'{Reaction}:Optimization of IS model did not converge owing to bond broken.')
            else:pass
        print(f'{z1,z2,v3,sss[0],sss_t[0]}')#
        opt=BFGS(Atoms,maxstep=0.05,trajectory=f'{p}ISopt.traj').run(fmax=0.01,steps=1000)
        if opt == True :
            print('Optimization converged successfully.')
        else: 
            print('Optimization did not converge within the maximum number of steps.')
    ''''''
    return Atoms
def checkISFS(Atoms,model:str,answerlist):
    Reaction = answerlist[0]
    bond_smi = answerlist[-2]
    broken_smi = answerlist[-1]
    def warp(A,answer,L):#[Reaction,bondatom,mainBodyIdx,subBodyIdx,bonded smiles,broken smiles]
        atoms = copy.deepcopy(A)
        test = checkBonds()
        test.poscar = Atoms
        test.AddAtoms()
        test.CheckAllBonds()
        bm2s = BuildMol2Smiles(test)
        bm2s.build()
        out_t = [bool(bm2s.smiles == answer[L]),None,None]#ttt can pass
        for i in range(2,4):
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
                out_t[i-1] = True
            else:
                out_t[i-1] = False
        temp=out_t[-1]
        out_t[-1]=out_t[-2]
        out_t[-2]=temp
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
            tl = warp(Atoms,answerlist,-1)
            return tl
        else:
            ValueError('wrong input')
    elif 'Remove' in Reaction:
        if model == 'FS':
            test = checkBonds()
            test.poscar = Atoms
            test.AddAtoms()
            test.CheckAllBonds()
            bm2s = BuildMol2Smiles(test)
            bm2s.build()
            return [bool(bm2s.smiles == bond_smi),bool(bm2s.ads != [])]#ttt can pass
        elif model == 'IS':
            tl = warp(Atoms,answerlist,-1)
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
    if os.path.exists(f'{p0}/IntermediateProcess/optimized_IS_FS/IS_opt.vasp'):
        IS = read(f'{p0}/IntermediateProcess/optimized_IS_FS/IS_opt.vasp')
        IS.calc = calc
    else:
        IS = read_data(f'{path}/{name}/IS.vasp','IS',answerlist,f'{path}/{name}/')
    if os.path.exists(f'{p0}/IntermediateProcess/optimized_IS_FS/FS_opt.vasp'):
        FS = read(f'{p0}/IntermediateProcess/optimized_IS_FS/FS_opt.vasp')
        FS.calc = calc
    else:
        FS = read_data(f'{path}/{name}/FS.vasp','FS',answerlist,f'{path}/{name}/')

    '''if os.path.exists(f'{p0}/IntermediateProcess/optimized_IS_FS/IS_opt.vasp') and os.path.exists(f'{p0}/IntermediateProcess/optimized_IS_FS/FS_opt.vasp'):
        IS, FS = read(f'{p0}/IntermediateProcess/optimized_IS_FS/IS_opt.vasp'), read(f'{p0}/IntermediateProcess/optimized_IS_FS/FS_opt.vasp')
        IS.calc = calc
        FS.calc = calc
    else:
        IS, FS = read_data(f'{path}/{name}/IS.vasp','IS',answerlist,f'{path}/{name}/'), read_data(f'{path}/{name}/FS.vasp','FS',answerlist,f'{path}/{name}/')'''

    IS_check = checkISFS(IS,'IS',answerlist)
    FS_check = checkISFS(FS,'FS',answerlist)
    with open('feedback.json','r') as j:
        file = j.read()
        if len(file)>0:
            ne = 'ne'
        else:
            ne = 'e'
    if ne == 'ne':
        with open ('feedback.json','r') as j:
            feedback = json.load(j)
    else:
        feedback ={}
    feedback[name] = {'IS':IS_check,'IS_fmax':float((((IS.get_forces())**2).sum(axis = 1)**0.5).max()),'FS':FS_check,'FS_fmax':float((((FS.get_forces())**2).sum(axis = 1)**0.5).max())}
    with open('feedback.json','w') as j:
        json.dump(feedback,j,indent=2)
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

                        

                        



            
            
