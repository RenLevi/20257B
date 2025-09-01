import ase.io
from nequip.ase import NequIPCalculator
from ase.optimize import BFGS,FIRE
import ase.io
import os
import sys
import logging
import MLP_opt.CheckNN as CheckNN
from ase.io import read
import numpy as np
import json
def get_fmax_from_traj(traj_file):
    # 读取轨迹文件的最后一个结构（单个Atoms对象）
    atoms = read(traj_file, index=-1)
    # 直接从Atoms对象获取力
    forces = atoms.get_forces()
    force_magnitudes = np.linalg.norm(forces, axis=1)
    fmax = np.max(force_magnitudes)
    return fmax
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
        w_pbc = []
        w_bond = []
        w_no_ads = []
        w_ads_H = []
        checkpass = []
        for i in range(1,random_number+1):
            print(f'{foldername} struct_{i} start')
            struct=ase.io.read(f'{sys_path}/{i}/POSCAR')
            struct.set_calculator(nequipModel)
            print(f' Starting optmization by NequIP model:')
            FIRE(struct).run(fmax=0.1,steps=500)
            BFGS(struct, trajectory=f'{sys_path}/{i}/nequipOpt.traj').run(fmax=0.05,steps=1000)
            fmax = get_fmax_from_traj(f'{sys_path}/{i}/nequipOpt.traj')
            if fmax > 0.05:
                w_fmax.append(i)
                print(f'struct_{i} fmax up to limit')
            else:
                CB = CheckNN.checkBonds()
                CB.poscar = struct
                if CB.CheckPBC == True:
                    CB.AddAtoms()
                    CB.CheckAllBonds()
                else:
                    w_pbc.append(i)
                BM2S = CheckNN.BuildMol2Smiles(CB)
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
        self.wp = w_pbc
        self.wb = w_bond
        self.wna = w_no_ads
        self.waH = w_ads_H
        return checkpass
'''class opt4ALLsystems():#顺序
    def __init__(self,input,random_number):
        self.input = input#path
        self.random = random_number
        self.folder = f'{input}/folder_name.txt'
    def txt_to_dict(self):
        dictionary = {}
        with open(self.folder, 'r') as file:
            for line in file:
                # 移除行首行尾的空白字符
                line = line.strip()
                # 忽略空行和注释行（假设注释行以'#'开头）
                if line and not line.startswith('#'):
                    # 分割键和值，最多分割一次
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key, value = parts
                        dictionary[key.strip()] = value.strip()
                    else:
                        print(f"无法解析的行：{line}")
                        break
        self.d =dictionary
        return dictionary
    def start_cal_order(self):
        fd = self.d
        floderlist = list(fd.keys())
        for name in floderlist:
            MLP4SYS = mlpopt4system(self.input,name)
            MLP4SYS.start(self.random)'''
class opt4ALLsystems_batch_by_batch():#并行顺序
    def __init__(self,input,random_number):
        self.input = input#path
        self.random = random_number
        self.folder = f'{input}/folder_name.json'
        with open(self.folder,'r') as f:
            dictionary = json.load(f)
        self.d = dictionary
    def start_split(self,batch):
        fd = self.d
        floderlist = list(fd.keys())
        n=len(floderlist)
        r=n%batch
        klow=int(n/batch)
        split = [klow+1 for _ in range(r)] + [klow for _ in range(10-r)]
        foldersplit=[]
        for i in split:
            il=floderlist[0:i]
            del floderlist[:i]
            foldersplit.append(il)
        self.fsp = foldersplit
        return foldersplit




        
            

        








    



            
            