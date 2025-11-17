import build_ISFS_model.readReaction as rR
import os
import json
from ase.io import read
import numpy as np
'''
    #:注释
    #...:用于测试，可以删除
    #?:存疑
'''
def get_fmax_from_traj(traj_file):
    # 读取轨迹文件的最后一个结构（单个Atoms对象）
    atoms = read(traj_file, index=-1)
    # 直接从Atoms对象获取力
    forces = atoms.get_forces()
    force_magnitudes = np.linalg.norm(forces, axis=1)
    fmax = np.max(force_magnitudes)
    return fmax
def find_uppercase_difference(strA, strB):
    """
    找出字符串B比字符串A多出的大写字母
    
    参数:
    strA: 原始字符串
    strB: 目标字符串
    
    返回:
    extra_uppercase: B比A多出的大写字母列表
    """
    # 提取两个字符串中的所有大写字母
    uppercase_A = [char for char in strA if char.isupper()]
    uppercase_B = [char for char in strB if char.isupper()]
    
    # 创建大写字母计数字典
    count_A = {}
    for char in uppercase_A:
        count_A[char] = count_A.get(char, 0) + 1
    
    count_B = {}
    for char in uppercase_B:
        count_B[char] = count_B.get(char, 0) + 1
    
    # 找出B比A多出的大写字母
    extra_uppercase = []
    for char, count in count_B.items():
        if char not in count_A:
            # B中有而A中没有的字母，全部算作多出的
            extra_uppercase.extend([char] * count)
        elif count > count_A[char]:
            # B中比A中多的部分
            extra_uppercase.extend([char] * (count - count_A[char]))
    
    return extra_uppercase
def read_file_line_by_line(file_path):#逐行读取txt文件并返回list数据
    reaction_list=[]
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            string = line.strip()  
            reaction_list.append(string)
        reaction_list.pop(0)
    return reaction_list
def collect_Alljson(path_d,path_test,check_json,batch):
    jobsubopt = f'{path_test}/jobsub/opt'
    record_all_p = f'{path_d}record_adscheck.json'
    with open (record_all_p,'w') as j1:
        pass
    for i in range(batch):
        json_i = f'{jobsubopt}/{i}/record.json'
        with open (json_i,'r') as j2:
            di = json.load(j2)
        
        with open(record_all_p, 'r') as f1:
            file = f1.read()
            if len(file)>0:
                ne = 'ne'
            else:
                ne = 'e'
        if ne == 'ne':
            with open (record_all_p,'r') as f2:
                old_data = json.load(f2)
        else:
            old_data ={}
        old_data.update(di)
        with open(record_all_p, 'w') as f3:
            json.dump(old_data,f3,indent=2)
    with open(record_all_p, 'r') as F:
        d_all = json.load(F)
    with open(check_json,'r') as J:
        d_check = json.load(J)
    if len(d_all) != len(d_check):
        return ValueError('some record loss!!!')
    else:
        pass
class molfile():
    def __init__(self,name,path_test,slab = None,random = 20):
        '''
        name:[cp,wf,wb,wna,waH]
        "[H]": [[1],[],[],[],[]]
        "[H]C([H])([H])O": [[1],[],[],[],[]]
        '''
        [cp,wf,wb,wna,waH] = [[],[],[],[],[]]
        species = f'{path_test}opt/system/species/'
        for i in range(1,random+1):
            cb = rR.checkBonds()
            cb.input(f'{species}{name}/{i}/nequipOpt.traj')
            cb.AddAtoms()
            cb.CheckAllBonds()
            bm2s = rR.BuildMol2Smiles(cb)
            bm2s.build()
            try:
                fmax = get_fmax_from_traj(f'{species}{name}/{i}/nequipOpt.traj')
            except Exception:
                fmax=-1
            if fmax > 0.05:
                wf.append(i)
                #print(f'struct_{i} fmax up to limit')
            elif fmax ==-1:
                wf.append(i)
                #print(f'struct_{i} fmax wrong')
            else:
                if bm2s.smiles == name:
                    if name != '[H]' and name != '[H][H]':
                        ch =[]
                        for a in bm2s.ads:
                            if a.elesymbol == 'H':ch.append(True)
                            else:ch.append(False)
                        if bm2s.ads != []:
                            if all(ch)==False:
                                cp.append(i)
                            else:
                                waH.append(i)
                        elif bm2s.ads == []:
                            wna.append(i)
                    else:
                        if bm2s.ads != []:
                            cp.append(i)
                elif bm2s.smiles != name:
                    wb.append(i)     
        if cp != []:
            E=[]
            for i in cp:
                last_atoms = read(f'{species}{name}/{i}/nequipOpt.traj',index=-1)
                final_energy = last_atoms.get_potential_energy()
                E.append(final_energy)
            min_E = min(E)
            id = E.index(min_E)
            self.model_p = f'{species}{name}/{cp[id]}/nequipOpt.traj'
        else:
            if len(wna+waH) > len(wb):#？存疑
                E = []
                for  i in wna:
                    last_atoms = read(f'{species}{name}/{i}/nequipOpt.traj',index=-1)
                    final_energy = last_atoms.get_potential_energy()
                    E.append(final_energy)
                min_E = min(E)
                id = E.index(min_E)
                self.model_p = f'{species}{name}/{wna[id]}/nequipOpt.traj'
            else:
                self.model_p = None
class PREforSearchTS():
    def __init__(self,path_test):
        self.mainfolder = path_test#/work/home/ac877eihwp/renyq/xxx/test/
        self.opt = path_test+'opt/'
        self.neb = path_test+'RDA_S/'
        self.file = {}
        self.slab = read(f'{path_test}opt/slab/Ru_hcp0001.vasp')
    def readDataPath(self):
        print('Start reading data from opt')
        def read_json(jsonfile):
            with open(jsonfile,'r') as j:
                dictionary = json.load(j)
            return dictionary
        self.foldername_json =f'{self.opt}system/folder_name.json'
        folder_dict=read_json(self.foldername_json)
        count_file = 0
        for file in folder_dict:
            mf = molfile(file,self.mainfolder,self.slab)
            self.file[file] = mf
            count_file +=1
        if count_file != len(folder_dict):
            ValueError  
        else:pass
        print('Finish reading data from opt')        
    def buildmodel(self,reaction_txt):
        slab = self.slab
        mainfolder = self.neb
        os.makedirs(mainfolder, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
        with open(f'{mainfolder}foldername.json', 'w') as file:
            pass
        reaction_list = read_file_line_by_line(reaction_txt)
        for reaction in reaction_list:
            rlist = rR.str2list(reaction)
            initial_mol = self.file[rlist[0][0]]
            final_mol = self.file[rlist[-1][0]]
            def warp(molstr,reactionlist):
                if molstr!='O/OH':
                    if molstr == 'H':
                        return '[H]'
                    elif molstr == 'OH':
                        return 'O[H]'
                    else:
                        return molstr
                else:
                    extra = find_uppercase_difference(rlist[-1][0],rlist[0][0])
                    if len(extra)  == 2:
                        return 'O[H]'
                    else:
                        return 'O'
            add_mol = self.file[warp(rlist[1][-4],rlist)]
            if initial_mol.model_p == None or final_mol.model_p == None:
                print(f'{reaction} -- IS:{initial_mol.model_p};FS:{final_mol.model_p}')
            else:
                RR = rR.STARTfromBROKENtoBONDED(initial_mol.model_p,add_mol.model_p,final_mol.model_p)
                RR.site_finder(slab)
                RR.run(reaction)
                subfolder = f'{mainfolder}{rlist[0][0]}_{rlist[-1][0]}/'
                data = {f'{rlist[0][0]}_{rlist[-1][0]}':[reaction,RR.tf]}
                with open(f'{mainfolder}foldername.json', 'r') as f:
                    file = f.read()
                    if len(file)>0:
                        ne = 'ne'
                    else:
                        ne = 'e'
                if ne == 'ne':
                    with open (f'{mainfolder}foldername.json','r') as f:
                        old_data = json.load(f)
                else:
                    old_data ={}
                old_data.update(data)
                with open(f'{mainfolder}foldername.json', 'w') as f:
                    json.dump(old_data,f,indent=2)
                os.makedirs(subfolder, exist_ok=True)
                RR.save(subfolder,'POSCAR')
        with open(f'{mainfolder}foldername.json', 'r') as f:
            self.d = json.load(f)               
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
    
class SearchTS4All(PREforSearchTS):
    def __init__(self,path_test):
        self.neb = path_test+'RDA_S/'
        with open(f'{self.neb}foldername.json', 'r') as f:
            self.d = json.load(f)







