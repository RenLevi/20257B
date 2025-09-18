import NEB.readReaction as rR
import os
import shutil
import json
from ase.io import read
'''
    #:注释
    #...:用于测试，可以删除
    #?:存疑
'''
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
    def __init__(self,checkdict,name,path_test):
        self.name =name
        self.report = checkdict[name]#[cp,wf,wb,wna,waH]
        '''
        name:[cp,wf,wb,wna,waH]
        "[H]": [[1],[],[],[],[]]
        "[H]C([H])([H])O": [[1],[],[],[],[]]
        '''
        [cp,wf,wb,wna,waH] = self.report
        species = f'{path_test}opt/system/species/'
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
class PREforNEB():
    def __init__(self,path_test):
        self.mainfolder = path_test#/work/home/ac877eihwp/renyq/xxx/test/
        self.opt = path_test+'opt/'
        self.neb = path_test+'neb/'
        self.file = {}
    def readDataPath(self):
        print('Start reading data from opt')
        def read_json(jsonfile):
            with open(jsonfile,'r') as j:
                dictionary = json.load(j)
            return dictionary
        opt_check_json = f'{self.opt}system/record_adscheck.json'
        self.foldername_json =f'{self.opt}system/folder_name.json'
        folder_dict=read_json(self.foldername_json)
        opt_check_dict = read_json(opt_check_json)
        count_file = 0
        for file in opt_check_dict:
            mf = molfile(opt_check_dict,file,self.mainfolder)
            self.file[file] = mf
            count_file +=1
        if count_file != len(folder_dict):
            ValueError
        else:pass
        print('Finish reading data from opt')
    def buildNEB(self,reaction_txt):
        mainfolder = self.neb
        os.makedirs(mainfolder, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
        with open(f'{mainfolder}foldername.json', 'w') as file:
            pass
        reaction_list = read_file_line_by_line(reaction_txt)
        for reaction in reaction_list:
            rlist = rR.str2list(reaction)
            initial_mol = self.file[rlist[0][0]]
            final_mol = self.file[rlist[-1][0]]
            if initial_mol.model_p == None or final_mol.model_p == None:
                pass
            else:
                RR = rR.readreaction(initial_mol.model_p,final_mol.model_p,reaction)
                RR.readfile()
                subfolder = f'{mainfolder}{rlist[0][0]}_{rlist[-1][0]}/'
                data = {f'{rlist[0][0]}_{rlist[-1][0]}':[reaction,RR.group1,RR.group2,RR.check,RR.split]}#File name ：[Reaction,idx,idx,bonded smiles,broken smiles]
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
if (__name__ == "__main__"):
    path0='/work/home/ac877eihwp/renyq/sella/test'
    pre_neb =PREforNEB(path_test=path0)
    pre_neb.readDataPath()
    pre_neb.buildNEB(path0)







