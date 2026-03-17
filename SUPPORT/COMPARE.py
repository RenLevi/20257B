import json
from pathlib import Path
from collections import Counter
from ase.io import read
from ase import Atoms
import numpy as np
import pandas as pd
import SUPPORT.support as sp
import os
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

class compare_NCS_RDAS():
    def __init__(self,NCS,RDAS):
        self.r = {}
        self.ncs = NCS
        self.rdas = RDAS
        self.preDATA = None
    def readfromtxt(self,txtpath):
        def read_file_line_by_line(file_path):#逐行读取txt文件并返回list数据
            reaction_list=[]
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    string = line.strip()  
                    reaction_list.append(string)
                reaction_list.pop(0)
            return reaction_list
        def str2list(reaction:str):
            r1 = reaction.split(">")
            r2 = []
            for i in r1:
                i1 = i.split()
                r2.append(i1)
            return r2
        def OorOH(OOH):
            issmiles = OOH[0][0]
            fssmiles = OOH[-1][0]
            type = OOH[1][0]
            addgroup = OOH[1][-4]
            if addgroup == 'O/OH':
                c1 = Counter(ch for ch in issmiles if ch.isalpha())
                c2 = Counter(ch for ch in fssmiles if ch.isalpha())
                all_keys = set(c1) | set(c2)
                ooh = sum(abs(c1.get(k, 0) - c2.get(k, 0)) for k in all_keys)
                if ooh == 2:
                    return 'OH'
                if ooh == 1:
                    return 'O'
                else:
                    print('error in check O or OH')
                    return ValueError('error in check O or OH')
            else:
                return None
        rlist = read_file_line_by_line(txtpath)
        for r in rlist:
            r0 = r
            r=str2list(r)
            self.r[f'{r[0][0]}_{r[-1][0]}']={
                                            'Reaction':r0,
                                            "IS":r[0][0],
                                            'FS':r[-1][0],
                                            'type':r[1][0],
                                            'AddGroup': r[1][-4],
                                            'Remove O/OH':OorOH(r)
                                            }
    def readAllJson(self,jsonpath,number):
        data = {}
        for i in range(number):
            jn = f'{i}.json'
            jp = Path(jsonpath)/jn
            with open (jp,'r') as f:
                old_data = json.load(f)
            data.update(old_data)
        self.preDATA = data
        with open('predata.json','w') as j:
            json.dump(self.preDATA,j,indent=2)
    def CheckModelFiles(self,filename,mainfolder,model):#model=NCS,RDAS
        def check_file_exists_pathlib(folder_path, file_name):
            file_path = Path(folder_path) / file_name
            return file_path.is_file()
        if self.r == {}:
            print('run readfromtxt first')
            return None
        else:
            for k in self.r:
                if model == 'NCS':
                    p = Path(mainfolder)/k
                elif model == 'RDAS':
                    p = Path(mainfolder)/k/'IntermediateProcess/results'
                if check_file_exists_pathlib(p,filename)==True:
                    self.r[k][f'{model}']=p/filename
                else:
                    self.r[k][f'{model}']=None
    def compare(self):
        compare_data ={}
        def Euclidean_distance(R1:Atoms, R2:Atoms):
            """
            计算两组结构的欧氏距离
            """
            p1 = R1.get_positions()
            p2 = R2.get_positions()
            assert len(R1) == len(R2)
            return np.sqrt(np.sum((p1 - p2) ** 2))
        compare_data[' '] = ['Reaction','Encs','Erdas','diff_E','diff_Eu','ncs_dist','rdas_dist','diff_dist']
        for k in self.r:
            print(self.r[k]['Reaction'])
            if  self.r[k]['NCS'] != None:
                ncs = read(self.r[k]['NCS'])
                Encs = ncs.get_potential_energy()
            elif self.r[k]['NCS'] == None:
                ncs = None
                Encs = None
            if self.r[k]['RDAS'] != None:
                rdas = read(self.r[k]['RDAS'])
                Erdas = rdas.get_potential_energy()
            elif self.r[k]['RDAS'] == None:
                rdas = None
                Erdas= None
            
            if rdas!=None and ncs!=None:
                diff = Euclidean_distance(ncs,rdas)
                diff_E = np.abs(Encs-Erdas)
                [id1,id2]=self.preDATA[k]["info list"][1]
                rdas_dist = rdas.get_distance(id1,id2)
                ncs_dist = ncs.get_distance(id1,id2)
                diff_dist = np.abs(ncs_dist-rdas_dist)
            else:
                diff_E = None
                diff = None
                rdas_dist = None
                ncs_dist = None
                diff_dist = None
            r = [self.r[k]['Reaction'],Encs,Erdas,diff_E,diff,ncs_dist,rdas_dist,diff_dist]
            compare_data[self.r[k]['Reaction']] = r
        df = pd.DataFrame(compare_data)
        df =df.T
        df.to_excel('compare.xlsx', index=False, sheet_name='compare.xlsx')
    def move_file_new_folder(self,folder):
        os.makedirs(folder,exist_ok=True)
        for k in self.r:
            print(self.r[k]['Reaction'])
            fp =Path(folder)/k
            os.makedirs(fp,exist_ok=True)
            os.makedirs(fp/'RDAS',exist_ok=True)
            os.makedirs(fp/'NCS',exist_ok=True)
            if self.r[k]['NCS'] != None and self.r[k]['RDAS'] != None:
                sp.copyFiles(self.r[k]['RDAS'],fp/'RDAS')
                self.r[k]['RDAS_freq']=fp/'RDAS'
                sp.copyFiles(self.r[k]['NCS'],fp/'NCS')
                self.r[k]['NCS_freq']=fp/'NCS'
            elif self.r[k]['NCS'] == None and self.r[k]['RDAS'] != None:
                sp.copyFiles(self.r[k]['RDAS'],fp/'RDAS')
                self.r[k]['RDAS_freq']=fp/'RDAS'
            elif self.r[k]['NCS'] != None and self.r[k]['RDAS'] == None:
                sp.copyFiles(self.r[k]['NCS'],fp/'NCS')
                self.r[k]['NCS_freq']=fp/'NCS'
            else:pass
        class PathEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Path):
                    return str(obj)
                return super().default(obj)
        with open('TS.json','w') as j:
            json.dump(self.r,j,indent=2,cls=PathEncoder)




                




def main():
    CNR = compare_NCS_RDAS('/public/home/ac877eihwp/renyq/C2/test/reactions','/public/home/ac877eihwp/renyq/C2/test/All4RDA_S')
    CNR.readAllJson(CNR.rdas,29)
    CNR.readfromtxt('/public/home/ac877eihwp/renyq/C2/test/RN/reactionslist.txt')
    CNR.CheckModelFiles('optimized_ts.xyz',CNR.ncs,'NCS')
    CNR.CheckModelFiles('TS.xyz',CNR.rdas,'RDAS')
    #CNR.compare()
    CNR.move_file_new_folder(folder='/public/home/ac877eihwp/renyq/C2/test/FQFE')

if __name__ == "__main__":
    main()

            
            

    
