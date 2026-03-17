import json
from pathlib import Path
from collections import Counter
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


class CreateInput4CATKINAS():
    def __init__(self,savePath=None,dict_input=None):
        self.r = {}
        self.p = savePath
        if dict_input:
            data = dict_input
        else:
            data = {
                    'Reaction mechanism':{},
                    'Reaction energy':{},
                    'Reaction condition':{},
                    'Control parameter':{}
            }
        with open('input.json','w') as j:
            json.dump(data,j)
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
            r=str2list(r)
            self.r[f'{r[0][0]}_{r[-1][0]}']={
                                            "IS":r[0][0],
                                            'FS':r[-1][0],
                                            'type':r[1][0],
                                            'AddGroup': r[1][-4],
                                            'Remove O/OH':OorOH(r)
                                            }
    def CheckModelFiles(self,filename,mainfolder,model):
        def check_file_exists_pathlib(folder_path, file_name):
            file_path = Path(folder_path) / file_name
            return file_path.is_file()
        if self.r == {}:
            print('run readfromtxt first')
            return None
        else:
            for k in self.r:
                p = Path(mainfolder)/k
                if check_file_exists_pathlib(p,filename)==True:
                    self.r[k][f'{model}_path']=p/filename
                else:
                    self.r[k][f'{model}_path']=None
    
    



CI4C = CreateInput4CATKINAS()
CI4C.readfromtxt('C:/Users/renyq/Desktop/20257b/reactionslist.txt')
