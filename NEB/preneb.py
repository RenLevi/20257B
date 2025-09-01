import readReaction as rR
import os
import shutil
import json
def read_file_line_by_line(file_path):#逐行读取txt文件并返回list数据
    reaction_list=[]
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            string = line.strip()  
            reaction_list.append(string)
        reaction_list.pop(0)
    return reaction_list
def copyFiles(source_file,dest_folder):
# 源文件路径source_file = '/path/to/source/file.txt'
# 目标文件夹路径dest_folder = '/path/to/destination/folder'
# 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    # 目标文件路径
    dest_file = os.path.join(dest_folder, os.path.basename(source_file))
    # 复制文件
    try:
        shutil.copy2(source_file, dest_file)
        print(f"文件已成功复制到 {dest_file}")
    except IOError as e:
        print(f"无法复制文件. {e}")
    except Exception as e:
        print(f"发生错误: {e}")
class molfile():
    def __init__(self,checkdict,name:str):
        self.name =name
        self.report = checkdict[name]#[cp,wf,wb,wna,waH]
        '''
        name:[cp,wf,wb,wna,waH]
        "[H]": [[1],[],[],[],[]]
        "[H]C([H])([H])O": [[1],[],[],[],[]]
        '''
        [cp,wf,wb,wna,waH] = self.report
        if cp != []:
            self.model_list = cp
        else:
            if wna != [] and wf == [] and wb == []:
                self.model_list =  wna+waH
            else:
                self.model_list =[]
                print

class PREforNEB():
    def __init__(self,pathforcal):
        self.mainfolder = pathforcal#/work/home/ac877eihwp/renyq/xxx/test/
        self.opt = pathforcal+'opt/'
        self.neb = pathforcal+'neb/'
        self.file = {}
    def readDataPath(self):
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
            mf = molfile(opt_check_dict,file)
            self.file[file] = mf
            count_file +=1
        if count_file != len(folder_dict):
            ValueError
        else:pass
        print('finish reading data from opt')
    def buildNEB(self,reaction_txt):
        mainfolder = self.neb
        os.makedirs(mainfolder, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
        with open(mainfolder+'foldername.json', 'w') as file:
            pass
        reaction_list = read_file_line_by_line(reaction_txt)
        for reaction in reaction_list:
            rlist = rR.str2list(reaction)
            initial_mol = self.file[rlist[0][0]]
            final_mol = self.file[rlist[-1][0]]
            if initial_mol.notuse == True or final_mol.notuse== True:
                pass
            else:
                subfolder = mainfolder + str(rlist[0][0])+'_'+str(rlist[-1][0])+'/'
                with open(mainfolder+'foldername.txt', 'a') as file:
                    reaction_floder_name = str(rlist[0][0])+'_'+str(rlist[-1][0])
                    file.write(f'{reaction_floder_name}:{reaction}\n')
                os.makedirs(subfolder, exist_ok=True)
                RR = rR.readreaction(initial_mol.path,final_mol.path,reaction)
                RR.readfile()
                RR.save(subfolder,'POSCAR')

if (__name__ == "__main__"):
    path0='/work/home/ac877eihwp/renyq/sella/test'
    pre_neb =PREforNEB(pathforcal=path0)
    pre_neb.readDataPath()
    pre_neb.buildNEB(path0)







