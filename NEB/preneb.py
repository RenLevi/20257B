import readReaction as rR
import os
import shutil
import ast
def is_number(s):
    """通用数值检测（允许整数、浮点数、科学计数法）"""
    try:
        float(s)
        return True
    except ValueError:
        return False
def txt_to_dict(filename):
    dictionary = {}
    with open(filename, 'r') as file:
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
    return dictionary
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
    def __init__(self,name:str,report:str,path:str):
        self.name =name
        self.report = ast.literal_eval(report)
        print(self.report)
        self.notuse=bool('WRONG' in self.report)
        if self.notuse == False:
            if self.report[-1] == "Ads":
                self.path = path+'struct_'+self.report[0]+'/opt.vasp'
                self.ads = True
            elif self.report[-1] =="NoAds":
                self.path = path+'struct_'+self.report[0]+'/opt.vasp'
                self.ads = False
        else:
            self.path = path+'struct_1'+'/opt.vasp'
            self.ads = False
class PREforNEB():
    def __init__(self,pathforcal):
        self.mainfolder = pathforcal#/work/home/ac877eihwp/renyq/sella/test
        self.output = pathforcal+'output/'
        self.neb = pathforcal+'neb/'
        self.file = {}
    def readDataPath(self):
        val_pass_txt = self.val+'mol_to_ad/checkbondpass.txt'
        output_pass_txt = self.output+'mol_to_ad/checkbondpass.txt'
        wrong_txt = self.val+'mol_to_ad/wrong.txt'
        self.foldername_txt =self.output+'mol_to_ad/floder_name.txt'
        folder_dict=txt_to_dict(self.foldername_txt)
        output_dict_pass = txt_to_dict(output_pass_txt)
        val_dict_pass = txt_to_dict(val_pass_txt)
        wrongdict = txt_to_dict(wrong_txt)
        count_file = 0
        for file in output_dict_pass:
            if file == None:pass
            else:
                mf = molfile(file,output_dict_pass[file],self.output+'mol_to_ad/'+file+'/')
                self.file[file] = mf
                count_file +=1
        for file in val_dict_pass:
            if file == None:pass
            else:
                mf = molfile(file,val_dict_pass[file],self.val+'mol_to_ad/'+file+'/')
                self.file[file] = mf
                count_file +=1
        for file in wrongdict:
            if file == None:pass
            else:
                mf = molfile(file,wrongdict[file],self.val+'mol_to_ad/'+file+'/')
                self.file[file] = mf
                count_file +=1
        if count_file != len(folder_dict):
            ValueError
        else:pass
        print('finish reading data from folders')
    def buildNEB(self,file_path4support):
        mainfolder = self.neb
        os.makedirs(mainfolder, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
        with open(mainfolder+'foldername.txt', 'w') as file:
            pass
        rtl_txt = self.mainfolder+'reactionslist.txt'
        reaction_list = read_file_line_by_line(rtl_txt)
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
                copyFiles(file_path4support,subfolder)

if (__name__ == "__main__"):
    path0='/work/home/ac877eihwp/renyq/sella/test'
    pre_neb =PREforNEB(pathforcal=path0)
    pre_neb.readDataPath()
    pre_neb.buildNEB(path0)







