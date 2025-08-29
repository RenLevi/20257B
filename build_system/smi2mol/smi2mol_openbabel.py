from openbabel import pybel
import numpy as np
from ase import Atoms
from ase.io import write
from rdkit import Chem
import re
import os
def add_brackets_around_letters(cnmol:str):# 使用正则表达式替换不在[]中的字母字符，前后添加[]:example:[H]CO==>[H][C][O]
    result = re.sub(r'(?<!\[)([a-zA-Z])(?!\])', r'[\g<1>]', cnmol)
    return result
def read_file_line_by_line(file_path):#逐行读取txt文件并返回list数据
    mol_list=[]
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            string = line.strip()  
            mol_list.append(string)
        mol_list.pop(0)
    return mol_list
def check_input_SMILES(smiles):#检查输入的smiles是否符合标准
    # 尝试将SMILES字符串转换为分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False  # SMILES格式不正确
    return True  # SMILES格式正确
def check_atoms_bonds(CHECKLIST):
    free_radical_position_list=[]
    for tp in CHECKLIST:
        if tp[-1] != tp[-2]:
            ValueError
        else:
            atom = tp[1]
            bonds = tp[-1]
            if atom == 'C' and bonds < 4:
                free_radical_position_list.append(tp)
            elif atom == 'O' and bonds < 2:
                free_radical_position_list.append(tp)
            elif atom == 'H' and bonds < 1:
                free_radical_position_list.append(tp)
            else:
                pass
    return free_radical_position_list
def enumerate_smiles(smiles):#列举同一分子smiles的等价表达,得到规范化的smiles
    mol = Chem.MolFromSmiles(smiles)
    smi = Chem.MolToSmiles(mol,doRandom=False,canonical=False)
    return smi
def find_free_radical(smiles,molecule_geo):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    cl = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()          # 原子索引
        element = atom.GetSymbol()        # 元素种类
        bond_count = atom.GetDegree()     # 成键数目（显式连接的原子数）
        total_degree = atom.GetTotalDegree()  # 总连接数（包括隐式氢）
        cl.append((atom_idx,element,bond_count,total_degree))
    freelist=check_atoms_bonds(cl)
    if len(freelist) == 1:
        free_radical_position = molecule_geo.positions[freelist[0][0]]
    elif len(freelist) > 1:
        num = len(freelist)
        sum_position = molecule_geo.positions[freelist[0][0]]
        for i in range(1,len(freelist)):
            atom_ids = freelist[i][0]
            atom_position = molecule_geo.positions[atom_ids]
            sum_position = sum_position + atom_position
        free_radical_position = sum_position/num
    else:
        free_radical_position = molecule_geo.positions[0]
    return free_radical_position    
def SMILES2ASEatoms(smi_from_sml):
        # 使用Openbabel创建分子对象
        smiles = enumerate_smiles(smi_from_sml[0])#!!!!
        molecule = pybel.readstring("smi", smiles)
        molecule.make3D(forcefield='mmff94', steps=100)
        # 创建ASE的Atoms对象
        molecule_geo = Atoms()
        molecule_geo.set_cell(np.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]]))
        molecule_geo.set_pbc((True, True, True))
        # 将Openbabel分子对象添加到ASE的Atoms对象中
        for atom in molecule:
            atom_type = atom.atomicnum
            atom_position = np.array([float(i) for i in atom.coords])
            molecule_geo.append(atom_type)
            molecule_geo.positions[-1] = atom_position
        # 调整分子位置,确定自由基位置
        free_radical_position = find_free_radical(smiles,molecule_geo)
        molecule_geo.translate(-free_radical_position)
        return molecule_geo
class smi2mol:
    def __init__(self,input,output):
        self.input = input#txt文件
        self.output = output
    def mollist2SMILES(self):#将CatNet的输出文件中的非标准SMILES转为标准的SMILES 
        #默认原子之间成键均为单健
        #得到分子数据的列表
        molecule_list = read_file_line_by_line(self.input)
        std_mol_list = []
        for mol_not_std_smi in molecule_list:
            std_mol = add_brackets_around_letters(mol_not_std_smi)
            std_mol_list.append((std_mol,mol_not_std_smi))#([H][C][H],[H]C[H])
        self.sml = std_mol_list
        return std_mol_list
    def creat_folder_and_file(self):
        saveFolder = f'{self.output}/species'
        recordfile = f'{self.output}/species_name.txt'
        self.SF = saveFolder
        self.RF = recordfile
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        with open (recordfile,'w') as f:
            pass
    def build_model(self):
        for smi_tp in self.sml:     
            if check_input_SMILES(smi_tp[0]) == False:
                print(f"The SMILES '{smi_tp[0]}' are incorrect")
                break
            else:
                ase_mol = SMILES2ASEatoms(smi_tp)
                #save xyz file
                file_name = f'{smi_tp[1]}.xyz'#可更改文件保存路径以及格式
                write(os.path.join(self.SF, file_name), ase_mol)
                with open(self.RF, 'a') as file:
                    file.write(f'{smi_tp[1]}:{file_name}\n')


