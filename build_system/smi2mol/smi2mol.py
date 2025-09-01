from rdkit.Chem import AllChem
import numpy as np
from ase import Atoms
from ase.io import write
from rdkit import Chem
import re
import os
import json
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
def SMILES2ASEatoms(smi_from_sml,optimize_geometry=True, forcefield='MMFF94'):
        # 使用Openbabel创建分子对象
        smiles = enumerate_smiles(smi_from_sml[0])#!!!!
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无法解析SMILES字符串: {smiles}")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        # 优化几何结构（如果需要）
        if optimize_geometry:
            if forcefield.upper() == 'MMFF94':
                AllChem.MMFFOptimizeMolecule(mol)
            elif forcefield.upper() == 'UFF':
                AllChem.UFFOptimizeMolecule(mol)
            else:
                raise ValueError(f"不支持的力场类型: {forcefield}")
    
        # 提取原子符号和坐标
        conf = mol.GetConformer()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        positions = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    
        # 创建ASE Atoms对象
        molecule_geo = Atoms(symbols=symbols, positions=positions)
        #molecule_geo.set_cell(np.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]]))
        #molecule_geo.set_pbc((True, True, True))
        # 可选：添加键信息作为自定义属性
        bonds = []
        for bond in mol.GetBonds():
            bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble()))
        molecule_geo.info['bonds'] = bonds
    
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
        recordfile = f'{self.output}/species_name.json'
        self.SF = saveFolder
        self.RF = recordfile
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        with open (recordfile,'a') as f:
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
                data = {smi_tp[1]:file_name}
                with open(self.RF, 'r') as f:
                    file = f.read()
                    if len(file)>0:
                        ne='not_empty'
                    else:
                        ne='empty'
                if ne == 'not_empty':
                    with open(self.RF,'r') as f:
                        old_data = json.load(f)
                else:
                    old_data = {}
                old_data.update(data)
                with open(self.RF,'w') as f:
                    json.dump(old_data,f,indent=2)


