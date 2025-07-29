from ase.io import read
from MLP_opt.JmolNN import bond
from ase import atom
from rdkit import Chem
import numpy as np
import re
#import copy
#for BuildMatrix2SMILES()
'''def find_zero_rows_and_columns(matrix):
    # 将输入转换为 NumPy 数组
    matrix = np.array(matrix)
    
    # 找出全零行的索引
    zero_rows = np.where(~matrix.any(axis=1))[0]
    
    # 找出全零列的索引
    zero_columns = np.where(~matrix.any(axis=0))[0]
    
    return zero_rows, zero_columns

def modify_zero_row_and_columns(matrix,i,j,new_value = 0):
    M = copy.deepcopy(matrix)
    M[i, :] = new_value
    M[:, j] = new_value
    return M
 
def smiles_to_adjacency_matrix_and_atoms(smiles):
    """
    SMILES表达式 --> RDkit对象  
    参数：
        SMILES表达式(str)
        返回:
        adjacency_matrix:邻接矩阵(np.array)
        atoms:对应元素列表(list)
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise ValueError("Invalid SMILES string")
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol, useBO=True)  # useBO=True表示考虑键级
    return np.array(adjacency_matrix),atoms

def adjacency_matrix_and_atoms_to_smiles(adjacency_matrix, atoms):
    """
    RDkit对象 --> SMILES表达式
    参数：
        adjacency_matrix:邻接矩阵(np.array)
        atoms:对应元素列表(list)
    返回:
        SMILES表达式(str)
    """
    
    mol = Chem.RWMol()
    for atom_symbol in atoms:
        atom = Chem.Atom(atom_symbol)
        mol.AddAtom(atom)
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if adjacency_matrix[i, j] > 0:
                bond_order = int(adjacency_matrix[i, j])
                if bond_order == 1:
                    bond_type = Chem.BondType.SINGLE
                elif bond_order == 2:
                    bond_type = Chem.BondType.DOUBLE
                elif bond_order == 3:
                    bond_type = Chem.BondType.TRIPLE
                else:
                    raise ValueError(f"Unsupported bond order: {bond_order}")
                mol.AddBond(i, j, bond_type)
    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception as e:
        raise ValueError("Failed to generate SMILES from adjacency matrix and atoms.") from e
    smiles = re.sub(r'(?<!\[)([a-zA-Z])(?!\])', r'[\g<1>]', smiles)
    NEW_mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(NEW_mol,doRandom=False,canonical=False)
    return smiles

def remove_row_and_column(matrix, i):
    """
    删除矩阵的第i行和第i列。
    
    参数:
        matrix (np.ndarray): 输入的矩阵。
        i (int): 要删除的行和列的索引(从0开始)。
    
    返回:
        np.ndarray: 删除第i行和第i列后的矩阵。
    """
    # 删除第i行
    matrix_without_i_row = np.delete(matrix, i, axis=0)
    
    # 删除第i列
    matrix_without_i_row_and_column = np.delete(matrix_without_i_row, i, axis=1)
    
    return matrix_without_i_row_and_column

def combine_matrices(A, B):
    """
    
    参数:
        A (np.ndarray): 左上子矩阵
        B (np.ndarray): 右下子矩阵
    
    返回:
        np.ndarray: 组合后的矩阵
    """
    # 创建零矩阵填充中间部分
    zeros_AB = np.zeros((A.shape[0], B.shape[1]))
    zeros_BA = np.zeros((B.shape[0], A.shape[1]))
    
    # 使用np.block组合矩阵
    combined_matrix = np.block([
        [A, zeros_AB],
        [zeros_BA, B]
    ])
    return combined_matrix

def modify_matrix(matrix, i, j, new_value):
    """
    修改矩阵的第i行、第j列的数值。

    参数:
        matrix (np.ndarray): 输入的矩阵。
        i (int): 行索引(从0开始)。
        j (int): 列索引(从0开始)。
        new_value (float or int): 新的数值。

    返回:
        np.ndarray: 修改后的矩阵。
    """
    # 检查索引是否超出矩阵范围
    if i >= matrix.shape[0] or j >= matrix.shape[1]:
        raise IndexError("索引超出矩阵范围")

    # 修改矩阵的第i行、第j列的数值
    matrix[i, j] = new_value
    return matrix

def add_group_to_mol(smi_group,smi_mol,m,n,l):
    """
    smi_group:被添加基团(H,O(H),C(R))
    smi_mol:主体
    m:基团连接主体的原子于基团中的编号
    n:主体连接基团的原子于主体中的编号
    l:成键键级
    """
    Mol_group = Chem.AddHs(Chem.MolFromSmiles(smi_group))
    Mol_mol = Chem.AddHs(Chem.MolFromSmiles(smi_mol))
    if Mol_group is None or Mol_mol is None:
        raise ValueError("无效的SMILES字符串")
    
    
    # 获取原子总数
    atom_count_group = Mol_group.GetNumAtoms()
    atom_count_mol = Mol_mol.GetNumAtoms()
    adj_matrix_G,atoms_G = smiles_to_adjacency_matrix_and_atoms(smi_group)
    adj_matrix_M,atoms_M = smiles_to_adjacency_matrix_and_atoms(smi_mol)
    adj_combine = combine_matrices(adj_matrix_M,adj_matrix_G)
    atoms_combine = atoms_M +atoms_G
    new_n = n + atom_count_mol
    adj_combine = modify_matrix(adj_combine,m,new_n,l)
    adj_combine = modify_matrix(adj_combine,new_n,m,l)
    smiles_combine = adjacency_matrix_and_atoms_to_smiles(adj_combine,atoms_combine)
    return smiles_combine

def remove_group_to_mol(smi_group,smi_mol,m):
    """
    smi_group:被删除基团(H,O(H),O)
    smi_mol:主体
    m:被删除基团在主体中的编号
    """
    adj_matrix_M,atoms_M = smiles_to_adjacency_matrix_and_atoms(smi_mol)
    adj_matrix_removed = modify_zero_row_and_columns(adj_matrix_M,m,m)
    zero_rows, zero_columns = find_zero_rows_and_columns(adj_matrix_removed)
    for i in sorted(zero_rows, reverse=True):
        adj_matrix_removed = remove_row_and_column(adj_matrix_removed,i)
    atoms_removed =atoms_M
    for index in sorted(zero_columns, reverse=True):
        del atoms_removed[index]
    smiles_removed = adjacency_matrix_and_atoms_to_smiles(adj_matrix_removed,atoms_removed)
    return smiles_removed

class BuildMatrix2SMILES():
    def __init__(self):
        self.atomlist = []
        self.smiles = ''
    def input(self,CB):
        all_atoms = len(CB.atoms)
        self.adjMatrix = np.zeros((all_atoms,all_atoms))
        for atom in CB.atoms:
            self.atomlist.append(atom.elesymbol)
            atom_id = atom.id
            bond_atom_dict = atom.bonddict
            for bond_atom in bond_atom_dict:
                bond_atom_id =bond_atom.id
                if self.adjMatrix[atom_id,bond_atom_id] == 0:
                    self.adjMatrix[atom_id,bond_atom_id] = 1
                    self.adjMatrix[bond_atom_id,atom_id] = 1
                else:
                    pass
        zero_rows, zero_columns = find_zero_rows_and_columns(self.adjMatrix)
        for i in sorted(zero_rows, reverse=True):
            self.adjMatrix = remove_row_and_column(self.adjMatrix,i)
        for index in sorted(zero_columns, reverse=True):
            del self.atomlist[index]
        smiles = adjacency_matrix_and_atoms_to_smiles(self.adjMatrix,self.atomlist)
        print(smiles)
        self.smiles = smiles'''
# new below
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
        
    def min_dis(self,atomID1,atomID2):
        distance = self.poscar.get_distance(atomID1,atomID2, mic=True)
        return distance
    def CheckBondwith2Atoms(self,main_atomID,sub_atomID):
        dis = self.min_dis(main_atomID,sub_atomID)
        main_atom  = self.atoms[main_atomID] 
        sub_atom = self.atoms[sub_atomID]
        if check_NON_metal_atoms(main_atom) == True or check_NON_metal_atoms(sub_atom) == True:
            if check_NON_metal_atoms(main_atom) == True and check_NON_metal_atoms(sub_atom) == True:
                if bond(main_atom.elesymbol,sub_atom.elesymbol,dis).judge_bondorder() == 1:
                    print(f'there is a bond with {main_atom.elesymbol}:{main_atomID} and {sub_atom.elesymbol}:{sub_atomID}.')
                    main_atom.bonddict[sub_atom] = sub_atom.number
                    sub_atom.bonddict[main_atom] = main_atom.number
                else:
                    print(f"there isn't a bond with {main_atom.elesymbol}:{main_atomID} and {sub_atom.elesymbol}:{sub_atomID}.")    
            else:
                if bond(main_atom.elesymbol,sub_atom.elesymbol,dis).judge_bondorder() == 1:
                    print(f'there is adsorption with {main_atom.elesymbol}:{main_atomID} and {sub_atom.elesymbol}:{sub_atomID}.')
                    if check_NON_metal_atoms(main_atom) == True:
                        self.adsorption.append(main_atom)
                    else:
                        self.adsorption.append(sub_atom)
        else:
            pass

    def CheckAllBonds(self):
        atoms = self.poscar
        for i, atom_i in enumerate(atoms):
            for j, atom_j in enumerate(atoms):
                if j > i:
                    self.CheckBondwith2Atoms(i,j)
                else:
                    pass
        print('finish checking ALL bonds')
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
        



if (__name__ == "__main__"):
    test = checkBonds()
    test.input('POSCAR')
    if test.CheckPBC() == True:
        test.AddAtoms()
        test.CheckAllBonds()
    else:
        pass
    output = BuildMol2Smiles(test)
    output.build()
    print(f'OUTPUT:{output.smiles}')
    if output.ads != []:
        print('ads')