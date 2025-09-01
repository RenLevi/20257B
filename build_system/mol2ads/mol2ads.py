import ast
from ase import Atoms
from ase.build import rotate
from ase.build import hcp0001
from ase.io import read, write
import numpy as np
from ase.constraints import FixAtoms
import copy
import os
import re
import json
def save_file(save_path,filename,model):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    write(os.path.join(save_path, filename), model, format='vasp', vasp5=True, direct=True)
    '''with open(os.path.join(save_path, filename), 'r') as f:
        lines = f.readlines()
        print("".join(lines[5:8]))  # 查看坐标行示例'''
## 分子操作
'''rotate'''
def rotate_mol(mol,rotate_matrix1,rotate_matrix2):#rotate molecule
    molcopy = copy.deepcopy(mol)
    (angle1,axis1) = rotate_matrix1
    (angle2,axis2) = rotate_matrix2
    molcopy.rotate(angle1,axis1,center=(0,0,0))
    molcopy.rotate(angle2,axis2,center=(0,0,0))
    return molcopy
'''place'''
def place_mol_on_surface(mol,surface,shift_vector):#place molecule
    surfacecopy = copy.deepcopy(surface)
    # 找到Ru(0001)表面最上层原子的z坐标最大值
    z_max = max(surfacecopy.positions[:, 2])
    # 计算分子的结合位点,将分子的质心移动到这个高度
    molecule_center = shift_vector + np.array([0,0,z_max])
    coordinates = mol.get_positions()
    average_coordinates = coordinates.mean(axis=0)
    mol.translate(molecule_center-average_coordinates)
    # 将分子添加到表面上
    system = surfacecopy + mol
    return system
'''random place'''
def random_place(size):
    x=size[0]
    y=size[1]
    #z=size[2]
    # 随机参数范围！！！
    x_range = [0,2.7*x]#Ru-Ru = 2.7A
    y_range = [0,2.7*y*((3**0.5)/2)]
    z_range = [2,3]#化学吸附
    x_sv = np.random.uniform(x_range[0], x_range[1])
    y_sv = np.random.uniform(y_range[0], y_range[1])
    z_sv = np.random.uniform(z_range[0], z_range[1])
    sv = [x_sv,y_sv,z_sv]
    theta_z = np.degrees(np.random.uniform(0, 2*np.pi))
    varphi_y = np.degrees(np.random.uniform(0, 2*np.pi))
    return sv,theta_z,varphi_y
## 检查原子之间距离（H：0.5埃；other atom：1埃）
def check_dist_between_atoms(structure):
    cutoff_H = 0.5  # H-other atoms(including H)
    cutoff_other = 1.0 #other atoms - other atoms(both except H)
    for i in range(len(structure)):
        for j in range(i+1, len(structure)):
            atom_i = structure[i]
            atom_j = structure[j]
            element_i = atom_i.symbol
            element_j = atom_j.symbol
            dist = structure.get_distances(i,j,mic=True)
            if element_i == 'H' or element_j == 'H':
                if dist < cutoff_H:
                    print(f'原子 {element_i}:{i} 和原子 {element_j}:{j} 之间的距离为 {dist}埃, 小于截断值 {cutoff_H}')
                    return False
                else:
                    pass
            else:
                if dist < cutoff_other:
                    print(f'原子 {element_i}:{i} 和原子 {element_j}:{j} 之间的距离为 {dist}埃, 小于截断值 {cutoff_other}')
                    return False
                else:
                    pass
    return True
## 检查分子是否位于表面以上 & 吸附原子是否位于分子最下方
def check_molecule_over_surface(surface,mol,sv):
    z_max = max(surface.positions[:,2])+0.5#高于表面0.5A的距离
    #molecule_center = sv + np.array([0,0,z_max])
    z_min_mol = min(mol.positions[:,2])
    if z_min_mol < z_max:
        print(f'部分原子距离催化剂表面不到0.5埃')
        return False
    '''elif molecule_center[2] > z_min_mol:
        print(f'吸附原子未位于最靠近表面位置')
        return False'''
    return True
class mol2ads:
    def __init__(self,input,output,metal):
        self.input = input#species_name.json
        self.output = output
        self.metal = metal
        with open(self.input, 'r') as file:
            dictionary = json.load(file)
        self.d =dictionary
    def creat_folder_and_file(self):
        saveFolder = f'{self.output}/species'
        self.SF = saveFolder
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
    def build_model(self,smi2molPATH,random_mol_num,size):
        metal = read(self.metal)
        adGroup_dict = self.d
        adGroup_namelist = list(adGroup_dict.keys())
        adGroup_filelist = list(adGroup_dict.values())
        group_num = len(adGroup_namelist)
        txt_name = f'{self.output}/folder_name.json'
        with open(txt_name, 'a') as file:
            pass
        for i in range(group_num):
            path_of_mol = f'{smi2molPATH}/{adGroup_filelist[i]}'
            adGroup_mol = read(path_of_mol)
            adGroup_name = adGroup_namelist[i]
            species_file_floder_name = adGroup_name
            data = {species_file_floder_name:adGroup_name}
            with open(txt_name, 'r') as f:
                file = f.read()
                if len(file)>0:
                    ne = 'ne'
                else:
                    ne = 'e'
            if ne == 'ne':
                with open (txt_name,'r') as f:
                    old_data = json.load(f)
            else:
                old_data ={}
            old_data.update(data)
            with open(txt_name, 'w') as f:
                json.dump(old_data,f,indent=2)
            j = 1
            while j <= random_mol_num:#随机模型数量
                sv,theta_z,varphi_y = random_place(size)
                mol = rotate_mol(adGroup_mol,(theta_z,'z'),(varphi_y,'y'))
                system = place_mol_on_surface(mol,metal,sv)
                if check_dist_between_atoms(system) == True and check_molecule_over_surface(metal,mol,sv) == True:
                    floder_n = f'{self.SF}/{adGroup_name}/{str(j)}'
                    file_n = 'POSCAR'
                    save_file(floder_n,file_n,system)
                    j=j+1
                else:
                    j=j
            print(f'Complete {adGroup_namelist[i]} model (total {random_mol_num})')

