import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core.periodic_table import Element
import operator
from pymatgen.io.cif import CifParser
from pymatgen.core import Molecule

def readfile(path,filemat):
    if filemat == 'poscar':
        structure = Structure.from_file(path)
    elif filemat == 'xyz':
        structure = Molecule.from_file(path)
    elif filemat == 'cif':
        parser = CifParser(path)
        structure = parser.get_structures()[0]  # 假设需要第一个结构
    return structure
def cal_w_vor(j_dict):
    w_sa = j_dict['solid_angle']
    w_fa = j_dict['area']
    w_vor = w_sa**2/w_fa
    return w_vor

def cal_w_dc(dij,dlow,dhigh):
    if dij<=dlow:
        w_dc = 1
    elif dij>dlow and dij<dhigh:
        w_dc = np.sqrt(1 + np.cos(np.pi * (dij - dlow) / (dhigh - dlow)))
    elif dij>=dhigh:
        w_dc = 0
    return w_dc 

def cal_w_en(Xi,Xj,delta_en):
    w_en = 1+delta_en*np.sqrt(np.abs(Xi-Xj))/3.3
    return w_en

def cal_w_ij(vor,dc,en):
    w_ij = vor*dc*en/max(vor,dc,en)
    return w_ij

def cal_AUC(x1,x2):
    def F(x):
        y = np.sqrt((-x**2+2*x))/2*(x-1-2*np.arctanh(np.sqrt((x/(x-2))))/np.sqrt((x**2-2*x)))
        return y
    if x1 < 0 or x2 < 0 or x1 > 1 or x2 > 1:
        raise ValueError("x1 和 x2 必须在 [0, 1] 范围内")
    # 计算 AUC
    auc = np.abs(F(x1)-F(x2))
    return auc
def cal_max_bonds(s_w:dict):
    indexlist = list(s_w.keys())
    weight = list(s_w.values())
    auc_dict = {}
    for i in range(len(weight)-1):
        auc = cal_AUC(weight[i],weight[i+1])
        auc_dict[f'{indexlist[i]}']=auc
    print(auc_dict)
    max_auc_key = max(auc_dict.items(), key=lambda item: item[1])[0]
    j_max_w = max_auc_key
    index = indexlist.index(int(j_max_w))
    bond_j = indexlist[:index+1]
    return bond_j

class CrystalNN:
    def __init__(self, delta_low=0.5, delta_high=1.0, delta_en=3.0):
        self.delta_low = delta_low  # 低距离截断参数 (Å)
        self.delta_high = delta_high  # 高距离截断参数 (Å)
        self.delta_en = delta_en  # 电负性权重参数
        self.get_w_list = {}
    def get_bonded(self,structure: Structure,i: int) -> dict:
        """
        计算晶体结构中每个原子的配位数
        :param structure: pymatgen的Structure对象
        :param n: 中心原子的索引
        :return: 字典，包含配位原子索引和权重
        """
        # 获取Voronoi分解结果
        vnn = VoronoiNN(allow_pathological=True)
        voro = vnn.get_voronoi_polyhedra(structure, i)#{n:{'site', 'normal', 'solid_angle', 'volume', 'face_dist', 'area', 'n_verts', 'verts', 'adj_neighbors'}}
        weights = {}
        for site_n in voro:
        
            #Voronoi权重
            face_dict = voro[site_n]
            w_vor = cal_w_vor(face_dict)
            # 距离权重
            neighborSite = face_dict['site']
            j = neighborSite.index
            d_ij = structure.get_distance(i,j)
            r_i = Element(structure[i].species_string).atomic_radius
            r_j = Element(structure[j].species_string).atomic_radius
            d_low = r_i + r_j + self.delta_low
            d_high = r_i + r_j + self.delta_high
            w_dc = cal_w_dc(d_ij,d_low,d_high)
            
            # 电负性权重
            x_i = Element(structure[i].species_string).X
            x_j = Element(structure[j].species_string).X
            w_en = cal_w_en(x_i,x_j,self.delta_en)
        
            # 归一化权重
            w_ij = cal_w_ij(w_vor,w_dc,w_en)
            weights[j] = w_ij
        # 按权重排序并计算最大概率配位数
        sorted_weight = dict(sorted(weights.items(), key=operator.itemgetter(1), reverse=True))
        bonded_neighbor_index_list= cal_max_bonds(sorted_weight)
        return bonded_neighbor_index_list


# 示例使用
if __name__ == "__main__":
    # 创建一个简单的面心立方结构
    # 创建 NaCl 晶体结构
    lattice = [[5.69, 0, 0], [0, 5.69, 0], [0, 0, 5.69]]  # 晶格参数
    species = ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"]  # 原子种类
    coords = [  # 分数坐标
        [0, 0, 0],      # Na
        [0.5, 0.5, 0],  # Na
        [0.5, 0, 0.5],  # Na
        [0, 0.5, 0.5],  # Na
        [0.5, 0, 0],    # Cl
        [0, 0.5, 0],    # Cl
        [0, 0, 0.5],    # Cl
        [0.5, 0.5, 0.5] # Cl
    ]
    structure = Structure(lattice, species, coords)
        # 初始化CrystalNN算法
    crystal_nn = CrystalNN()
    #structure = readfile('POSCAR','poscar')
    # 计算第一个原子的配位数
    result = crystal_nn.get_bonded(structure,0)
    print(structure[0].species_string)
    print(result)
    #print(f"配位数: {result['coordination_number']}")
    #print(f"权重: {result['weights']}")