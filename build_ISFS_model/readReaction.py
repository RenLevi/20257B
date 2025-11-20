from scipy.spatial import cKDTree
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from build_ISFS_model.CheckNN import *
from ase.io import write,read
import copy
import os
import re
from rdkit import Chem
from ase import Atoms
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from collections import defaultdict
#import matplotlib.pyplot as plt
def element_constrained_hungarian(atoms_A, atoms_B):
    """
    使用匈牙利算法匹配两个ASE结构中的原子，确保只有相同元素的原子才会匹配
    
    参数:
    atoms_A, atoms_B: ASE Atoms对象
    
    返回:
    matches: 匹配结果列表，每个元素为字典，包含匹配的原子信息
    total_distance: 总匹配距离
    """
    # 检查原子数量是否相同
    if len(atoms_A) != len(atoms_B):
        raise ValueError("两个结构中的原子数量不同")
    
    # 获取元素符号
    symbols_A = atoms_A.get_chemical_symbols()
    symbols_B = atoms_B.get_chemical_symbols()
    
    # 获取所有唯一的元素类型
    unique_elements = set(symbols_A) | set(symbols_B)
    
    # 按元素类型分组原子
    elements_A = {elem: [] for elem in unique_elements}
    elements_B = {elem: [] for elem in unique_elements}
    
    for i, elem in enumerate(symbols_A):
        elements_A[elem].append(i)
    
    for i, elem in enumerate(symbols_B):
        elements_B[elem].append(i)
    
    # 检查每个元素的原子数量是否相同
    for elem in unique_elements:
        count_A = len(elements_A[elem])
        count_B = len(elements_B[elem])
        if count_A != count_B:
            raise ValueError(f"元素 {elem} 的原子数量不同: {count_A} vs {count_B}")
    
    # 提取所有位置
    positions_A = atoms_A.get_positions()
    positions_B = atoms_B.get_positions()
    
    # 为每个元素类型分别进行匹配
    all_matches = []
    total_distance = 0.0
    
    for elem in unique_elements:
        if not elements_A[elem]:  # 如果没有这种元素，跳过
            continue
            
        # 获取该元素在A和B中的原子索引
        indices_A = elements_A[elem]
        indices_B = elements_B[elem]
        
        # 提取这些原子的位置
        pos_A = positions_A[indices_A]
        pos_B = positions_B[indices_B]
        
        # 计算距离矩阵
        distance_matrix = cdist(pos_A, pos_B, metric='euclidean')
        
        # 使用匈牙利算法找到最优匹配
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        
        # 构建匹配结果
        for i, j in zip(row_ind, col_ind):
            idx_A = indices_A[i]
            idx_B = indices_B[j]
            dist = distance_matrix[i, j]
            
            match_info = {
                'index_A': idx_A,
                'index_B': idx_B,
                'element': elem,
                'distance': dist,
                'position_A': positions_A[idx_A],
                'position_B': positions_B[idx_B]
            }
            all_matches.append(match_info)
            total_distance += dist
    
    # 按A结构的索引排序匹配结果
    all_matches.sort(key=lambda x: x['index_A'])
    
    return all_matches, total_distance
def visualize_matching_results(atoms_A, atoms_B, matches):
    """
    可视化匹配结果
    """
    print(f"结构A: {atoms_A.get_chemical_formula()}")
    print(f"结构B: {atoms_B.get_chemical_formula()}")
    print("\n原子匹配结果:")
    print("A索引  B索引  元素     距离(Å)")
    print("-" * 30)
    
    for match in matches:
        print(f"{match['index_A']:4}  {match['index_B']:4}   {match['element']:2}     {match['distance']:.4f}")
    
    # 计算平均距离
    avg_distance = np.mean([m['distance'] for m in matches])
    print(f"\n平均匹配距离: {avg_distance:.4f} Å")
    
    return avg_distance
def calculate_midpoint(*points):
    """
    计算多个点的几何中心（中点）
    支持二维或三维点
    """
    if not points:
        return None
    
    # 转换为numpy数组
    points_array = np.array(points)
    
    # 计算平均值（中点）
    midpoint = np.mean(points_array, axis=0)
    
    return midpoint
'''查询吸附位点'''
class SurfaceSiteFinder:
    def __init__(self, atoms: Atoms, surface_direction: int = 2):
        """
        初始化表面位点查找器
        
        参数:
        atoms: ASE Atoms 对象，表示平板结构
        surface_direction: 表面法线方向 (0=x, 1=y, 2=z)
        """
        self.atoms = atoms
        self.surface_direction = surface_direction
        self.grid_points = None
        self.wrapped_points = None
        self.site_atoms = defaultdict(list)
        self.site_positions = {}
        self.site_special_vectors = {}
        self.site_vectors = {}
        self.cell = self.atoms.get_cell()
        self.pbc = self.atoms.get_pbc()
        self._generate_replicas()
    def _generate_replicas(self):
        """生成必要的镜像原子以处理周期性边界条件"""
        # 确定每个方向需要复制的数量
        # 对于最近邻搜索，通常只需要相邻的镜像
        replicas = []
        for i, pbc in enumerate(self.pbc):
            if pbc:
                replicas.append([-1, 0, 1])
            else:
                replicas.append([0])
        
        # 生成所有可能的复制组合
        replica_offsets = np.array(np.meshgrid(*replicas)).T.reshape(-1, 3)
        
        # 存储所有位置（原始+镜像）
        self.all_positions = []
        self.original_indices = []  # 记录每个位置对应的原始原子索引
        
        # 原始原子位置
        original_positions = self.atoms.get_positions()
        for i, pos in enumerate(original_positions):
            self.all_positions.append(pos)
            self.original_indices.append(i)
        # 镜像原子位置
        for offset in replica_offsets:
            # 跳过零偏移（原始位置）
            if np.all(offset == 0):
                continue
            # 应用周期性偏移
            offset_positions = original_positions + offset @ self.cell
            
            for i, pos in enumerate(offset_positions):
                self.all_positions.append(pos)
                self.original_indices.append(i)
        
        self.all_positions = np.array(self.all_positions)
        self.original_indices = np.array(self.original_indices)
    def create_grid(self, grid_spacing: float = 0.1, height_above_surface: float = 5.0):
        """
        在表面上创建密集网格
        
        参数:
        grid_spacing: 网格点间距 (Å)
        height_above_surface: 网格在表面上方的初始高度 (Å)
        """
        # 获取表面原子的坐标
        positions = self.atoms.get_positions()
        
        # 确定表面方向
        if self.surface_direction == 0:  # x方向为表面法线
            surface_coords = positions[:, 1:]
            max_height = np.max(positions[:, 0])
        elif self.surface_direction == 1:  # y方向为表面法线
            surface_coords = positions[:, [0, 2]]
            max_height = np.max(positions[:, 1])
        else:  # z方向为表面法线 (默认)
            surface_coords = positions[:, :2]
            max_height = np.max(positions[:, 2])
        
        # 确定网格的边界
        x_min, y_min = np.min(surface_coords, axis=0)
        x_max, y_max = np.max(surface_coords, axis=0)
        
        # 扩展边界以确保覆盖整个表面
        x_min, x_max = x_min - 2.0, x_max + 2.0
        y_min, y_max = y_min - 2.0, y_max + 2.0
        
        # 创建网格点
        x_grid = np.arange(x_min, x_max, grid_spacing)
        y_grid = np.arange(y_min, y_max, grid_spacing)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        # 设置网格高度
        if self.surface_direction == 0:
            zz = np.full_like(xx, max_height + height_above_surface)
            self.grid_points = np.vstack([zz.ravel(), xx.ravel(), yy.ravel()]).T
        elif self.surface_direction == 1:
            zz = np.full_like(xx, max_height + height_above_surface)
            self.grid_points = np.vstack([xx.ravel(), zz.ravel(), yy.ravel()]).T
        else:
            zz = np.full_like(xx, max_height + height_above_surface)
            self.grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
            
        return self.grid_points
    def wrap_grid_to_surface(self, contact_distance: float = 2.0, step_size: float = 0.1,height_above_surface=3.0):
        """
        将网格点向表面平移，直到接近原子
        
        参数:
        contact_distance: 接触距离阈值 (Å)
        step_size: 平移步长 (Å)
        """
        if self.grid_points is None:
            raise ValueError("请先创建网格点")
            
        # 创建原子位置的KD树用于快速最近邻搜索
        atom_positions = self.all_positions
        tree = cKDTree(atom_positions)
        
        # 初始化包裹后的点
        wrapped_points = self.grid_points.copy()
        
        # 确定平移方向
        if self.surface_direction == 0:
            direction = np.array([-1, 0, 0])
        elif self.surface_direction == 1:
            direction = np.array([0, -1, 0])
        else:
            direction = np.array([0, 0, -1])
            
        # 逐步平移网格点
        max_steps = int(height_above_surface / step_size) + 10
        for step in range(max_steps):
            # 计算每个点到最近原子的距离
            distances, indices = tree.query(wrapped_points)
            
            # 找到尚未接触原子的点
            not_contacted = distances > contact_distance
            
            if not np.any(not_contacted):
                break
                
            # 将这些点向表面方向移动
            wrapped_points[not_contacted] += direction * step_size
        
        self.wrapped_points = wrapped_points
        return wrapped_points   
    def find_sites(self, contact_distance: float = 2.0, multi_site_threshold: float = 2):
        """
        识别表面位点
        
        参数:
        contact_distance: 接触距离阈值 (Å)
        multi_site_threshold: 多重位点识别阈值 (Å)
        """
        if self.wrapped_points is None:
            raise ValueError("请先执行网格包裹")
            
        # 创建原子位置的KD树
        atom_positions = self.all_positions
        tree = cKDTree(atom_positions)
        
        # 对于每个包裹后的网格点，找到接触的原子
        for i, point in enumerate(self.wrapped_points):
            # 找到距离此点在一定范围内的所有原子
            indices = tree.query_ball_point(point, contact_distance)
            
            if indices:
                # 将原子索引转换为可哈希的元组
                atom_tuple = tuple(sorted(indices))
                self.site_atoms[atom_tuple].append(point)
        
        # 识别位点类型并计算位点位置
        for atom_indices, points in self.site_atoms.items():
            if len(atom_indices) == 1:
                # 顶位 - 使用原子位置
                atom_idx = atom_indices[0]
                self.site_positions[atom_indices] = atom_positions[atom_idx]
                self.site_special_vectors[atom_indices]=None
            elif len(atom_indices) == 2:
                #桥位
                site_atoms = atom_positions[list(atom_indices)]
                self.site_positions[atom_indices] = np.mean(site_atoms, axis=0)
                self.site_special_vectors[atom_indices]=(site_atoms[-1]-site_atoms[0])/np.linalg.norm(site_atoms[-1]-site_atoms[0])
            else:
                # 桥位或多重位点 - 使用原子位置的平均值
                site_atoms = atom_positions[list(atom_indices)]
                self.site_positions[atom_indices] = np.mean(site_atoms, axis=0)
                self.site_special_vectors[atom_indices]=None
        
        return self.site_atoms, self.site_positions,self.site_special_vectors
    def classify_sites(self, multi_site_threshold: float = 2):
        """
        分类位点类型
        
        参数:
        multi_site_threshold: 多重位点识别阈值 (Å)
        """
        site_types = {}
        
        for atom_indices in self.site_atoms.keys():
            if len(atom_indices) == 1:
                site_types[atom_indices] = "top"
            elif len(atom_indices) == 2:
                site_types[atom_indices] = "bridge"
            else:
                # 检查是否构成多重位点
                atom_positions = self.all_positions[list(atom_indices)]
                centroid = np.mean(atom_positions, axis=0)
                
                # 计算原子到质心的最大距离
                max_distance = np.max(np.linalg.norm(atom_positions - centroid, axis=1))
                
                if max_distance < multi_site_threshold:
                    site_types[atom_indices] = f"{len(atom_indices)}th_multifold"
                else:
                    site_types[atom_indices] = "complex"
        
        return site_types
    def visualize(self, show_grid: bool = False, show_wrapped: bool = True):
        """
        可视化结果
        
        参数:
        show_grid: 是否显示初始网格
        show_wrapped: 是否显示包裹后的网格
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制原子
        positions = self.atoms.get_positions()
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='blue', s=150, label='atom')
        
        # 绘制初始网格点（如果要求）
        if show_grid and self.grid_points is not None:
            ax.scatter(self.grid_points[:, 0], self.grid_points[:, 1], self.grid_points[:, 2],
                      c='gray', s=5, alpha=0.3, label='initial grid')
        
        # 绘制包裹后的网格点（如果要求）
        if show_wrapped and self.wrapped_points is not None:
            ax.scatter(self.wrapped_points[:, 0], self.wrapped_points[:, 1], self.wrapped_points[:, 2],
                      c='lightgreen', s=5, alpha=0.5, label='warpped grid')
        
        # 绘制位点位置
        colors = ['red', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        color_idx = 0
        
        for atom_indices, position in self.site_positions.items():
            site_type = self.classify_sites().get(atom_indices, "未知")
            
            if site_type == "top":
                color = 'red'
                marker = 'o'
                size = 50
                alpha = 1
            elif site_type == "bridge":
                color = 'orange'
                marker = 's'
                size = 40
                alpha = 1
            elif "multifold" in site_type:
                '''color = colors[color_idx % len(colors)]
                color_idx += 1'''
                color = 'cyan'
                marker = 'D'
                size = 30
                alpha = 1
            else:
                color = 'gray'
                marker = 'x'
                size = 20
                alpha = 1
            
            ax.scatter(position[0], position[1], position[2], 
                      c=color, marker=marker, s=size, label=site_type,alpha = alpha)
            
            # 绘制向量（从位点到网格点平均位置）
            '''if atom_indices in self.site_vectors:
                vector = self.site_vectors[atom_indices]
                ax.quiver(position[0], position[1], position[2],
                         vector[0]-position[0], vector[1]-position[1], vector[2]-position[2],
                         color=color, arrow_length_ratio=0.1)'''
        
        # 设置图表属性
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title('sites of surface')
        
        # 避免重复的图例标签
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()
def find_nearest_point_kdtree(points, target_point):
    """
    使用KD树找到最近的点
    """
    # 构建KD树
    tree = cKDTree(points)
    
    # 查询最近的点
    distance, index = tree.query(target_point.reshape(1, -1))
    
    nearest_point = points[index[0]]
    
    return nearest_point, distance[0], index[0]
class DistanceQuery:
    def __init__(self, points):
        """
        初始化KDTree索引
        """
        self.points = np.array(points)
        self.tree = cKDTree(self.points)
    
    def find_points_at_distance(self, query_point, target_distance, tolerance=0.1):
        """
        使用KDTree高效查找特定距离的点
        
        策略：先找到半径范围内的点，再精确筛选
        """
        # 搜索半径范围 [target_distance - tolerance, target_distance + tolerance]
        min_dist = max(0, target_distance - tolerance)
        max_dist = target_distance + tolerance
        
        # 使用query_ball_point找到范围内的点
        indices = self.tree.query_ball_point(query_point, max_dist)
        
        # 精确筛选
        result_indices = []
        for idx in indices:
            dist = np.linalg.norm(self.points[idx] - query_point)
            if abs(dist - target_distance) < tolerance:
                result_indices.append(idx)
        
        return result_indices
    
    def find_all_distances(self, query_point):
        """
        返回所有点到查询点的距离，便于后续分析
        """
        return np.linalg.norm(self.points - query_point, axis=1)
def find_site(atoms,adsatom:list,finder:SurfaceSiteFinder): 
    out = []
    sites, positions,special_vector = finder.find_sites(contact_distance=2.3)
    site_types = finder.classify_sites(multi_site_threshold=2)
    site_positions_list =[]
    atom_indices_list = []
    for i in positions:
        site_positions_list.append(positions[i])
        atom_indices_list.append(i)
    for adsA in adsatom:
        tp = atoms.positions[adsA.id]
        nearest, distance, idx = find_nearest_point_kdtree(np.array(site_positions_list),tp)
        vector = tp - nearest
        atom_indices = atom_indices_list[idx]
        site_type = site_types[atom_indices]
        out.append([nearest, distance,adsA.id,atom_indices,site_type, vector])
    return out
def svd_rotation_matrix(a, b):
    """
    使用SVD分解计算旋转矩阵
    
    参数:
    a, b: 三维单位向量 (numpy数组)
    
    返回:
    R: 3x3旋转矩阵
    """
    # 确保输入是单位向量
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    
    # 计算协方差矩阵
    H = np.outer(a, b)
    
    # SVD分解
    U, S, Vt = np.linalg.svd(H)
    
    # 计算旋转矩阵
    R = np.dot(Vt.T, U.T)
    
    # 处理反射情况
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    return R
def Euclidean_distance(R1:Atoms, R2:Atoms):
    """
    计算两组结构的欧氏距离
    """
    p1 = R1.get_positions()
    p2 = R2.get_positions()
    assert len(R1) == len(R2)
    return np.sqrt(np.sum((p1 - p2) ** 2))
def cal_dist_between_nobondedatoms(o1p,a2):
    d = 0
    for atom2 in a2:
        d+=np.linalg.norm(atom2-o1p)
    return d
def find_min_sum_distance_point_vectorized(points, point_a, point_b):
    """
    使用向量化计算找到到两点距离和最小的点（更高效）
    
    参数:
    points: 三维点集，形状为(n, 3)的numpy数组
    point_a, point_b: 两个已知的三维点
    
    返回:
    min_point: 到两点距离和最小的点
    min_distance: 最小的距离和
    min_index: 最小点在点集中的索引
    """
    # 向量化计算所有点到point_a和point_b的距离
    dist_a = np.linalg.norm(points - point_a, axis=1)
    dist_b = np.linalg.norm(points - point_b, axis=1)
    total_dists = dist_a + dist_b
    
    # 找到最小值的索引
    min_index = np.argmin(total_dists)
    min_point = points[min_index]
    min_distance = total_dists[min_index]
    
    return min_point, min_distance, min_index
def find_max_sum_distance_point_vectorized(points, point_a, point_b):
    """
    使用向量化计算找到到两点距离和最大的点（更高效）
    
    参数:
    points: 三维点集，形状为(n, 3)的numpy数组
    point_a, point_b: 两个已知的三维点
    
    返回:
    max_point: 到两点距离和最大的点
    max_distance: 最大的距离和
    max_index: 最大点在点集中的索引
    """
    # 向量化计算所有点到point_a和point_b的距离
    dist_a = np.linalg.norm(points - point_a, axis=1)
    dist_b = np.linalg.norm(points - point_b, axis=1)
    total_dists = dist_a + dist_b
    
    # 找到最小值的索引
    max_index = np.argmax(total_dists)
    max_point = points[max_index]
    max_distance = total_dists[max_index]
    
    return max_point, max_distance, max_index
def select_site_with_max_dist(result_idxlist,base_mol,points,centeridx):
    sitepositions0=[]
    idxlist = []
    total_dists = np.zeros((1,len(result_idxlist)))
    for idx in result_idxlist:
        sitepositions0.append(points[idx])
        idxlist.append(idx)
    sitepositions=np.array(sitepositions0)
    for a in base_mol:
        if a.index == centeridx :
            point_a = a.position
            dist_a = np.linalg.norm(sitepositions - point_a,axis=1)
            total_dists = dist_a-dist_a
    for a in base_mol:
        if a.index != centeridx and a.symbol != 'Ru':
            point_a = a.position
            dist_a = np.linalg.norm(sitepositions - point_a,axis=1)
            total_dists += dist_a
        
    if np.all(total_dists == 0):
        max_index = 0
        max_point = sitepositions0[max_index]
    else:
        max_index = np.argmax(total_dists)
        max_point = sitepositions0[max_index]
        max_distance = total_dists[0][max_index]
    return idxlist[max_index],max_point
'''查询成键反应原子对'''
def str2list(reaction:str):
    r1 = reaction.split(">")
    r2 = []
    for i in r1:
        i1 = i.split()
        r2.append(i1)
    return r2
def add_brackets_around_letters(cnmol:str):# 使用正则表达式替换不在[]中的字母字符，前后添加[]:example:[H]CO==>[H][C][O]
    result = re.sub(r'(?<!\[)([a-zA-Z])(?!\])', r'[\g<1>]', cnmol)
    return result
def checkbond_a3(reaction:list,a1,a3):
    mol1 = a1.bms.mol
    mol2 =a3.bms.mol
    bms1 = a1.bms
    bms2 = a3.bms
    reactiontype = reaction[1][0]
    clean_list_reaction =[]
    for id in range(len(reaction[1])):
        if bool(re.search(r'\d', reaction[1][id])) == False:
            clean_list_reaction.append(reaction[1][id])
    addatom = clean_list_reaction[1]#reaction[1][-3]
    bondedatom = clean_list_reaction[-1]#reaction[1][-1]
    def COMBINE(mol,add):
        # 创建分子时禁用化合价检查
        params = Chem.SmilesParserParams()
        params.removeHs = False  # 不自动移除氢原子
        params.sanitize = False  # 禁用所有检查
        addmol = Chem.MolFromSmiles(add_brackets_around_letters(add),params=params)
        combined_mol = rdmolops.CombineMols(mol, addmol)
        return combined_mol
    def addATOM():
        if reactiontype == 'Add':
            if len(addatom) > 1 and 'C' in addatom:
                return 'C'
            elif addatom =='OH':
                return 'O'
            else:
                return addatom
        else:
            if addatom == 'O/OH':
                return 'O'
            else:
                return addatom
    def warp(cs12,add=addatom):
        check_mol = copy.deepcopy(cs12[1])
        check_mol = COMBINE(check_mol,add)
        bonds = cs12[0].GetBonds()
        AA=addATOM()
        aset = {AA,bondedatom}
        print(aset)
        check=100
        outlist = [None,None,None,None]
        for bond in bonds:
            mol = copy.deepcopy(cs12[0])
            begin_atom_id = bond.GetBeginAtomIdx()
            end_atom_id = bond.GetEndAtomIdx()
            begin_atom = mol.GetAtomWithIdx(begin_atom_id)
            end_atom = mol.GetAtomWithIdx(end_atom_id)
            qset = {begin_atom.GetSymbol(),end_atom.GetSymbol()}
            bms = cs12[2]
            cb = bms.cb
            if qset == aset:
                mol.RemoveBond(begin_atom_id, end_atom_id)
                if subHH(Chem.MolToSmiles(mol)) == Chem.MolToSmiles(check_mol):
                    Bid=begin_atom.GetIdx()+bms.metal
                    Eid=end_atom.GetIdx()+bms.metal
                    a1 = cb.atoms[Bid]
                    a2 = cb.atoms[Eid]
                    p1 = a1.xyz
                    p2 = a2.xyz
                    sumZ=p1[2]+p2[2]
                    #print(sumZ,check)
                    if sumZ<=check:
                        check = sumZ
                        outlist = [begin_atom.GetIdx(),end_atom.GetIdx(),Chem.MolToSmiles(cs12[0]),Chem.MolToSmiles(check_mol)]
                    else:
                        check = check
        return outlist[0],outlist[1],outlist[2],outlist[3]
    if reactiontype == 'Add':
        cs12 = (mol2,mol1,bms2,bms1)
        return warp(cs12)
    elif reactiontype == 'Remove':
        cs12 = (mol1,mol2,bms1,bms2)
        if addatom == 'O/OH':
            o1,o2,o3,o4 = warp(cs12,add='O')
            if o1 == None and o2 == None and o3 == None and o4 == None:
                return warp(cs12,add='OH')
            else:
                return o1,o2,o3,o4
        else:
            return warp(cs12)        
def checkbond_a1a2(reaction:list,a1,a2,a3):
    mol1 = a1.bms.mol
    mol2 = a2.bms.mol
    mol3 = a3.bms.mol
    reactiontype = reaction[1][0]
    clean_list_reaction =[]
    for id in range(len(reaction[1])):
        if bool(re.search(r'\d', reaction[1][id])) == False:
            clean_list_reaction.append(reaction[1][id])
    addatom = clean_list_reaction[1]#reaction[1][-3]
    bondedatom = clean_list_reaction[-1]#reaction[1][-1]
    def addATOM():
        if reactiontype == 'Add':
            if len(addatom) > 1 and 'C' in addatom:
                return 'C'
            elif addatom =='OH':
                return 'O'
            else:
                return addatom
        else:
            if addatom == 'O/OH':
                return 'O'
            else:
                return addatom
    AA = addATOM()
    aset = {AA,bondedatom}
    adsAtomsIna1 = a1.ads
    indices_to_mol = [atom.index for atom in a1.atoms if atom.symbol != 'Ru']
    adsAtomsIna2 = a2.ads
    for adsA1 in adsAtomsIna1:
        for adsA2 in adsAtomsIna2:
            qset = {adsA1.elesymbol,adsA2.elesymbol}
            if qset == aset:
                mol_broken = rdmolops.CombineMols(mol1,mol2)
                rwmol = Chem.RWMol(mol_broken)
                ida1 = adsA1.id-64
                ida2 = adsA2.id-64+mol1.GetNumAtoms()
                print(ida1,ida2,adsA1.id,adsA2.id)
                rwmol.AddBond(ida1, ida2, Chem.BondType.SINGLE)
                if Chem.MolToSmiles(rwmol) == Chem.MolToSmiles(mol3):
                    return adsA1.id,adsA2.id,'ad2ad',(ida1,ida2)
                else:pass
    for A1idx in indices_to_mol:
        A1 = a1.cb.atoms[A1idx]
        for adsA2 in adsAtomsIna2:
            qset = {A1.elesymbol,adsA2.elesymbol}
            if qset == aset:
                mol_broken = rdmolops.CombineMols(mol1,mol2)
                rwmol = Chem.RWMol(mol_broken)
                ida1 = A1.id-64
                ida2 = adsA2.id-64+mol1.GetNumAtoms()
                rwmol.AddBond(ida1, ida2, Chem.BondType.SINGLE)
                if Chem.MolToSmiles(rwmol) == Chem.MolToSmiles(mol3):
                    return A1.id,adsA2.id,'Nad2ad',(ida1,ida2)
                else:pass
    return False,False,False,False
class NN_system():
    def __init__(self):
        self.cb = None
        self.bms = None
        self.ads = None
        self.only_mol = None
        self.ads_data = None
        self.atoms = None
    def RunCheckNN_FindSite(self,file,finder):
        print(file)
        cb = checkBonds()
        if type(file) == str:
            cb.input(file)
        else:cb.poscar=file
        cb.AddAtoms()
        cb.CheckAllBonds()
        bms=BuildMol2Smiles(cb)
        bms.build()
        self.cb = cb 
        self.bms = bms
        self.ads = cb.adsorption
        atoms = cb.poscar
        self.atoms = atoms
        indices_to_mol = [atom.index for atom in atoms if atom.symbol != 'Ru']
        self.only_mol = atoms[indices_to_mol]
        self.ads_data = find_site(cb.poscar,cb.adsorption,finder)
        print(self.ads_data)
        return self
        
"""
1.复数吸附位点吸附的中间体在催化剂表面难以发生迁移
2.复数吸附的中间体基元反应前后整体质心移动幅度小
#3.化学键的形成与断裂发生在吸附原子之间,至少有一个吸附原子
"""

class STARTfromBROKENtoBONDED():
    def __init__(self,atoms1,atoms2,atoms3):
        self.atoms = (atoms1,atoms2,atoms3)
        '''
        a1(big)+a2(small)=a3(bigger)
        '''
    def site_finder(self,slab):
        self.slab = slab
        finder = SurfaceSiteFinder(slab)
        # 创建网格
        grid_points = finder.create_grid(grid_spacing=0.1, height_above_surface=3.0)
        # 包裹网格到表面
        wrapped_points = finder.wrap_grid_to_surface(contact_distance=2, step_size=0.1,height_above_surface=3.0)
        # 查找位点
        sites, positions,special_vector = finder.find_sites(contact_distance=2)
        # 分类位点
        site_types = finder.classify_sites(multi_site_threshold=2)
        self.site = finder
        self.site_types = site_types
        self.site_positions = positions
        self.special_vectors = special_vector
        return self.site
    def run(self,reaction:str):
        print(reaction)
        self.r = str2list(reaction)
        (atoms1,atoms2,atoms3)=self.atoms
        a1,a2,a3= NN_system(),NN_system(),NN_system()
        a1.RunCheckNN_FindSite(atoms1,self.site)
        a2.RunCheckNN_FindSite(atoms2,self.site)
        a3.RunCheckNN_FindSite(atoms3,self.site)
        top = {}
        bridge = {}
        hcc = {}
        for atom_indices, site_type in self.site_types.items():
            if site_type == 'top':
                top[atom_indices]=self.site_positions[atom_indices]
            elif site_type == 'bridge':
                bridge[atom_indices]=self.site_positions[atom_indices]
            else:
                hcc[atom_indices]=self.site_positions[atom_indices]
        def warp(rl,a1,a2,a3):
            o1,o2,self.tf,ids_mol = checkbond_a1a2(rl,a1,a2,a3)#id in atoms
            bid_mol,eid_mol,_,_ = checkbond_a3(rl,a1,a3)
            if o1 == False or o2 == False:
                return False,False
            base_mol = a1.atoms
            total_atoms = len(base_mol)
            mola2 = copy.deepcopy(a2.only_mol)#ase atoms
            topsitepl = list(top.values())
            topsitekl = list(top.keys())
            bridegsitepl = list(bridge.values())
            bridgesitekl = list(bridge.keys())
            hccsitepl = list(hcc.values())
            hccsitekl = list(hcc.keys())
            if len(a2.ads_data) == 1:
                a2_ads_data0 = a2.ads_data[0]#[[nearest, distance,adsA.id,atom_indices,site_type, vector]]
                a2zv=a2_ads_data0[-1]#vector
                a2st=a2_ads_data0[-2]#site_type
                a2ai=a2_ads_data0[-3]#atom_indices
                a2sp=self.site_positions[a2ai]
                a2spv= self.special_vectors[a2ai]
                if a2st == 'top':
                    bap_a1 = base_mol[o1].position
                    DQ = DistanceQuery(topsitepl)
                    result_idxlist = DQ.find_points_at_distance(query_point=bap_a1,target_distance=3,tolerance=0.5)
                    _,sp4a2 = select_site_with_max_dist(result_idxlist,base_mol,topsitepl,o1)
                    print('top')
                    print(bap_a1)
                    print(sp4a2)
                    v_trans = sp4a2-a2sp
                    mola2.positions += v_trans
                    a1a2sys = base_mol+mola2
                elif a2st == 'bridge':
                    bap_a1 = base_mol[o1].position
                    DQ = DistanceQuery(bridegsitepl)
                    result_idxlist = DQ.find_points_at_distance(query_point=bap_a1,target_distance=3,tolerance=0.5)
                    _,sp4a2 = select_site_with_max_dist(result_idxlist,base_mol,bridegsitepl,o1)
                    print('bridge')
                    print(bap_a1)
                    print(sp4a2)
                    sai = bridgesitekl[result_idxlist[0]]
                    spv = self.special_vectors[sai]
                    v_trans = sp4a2-a2sp
                    R = svd_rotation_matrix(a2spv,spv)
                    for a in mola2:
                        a.position = np.dot(R,a.position)
                    mola2.positions += v_trans
                    a1a2sys = base_mol+mola2
                else:
                    bap_a1 = base_mol[o1].position
                    DQ = DistanceQuery(hccsitepl)
                    result_idxlist = DQ.find_points_at_distance(query_point=bap_a1,target_distance=3,tolerance=0.5)
                    _,sp4a2 = select_site_with_max_dist(result_idxlist,base_mol,hccsitepl,o1)
                    print('hcc')
                    print(bap_a1)
                    print(sp4a2)
                    v_trans = sp4a2-a2sp
                    mola2.positions += v_trans
                    a1a2sys = base_mol+mola2
            elif len(a2.ads_data) > 1:
                bap_a1 = base_mol[o1].position
                a2_ads_data = a2.ads_data
                site1 = a2_ads_data[0]
                site2 = a2_ads_data[-1]#[nearest, distance,adsA.id,atom_indices,site_type, vector]
                v21 = site1[0]- site2[0]
                distsite12=np.linalg.norm(site1[0]- site2[0])
                sitepldict = {'top':topsitepl,'bridge':bridegsitepl,'3th_multifold':hccsitepl}
                DQ4site1 = DistanceQuery(sitepldict[site1[-2]])
                result_idxlist = DQ4site1.find_points_at_distance(query_point=bap_a1,target_distance=3,tolerance=1)
                _,max_point1 = select_site_with_max_dist(result_idxlist,base_mol,sitepldict[site1[-2]],o1)
                DQ4site2 = DistanceQuery(sitepldict[site2[-2]])
                site2_idxlist=DQ4site2.find_points_at_distance(max_point1,distsite12,tolerance=0.5)
                v_trans = max_point1-site1[0]
                assert len(site2_idxlist) != 0
                nobondedatomdist = []
                for id  in site2_idxlist:
                    point2 = sitepldict[site2[-2]][id]
                    v_21 = point2-min_point1
                    R = svd_rotation_matrix(v21,v_21)
                    mola2 = copy.deepcopy(a2.only_mol)
                    for a in mola2:
                        a.position = np.dot(R,a.position)
                    mola2.positions += v_trans
                    nbad_info = {
                    'id':id,
                    'point2':point2,
                    'SUM_dist':cal_dist_between_nobondedatoms(base_mol[o1].position,mola2)
                    }
                    nobondedatomdist.append(nbad_info)
                euolist.sort(key=lambda x: x['SUM_dist'])
                max_point2 = euolist[-1]['point2']
                v_21 = max_point2-max_point1
                R = svd_rotation_matrix(v21,v_21)
                mola2 = copy.deepcopy(a2.only_mol)
                indices_to_mol = [atom.index for atom in a3sys if atom.symbol != 'Ru']
                for a in mola2:
                    a.position = np.dot(R,a.position)
                mola2.positions += v_trans
                a1a2sys = base_mol+mola2
            def get_atom_info(atoms, index):
                """获取单个原子的位置和元素符号"""
                position = copy.deepcopy(atoms.positions[index])  # 原子位置 (x, y, z)
                symbol = copy.deepcopy(atoms.symbols[index])      # 元素符号
                return position, symbol
            indices_to_mol = [atom.index for atom in a1a2sys if atom.symbol != 'Ru']
            a1a2sys_only_mol = copy.deepcopy(a1a2sys[indices_to_mol])
            a3_only_mol = copy.deepcopy(a3.only_mol)
            beginPOS,beginSYM=get_atom_info(a3_only_mol,bid_mol)
            endPOS,endSYM=get_atom_info(a3_only_mol,eid_mol)
            a3_center = calculate_midpoint(beginPOS,endPOS)
            mainPOS,mainSYM=get_atom_info(a1a2sys_only_mol,ids_mol[0])
            subPOS,subSYM=get_atom_info(a1a2sys_only_mol,ids_mol[-1])
            a1a2_center = calculate_midpoint(mainPOS,subPOS)
            v_core_a1a2=mainPOS-subPOS
            if beginSYM == mainSYM:
                v_core_a3 = beginPOS-endPOS
            else:
                v_core_a3 = endPOS-beginPOS
            #移动a3中的分子
            v_trans = a1a2_center-a3_center
            R = svd_rotation_matrix(v_core_a3,v_core_a1a2)
            a3_only_mol.positions+=v_trans
            for a in a3_only_mol:
                a.position = np.dot(R,a.position)
            matches, total_dist = element_constrained_hungarian(a3_only_mol, a1a2sys_only_mol)
            '''
            match_info = {
                'index_A': idx_A,
                'index_B': idx_B,
                'element': elem,
                'distance': dist,
                'position_A': positions_A[idx_A],
                'position_B': positions_B[idx_B]
            }
            按A结构的索引排序匹配结果
            '''
            new_order = []
            for match_info in matches:
                new_order.append(match_info['index_B'])
            a1a2sys_only_mol = a1a2sys_only_mol[new_order]
            self.group2 = self.slab+a1a2sys_only_mol
            
            a3_ads_data = a3.ads_data
            if len(a3_ads_data) == 1:
                [nearest, distance,adsAid,atom_indices,site_type, vector] = a3_ads_data[0]
                if site_type != 'bridge':
                    if site_type == 'top':
                        sitepl = topsitepl
                    else:
                        sitepl = hccsitepl
                    min_point, _, _ = find_min_sum_distance_point_vectorized(sitepl,mainPOS,subPOS)
                    v_trans = min_point - nearest
                    R = svd_rotation_matrix(v_core_a3,v_core_a1a2)
                    a3sys = copy.deepcopy(a3.atoms)
                    indices_to_mol = [atom.index for atom in a3sys if atom.symbol != 'Ru']
                    for id in indices_to_mol:
                        a3sys[id].position += v_trans
                        a3sys[id].position = np.dot(R,a3sys[id].position)
                else:
                    sitepl = bridegsitepl
                    a3spv = self.special_vectors[atom_indices]
                    min_point, _, min_index = find_min_sum_distance_point_vectorized(sitepl,mainPOS,subPOS)
                    sai = bridgesitekl[min_index]
                    spv = self.special_vectors[sai]
                    v_trans = min_point - nearest
                    R1 = svd_rotation_matrix(a3spv,spv)
                    a3sys = copy.deepcopy(a3.atoms)
                    indices_to_mol = [atom.index for atom in a3sys if atom.symbol != 'Ru']
                    for id in indices_to_mol:
                        a3sys[id].position += v_trans
                        a3sys[id].position = np.dot(R1,a3sys[id].position)
                    eou1 =Euclidean_distance(a3sys,self.group2)
                    R2 = svd_rotation_matrix(a3spv,-spv)
                    a3sys = copy.deepcopy(a3.atoms)
                    indices_to_mol = [atom.index for atom in a3sys if atom.symbol != 'Ru']
                    for id in indices_to_mol:
                        a3sys[id].position += v_trans
                        a3sys[id].position = np.dot(R2,a3sys[id].position)
                    eou2 =Euclidean_distance(a3sys,self.group2)
                    if eou1 <= eou2:
                        a3sys = copy.deepcopy(a3.atoms)
                        indices_to_mol = [atom.index for atom in a3sys if atom.symbol != 'Ru']
                        for id in indices_to_mol:
                            a3sys[id].position += v_trans
                            a3sys[id].position = np.dot(R1,a3sys[id].position)
                    else:
                        a3sys = copy.deepcopy(a3.atoms)
                        indices_to_mol = [atom.index for atom in a3sys if atom.symbol != 'Ru']
                        for id in indices_to_mol:
                            a3sys[id].position += v_trans
                            a3sys[id].position = np.dot(R2,a3sys[id].position)
            elif len(a3_ads_data) < 1:
                a3sys = copy.deepcopy(a3.atoms)
                indices_to_mol = [atom.index for atom in a3sys if atom.symbol != 'Ru']
                v_trans = v_core_a1a2-v_core_a3
                R = svd_rotation_matrix(v_core_a3,v_core_a1a2)
                for id in indices_to_mol:
                    a3sys[id].position += v_trans
                    a3sys[id].position = np.dot(R,a3sys[id].position)
            elif len(a3_ads_data) > 1:
                site1 = a3_ads_data[0]
                site2 = a3_ads_data[1]#[nearest, distance,adsA.id,atom_indices,site_type, vector]
                v21 = site1[0]- site2[0]
                distsite12=np.linalg.norm(site1[0]- site2[0])
                sitepldict = {'top':topsitepl,'bridge':bridegsitepl,'3th_multifold':hccsitepl}
                min_point1, _, _ = find_min_sum_distance_point_vectorized(sitepldict[site1[-2]],mainPOS,subPOS)
                v_trans = min_point1-site1[0]
                DQ4site2 = DistanceQuery(sitepldict[site2[-2]])
                site2_idxlist=DQ4site2.find_points_at_distance(min_point1,distsite12,tolerance=1)
                assert len(site2_idxlist) != 0
                euolist = []
                for id  in site2_idxlist:
                    point2 = sitepldict[site2[-2]][id]
                    v_21 = point2-min_point1
                    R = svd_rotation_matrix(v21,v_21)
                    a3sys = copy.deepcopy(a3.atoms)
                    indices_to_mol = [atom.index for atom in a3sys if atom.symbol != 'Ru']
                    for id in indices_to_mol:
                        a3sys[id].position += v_trans
                        a3sys[id].position = np.dot(R,a3sys[id].position)
                    euo_info = {
                    'id':id,
                    'point2':point2,
                    'Euclidean_distance':Euclidean_distance(a3sys,self.group2)
                    }
                    euolist.append(euo_info)
                euolist.sort(key=lambda x: x['Euclidean_distance'])
                min_point2 = euolist[0]['point2']
                v_21 = min_point2-min_point1
                R = svd_rotation_matrix(v21,v_21)
                a3sys = copy.deepcopy(a3.atoms)
                indices_to_mol = [atom.index for atom in a3sys if atom.symbol != 'Ru']
                for id in indices_to_mol:
                    a3sys[id].position += v_trans
                    a3sys[id].position = np.dot(R,a3sys[id].position)
            else:print('怎么可能吸附位点数量同时不满足大于1，等于1，小于1；它是分数吗')
            self.group1 = a3sys
            return self.group2,self.group1
        print(a1.bms.smiles,len(a1.ads_data),a2.bms.smiles,len(a2.ads_data))
        self.IS,self.FS = warp(self.r,a1,a2,a3)
        return self.IS,self.FS
    def save(self,path,format):
        # 保存为POSCAR文件（VASP格式）
        if format=='poscar' or 'POSCAR' or 'vasp':
            if self.IS != False:
                write(path+'IS.vasp', self.IS, format='vasp', vasp5=True)  # vasp5=True添加元素名称
                write(path+'FS.vasp', self.FS, format='vasp', vasp5=True)  # vasp5=True添加元素名称
            else:
                print('the Reaction wrong')
        else:
            print('format should be .vasp')