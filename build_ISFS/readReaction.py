from rdkit import Chem
from rdkit.Chem import rdmolops
from build_ISFS.CheckNN import *
from ase.io import write
import numpy as np
import copy
from ase import Atoms
import re
from ase.optimize import BFGS,FIRE
from rdkit import Chem
from nequip.ase import NequIPCalculator
from ase.data import covalent_radii
from scipy.spatial import cKDTree
from collections import defaultdict
import matplotlib.pyplot as plt
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
                replicas.append([-2, -1, 0, 1, 2])
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
        x_min, x_max = x_min - 8.0, x_max + 8.0
        y_min, y_max = y_min - 8.0, y_max + 8.0
        
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
def find_site(atoms,adsatom:list,finder:SurfaceSiteFinder): 
    out = []
    sites, positions,special_vector = finder.find_sites(contact_distance=2.5)
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
class NN_system():
    def __init__(self):
        self.cb = None
        self.bms = None
        self.ads = None
        self.only_mol = None
        self.ads_data = None
        self.atoms = None
    def RunCheckNN_FindSite(self,file,finder):
        cb = checkBonds()
        if type(file) == str:
            print(file)
            cb.input(file)
        else:cb.poscar=file
        cb.AddAtoms()
        cb.CheckAllBonds()
        bms=BuildMol2Smiles(cb)
        bms.build()
        self.cb = cb 
        self.bms = bms
        self.ads = cb.adsorptAtom
        atoms = cb.poscar
        self.atoms = atoms
        indices_to_mol = [atom.index for atom in atoms if atom.symbol != 'Ru']
        self.only_mol = atoms[indices_to_mol]
        self.ads_data = find_site(cb.poscar,cb.adsorptAtom,finder)
        #print(self.ads_data)
        return self
def check_NON_metal_atoms(atom):
    non_metal_list =[1,2,5,6,7,8,9,10,14,15,16,17,18,33,34,35,36,52,53,54,85,86,117,118]
    if atom.number in non_metal_list:
        return True
    else:
        return False
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
def rotate_point_set(points, A, B, center=None):
    """
    旋转点集使得向量A与向量B同向
    
    参数:
        points: 点集 (n x 3 numpy array)
        A: 原始向量
        B: 目标向量
        center: 旋转中心，如果为None则使用点集中心
    
    返回:
        rotated_points: 旋转后的点集
        rotation_matrix: 使用的旋转矩阵
    """
    # 计算旋转矩阵
    rotation_matrix = svd_rotation_matrix(A,B)#rotate_vector_to_target(A, B)
    
    # 确定旋转中心
    if center is None:
        center = np.mean(points, axis=0)
    
    # 将点集平移到旋转中心，旋转，再平移回去
    translated_points = points - center
    rotated_translated_points = np.dot(translated_points, rotation_matrix.T)
    rotated_points = rotated_translated_points + center
    
    return rotated_points, rotation_matrix
def verify_rotation(original_A, rotated_A, target_B):
    """验证旋转是否正确"""
    original_A_norm = original_A / np.linalg.norm(original_A)
    rotated_A_norm = rotated_A / np.linalg.norm(rotated_A)
    target_B_norm = target_B / np.linalg.norm(target_B)
    
    dot_product = np.dot(rotated_A_norm, target_B_norm)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi
    
    print(f"旋转后向量与目标向量的夹角: {angle:.6f} 度")
    print(f"方向一致性: {dot_product:.10f} (应该接近1.0)")
    
    return abs(dot_product - 1.0) < 1e-10
def add_brackets_around_letters(cnmol:str):# 使用正则表达式替换不在[]中的字母字符，前后添加[]:example:[H]CO==>[H][C][O]
    result = re.sub(r'(?<!\[)([a-zA-Z])(?!\])', r'[\g<1>]', cnmol)
    return result
def subHH(STR):
    result = re.sub(r'\[HH\]', '[H]', STR)
    return result
def str2list(reaction:str):
    r1 = reaction.split(">")
    r2 = []
    for i in r1:
        i1 = i.split()
        r2.append(i1)
    return r2
def checkbond(reaction:list,bms1,bms2):
    mol1 = bms1.mol
    mol2 =bms2.mol
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
        numbeforeadd=check_mol.GetNumAtoms()
        check_mol = COMBINE(check_mol,add)
        mol = cs12[0]
        if check_mol.GetNumAtoms() != mol.GetNumAtoms():
            return [False,False,False,False]
        bonds = cs12[0].GetBonds()
        AA=addATOM()
        aset = {AA,bondedatom}
        check=100
        outlist = [None,None,None,None]
        bms_check = cs12[-1]
        adsIDx = []
        for a in bms_check.ads:
            id = a.id
            adsIDx.append(id)
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
                if begin_atom_id >= numbeforeadd:
                    checkid = end_atom_id
                else:
                    checkid = begin_atom_id
                if checkid + bms_check.metal in adsIDx:
                    mol.RemoveBond(begin_atom_id, end_atom_id)
                    if subHH(Chem.MolToSmiles(mol)) == Chem.MolToSmiles(check_mol):
                        outlist = [begin_atom.GetIdx(),end_atom.GetIdx(),Chem.MolToSmiles(cs12[0]),Chem.MolToSmiles(check_mol)]
        for bond in bonds:
            mol = copy.deepcopy(cs12[0])
            begin_atom_id = bond.GetBeginAtomIdx()
            end_atom_id = bond.GetEndAtomIdx()
            begin_atom = mol.GetAtomWithIdx(begin_atom_id)
            end_atom = mol.GetAtomWithIdx(end_atom_id)
            qset = {begin_atom.GetSymbol(),end_atom.GetSymbol()}
            bms = cs12[2]
            ads = [] 
            for Natom in bms.ads:
                ads.append(Natom.id)
            if qset == aset:
                bmsidB=begin_atom.GetIdx()+bms.metal
                bmsidE=end_atom.GetIdx()+bms.metal
                mol.RemoveBond(begin_atom_id, end_atom_id)
                if subHH(Chem.MolToSmiles(mol)) == Chem.MolToSmiles(check_mol):
                    outlist = [begin_atom.GetIdx(),end_atom.GetIdx(),Chem.MolToSmiles(cs12[0]),Chem.MolToSmiles(check_mol)]
        return outlist[0],outlist[1],outlist[2],outlist[3]
    if reactiontype == 'Add':
        cs12 = (mol2,mol1,bms2,bms1)
        return warp(cs12)
    elif reactiontype == 'Remove':
        cs12 = (mol1,mol2,bms1,bms2)
        if addatom == 'O/OH':
            o1,o2,o3,o4 = warp(cs12,add='O')
            if o1 == False and o2 == False and o3 == False and o4 == False:
                return warp(cs12,add='OH')
            else:
                return o1,o2,o3,o4
        else:
            return warp(cs12)          
'''def check_molecule_over_surface(atoms):
    zlist = []
    molz = []
    for atom in atoms:
        if atom.symbol not in ['C','H','O']:
            zlist.append(atom.position[2])
    z_max = max(zlist)
    for atom in atoms:
        if atom.symbol in ['C','H','O']:
            molz.append(atom.position[2])
    z_min = min(molz)
    if z_min <= z_max+0.5:
        print(f'部分原子位于催化剂表面以下')
        return False
    else:    return True'''
def check_molecule_above_surface(atoms,threshold=0.5):
    molindice =[atom.index for atom in atoms if atom.symbol in ['C','H','O']]
    metalindice =[atom.index for atom in atoms if atom.symbol not in ['C','H','O']]
    points = atoms.positions[metalindice]
    for id in molindice:
        nearest_point, distance, _ = find_nearest_point_kdtree(points, atoms[id].position)
        if distance <= threshold:
            print(f'分子原子 {id} 位于催化剂表面以下或过近，距离为 {distance:.2f} Å')
            return False
    return True
def Euclidean_distance(R1:Atoms, R2:Atoms,bondids):
    """
    计算两组结构的欧氏距离
    """
    deltap2 = []
    assert len(R1) == len(R2)
    for id in bondids:
        p1 = R1.positions[id]
        p2 = R2.positions[id]
        deltap2.append(p1-p2)
    deltap2 = np.array(deltap2)
    return np.sqrt(np.sum((deltap2) ** 2))
def check_surface_or_bulk_changed(atoms,template,threshold=0.5):
    metalindice =[atom.index for atom in atoms if check_NON_metal_atoms(atom)==False]
    diff = Euclidean_distance(atoms,template,metalindice)
    if diff/np.sqrt(len(metalindice)) <= threshold:return True
    else:return False
def rotate_mol_find_best_distance(a2,a1,step_degree=60,opt = 'min',center=None,rotate_axis=np.array([0, 0, 1])):#wrong
    """
    旋转分子以找到与另一个分子之间的最大/小距离配置
    
    参数:
    atoms1: ASE Atoms 对象，表示第一个分子
    atoms2: ASE Atoms 对象，表示第二个分子
    step_degree: 每次旋转的角度步长（度）
    
    返回:
    min_distance: 最小距离
    best_atoms1: 旋转后的第一个分子配置
    """
    if opt == 'min':
        check_distance = float('inf')
    elif opt == 'max':
        check_distance = 0
    best_rotate = None
    best_model = None
    # 遍历所有旋转角度组合
    for alpha in np.arange(0,360, step_degree):
                # 创建旋转矩阵
                rotation_axis = rotate_axis # Z轴向量
                # 复制并旋转第一个分子
                rotated_a2 = a2.copy()
                rotated_a2.rotate(rotation_axis,alpha,center=center,rotate_cell=False)
                # 计算两分子之间的距离
                dist = np.linalg.norm(a1.positions-rotated_a2.get_center_of_mass(),axis=1)
                total_sum = np.sum(dist)
                # 更新最小距离和最佳配置
                if opt == 'max':
                    if  total_sum > check_distance:
                        check_distance = total_sum
                        best_rotate = alpha
                        best_model = rotated_a2
                elif opt == 'min':
                    if  total_sum < check_distance:
                        check_distance = total_sum
                        best_rotate = alpha
                        best_model = rotated_a2
                    
    return check_distance,best_model ,best_rotate
def adjust_distance_old(CB,
                    notmove,nmGidx,
                    move,mGidx,
                    new_distance=3,
                    delta=0,
                    alpha=0,
                    noads=False
                    ):
    """
    调整两个原子之间的距离
    """
    bondlist = check_neighbor(notmove,CB)
    atoms = copy.deepcopy(CB.poscar)
    pos1 = atoms.positions[notmove]
    pos2 = atoms.positions[move]

    if noads == False:pass
    else:
        molIdxlist=[]
        for atom in atoms:
            if atom.symbol in ['C','H','O']:
                molIdxlist.append(atom.index)
            else:pass
        group = atoms[molIdxlist]
        v_important = pos2-pos1
        z= np.array([0,0,-1])
        rotated_points,_ = rotate_point_set(group.positions,v_important,z,center=pos1)
        #group.translate((0,0,19-pos1[2]))
        atoms.positions[molIdxlist] = rotated_points 
    p2=atoms.positions[move]
    p1=atoms.positions[notmove]
    main = atoms[nmGidx]
    sub = atoms[mGidx]
    if np.abs(p2[1]-p1[1])<=0.1 and np.abs(p2[0]-p1[0])<=0.1:
        if  len(bondlist) == 1:
            shift =np.array([0,0.1,0])
        else:
            if len(bondlist)==2:
                v=[]
                for ba in bondlist:
                    if ba != move:
                        v.append(atoms.positions[ba]-p1)
                vd = -v[0]/np.linalg.norm(v[0])
                shift =vd
            if len(bondlist)==3:
                v=[]
                for ba in bondlist:
                    if ba != move:
                        v.append(atoms.positions[ba]-p1)
                v1 = v[0]
                v2 = v[-1]
                vd = np.cross(v1,v2)/np.linalg.norm(np.cross(v1,v2))
                shift = np.array([vd[0],vd[1],0])/np.linalg.norm(np.array([vd[0],vd[1],0]))/10
            else:
                shift = np.array([0,0.1,0])
    else:
        shift = np.array([0,0,0])
    p21=p2-p1
    ph=np.array([0,0,p1[-1]-p2[-1]])
    p21xy=np.array([p21[0],p21[1],0])
    vd = (p21+ph+shift)/np.linalg.norm(p21+ph+shift)*new_distance-p21xy+ph
    v_final =copy.deepcopy(vd)

    for id in mGidx:
        atoms.positions[id] = atoms.positions[id] + v_final 
    for atom in atoms:
        if atom.symbol in ['C','H','O']:
            aid = atom.index
            atoms.positions[aid] = atoms.positions[aid]+np.array([0,0,alpha+delta])

    '''if noads == False:pass
    else:
        addgroup = atoms[mGidx]
        for a in addgroup:
            if np.allclose(a.position, pos2, atol=1e-6):
                addgroup_pos2_idx = a.index
        v_important = pos2-pos1
        z= np.array([0,0,-1])
        theta = angle_between_vectors(v_important,z)
        axis_vz= np.cross(v_important,z)
        addgroup.rotate(v=axis_vz,a=theta,center=pos2)
        addval=addgroup.positions[addgroup_pos2_idx]-pos1
        if are_vectors_parallel(addval,np.array([0,0,-1])) == False:
            addgroup.rotate(v=axis_vz,a=-2*theta,center=pos1)
        atoms.positions[mGidx]=addgroup.positions'''
    return atoms
def update_positions(atoms,move,notmove,nmGidx,mGidx):
    p2= atoms.positions[move]
    p1 = atoms.positions[notmove]
    main = atoms[nmGidx]
    sub = atoms[mGidx]
    return p1,p2,main,sub
def adjust_distance(CB,
                    notmove,nmGidx,
                    move,mGidx,
                    new_distance=3,
                    delta=0,
                    alpha=0,
                    noads=False,
                    ):
    """
    调整两个原子之间的距离
    """
    atoms = copy.deepcopy(CB.poscar)
    pos1 = atoms.positions[notmove]
    pos2 = atoms.positions[move]
    main = atoms[nmGidx]
    sub = atoms[mGidx]

    if noads == False:pass
    else:
        v_important = pos2-pos1
        z= np.array([0,0,-1])
        rotated_points,_ = rotate_point_set(main.positions,v_important,z,center=pos1)
        main.positions = rotated_points
        main.translate((0,0,18-pos1[2]))
        v_important = pos1-pos2
        rotated_points,_ = rotate_point_set(sub.positions,v_important,z,center=pos2)
        sub.positions = rotated_points
        sub.translate((0,0,18-pos2[2]))
        atoms.positions[nmGidx]=main.positions
        atoms.positions[mGidx]=sub.positions
    p1,p2,main,sub = update_positions(atoms,move,notmove,nmGidx,mGidx)
    p21=p2-p1
    p21xy = np.array([p21[0],p21[1],0])
    if np.linalg.norm(p21xy) <=0.01:
        p21xy = np.array([1,0,0])
    rotate_points,_ = rotate_point_set(sub.positions,p21,p21xy,center=p1)
    sub.positions = rotate_points
    rotate_points,_ = rotate_point_set(sub.positions,-p21,np.array([0,0,-1]),center=p2)
    sub.positions = rotate_points
    atoms.positions[mGidx]=sub.positions
    p1,p2,main,sub = update_positions(atoms,move,notmove,nmGidx,mGidx)
    p21=p2-p1
    v_shift = p21/np.linalg.norm(p21)*new_distance - p21
    sub.translate(v_shift)
    atoms.positions[mGidx]=sub.positions
    pos1,pos2,main,sub = update_positions(atoms,move,notmove,nmGidx,mGidx)
    _,rotate_best_sub,_ = rotate_mol_find_best_distance(sub,main,step_degree=30,opt = 'max',center=pos1,rotate_axis=np.array([0, 0, 1]))
    atoms.positions[mGidx]=rotate_best_sub.positions
    pos1,pos2,main,sub = update_positions(atoms,move,notmove,nmGidx,mGidx)
    _,rotate_best_sub,_ = rotate_mol_find_best_distance(sub,main,step_degree=30,opt = 'max',center=pos2,rotate_axis=np.array([0, 0, 1]))
    atoms.positions[mGidx]=rotate_best_sub.positions
    pos1,pos2,main,sub = update_positions(atoms,move,notmove,nmGidx,mGidx)
    atoms.positions[nmGidx]=main.positions
    atoms.positions[mGidx]=sub.positions
    atoms.translate(np.array([0,0,alpha+delta]))
    return atoms
def check_neighbor(id,cb):
    idx = []
    centeratom = cb.atoms[id]
    bonddict = centeratom.bonddict
    for atom in bonddict:
       if atom.id not in idx:
            idx.append(atom.id)
    return idx
def spilt_group(id_in,id_notin,cb):
    def warp(pls:list):
        if id_in in pls:
            pls.remove(id_in)
        if id_notin in pls:
            pls.remove(id_notin)
    pl = check_neighbor(id_in,cb)
    warp(pl)
    pl_num =len(pl)
    count = 0
    while count != 2:
        for pi in pl:
            pil=check_neighbor(pi,cb)
            pl =list(set(pl+pil))
        warp(pl)
        if len(pl) ==  pl_num:count+=1
        else:pl_num = len(pl)
    pl.append(id_in)
    return pl
'''class MultiDistanceAwareOptimizer(BFGS):
    """监控多个原子对距离的优化器"""
    def __init__(self, atoms, distance_constraints, force_scale=2, 
                 trajectory=None, logfile=None):
        """
        参数:
        distance_constraints: 列表，每个元素为 (atom_i, atom_j, limit_distance, mode)
        mode = 0 : dij > limt_d时增加吸引力
        mode = 1 :dij < limit_d时增加排斥力
        """
        super().__init__(atoms=atoms, trajectory=trajectory, logfile=logfile)
        self.distance_constraints = distance_constraints
        self.force_scale = force_scale
    def get_min_image_distance(atoms, i, j):
        """
        计算晶体中两个原子之间的最小镜像距离
        
        参数:
        atoms: ASE Atoms 对象
        i, j: 原子编号
        
        返回:
        distance: 最小镜像距离
        vector: 最小镜像向量
        """
        # 获取原子位置
        pos_i = atoms.positions[i]
        pos_j = atoms.positions[j]
        
        # 获取晶胞向量
        cell = atoms.cell
        pbc = atoms.pbc
        
        # 计算最小镜像向量
        # 使用ASE内置函数计算
        from ase.geometry import find_mic
        vector = pos_j - pos_i
        mic_vector = find_mic(vector, cell, pbc)[0]
        
        # 计算距离
        distance = np.linalg.norm(mic_vector)
        
        return distance, mic_vector
    def check_and_adjust_forces(self, forces):
        """检查多个原子对距离并调整力"""
        for i, j, limit_dist, mode in self.distance_constraints:
            distance,vector = self.get_min_image_distance(self.atoms, i, j)
            if mode == 0:
                if distance > limit_dist:
                    direction = vector
                    direction_unit = direction / np.linalg.norm(direction)
                    
                    current_force = np.linalg.norm(forces[i] - forces[j])/2
                    extra_force = self.force_scale * current_force * (distance - limit_dist)
                    
                    forces[i] += extra_force * direction_unit
                    forces[j] -= extra_force * direction_unit
                    print(f"mode {mode}:调整原子对 ({i},{j}) 受力，距离: {distance:.4f} > {limit_dist:.4f} Å")
            elif mode == 1:
                if distance < limit_dist:
                    direction = vector
                    direction_unit = direction / np.linalg.norm(direction)
                    current_force = np.linalg.norm(forces[i] - forces[j])/2
                    extra_force = self.force_scale * current_force * (distance - limit_dist)
                    forces[i] += extra_force * direction_unit
                    forces[j] -= extra_force * direction_unit
                    print(f"mode {mode}:调整原子对 ({i},{j}) 受力，距离: {distance:.4f} < {limit_dist:.4f} Å")
            elif mode == -1:#强烈排斥
                if distance <= limit_dist:
                    direction = vector
                    direction_unit = direction / np.linalg.norm(direction)
                    current_force = np.linalg.norm(forces[i] - forces[j])/2
                    extra_force = 5*self.force_scale * current_force * (distance - limit_dist)
                    forces[i] += extra_force * direction_unit
                    forces[j] -= extra_force * direction_unit
                    print(f"mode {mode}:调整原子对 ({i},{j}) 受力，距离: {distance:.4f} <={limit_dist:.4f} Å")
        return forces
    def step(self, forces=None):
        if forces is None:
            forces = self.atoms.get_forces()
        adjusted_forces = self.check_and_adjust_forces(forces)
        super().step(adjusted_forces)
    def run(self, fmax=0.05, steps=...):
        super().run(fmax, steps)'''
class MultiDistanceAwareOptimizer(BFGS):
    """监控多个原子对距离的优化器"""
    def __init__(self, atoms, distance_constraints, force_scale=2, 
                 trajectory=None, logfile=None):
        """
        参数:
        distance_constraints: 列表，每个元素为 (atom_i, atom_j, limit_distance, mode)
        mode = 0 : dij > limt_d时增加吸引力
        mode = 1 :dij < limit_d时增加排斥力
        """
        self.tem = copy.deepcopy(atoms)
        super().__init__(atoms=atoms, trajectory=trajectory, logfile=logfile)
        self.distance_constraints = distance_constraints
        self.force_scale = force_scale
        self.continue_optimization = True  # 添加继续优化标志
        self.stop_condition = None  # 停止条件函数
        self.sc = 0#正常
        
    def set_stop_condition(self, condition_func):
        """
        设置停止条件函数
        
        参数:
        condition_func: 返回True/False的函数，接受atoms作为参数
        """
        self.stop_condition = condition_func
    
    @staticmethod
    def get_min_image_distance(atoms, i, j):
        """
        计算晶体中两个原子之间的最小镜像距离
        """
        # 获取原子位置
        pos_i = atoms.positions[i]
        pos_j = atoms.positions[j]
        
        # 获取晶胞向量
        cell = atoms.cell
        pbc = atoms.pbc
        
        # 计算最小镜像向量
        from ase.geometry import find_mic
        vector = pos_j - pos_i
        mic_vector = find_mic(vector, cell, pbc)[0]
        
        # 计算距离
        distance = np.linalg.norm(mic_vector)
        
        return distance, mic_vector
    
    def check_and_adjust_forces(self, forces):
        """检查多个原子对距离并调整力"""
        for i, j, limit_dist, mode in self.distance_constraints:
            distance, vector = self.get_min_image_distance(self.atoms, i, j)
            if mode == 0:
                if distance > limit_dist:
                    direction = vector
                    direction_unit = direction / np.linalg.norm(direction)
                    
                    current_force = np.linalg.norm(forces[i] - forces[j])/2
                    extra_force = self.force_scale * current_force * (distance - limit_dist)
                    
                    forces[i] += extra_force * direction_unit
                    forces[j] -= extra_force * direction_unit
                    #print(f"mode {mode}:调整原子对 ({i},{j}) 受力，距离: {distance:.4f} > {limit_dist:.4f} Å")
            elif mode == 1:
                if distance < limit_dist:
                    direction = vector
                    direction_unit = direction / np.linalg.norm(direction)
                    current_force = np.linalg.norm(forces[i] - forces[j])/2
                    extra_force = self.force_scale * current_force * (distance - limit_dist)
                    forces[i] += extra_force * direction_unit
                    forces[j] -= extra_force * direction_unit
                    #print(f"mode {mode}:调整原子对 ({i},{j}) 受力，距离: {distance:.4f} < {limit_dist:.4f} Å")
            '''elif mode == -1:
                if distance <= limit_dist:
                    direction = vector
                    direction_unit = direction / np.linalg.norm(direction)
                    current_force = np.linalg.norm(forces[i] - forces[j])/2
                    extra_force = 5*self.force_scale * current_force * (distance - limit_dist)
                    forces[i] += extra_force * direction_unit
                    forces[j] -= extra_force * direction_unit
                    #print(f"mode {mode}:调整原子对 ({i},{j}) 受力，距离: {distance:.4f} <={limit_dist:.4f} Å")'''
        return forces
    
    def step(self, forces=None):
        # 检查是否应该继续优化
        if not self.continue_optimization:
            #print("优化已暂停")
            return
        
        # 检查停止条件
        if self.stop_condition and not self.stop_condition(self.atoms,self.tem):
            print("晶体解离，暂停优化")
            self.continue_optimization = False
            self.sc = 1#boom
            return
        
        if forces is None:
            forces = self.atoms.get_forces()
        
        adjusted_forces = self.check_and_adjust_forces(forces)
        super().step(adjusted_forces)
    
    def run(self, fmax=0.05, steps=None):
        """
        运行优化，支持暂停
        
        参数:
        fmax: 最大力收敛标准
        steps: 最大步数，如果为None则无限制
        """
        self.fmax = fmax
        self.max_steps = steps
        
        # 重置优化标志
        self.continue_optimization = True
        
        for converged in self.irun(fmax=fmax, steps=steps):
            # 检查是否应该继续
            if not self.continue_optimization:
                #print(f"优化在第 {self.nsteps} 步暂停")
                break
        
        return converged if self.continue_optimization else False
    
    def resume(self):
        """恢复暂停的优化"""
        self.continue_optimization = True
        print("优化恢复")
    
    def pause(self):
        """暂停优化"""
        self.continue_optimization = False
        print("优化暂停")      
class readreaction():
    def __init__(self,file1,file2,reaction,MPLS):# file1> reaction > file2
        self.mol1 = file1
        self.mol2 = file2
        self.r = str2list(reaction)
        self.r_str = reaction
        self.group1 = []
        self.group2 = []
        self.changebondatom = None
        self.stop = False
        self.mlps = MPLS
        self.OUT1=None
        self.OUT2=None
    def readfile(self):
        def warp(id1,id2,cb):
            group1 = spilt_group(id1,id2,cb)
            group2 = spilt_group(id2,id1,cb)
            if len(group1) >1 or len(group2) >1:
                if len(group1) >= len(group2):
                    return id1,id2
                else:
                    return id2,id1

            else:
                atoms = cb.poscar
                p1 = atoms.positions[id1]
                p2 = atoms.positions[id2]
                z1=p1[-1]
                z2=p2[-1]
                if z1 <= z2:
                    return id1,id2
                else:
                    return id2,id1
        CB1 = checkBonds()
        CB1.input(self.mol1)
        CB1.AddAtoms()
        CB1.CheckAllBonds()
        CB2 = checkBonds()
        CB2.input(self.mol2)
        CB2.AddAtoms()
        CB2.CheckAllBonds()
        BMS1 = BuildMol2Smiles(CB1)
        BMS1.build()
        BMS2 = BuildMol2Smiles(CB2)
        BMS2.build()
        begin_id,end_id,smilesFORcheck,smilesFORspilt = checkbond(self.r,BMS1,BMS2)
        
        if begin_id == None or end_id == None:
            print(f'{self.r_str}:checkbond wrong! {begin_id,end_id}')
            self.stop = True
        else:
            Bid_infile = begin_id +BMS1.metal 
            Eid_infile = end_id +BMS1.metal
            reactiontype = self.r[1][0]
            '''
            确保FS为成键后产物
            '''
            if reactiontype == 'Add':
                CB = CB2
                self.OUT1 = CB2.poscar
                self.molINFO = [CB2,BMS2]
            else:
                CB = CB1
                self.OUT1 = CB1.poscar
                self.molINFO = [CB1,BMS1]
            if bool(CB.adsorption) == False:
                noads = True
            else:
                noads = False
            notmove,move = warp(Bid_infile,Eid_infile,CB)
            notmoveGroupIdx = spilt_group(notmove,move,CB)
            moveGroupIdx = spilt_group(move,notmove,CB)
            self.group1 = notmoveGroupIdx#main body
            self.group2 = moveGroupIdx#sub body
            self.changebondatom = (notmove,move)#(Bid_infile,Eid_infile)
            newmol = adjust_distance(CB,notmove,notmoveGroupIdx,move,moveGroupIdx,alpha=0,noads=noads)
            if check_molecule_above_surface(newmol) == False:
                    for i in range(1,31):
                        newmol = adjust_distance(CB,notmove,notmoveGroupIdx,move,moveGroupIdx,alpha=0,delta=0.1*i,noads=noads)
                        if check_molecule_above_surface(newmol) == True:
                            print(f'delta Z applied:{0.1*i} Å')
                            break
            self.OUT2 = newmol
            self.check =smilesFORcheck 
            self.split =smilesFORspilt
    def run_MDAO(self,path):
        self.path2save = path
        calc = NequIPCalculator.from_deployed_model(self.mlps, device='cpu')
        twogroups=self.OUT2
        self.beforeMDAO = twogroups.copy()
        twogroups.calc = calc
        distance_constraints = []
        CB = self.molINFO[0]
        bondsetlist = CB.bondsetlist
        changebondatom = self.changebondatom
        for i in range(len(twogroups)):
            for j in range(len(twogroups)):
                if i >= j:pass
                else:
                    if check_NON_metal_atoms(twogroups[i]) == False or check_NON_metal_atoms(twogroups[j])==False:
                        RuH = ['Ru','H']
                        if twogroups[i].symbol in RuH and twogroups[j].symbol in RuH and twogroups[i].symbol != twogroups[j].symbol:
                            limit_distance = covalent_radii[twogroups.get_atomic_numbers()[i]]+covalent_radii[twogroups.get_atomic_numbers()[j]]+0.5
                            distance_constraints.append((i, j, limit_distance, 1))
                    else:
                        if changebondatom in [(i,j),(j,i)]:
                            limit_distance = 3#covalent_radii[twogroups.get_atomic_numbers()[i]]+covalent_radii[twogroups.get_atomic_numbers()[j]+1]
                            if (i, j,limit_distance, 1) not in distance_constraints:
                                distance_constraints.append((i, j,limit_distance, 1))
                        else:
                            if (i,j) in bondsetlist or (j,i) in bondsetlist:
                                limit_distance = covalent_radii[twogroups.get_atomic_numbers()[i]]+covalent_radii[twogroups.get_atomic_numbers()[j]]-0.5
                                if (i, j, limit_distance, 0) not in distance_constraints:
                                    distance_constraints.append((i, j, limit_distance, 0))
                            else:
                                limit_distance = covalent_radii[twogroups.get_atomic_numbers()[i]]+covalent_radii[twogroups.get_atomic_numbers()[j]]+0.5
                                if (i, j, limit_distance, 1) not in distance_constraints:
                                    distance_constraints.append((i, j, limit_distance, 1))

        self.distance_constraints = distance_constraints
        opt = MultiDistanceAwareOptimizer(
                                        atoms=twogroups,
                                        distance_constraints=distance_constraints,
                                        force_scale=2,
                                        logfile=f'{path}MDAO.log',
                                        trajectory=f'{path}MDAO.traj'
                                        )
    
        # 运行优化
        print("\n开始MDAO优化")
        opt.set_stop_condition(check_surface_or_bulk_changed)
        mdao_fmax_bool=opt.run(fmax=0.05,steps=250)
        opt.step()
        print(f'\nMDAO结束:{mdao_fmax_bool}')
        print('\n开始BFGS优化')
        bfgs = BFGS(twogroups, logfile=f'{path}BFGS.log', trajectory=f'{path}BFGS.traj')
        bfgs_fmax_bool=bfgs.run(fmax=0.01,steps=1000)
        print(f'\nBFGS结束:{bfgs_fmax_bool}')
        self.OUT2 = twogroups
        self.opt_check = [bool(mdao_fmax_bool),bool(bfgs_fmax_bool)]
    def check_result(self,path):
        if self.opt_check[-1]==True:
            twogroups=copy.deepcopy(self.OUT2)
            ccb = checkBonds()
            ccb.poscar = twogroups
            ccb.AddAtoms()
            ccb.CheckAllBonds()
            cbms = BuildMol2Smiles(ccb)
            cbms.build()
            ADS = cbms.ads
            G1ID = self.group1
            G2ID = self.group2
            g1ads = 0
            g2ads = 0
            for ad in ADS:
                adid = ad.id
                if adid in G1ID:
                    g1ads +=1
                elif adid in G2ID:
                    g2ads +=1
            print(f'\n优化后分子SMILES:{cbms.smiles},check:{self.split},output:{[bool(cbms.smiles == self.split),bool(g1ads!=0),bool(g2ads!=0)]}')
            self.OUT2 = twogroups
            self.check_result_out = [cbms.smiles == self.split, g1ads!=0, g2ads!=0]
        else:
            twogroups=copy.deepcopy(self.OUT2)
            print('\n开始FIRE优化')
            fire = FIRE(twogroups, logfile=f'{path}FIRE.log', trajectory=f'{path}FIRE.traj')
            fire_fmax_bool=fire.run(fmax=0.01,steps=1000)
            print(f'\nFIRE结束:{fire_fmax_bool}')
            ccb = checkBonds()
            ccb.poscar = twogroups
            ccb.AddAtoms()
            ccb.CheckAllBonds()
            cbms = BuildMol2Smiles(ccb)
            cbms.build()
            ADS = cbms.ads
            G1ID = self.group1
            G2ID = self.group2
            g1ads = 0
            g2ads = 0
            for ad in ADS:
                adid = ad.id
                if adid in G1ID:
                    g1ads +=1
                elif adid in G2ID:
                    g2ads +=1
            print(f'\n优化后分子SMILES:{cbms.smiles},check:{self.split},output:{[bool(cbms.smiles == self.split),bool(g1ads!=0),bool(g2ads!=0)]}')
            self.opt_check.append(bool(fire_fmax_bool))
            self.OUT2 = twogroups
            self.check_result_out = [cbms.smiles == self.split, g1ads!=0, g2ads!=0]

    def save(self,path,format):
        reactiontype = self.r[1][0]
        if reactiontype == 'Add':
                self.IS = self.OUT2
                self.FS = self.OUT1
        else:
                self.IS = self.OUT1
                self.FS = self.OUT2
        # 保存为POSCAR文件（VASP格式）
        if format=='poscar' or 'POSCAR' or 'vasp':
            if self.stop == False:
                write(path+'IS.vasp', self.IS, format='vasp', vasp5=True)  # vasp5=True添加元素名称
                write(path+'FS.vasp', self.FS, format='vasp', vasp5=True)  # vasp5=True添加元素名称
                write(path+'beforeMDAO.vasp', self.beforeMDAO, format='vasp', vasp5=True)  # vasp5=True添加元素名称
        else:
            print('format should be .vasp')
