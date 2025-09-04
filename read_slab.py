from ase.io import read
import numpy as np
def analyze_poscar_pbc(poscar_file):
    """
    读取 POSCAR 文件并分析其周期性边界条件
    
    参数:
    poscar_file: POSCAR 文件路径
    
    返回:
    包含 PBC 信息的字典
    """
    # 读取 POSCAR 文件
    atoms = read(poscar_file)
    
    # 获取晶胞向量
    cell = atoms.get_cell()
    
    # 计算晶胞大小（每个方向的长度）
    a = np.linalg.norm(cell[0])  # a 方向长度
    b = np.linalg.norm(cell[1])  # b 方向长度
    c = np.linalg.norm(cell[2])  # c 方向长度
    
    # 计算晶胞角度
    alpha = np.degrees(np.arccos(np.dot(cell[1], cell[2]) / (b * c)))
    beta = np.degrees(np.arccos(np.dot(cell[0], cell[2]) / (a * c)))
    gamma = np.degrees(np.arccos(np.dot(cell[0], cell[1]) / (a * b)))
    
    # 获取 PBC 设置（哪些方向是周期性的）
    pbc_flags = atoms.get_pbc()
    
    # 获取晶胞体积
    volume = atoms.get_volume()
    
    # 返回所有信息
    return {
        'cell_vectors': cell,
        'cell_lengths': (a, b, c),
        'cell_angles': (alpha, beta, gamma),
        'pbc_flags': pbc_flags,
        'volume': volume,
        'pbc_directions': get_pbc_directions(pbc_flags)
    }

def get_pbc_directions(pbc_flags):
    """
    将 PBC 标志转换为方向描述
    
    参数:
    pbc_flags: 三个布尔值的元组或列表，表示 x, y, z 方向的周期性
    
    返回:
    方向描述字符串
    """
    directions = []
    if pbc_flags[0]:
        directions.append('x')
    if pbc_flags[1]:
        directions.append('y')
    if pbc_flags[2]:
        directions.append('z')
    
    if not any(pbc_flags):
        return "无周期性边界条件"
    elif all(pbc_flags):
        return "三维周期性 (x, y, z)"
    else:
        return f"{len(directions)}维周期性 ({', '.join(directions)})"
class test():
    def __init__(self):
        self.r_Ru = 2.7
    def read_slab(self,path):
        struct = read(path)
        self.struct = struct
        self.analyzeOFstruct = analyze_poscar_pbc(path)
    def build_grid4surface(self):
        (a,b,c) = self.analyzeOFstruct['cell_lengths']
        (alpha, beta, gamma) = self.analyzeOFstruct['cell_angles']