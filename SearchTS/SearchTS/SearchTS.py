import numpy as np
from ase import Atoms
from ase.optimize import BFGS, LBFGS
from ase.io import read, write
import matplotlib.pyplot as plt
from ase.constraints import FixAtoms
# 线性插值生成中间结构
def interpolate_structure(initial_file, final_file, fraction=0.5, output_file=None):
    """
    线性插值生成任意比例的中间结构
    
    Parameters:
    - initial_file: 初始结构文件
    - final_file: 最终结构文件
    - fraction: 插值比例 (0=初始结构, 1=最终结构, 0.5=50%中间结构)
    - output_file: 输出文件路径
    """
    # 读取结构
    initial = read(initial_file)
    final = read(final_file)
    
    # 验证一致性
    if len(initial) != len(final):
        raise ValueError("原子数量不一致!")
    
    if initial.get_chemical_symbols() != final.get_chemical_symbols():
        raise ValueError("原子种类或顺序不一致!")
    
    # 线性插值
    pos_initial = initial.get_positions()
    pos_final = final.get_positions()
    
    # 计算插值后的坐标
    pos_interpolated = pos_initial + fraction * (pos_final - pos_initial)
    
    # 创建新结构
    new_structure = initial.copy()
    new_structure.set_positions(pos_interpolated)
    
    # 保存
    if output_file:
        write(output_file, new_structure)
        print(f"{fraction*100}% 结构已保存到: {output_file}")
    
    return new_structure
# 计算两组结构的欧氏距离
def Euclidean_distance(R1:Atoms, R2:Atoms):
    """
    计算两组结构的欧氏距离
    """
    p1 = R1.get_positions()
    p2 = R2.get_positions()
    assert len(R1) == len(R2)
    return np.sqrt(np.sum((p1 - p2) ** 2))
# 能量阈限的结构优化 
def optimize_with_energy_criterion(atoms, calculator, energy_threshold=1e-2, 
                                   max_steps=100, trajectory_file='optimization.traj'):
    """
    使用标准优化器但基于能量收敛准则
    
    Parameters:
    - atoms: 要优化的原子结构
    - calculator: 计算器
    - energy_threshold: 能量收敛阈值 (eV)
    - max_steps: 最大优化步数
    - trajectory_file: 轨迹文件路径
    """
    
    # 设置计算器
    atoms.set_calculator(calculator)
    
    # 初始化优化器
    optimizer = BFGS(atoms, trajectory=trajectory_file, logfile='optimization.log')
    
    # 记录能量历史
    energy_history = []
    
    # 手动执行优化步骤
    for step in range(max_steps):
        # 获取当前能量
        current_energy = atoms.get_potential_energy()
        energy_history.append(current_energy)
        
        print(f"步骤 {step}: 能量 = {current_energy:.6f} eV")
        
        # 检查能量收敛
        if len(energy_history) > 1:
            energy_diff = abs(energy_history[-1] - energy_history[-2])
            print(f"  能量变化: {energy_diff:.6f} eV")
            
            if energy_diff < energy_threshold:
                print(f"\n能量收敛于步骤 {step}!")
                print(f"最终能量变化: {energy_diff:.6f} eV < 阈值 {energy_threshold} eV")
                break
        
        # 执行一步优化
        try:
            optimizer.step()
        except StopIteration:
            print("优化器自然收敛")
            break
    
    else:
        print(f"达到最大步数 {max_steps}")
    
    # 绘制能量收敛图
    #plot_energy_convergence(energy_history, energy_threshold)
    
    return atoms, energy_history
# 能量作图（optional)
def plot_energy_convergence(energy_history, threshold):
    """绘制能量收敛图"""
    plt.figure(figsize=(10, 6))
    
    # 计算能量变化
    energy_changes = []
    for i in range(1, len(energy_history)):
        energy_changes.append(abs(energy_history[i] - energy_history[i-1]))
    
    # 绘制能量历史
    plt.subplot(2, 1, 1)
    plt.plot(energy_history, 'b-o', markersize=3)
    plt.ylabel('能量 (eV)')
    plt.title('优化能量历史')
    plt.grid(True)
    
    # 绘制能量变化
    plt.subplot(2, 1, 2)
    plt.semilogy(range(1, len(energy_history)), energy_changes, 'r-o', markersize=3)
    plt.axhline(y=threshold, color='g', linestyle='--', label=f'收敛阈值: {threshold} eV')
    plt.xlabel('优化步骤')
    plt.ylabel('能量变化 (eV)')
    plt.title('能量变化历史 (对数坐标)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('energy_convergence.png', dpi=300)
    plt.show()
#计算方向性
def sigmaCopt(delta_dIS,delta_dFS):
    """
    计算sigmaCopt
    """
    if delta_dIS < 0 and delta_dFS > 0:
        out = 'IS'
    elif delta_dIS > 0 and delta_dFS < 0:
        out = 'FS'
    elif delta_dIS*delta_dFS > 0:
        out = 'Nondirectional'
    else:
        ValueError("Error in sigmaCopt calculation!")
    return out
def D_criteria(delta_dIS,delta_dFS):
    """
    计算D_criteria
    """
    if delta_dIS*delta_dFS > 0:
        return True
    elif np.abs(delta_dIS) < 0.05 or np.abs(delta_dFS) < 0.05:   
        return True
    else:
        return False
    """
    计算D_criteria
    """
    if delta_dIS*delta_dFS > 0:
        return True
    elif np.abs(delta_dIS) < 0.05 or np.abs(delta_dFS) < 0.05:   
        return True
    else:
        return False
class RDA_D():
    def __init__(self,ISfile,FSfile,path):
        self.ISfile = ISfile
        self.FSfile = FSfile
        self.path = path
    def readData(self):
        IS = read(self.ISfile)
        FS = read(self.FSfile)
        def warp(atoms):
            """打印并返回固定原子索引"""
            fixed_atoms = []
            for i, constraint in enumerate(atoms.constraints):
                if isinstance(constraint, FixAtoms):
                    indices = constraint.index
                    fixed_atoms.extend(indices.tolist())
                    print(f"  固定原子数量: {len(indices)}")
                    print(f"  固定原子索引 (0-based): {indices.tolist()}")
            return fixed_atoms
        IS_fixed = warp(IS)
        FS_fixed = warp(FS)
        assert IS_fixed == FS_fixed, "IS和FS的固定原子不一致!"
        self.fixed_atoms = IS_fixed
        return IS_fixed
    def run(self, calculator):
        # 线性插值生成中间结构
        intermediate_structure = interpolate_structure(self.ISfile, self.FSfile, fraction=0.5, output_file=f'{self.path}/R_aloha.traj')
        # 设置计算器（这里以Lennard-Jones为例，用户应根据实际情况设置）
        calc = calculator
        # 使用能量收敛准则优化中间结构
        optimized_atoms, energy_history = optimize_with_energy_criterion(intermediate_structure, calc, 
                                                                        energy_threshold=0.01, 
                                                                        max_steps=100, 
                                                                        trajectory_file=f'{self.path}/R1.traj')
        return optimized_atoms, energy_history
    

    