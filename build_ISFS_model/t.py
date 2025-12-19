import numpy as np
from ase.optimize import BFGS
from ase import Atoms
from ase.io import read
from nequip.ase import NequIPCalculator
class DistanceAwareOptimizer(BFGS):
    """
    自定义优化器：监控特定原子对距离并在距离过大时调整力
    """
    def __init__(self, atoms, atom_indices, max_distance, force_scale=0.1, 
                 trajectory=None, logfile=None, master=None):
        """
        参数:
        atoms: Atoms对象
        atom_indices: 要监控的原子对索引，例如 (0, 1)
        max_distance: 允许的最大距离（埃）
        force_scale: 力调整的缩放因子
        """
        super().__init__(atoms, trajectory, logfile, master)
        self.atom_indices = atom_indices
        self.max_distance = max_distance
        self.force_scale = force_scale
        
    def check_and_adjust_forces(self, forces):
        """检查原子距离并在需要时调整力"""
        i, j = self.atom_indices
        positions = self.atoms.positions
        
        # 计算当前原子距离
        distance = np.linalg.norm(positions[i] - positions[j])
        print(f"当前原子 {i} 和 {j} 之间的距离: {distance:.4f} Å")
        
        # 如果距离超过阈值，调整力
        if distance > self.max_distance:
            print(f"距离超过阈值 {self.max_distance} Å，调整受力...")
            
            # 计算原子间方向向量
            direction = positions[j] - positions[i]
            direction_unit = direction / np.linalg.norm(direction)
            
            # 计算当前原子间作用力的大小
            current_force_magnitude = np.linalg.norm(forces[i] - forces[j])
            
            # 添加额外的吸引力（指向对方）
            # 力的大小与超出阈值的程度和当前力的大小相关
            extra_force_magnitude = self.force_scale * current_force_magnitude * (distance - self.max_distance)
            
            # 应用调整后的力
            forces[i] += extra_force_magnitude * direction_unit
            forces[j] -= extra_force_magnitude * direction_unit
            
            print(f"已调整受力，额外力大小: {extra_force_magnitude:.6f} eV/Å")
        
        return forces
    
    def step(self, forces=None):
        """重写step方法，在每一步优化中检查并调整力"""
        if forces is None:
            forces = self.atoms.get_forces()
        
        # 检查距离并调整力
        adjusted_forces = self.check_and_adjust_forces(forces)
        
        # 使用调整后的力进行优化步骤
        super().step(adjusted_forces)

# ========== 使用示例 ==========
def example_usage():
    """使用示例"""
    model_path = 'prototypeModel.pth'
    calc = NequIPCalculator.from_deployed_model(model_path, device='cpu')
    
    # 创建一个示例分子（一氧化碳）
    atoms = read('000.traj',index=0)
    atoms.calc = calc  # 使用NequIP计算器
    # 设置要监控的原子对（C和O原子）
    carbon_idx, oxygen_idx = 66, 67  # 在CO分子中
    
    print("初始结构:")
    print(f"C 原子位置: {atoms.positions[carbon_idx]}")
    print(f"O 原子位置: {atoms.positions[oxygen_idx]}")
    
    initial_distance = np.linalg.norm(atoms.positions[carbon_idx] - atoms.positions[oxygen_idx])
    print(f"初始C-O距离: {initial_distance:.4f} Å")
    
    # 创建自定义优化器
    # 设置最大允许距离为1.5 Å（CO键长约为1.13 Å）
    opt = DistanceAwareOptimizer(
        atoms=atoms,
        atom_indices=(carbon_idx, oxygen_idx),
        max_distance=1.5,
        force_scale=0.2,
        logfile='-'
    )
    
    # 运行优化
    print("\n开始结构优化...")
    opt.run(fmax=0.05, steps=100)
    
    print("\n优化后结构:")
    print(f"C 原子位置: {atoms.positions[carbon_idx]}")
    print(f"O 原子位置: {atoms.positions[oxygen_idx]}")
    final_distance = np.linalg.norm(atoms.positions[carbon_idx] - atoms.positions[oxygen_idx])
    print(f"最终C-O距离: {final_distance:.4f} Å")

# ========== 高级版本：监控多个原子对 ==========
class MultiDistanceAwareOptimizer(BFGS):
    """监控多个原子对距离的优化器"""
    
    def __init__(self, atoms, distance_constraints, force_scale=0.1, 
                 trajectory=None, logfile=None, master=None):
        """
        参数:
        distance_constraints: 列表，每个元素为 (atom_i, atom_j, max_distance)
        """
        super().__init__(atoms, trajectory, logfile, master)
        self.distance_constraints = distance_constraints
        self.force_scale = force_scale
        
    def check_and_adjust_forces(self, forces):
        """检查多个原子对距离并调整力"""
        positions = self.atoms.positions
        
        for i, j, max_dist in self.distance_constraints:
            distance = np.linalg.norm(positions[i] - positions[j])
            
            if distance > max_dist:
                direction = positions[j] - positions[i]
                direction_unit = direction / np.linalg.norm(direction)
                
                current_force = np.linalg.norm(forces[i] - forces[j])
                extra_force = self.force_scale * current_force * (distance - max_dist)
                
                forces[i] += extra_force * direction_unit
                forces[j] -= extra_force * direction_unit
                
                print(f"调整原子对 ({i},{j}) 受力，距离: {distance:.4f} Å")
        
        return forces
    
    def step(self, forces=None):
        if forces is None:
            forces = self.atoms.get_forces()
        
        adjusted_forces = self.check_and_adjust_forces(forces)
        super().step(adjusted_forces)

if __name__ == "__main__":
    example_usage()