import torch
from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms
from sella import Sella
from nequip.ase import NequIPCalculator

# 加载预训练的势函数
model_path = '/work/home/ac877eihwp/renyq/LUNIX_all/mlp_opt/prototypeModel.pth'
calculator = NequIPCalculator.from_deployed_model(model_path)

# 从 POSCAR 文件中读取初始结构
initial_structure = read('POSCAR')
fixed_indices = list(range(0, 32))
initial_structure.constraints = [FixAtoms(indices=fixed_indices)]
# 设置计算器
initial_structure.set_calculator(calculator)

# 使用 Sella 进行过渡态搜索
opt = Sella(initial_structure, trajectory='ts_search.traj')
opt.run(fmax=0.05)  # fmax 是力的收敛标准

# 保存优化后的结构
write('optimized_ts.xyz', initial_structure)
