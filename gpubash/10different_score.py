#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
111.py - MCTS for Z-matrix based structure optimization
修改说明:
- 去掉了原子重叠检查（删除了 check_atomic_overlap() 和在 simulation 中的对应调用分支）
- 对原代码进行了注释增强：对函数、重要变量和关键行添加了更详细的注释（逐步说明目的和影响）
- 保持原有逻辑（如随机参考三个原子、离散化、FIRE 优化、NequIP/ DummyCalculator 回退等）
- 修改参考原子选择逻辑，确保三个参考原子不连续
- 新增功能：记录异常原子并施加惩罚，同时处理优化前后的能量
"""

# -----------------------------
# 基本库导入
# -----------------------------
import numpy as np  # 数值计算（向量、数组、线性代数等）
import re  # 正则，用于解析 Z-matrix 文本行
import math  # 数学函数（例如 sqrt、log）
from ase import Atoms  # ASE 的 Atoms 类，用来表示原子体系（positions, symbols 等）
from ase.io import read, write  # ASE 文件读写函数（支持 xyz, cif 等）
from ase.optimize import FIRE  # FIRE 优化器，用于结构优化（geometry relaxation）
from ase.data import atomic_numbers  # （保留）原子序号字典（如果需要）

try:
    from nequip.ase import NequIPCalculator  # 尝试导入 NequIP 计算器（若可用）
except Exception:
    NequIPCalculator = None  # 如果导入失败，保持为 None，后面用 DummyCalculator 代替
import random  # 随机数与随机选择
import json  # 用于保存结果为 JSON
import os  # 文件/目录操作
import time  # 计时
from datetime import datetime  # 时间戳生成
import pandas as pd  # 读写 CSV、构造 DataFrame（离散化表格化）
import argparse  # 命令行参数解析

# 自定义转换器模块（负责 Z-matrix <-> Cartesian 转换）
from converter3 import Converter


# -----------------------------
# Timer helper：上下文管理器式计时（用于记录阶段时间）
# -----------------------------
class Timer:
    def __init__(self, name="Operation"):
        self.name = name  # 阶段名（用于日志/打印）
        self.start_time = None  # 记录开始时间
        self.end_time = None  # 记录结束时间
        self.elapsed = 0.0  # 存放耗时

    def __enter__(self):
        self.start_time = time.time()  # 进入上下文时记录当前时间
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()  # 退出上下文时记录结束时间
        self.elapsed = self.end_time - self.start_time  # 计算耗时（秒）


# -----------------------------
# 1. 从 XYZ 文件读取结构（返回 ASE Atoms）
# -----------------------------
def read_structure_from_xyz(xyz_file):
    """
    读取 xyz 文件并返回 ASE Atoms 对象。
    - 设定 cell 为 None、pbc=False（即非周期性体系）
    - 捕获异常并返回 None
    """
    try:
        with Timer("读取XYZ文件"):
            atoms = read(xyz_file)  # ASE 自动解析 xyz 文件到 Atoms
        print(f"成功从 {xyz_file} 读取结构，包含 {len(atoms)} 个原子")
        atoms.set_cell(None)  # 去掉晶格信息（以 Cartesian 原子坐标为主）
        atoms.set_pbc(False)  # 关闭周期边界条件
        print("移除晶格信息，使用非周期性边界条件")
        return atoms
    except Exception as e:
        print(f"读取XYZ文件时出错: {e}")
        return None


# -----------------------------
# 2. 把 ASE Atoms 写成临时 xyz，再调用 Converter 得到 Z-matrix（文本行列表）
# -----------------------------
def xyz_to_zmat(atoms):
    """
    使用 Converter 把 Cartesian（xyz）转换为 Z-matrix 文本行（跳过文件头两行）
    返回 (zmat_lines, lattice, pbc) 目前 lattice/pbc 暂未使用，返回 None/False 与旧代码保持兼容
    """
    converter = Converter()
    temp_xyz = "temp_cartesian.xyz"
    temp_zmat = "temp_zmatrix.zmat"
    with Timer("保存临时XYZ文件"):
        write(temp_xyz, atoms)  # 用 ASE 写出临时 xyz
    with Timer("XYZ到Z矩阵转换"):
        converter.run_cartesian(input_file=temp_xyz, output_file=temp_zmat)  # converter 执行转换
    with Timer("读取Z矩阵文件"):
        with open(temp_zmat, 'r') as f:
            zmat_lines = f.readlines()[2:]  # 跳过前两行（#ZMATRIX / 注释）
    # 清理临时文件（忽略删除错误）
    try:
        os.remove(temp_xyz)
    except Exception:
        pass
    try:
        os.remove(temp_zmat)
    except Exception:
        pass
    # 返回去掉首尾空格后的行列表，和占位的 lattice/pbc
    return [line.strip() for line in zmat_lines], None, False


# -----------------------------
# helper: 从 zmat 行中提取每个原子在文件中给出的键长（用于确定全局距离 bins）
# -----------------------------
def extract_actual_distances(zmat_lines):
    """
    遍历 zmat 的每一行，解析出 distance（bond）字段（如果存在），构造一个原子级别的实际距离列表：
    - 第一原子距离设为 0.0（没有参考）
    - 若解析失败或字段缺失，则使用默认 1.5
    """
    distances = []
    for i, line in enumerate(zmat_lines):
        parts = re.split(r'\s+', line.strip())  # 用空白分割
        if i == 0:
            distances.append(0.0)  # 第一个原子没有 bond 值
        elif len(parts) > 2:
            try:
                distances.append(float(parts[2]))  # Z-matrix 中通常第 3 项是距离值
            except Exception:
                distances.append(1.5)  # 解析失败回退默认值 1.5 Å
        else:
            distances.append(1.5)
    return distances


# -----------------------------
# 创建距离区间（bins）
# -----------------------------
def create_global_distance_bins(actual_distances, n_bins=5):
    """
    为 bond length 创建统一的全局区间（目前固定从 1.0 到 2.8，等间距）
    - 返回 bins 的边界数组，长度为 n_bins+1
    """
    return np.linspace(1.0, 2.8, n_bins + 1)


def create_fixed_bins(feature_type, n_bins=10):
    """
    为角度(angle)或二面角(dihedral)创建固定的离散区间边界数组
    - angle: 60 -> 180 度
    - dihedral: -180 -> +180 度
    """
    if feature_type == 'angle':
        return np.linspace(60.0, 180.0, n_bins + 1)
    elif feature_type == 'dihedral':
        return np.linspace(-180.0, 180.0, n_bins + 1)
    else:
        raise ValueError(f"未知的特征类型: {feature_type}")


# -----------------------------
# 解析 Z-matrix 文本 -> 构建 all_rows（每一行可能的离散组合）以及对应的 bins 列表
# -----------------------------
def parse_zmat(zmat_lines, n_bins=10):
    """
    核心：
    - 计算实际距离序列 -> 用于全局 distance bins
    - 为每一行构建一个 list，里面是该行所有离散化的组合 (element, bond_bin_idx, angle_bin_idx, dihedral_bin_idx)
      - 第 1 原子：(element,)
      - 第 2 原子：(element, bond_idx)
      - 第 3 原子：(element, bond_idx, angle_idx)
      - 第 4+ 原子：(element, bond_idx, angle_idx, dihedral_idx)
    - 返回: all_rows, distance_bins_list, angle_bins, dihedral_bins, global_distance_bins
    """
    with Timer("提取实际距离"):
        actual_distances = extract_actual_distances(zmat_lines)

    with Timer("创建全局距离区间"):
        global_distance_bins = create_global_distance_bins(actual_distances, 5)

    # distance_bins_list: 对每一行是否有 distance bins 的引用（第一行没有）
    distance_bins_list = []
    for i in range(len(actual_distances)):
        if i == 0:
            distance_bins_list.append(None)
        else:
            distance_bins_list.append(global_distance_bins)

    with Timer("创建角度和二面角区间"):
        angle_bins = create_fixed_bins('angle', n_bins)
        dihedral_bins = create_fixed_bins('dihedral', n_bins)

    all_rows = []
    with Timer("解析Z矩阵并创建所有可能的行组合"):
        for i, line in enumerate(zmat_lines):
            parts = re.split(r'\s+', line.strip())
            if not parts:
                continue
            element = parts[0]  # 原子类型（符号）
            # 根据行编号决定该行应包含的离散变量（bond/angle/dihedral）
            if i == 0:
                # 第 1 原子只有元素符号
                row_combos = [(element,)]
            elif i == 1:
                # 第 2 原子只有 bond（距离）索引
                row_combos = [(element, bond_idx) for bond_idx in range(len(global_distance_bins) - 1)]
            elif i == 2:
                # 第 3 原子包含 bond 和 angle 两个离散维度
                row_combos = [
                    (element, bond_idx, angle_idx)
                    for bond_idx in range(len(global_distance_bins) - 1)
                    for angle_idx in range(len(angle_bins) - 1)
                ]
            else:
                # 第 4 个及以后原子包含 bond, angle, dihedral 三个离散维度
                row_combos = [
                    (element, bond_idx, angle_idx, dihedral_idx)
                    for bond_idx in range(len(global_distance_bins) - 1)
                    for angle_idx in range(len(angle_bins) - 1)
                    for dihedral_idx in range(len(dihedral_bins) - 1)
                ]
            all_rows.append(row_combos)

    return all_rows, distance_bins_list, angle_bins, dihedral_bins, global_distance_bins


# -----------------------------
# build z-matrix line from a selection tuple
# -----------------------------
def build_zmat_line_from_selection(i, sel, distance_bins_list, angle_bins, dihedral_bins):
    """
    根据给定的离散选择 sel 构建对应的 Z-matrix 行文本：
    - 对于每个索引 i（原子序号），把所选 bin 的中点值写成数值
    - 对于 4+ 原子，随机选择三个不连续的参考原子索引（1..i）
    """
    if i == 0:
        # 第一个原子：只有元素符号
        return f"{sel[0]}"
    elif i == 1:
        # 第二原子：元素 + 参考原子 1 的键长
        bond_idx = sel[1]
        bond_value = (distance_bins_list[i][bond_idx] + distance_bins_list[i][bond_idx + 1]) / 2
        return f"{sel[0]} 1 {bond_value:.6f}"
    elif i == 2:
        # 第三原子：元素 + 参考1 的键长 + 参考2 的角度
        bond_idx, angle_idx = sel[1], sel[2]
        bond_value = (distance_bins_list[i][bond_idx] + distance_bins_list[i][bond_idx + 1]) / 2
        angle_value = (angle_bins[angle_idx] + angle_bins[angle_idx + 1]) / 2
        return f"{sel[0]} 1 {bond_value:.6f} 2 {angle_value:.6f}"
    else:
        # 第四及以后原子：元素 + 随机选择三个不连续的参考原子 + 对应的值（bond/angle/dihedral）
        bond_idx, angle_idx, dihedral_idx = sel[1], sel[2], sel[3]
        bond_value = (distance_bins_list[i][bond_idx] + distance_bins_list[i][bond_idx + 1]) / 2
        angle_value = (angle_bins[angle_idx] + angle_bins[angle_idx + 1]) / 2
        dihedral_value = (dihedral_bins[dihedral_idx] + dihedral_bins[dihedral_idx + 1]) / 2

        # 随机选择三个不连续的参考原子，要求索引范围为 1..i（原 Z-matrix 是 1 起始）
        # 确保三个参考原子不连续（任意两个参考原子索引差不等于1）
        max_attempts = 100
        ref_atoms = []
        for attempt in range(max_attempts):
            ref_atoms = random.sample(range(1, i + 1), 3)
            sorted_ref = sorted(ref_atoms)
            # 检查是否有连续的情况
            has_consecutive = False
            for j in range(2):
                if sorted_ref[j + 1] - sorted_ref[j] == 1:
                    has_consecutive = True
                    break
            if not has_consecutive:
                break
        ref1, ref2, ref3 = ref_atoms
        return f"{sel[0]} {ref1} {bond_value:.6f} {ref2} {angle_value:.6f} {ref3} {dihedral_value:.6f}"


# -----------------------------
# Z-matrix -> Cartesian (xyz) using Converter
# -----------------------------
def zmat_to_xyz_with_values(zmat_lines, lattice=None, pbc=None, scale_factor=1.0):
    """
    把给定的 Z-matrix 行写到临时文件，调用 Converter 生成 xyz，再用 ASE 读回 Atoms。
    - 可选择对返回坐标做整体缩放（scale_factor）
    - 返回 ASE Atoms 或者在失败时返回 None
    """
    converter = Converter()
    temp_zmat = "temp_zmatrix.zmat"
    temp_xyz = "temp_cartesian.xyz"

    # 把 Z-matrix 行写入临时文件（带 #ZMATRIX 头以符合 converter 的输入格式）
    with open(temp_zmat, 'w') as f:
        f.write("#ZMATRIX\n#\n")
        for line in zmat_lines:
            f.write(line + "\n")
    try:
        # 调用 converter 把 zmat 转成 xyz（若转换失败会抛异常）
        converter.run_zmatrix(input_file=temp_zmat, output_file=temp_xyz)
    except Exception as e:
        # 若转换报错，清理临时文件并返回 None
        try:
            os.remove(temp_zmat)
        except Exception:
            pass
        try:
            if os.path.exists(temp_xyz):
                os.remove(temp_xyz)
        except Exception:
            pass
        return None

    # 读取生成的 xyz 并构建 Atoms 对象
    with open(temp_xyz, 'r'):
        pass  # 仅用来确保文件存在（占位）
    atoms = read(temp_xyz)
    atoms.set_cell(None)  # 去掉晶格
    atoms.set_pbc(False)  # 非周期性
    if scale_factor != 1.0:
        positions = atoms.get_positions()
        positions *= scale_factor  # 整体缩放坐标
        atoms.set_positions(positions)

    # 清理临时文件（忽略删除错误）
    try:
        os.remove(temp_zmat)
    except Exception:
        pass
    try:
        os.remove(temp_xyz)
    except Exception:
        pass
    return atoms


# -----------------------------
# MCTS 节点类
# -----------------------------
class MCTSNode:
    def __init__(self, row_idx, selection, parent=None):
        # 基本树结构信息
        self.row_idx = row_idx  # 该节点对应的行索引（第几个原子，0-based）
        self.selection = selection  # 存放该节点表示的离散选择元组（如 (element, bond_idx, ...)）
        self.parent = parent  # 父节点引用（根节点 parent=None）
        self.children = []  # 子节点列表

        # 评估与统计信息
        self.visits = 0  # 被访问（update/backprop）次数
        self.composite_score = 0.0  # 累积评分（复合分数，越高越好）
        self.cumulative_energy = 0.0  # 累计的原子能量（用于分析/写 log）
        self.expanded = False  # 是否已展开（是否已经创建过子节点）
        self.uct_cache = float('inf')  # UCT 缓存（便于快速选择），初始为 inf 以鼓励首次探索

    def update_uct(self, exploration_weight=0.1):
        """
        计算并缓存 UCT 值（供 best_child 使用）
        UCT 形式：
        - 若节点未访问（visits==0）：使用 prior + exploration
            prior = parent.composite_score / max(1, parent.visits)  （若有父节点）
            exploration = exploration_weight * sqrt( log(max(1,parent.visits+1)) )
        - 否则：标准 UCT：
            uct = (composite_score / visits) + exploration_weight * sqrt( log(parent.visits+1) / visits )
        目的是把 exploitation（平均已获得得分）与 exploration（罕见节点优先）结合起来。
        """
        if self.visits == 0:
            prior = 0.0 if self.parent is None else (self.parent.composite_score / max(1, self.parent.visits))
            # parent.visits 或 0 时通过 max(1, ...) 防止 log(0)
            self.uct_cache = prior + exploration_weight * math.sqrt(math.log(max(1, (self.parent.visits or 0) + 1)))
        else:
            # 注意用 parent.visits + 1 防止 log(0)
            self.uct_cache = (self.composite_score / self.visits) + \
                             exploration_weight * math.sqrt(math.log(self.parent.visits + 1) / self.visits)

    def uct_value(self):
        # 返回缓存的 UCT 值（若需要可先调用 update_uct）
        return self.uct_cache

    def add_child(self, child_node):
        # 把 child_node 添加到当前节点的子列表里
        self.children.append(child_node)

    def update(self, total_energy_pre, total_energy_post, cumulative_atomic_energy_pre, cumulative_atomic_energy_post,
               abnormal_atoms=None, exploration_weight=0.1):
        """
        在 backpropagation 中被调用以更新节点统计信息：
        - visits 自增（访问计数）
        - 根据异常原子列表，对异常原子对应的节点施加惩罚
        - composite_score 增加基于前后能量的 reward
        """
        self.visits += 1

        # 检查当前节点是否对应异常原子
        penalty = 0.0
        if abnormal_atoms and self.row_idx in abnormal_atoms:
            # 对异常原子施加大惩罚
            penalty = 1000.0
            print(f"节点 {self.row_idx} 对应异常原子，施加惩罚 {penalty}")

        # 设计选择：能量越小越好 => 把能量取负数作为得分（前后能量各占50%权重）
        structure_score_pre = -total_energy_pre * 0.25
        structure_score_post = -total_energy_post * 0.25
        cumulative_score_pre = -cumulative_atomic_energy_pre * 0.25
        cumulative_score_post = -cumulative_atomic_energy_post * 0.25

        # 应用惩罚
        reward = structure_score_pre + structure_score_post + cumulative_score_pre + cumulative_score_post - penalty
        self.composite_score += reward

        # 如果有父节点，则根据最新的 composite_score/visits 刷新 uct_cache（用于下一次选择）
        if self.parent is not None:
            self.update_uct(exploration_weight)


# -----------------------------
# MCTS 主类：包含选择、扩展、模拟、回传以及运行循环
# -----------------------------
class MCTS:
    def __init__(self, all_rows, distance_bins_list, angle_bins, dihedral_bins, calculator,
                 n_iterations=10,
                 log_dir="mcts_logs",
                 lattice=None,
                 pbc=None):
        # Problem definition：离散化后的所有选择空间与 bins
        self.all_rows = all_rows
        self.distance_bins_list = distance_bins_list
        self.angle_bins = angle_bins
        self.dihedral_bins = dihedral_bins
        self.calculator = calculator

        # 运行参数
        self.n_iterations = n_iterations
        self.global_distance_bins = distance_bins_list[1] if len(distance_bins_list) > 1 else None

        # lattice/pbc 在当前流程中并未使用（占位）
        self.lattice = None
        self.pbc = False

        # 根节点：引用 all_rows[0] 的第一个选择（通常是第一个原子的元素符号）
        first_atom_selection = all_rows[0][0]
        self.root = MCTSNode(0, first_atom_selection)

        # 日志文件夹与文件准备（会创建若不存在）
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "zmat_paths"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "optimized_structures"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "initial_structures"), exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = open(os.path.join(log_dir, f"mcts_log_{timestamp}.txt"), "w")
        self.log_file.write("Iteration,TotalEnergyPre,TotalEnergyPost,Path,ZMatrix\n")
        self.score_log_file = open(os.path.join(log_dir, f"score_log_{timestamp}.txt"), "w")
        self.score_log_file.write("Iteration,TotalEnergyPre,TotalEnergyPost,AtomicEnergiesPre,AtomicEnergiesPost\n")

        # 新增异常原子日志文件
        self.abnormal_log_file = open(os.path.join(log_dir, f"abnormal_log_{timestamp}.txt"), "w")
        self.abnormal_log_file.write("Iteration,AbnormalAtoms,Reason\n")

        self.time_log_file = open(os.path.join(log_dir, f"time_log_{timestamp}.txt"), "w")
        self.time_log_file.write("Iteration,Phase,Time\n")
        self.iteration_times = []

    def get_path_from_root(self, node):
        """
        从给定节点向上追溯到根节点，收集每个节点的 selection（从根到叶的顺序）
        - 返回 selection tuple 列表（用于构建完整的 Z-matrix）
        """
        path = []
        current = node
        while current:
            path.append(current.selection)
            current = current.parent
        path.reverse()
        return path

    # expand_node: 将给定节点的下一行所有可能 selection 都作为子节点一次性创建
    def expand_node(self, node):
        """
        扩展 node：如果 node 不是最后一行且还未扩展：
        - next_row_idx = node.row_idx + 1
        - 遍历 all_rows[next_row_idx] 中所有 selection，为每个 selection 创建 MCTSNode 并 add_child
        - 标记 node.expanded = True
        注意：这里采用"完整展开下一层"的策略（非渐进式随机扩展）
        """
        if node.row_idx >= len(self.all_rows) - 1 or node.expanded:
            return
        next_row_idx = node.row_idx + 1
        possible_selections = self.all_rows[next_row_idx]
        for selection in possible_selections:
            # 直接复用 selection 的引用（不做深拷贝）
            new_node = MCTSNode(next_row_idx, selection, parent=node)
            node.add_child(new_node)
        node.expanded = True

    # simulation: 从叶节点构建完整路径并评估能量（结构优化 + 能量计算）
    def simulation(self, node, iteration):
        """
        给定叶节点 node（select_path 返回的叶节点），构建从根到叶的完整 path（selection 列表）；
        - 若 path 长度 != len(all_rows)：视为失败，返回 inf 能量惩罚
        - 构建 zmat_lines（调用 build_zmat_line_from_selection）
        - 通过 zmat_to_xyz_with_values 转成 ASE Atoms
        - 计算优化前能量
        - 对 Atoms 做结构优化（FIRE），然后计算优化后能量
        - 发生任何异常或能量异常时记录异常原子
        - 对于异常结构，不保存文件
        """
        path = self.get_path_from_root(node)
        if len(path) != len(self.all_rows):
            # 若路径未达到全长，返回惩罚（无法构建完整结构）
            print(f"错误：路径不完整，期望 {len(self.all_rows)} 个原子，实际 {len(path)} 个原子")
            return float('inf'), float('inf'), np.full(len(self.all_rows), float('inf')), np.full(len(self.all_rows),
                                                                                                  float(
                                                                                                      'inf')), [], path

        # 构建 Z-matrix 文本行
        zmat_lines = []
        for i, selection in enumerate(path):
            zmat_lines.append(build_zmat_line_from_selection(i, selection, self.distance_bins_list, self.angle_bins,
                                                             self.dihedral_bins))

        # 保存 zmat 路径以便复现
        zmat_path_file = os.path.join(self.log_dir, "zmat_paths", f"iteration_{iteration:04d}.zmat")
        with open(zmat_path_file, "w") as f:
            f.write("\n".join(zmat_lines))

        # 转换 Z-matrix 到 Cartesian Atoms
        atoms = zmat_to_xyz_with_values(zmat_lines, lattice=self.lattice, pbc=self.pbc, scale_factor=1.0)
        if atoms is None:
            # 转换失败：给予惩罚并记录日志
            print(f"迭代 {iteration}: Z矩阵转换失败，使用高能量惩罚")
            total_energy_pre = float('inf')
            total_energy_post = float('inf')
            atomic_energies_pre = np.full(len(self.all_rows), float('inf'))
            atomic_energies_post = np.full(len(self.all_rows), float('inf'))
            abnormal_atoms = list(range(len(self.all_rows)))  # 所有原子都标记为异常
            path_str = " -> ".join([str(p) for p in path])
            zmat_str = " | ".join(zmat_lines)
            self.log_file.write(
                f"{iteration},{total_energy_pre:.6f},{total_energy_post:.6f},\"{path_str}\",\"{zmat_str}\"\n")
            self.log_file.flush()
            self.abnormal_log_file.write(f"{iteration},\"all\",\"Z-matrix conversion failed\"\n")
            self.abnormal_log_file.flush()
            return total_energy_pre, total_energy_post, atomic_energies_pre, atomic_energies_post, abnormal_atoms, path

        # 保存初始结构（转换后但优化前）
        initial_structure_file = os.path.join(self.log_dir, "initial_structures", f"iteration_{iteration:04d}.xyz")
        try:
            write(initial_structure_file, atoms)
        except Exception as e:
            print(f"迭代 {iteration}: 保存初始结构时出错: {e}")

        # 初始化异常原子列表和原因
        abnormal_atoms = []
        abnormal_reason = ""

        try:
            # 绑定计算器并计算优化前能量
            atoms.calc = self.calculator
            atoms.set_cell(None)
            atoms.set_pbc(False)

            # 计算优化前能量
            energy_start = time.time()
            total_energy_pre = atoms.get_potential_energy()
            atomic_energies_pre = atoms.get_potential_energies()
            energy_time = time.time() - energy_start
            self.time_log_file.write(f"{iteration},PreOptimizationEnergy,{energy_time:.6f}\n")
            self.time_log_file.flush()

            # 检查优化前能量异常
            if np.isnan(total_energy_pre) or np.any(np.isnan(atomic_energies_pre)) or abs(total_energy_pre) > 1e4:
                abnormal_atoms = list(range(len(atoms)))
                abnormal_reason = "Pre-optimization energy abnormal"
                print(f"迭代 {iteration}: 优化前能量异常 {total_energy_pre}，跳过优化")

                # 记录异常
                self.abnormal_log_file.write(f"{iteration},\"{abnormal_atoms}\",\"{abnormal_reason}\"\n")
                self.abnormal_log_file.flush()

                # 不进行优化，直接返回
                path_str = " -> ".join([str(p) for p in path])
                zmat_str = " | ".join(zmat_lines)
                self.log_file.write(
                    f"{iteration},{total_energy_pre:.6f},{total_energy_pre:.6f},\"{path_str}\",\"{zmat_str}\"\n")
                self.log_file.flush()
                atomic_energies_pre_str = " ".join([f"{e:.6f}" for e in atomic_energies_pre])
                self.score_log_file.write(
                    f"{iteration},{total_energy_pre:.6f},{total_energy_pre:.6f},\"{atomic_energies_pre_str}\",\"{atomic_energies_pre_str}\"\n")
                self.score_log_file.flush()

                # 删除已保存的初始结构文件
                try:
                    os.remove(initial_structure_file)
                except Exception:
                    pass

                return total_energy_pre, total_energy_pre, atomic_energies_pre, atomic_energies_pre, abnormal_atoms, path

            # 执行结构优化
            opt_start = time.time()
            opt = FIRE(atoms=atoms, trajectory=None, logfile=None)
            opt.run(fmax=0.05, steps=2000)  # 进行几何优化
            opt_time = time.time() - opt_start
            self.time_log_file.write(f"{iteration},StructureOptimization,{opt_time:.6f}\n")
            self.time_log_file.flush()

            if not opt.converged():
                # 优化未收敛：标记为异常
                abnormal_atoms = list(range(len(atoms)))
                abnormal_reason = "Optimization not converged"
                print(f"迭代 {iteration}: 结构优化未收敛，标记为异常")

            # 计算优化后能量
            energy_start = time.time()
            total_energy_post = atoms.get_potential_energy()
            atomic_energies_post = atoms.get_potential_energies()
            energy_time = time.time() - energy_start
            self.time_log_file.write(f"{iteration},PostOptimizationEnergy,{energy_time:.6f}\n")
            self.time_log_file.flush()

            # 检查优化后能量异常
            if np.isnan(total_energy_post) or np.any(np.isnan(atomic_energies_post)) or abs(total_energy_post) > 1e4:
                # 找出具体哪些原子能量异常
                abnormal_atoms = []
                for i, e in enumerate(atomic_energies_post):
                    if np.isnan(e) or np.isinf(e) or abs(e) > 1000:  # 设置原子能量异常阈值
                        abnormal_atoms.append(i)

                if not abnormal_atoms:  # 如果没有找到具体异常原子，标记所有原子
                    abnormal_atoms = list(range(len(atoms)))

                abnormal_reason = "Post-optimization energy abnormal"
                print(f"迭代 {iteration}: 优化后能量异常 {total_energy_post}，异常原子: {abnormal_atoms}")

        except Exception as e:
            # 在优化或能量计算过程中发生异常：标记所有原子为异常
            print(f"迭代 {iteration}: 优化过程中出错: {e}")
            total_energy_pre = float('inf')
            total_energy_post = float('inf')
            atomic_energies_pre = np.full(len(self.all_rows), float('inf'))
            atomic_energies_post = np.full(len(self.all_rows), float('inf'))
            abnormal_atoms = list(range(len(self.all_rows)))
            abnormal_reason = f"Optimization error: {str(e)}"

        # 记录异常信息
        if abnormal_atoms:
            self.abnormal_log_file.write(f"{iteration},\"{abnormal_atoms}\",\"{abnormal_reason}\"\n")
            self.abnormal_log_file.flush()

            # 不保存优化后的结构文件
            try:
                if os.path.exists(initial_structure_file):
                    os.remove(initial_structure_file)
            except Exception:
                pass
        else:
            # 保存优化后的结构（若无异常）
            structure_file = os.path.join(self.log_dir, "optimized_structures", f"iteration_{iteration:04d}.xyz")
            try:
                write(structure_file, atoms)
            except Exception as e:
                print(f"迭代 {iteration}: 保存优化结构时出错: {e}")

        # 记录日志（路径、zmat、能量）
        path_str = " -> ".join([str(p) for p in path])
        zmat_str = " | ".join(zmat_lines)
        self.log_file.write(
            f"{iteration},{total_energy_pre:.6f},{total_energy_post:.6f},\"{path_str}\",\"{zmat_str}\"\n")
        self.log_file.flush()

        atomic_energies_pre_str = " ".join([f"{e:.6f}" for e in atomic_energies_pre])
        atomic_energies_post_str = " ".join([f"{e:.6f}" for e in atomic_energies_post])
        self.score_log_file.write(
            f"{iteration},{total_energy_pre:.6f},{total_energy_post:.6f},\"{atomic_energies_pre_str}\",\"{atomic_energies_post_str}\"\n")
        self.score_log_file.flush()

        return total_energy_pre, total_energy_post, atomic_energies_pre, atomic_energies_post, abnormal_atoms, path

    def backpropagation(self, node, total_energy_pre, total_energy_post, atomic_energies_pre, atomic_energies_post,
                        abnormal_atoms):
        """
        把 simulation 得到的能量回传到根节点：
        - 先把无效能量替换为大的惩罚值（若是 inf / NaN）
        - 从叶节点沿 parent 指针回溯到根，收集路径节点列表并反转为 root->leaf 顺序
        - 计算 cumulative_atomic_energies = np.cumsum(atomic_energies)
        - 对路径上的每个节点 i，调用 node.update() 并传递异常原子信息
        """
        if np.isinf(total_energy_pre) or np.any(np.isinf(atomic_energies_pre)) or np.isnan(total_energy_pre):
            total_energy_pre = 1e6
            atomic_energies_pre = np.full(len(self.all_rows), 1e5)

        if np.isinf(total_energy_post) or np.any(np.isinf(atomic_energies_post)) or np.isnan(total_energy_post):
            total_energy_post = 1e6
            atomic_energies_post = np.full(len(self.all_rows), 1e5)

        # 收集从根到当前叶节点的节点列表（顺序为 root -> leaf）
        current = node
        path = []
        while current:
            path.append(current)
            current = current.parent
        path.reverse()

        cumulative_atomic_energies_pre = np.cumsum(atomic_energies_pre)
        cumulative_atomic_energies_post = np.cumsum(atomic_energies_post)

        for i, nd in enumerate(path):
            cumulative_atomic_energy_pre = cumulative_atomic_energies_pre[i]
            cumulative_atomic_energy_post = cumulative_atomic_energies_post[i]
            nd.update(total_energy_pre, total_energy_post, cumulative_atomic_energy_pre, cumulative_atomic_energy_post,
                      abnormal_atoms)

    def best_child(self, node, exploration_weight=0.1):
        """
        在 node 的 children 中选择 UCT 最大的子节点：
        - 先对每个 child 调用 child.update_uct(exploration_weight) 统一刷新缓存
        - 把 uct 值取成 numpy 数组，取最大值并找出所有等于最大值的索引
        - 若有多个并列最大，则随机从这些并列中抽取一个（random tie-break）
        """
        if not node.children:
            return None
        # 更新每个子节点的 UCT 缓存以保证一致性
        for child in node.children:
            child.update_uct(exploration_weight)
        uct_values_array = np.array([child.uct_value() for child in node.children])
        max_uct = np.max(uct_values_array)
        best_indices = np.where(uct_values_array == max_uct)[0]
        # 若多个并列，随机选择一个以避免偏置
        return random.choice([node.children[i] for i in best_indices])

    def select_path(self):
        """
        从 self.root 开始向下选择节点直到叶节点（row_idx == 最后一行）：
        - 若当前节点未扩展，先 expand_node(current)
        - 若当前无子节点（极端情况），fallback: 随机创建一个子节点
        - 否则通过 best_child(current) 选出下一步
        - 返回 (path, leaf_node) 其中 path 是 selection 的列表（从 root 到 leaf）
        """
        path = []
        current = self.root
        path.append(current.selection)

        # 逐行向下直到最后一行
        while current.row_idx < len(self.all_rows) - 1:
            # 若当前节点还没展开，则做一次完整的展开（产生所有下一行子节点）
            if not current.expanded:
                expand_start = time.time()
                self.expand_node(current)
                expand_time = time.time() - expand_start
                self.time_log_file.write(f"{len(self.iteration_times)},NodeExpansion,{expand_time:.6f}\n")
                self.time_log_file.flush()

            # 极端：如果没有子节点（例如 all_rows[next] 为空），则创建一个随机后代作为 fallback
            if not current.children:
                next_selection = random.choice(self.all_rows[current.row_idx + 1])
                next_node = MCTSNode(current.row_idx + 1, next_selection, parent=current)
                current.add_child(next_node)
                current = next_node
                path.append(current.selection)
                continue

            # 正常通过 UCT 选择最优子节点（带 tie-break）
            next_node = self.best_child(current)
            if next_node is None:
                next_node = random.choice(current.children)
            current = next_node
            path.append(current.selection)

        return path, current

    def run(self):
        """
        主循环：执行 n_iterations 次 MCTS 迭代，每次包含：
        - selection -> 得到叶节点
        - simulation -> 基于叶节点构建并优化结构，返回能量
        - backpropagation -> 把能量回传并更新节点统计
        迭代结束后从根节点沿最被访问的分支恢复 best_path
        """
        for i in range(self.n_iterations):
            iteration_start = time.time()

            # selection
            select_start = time.time()
            path, leaf_node = self.select_path()
            select_time = time.time() - select_start
            self.time_log_file.write(f"{i},Selection,{select_time:.6f}\n")
            self.time_log_file.flush()

            # simulation
            simulation_start = time.time()
            total_energy_pre, total_energy_post, atomic_energies_pre, atomic_energies_post, abnormal_atoms, full_path = self.simulation(
                leaf_node, i)
            simulation_time = time.time() - simulation_start
            self.time_log_file.write(f"{i},Simulation,{simulation_time:.6f}\n")
            self.time_log_file.flush()

            # backprop
            backprop_start = time.time()
            self.backpropagation(leaf_node, total_energy_pre, total_energy_post, atomic_energies_pre,
                                 atomic_energies_post, abnormal_atoms)
            backprop_time = time.time() - backprop_start
            self.time_log_file.write(f"{i},Backpropagation,{backprop_time:.6f}\n")
            self.time_log_file.flush()

            iteration_time = time.time() - iteration_start
            self.iteration_times.append(iteration_time)
            self.time_log_file.write(f"{i},TotalIteration,{iteration_time:.6f}\n")
            self.time_log_file.flush()

            # 控制台输出当前迭代信息（能量/耗时）
            print(
                f"Iteration {i + 1}/{self.n_iterations}, Pre-Energy: {total_energy_pre:.6f} eV, Post-Energy: {total_energy_post:.6f} eV, Time: {iteration_time:.2f}s")

        # 关闭日志文件
        self.log_file.close()
        self.score_log_file.close()
        self.abnormal_log_file.close()
        self.time_log_file.close()

        # 恢复 best_path：从根节点每步选择访问次数最多的子节点（贪心）
        best_path = []
        current = self.root
        best_path.append(current.selection)
        while current.children:
            best_child = max(current.children, key=lambda child: child.visits)
            current = best_child
            best_path.append(current.selection)

        if current.visits > 0:
            best_score = current.composite_score / current.visits
        else:
            best_score = float('inf')

        return best_path, best_score


# -----------------------------
# 保存离散化的 Z-matrix（all_rows）到 CSV，便于人工检查或后续加载
# -----------------------------
def save_discretized_zmat_to_csv(all_rows, distance_bins_list, angle_bins, dihedral_bins, global_distance_bins,
                                 filename="discretized_zmatrix.csv"):
    """
    把 all_rows 的离散表示保存到 CSV，每一行对应原子行，列包含每个候选节点的类型、bond 区间、angle 区间、dihedral 区间
    - 有助于离散空间的可视化与后续用 load_discretized_zmat_from_csv 重新加载
    """
    max_nodes = max(len(row) for row in all_rows)
    data = []
    for row_idx, row_combos in enumerate(all_rows):
        element = row_combos[0][0] if row_combos else ""
        row_data = {
            "Row": row_idx + 1,
            "Element": element
        }
        for node_idx, node in enumerate(row_combos):
            node_type = node[0]
            bond_info = ""
            angle_info = ""
            dihedral_info = ""
            if len(node) > 1 and global_distance_bins is not None:
                bond_idx = node[1]
                bond_info = f"{global_distance_bins[bond_idx]:.3f}-{global_distance_bins[bond_idx + 1]:.3f}"
            if len(node) > 2:
                angle_idx = node[2]
                angle_info = f"{angle_bins[angle_idx]:.3f}-{angle_bins[angle_idx + 1]:.3f}"
            if len(node) > 3:
                dihedral_idx = node[3]
                dihedral_info = f"{dihedral_bins[dihedral_idx]:.3f}-{dihedral_bins[dihedral_idx + 1]:.3f}"
            row_data[f"Node_{node_idx + 1}_Type"] = node_type
            row_data[f"Node_{node_idx + 1}_Bond"] = bond_info
            row_data[f"Node_{node_idx + 1}_Angle"] = angle_info
            row_data[f"Node_{node_idx + 1}_Dihedral"] = dihedral_info
        # 填充空白列
        for i in range(len(row_combos), max_nodes):
            row_data[f"Node_{i + 1}_Type"] = ""
            row_data[f"Node_{i + 1}_Bond"] = ""
            row_data[f"Node_{i + 1}_Angle"] = ""
            row_data[f"Node_{i + 1}_Dihedral"] = ""
        data.append(row_data)
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"离散化Z矩阵已保存到 {filename}")


# -----------------------------
# 从 CSV 恢复离散化 Z-matrix（load）
# -----------------------------
def load_discretized_zmat_from_csv(filename):
    """
    从上面保存的 CSV 恢复 all_rows、distance_bins_list、angle_bins、dihedral_bins、global_distance_bins
    - 对缺失的索引或区间采用随机回退
    """
    with Timer("读取CSV文件"):
        df = pd.read_csv(filename)
    df = df.sort_values('Row')
    node_cols = [col for col in df.columns if col.startswith('Node_') and '_Type' in col]
    max_nodes = len(node_cols)
    angle_bins = create_fixed_bins('angle', 10)
    dihedral_bins = create_fixed_bins('dihedral', 10)
    global_distance_bins = None

    # 尝试从第一行的 Node_1_Bond 列推断全局 distance bins 的边界（采用第一个出现的区间）
    for index, row in df.iterrows():
        bond_col = f'Node_1_Bond'
        if pd.notna(row.get(bond_col)) and row[bond_col] != '':
            bond_str = row[bond_col]
            parts = bond_str.split('-')
            if len(parts) == 2:
                try:
                    min_val = float(parts[0])
                    max_val = float(parts[1])
                    global_distance_bins = np.linspace(min_val, max_val, 6)
                    break
                except ValueError:
                    continue
    if global_distance_bins is None:
        global_distance_bins = create_global_distance_bins([], 5)

    distance_bins_list = []
    for index, row in df.iterrows():
        row_idx = int(row['Row']) - 1
        if row_idx == 0:
            distance_bins_list.append(None)
        else:
            distance_bins_list.append(global_distance_bins)

    all_rows = []
    for index, row in df.iterrows():
        row_nodes = []
        row_idx = int(row['Row']) - 1
        for i in range(1, max_nodes + 1):
            type_col = f'Node_{i}_Type'
            bond_col = f'Node_{i}_Bond'
            angle_col = f'Node_{i}_Angle'
            dihedral_col = f'Node_{i}_Dihedral'
            if pd.notna(row.get(type_col)) and row[type_col] != '':
                element = row[type_col]
                bond_str = row[bond_col] if pd.notna(row.get(bond_col)) else ''
                angle_str = row[angle_col] if pd.notna(row.get(angle_col)) else ''
                dihedral_str = row[dihedral_col] if pd.notna(row.get(dihedral_col)) else ''

                def parse_interval(interval_str):
                    if interval_str == '':
                        return None
                    parts = interval_str.split('-')
                    if len(parts) == 2:
                        try:
                            return float(parts[0]), float(parts[1])
                        except ValueError:
                            return None
                    return None

                bond_idx = None
                if bond_str != '' and global_distance_bins is not None:
                    low_high = parse_interval(bond_str)
                    if low_high is not None:
                        low, high = low_high
                        for idx in range(len(global_distance_bins) - 1):
                            if abs(global_distance_bins[idx] - low) < 1e-5 and abs(
                                    global_distance_bins[idx + 1] - high) < 1e-5:
                                bond_idx = idx
                                break
                angle_idx = None
                if angle_str != '':
                    low_high = parse_interval(angle_str)
                    if low_high is not None:
                        low, high = low_high
                        for idx in range(len(angle_bins) - 1):
                            if abs(angle_bins[idx] - low) < 1e-5 and abs(angle_bins[idx + 1] - high) < 1e-5:
                                angle_idx = idx
                                break
                dihedral_idx = None
                if dihedral_str != '':
                    low_high = parse_interval(dihedral_str)
                    if low_high is not None:
                        low, high = low_high
                        for idx in range(len(dihedral_bins) - 1):
                            if abs(dihedral_bins[idx] - low) < 1e-5 and abs(dihedral_bins[idx + 1] - high) < 1e-5:
                                dihedral_idx = idx
                                break

                # 根据行号构建 tuple（若缺失索引则用随机回退）
                if row_idx == 0:
                    node = (element,)
                elif row_idx == 1:
                    if bond_idx is not None:
                        node = (element, bond_idx)
                    else:
                        node = (element, random.randint(0, len(global_distance_bins) - 2))
                elif row_idx == 2:
                    if bond_idx is not None and angle_idx is not None:
                        node = (element, bond_idx, angle_idx)
                    else:
                        node = (element, random.randint(0, len(global_distance_bins) - 2),
                                random.randint(0, len(angle_bins) - 2))
                else:
                    if bond_idx is not None and angle_idx is not None and dihedral_idx is not None:
                        node = (element, bond_idx, angle_idx, dihedral_idx)
                    else:
                        node = (element, random.randint(0, len(global_distance_bins) - 2),
                                random.randint(0, len(angle_bins) - 2),
                                random.randint(0, len(dihedral_bins) - 2))
                row_nodes.append(node)
        all_rows.append(row_nodes)
    return all_rows, distance_bins_list, angle_bins, dihedral_bins, global_distance_bins


# -----------------------------
# main_from_xyz：从 xyz 文件开始完整运行 MCTS 流程
# -----------------------------
def main_from_xyz(xyz_file):
    n_bins = 10
    n_iterations = 50000  # 默认迭代次数（可根据需要调整）
    log_dir = "mcts_logs"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    time_log_file = open(f"global_time_log_{timestamp}.txt", "w")
    time_log_file.write("Step,Time\n")

    print(f"从 {xyz_file} 读取结构...")
    with Timer("读取XYZ文件") as t:
        atoms = read_structure_from_xyz(xyz_file)
    time_log_file.write(f"ReadStructure,{t.elapsed:.6f}\n")
    if atoms is None:
        print("无法读取XYZ文件，退出程序")
        time_log_file.close()
        return

    lattice = None
    pbc = False

    with Timer("保存初始结构") as t:
        write('initial_structure.xyz', atoms)
    time_log_file.write(f"SaveInitialStructure,{t.elapsed:.6f}\n")

    print("加载势函数计算器...")
    with Timer("加载计算器") as t:
        try:
            if NequIPCalculator is not None:
                # 若 NequIP 可用，直接加载模型（请确保模型路径与设备配置正确）
                calculator = NequIPCalculator.from_deployed_model(
                    model_path='C.nequip.pth',
                    device='cuda'
                )
            else:
                # 回退到 DummyCalculator（用于测试或无 model 环境）
                raise RuntimeError("NequIPCalculator 未安装")
        except Exception as e:
            print(f"加载计算器时出错: {e}")
            print("使用哑计算器进行测试")
            from ase.calculators.calculator import Calculator, all_changes

            class DummyCalculator(Calculator):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)

                def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
                    super().calculate(atoms, properties, system_changes)
                    # 固定返回值（便于调试）
                    self.results = {'energy': -100.0, 'energies': np.full(len(atoms), -10.0)}

            calculator = DummyCalculator()
    time_log_file.write(f"LoadCalculator,{t.elapsed:.6f}\n")

    # 对初始结构做一次优化并保存（作为参考）
    print("优化初始结构...")
    atoms.calc = calculator
    atoms.set_cell(None)
    atoms.set_pbc(False)
    with Timer("初始结构优化") as t:
        opt = FIRE(atoms=atoms, trajectory=None, logfile=None)
        opt.run(fmax=0.05, steps=2000)
    time_log_file.write(f"InitialOptimization,{t.elapsed:.6f}\n")

    with Timer("保存优化后的初始结构") as t:
        write('optimized_initial.xyz', atoms)
    time_log_file.write(f"SaveOptimizedInitial,{t.elapsed:.6f}\n")
    print("初始结构优化完成，已保存到 optimized_initial.xyz")

    # 转换为 Z-matrix（并写入 initial_optimized.zmat）
    print("转换为Z矩阵...")
    with Timer("转换为Z矩阵") as t:
        zmat_lines, lattice, pbc = xyz_to_zmat(atoms)
    time_log_file.write(f"ConvertToZMatrix,{t.elapsed:.6f}\n")

    with Timer("保存Z矩阵") as t:
        with open("initial_optimized.zmat", "w") as f:
            f.write("#ZMATRIX\n#\n")
            f.write("\n".join(zmat_lines))
    time_log_file.write(f"SaveZMatrix,{t.elapsed:.6f}\n")

    # 离散化 Z-matrix（得到 all_rows）
    print("离散化Z矩阵...")
    with Timer("离散化Z矩阵") as t:
        all_rows, distance_bins_list, angle_bins, dihedral_bins, global_distance_bins = parse_zmat(zmat_lines, n_bins)
    time_log_file.write(f"DiscretizeZMatrix,{t.elapsed:.6f}\n")

    # 保存到 CSV 以便后续或人工检查
    with Timer("保存离散化Z矩阵到CSV") as t:
        save_discretized_zmat_to_csv(all_rows, distance_bins_list, angle_bins, dihedral_bins, global_distance_bins,
                                     "discretized_zmatrix.csv")
    time_log_file.write(f"SaveDiscretizedZMatrix,{t.elapsed:.6f}\n")

    # 运行 MCTS 主流程
    print("运行MCTS优化...")
    with Timer("MCTS运行") as t:
        mcts = MCTS(all_rows, distance_bins_list, angle_bins, dihedral_bins, calculator, n_iterations,
                    log_dir, lattice, pbc)
        best_path, best_score = mcts.run()
    time_log_file.write(f"MCTSRun,{t.elapsed:.6f}\n")

    # 根据 best_path 生成结构并保存
    print("生成最佳结构...")
    zmat_lines_best = []
    with Timer("构建最佳Z矩阵") as t:
        for i, selection in enumerate(best_path):
            zmat_lines_best.append(
                build_zmat_line_from_selection(i, selection, distance_bins_list, angle_bins, dihedral_bins))
    time_log_file.write(f"BuildBestZMatrix,{t.elapsed:.6f}\n")

    with Timer("Z矩阵到XYZ转换") as t:
        best_atoms = zmat_to_xyz_with_values(zmat_lines_best, lattice=None, pbc=False)
    time_log_file.write(f"ConvertBestZMatrixToXYZ,{t.elapsed:.6f}\n")

    with Timer("保存最佳结构") as t:
        write("optimized_structure.xyz", best_atoms)
    time_log_file.write(f"SaveBestStructure,{t.elapsed:.6f}\n")

    # 保存 summary JSON
    with Timer("保存优化结果") as t:
        result = {
            "best_score": best_score,
            "best_path": [str(node) for node in best_path],
            "distance_bins_list": [bins.tolist() if bins is not None else None for bins in distance_bins_list],
            "angle_bins": angle_bins.tolist(),
            "dihedral_bins": dihedral_bins.tolist(),
            "global_distance_bins": global_distance_bins.tolist()
        }
        with open("mcts_results.json", "w") as f:
            json.dump(result, f, indent=2)
    time_log_file.write(f"SaveResults,{t.elapsed:.6f}\n")
    time_log_file.close()

    print("优化完成！最佳能量:", best_score)
    print("最佳结构已保存到 optimized_structure.xyz")
    print("详细结果已保存到 mcts_results.json")
    print(f"所有迭代的Z矩阵路径已保存到 {log_dir}/zmat_paths/")
    print(f"所有迭代的初始结构已保存到 {log_dir}/initial_structures/")
    print(f"所有迭代的优化结构已保存到 {log_dir}/optimized_structures/")


# -----------------------------
# main_from_csv：从已离散化的 CSV 恢复并运行 MCTS（与 main_from_xyz 类似）
# -----------------------------
def main_from_csv(csv_file, n_iterations=10, log_dir="mcts_logs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    time_log_file = open(f"global_time_log_{timestamp}.txt", "w")
    time_log_file.write("Step,Time\n")

    print("从CSV加载离散化Z矩阵...")
    with Timer("从CSV加载离散化Z矩阵") as t:
        all_rows, distance_bins_list, angle_bins, dihedral_bins, global_distance_bins = load_discretized_zmat_from_csv(
            csv_file)
    time_log_file.write(f"LoadDiscretizedZMatrixFromCSV,{t.elapsed:.6f}\n")

    print("加载势函数计算器...")
    with Timer("加载计算器") as t:
        try:
            if NequIPCalculator is not None:
                calculator = NequIPCalculator.from_deployed_model(
                    model_path='C.nequip.pth',
                    device='cuda'
                )
            else:
                raise RuntimeError("NequIPCalculator 未安装")
        except Exception as e:
            print(f"加载计算器时出错: {e}")
            print("使用哑计算器进行测试")
            from ase.calculators.calculator import Calculator, all_changes

            class DummyCalculator(Calculator):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)

                def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
                    super().calculate(atoms, properties, system_changes)
                    self.results = {'energy': -100.0, 'energies': np.full(len(atoms), -10.0)}

            calculator = DummyCalculator()
    time_log_file.write(f"LoadCalculator,{t.elapsed:.6f}\n")

    print("运行MCTS优化...")
    with Timer("MCTS运行") as t:
        mcts = MCTS(all_rows, distance_bins_list, angle_bins, dihedral_bins, calculator, n_iterations, log_dir, None,
                    False)
        best_path, best_score = mcts.run()
    time_log_file.write(f"MCTSRun,{t.elapsed:.6f}\n")

    print("生成最佳结构...")
    zmat_lines_best = []
    with Timer("构建最佳Z矩阵") as t:
        for i, selection in enumerate(best_path):
            zmat_lines_best.append(
                build_zmat_line_from_selection(i, selection, distance_bins_list, angle_bins, dihedral_bins))
    time_log_file.write(f"BuildBestZMatrix,{t.elapsed:.6f}\n")

    with Timer("Z矩阵到XYZ转换") as t:
        best_atoms = zmat_to_xyz_with_values(zmat_lines_best, lattice=None, pbc=False)
    time_log_file.write(f"ConvertBestZMatrixToXYZ,{t.elapsed:.6f}\n")

    with Timer("保存最佳结构") as t:
        write("optimized_structure.xyz", best_atoms)
    time_log_file.write(f"SaveBestStructure,{t.elapsed:.6f}\n")

    with Timer("保存优化结果") as t:
        result = {
            "best_score": best_score,
            "best_path": [str(node) for node in best_path],
            "distance_bins_list": [bins.tolist() if bins is not None else None for bins in distance_bins_list],
            "angle_bins": angle_bins.tolist(),
            "dihedral_bins": dihedral_bins.tolist(),
            "global_distance_bins": global_distance_bins.tolist()
        }
        with open("mcts_results.json", "w") as f:
            json.dump(result, f, indent=2)
    time_log_file.write(f"SaveResults,{t.elapsed:.6f}\n")
    time_log_file.close()

    print("优化完成！最佳能量:", best_score)
    print("最佳结构已保存到 optimized_structure.xyz")
    print("详细结果已保存到 mcts_results.json")
    print(f"所有迭代的Z矩阵路径已保存到 {log_dir}/zmat_paths/")
    print(f"所有迭代的初始结构已保存到 {log_dir}/initial_structures/")
    print(f"所有迭代的优化结构已保存到 {log_dir}/optimized_structures/")


# -----------------------------
# CLI 入口：支持 --xyz 或 --csv 两种启动方式
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MCTS for structure optimization')
    parser.add_argument('--xyz', type=str, help='Input XYZ file containing the initial structure')
    parser.add_argument('--csv', type=str, help='Input CSV file of discretized zmatrix')
    args = parser.parse_args()

    if args.xyz:
        main_from_xyz(args.xyz)
    elif args.csv:
        main_from_csv(args.csv)
    else:
        print("请提供输入文件: 使用 --xyz 指定XYZ文件或 --csv 指定CSV文件")
        print("示例: python 111.py --xyz my_structure.xyz")