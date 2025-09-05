import numpy as np
from scipy.spatial import distance_matrix
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.local_env import CrystalNN
import matplotlib.pyplot as plt
import os
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
def calculate_coordination_with_gcn(input_file, pltout=True, output_prefix="output"):
    """
    计算结构文件中所有原子的配位数和广义配位数 (GCN)
    
    参数:
        input_file: 输入结构文件路径 (POSCAR, CIF, 等)
        pltout: 是否生成图表
        output_prefix: 输出文件前缀 (默认为"output")
        cnn_kwargs: CrystalNN 的任意关键字参数
    """
    # 1. 读取结构文件
    try:
        '''-------------------'''
        # 读取单个结构
        ase_atoms = read(input_file, index=-1)  # 读取第一帧

        # 使用适配器转换
        adaptor = AseAtomsAdaptor()
        structure = adaptor.get_structure(ase_atoms)
        '''-------------------'''
        #structure = Structure.from_file(input_file)
        filename = os.path.basename(input_file)
        print(f"成功读取结构文件: {filename}")
        print(f"  晶胞参数: {structure.lattice.parameters}")
        print(f"  原子种类: {', '.join(set(site.species_string for site in structure))}")
        print(f"  原子总数: {len(structure)}")
        print(f"  周期性: {structure.lattice.pbc}")
    except Exception as e:
        print(f"错误: 无法读取结构文件 {input_file}")
        print(f"详细信息: {e}")
        return None
    
    # 2. 初始化 CrystalNN (带默认参数)
    default_params = {
        'weighted_cn': False,
        'cation_anion': False,
        'distance_cutoffs': (0.5, 1),
        'x_diff_weight': 3.0,
        'porous_adjustment': False,
        'search_cutoff': 5.0
    }
    
    # 更新用户提供的参数
    try:
        cnn = CrystalNN(**default_params)
        print("\nCrystalNN 参数配置:")
        for key, value in default_params.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"错误: 初始化 CrystalNN 失败")
        print(f"详细信息: {e}")
        return None
    # 3. 计算所有原子的配位数
    coordination_numbers = []
    nn_info_list = []
    print("\n计算配位数中...")
    ct = []
    for i in range(len(structure)):
        cn = cnn.get_cn(structure, i)
        ct.append(cn)
    cnmax = max(ct)
    # 检查结构是否具有周期性
    has_periodicity = all(structure.lattice.pbc)
    for i in range(len(structure)):
        try:
            # 获取配位数
            cn = cnn.get_cn(structure, i)
            # 处理表面原子（如果适用）
            if has_periodicity:
                first_site = structure[i]
                if 'selective_dynamics' in first_site.properties:
                    all_false_sd = np.all(first_site.properties["selective_dynamics"])
                    if not all_false_sd:
                        cn = cnmax
            coordination_numbers.append(cn)
            # 获取近邻信息 (包含原始索引)
            nn_info = cnn.get_nn_info(structure, i)
            # 添加原始索引到邻居信息中
            for neighbor in nn_info:
                neighbor['site_index'] = neighbor['site_index']  # 保留原始索引
            nn_info_list.append(nn_info)
            
            # 打印进度
            if (i + 1) % 50 == 0 or (i + 1) == len(structure):
                print(f"  已完成 {i+1}/{len(structure)} 个原子")
                
        except Exception as e:
            print(f"  警告: 原子 {i} ({structure[i].species_string}) 计算失败 - {str(e)}")
            coordination_numbers.append(float('nan'))
            nn_info_list.append([])

    # 4. 计算广义配位数 (GCN) - 考虑周期性
    print("\n计算广义配位数 (GCN) 中...")
    if has_periodicity:
        print("结构周期性,标准GCN计算")
        # 使用标准GCN计算（周期性）
        gcn_values = np.zeros(len(structure))
        for i in range(len(structure)):
            first_site = structure[i]
            if 'selective_dynamics' in first_site.properties:
                all_false_sd = np.all(first_site.properties["selective_dynamics"])
            cn_i = coordination_numbers[i]
            if np.isnan(cn_i) or cn_i == 0:
                gcn_values[i] = float('nan')
                continue
                
            gcn_sum = 0
            valid_neighbors = 0
            if all_false_sd ==False:
                gcn_values[i] = cn_i
            else:
                for neighbor in nn_info_list[i]:
                    j = neighbor['site_index']
                    if np.isnan(coordination_numbers[j]) or coordination_numbers[j] == 0:
                        continue
                    gcn_sum += coordination_numbers[j]
                    valid_neighbors += 1
                if valid_neighbors > 0:
                    if 'C' in first_site.species:
                        gcn_values[i] = gcn_sum / 4
                    elif 'O' in first_site.species:
                        gcn_values[i] = gcn_sum / 2
                    elif 'H' in first_site.species:
                        gcn_values[i] = gcn_sum / 1
                    else:
                        gcn_values[i] = gcn_sum / 12
                else:
                    gcn_values[i] = float('nan')
    else:
        print("结构非周期性,标准GCN计算")
        # 使用标准GCN计算（非周期性）
        gcn_values = np.zeros(len(structure))
        for i in range(len(structure)):
            cn_i = coordination_numbers[i]
            if np.isnan(cn_i) or cn_i == 0:
                gcn_values[i] = float('nan')
                continue
                
            gcn_sum = 0
            valid_neighbors = 0
            
            for neighbor in nn_info_list[i]:
                j = neighbor['site_index']
                if np.isnan(coordination_numbers[j]) or coordination_numbers[j] == 0:
                    continue
                    
                gcn_sum += coordination_numbers[j]
                valid_neighbors += 1
            
            if valid_neighbors > 0:
                    if 'C' in first_site.species:
                        gcn_values[i] = gcn_sum / 4
                    elif 'O' in first_site.species:
                        gcn_values[i] = gcn_sum / 2
                    elif 'H' in first_site.species:
                        gcn_values[i] = gcn_sum / 1
                    else:
                        gcn_values[i] = gcn_sum / 12
            else:
                gcn_values[i] = float('nan')
    # 5. 分析结果
    # 按元素类型分组
    element_data = {}
    for i, site in enumerate(structure):
        element = site.species_string
        if element not in element_data:
            element_data[element] = {
                'count': 0,
                'cn_list': [],
                'gcn_list': [],
                'min_cn': float('inf'),
                'max_cn': 0,
                'min_gcn': float('inf'),
                'max_gcn': 0,
                'positions': []
            }
        
        element_data[element]['count'] += 1
        element_data[element]['cn_list'].append(coordination_numbers[i])
        element_data[element]['gcn_list'].append(gcn_values[i])
        element_data[element]['positions'].append(site.coords)
        
        # 更新最小/最大配位数
        if coordination_numbers[i] < element_data[element]['min_cn']:
            element_data[element]['min_cn'] = coordination_numbers[i]
        if coordination_numbers[i] > element_data[element]['max_cn']:
            element_data[element]['max_cn'] = coordination_numbers[i]
        
        # 更新最小/最大GCN
        if not np.isnan(gcn_values[i]) and gcn_values[i] < element_data[element]['min_gcn']:
            element_data[element]['min_gcn'] = gcn_values[i]
        if not np.isnan(gcn_values[i]) and gcn_values[i] > element_data[element]['max_gcn']:
            element_data[element]['max_gcn'] = gcn_values[i]
    
    # 计算统计数据
    valid_cn = [cn for cn in coordination_numbers if not np.isnan(cn)]
    valid_gcn = [gcn for gcn in gcn_values if not np.isnan(gcn)]
    overall_avg_cn = np.mean(valid_cn) if valid_cn else float('nan')
    overall_avg_gcn = np.mean(valid_gcn) if valid_gcn else float('nan')
    
    # 6. 显示结果
    print("\n" + "="*60)
    print("配位数和广义配位数计算完成!")
    print("="*60)
    print(f"结构文件: {input_file}")
    print(f"原子总数: {len(structure)}")
    print(f"整体平均配位数: {overall_avg_cn:.4f}")
    print(f"整体平均GCN: {overall_avg_gcn:.4f}")
    print("\n按元素统计:")
    print("元素 | 原子数 | 平均CN | 平均GCN | CN-GCN差")
    print("-"*60)
    for element, data in element_data.items():
        if data['cn_list']:
            avg_cn = np.nanmean(data['cn_list'])
            avg_gcn = np.nanmean(data['gcn_list'])
            cn_gcn_diff = avg_cn - avg_gcn
            print(f"{element:4} | {data['count']:6} | {avg_cn:6.3f} | {avg_gcn:7.3f} | {cn_gcn_diff:7.3f}")
    
    if pltout:
        # 7. 可视化结果
        plt.figure(figsize=(15, 10))
        
        # CN分布直方图
        plt.subplot(2, 2, 1)
        plt.hist(valid_cn, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Coordination Number (CN)')
        plt.ylabel('Sum of Atoms')
        plt.title('Coordination Number Distribution')
        plt.grid(True, alpha=0.3)
        
        # GCN分布直方图
        plt.subplot(2, 2, 2)
        plt.hist(valid_gcn, bins=20, alpha=0.7, color='salmon', edgecolor='black')
        plt.xlabel('Generalized Coordination Number (GCN)')
        plt.ylabel('Sum of Atoms')
        plt.title('Generalized Coordination Number Distribution')
        plt.grid(True, alpha=0.3)
        
        # 3D空间分布 (CN)
        try:
            ax = plt.subplot(2, 2, 3, projection='3d')
            
            # 获取所有有效原子的坐标
            coords = []
            cn_vals = []
            for i, site in enumerate(structure):
                if not np.isnan(coordination_numbers[i]):
                    coords.append(site.coords)
                    cn_vals.append(coordination_numbers[i])
            
            if coords:
                coords = np.array(coords)
                cn_vals = np.array(cn_vals)
                
                # 创建颜色映射 (CN)
                sc = ax.scatter(coords[:,0], coords[:,1], coords[:,2], 
                              c=cn_vals, cmap='viridis', s=50, 
                              alpha=0.8, label='CN')
                plt.colorbar(sc, ax=ax, label='CN')            
                ax.set_xlabel('X (Å)')
                ax.set_ylabel('Y (Å)')
                ax.set_zlabel('Z (Å)')
                ax.set_title('Atomic Positions - CN')
        except Exception as e:
            print(f"警告: CN 3D 绘图失败 - {str(e)}")
        
        # 3D空间分布 (GCN)
        try:
            ax = plt.subplot(2, 2, 4, projection='3d')
            
            # 获取所有有效原子的坐标
            coords = []
            gcn_vals = []
            for i, site in enumerate(structure):
                if not np.isnan(gcn_values[i]):
                    coords.append(site.coords)
                    gcn_vals.append(gcn_values[i])
            
            if coords:
                coords = np.array(coords)
                gcn_vals = np.array(gcn_vals)
                
                # 创建颜色映射 (GCN)
                sc = ax.scatter(coords[:,0], coords[:,1], coords[:,2], 
                              c=gcn_vals, cmap='viridis', s=50, 
                              alpha=0.8, label='GCN')
                plt.colorbar(sc, ax=ax, label='GCN')
                ax.set_xlabel('X (Å)')
                ax.set_ylabel('Y (Å)')
                ax.set_zlabel('Z (Å)')
                ax.set_title('Atomic Positions - GCN')
        except Exception as e:
            print(f"警告: GCN 3D 绘图失败 - {str(e)}")
        
        plt.tight_layout()
        plot_output = f"{output_prefix}_coordination_gcn_plot.png"
        plt.savefig(plot_output, dpi=300)
        print(f"可视化结果已保存至: {plot_output}")
    
    # 8. 保存结果到文件
    txt_output = f"{output_prefix}_coordination_gcn.txt"
    with open(txt_output, 'w',encoding='utf-8') as f:
        f.write(f"结构文件: {input_file}\n")
        f.write(f"原子总数: {len(structure)}\n")
        f.write(f"晶胞参数: a={structure.lattice.a:.4f} Å, b={structure.lattice.b:.4f} Å, c={structure.lattice.c:.4f} Å\n")
        f.write(f"晶胞角度: α={structure.lattice.alpha:.2f}°, β={structure.lattice.beta:.2f}°, γ={structure.lattice.gamma:.2f}°\n")
        f.write(f"整体平均配位数: {overall_avg_cn:.4f}\n")
        f.write(f"整体平均GCN: {overall_avg_gcn:.4f}\n\n")
        
        f.write("按元素统计:\n")
        f.write("元素 | 原子数 | 平均CN | 平均GCN | CN-GCN差\n")
        f.write("-"*60 + "\n")
        for element, data in element_data.items():
            avg_cn = np.nanmean(data['cn_list'])
            avg_gcn = np.nanmean(data['gcn_list'])
            cn_gcn_diff = avg_cn - avg_gcn
            f.write(f"{element:4} | {data['count']:6} | {avg_cn:6.3f} | {avg_gcn:7.3f} | {cn_gcn_diff:7.3f}\n")
        
        f.write("\n详细原子数据 (分数坐标):\n")
        f.write("索引 | 元素 | x | y | z | CN | GCN | CN-GCN差 | 近邻原子\n")
        f.write("-"*90 + "\n")
        
        # 获取晶格结构用于距离计算
        lattice = structure.lattice

        for i, site in enumerate(structure):
            center_coords = site.coords  # 中心原子坐标
            frac_coords = site.frac_coords
            
            neighbor_info = []
            for n in nn_info_list[i]:  # 遍历所有近邻
                # 计算近邻原子的实际位置（考虑周期性边界）
                neighbor_site = n['site']
                image_offset = n['image']
                
                # 将分数坐标偏移转换为笛卡尔坐标偏移
                cartesian_offset = lattice.get_cartesian_coords(image_offset)
                
                # 计算近邻原子的实际坐标
                neighbor_coords = neighbor_site.coords + cartesian_offset
                
                # 计算距离
                distance = np.linalg.norm(center_coords - neighbor_coords)
                
                neighbor_info.append(f"{neighbor_site.species_string}({distance:.3f}Å)")
            
            neighbors = ", ".join(neighbor_info[:4])  # 只显示前4个近邻
            if len(neighbor_info) > 4:
                neighbors += f" +{len(neighbor_info)-4} more"
            
            cn_gcn_diff = coordination_numbers[i] - gcn_values[i] if not np.isnan(coordination_numbers[i]) and not np.isnan(gcn_values[i]) else float('nan')
            
            f.write(f"{i:5} | {site.species_string:4} | "
                    f"{frac_coords[0]:.6f} | {frac_coords[1]:.6f} | {frac_coords[2]:.6f} | "
                    f"{coordination_numbers[i]:6.3f} | {gcn_values[i]:6.3f} | "
                    f"{cn_gcn_diff:8.3f} | {neighbors}\n")
    
    print(f"详细结果已保存至: {txt_output}")
    
    # 9. 返回结果以便进一步分析
    return {
        'structure': structure,
        'coordination_numbers': coordination_numbers,
        'gcn_values': gcn_values,
        'nn_info': nn_info_list,
        'element_data': element_data,
        'plot': plt,
        'text_output': txt_output,
        'plot_output': plot_output if pltout else None
    }

# 在 Notebook 中使用示例
if __name__ == "__main__":
    # 示例用法
    result = calculate_coordination_with_gcn(
        input_file="opt/system/species/[H]C([H])(O)O/11/nequipOpt.traj",  # 替换为您的结构文件路径
        pltout=True,
        output_prefix="my_analysis")
    
    # 显示图表
    if result:
        result['plot'].show()