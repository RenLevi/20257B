from ase.io import read, write

def split_xyz(input_file, output_prefix="structure"):
    name=['IS','TS','FS']
    with open(input_file, 'r') as f:
        # 读取并过滤空行，保留原始内容（去除行尾换行符）
        lines = [line.rstrip('\n') for line in f if line.strip()]
    
    index = 0
    structure_count = 0
    
    while index < len(lines):
        try:
            # 尝试解析原子数量
            num_atoms = int(lines[index])
        except ValueError:
            print(f"错误：第{index+1}行无法解析原子数量")
            break
        
        # 计算需要读取的行数
        required_lines = num_atoms + 2
        available_lines = len(lines) - index
        
        # 检查行数是否足够
        if available_lines < required_lines:
            print(f"错误：结构{structure_count+1}不完整，需要{required_lines}行，剩余{available_lines}行")
            break
        
        # 提取当前结构内容
        structure_content = lines[index:index+required_lines]
        
        # 生成输出文件名
        output_file = f"{output_prefix}_{name[structure_count]}.xyz"
        
        # 写入文件
        with open(output_file, 'w') as fout:
            fout.write('\n'.join(structure_content) + '\n')
        print(f"已创建文件：{output_file}")
        
        # 移动索引并增加计数器
        index += required_lines
        structure_count += 1
    
    print(f"处理完成，共分割出{structure_count}个结构")

def xyz_to_poscar_ase(xyz_file, poscar_file):
    """
    使用 ASE 将 XYZ 文件转换为 POSCAR 格式
    
    参数:
        xyz_file (str): 输入的 XYZ 文件名
        poscar_file (str): 输出的 POSCAR 文件名
    """
    # 读取 XYZ 文件
    atoms = read(xyz_file)
    
    # 写入 POSCAR 文件
    write(poscar_file, atoms, format='vasp')
    
    print(f"成功将 {xyz_file} 转换为 {poscar_file}")

# 使用示例
if __name__ == "__main__":
    input_path = 'R7.xyz'
    output_base = "structure"
    
    split_xyz(input_path, output_base)
    xyz_to_poscar_ase("structure_TS.xyz", "POSCAR")