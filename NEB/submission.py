
import os
import subprocess
def txt_to_dict(filename):
    dictionary = {}
    with open(filename, 'r') as file:
        for line in file:
            # 移除行首行尾的空白字符
            line = line.strip()
            # 忽略空行和注释行（假设注释行以'#'开头）
            if line and not line.startswith('#'):
                # 分割键和值，最多分割一次
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key, value = parts
                    dictionary[key.strip()] = value.strip()
                else:
                    print(f"无法解析的行：{line}")
    return dictionary
def run_command_in_directory(directory, command):
    """
    在指定目录下运行命令
    
    参数:
        directory (str): 要切换到的目录路径
        command (str or list): 要执行的命令(字符串或列表形式)
    """
    # 保存当前工作目录
    original_dir = os.getcwd()
    
    try:
        # 切换到目标目录
        os.chdir(directory)
        print(f"当前工作目录已切换到: {os.getcwd()}")
        
        # 执行命令
        print(f"正在执行命令: {command}")
        result = subprocess.run(command, shell=isinstance(command, str), 
                               check=True, text=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 输出命令执行结果
        print("命令输出:")
        print(result.stdout)
        if result.stderr:
            print("错误信息:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败，返回码: {e.returncode}")
        print(f"错误信息: {e.stderr}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        # 无论成功与否，都恢复原始工作目录
        os.chdir(original_dir)
        print(f"已恢复工作目录到: {original_dir}")

if __name__ == "__main__":
    # 示例用法
    path1 = '/work/home/ac877eihwp/renyq/sella/cal/neb/foldername.txt'
    path2 = '/work/home/ac877eihwp/renyq/sella/cal/neb/'
    folder_dict = txt_to_dict(path1)
    namelist = list(folder_dict.keys())
    smileslist =list(folder_dict.values())
    for foldername in namelist:
        fp = path2+foldername
        fp = path2+foldername
        target_directory = fp  # 替换为你的目标目录
        command_to_run = "sbatch jobsearchts.sh"  # 替换为你想运行的命令
    
    # 也可以使用列表形式传递命令(推荐，更安全)
    # command_to_run = ["ls", "-l"]
        run_command_in_directory(target_directory, command_to_run)