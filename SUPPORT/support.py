import subprocess
import os
import shutil
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
        pass
def copyFiles(source_file,dest_folder):
# 源文件路径source_file = '/path/to/source/file.txt'
# 目标文件夹路径dest_folder = '/path/to/destination/folder'
# 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    # 目标文件路径
    dest_file = os.path.join(dest_folder, os.path.basename(source_file))
    # 复制文件
    try:
        shutil.copy2(source_file, dest_file)
        print(f"文件已成功复制到 {dest_file}")
    except IOError as e:
        print(f"无法复制文件. {e}")
    except Exception as e:
        print(f"发生错误: {e}")
def copyFolder(src, dst, exclude_dirs=None):
    """
    复制文件夹，排除指定目录
    
    参数:
        src (str): 源文件夹路径
        dst (str): 目标文件夹路径
        exclude_dirs (list): 要排除的目录列表
    """
    if exclude_dirs is None:
        exclude_dirs = ['__pycache__', '.git', '.idea']
    
    try:
        # 确保目标目录存在
        os.makedirs(dst, exist_ok=True)
        
        # 遍历源文件夹中的所有项目
        for item in os.listdir(src):
            if item in exclude_dirs:
                print(f"跳过目录: {item}")
                continue
                
            source_item = os.path.join(src, item)
            dest_item = os.path.join(dst, item)
            
            if os.path.isdir(source_item):
                # 如果是目录，递归复制
                copyFolder(source_item, dest_item, exclude_dirs)
            else:
                # 如果是文件，直接复制
                shutil.copy2(source_item, dest_item)
        
        print(f"成功复制文件夹: {src} -> {dst}")
        return True
    except Exception as e:
        print(f"复制失败: {e}")
        return False