import shutil
import os
import json
def main():
    # 要删除的文件夹路径
    folder_path = input("请输入要删除的文件夹路径: ")
    
    # 验证路径
    if not os.path.exists(folder_path):
        print("指定的路径不存在！")
        return
    
    if not os.path.isdir(folder_path):
        print("指定的路径不是文件夹！")
        return
    
    # 确认删除
    confirm = input(f"确定要删除文件夹 '{folder_path}' 及其所有内容吗？(y/n): ")
    if confirm.lower() != 'y':
        print("操作已取消")
        return
    
    # 执行删除
    try:
        shutil.rmtree(folder_path)
        print("文件夹删除成功！")
    except Exception as e:
        print(f"删除失败: {e}")
def delfloder(path):
    # 要删除的文件夹路径
    folder_path = path
    
    # 验证路径
    if not os.path.exists(folder_path):
        print("指定的路径不存在！")
        return
    
    if not os.path.isdir(folder_path):
        print("指定的路径不是文件夹！")
        return
    # 执行删除
    try:
        shutil.rmtree(folder_path)
        print(f"{folder_path}文件夹删除成功！")
    except Exception as e:
        print(f"删除失败: {e}")

if __name__ == "__main__":
    p0='/public/home/ac877eihwp/renyq/model/RDA_S'
    with open('/public/home/ac877eihwp/renyq/model/RDA_S/foldername.json','r') as f:
        dictTS = json.load(f)
    for name in dictTS:
        completePath = f'{p0}/{name}/IntermediateProcess/'
        delfloder(f'{completePath}step1')
        delfloder(f'{completePath}step2')
        delfloder(f'{completePath}step3')
        delfloder(f'{completePath}results')
        delfloder(f'{p0}/{name}/freq_calculation')
        os.mkdir(f'{completePath}step1')
        os.mkdir(f'{completePath}step2')
        os.mkdir(f'{completePath}step3')
        os.mkdir(f'{completePath}results')
