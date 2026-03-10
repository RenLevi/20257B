import json
from ase.io import read
import pandas as pd
import os
import copy
import numpy as np
def find_file_listdir(directory, filename):
    """
    使用 os.listdir() 查找文件
    """
    try:
        files = os.listdir(directory)
        return filename in files
    except FileNotFoundError:
        print(f"目录不存在: {directory}")
        return False
def get_single_filename(folder_path):
    """获取文件夹中唯一的文件名"""
    all_items = os.listdir(folder_path)
    files = [item for item in all_items if os.path.isfile(os.path.join(folder_path, item))]
    
    if len(files) == 1:
        return files[0]
    elif len(files) == 0:
        return "文件夹为空"
    else:
        return f"文件夹中有 {len(files)} 个文件，不止一个文件"

p0='/public/home/ac877eihwp/renyq/model/RDA_S'
with open('/public/home/ac877eihwp/renyq/model/RDA_S/foldername.json','r') as f:
    dictTS = json.load(f)
data = {}
#File name ：[Reaction,atom bond changed,idx,idx,bonded smiles,broken smiles]
for name in dictTS:
    IS = read(f'{p0}/{name}/ISopt.traj',index=-1)
    FS = read(f'{p0}/{name}/FSopt.traj',index=-1)
    rl=dictTS[name]
    id1=rl[1][0]
    id2=rl[1][1]   
    distIS=IS.get_distance(id1,id2)
    distFS=FS.get_distance(id1,id2)
    ISE = IS.get_potential_energy()
    FSE = FS.get_potential_energy() 
    fn = get_single_filename(f'{p0}/{name}/IntermediateProcess/results/')
    if fn != "文件夹为空":
        TS = read(f'{p0}/{name}/IntermediateProcess/results/{fn}')
        distTS=TS.get_distance(id1,id2)
        TSE = TS.get_potential_energy()
        DELTA=TSE-ISE
    else:
        distTS=None
        TSE=None
        DELTA=None
    l = [dictTS[name][0],ISE-ISE,DELTA,FSE-ISE,distIS,distTS,distFS]
    data[name]=l   
df = pd.DataFrame(data)
df = df.T
df.to_excel('RDAS.xlsx', index=False, sheet_name='RDAS.xlsx')