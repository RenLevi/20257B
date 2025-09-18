import os
import json
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

p0 = '/work/home/ac877eihwp/renyq/20250828TT/test/neb'
fj = p0+'/foldername.json'
C= 0
with open (fj,'r') as j:
    dictFJ=json.load(j)
for name in dictFJ:
    ffl = find_file_listdir(f'{p0}/{name}','optimized_ts.xyz')
    if ffl == True:
        C=C+1
    else:
        C=C
print(f'任务路径:{p0}\n任务总数:{C}/{len(dictFJ)}\n任务进度:{C/len(dictFJ)*100:.3f}%')