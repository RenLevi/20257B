import json
from pathlib import Path
import os
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
# 指定文件路径，替换为你的 json 文件名
json_path = Path(r"/public/home/ac877eihwp/renyq/C2/test/opt/system/folder_name.json")

try:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"文件未找到：{json_path}")
except json.JSONDecodeError as e:
    print(f"解析 JSON 失败：{e}")
else:
    # 美化输出，确保中文不被转义
    #print(json.dumps(data, ensure_ascii=False, indent=2))
    pass
count = 0
for name in data:
    fp = f'/public/home/ac877eihwp/renyq/C2/test/opt/system/species/{name}'
    for i in range(1,21):
        cfp = f'{fp}/{i}'
        FFL = find_file_listdir(cfp,'nequipOpt.traj')
        if FFL ==True:
            count+=1
        else:
            print(cfp)
print(f'{count}/{len(data)*20}')




