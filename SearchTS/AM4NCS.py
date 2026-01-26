import json
import os
import SUPPORT.support as sp
def read_file_line_by_line(file_path):#逐行读取txt文件并返回list数据
    reaction_list=[]
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            string = line.strip()  
            reaction_list.append(string)
        reaction_list.pop(0)
    return reaction_list
def batch_process(data, batch_size):
    for i in range(0, len(data), batch_size):


        yield data[i:i + batch_size]
i = 0
fdl = read_file_line_by_line('/public/home/ac877eihwp/renyq/C2/test/RN/reactionslist.txt')
folder = 'NCS'
for batch in batch_process(fdl,11):
    data = {'INAME':i,
        'folderpath':batch,
        'MLPs_model_path':'/public/home/ac877eihwp/renyq/prototypeModel.pth',
        'path':'/public/home/ac877eihwp/renyq/C2/test/reactions'
        }
    os.makedirs(name=f'test/jobsub/{folder}/{i}',exist_ok=True)
    with open(f'test/jobsub/{folder}/{i}/config.json','w') as j:
        json.dump(data,j)
    sp.copyFiles('SearchTS/NEB_Combine_Sella.py',f'test/jobsub/{folder}/{i}')
    sp.copyFiles('SearchTS/jobNCS.sh',f'test/jobsub/{folder}/{i}')
    sp.run_command_in_directory(directory=f'test/jobsub/{folder}/{i}',command='qsub jobNCS.sh')
    i += 1