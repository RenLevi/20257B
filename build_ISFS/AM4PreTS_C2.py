#import build_ISFS.pre4SearchTS as pre4TS
import SUPPORT.support as sp
import json
import os
'''#NEB搜索过渡态#并列执行
print('-'*10,"start",'-'*20)
Pre4TS = pre4TS.PREforSearchTS('/public/home/ac877eihwp/renyq/C2/test/')
print('-'*10,"start site finder",'-'*8)
Pre4TS.site_finder()
print('-'*10,"start readData",'-'*11)
Pre4TS.readDataPath()
print('-'*10,"start build model",'-'*8)
Pre4TS.buildmodel('/public/home/ac877eihwp/renyq/C2/test/RN/{reactions}list.txt')
Pre4TS.start_split(10)'''
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
reactions = 'reactions'
for batch in batch_process(fdl,11):
    data = {'INAME':i,
        'folderpath':batch,
        'MLPs_model_path':'/public/home/ac877eihwp/renyq/prototypeModel.pth'
        }
    os.makedirs(name=f'test/jobsub/{reactions}/{i}',exist_ok=True)
    with open(f'test/jobsub/{reactions}/{i}/config.json','w') as j:
        json.dump(data,j)
    sp.copyFiles('build_ISFS/preTS.py',f'test/jobsub/{reactions}/{i}')
    sp.copyFiles('build_ISFS/jobpreTS.sh',f'test/jobsub/{reactions}/{i}')
    sp.copyFolder('build_ISFS',f'test/jobsub/{reactions}/{i}/build_ISFS')
    sp.run_command_in_directory(directory=f'test/jobsub/{reactions}/{i}',command='qsub jobpreTS.sh')
    i += 1




