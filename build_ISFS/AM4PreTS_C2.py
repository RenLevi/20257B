import build_ISFS.pre4SearchTS as pre4TS
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
Pre4TS.buildmodel('/public/home/ac877eihwp/renyq/C2/test/RN/reactionslist.txt')
Pre4TS.start_split(10)'''
def read_file_line_by_line(file_path):#逐行读取txt文件并返回list数据
    reaction_list=[]
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            string = line.strip()  
            reaction_list.append(string)
        reaction_list.pop(0)
    return reaction_list
def start_split(fdl,batch):
        floderlist = fdl
        n=len(floderlist)
        r=n%batch
        klow=int(n/batch)
        split = [klow+1 for _ in range(r)] + [klow for _ in range(10-r)]
        foldersplit=[]
        for i in split:
            il=floderlist[0:i]
            del floderlist[:i]
            foldersplit.append(il)
        return foldersplit

fdl = read_file_line_by_line('/public/home/ac877eihwp/renyq/C2/test/RN/reactionslist.txt')
foldersplitlist = start_split(fdl,20)
for i in range(len(foldersplitlist)):
    fl = foldersplitlist[i]
    data = {
        'path':'/public/home/ac877eihwp/renyq/C2/test/RDA_S',
        #'record':'/work/home/ac877eihwp/renyq/20250828TT/test/opt/system/record_adscheck.json',
        'folderpath':fl
        }
    os.makedirs(name=f'test/jobsub/RDA_Spre/{i}',exist_ok=True)
    with open(f'test/jobsub/RDA_Spre/{i}/config.json','w') as j:
        json.dump(data,j)
    sp.copyFiles('build_ISFS/preTS.py',f'test/jobsub/RDA_Spre/{i}')
    sp.copyFiles('build_ISFS/jobpreTS.sh',f'test/jobsub/RDA_Spre/{i}')
    sp.copyFolder('build_ISFS',f'test/jobsub/RDA_Spre/{i}/build_ISFS')
    sp.run_command_in_directory(directory=f'test/jobsub/RDA_Spre/{i}',command='qsub jobpreTS.sh')




