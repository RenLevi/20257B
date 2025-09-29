import preSearchTS.pre4SearchTS as pre4TS
import SUPPORT.support as sp
import json
import os
#NEB搜索过渡态#并列执行
Pre4TS = pre4TS.PREforSearchTS('/work/home/ac877eihwp/renyq/20250828TT/test/')
Pre4TS.readDataPath()
Pre4TS.buildmodel('/work/home/ac877eihwp/renyq/20250828TT/test/RN/reactionslist.txt')
Pre4TS.start_split(10)
foldersplitlist = Pre4TS.fsp
for i in range(len(foldersplitlist)):
    fl = foldersplitlist[i]
    data = {
        'path':'/work/home/ac877eihwp/renyq/20250828TT/test/RDA_S',
        #'record':'/work/home/ac877eihwp/renyq/20250828TT/test/opt/system/record_adscheck.json',
        'folderpath':fl
        }
    os.makedirs(name=f'test/jobsub/RDA_S/{i}',exist_ok=True)
    with open(f'test/jobsub/RDA_S/{i}/config.json','w') as j:
        json.dump(data,j)
    sp.copyFiles('preSearchTS/opt4SearchTS.py',f'test/jobsub/RDA_S_TS/{i}')
    sp.copyFiles('preSearchTS/jobpre4TS.sh',f'test/jobsub/RDA_S_TS/{i}')
    sp.run_command_in_directory(directory=f'test/jobsub/RDA_S_TS/{i}',command='sbatch jobpre4TS.sh')

    



