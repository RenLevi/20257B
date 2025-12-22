import build_ISFS_br2bo.pre4SearchTS as pre4TS
import SUPPORT.support as sp
import json
import os
#NEB搜索过渡态#并列执行
Pre4TS = pre4TS.PREforSearchTS('/public/home/ac877eihwp/renyq/C2/test/')
Pre4TS.site_finder()
Pre4TS.readDataPath()
Pre4TS.buildmodel('/public/home/ac877eihwp/renyq/C2/test/RN/reactionslist.txt')
Pre4TS.start_split(10)
foldersplitlist = Pre4TS.fsp
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
    sp.copyFiles('preSearchTS/opt4SearchTS.py',f'test/jobsub/RDA_Spre/{i}')
    sp.copyFiles('preSearchTS/jobpre4TS.sh',f'test/jobsub/RDA_Spre/{i}')
    #sp.run_command_in_directory(directory=f'test/jobsub/RDA_Spre/{i}',command='qsub jobpre4TS.sh')

    



