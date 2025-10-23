import json
import os
import SUPPORT.support as sp
from preSearchTS.pre4SearchTS import SearchTS4All

STS4A = SearchTS4All('/public/home/ac877eihwp/renyq/model/')
foldersplitlist = STS4A.start_split(10)
for i in range(len(foldersplitlist)):
    fl = foldersplitlist[i]
    data = {
        'path':'/public/home/ac877eihwp/renyq/model/RDA_S',
        #'record':'/work/home/ac877eihwp/renyq/20250828TT/test/opt/system/record_adscheck.json',
        'folderpath':fl
        }
    os.makedirs(name=f'model/jobsub/RDA_S/{i}',exist_ok=True)
    with open(f'model/jobsub/RDA_S/{i}/config.json','w') as j:
        json.dump(data,j)
    sp.copyFiles('RDA_S/SearchTS_test.py',f'model/jobsub/RDA_S/{i}')
    sp.copyFiles('RDA_S/jobsub.sh',f'model/jobsub/RDA_S/{i}')
    sp.run_command_in_directory(directory=f'model/jobsub/RDA_S/{i}',command='qsub jobsub.sh')