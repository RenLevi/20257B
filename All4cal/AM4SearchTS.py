import json
import os
import SUPPORT.support as sp
from preSearchTS.pre4SearchTS import SearchTS4All

STS4A = SearchTS4All('/work/home/ac877eihwp/renyq/20250828TT/test/')
foldersplitlist = STS4A.start_split(10)
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
    sp.copyFiles('RDA_S/SearchTS.py',f'test/jobsub/RDA_S/{i}')
    sp.copyFiles('RDA_S/jobsearchts.sh',f'test/jobsub/RDA_S/{i}')
    sp.run_command_in_directory(directory=f'test/jobsub/RDA_S/{i}',command='sbatch jobsearchts.sh')