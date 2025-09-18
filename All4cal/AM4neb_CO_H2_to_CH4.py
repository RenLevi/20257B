import NEB.preneb as pb
import SUPPORT.support as sp
import json
import os
#NEB搜索过渡态#并列执行
Pre4NEB = pb.PREforNEB('/work/home/ac877eihwp/renyq/20250828TT/test/')
Pre4NEB.readDataPath()
Pre4NEB.buildNEB('/work/home/ac877eihwp/renyq/20250828TT/test/RN/reactionslist.txt')
Pre4NEB.start_split(10)
foldersplitlist = Pre4NEB.fsp
for i in range(len(foldersplitlist)):
    fl = foldersplitlist[i]
    data = {
        'path':'/work/home/ac877eihwp/renyq/20250828TT/test/neb',
        #'record':'/work/home/ac877eihwp/renyq/20250828TT/test/opt/system/record_adscheck.json',
        'folderpath':fl
        }
    os.makedirs(name=f'test/jobsub/neb/{i}',exist_ok=True)
    with open(f'test/jobsub/neb/{i}/config.json','w') as j:
        json.dump(data,j)
    sp.copyFiles('NEB/neb_searchTS.py',f'test/jobsub/neb/{i}')
    sp.copyFiles('NEB/jobsearchts.sh',f'test/jobsub/neb/{i}')
    sp.run_command_in_directory(directory=f'test/jobsub/neb/{i}',command='sbatch jobSubmit.sh')

    



