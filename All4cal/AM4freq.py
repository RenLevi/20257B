import json
import os
import SUPPORT.support as sp
def start_split(fd,batch):
    floderlist = list(fd.keys())
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

p0='/work/home/ac877eihwp/renyq/20250828TT/test/neb_05'
with open (f'{p0}/foldername.json','r') as j:
    dictTS=json.load(j)
foldersplitlist = start_split(dictTS,10)
for i in range(len(foldersplitlist)):
    fl = foldersplitlist[i]
    data = {
        'path':'/work/home/ac877eihwp/renyq/20250828TT/test/neb_05',
        #'record':'/work/home/ac877eihwp/renyq/20250828TT/test/opt/system/record_adscheck.json',
        'folderpath':fl
        }
    os.makedirs(name=f'test/jobsub/freq/{i}',exist_ok=True)
    with open(f'test/jobsub/freq/{i}/config.json','w') as j:
        json.dump(data,j)
    sp.copyFiles('freq_mlp/freq.py',f'test/jobsub/freq/{i}')
    sp.copyFiles('freq_mlp/jobsubfreq.sh',f'test/jobsub/freq/{i}')
    sp.run_command_in_directory(directory=f'test/jobsub/freq/{i}',command='sbatch jobsubfreq.sh')