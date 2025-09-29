import json
class SearchTS4All():
    def __init__(self,path_test):
        self.neb = path_test+'RDA_S/'
        with open(f'{self.neb}foldername.json', 'r') as f:
            self.d = json.load(f)
    def start_split(self,batch):
        fd = self.d
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
        self.fsp = foldersplit
        return foldersplit
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
    sp.copyFiles('preSearchTS/opt4SearchTS.py',f'test/jobsub/RDA_S/{i}')
    sp.copyFiles('preSearchTS/jobpre4TS.sh',f'test/jobsub/RDA_S/{i}')
    sp.run_command_in_directory(directory=f'test/jobsub/RDA_S/{i}',command='sbatch jobpre4TS.sh')