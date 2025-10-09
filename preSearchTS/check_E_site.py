import json
from ase.io import read
class molfile():
    def __init__(self,checkdict,name,path_test):
        self.name =name
        self.report = checkdict[name]#[cp,wf,wb,wna,waH]
        '''
        name:[cp,wf,wb,wna,waH]
        "[H]": [[1],[],[],[],[]]
        "[H]C([H])([H])O": [[1],[],[],[],[]]
        '''
        [cp,wf,wb,wna,waH] = self.report
        species = f'{path_test}opt/system/species/'
        if cp != []:
            E=[]
            for i in cp:
                last_atoms = read(f'{species}{name}/{i}/nequipOpt.traj',index=-1)
                final_energy = last_atoms.get_potential_energy()
                E.append(final_energy)
            min_E = min(E)
            id = E.index(min_E)
            self.model_p = f'{species}{name}/{cp[id]}/nequipOpt.traj'
        else:
            if len(wna+waH) > len(wb):#？存疑
                E = []
                for  i in wna:
                    last_atoms = read(f'{species}{name}/{i}/nequipOpt.traj',index=-1)
                    final_energy = last_atoms.get_potential_energy()
                    E.append(final_energy)
                min_E = min(E)
                id = E.index(min_E)
                self.model_p = f'{species}{name}/{wna[id]}/nequipOpt.traj'
            else:
                self.model_p = None