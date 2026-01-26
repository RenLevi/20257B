from ase.optimize import BFGS, FIRE
from ase.io import read, write
from nequip.ase import NequIPCalculator
from ase.neb import NEB
from ase.constraints import FixAtoms
from sella import Sella
from nequip.ase import NequIPCalculator
import copy
import json

model_path = '/work/home/ac877eihwp/renyq/prototypeModel.pth'
calc = NequIPCalculator.from_deployed_model(model_path, device='cpu')
def run_neb(IS, FS, nImg, out_file, p):
    steps = 2000
    a = False
    while not a:
        a = cycle_neb(IS, FS, nImg, steps, p)
        nImg += 1
    filename = '%s.xyz' % out_file
    write(f'{p}/{filename}', a[:])
    TS = sorted(a, key=lambda x: x.get_potential_energy())[-1]
    return TS

def cycle_neb(IS, FS, nImg, steps, p):
    a = [IS.copy() for i in range(nImg+1)] + [FS]
    for i in a: 
        i.calc = calc#NequIPCalculator.from_deployed_model(model_path, device='cpu')
    neb = NEB(a, climb=True,k=5)
    neb.interpolate(method='idpp', mic=True)
    write(f'{p}/neb_interpolated_{nImg}img.xyz', a)  # <--- 新增保存插值路径
    if FIRE(neb).run(fmax=0.3, steps=steps):#
        return a
    else:
        return False

with open('config.json','r') as j:
    data = json.load(j)
path = data['path']
folderpath=data['folderpath']
INAME = data['INAME']
for name in folderpath:
    PathName = f'{path}/{name}'
    def CHECKISFS(iname,name,p):
        with open(f'{p}/{iname}.json','r') as j:
            dictrns = json.load(j)
        rnd = dictrns[name]
        keylist = list(rnd.key())
        if "final ads" in keylist:
            if rnd["final ads"] == [True,True,True]:
                return True
            else:
                return False
        else:
            return False
    def warp(IS,FS,p):    
        print(f'Start NEB')
        TS = run_neb(IS, FS, 8, 'TS', p)
        d = [IS, TS, FS]
        dn=['IS','TS','FS']
        print('-'*50)
        for i in d:
            i.calc = NequIPCalculator.from_deployed_model(model_path, device='cpu')
            print('%8s : %.3f' % (dn[d.index(i)], i.get_potential_energy()))
        write(f'{path}/{name}/R7.xyz', d[:])#R7.xyz include IS TS(0.3) FS
        print('Finish NEB search TS')
        print('Start Sella')
        initial_structure = copy.deepcopy(TS)
        fixed_indices = list(range(0, 32))
        initial_structure.constraints = [FixAtoms(indices=fixed_indices)]
        initial_structure.set_calculator(calc)
        opt = Sella(initial_structure, trajectory=f'{path}/{name}/ts_search.traj')
        opt.run(fmax=0.05)  # fmax 是力的收敛标准
        write(f'{path}/{name}/optimized_ts.xyz', initial_structure)
        print('Finish Sella optimization')
    if CHECKISFS(INAME,name,path):
        IS = read(f'{path}/{name}/IS.vasp')
        FS = read(f'{path}/{name}/FS.vasp')
        warp(IS,FS,PathName)
        with open(f'{path}/{INAME}.json','r') as j:
            dictij = json.load(j)
        dictij[name]['TS searched'] = True
        with open(f'{path}/{INAME}.json','w') as j:
            json.dump(dictij,j)
    else:
        with open(f'{path}/{INAME}.json','r') as j:
            dictij = json.load(j)
        dictij[name]['TS searched'] = False
        with open(f'{path}/{INAME}.json','w') as j:
            json.dump(dictij,j)


                        

                        



            
            
