from ase.optimize import BFGS, FIRE
from ase.io import read, write
from nequip.ase import NequIPCalculator
from ase.neb import NEB
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms
from sella import Sella
from nequip.ase import NequIPCalculator
import copy
import json
model_path = '/work/home/ac877eihwp/renyq/LUNIX_all/mlp_opt/prototypeModel.pth'
calc = NequIPCalculator.from_deployed_model(model_path, device='cpu')


def var_name(var, all_var=locals()):
    return [var_name for var_name in all_var if all_var[var_name] is var][0]


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
        i.calc = NequIPCalculator.from_deployed_model(model_path, device='cpu')
    neb = NEB(a, climb=True)
    neb.interpolate(method='idpp', mic=True)
    write(f'{p}/neb_interpolated_{nImg}img.xyz', a)  # <--- 新增保存插值路径
    if FIRE(neb).run(fmax=0.5, steps=steps):#
        return a
    else:
        return False

def read_data(file_name):
    Atoms = read(file_name)
    Atoms.calc = calc
    FIRE(Atoms).run(fmax=0.1)#
    BFGS(Atoms,maxstep=0.05).run(fmax=0.01)#
    return Atoms
'''--------------------------------'''
with open('config.json','r') as j:
    data = json.load(j)
path = data['path']
folderpath=data['folderpath']
for name in folderpath:
    IS, FS = read_data(f'{path}/{name}/IS.vasp'), read_data(f'{path}/{name}/FS.vasp')
    p0 = f'{path}/{name}'
    print('Start NEB')
    TS = run_neb(IS, FS, 8, 'TS', p0)
    d = [IS, TS, FS]
    print('-'*50)
    for i in d:
        i.calc = NequIPCalculator.from_deployed_model(model_path, device='cpu')
        print('%8s : %.3f' % (var_name(i), i.get_potential_energy()))
    write(f'{path}/{name}/R7.xyz', d[:])#R7.xyz include IS TS(0.3) FS
    print('Finish NEB search TS')
    print('Start Sella')
    initial_structure = copy.deepcopy(TS)
    fixed_indices = list(range(0, 32))
    initial_structure.constraints = [FixAtoms(indices=fixed_indices)]
    initial_structure.set_calculator(calc)
    opt = Sella(initial_structure, trajectory='ts_search.traj')
    opt.run(fmax=0.05)  # fmax 是力的收敛标准
    write(f'{path}/{name}/optimized_ts.xyz', initial_structure)