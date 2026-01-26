from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator
from nequip.ase import NequIPCalculator
from ase.vibrations import Vibrations
from ase.constraints import FixAtoms
import json
import os
def get_single_filename(folder_path):
    """获取文件夹中唯一的文件名"""
    all_items = os.listdir(folder_path)
    files = [item for item in all_items if os.path.isfile(os.path.join(folder_path, item))]
    if len(files) == 1:
        return files[0]
    elif len(files) == 0:
        return "文件夹为空"
    else:
        return f"文件夹中有 {len(files)} 个文件，不止一个文件"
model_path = '/work/home/ac877eihwp/renyq/LUNIX_all/mlp_opt/prototypeModel.pth'
calculator = NequIPCalculator.from_deployed_model(model_path)
with open('config.json','r') as j:
    data = json.load(j)
path = data['path']
folderpath=data['folderpath']
INAME = data['INAME']
for name in folderpath:
    PathName = f'{path}/{name}'
    with open (f'{path}/{INAME}.json','r') as j:
        dictrn = json.load(j)
    rnd = dictrn[name]
    if rnd['TS searched'] == True:
        atoms = read(f'{PathName}/optimized_ts.xyz')
        atoms.calc = calculator
        indices_to_fix = [atom.index for atom in atoms if atom.symbol == 'Ru']
        constraint = FixAtoms(indices=indices_to_fix)
        atoms.set_constraint(constraint)
        vib = Vibrations(atoms, name = f'{PathName}/freq_calculation', delta = 0.01, nfree = 2)
        vib.run()
        vib.summary(log=f'{PathName}/frequency_summary.txt')
        frequencies = vib.get_frequencies()
        print("\n振动频率 (cm⁻¹):")
        with open(f'{path}/{INAME}.json','r') as j:
            dictij = json.load(j)
        dictij[name]['freq&free Energy'] = True
        with open(f'{path}/{INAME}.json','w') as j:
            json.dump(dictij,j)
    else:
        with open(f'{path}/{INAME}.json','r') as j:
            dictij = json.load(j)
        dictij[name]['freq&free Energy'] = False
        with open(f'{path}/{INAME}.json','w') as j:
            json.dump(dictij,j)

