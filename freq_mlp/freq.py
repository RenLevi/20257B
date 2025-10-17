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
for name in folderpath:
    p0 = f'{path}/{name}'
    '''fn = get_single_filename(f'{p0}//IntermediateProcess/results')
    atoms = read(f'{p0}//IntermediateProcess/results/{fn}')'''
    atoms = read(f'{p0}/optimized_ts.xyz')
    print(name,len(atoms))
    atoms.calc = calculator
    indices_to_fix = [atom.index for atom in atoms if atom.symbol == 'Ru']
    constraint = FixAtoms(indices=indices_to_fix)
    atoms.set_constraint(constraint)
    vib = Vibrations(atoms, name = f'{p0}/freq_calculation', delta = 0.01, nfree = 2)
    vib.run()
    vib.summary(log=f'{p0}/frequency_summary.txt')
    frequencies = vib.get_frequencies()
    #print("\n振动频率 (cm⁻¹):")

