from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator
from nequip.ase import NequIPCalculator
from ase.vibrations import Vibrations
from ase.constraints import FixAtoms
import json
import os
from pathlib import Path
with open('config.json','r') as j:
    data = json.load(j)
path = data['path']
folderpath=data['folderpath']
INAME = data['INAME']
model_path = data['MLPs_model_path']
calc = NequIPCalculator.from_deployed_model(model_path, device='cpu')
for name in folderpath:
    with open (f'{path}/TS.json','r') as j:
        dictTS = json.load(j)
    for k in dictTS:
        if dictTS[k]["Reaction"]==name:
            if dictTS[k]["RDAS_freq"] != None:
                PN = Path(dictTS[k]["RDAS_freq"])
                atoms = read(PN/'TS.xyz')
                atoms.calc = calc
                indices_to_fix = [atom.index for atom in atoms if atom.symbol == 'Ru']
                constraint = FixAtoms(indices=indices_to_fix)
                atoms.set_constraint(constraint)
                vib = Vibrations(atoms, name = f'{PN}/freq_calculation', delta = 0.01, nfree = 2)
                vib.run()
                vib.summary(log=f'{PN}/frequency_summary.txt')
                frequencies = vib.get_frequencies()
            if dictTS[k]["NCS_freq"] != None:
                PN = Path(dictTS[k]["NCS_freq"])
                atoms = read(PN/'optimized_ts.xyz')
                atoms.calc = calc
                indices_to_fix = [atom.index for atom in atoms if atom.symbol == 'Ru']
                constraint = FixAtoms(indices=indices_to_fix)
                atoms.set_constraint(constraint)
                vib = Vibrations(atoms, name = f'{PN}/freq_calculation', delta = 0.01, nfree = 2)
                vib.run()
                vib.summary(log=f'{PN}/frequency_summary.txt')
                frequencies = vib.get_frequencies()

