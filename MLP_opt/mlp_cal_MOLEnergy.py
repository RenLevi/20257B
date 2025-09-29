
import ase.io
from nequip.ase import NequIPCalculator
from ase.optimize import BFGS
import ase.io
import os
import sys
import logging

#vasp_energies = []
#nequip_energies = []
#ss=ase.io.read('OUTCAR', index=':')
#for i,s in enumerate(ss):
#  vaspE=s.get_potential_energy()
#  s.set_calculator(nequipModel)
#  nequipE=s.get_potential_energy()
#  log_message = f"{i}: vaspE = {vaspE}, nequipE = {nequipE}"
#  logging.info(log_message)
#  print(log_message)
# vasp_energies.append(vaspE)
#  nequip_energies.append(nequipE)
model_path = '/work/home/ac877eihwp/renyq/prototypeModel.pth'
calc = NequIPCalculator.from_deployed_model(model_path, device='cpu')
struct=ase.io.read('[H]O[H].xyz')
struct.calc = calc
print(f' Starting optmization by NequIP model:')
#optJob=BFGS(struct, trajectory='nequipOpt.traj')
optJob=BFGS(struct, trajectory='O.vasp')
optJob.run(fmax=0.05,steps=1000)