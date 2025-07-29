from ase import Atoms
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from nequip.ase import NequIPCalculator
import ase.io
from numpy import array
from ase.thermochemistry import HinderedThermo
nequipModel=NequIPCalculator.from_deployed_model(model_path='/work/home/ac877eihwp/renyq/sella/LUNIX_all/mlp_opt/prototypeModel.pth',device='cpu')
struct=ase.io.read('POSCAR')
struct.set_calculator(nequipModel)
BFGS(struct).run(fmax=0.01)
vib = Vibrations(struct)
vib.run()
vib.summary()
