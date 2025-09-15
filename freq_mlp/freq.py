from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator
from nequip.ase import NequIPCalculator
from ase.vibrations import Vibrations

model_path = '/work/home/ac877eihwp/renyq/LUNIX_all/mlp_opt/prototypeModel.pth'
calculator = NequIPCalculator.from_deployed_model(model_path)

atoms = read('optimized_ts.xyz')

atoms.set_calculator(calculator)

vib = Vibrations(atoms, name = 'freq_calculation', delta = 0.01, nfree = 2)
vib.run()
vib.summary(log='frequency_summary.txt')
frequencies = vib.get_frequencies()
print("\n振动频率 (cm⁻¹):")