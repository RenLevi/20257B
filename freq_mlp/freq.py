from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator
from nequip.ase import NequIPCalculator
from ase.vibrations import Vibrations
from ase.constraints import FixAtoms
model_path = '/work/home/ac877eihwp/renyq/LUNIX_all/mlp_opt/prototypeModel.pth'
calculator = NequIPCalculator.from_deployed_model(model_path)

atoms = read('TS_RDA_S_RdncCopt.xyz')
atoms.calc = calculator
indices_to_fix = [atom.index for atom in atoms if atom.symbol == 'Ru']
constraint = FixAtoms(indices=indices_to_fix)
atoms.set_constraint(constraint)
vib = Vibrations(atoms, name = 'freq_calculation', delta = 0.01, nfree = 2)
vib.run()
vib.summary(log='frequency_summary.txt')
frequencies = vib.get_frequencies()
print("\n振动频率 (cm⁻¹):")

