import json
from ase import Atoms
p0 = '/work/home/ac877eihwp/renyq/20250828TT/test'
with open('/work/home/ac877eihwp/renyq/20250828TT/test/RDA_S/foldername.json','r') as j:
    dict_folder = json.load(j)
for name in dict_folder:
    p = f'{p0}/{name}'
