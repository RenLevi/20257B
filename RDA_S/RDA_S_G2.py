import json
from SearchTS_G2 import RDA_S
import os
with open('config.json','r') as j:
    data = json.load(j)
INAME = data['INAME']
path = data['path']
folderpath=data['folderpath']
model_path = data['MLPs_model_path']
with open(f'{path}/{INAME}.json','r') as j:
    data = json.load(j)
for name in data:
    pn = f'{path}/{name}'
    pn_save = f'/public/home/ac877eihwp/renyq/C2/test/RDASg2/{name}'
    os.makedirs(pn_save, exist_ok=True)
    reaction_info = data[name]
    if "final ads" in reaction_info:
        if reaction_info["final ads"] == [True,True,True]:
            RDA_S4STS = RDA_S(ISfile=f'{pn}/IS.vasp', FSfile=f'{pn}/FS.vasp', path=pn_save)
            RDA_S4STS.initialize_floder()
            RDA_S4STS.opt(model_path)
            RDA_S4STS.run_idpp(model_path)
