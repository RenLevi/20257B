import json
from SearchTS import RDA_S

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
    reaction_info = data[name]
    if "final ads" in reaction_info:
        if reaction_info["final ads"] == [True,True,True]:
            RDA_S4STS = RDA_S(ISfile=f'{pn}/IS.vasp', FSfile=f'{pn}/FS.vasp', path=pn)
            RDA_S4STS.initialize_floder()
            RDA_S4STS.opt(model_path)
            RDA_S4STS.run(model_path)