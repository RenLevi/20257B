import json
'''p0 = '/work/home/ac877eihwp/renyq/20250828TT/test'
with open('/work/home/ac877eihwp/renyq/20250828TT/test/RDA_S/foldername.json','r') as j:
    dict_folder = json.load(j)
for name in dict_folder:
    p = f'{p0}/{name}'''
def json_r_w(name,data):
    with open(name, 'r') as f:
        file = f.read()
        if len(file)>0:
            ne = 'ne'
        else:
            ne = 'e'
    if ne == 'ne':
        with open (name,'r') as f:
            old_data = json.load(f)
    else:
        old_data ={}
    old_data.update(data)
    with open(name, 'w') as f:
        json.dump(old_data,f,indent=2)
with open('Allfeedback.json','w') as j:
    pass
for i in range(10):
    with open(f'{i}/feedback.json','r') as j:
        d = json.load(j)
    json_r_w(f'Allfeedback.json',d)