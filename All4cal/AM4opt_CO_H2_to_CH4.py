import build_metal_model.metal_surface as ms
import build_system.smi2mol.smi2mol as s2m
import build_system.mol2ads.mol2ads as m2a
import MLP_opt.MLPopt as mlpopt
import SUPPORT.support as sp
import json
import os
#催化剂模型
surface = ms.metal(element='Ru',size=(4,4,4),vacuum=10,path=f'test/opt/')
surface.build_model()
surface.save_model()
print('Complete matel model')
#气相分子模型
smiles2ase = s2m.smi2mol(input='test/RN/mollist.txt',output='test/opt/molecule')
smiles2ase.mollist2SMILES()
smiles2ase.creat_folder_and_file()
smiles2ase.build_model()
#非含碳分子气相模型
with open ('test/RN/mol_without_C.txt','w') as f:
    f.write('We total got 5 mol\n')
    f.write('[H][H]\n')
    f.write('[H]\n')
    f.write('[H]O[H]\n')
    f.write('O\n')
    f.write('OH\n')
smiles2ase = s2m.smi2mol(input='test/RN/mol_without_C.txt',output='test/opt/molecule')
smiles2ase.mollist2SMILES()
smiles2ase.creat_folder_and_file()
smiles2ase.build_model()
print('Complete molecule(g) model')

#预吸附随机放置模型（要求随机散布较均匀！未实现,对于干净表面，随机要求不高）
molecule2system = m2a.mol2ads(
    input='test/opt/molecule/species_name.json',
    output='test/opt/system',
    metal='test/opt/slab/Ru_hcp0001.vasp')
molecule2system.creat_folder_and_file()
molecule2system.build_model(smi2molPATH='test/opt/molecule/species',random_mol_num=20,size=(4,4,4))

#MLP优化结构#并列执行
OPT4ALLSYS = mlpopt.opt4ALLsystems_batch_by_batch(input='test/opt/system',random_number=20)
OPT4ALLSYS.start_split(10)
foldersplitlist = OPT4ALLSYS.fsp
for i in range(len(foldersplitlist)):
    fl = foldersplitlist[i]
    data = {
        'path':'/work/home/ac877eihwp/renyq/20250828TT/test/opt/system',
        'record':'/work/home/ac877eihwp/renyq/20250828TT/test/opt/system/record_adscheck.json',
        'folderpath':fl,
        'random_number':20
        }
    os.makedirs(name=f'test/jobsub/opt/{i}',exist_ok=True)
    with open(f'test/jobsub/opt/{i}/config.json','w') as j:
        json.dump(data,j)
    sp.copyFiles('MLP_opt/mlp_calEnergy.py',f'test/jobsub/opt/{i}')
    sp.copyFiles('MLP_opt/jobsubopt.sh',f'test/jobsub/opt/{i}')
    sp.run_command_in_directory(directory=f'test/jobsub/opt/{i}',command='sbatch jobsubopt.sh')

    



