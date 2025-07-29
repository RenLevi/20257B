import build_metal_model.metal_surface as ms
import build_system.smi2mol.smi2mol as s2m
import build_system.mol2ads.mol2ads as m2a
import MLP_opt.MLPopt as mlpopt
#催化剂模型
surface = ms.metal(element='Ru',size=(4,4,4),vacuum=10,path=f'test/output/')
surface.build_model()
surface.save_model()
print('Complete matel model')
#气相分子模型
smiles2ase = s2m.smi2mol(input='test/RN/mollist.txt',output='test/output/molecule')
smiles2ase.mollist2SMILES()
smiles2ase.creat_folder_and_file()
smiles2ase.build_model()
print('Complete molecule(g) model')
#预吸附随机放置模型（要求随机散布较均匀！未实现）
molecule2system = m2a.mol2ads(input='test/output/molecule/species_name.txt',output='test/output/system',metal='test/output/slab/Ru_hcp0001.vasp')
molecule2system.txt_to_dict()
molecule2system.creat_folder_and_file()
molecule2system.build_model(smi2molPATH='test/output/molecule/species',random_mol_num=10,size=(4,4,4))
#MLP优化结构#顺序执行
OPT4ALLSYS = mlpopt.opt4ALLsystems(input='test/output/system',random_number=10)
OPT4ALLSYS.txt_to_dict()
OPT4ALLSYS.start_cal(record='test/output/system/record.txt')





