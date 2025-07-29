from rdkit.Chem import AllChem as Chem
from MEACRNG.CRNG.ReactionNetworkGenerator import SpeciesNode, NetworkGenerator
from MEACRNG.CRNG.ReactionLibrary.C1ReactionLib import C1OneReactantOneProductReactionLib
from MEACRNG.CRNG.ReactionLibrary.C2ReactionLib import C2OneReactionOneProductReactionLib
from MEACRNG.CRNG.ReactionLibrary.CnReactionLib import CnOneReactionOneProductReactionLib
from MEACRNG.CRNG.ValidMolCheckFuncLibrary.ValidMolCheckFuncLib import CarbonValidMolCheckFunc
from MEACRNG.Tools.utils import ReactionLibUtils
from MEACRNG.CRNG.ReactionLibrary.MetaReactions import OneReactantOneProductMetaReation, ReactionType, Reaction
from MEACRNG.Tools.Smiles2formula import expression_transform
import os
# allowed steps,for multi carbon products，use functions in MetaReation or write a new lib

reactions = ReactionLibUtils.merge_reactions([

    CnOneReactionOneProductReactionLib.reaction_add_carbon_species_on_jth_carbon,
    CnOneReactionOneProductReactionLib.reaction_add_H_on_carbons,
    CnOneReactionOneProductReactionLib.reaction_add_OH_on_carbons,
    CnOneReactionOneProductReactionLib.reaction_add_O_single_bond_on_carbons,
    CnOneReactionOneProductReactionLib.reaction_add_H_on_the_Os,
    CnOneReactionOneProductReactionLib.reaction_remove_H_on_carbons,
    CnOneReactionOneProductReactionLib.reaction_remove_H_on_Os,
    CnOneReactionOneProductReactionLib.reaction_remove_first_O_or_OH_on_carbons,
    CnOneReactionOneProductReactionLib.reaction_remove_second_O_or_OH_on_carbon
])

# boundary condition check
check_func = [
    CarbonValidMolCheckFunc.no_more_than_four_bond_in_carbon_check,
    CarbonValidMolCheckFunc.no_more_than_M_bond_in_atom_N_check(n='O',m=2),
    CarbonValidMolCheckFunc.valid_C_O_bond_structure_check_when_only_use_single_CO_bond_simple,
    CarbonValidMolCheckFunc.no_more_than_three_carbon_check,
    CarbonValidMolCheckFunc.num_of_C_and_O_should_smaller_equal_than_4,
    CarbonValidMolCheckFunc.num_of_CHO_should_bigger_equal_than_1
    
]

#reactants and results in Smiles，
CH3CH2CH3 = SpeciesNode(Chem.AddHs(Chem.MolFromSmiles('CCC')))# 'AddHs' will fill H，eg. C+4H==CH4
CO = SpeciesNode(Chem.MolFromSmiles('CO'))
#CH3 = SpeciesNode(Chem.MolFromSmiles('[H]C([H])[H]',sanitize=False))# sanitize=False，input smile which keeps H directly.

# set reactant and product
ng = NetworkGenerator(
    initial_reactants=[CO],
    final_results=[CH3CH2CH3],
    valid_mol_check_func_list=check_func)


ng.species_do_one_reactant_one_product_reaction_until_no_new_product(reactions)
ng.remove_repeated_reaction_regardless_direction(True)
ng.sort_reaction_by_reaction_str()
ng.group_reaction_by_reaction_name()
print(ng.get_total_readable_reaction_string())
all_mol, all_mol_name,name_to_mol_dict = ng.get_all_involved_mols_and_mol_name_to_mol_dict()
print('total generate %s species'%len(all_mol_name))
print('outputing smiles coding species')
print("\n".join(all_mol_name))
'''
For expression_transform function, if species contain elements beyond C H O, 
this function will directly return the original smiles formula.
This functiton is still in test, please check whether the output is matched with its original Smiles coding.
'''
print('\nChanging Smiles codes into string')
for read in all_mol_name:
    print(read + '   ==   ' + expression_transform(read))# Please change the print way to meet your demand.
    # print(expression_transform(read))
print('')

ng.check_product_in_dict(True)
#!!!

SavePath = 'test/RN/'

#!!!
if not os.path.exists(SavePath):
        os.makedirs(SavePath)
with open(f'{SavePath}reactionslist.txt', 'w', encoding='gbk') as f:
    f.write("We total got %s reaction \n" % len(ng.steps))
    f.write(ng.get_total_readable_reaction_string(color=0))
    f.close()

with open(f'{SavePath}mollist.txt', 'w', encoding='UTF-8') as f:
    f.write("We total got %s mol \n" % len(all_mol_name))
    for i in all_mol_name:
        f.write(i + '\n')
    f.close()
# write reactionslist and molist

ng.plot_network_to_pdf_file(filepath=f"{SavePath}network_SRM_smiles1.pdf")
ng.plot_network_to_pdf_file_labeltransed(filepath=f"{SavePath}network_SRM_string1.pdf")

