from MEACRNG.CRNG.ReactionLibrary.MetaReactions import ReactionType, Reaction
from rdkit.Chem import AllChem as Chem
from MEACRNG.MolEncoder.C1MolLib import C1MolLib
from MEACRNG.CRNG.ReactionLibrary.MetaReactions import OneReactantOneProductMetaReation


# Cn的Lib是基于C1建立的，而并非从0开始
n = 2    
class CnOneReactionOneProductReactionLib:
    reaction_add_carbon_species_on_jth_Os = [
        Reaction(
            OneReactantOneProductMetaReation.reaction_add_mol_on_nth_O_with_carbon_and_make_CO_single_bond(i, j),
            f"Add {Chem.MolToSmiles(i)} on {j}th O" ,
            ReactionType.OneReactantOneProduct)
        for i in [
            Chem.RWMol(Chem.MolFromSmarts("C")),
            C1MolLib.CH,
            C1MolLib.CH2,
            C1MolLib.CH3,
            C1MolLib.CO,
            C1MolLib.CHO,
            C1MolLib.COH,
            C1MolLib.CH2O,
            C1MolLib.CHOH
        ] for j in list(range(1,2*n))
    ]
    reaction_add_carbon_species_on_jth_carbon = [
        Reaction(
            OneReactantOneProductMetaReation.reaction_add_mol_on_nth_carbon_and_make_CC_single_bond(i, j),
            f"Add {Chem.MolToSmiles(i)} on {j}th C" ,
            ReactionType.OneReactantOneProduct)
        for i in [
            Chem.RWMol(Chem.MolFromSmarts("C")),
            C1MolLib.CH,
            C1MolLib.CH2,
            C1MolLib.CH3,
            C1MolLib.CO,
            C1MolLib.CHO,
            C1MolLib.COH,
            C1MolLib.CH2O,
            C1MolLib.CHOH
        ] for j in list(range(1,n))
    ]
    nlist =list(range(1,n+1))#[1,2,3]
    reaction_add_H_on_carbons = [Reaction(
        OneReactantOneProductMetaReation.reaction_add_atom_A_on_nth_atom_B_with_m_Hs_to_B_if_B_have_bonds_fewer_equal_than_l(
            1, i, "C", 0, 3),
        "Add H on %sth C" % i,
        ReactionType.OneReactantOneProduct) for i in nlist]

    reaction_add_H_on_the_Os = [Reaction(
        OneReactantOneProductMetaReation.reaction_add_atom_A_on_nth_atom_B_with_m_Hs_to_B_if_B_have_bonds_fewer_equal_than_l(
            1, i, "O", 0, 1),
        "Add H on %sth O" % i,
        ReactionType.OneReactantOneProduct) for i in nlist]

    reaction_add_OH_on_carbons = [Reaction(
        OneReactantOneProductMetaReation.reaction_add_atom_A_on_nth_atom_B_with_m_Hs_to_B_if_B_have_bonds_fewer_equal_than_l(
            8, i, "C", 1, 3),
        "Add OH on %sth C" % i,
        ReactionType.OneReactantOneProduct) for i in nlist]

    reaction_add_O_single_bond_on_carbons = [Reaction(
        OneReactantOneProductMetaReation.reaction_add_atom_A_on_nth_atom_B_with_m_Hs_to_B_if_B_have_bonds_fewer_equal_than_l(
            8, i, "C", 0, 3, Chem.BondType.SINGLE),
        "Add O on %sth C" % i,
        ReactionType.OneReactantOneProduct) for i in nlist]

    reaction_remove_H_on_Os = [Reaction(
        OneReactantOneProductMetaReation.reaction_remove_kth_atom_A_on_nth_atom_B(
            1, "H", i, "O"),
        "Remove H on %sth O" % i,
        ReactionType.OneReactantOneProduct) for i in range(1, 2*n+1)]

    reaction_remove_H_on_carbons = [Reaction(
        OneReactantOneProductMetaReation.reaction_remove_kth_atom_A_on_nth_atom_B(
            1, "H", i, "C"),
        "Remove H on %sth C" % i,
        ReactionType.OneReactantOneProduct) for i in nlist]

    reaction_remove_first_O_or_OH_on_carbons = [Reaction(
        OneReactantOneProductMetaReation.reaction_remove_kth_atom_A_on_nth_atom_B(
            1, "O", i, "C", True),
        "Remove 1st O/OH on %sth C" % i,
        ReactionType.OneReactantOneProduct) for i in nlist]#wrong

    reaction_remove_second_O_or_OH_on_carbon = [Reaction(
        OneReactantOneProductMetaReation.reaction_remove_kth_atom_A_on_nth_atom_B(
            2, "O", i, "C", True),
        "Remove 2nd O/OH on %sth C" % i,
        ReactionType.OneReactantOneProduct) for i in nlist]#wrong
 
