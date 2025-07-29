from rdkit.Chem import AllChem as Chem
from MEACRNG.Tools.utils import get_index_and_total_bond_num_of_the_nth_atom_of_element_with_total_m_bonds
from MEACRNG.Tools import utils as u
from MEACRNG.MolEncoder.MolMaker import MolMaker

class ReactionType:# TOUNDER
    OneReactantOneProduct = 1


class Reaction(object):
    def __init__(self, reaction_func, name, reaction_type):
        self.reaction_func = reaction_func
        self.name = name
        self.reaction_type = reaction_type


class OneReactantOneProductMetaReation:





    @staticmethod
    def reaction_bonding_an_new_atom_to_the_atom_with_certain_index_with_certain_bond_type(
            atom_index,
            atomic_number_of_the_atom_to_add,
            bond_type=Chem.BondType.SINGLE,
            num_of_H_add_to_the_new_atom=0):
        def warp(mol):
            if isinstance(mol, bool): return mol
            mol.AddAtom(Chem.Atom(atomic_number_of_the_atom_to_add))
            index_of_add_atom = mol.GetNumAtoms() - 1
            mol.AddBond(atom_index, index_of_add_atom, bond_type)

            for _ in range(num_of_H_add_to_the_new_atom):
                mol.AddAtom(Chem.Atom(1))
                mol.AddBond(index_of_add_atom, mol.GetNumAtoms() - 1, Chem.BondType.SINGLE)
            return mol
        return warp




    @staticmethod
    def reaction_remove_the_kth_bond_with_certain_type_and_connect_to_certain_atom_in_base_atom(
            k=1,
            bond_type=None,
            bonded_atom="H",
            base_atom_idx=0,
            brake_all_bonds_in_bonded_atom=False
    ):
        def warp(mol):
            now_k = 0
            if isinstance(mol, bool): return mol
            atoms = mol.GetAtoms()
            base_atom = atoms[base_atom_idx]
            assert base_atom.GetIdx() == base_atom_idx
            for b in base_atom.GetBonds():
                if (bond_type is None) or (b.GetBondType() == bond_type):
                    now_k += 1
                    if now_k == k:
                        atom1 = b.GetBeginAtom()
                        atom2 = b.GetEndAtom()
                        atom1_s = atom1.GetSymbol()
                        atom2_s = atom2.GetSymbol()
                        if atom1_s == atom2_s:
                            another_atom = atom1
                        else:
                            if atom1_s == base_atom.GetSymbol():
                                another_atom = atom2
                            else:
                                another_atom = atom1
                        if another_atom.GetSymbol() == bonded_atom:
                            mol.RemoveBond(base_atom_idx, another_atom.GetIdx())
                            if brake_all_bonds_in_bonded_atom:
                                for b in another_atom.GetBonds():
                                    mol.RemoveBond(b.GetBeginAtom().GetIdx(), b.GetEndAtom().GetIdx())
                            return mol
            return False

        return warp

    @staticmethod
    def reaction_remove_atoms_without_bonds(remove_ignore="C"):
        def warp(mol):
            # TODO: single C and single H will be removed!
            if isinstance(mol, bool): return mol
            will_remove_idx = []
            # only one atom
            if mol.GetNumAtoms() == 1:
                return False
            for a in mol.GetAtoms():
                if remove_ignore is not None:
                    if a.GetSymbol() == remove_ignore:
                        continue
                if len(a.GetBonds()) == 0:
                    will_remove_idx.append(a.GetIdx())

            # will remove all atoms
            if len(will_remove_idx) == mol.GetNumAtoms():
                return False

            # the length of index will -1 if a atom is removed, so use offset
            will_remove_idx.sort()
            offset = 0
            for i in will_remove_idx:
                mol.RemoveAtom(i - offset)
                offset += 1
            return mol

        return warp

    @staticmethod
    def reaction_add_atom_A_on_nth_atom_B_with_m_Hs_to_B_if_B_have_bonds_fewer_equal_than_l(
            atomic_number_of_atom_A,
            n,
            symbol_of_atom_B,
            m,
            l,
            bond_type=Chem.BondType.SINGLE
    ):
        assert n >= 1

        def warp(mol):
            if isinstance(mol, bool): return mol
            index, total_bond_num = get_index_and_total_bond_num_of_the_nth_atom_of_element_with_total_m_bonds(
                mol,
                n,
                symbol_of_atom_B,
                m=-1)
            if index is None:
                return False
            if total_bond_num > l:
                return False
            r1 = OneReactantOneProductMetaReation.reaction_bonding_an_new_atom_to_the_atom_with_certain_index_with_certain_bond_type(
                index,
                atomic_number_of_the_atom_to_add=atomic_number_of_atom_A,
                bond_type=bond_type,
                num_of_H_add_to_the_new_atom=m
            )
            return r1(mol)

        return warp

    @staticmethod
    def reaction_remove_kth_atom_A_on_nth_atom_B(
            k,
            symbol_of_atom_A,
            n,
            symbol_of_atom_B,
            brake_all_bonds_in_bonded_atom=False,
            allow_bond_type=None,
            atom_to_ignore_remove_when_have_no_bond_connected="C"):
        '''
        example:
        reaction1 = reaction_remove_atom_A_on_nth_atom_B_in_mol("H",1,"C")
        the reaction of remove H on first carbon
        reaction1(mol)
        '''
        assert n >= 1

        def warp(mol):
            if isinstance(mol, bool): return mol
            index, total_bond_num = get_index_and_total_bond_num_of_the_nth_atom_of_element_with_total_m_bonds(
                mol,
                n,
                symbol_of_atom_B,
                m=-1)
            if index is None:
                return False
            r1 = OneReactantOneProductMetaReation.reaction_remove_the_kth_bond_with_certain_type_and_connect_to_certain_atom_in_base_atom(
                k=k,
                base_atom_idx=index,
                bond_type=allow_bond_type,
                bonded_atom=symbol_of_atom_A,
                brake_all_bonds_in_bonded_atom=brake_all_bonds_in_bonded_atom
            )
            r2 = OneReactantOneProductMetaReation.reaction_remove_atoms_without_bonds(
                remove_ignore=atom_to_ignore_remove_when_have_no_bond_connected
            )
            return r2(r1(mol))

        return warp

    @staticmethod
    def reaction_add_mol_on_nth_carbon_with_O_and_make_CC_single_bond(mol, n):
        def warp(target_mol):
            if isinstance(target_mol, bool): return mol
            mol_to_add = Chem.RWMol(mol)
            now_n = 0
            for a in target_mol.GetAtoms():
                if a.GetSymbol() == "C":
                    for b in a.GetBonds():
                        if b.GetBeginAtom().GetSymbol() == "O" or b.GetEndAtom().GetSymbol() == "O":
                            now_n += 1
                            break
                if now_n >= n:
                    C_index_in_target = a.GetIdx()
                    break
            if now_n == 0:
                return False
            C_index_in_mol_to_add, _ = u.get_index_and_total_bond_num_of_the_nth_atom_of_element_with_total_m_bonds(
                mol_to_add, 1, "C")

            new_mol = MolMaker.add_mol_B_to_mol_A_and_make_bonds_in_ith_atom_in_A_with_jth_atom_in_B(
                mol_to_add,
                target_mol,
                C_index_in_mol_to_add,
                C_index_in_target,
                Chem.BondType.SINGLE)
            return new_mol

        return warp

    @staticmethod
    def reaction_add_mol_on_nth_carbon_and_make_CC_single_bond(mol,n):
        def warp(target_mol):
            if isinstance(target_mol, bool): return mol
            mol_to_add = Chem.RWMol(mol)
            now_n = 0
            for a in target_mol.GetAtoms():
                if a.GetSymbol() == "C":
                        now_n += 1
                if now_n >= n:
                    C_index_in_target = a.GetIdx()
                    break
            if  now_n < n:
                C_index_in_target = target_mol.GetAtoms()[0].GetIdx()
            if now_n == 0:
                return False
            C_index_in_mol_to_add, _ = u.get_index_and_total_bond_num_of_the_nth_atom_of_element_with_total_m_bonds(
                mol_to_add, 1, "C")

            new_mol = MolMaker.add_mol_B_to_mol_A_and_make_bonds_in_ith_atom_in_A_with_jth_atom_in_B(
                mol_to_add,
                target_mol,
                C_index_in_mol_to_add,
                C_index_in_target,
                Chem.BondType.SINGLE)
            return new_mol

        return warp

    @staticmethod
    def reaction_add_mol_on_last_carbon_and_make_CC_single_bond(mol):
        def warp(target_mol):
            if isinstance(target_mol, bool): return mol
            mol_to_add = Chem.RWMol(mol)
            now_n = 0
            for a in target_mol.GetAtoms():
                if a.GetSymbol() == "C":
                    for b in a.GetBonds():
                        if b.GetBeginAtom().GetSymbol() == "C" and b.GetEndAtom().GetSymbol() == "C":
                            now_n += 1
                    if now_n == 1 or now_n == 0:
                        C_index_in_target = a.GetIdx()
                        break
                    now_n = 0

            C_index_in_mol_to_add, _ = u.get_index_and_total_bond_num_of_the_nth_atom_of_element_with_total_m_bonds(
                mol_to_add, 1, "C")

            new_mol = MolMaker.add_mol_B_to_mol_A_and_make_bonds_in_ith_atom_in_A_with_jth_atom_in_B(
                mol_to_add,
                target_mol,
                C_index_in_mol_to_add,
                C_index_in_target,
                Chem.BondType.SINGLE)
            return new_mol

        return warp

    @staticmethod
    def reaction_add_mol_on_first_carbon_and_make_CC_single_bond(mol):
        def warp(target_mol):

            if isinstance(target_mol, bool): return mol
            mol_to_add = Chem.RWMol(mol)
            now_n = 0
            for a in target_mol.GetAtoms():
                if a.GetSymbol() == "C":
                    for b in a.GetBonds():
                        if b.GetBeginAtom().GetSymbol() == "C" and b.GetEndAtom().GetSymbol() == "C":
                            now_n += 1
                    if now_n == 1 or now_n == 0:
                        C_index_in_target = a.GetIdx()
                    now_n = 0

            C_index_in_mol_to_add, _ = u.get_index_and_total_bond_num_of_the_nth_atom_of_element_with_total_m_bonds(
                mol_to_add, 1, "C")


            new_mol = MolMaker.add_mol_B_to_mol_A_and_make_bonds_in_ith_atom_in_A_with_jth_atom_in_B(
                mol_to_add,
                target_mol,
                C_index_in_mol_to_add,
                C_index_in_target,
                Chem.BondType.SINGLE)
            return new_mol

        return warp

    def reaction_remove_mol_on_carbon(mol):
        def warp(target_mol):

            if isinstance(target_mol, bool): return mol
            mol_to_add = Chem.RWMol(mol)
            now_n = 0
            for a in target_mol.GetAtoms():
                if a.GetSymbol() == "C":
                    for b in a.GetBonds():
                        if b.GetBeginAtom().GetSymbol() == "C" and b.GetEndAtom().GetSymbol() == "C":
                            now_n += 1
                    if now_n == 1 or now_n == 0:
                        C_index_in_target = a.GetIdx()
                    now_n = 0

            C_index_in_mol_to_add, _ = u.get_index_and_total_bond_num_of_the_nth_atom_of_element_with_total_m_bonds(
                mol_to_add, 1, "C")


            new_mol = MolMaker.add_mol_B_to_mol_A_and_make_bonds_in_ith_atom_in_A_with_jth_atom_in_B(
                mol_to_add,
                target_mol,
                C_index_in_mol_to_add,
                C_index_in_target,
                Chem.BondType.SINGLE)
            return new_mol

        return warp

    @staticmethod
    def reaction_add_mol_on_nth_carbon_with_O_and_make_CC_single_bond_bak(mol, n):
        def warp(target_mol):
            if isinstance(target_mol, bool): return mol
            mol_to_add = Chem.RWMol(mol)
            now_n = 0
            for a in target_mol.GetAtoms():
                if a.GetSymbol() == "C":
                    now_n += 1
            if now_n == 0:
                return False
            if now_n != 0:
                C_index_in_target = target_mol.GetAtoms()[0].GetIdx()

            C_index_in_mol_to_add, _ = u.get_index_and_total_bond_num_of_the_nth_atom_of_element_with_total_m_bonds(
                mol_to_add, 1, "C")

            new_mol = MolMaker.add_mol_B_to_mol_A_and_make_bonds_in_ith_atom_in_A_with_jth_atom_in_B(
                mol_to_add,
                target_mol,
                C_index_in_mol_to_add,
                C_index_in_target,
                Chem.BondType.SINGLE)
            return new_mol

        return warp