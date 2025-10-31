from MEACRNG.Tools.utils import get_total_bond_num_regarding_triple_double_single_as_3_2_1
from MEACRNG.Tools.utils import get_num_of_single_double_triple_bonds_and_number_of_atom_B_with_centerd_atom_A
from rdkit.Chem import AllChem as Chem
from MEACRNG.CRNG.ReactionNetworkGenerator import NetworkGenerator
class CarbonValidMolCheckFunc:

    @staticmethod
    def num_of_C_and_O_should_smaller_equal_than_2(mol):
        '''
        分子中的C和O的个数之和需要小于等于2，
        不允许COO这个骨架！
        '''
        num_carbon = 0
        num_O = 0
        for a in mol.GetAtoms():
            if a.GetSymbol() == "C":
                num_carbon += 1
            elif a.GetSymbol() == "O":
                num_O += 1
        if num_carbon + num_O > 2:
            return False
        return True

    @staticmethod
    def num_of_C_and_O_should_smaller_equal_than_3(mol):
        '''
        分子中的C和O的个数之和需要小于等于3，
        实际上是不允许OCCO这个骨架！
        '''
        num_carbon = 0
        num_O = 0
        for a in mol.GetAtoms():
            if a.GetSymbol() == "C":
                num_carbon += 1
            elif a.GetSymbol() == "O":
                num_O += 1
        if num_carbon + num_O > 3:
            return False
        return True

    @staticmethod
    def num_of_O_should_smaller_equal_than_2(mol):
        '''
        分子中O的个数之和需要小于等于2，
        '''
        num_O = 0
        for a in mol.GetAtoms():
            if a.GetSymbol() == "O":
                num_O += 1
        if num_O > 2:
            return False
        return True






    @staticmethod
    def num_of_atom_N_should_smaller_equal_than_M(n='H',m=0):
        '''
        限制单个原子个数的函数
        分子中'N'原子的个数之和需要小于等于'M'，
        '''
        def warp(mol):
            num = 0
            for a in mol.GetAtoms():
                if a.GetSymbol() == n:
                    num += 1
            if num > m:
                return False
            return True
        return warp

    @staticmethod
    def sum_atom_num_in_list_N_should_smaller_equal_than_M(n=[],m=0):
        '''
        限制原子list中总个数的函数
        list中所有原子的个数之和需要小于等于'M'，
        '''
        def warp(mol):
            num = 0
            for a in mol.GetAtoms():
                if a.GetSymbol() in n:
                    num += 1
            if num > m:
                return False
            return True
        return warp

    @staticmethod
    def no_more_than_M_bond_in_atom_N_check(n='C',m=4):
        '''
        一个N原子上最多连接m个键
        :param mol:
        :return:
        '''
        def warp(mol):
            for a in mol.GetAtoms():
                if a.GetSymbol() == n:
                    num_bonds = get_total_bond_num_regarding_triple_double_single_as_3_2_1(a)
                    if num_bonds > m:
                        return False
            return True
        return warp





    @staticmethod
    def num_of_H_should_bigger_equal_than_1(mol):
        '''
        碳上最少要有一个H，C除外
        '''

        # print(Chem.MolToSmiles(mol))
        for a in mol.GetAtoms():
            if a.GetSymbol() == "C":
                num_C = 0
                for b in a.GetBonds():
                    if b.GetBeginAtom().GetSymbol() == "C" and b.GetEndAtom().GetSymbol() == 'C':
                        num_C += 1
                # print('num_C'+str(num_C))
                if num_C == 0:
                    return True
                if num_C > 0:
                    num_H = 0
                    num_O = 0
                    for b in a.GetBonds():
                        if b.GetBeginAtom().GetSymbol() == "H" or b.GetEndAtom().GetSymbol() == 'H':
                            num_H += 1
                        # if b.GetBeginAtom().GetSymbol() == "O" or b.GetEndAtom().GetSymbol() == 'O':
                            # num_O += 1
                    # print('num_H' + str(num_H+num_O))
                    if num_H + num_O == 0:
                        return False

        return True

    @staticmethod
    def num_of_CHO_should_bigger_equal_than_1(mol):
        '''
        碳上最少要有一个H，C除外
        '''

        # print(Chem.MolToSmiles(mol))
        for a in mol.GetAtoms():
            if a.GetSymbol() == "C":
                num_C = 0
                for b in a.GetBonds():
                    if b.GetBeginAtom().GetSymbol() == "C" and b.GetEndAtom().GetSymbol() == 'C':
                        num_C += 1
                # print('num_C'+str(num_C))
                if num_C == 0:
                    return True
                if num_C > 0:
                    num_H = 0
                    num_O = 0
                    for b in a.GetBonds():
                        if b.GetBeginAtom().GetSymbol() == "H" or b.GetEndAtom().GetSymbol() == 'H':
                            num_H += 1
                        # if b.GetBeginAtom().GetSymbol() == "O" or b.GetEndAtom().GetSymbol() == 'O':
                            # num_O += 1
                    # print('num_H' + str(num_H+num_O))
                    if num_H + num_O == 0:
                        return False

        return True

    @staticmethod
    def num_of_C_and_O_should_smaller_equal_than_4(mol):
        '''
        分子中的C和O的个数之和需要小于等于4，C总是和O连在一起，因此中的CO骨架大小小于等于4
        '''
        num_carbon = 0
        num_O = 0
        for a in mol.GetAtoms():
            if a.GetSymbol() == "C":
                num_carbon += 1
            elif a.GetSymbol() == "O":
                num_O += 1
        if num_carbon + num_O > 4:
            return False
        return True

    @staticmethod
    def num_of_O_should_smaller_equal_than_2(mol):
        '''
        分子中的O的个数之和需要小于等于3
        '''
        num_O = 0
        for a in mol.GetAtoms():
            if a.GetSymbol() == "O":
                num_O += 1
        if num_O > 2:
            return False
        return True

    @staticmethod
    def no_O_when_have_two_carbon(mol):
        '''
        当有分子中有两个C的时候，如果此时没有O，就不要这个物种
        '''
        num_carbon = 0
        num_O = 0
        for a in mol.GetAtoms():
            if a.GetSymbol() == "C":
                num_carbon += 1
            elif a.GetSymbol() == "O":
                num_O += 1
        if num_carbon == 2 and num_O == 0:
            return False
        return True



    @staticmethod
    def no_more_than_two_carbon_check(mol):
        '''
        碳链最长只能有两个
        :param mol:
        :return:
        '''
        num_carbon = 0
        for a in mol.GetAtoms():
            if a.GetSymbol() == "C":
                num_carbon += 1
        if num_carbon <= 0:
            return False
        if num_carbon > 2:
            return False
        return True

    @staticmethod
    def no_more_than_three_carbon_check(mol):
        '''
        碳链最长只能有三个
        :param mol:
        :return:
        '''
        num_carbon = 0
        for a in mol.GetAtoms():
            if a.GetSymbol() == "C":
                num_carbon += 1
        if num_carbon <= 0:
            return False
        if num_carbon > 3:
            return False
        return True

    @staticmethod
    def no_more_than_four_carbon_check(mol):
        '''
        碳链最长只能有四个
        :param mol:
        :return:
        '''
        num_carbon = 0
        for a in mol.GetAtoms():
            if a.GetSymbol() == "C":
                num_carbon += 1
        if num_carbon <= 0:
            return False
        if num_carbon > 4:
            return False
        return True

    @staticmethod
    def no_more_than_five_carbon_check(mol):
        '''
        碳链最长只能有五个
        :param mol:
        :return:
        '''
        num_carbon = 0
        for a in mol.GetAtoms():
            if a.GetSymbol() == "C":
                num_carbon += 1
        if num_carbon <= 0:
            return False
        if num_carbon > 5:
            return False
        return True

    @staticmethod
    def no_more_than_six_carbon_check(mol):
        '''
        碳链最长只能有六个
        :param mol:
        :return:
        '''
        num_carbon = 0
        for a in mol.GetAtoms():
            if a.GetSymbol() == "C":
                num_carbon += 1
        if num_carbon <= 0:
            return False
        if num_carbon > 6:
            return False
        return True

    @staticmethod
    def no_more_than_four_bond_in_carbon_check(mol):
        '''
        一个C上最多连接四个键
        :param mol:
        :return:
        '''
        for a in mol.GetAtoms():
            if a.GetSymbol() == "C":
                num_bonds = get_total_bond_num_regarding_triple_double_single_as_3_2_1(a)

                if num_bonds > 4:
                    return False
        return True

    @staticmethod
    def valid_C_O_bond_structure_check_when_only_use_single_CO_bond_simple(mol):
        '''
        valid:
        only 1 O bond C
        when 2 O on C, can not be C-OH(-OH)
        '''

        def check_if_H_bonded_O(O_atom):
            for b in O_atom.GetBonds():
                if b.GetBeginAtom().GetSymbol() == "H" or b.GetEndAtom().GetSymbol() == "H":
                    return True
            return False

        for a in mol.GetAtoms():
            if a.GetSymbol() == "C":
                num_C_to_O_bond = 0
                num_C_to_OH_bond = 0
                for b in a.GetBonds():
                    if b.GetBeginAtom().GetSymbol() == "O":
                        num_C_to_O_bond += 1
                        if check_if_H_bonded_O(b.GetBeginAtom()):
                            num_C_to_OH_bond += 1
                    elif b.GetEndAtom().GetSymbol() == "O":
                        num_C_to_O_bond += 1
                        if check_if_H_bonded_O(b.GetEndAtom()):
                            num_C_to_OH_bond += 1

                if num_C_to_O_bond >= 3:
                    return False
                if num_C_to_OH_bond >= 2:
                    return False
        return True

    @staticmethod
    def valid_C_O_bond_structure_check(mol):
        '''
        valid:
        only 1 O bond C
        only C(=O)-O when two O bond C
        '''
        for a in mol.GetAtoms():
            if a.GetSymbol() == "C":
                numO, num_s, num_d, num_t = get_num_of_single_double_triple_bonds_and_number_of_atom_B_with_centerd_atom_A(
                    "O", a)
                if numO <= 1:
                    return True
                if numO == 2:
                    # only C(=O)-O or O=C=O
                    if num_d == 2:
                        return True
                    if num_s == 1 and num_d == 1:
                        return True
                    else:
                        return False
                return False

    @staticmethod
    def unsaturation_check(mol):
        '''
        不饱和度检测，暂时没有实际应用
        :param mol:
        :return:
        '''
        num_carbon = 0
        num_bonds = 0
        for a in mol.GetAtoms():
            if a.GetSymbol() == "C":
                num_carbon += 1
                num_bonds += get_total_bond_num_regarding_triple_double_single_as_3_2_1(a)
        if num_bonds == (num_carbon * 4 ) :
            return False
        return True

    @staticmethod
    def remove_specifal_mol_C6(mol):
        '''
        如果生成物中含有remove_list中的物种，将会被剔除
        请注意list中的物种格式
        :param mol:
        :return:
        '''
        remove_list = [
            'C',
            'OCCO'
        ]
        if Chem.MolToSmiles(mol) in remove_list:
            return False
        return True
    
    @staticmethod
    def remove_mol_had_two_groups(mol):
        '''
        如果生成物中含有多个基团，将会被剔除
        '''
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            return False
        elif len(frags)==1:
            return True
        else:
            return False



if __name__ == '__main__':
    from MolEncoder import C1MolLib

    HCOO = C1MolLib.HCOO

    CarbonValidMolCheckFunc.valid_
    C_O_bond_structure_check(HCOO)

