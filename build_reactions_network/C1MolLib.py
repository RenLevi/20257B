from MEACRNG.MolEncoder.MolMaker import MolMaker, MMConnectConfig
from rdkit.Chem import AllChem as Chem


class C1MolMarker:
    @staticmethod
    def make_base_on_C(mmcc_list):# 在C的基础上加原子，并设置键种类，返回编辑好的rdkit分子。mmcc_list是要加入键种类（包括原子）的list
        return Chem.MolFromSmarts(
            Chem.MolToSmarts(
                MolMaker.connect_bonds_from_config_to_first_atom_A_in_initial_mol(
                    Chem.RWMol(Chem.MolFromSmarts("C")),
                    mmcc_list,
                    "C")))


class C1MolLib:
    H_single = MMConnectConfig(Chem.BondType.SINGLE,# 设置-H键
                               1,
                               0)
    O_double = MMConnectConfig(Chem.BondType.DOUBLE,# 设置=O键
                               8,
                               0)
    O_single = MMConnectConfig(Chem.BondType.SINGLE,# 设置-O键
                               8,
                               0)
    OH_single = MMConnectConfig(Chem.BondType.SINGLE,# 设置-OH键
                                8,
                                1)
    CH = C1MolMarker.make_base_on_C([H_single])# 这里制作分子就显而易见了，注意吸附态时不考虑双键
    CH2 = C1MolMarker.make_base_on_C([H_single] * 2)
    CH3 = C1MolMarker.make_base_on_C([H_single] * 3)
    CO = C1MolMarker.make_base_on_C([O_single])
    COH = C1MolMarker.make_base_on_C([OH_single])
    CH2O = C1MolMarker.make_base_on_C([H_single,H_single,O_single])
    CHOH = C1MolMarker.make_base_on_C([H_single,OH_single])
    CH2OH = C1MolMarker.make_base_on_C([H_single,H_single,OH_single])
    CHO = C1MolMarker.make_base_on_C([O_single, H_single])
    HCOO_double_CO_bond = C1MolMarker.make_base_on_C([H_single, O_single, O_double])
    CO2_double_CO_bond = C1MolMarker.make_base_on_C([O_double, O_double])
    # in adsorbate we do not consider double bond!
    COOH_single_CO_bond = C1MolMarker.make_base_on_C([O_single, OH_single])
    CO2_single_CO_bond = C1MolMarker.make_base_on_C([O_single, O_single])
    


