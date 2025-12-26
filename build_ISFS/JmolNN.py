from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN,VoronoiNN,JmolNN
from pymatgen.io.ase import AseAtomsAdaptor
class bond():
    def __init__(self,atoms):
        self.structure = AseAtomsAdaptor.get_structure(atoms)
    def judge_bondorder(self):
        structure = self.structure
        # 获取元素的原子序数
        jnn = JmolNN()
        neighbors_info_list = []
        neighbors_idx_list = []
        for i in range(0,len(structure)):
            neighbors_info = jnn.get_nn_info(structure, i)
            neighbors_info_list.append(neighbors_info)
            neighbors_idx = []
            for dict_i in neighbors_info:
                neighbors_idx.append(dict_i['site_index'])
            neighbors_idx_list.append(neighbors_idx)
                
        return neighbors_info_list,neighbors_idx_list

