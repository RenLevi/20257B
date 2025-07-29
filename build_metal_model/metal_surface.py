from ase.build import hcp0001,fcc111,fcc110
from ase.constraints import FixAtoms
import os
from ase.io import write
class metal():
    def __init__(self,element,size,vacuum,path,fix_col=2,type='hcp0001'):#元素，尺寸，真空层厚度，保存格式，保存路径:mol_to_ad，固定底层原子层数
        self.s = size
        self.e = element
        self.v = vacuum
        self.p = path
        self.fix = fix_col
        self.type = type
    def build_model(self):
        x = self.s[0]
        y = self.s[1]
        z = self.s[2]
        matel_surface = hcp0001(self.e, size=self.s, vacuum=self.v)
        z_coords = matel_surface.positions[:, 2]
        threshold = self.fix*x*y
        fixed_indices = list(range(0, threshold))
        matel_surface.constraints = [FixAtoms(indices=fixed_indices)]
        self.model = matel_surface
        return matel_surface
    def save_model(self):
        slabname = f'{self.e}_{self.type}.vasp'
        slab_save_path = f'{self.p}/slab'
        def save_file(save_path,filename,model):
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            write(os.path.join(save_path, filename), model, format='vasp', vasp5=True, direct=True)
            '''with open(os.path.join(save_path, filename), 'r') as f:
                lines = f.readlines()
                print("".join(lines[5:8]))  # 查看坐标行示例'''
        save_file(slab_save_path,slabname,self.model)



