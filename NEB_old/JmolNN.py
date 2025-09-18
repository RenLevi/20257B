from ase.data import covalent_radii, atomic_numbers
class bond():
    def __init__(self, ele1,ele2,dis):
        self.ele1 = ele1
        self.ele2 = ele2
        self.length = dis
    def judge_bondorder(self):
        # 获取元素的原子序数
        z_i = atomic_numbers[self.ele1]
        z_j = atomic_numbers[self.ele2]
        r_i = covalent_radii[z_i]
        r_j = covalent_radii[z_j]
        delta = 0.45
        if self.length <= r_i+r_j+delta:
            return 1
        else:
            return 0


