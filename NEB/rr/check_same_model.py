from ase.io import read
from ase.data import covalent_radii, atomic_numbers
from ase import atom
import numpy as np
import json
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
def check_NON_metal_atoms(atom):
    non_metal_list =[1,2,5,6,7,8,9,10,14,15,16,17,18,33,34,35,36,52,53,54,85,86,117,118]
    if atom.number in non_metal_list:
        return True
    else:
        return False
class N_atom:
    def __init__(self, coord, element,number,index):
        self.xyz = coord
        self.id = index
        self.elesymbol = element
        self.number = number
        self.bonddict = {}
        self.bondtype = {}
        self.charge = 0
class checksame():
    def __init__(self):
        self.atoms = []
        self.poscar = atom
        self.ads_metal = []
    def input(self,filename):
        self.poscar = read(filename)
    def AddAtoms(self):
        atoms= self.poscar
        atoms_info = []
        for i, atom in enumerate(atoms):
            atominfo = N_atom(atom.position,atom.symbol,atom.number,i)
            atoms_info.append(atominfo)
        self.atoms = atoms_info   
    def min_dis(self,atomID1,atomID2):
        distance = self.poscar.get_distance(atomID1,atomID2, mic=True)
        return distance
    def CheckBondwith2Atoms(self,main_atomID,sub_atomID):
        dis = self.min_dis(main_atomID,sub_atomID)
        main_atom  = self.atoms[main_atomID] 
        sub_atom = self.atoms[sub_atomID]
        if check_NON_metal_atoms(main_atom) == True or check_NON_metal_atoms(sub_atom) == True:
            if check_NON_metal_atoms(main_atom) == True and check_NON_metal_atoms(sub_atom) == True:
                if bond(main_atom.elesymbol,sub_atom.elesymbol,dis).judge_bondorder() == 1:
                    main_atom.bonddict[sub_atom] = sub_atom.number
                    sub_atom.bonddict[main_atom] = main_atom.number
                else:
                    pass
            else:
                if bond(main_atom.elesymbol,sub_atom.elesymbol,dis).judge_bondorder() == 1:
                    if check_NON_metal_atoms(main_atom) == False:
                        self.ads_metal.append(main_atom)
                    else:
                        self.ads_metal.append(sub_atom)
        else:
            pass
    def CheckAllBonds(self):
        atoms = self.poscar
        for i, atom_i in enumerate(atoms):
            for j, atom_j in enumerate(atoms):
                if j > i:
                    self.CheckBondwith2Atoms(i,j)
                else:
                    pass
        #print('finish checking ALL bonds')
    def cal_metal_vector(self):
        m = []
        if self.ads_metal == []:
            print('NO ads')
        else:
            for i in self.ads_metal:
                m.append([i])
            if len(m) == 1:
                p = m[0].xyz-m[0].xyz
            else:
                p = np.zeros((len(m),len(m),3))
                for i in range(len(m)):
                    for j in range(len(m)):
                        p[i][j]=m[i].xyz-m[j].xyz
            return p
                        

    def cal_vector(self):
        a=[]
        m=[]
        if self.ads_metal == []:
            print('NO ads')
        else:
            for Natom in self.atoms:
                if check_NON_metal_atoms(Natom) == True:
                    a.append([Natom])
                else:
                    pass
            for i in self.ads_metal:
                m.append([i])
        PA = np.array(a)
        PM = np.array(m)
        PAIM = np.repeat(PA, len(m),axis=1).T
        PMIA = np.repeat(PM, len(a), axis=1)
        result = np.zeros((len(m),len(a),3))
        for i in range(len(m)):
            for j in range(len(a)):
                result[i][j]=PAIM[i][j].xyz-PMIA[i][j].xyz
        return result
class in_out_v():
    def __init__(self,record_json,path):#path = /work/home/ac877eihwp/renyq/20250828TT/test/
        with open(record_json,'r') as f:
            dictionary = json.load(f)
        self.d = dictionary
        self.p = path
    def cal_all(self):
        dict4cal = self.d
        for name in dict4cal:
            cpl = dict4cal[name][0]
            if cpl == []:
                print('NO model check pass')
            else:
                V = []
                for i in cpl:
                    CB =checksame()
                    CB.input(f'{self.p}opt/system/species/{name}/{i}/nequipOpt.traj')
                    CB.AddAtoms()
                    CB.CheckAllBonds()
                    vc = CB.cal_vector()
                    V.append(vc)
                    print(name,i,vc.shape)

        


if (__name__ == "__main__"):
    iov = in_out_v('record_adscheck.json','test/')
    iov.cal_all()
