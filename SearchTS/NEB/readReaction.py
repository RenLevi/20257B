from rdkit import Chem
from rdkit.Chem import rdmolops
from NEB.CheckNN import *
from ase.io import write
import numpy as np
import copy
import os
def are_vectors_parallel(v1, v2, tol=1e-6):
    """
    检查两向量是否方向相同（或相反）。
    返回:
        True  (方向相同: 点积 ≈ 1)
        false  (方向相反: 点积 ≈ -1)
        False (其他情况)
    """
    v1_unit = v1 / np.linalg.norm(v1)
    v2_unit = v2 / np.linalg.norm(v2)
    dot_product = np.dot(v1_unit, v2_unit)
    return np.isclose(dot_product, 1.0, atol=tol)
def angle_between_vectors(v1, v2):
    """使用NumPy的线性代数函数"""
    # 归一化向量
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    # 计算夹角的余弦值
    cos_theta = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))
def add_brackets_around_letters(cnmol:str):# 使用正则表达式替换不在[]中的字母字符，前后添加[]:example:[H]CO==>[H][C][O]
    result = re.sub(r'(?<!\[)([a-zA-Z])(?!\])', r'[\g<1>]', cnmol)
    return result
def subHH(STR):
    result = re.sub(r'\[HH\]', '[H]', STR)
    return result
def str2list(reaction:str):
    r1 = reaction.split(">")
    r2 = []
    for i in r1:
        i1 = i.split()
        r2.append(i1)
    return r2
def checkbond(reaction:list,bms1,bms2):
    mol1 = bms1.mol
    mol2 =bms2.mol
    reactiontype = reaction[1][0]
    addatom = reaction[1][-3]#
    bondedatom = reaction[1][-1]
    def COMBINE(mol,add):
        # 创建分子时禁用化合价检查
        params = Chem.SmilesParserParams()
        params.removeHs = False  # 不自动移除氢原子
        params.sanitize = False  # 禁用所有检查
        addmol = Chem.MolFromSmiles(add_brackets_around_letters(add),params=params)
        combined_mol = rdmolops.CombineMols(mol, addmol)
        return combined_mol
    def addATOM():
        if reactiontype == 'Add':
            if len(addatom) > 1 and 'C' in addatom:
                return 'C'
            elif addatom =='OH':
                return 'O'
            else:
                return addatom
        else:
            if addatom == 'O/OH':
                return 'O'
            else:
                return addatom
    def warp(cs12,add=addatom):
        check_mol = copy.deepcopy(cs12[1])
        check_mol = COMBINE(check_mol,add)
        bonds = cs12[0].GetBonds()
        AA=addATOM()
        aset = {AA,bondedatom}
        for bond in bonds:
            mol = copy.deepcopy(cs12[0])
            begin_atom_id = bond.GetBeginAtomIdx()
            end_atom_id = bond.GetEndAtomIdx()
            begin_atom = mol.GetAtomWithIdx(begin_atom_id)
            end_atom = mol.GetAtomWithIdx(end_atom_id)
            qset = {begin_atom.GetSymbol(),end_atom.GetSymbol()}
            if qset == aset:
                mol.RemoveBond(begin_atom_id, end_atom_id)
                if subHH(Chem.MolToSmiles(mol)) == Chem.MolToSmiles(check_mol):
                    return begin_atom.GetIdx(),end_atom.GetIdx(),Chem.MolToSmiles(cs12[0]),Chem.MolToSmiles(check_mol)
                else:
                    pass
            else:
                pass
        return False, False, False, False
    if reactiontype == 'Add':
        cs12 = (mol2,mol1)
        return warp(cs12)
    elif reactiontype == 'Remove':
        cs12 = (mol1,mol2)
        if addatom == 'O/OH':
            o1,o2,o3,o4 = warp(cs12,add='O')
            if o1 == False and o2 == False and o3 == False:
                return warp(cs12,add='OH')
            else:
                return o1,o2,o3,o4
        else:
            return warp(cs12)          
def check_molecule_over_surface(atoms):
    zlist = []
    molz = []
    for atom in atoms:
        if atom.symbol not in ['C','H','O']:
            zlist.append(atom.position[2])
    z_max = max(zlist)
    for atom in atoms:
        if atom.symbol in ['C','H','O']:
            molz.append(atom.position[2])
    z_min = min(molz)
    if z_min <= z_max:
        print(f'部分原子位于催化剂表面以下')
        return False
    else:    return True
def adjust_distance(CB,
                    notmove,nmGidx,
                    move,mGidx,
                    new_distance=3,
                    delta=0,
                    alpha=0,
                    noads=False
                    ):
    """
    调整两个原子之间的距离
    
    参数:
        atoms: ASE Atoms 对象
        index1: 第一个原子的索引
        index2: 第二个原子的索引
        new_distance: 新的距离 (Å)
    """
    bondlist = check_neighbor(notmove,CB)
    atoms = copy.deepcopy(CB.poscar)
    pos1 = atoms.positions[notmove]
    pos2 = atoms.positions[move]
    if noads == False:pass
    else:
        molIdxlist=[]
        for atom in atoms:
            if atom.symbol in ['C','H','O']:
                molIdxlist.append(atom.index)
            else:pass
        group = atoms[molIdxlist]
        for a in group:
            if np.allclose(a.position, pos2, atol=1e-6):
                group_pos2_idx = a.index
            if np.allclose(a.position, pos1, atol=1e-6):
                group_pos1_idx = a.index
        v_important = pos2-pos1
        z= np.array([0,0,-1])
        theta = angle_between_vectors(v_important,z)
        axis_vz= np.cross(v_important,z)
        if np.linalg.norm(axis_vz)>1e-6:
            group.rotate(v=axis_vz,a=theta,center=pos1)
        val=group.positions[group_pos2_idx]-group.positions[group_pos1_idx]
        if are_vectors_parallel(val,np.array([0,0,-1])) == False:
            group.rotate(v=axis_vz,a=-2*theta,center=pos1)
        group.translate((0,0,19-pos1[2]))
        atoms.positions[molIdxlist] = group.positions
    p2=pos2
    p1=pos1
    if np.abs(p2[1]-p1[1])<=0.1 and np.abs(p2[0]-p1[0])<=0.1:
        if  len(bondlist) == 1:
            shift =np.array([0,0.1,0])
        else:
            if len(bondlist)==2:
                v=[]
                for ba in bondlist:
                    if ba != move:
                        v.append(atoms.positions[ba]-p1)
                vd = -v[0]/np.linalg.norm(v[0])
                shift =vd
            if len(bondlist)==3:
                v=[]
                for ba in bondlist:
                    if ba != move:
                        v.append(atoms.positions[ba]-p1)
                v1 = v[0]
                v2 = v[-1]
                vd = np.cross(v1,v2)/np.linalg.norm(np.cross(v1,v2))
                shift = np.array([vd[0],vd[1],0])/np.linalg.norm(np.array([vd[0],vd[1],0]))/10
            else:
                shift = np.array([0,0.1,0])
    else:
        shift = np.array([0,0,0])
    p21=p2-p1
    ph=np.array([0,0,p1[-1]-p2[-1]])
    p21xy=np.array([p21[0],p21[1],0])
    vd = (p21+ph+shift)/np.linalg.norm(p21+ph+shift)*new_distance-p21xy+ph
    v_final =copy.deepcopy(vd)
    # 移动第二个原子到新位置
    for id in mGidx:
        atoms.positions[id] = atoms.positions[id] + v_final 
    for atom in atoms:
        if atom.symbol in ['C','H','O']:
            aid = atom.index
            atoms.positions[aid] = atoms.positions[aid]+np.array([0,0,alpha+delta])
    if noads == False:pass
    else:
        addgroup = atoms[mGidx]
        for a in addgroup:
            if np.allclose(a.position, pos2, atol=1e-6):
                addgroup_pos2_idx = a.index
        v_important = pos2-pos1
        z= np.array([0,0,-1])
        theta = angle_between_vectors(v_important,z)
        axis_vz= np.cross(v_important,z)
        addgroup.rotate(v=axis_vz,a=theta,center=pos2)
        addval=addgroup.positions[addgroup_pos2_idx]-pos1
        if are_vectors_parallel(addval,np.array([0,0,-1])) == False:
            addgroup.rotate(v=axis_vz,a=-2*theta,center=pos1)
        atoms.positions[mGidx]=addgroup.positions
    return atoms
def check_neighbor(id,cb):
    idx = []
    centeratom = cb.atoms[id]
    bonddict = centeratom.bonddict
    for atom in bonddict:
       if atom.id not in idx:
            idx.append(atom.id)
    return idx
def spilt_group_for_C1(id_in,id_notin,cb):
    pl = check_neighbor(id_in,cb)
    pl.append(id_in)
    pl.remove(id_notin)
    pl=list(set(pl))
    return pl
def spilt_group(id_in,id_notin,cb):
    def warp(pls:list):
        if id_in in pls:
            pls.remove(id_in)
        if id_notin in pls:
            pls.remove(id_notin)
    pl = check_neighbor(id_in,cb)
    warp(pl)
    pl_num =len(pl)
    count = 0
    while count != 2:
        for pi in pl:
            pil=check_neighbor(pi,cb)
            pl =list(set(pl+pil))
        warp(pl)
        if len(pl) ==  pl_num:count+=1
        else:pl_num = len(pl)
    pl.append(id_in)
    return pl
class readreaction():
    def __init__(self,file1,file2,reaction,noads=False):# file1> reaction > file2
        self.mol1 = file1
        self.mol2 = file2
        self.r = str2list(reaction)
        self.r_str = reaction
        self.noads = noads
        self.group1 = []
        self.group2 = []
    def readfile(self):
        def warp(id1,id2,cb):
            atoms = cb.poscar
            p1 = atoms.positions[id1]
            p2 = atoms.positions[id2]
            z1=p1[-1]
            z2=p2[-1]
            if z2 >= z1:
                return id1,id2
            else:return id2,id1
        CB1 = checkBonds()
        CB1.input(self.mol1)
        CB1.AddAtoms()
        CB1.CheckAllBonds()
        CB2 = checkBonds()
        CB2.input(self.mol2)
        CB2.AddAtoms()
        CB2.CheckAllBonds()
        BMS1 = BuildMol2Smiles(CB1)
        BMS1.build()
        BMS2 = BuildMol2Smiles(CB2)
        BMS2.build()
        begin_id,end_id,smilesFORcheck,smilesFORspilt = checkbond(self.r,BMS1,BMS2)
        Bid_infile = begin_id +BMS1.metal 
        Eid_infile = end_id +BMS1.metal
        reactiontype = self.r[1][0]
        addatom = self.r[1][1]
        bondedatom = self.r[1][-1]
        '''
        确保FS为成键后产物
        '''
        if reactiontype == 'Add':
            CB = CB2
            self.nebFS = CB2.poscar
        else:
            CB = CB1
            self.nebFS = CB1.poscar
        if bool(CB.adsorption) == False:
            noads = True
        else:
            noads = False
        notmove,move = warp(Bid_infile,Eid_infile,CB)
        notmoveGroupIdx = spilt_group(notmove,move,CB)
        moveGroupIdx = spilt_group(move,notmove,CB)
        self.group1 = notmoveGroupIdx#main body
        self.group2 = moveGroupIdx#sub body
        newmoll=[]
        for a in [0,0.2,0.4,0.6,0.8,1.0]:
            newmol = adjust_distance(CB,notmove,notmoveGroupIdx,move,moveGroupIdx,alpha=a,noads=noads)
            if check_molecule_over_surface(newmol) == False:
                    for i in range(1,20):
                        newmol = adjust_distance(CB,notmove,notmoveGroupIdx,move,moveGroupIdx,alpha=a,delta=0.1*i,noads=noads)
                        if check_molecule_over_surface(newmol) == True:
                            print('higher over surface')
                            break
            newmoll.append(newmol)
        self.nebIS = newmoll
        self.check =smilesFORcheck 
        self.split =smilesFORspilt
    def save(self,path,format):
        # 保存为POSCAR文件（VASP格式）
        if format=='poscar' or 'POSCAR' or 'vasp':
            os.makedirs(f'{path}ISs', exist_ok=True)
            for a in range(6):
                write(f'{path}ISs/{a*2}.vasp', self.nebIS[a], format='vasp', vasp5=True)
            #write(path+'IS.vasp', self.nebIS, format='vasp', vasp5=True)  # vasp5=True添加元素名称
            write(path+'FS.vasp', self.nebFS, format='vasp', vasp5=True)  # vasp5=True添加元素名称
        else:
            print('format should be .vasp')
        ''' #test IS whether the bond ia breaked     
        test = checkBonds()
        test.input(path+'IS.vasp')
        test.AddAtoms()
        test.CheckAllBonds()
            #return TypeError   
        output = BuildMol2Smiles(test)
        output.build()
        if output.smiles == self.check:
            print(f'IS:{output.smiles},Check:{self.check}',output.smiles == self.check,'\n'
              'Error:the bond that should be breaked is not breaked'
              )
            return ValueError
        if output.ads == []:
            print('Warning:there is not adsportion in model')'''        
            



        
if (__name__ == "__main__"):
        # 使用示例
    file1 = "test/opt/system/species/[H]OC([H])[H]/1/nequipOpt.traj"
    file2 = "test/opt/system/species/[H]OC([H])([H])O/1/nequipOpt.traj"
    reaction ='[H]OC([H])[H] > Add O on C > [H]OC([H])([H])O'
    PATH =  "test/"
    RR = readreaction(file1,file2,reaction)
    RR.readfile()
    RR.save(PATH,'POSCAR')






        

                






    

