from rdkit import Chem
from rdkit.Chem import rdmolops
from build_ISFS_br2bo.preSearchTS.CheckNN import *
from ase.io import write
import numpy as np
import copy
import os
import re
from ase.optimize import BFGS
from rdkit import Chem
from nequip.ase import NequIPCalculator

def svd_rotation_matrix(a, b):
    """
    使用SVD分解计算旋转矩阵
    
    参数:
    a, b: 三维单位向量 (numpy数组)
    
    返回:
    R: 3x3旋转矩阵
    """
    # 确保输入是单位向量
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    
    # 计算协方差矩阵
    H = np.outer(a, b)
    
    # SVD分解
    U, S, Vt = np.linalg.svd(H)
    
    # 计算旋转矩阵
    R = np.dot(Vt.T, U.T)
    
    # 处理反射情况
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    return R
def rotate_point_set(points, A, B, center=None):
    """
    旋转点集使得向量A与向量B同向
    
    参数:
        points: 点集 (n x 3 numpy array)
        A: 原始向量
        B: 目标向量
        center: 旋转中心，如果为None则使用点集中心
    
    返回:
        rotated_points: 旋转后的点集
        rotation_matrix: 使用的旋转矩阵
    """
    # 计算旋转矩阵
    rotation_matrix = svd_rotation_matrix(A,B)#rotate_vector_to_target(A, B)
    
    # 确定旋转中心
    if center is None:
        center = np.mean(points, axis=0)
    
    # 将点集平移到旋转中心，旋转，再平移回去
    translated_points = points - center
    rotated_translated_points = np.dot(translated_points, rotation_matrix.T)
    rotated_points = rotated_translated_points + center
    
    return rotated_points, rotation_matrix
def verify_rotation(original_A, rotated_A, target_B):
    """验证旋转是否正确"""
    original_A_norm = original_A / np.linalg.norm(original_A)
    rotated_A_norm = rotated_A / np.linalg.norm(rotated_A)
    target_B_norm = target_B / np.linalg.norm(target_B)
    
    dot_product = np.dot(rotated_A_norm, target_B_norm)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi
    
    print(f"旋转后向量与目标向量的夹角: {angle:.6f} 度")
    print(f"方向一致性: {dot_product:.10f} (应该接近1.0)")
    
    return abs(dot_product - 1.0) < 1e-10
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
    clean_list_reaction =[]
    for id in range(len(reaction[1])):
        if bool(re.search(r'\d', reaction[1][id])) == False:
            clean_list_reaction.append(reaction[1][id])
    addatom = clean_list_reaction[1]#reaction[1][-3]
    bondedatom = clean_list_reaction[-1]#reaction[1][-1]
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
        numbeforeadd=check_mol.GetNumAtoms()
        check_mol = COMBINE(check_mol,add)
        mol = cs12[0]
        if check_mol.GetNumAtoms() != mol.GetNumAtoms():
            return [False,False,False,False]
        bonds = cs12[0].GetBonds()
        AA=addATOM()
        aset = {AA,bondedatom}
        check=100
        outlist = [None,None,None,None]
        bms_check = cs12[-1]
        adsIDx = []
        for a in bms_check.ads:
            id = a.id
            adsIDx.append(id)
        for bond in bonds:
            mol = copy.deepcopy(cs12[0])
            begin_atom_id = bond.GetBeginAtomIdx()
            end_atom_id = bond.GetEndAtomIdx()
            begin_atom = mol.GetAtomWithIdx(begin_atom_id)
            end_atom = mol.GetAtomWithIdx(end_atom_id)
            qset = {begin_atom.GetSymbol(),end_atom.GetSymbol()}
            bms = cs12[2]
            cb = bms.cb
            if qset == aset:
                if begin_atom_id >= numbeforeadd:
                    checkid = end_atom_id
                else:
                    checkid = begin_atom_id
                if checkid + bms_check.metal in adsIDx:
                    mol.RemoveBond(begin_atom_id, end_atom_id)
                    if subHH(Chem.MolToSmiles(mol)) == Chem.MolToSmiles(check_mol):
                        outlist = [begin_atom.GetIdx(),end_atom.GetIdx(),Chem.MolToSmiles(cs12[0]),Chem.MolToSmiles(check_mol)]
        for bond in bonds:
            mol = copy.deepcopy(cs12[0])
            begin_atom_id = bond.GetBeginAtomIdx()
            end_atom_id = bond.GetEndAtomIdx()
            begin_atom = mol.GetAtomWithIdx(begin_atom_id)
            end_atom = mol.GetAtomWithIdx(end_atom_id)
            qset = {begin_atom.GetSymbol(),end_atom.GetSymbol()}
            bms = cs12[2]
            ads = [] 
            for Natom in bms.ads:
                ads.append(Natom.id)
            if qset == aset:
                bmsidB=begin_atom.GetIdx()+bms.metal
                bmsidE=end_atom.GetIdx()+bms.metal
                mol.RemoveBond(begin_atom_id, end_atom_id)
                if subHH(Chem.MolToSmiles(mol)) == Chem.MolToSmiles(check_mol):
                    outlist = [begin_atom.GetIdx(),end_atom.GetIdx(),Chem.MolToSmiles(cs12[0]),Chem.MolToSmiles(check_mol)]
        return outlist[0],outlist[1],outlist[2],outlist[3]
    if reactiontype == 'Add':
        cs12 = (mol2,mol1,bms2,bms1)
        return warp(cs12)
    elif reactiontype == 'Remove':
        cs12 = (mol1,mol2,bms1,bms2)
        if addatom == 'O/OH':
            o1,o2,o3,o4 = warp(cs12,add='O')
            if o1 == False and o2 == False and o3 == False and o4 == False:
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
def adjust_distance_old(CB,
                    notmove,nmGidx,
                    move,mGidx,
                    new_distance=3,
                    delta=0,
                    alpha=0,
                    noads=False
                    ):
    """
    调整两个原子之间的距离
    """
    bondlist = check_neighbor(notmove,CB)
    atoms = copy.deepcopy(CB.poscar)
    pos1 = atoms.positions[notmove]
    pos2 = atoms.positions[move]

    '''if noads == False:pass
    else:
        molIdxlist=[]
        for atom in atoms:
            if atom.symbol in ['C','H','O']:
                molIdxlist.append(atom.index)
            else:pass
        group = atoms[molIdxlist]
        v_important = pos2-pos1
        z= np.array([0,0,-1])
        rotated_points,_ = rotate_point_set(group.positions,v_important,z,center=pos1)
        #group.translate((0,0,19-pos1[2]))
        atoms.positions[molIdxlist] = rotated_points'''
        
    p2=atoms.positions[move]
    p1=atoms.positions[notmove]
    main = atoms[nmGidx]
    sub = atoms[mGidx]
    MASSCENTER_main = main.get_center_of_mass()
    MASSCENTER_sub = sub.get_center_of_mass()
    V_MAIN = MASSCENTER_main-p1
    V_SUB = MASSCENTER_sub -p2
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

    for id in mGidx:
        atoms.positions[id] = atoms.positions[id] + v_final 
    for atom in atoms:
        if atom.symbol in ['C','H','O']:
            aid = atom.index
            atoms.positions[aid] = atoms.positions[aid]+np.array([0,0,alpha+delta])

    '''if noads == False:pass
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
        atoms.positions[mGidx]=addgroup.positions'''
    return atoms

def adjust_distance(CB,
                    notmove,nmGidx,
                    move,mGidx,
                    new_distance=3,
                    delta=0,
                    alpha=0,
                    ):
    """
    调整两个原子之间的距离
    """
    atoms = copy.deepcopy(CB.poscar)
    pos1 = atoms.positions[notmove]
    pos2 = atoms.positions[move]
    p2=atoms.positions[move]
    p1=atoms.positions[notmove]
    p21=p2-p1
    vd = p21/np.linalg.norm(p21)*new_distance-p21
    for id in mGidx:
        atoms.positions[id] = atoms.positions[id] + vd
    for atom in atoms:
        if atom.symbol in ['C','H','O']:
            aid = atom.index
            atoms.positions[aid] = atoms.positions[aid]+np.array([0,0,alpha+delta])
    return atoms
def check_neighbor(id,cb):
    idx = []
    centeratom = cb.atoms[id]
    bonddict = centeratom.bonddict
    for atom in bonddict:
       if atom.id not in idx:
            idx.append(atom.id)
    return idx
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
class MultiDistanceAwareOptimizer(BFGS):
    """监控多个原子对距离的优化器"""
    
    def __init__(self, atoms, distance_constraints, force_scale=0.1, 
                 trajectory=None, logfile=None, master=None):
        """
        参数:
        distance_constraints: 列表，每个元素为 (atom_i, atom_j, limit_distance, mode)
        mode = 0 : dij > limt_d时增加吸引力
        mode = 1 :dij < limit_d时增加排斥力
        """
        super().__init__(atoms, trajectory, logfile, master)
        self.distance_constraints = distance_constraints
        self.force_scale = force_scale
        
    def check_and_adjust_forces(self, forces):
        """检查多个原子对距离并调整力"""
        positions = self.atoms.positions
        for i, j, limit_dist, mode in self.distance_constraints:
            distance = np.linalg.norm(positions[i] - positions[j])
            if mode == 0:
                if distance > limit_dist:
                    direction = positions[j] - positions[i]
                    direction_unit = direction / np.linalg.norm(direction)
                    
                    current_force = np.linalg.norm(forces[i] - forces[j])
                    extra_force = self.force_scale * current_force * (distance - limit_dist)
                    
                    forces[i] += extra_force * direction_unit
                    forces[j] -= extra_force * direction_unit
                    
                    print(f"调整原子对 ({i},{j}) 受力，距离: {distance:.4f} Å")
            elif mode == 1:
                if distance < limit_dist:
                    direction = positions[j] - positions[i]
                    direction_unit = direction / np.linalg.norm(direction)
                    
                    current_force = np.linalg.norm(forces[i] - forces[j])
                    extra_force = self.force_scale * current_force * (distance - limit_dist)-np.array([0,0.1,0])
                    
                    forces[i] += extra_force * direction_unit
                    forces[j] -= extra_force * direction_unit
                    
                    print(f"调整原子对 ({i},{j}) 受力，距离: {distance:.4f} Å")
        return forces
    
    def step(self, forces=None):
        if forces is None:
            forces = self.atoms.get_forces()
        
        adjusted_forces = self.check_and_adjust_forces(forces)
        super().step(adjusted_forces)
class readreaction():
    def __init__(self,file1,file2,reaction):# file1> reaction > file2
        self.mol1 = file1
        self.mol2 = file2
        self.r = str2list(reaction)
        self.r_str = reaction
        self.group1 = []
        self.group2 = []
        self.changebondatom = None
        self.stop = False
    def readfile(self):
        def warp(id1,id2,cb):
            id1s = spilt_group(id1,id2,cb)
            id2s = spilt_group(id2,id1,cb)
            atoms = cb.poscar
            id1mol = atoms[id1s]
            id2mol = atoms[id2s]
            mass1 = sum(id1mol.get_masses())
            mass2 = sum(id2mol.get_masses())
            p1 = atoms.positions[id1]
            p2 = atoms.positions[id2]
            z1=p1[-1]
            z2=p2[-1]
            if len(id1s) >1 or len(id2s) >1:
                if mass1 >= mass2:
                    return id1,id2
                else:return id2,id1
            else:
                if z1 <= z2:
                    return id1,id2
                else:
                    return id2,id1
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
        
        if begin_id == None:
            print(f'{self.r_str}:checkbond wrong!')
            self.stop = True
        else:
            Bid_infile = begin_id +BMS1.metal 
            Eid_infile = end_id +BMS1.metal
            reactiontype = self.r[1][0]
            '''
            确保FS为成键后产物
            '''
            if reactiontype == 'Add':
                CB = CB2
                self.mol1 = CB2.poscar
                self.molINFO = [CB2,BMS2]
            else:
                CB = CB1
                self.mol1 = CB1.poscar
                self.molINFO = [CB1,BMS1]
            if bool(CB.adsorption) == False:
                noads = True
            else:
                noads = False
            notmove,move = warp(Bid_infile,Eid_infile,CB)
            notmoveGroupIdx = spilt_group(notmove,move,CB)
            moveGroupIdx = spilt_group(move,notmove,CB)
            self.group1 = notmoveGroupIdx#main body
            self.group2 = moveGroupIdx#sub body
            self.changebondatom = (notmove,move)#(Bid_infile,Eid_infile)
            newmol = adjust_distance(CB,notmove,notmoveGroupIdx,move,moveGroupIdx,alpha=0)
            if check_molecule_over_surface(newmol) == False:
                    for i in range(1,20):
                        newmol = adjust_distance(CB,notmove,notmoveGroupIdx,move,moveGroupIdx,alpha=0,delta=0.1*i)
                        if check_molecule_over_surface(newmol) == True:
                            print(f'higher over surface:alpha={0.1*i}')
                            break
            self.mol2 = newmol
            self.check =smilesFORcheck 
            self.split =smilesFORspilt
            '''if reactiontype == 'Add':
                self.IS = self.mol2
                self.FS = self.mol1
            else:
                self.IS = self.mol1
                self.FS = self.mol2'''
    def run_MDAO(self,MLPs_model_path):
        calc = NequIPCalculator.from_deployed_model(MLPs_model_path, device='cpu')
        twogroups=self.mol2
        twogroups.calc = calc
        distance_constraints = []
        CB = self.molINFO[0]
        bondsetlist = CB.bondsetlist
        changebondatom = self.changebondatom
        for biatoms in bondsetlist:
            (ith_atom_id,jth_atom_id) = biatoms
            if ith_atom_id not in changebondatom or jth_atom_id not in changebondatom:
                distance_constraints.append((ith_atom_id, jth_atom_id, limit_distance, 0))
            else:
                distance_constraints.append((ith_atom_id, jth_atom_id,3, 1))
            




        

    def save(self,path,format):
        # 保存为POSCAR文件（VASP格式）
        if format=='poscar' or 'POSCAR' or 'vasp':
            if self.stop == False:
                write(path+'IS.vasp', self.IS, format='vasp', vasp5=True)  # vasp5=True添加元素名称
                write(path+'FS.vasp', self.FS, format='vasp', vasp5=True)  # vasp5=True添加元素名称

        else:
            print('format should be .vasp')

if (__name__ == "__main__"):
        # 使用示例
    file1 = "test/opt/system/species/[H]OC([H])[H]/1/nequipOpt.traj"
    file2 = "test/opt/system/species/[H]OC([H])([H])O/1/nequipOpt.traj"
    reaction ='[H]OC([H])[H] > Add O on C > [H]OC([H])([H])O'
    PATH =  "test/"
    RR = readreaction(file1,file2,reaction)
    RR.readfile()
    RR.save(PATH,'POSCAR')






        

                






    

