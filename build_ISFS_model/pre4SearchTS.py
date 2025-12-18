import build_ISFS_model.readReaction as rR
import os
import json
from ase.io import read
import numpy as np
from collections import Counter,defaultdict
import matplotlib.pyplot as plt
from ase import Atoms
from scipy.spatial import cKDTree
'''
    #:注释
    #...:用于测试，可以删除
    #?:存疑
'''
def json_r_w(name,data):
    with open(name, 'r') as f:
        file = f.read()
        if len(file)>0:
            ne = 'ne'
        else:
            ne = 'e'
    if ne == 'ne':
        with open (name,'r') as f:
            old_data = json.load(f)
    else:
        old_data ={}
    old_data.update(data)
    with open(name, 'w') as f:
        json.dump(old_data,f,indent=2)
def get_fmax_from_traj(traj_file):
    # 读取轨迹文件的最后一个结构（单个Atoms对象）
    atoms = read(traj_file, index=-1)
    # 直接从Atoms对象获取力
    forces = atoms.get_forces()
    force_magnitudes = np.linalg.norm(forces, axis=1)
    fmax = np.max(force_magnitudes)
    return fmax
def find_uppercase_difference(strA, strB):
    """
    找出字符串B比字符串A多出的大写字母
    
    参数:
    strA: 原始字符串
    strB: 目标字符串
    
    返回:
    extra_uppercase: B比A多出的大写字母列表
    """
    # 提取两个字符串中的所有大写字母
    uppercase_A = [char for char in strA if char.isupper()]
    uppercase_B = [char for char in strB if char.isupper()]
    
    # 创建大写字母计数字典
    count_A = {}
    for char in uppercase_A:
        count_A[char] = count_A.get(char, 0) + 1
    
    count_B = {}
    for char in uppercase_B:
        count_B[char] = count_B.get(char, 0) + 1
    
    # 找出B比A多出的大写字母
    extra_uppercase = []
    for char, count in count_B.items():
        if char not in count_A:
            # B中有而A中没有的字母，全部算作多出的
            extra_uppercase.extend([char] * count)
        elif count > count_A[char]:
            # B中比A中多的部分
            extra_uppercase.extend([char] * (count - count_A[char]))
    
    return extra_uppercase
def read_file_line_by_line(file_path):#逐行读取txt文件并返回list数据
    reaction_list=[]
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            string = line.strip()  
            reaction_list.append(string)
        reaction_list.pop(0)
    return reaction_list
class SurfaceSiteFinder:
    def __init__(self, atoms: Atoms, surface_direction: int = 2):
        """
        初始化表面位点查找器
        
        参数:
        atoms: ASE Atoms 对象，表示平板结构
        surface_direction: 表面法线方向 (0=x, 1=y, 2=z)
        """
        self.atoms = atoms
        self.surface_direction = surface_direction
        self.grid_points = None
        self.wrapped_points = None
        self.site_atoms = defaultdict(list)
        self.site_positions = {}
        self.site_special_vectors = {}
        self.site_vectors = {}
        self.cell = self.atoms.get_cell()
        self.pbc = self.atoms.get_pbc()
        self._generate_replicas()
    def _generate_replicas(self):
        """生成必要的镜像原子以处理周期性边界条件"""
        # 确定每个方向需要复制的数量
        # 对于最近邻搜索，通常只需要相邻的镜像
        replicas = []
        for i, pbc in enumerate(self.pbc):
            if pbc:
                replicas.append([-2, -1, 0, 1, 2])
            else:
                replicas.append([0])
        
        # 生成所有可能的复制组合
        replica_offsets = np.array(np.meshgrid(*replicas)).T.reshape(-1, 3)
        
        # 存储所有位置（原始+镜像）
        self.all_positions = []
        self.original_indices = []  # 记录每个位置对应的原始原子索引
        
        # 原始原子位置
        original_positions = self.atoms.get_positions()
        for i, pos in enumerate(original_positions):
            self.all_positions.append(pos)
            self.original_indices.append(i)
        # 镜像原子位置
        for offset in replica_offsets:
            # 跳过零偏移（原始位置）
            if np.all(offset == 0):
                continue
            # 应用周期性偏移
            offset_positions = original_positions + offset @ self.cell
            
            for i, pos in enumerate(offset_positions):
                self.all_positions.append(pos)
                self.original_indices.append(i)
        
        self.all_positions = np.array(self.all_positions)
        self.original_indices = np.array(self.original_indices)
    def create_grid(self, grid_spacing: float = 0.1, height_above_surface: float = 5.0):
        """
        在表面上创建密集网格
        
        参数:
        grid_spacing: 网格点间距 (Å)
        height_above_surface: 网格在表面上方的初始高度 (Å)
        """
        # 获取表面原子的坐标
        positions = self.atoms.get_positions()
        
        # 确定表面方向
        if self.surface_direction == 0:  # x方向为表面法线
            surface_coords = positions[:, 1:]
            max_height = np.max(positions[:, 0])
        elif self.surface_direction == 1:  # y方向为表面法线
            surface_coords = positions[:, [0, 2]]
            max_height = np.max(positions[:, 1])
        else:  # z方向为表面法线 (默认)
            surface_coords = positions[:, :2]
            max_height = np.max(positions[:, 2])
        
        # 确定网格的边界
        x_min, y_min = np.min(surface_coords, axis=0)
        x_max, y_max = np.max(surface_coords, axis=0)
        
        # 扩展边界以确保覆盖整个表面
        x_min, x_max = x_min - 8.0, x_max + 8.0
        y_min, y_max = y_min - 8.0, y_max + 8.0
        
        # 创建网格点
        x_grid = np.arange(x_min, x_max, grid_spacing)
        y_grid = np.arange(y_min, y_max, grid_spacing)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        # 设置网格高度
        if self.surface_direction == 0:
            zz = np.full_like(xx, max_height + height_above_surface)
            self.grid_points = np.vstack([zz.ravel(), xx.ravel(), yy.ravel()]).T
        elif self.surface_direction == 1:
            zz = np.full_like(xx, max_height + height_above_surface)
            self.grid_points = np.vstack([xx.ravel(), zz.ravel(), yy.ravel()]).T
        else:
            zz = np.full_like(xx, max_height + height_above_surface)
            self.grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
            
        return self.grid_points
    def wrap_grid_to_surface(self, contact_distance: float = 2.0, step_size: float = 0.1,height_above_surface=3.0):
        """
        将网格点向表面平移，直到接近原子
        
        参数:
        contact_distance: 接触距离阈值 (Å)
        step_size: 平移步长 (Å)
        """
        if self.grid_points is None:
            raise ValueError("请先创建网格点")
            
        # 创建原子位置的KD树用于快速最近邻搜索
        atom_positions = self.all_positions
        tree = cKDTree(atom_positions)
        
        # 初始化包裹后的点
        wrapped_points = self.grid_points.copy()
        
        # 确定平移方向
        if self.surface_direction == 0:
            direction = np.array([-1, 0, 0])
        elif self.surface_direction == 1:
            direction = np.array([0, -1, 0])
        else:
            direction = np.array([0, 0, -1])
            
        # 逐步平移网格点
        max_steps = int(height_above_surface / step_size) + 10
        for step in range(max_steps):
            # 计算每个点到最近原子的距离
            distances, indices = tree.query(wrapped_points)
            
            # 找到尚未接触原子的点
            not_contacted = distances > contact_distance
            
            if not np.any(not_contacted):
                break
                
            # 将这些点向表面方向移动
            wrapped_points[not_contacted] += direction * step_size
        
        self.wrapped_points = wrapped_points
        return wrapped_points   
    def find_sites(self, contact_distance: float = 2.0, multi_site_threshold: float = 2):
        """
        识别表面位点
        
        参数:
        contact_distance: 接触距离阈值 (Å)
        multi_site_threshold: 多重位点识别阈值 (Å)
        """
        if self.wrapped_points is None:
            raise ValueError("请先执行网格包裹")
            
        # 创建原子位置的KD树
        atom_positions = self.all_positions
        tree = cKDTree(atom_positions)
        
        # 对于每个包裹后的网格点，找到接触的原子
        for i, point in enumerate(self.wrapped_points):
            # 找到距离此点在一定范围内的所有原子
            indices = tree.query_ball_point(point, contact_distance)
            
            if indices:
                # 将原子索引转换为可哈希的元组
                atom_tuple = tuple(sorted(indices))
                self.site_atoms[atom_tuple].append(point)
        
        # 识别位点类型并计算位点位置
        for atom_indices, points in self.site_atoms.items():
            if len(atom_indices) == 1:
                # 顶位 - 使用原子位置
                atom_idx = atom_indices[0]
                self.site_positions[atom_indices] = atom_positions[atom_idx]
                self.site_special_vectors[atom_indices]=None
            elif len(atom_indices) == 2:
                #桥位
                site_atoms = atom_positions[list(atom_indices)]
                self.site_positions[atom_indices] = np.mean(site_atoms, axis=0)
                self.site_special_vectors[atom_indices]=(site_atoms[-1]-site_atoms[0])/np.linalg.norm(site_atoms[-1]-site_atoms[0])
            else:
                # 桥位或多重位点 - 使用原子位置的平均值
                site_atoms = atom_positions[list(atom_indices)]
                self.site_positions[atom_indices] = np.mean(site_atoms, axis=0)
                self.site_special_vectors[atom_indices]=None
        
        return self.site_atoms, self.site_positions,self.site_special_vectors
    def classify_sites(self, multi_site_threshold: float = 2):
        """
        分类位点类型
        
        参数:
        multi_site_threshold: 多重位点识别阈值 (Å)
        """
        site_types = {}
        
        for atom_indices in self.site_atoms.keys():
            if len(atom_indices) == 1:
                site_types[atom_indices] = "top"
            elif len(atom_indices) == 2:
                site_types[atom_indices] = "bridge"
            else:
                # 检查是否构成多重位点
                atom_positions = self.all_positions[list(atom_indices)]
                centroid = np.mean(atom_positions, axis=0)
                
                # 计算原子到质心的最大距离
                max_distance = np.max(np.linalg.norm(atom_positions - centroid, axis=1))
                
                if max_distance < multi_site_threshold:
                    site_types[atom_indices] = f"{len(atom_indices)}th_multifold"
                else:
                    site_types[atom_indices] = "complex"
        
        return site_types
    def visualize(self, show_grid: bool = False, show_wrapped: bool = True):
        """
        可视化结果
        
        参数:
        show_grid: 是否显示初始网格
        show_wrapped: 是否显示包裹后的网格
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制原子
        positions = self.atoms.get_positions()
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='blue', s=150, label='atom')
        
        # 绘制初始网格点（如果要求）
        if show_grid and self.grid_points is not None:
            ax.scatter(self.grid_points[:, 0], self.grid_points[:, 1], self.grid_points[:, 2],
                      c='gray', s=5, alpha=0.3, label='initial grid')
        
        # 绘制包裹后的网格点（如果要求）
        if show_wrapped and self.wrapped_points is not None:
            ax.scatter(self.wrapped_points[:, 0], self.wrapped_points[:, 1], self.wrapped_points[:, 2],
                      c='lightgreen', s=5, alpha=0.5, label='warpped grid')
        
        # 绘制位点位置
        colors = ['red', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        color_idx = 0
        
        for atom_indices, position in self.site_positions.items():
            site_type = self.classify_sites().get(atom_indices, "未知")
            
            if site_type == "top":
                color = 'red'
                marker = 'o'
                size = 50
                alpha = 1
            elif site_type == "bridge":
                color = 'orange'
                marker = 's'
                size = 40
                alpha = 1
            elif "multifold" in site_type:
                '''color = colors[color_idx % len(colors)]
                color_idx += 1'''
                color = 'cyan'
                marker = 'D'
                size = 30
                alpha = 1
            else:
                color = 'gray'
                marker = 'x'
                size = 20
                alpha = 1
            
            ax.scatter(position[0], position[1], position[2], 
                      c=color, marker=marker, s=size, label=site_type,alpha = alpha)
            
            # 绘制向量（从位点到网格点平均位置）
            '''if atom_indices in self.site_vectors:
                vector = self.site_vectors[atom_indices]
                ax.quiver(position[0], position[1], position[2],
                         vector[0]-position[0], vector[1]-position[1], vector[2]-position[2],
                         color=color, arrow_length_ratio=0.1)'''
        
        # 设置图表属性
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title('sites of surface')
        
        # 避免重复的图例标签
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()
class molfile():
    def __init__(self,name,path_test,slab = None,random = 20):
        [cp,wf,wb,wna,waH] = [[],[],[],[],[]]
        species = f'{path_test}opt/system/species/'
        self.species = species
        self.name = name
        for i in range(1,random+1):
            cb = rR.checkBonds()
            cb.input(f'{species}{name}/{i}/nequipOpt.traj')
            cb.AddAtoms()
            cb.CheckAllBonds()
            bm2s = rR.BuildMol2Smiles(cb)
            bm2s.build()
            try:
                fmax = get_fmax_from_traj(f'{species}{name}/{i}/nequipOpt.traj')
            except Exception:
                fmax=-1
            if fmax > 0.05:
                wf.append(i)
            elif fmax ==-1:
                wf.append(i)
            else:
                if bm2s.smiles == name:
                    if name != '[H]' and name != '[H][H]':
                        ch =[]
                        for a in bm2s.ads:
                            if a.elesymbol == 'H':ch.append(True)
                            else:ch.append(False)
                        if bm2s.ads != []:
                            if all(ch)==False:
                                cp.append(i)
                            else:
                                waH.append(i)
                        elif bm2s.ads == []:
                            wna.append(i)
                    else:
                        if bm2s.ads != []:
                            cp.append(i)
                elif bm2s.smiles != name:
                    wb.append(i)   
        self.first_check =  [cp,wf,wb,wna,waH]
    def site_finder(self,slab):
            self.slab = slab
            finder = SurfaceSiteFinder(slab)
            # 创建网格
            grid_points = finder.create_grid(grid_spacing=0.1, height_above_surface=3.0)
            # 包裹网格到表面
            wrapped_points = finder.wrap_grid_to_surface(contact_distance=2, step_size=0.1,height_above_surface=3.0)
            # 查找位点
            sites, positions,special_vector = finder.find_sites(contact_distance=2.5)
            # 分类位点
            site_types = finder.classify_sites(multi_site_threshold=2)
            self.site = finder
            self.site_types = site_types
            self.site_positions = positions
            self.special_vectors = special_vector
            return self.site
    def check_site(self):
        
        [cp,wf,wb,wna,waH] = self.first_check
        species = self.species
        name = self.name
        if cp == []:
            if len(wna+waH) > len(wb):#？存疑
                E = []
                for  i in wna:
                    last_atoms = read(f'{species}{name}/{i}/nequipOpt.traj',index=-1)
                    final_energy = last_atoms.get_potential_energy()
                    E.append(final_energy)
                min_E = min(E)
                id = E.index(min_E)
                self.model_p = f'{species}{name}/{wna[id]}/nequipOpt.traj'
            else:
                self.model_p = None
        else:
            dict_cp ={}
            for i in cp:
                last_atoms = read(f'{species}{name}/{i}/nequipOpt.traj',index=-1)
                last_atomsNN = rR.NN_system()
                last_atomsNN.RunCheckNN_FindSite(last_atoms,self.site)
                ads_data = last_atomsNN.ads_data
                site_list =[]
                for ads in ads_data:#ads:[nearest, distance,adsA.id,atom_indices,site_type, vector]
                    site_list.append(ads[-2])
                site_count = Counter(site_list)
                if site_count not in dict_cp:
                    dict_cp[site_count] = [i]
                else:
                    dict_cp[site_count].append(i)
            print(dict_cp)


                
                
                

                    

class PREforSearchTS():
    def __init__(self,path_test):
        self.mainfolder = path_test#/work/home/ac877eihwp/renyq/xxx/test/
        self.opt = path_test+'opt/'
        self.neb = path_test+'RDA_S/'
        self.file = {}
        self.slab = read(f'{path_test}opt/slab/Ru_hcp0001.vasp')
    def readDataPath(self):
        print('Start reading data from opt')
        def read_json(jsonfile):
            with open(jsonfile,'r') as j:
                dictionary = json.load(j)
            return dictionary
        self.foldername_json =f'{self.opt}system/folder_name.json'
        folder_dict=read_json(self.foldername_json)
        count_file = 0
        for file in folder_dict:
            mf = molfile(file,self.mainfolder,self.slab)
            mf.site_finder(self.slab)
            mf.check_site()
            self.file[file] = mf
            count_file +=1
            print(f'{file}:{count_file}')
        assert 1!=1
        if count_file != len(folder_dict):
            ValueError  
        else:pass
        print('Finish reading data from opt')        
    def buildmodel(self,reaction_txt):
        slab = self.slab
        mainfolder = self.neb
        os.makedirs(mainfolder, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
        with open(f'{mainfolder}foldername.json', 'w') as file:
            pass
        reaction_list = read_file_line_by_line(reaction_txt)
        for reaction in reaction_list:
            rlist = rR.str2list(reaction)
            initial_mol = self.file[rlist[0][0]]
            final_mol = self.file[rlist[-1][0]]
            def warp(molstr,reactionlist):
                if molstr!='O/OH':
                    if molstr == 'H':
                        return '[H]'
                    elif molstr == 'OH':
                        return '[H]O'
                    else:
                        return molstr
                else:
                    extra = find_uppercase_difference(rlist[-1][0],rlist[0][0])
                    if len(extra)  == 2:
                        return '[H]O'
                    else:
                        return 'O'
            add_mol = self.file[warp(rlist[1][-4],rlist)]
            if initial_mol.model_p == None or final_mol.model_p == None:
                print(f'{reaction} -- IS:{initial_mol.model_p};FS:{final_mol.model_p}')
            else:
                RR = rR.STARTfromBROKENtoBONDED(initial_mol.model_p,add_mol.model_p,final_mol.model_p)
                RR.site_finder(slab)
                RR.run(reaction)
                if RR.IS == False or RR.FS == False:
                    print(f'{reaction} -- IS{RR.IS} or FS{RR.FS} build wrong')
                else:
                    ISNNSYS,FSNNSYS=RR.NNsystemInfo()
                    subfolder = f'{mainfolder}{rlist[0][0]}_{rlist[-1][0]}/'
                    data = {f'{rlist[0][0]}_{rlist[-1][0]}':[reaction,RR.tf,RR.bondatoms,ISNNSYS.bms.smiles,FSNNSYS.bms.smiles]}
                    with open(f'{mainfolder}foldername.json', 'r') as f:
                        file = f.read()
                        if len(file)>0:
                            ne = 'ne'
                        else:
                            ne = 'e'
                    if ne == 'ne':
                        with open (f'{mainfolder}foldername.json','r') as f:
                            old_data = json.load(f)
                    else:
                        old_data ={}
                    old_data.update(data)
                    with open(f'{mainfolder}foldername.json', 'w') as f:
                        json.dump(old_data,f,indent=2)
                    os.makedirs(subfolder, exist_ok=True)
                    RR.save(subfolder,'POSCAR')
        with open(f'{mainfolder}foldername.json', 'r') as f:
            self.d = json.load(f)               
    def start_split(self,batch):
        fd = self.d
        floderlist = list(fd.keys())
        n=len(floderlist)
        r=n%batch
        klow=int(n/batch)
        split = [klow+1 for _ in range(r)] + [klow for _ in range(10-r)]
        foldersplit=[]
        for i in split:
            il=floderlist[0:i]
            del floderlist[:i]
            foldersplit.append(il)
        self.fsp = foldersplit
        return foldersplit



'''class SearchTS4All(PREforSearchTS):
    def __init__(self,path_test):
        self.neb = path_test+'RDA_S/'
        with open(f'{self.neb}foldername.json', 'r') as f:
            self.d = json.load(f)
'''






