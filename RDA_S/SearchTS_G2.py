import numpy as np
from ase import Atoms
from ase.optimize import BFGS, LBFGS, FIRE
from ase.io import read, write
import matplotlib.pyplot as plt
from ase.constraints import FixAtoms
import copy
import os
from sella import Sella
from nequip.ase import NequIPCalculator
from ase.mep import NEB
from ase.vibrations import Vibrations
import shutil
def get_single_filename(folder_path):
    """获取文件夹中唯一的文件名"""
    all_items = os.listdir(folder_path)
    files = [item for item in all_items if os.path.isfile(os.path.join(folder_path, item))]
    
    if len(files) == 1:
        return files[0]
    elif len(files) == 0:
        return "文件夹为空"
    else:
        return f"文件夹中有 {len(files)} 个文件，不止一个文件"
# 线性插值生成中间结构
def interpolate_structure(
                            initial_model, 
                            final_model, 
                            fraction=0.5, 
                            output_file=None):
    """
    线性插值生成任意比例的中间结构
    
    Parameters:
    - initial_file: 初始结构文件
    - final_file: 最终结构文件
    - fraction: 插值比例 (0=初始结构, 1=最终结构, 0.5=50%中间结构)
    - output_file: 输出文件路径
    """
    # 读取结构
    initial = copy.deepcopy(initial_model)
    final = copy.deepcopy(final_model)
    
    # 验证一致性
    if len(initial) != len(final):
        raise ValueError("原子数量不一致!")
    
    if initial.get_chemical_symbols() != final.get_chemical_symbols():
        raise ValueError("原子种类或顺序不一致!")
    
    # 线性插值
    pos_initial = initial.get_positions()
    pos_final = final.get_positions()
    
    # 计算插值后的坐标
    pos_interpolated = pos_initial + fraction * (pos_final - pos_initial)
    
    # 创建新结构
    new_structure = copy.deepcopy(initial)
    new_structure.set_positions(pos_interpolated)
    
    # 保存
    if output_file:
        write(output_file, new_structure)
        print(f"{fraction*100}% 结构已保存到: {output_file}")
    
    return new_structure
#IDPP插值生成中间结构
def interpolate_structure_from_idpp(
                                    initial_file, final_file, 
                                    n_images=3, 
                                    fraction=0.5,
                                    output_file=None):
    """
    使用 ASE 的 IDPP 方法在初始结构和最终结构之间插值生成中间结构。
    
    参数:
    initial_file: 初始结构文件路径 (POSCAR 格式)
    final_file: 最终结构文件路径 (POSCAR 格式)
    n_images: 总路径上的副本数 (包含初态和末态)
    fmax: IDPP 优化的力收敛标准 (eV/A)
    optimizer: 使用的优化算法 (如 BFGS)
    steps: 最大优化步数
    k: 弹簧常数 (eV/A)
    
    返回:
    images: 包含初态、末态和中间副本的结构列表
    """
    # 1. 读取初始和最终结构
    if type(initial_file) == str:
        initial = read(initial_file)
    else:
        initial = copy.deepcopy(initial_file)
    if type(final_file) == str:
        final = read(final_file)
    else:
        final = copy.deepcopy(final_file)
    # 2. 初始化路径上的结构列表 (初态 + (n_images-2)个中间副本 + 末态)
    images = [initial]
    images += [initial.copy() for i in range(n_images - 2)]
    images += [final]
    # 3. 创建 NEB 对象
    neb = NEB(images, climb=False)   # 这里不需要爬坡，只做路径生成
    neb.interpolate(method='idpp')
    interpolate_structure = images[int((n_images-1)*fraction)]  
    if output_file:
        write(output_file, interpolate_structure)
        print(f"{fraction*100}% 结构已保存到: {output_file}")
    return interpolate_structure, images
# 计算两组结构的欧氏距离
def Euclidean_distance(R1:Atoms, R2:Atoms):
    """
    计算两组结构的欧氏距离
    """
    p1 = R1.get_positions()
    p2 = R2.get_positions()
    assert len(R1) == len(R2)
    return np.sqrt(np.sum((p1 - p2) ** 2))
# 能量阈限的结构优化 
def optimize_with_energy_criterion(model, calculator, energy_threshold=1e-2, 
                                   max_steps=100, trajectory_file=None,log_file=None):
    """
    使用标准优化器但基于能量收敛准则
    
    Parameters:
    - atoms: 要优化的原子结构
    - calculator: 计算器
    - energy_threshold: 能量收敛阈值 (eV)
    - max_steps: 最大优化步数
    - trajectory_file: 轨迹文件路径
    """
    atoms =copy.deepcopy(model)
    atoms.calc = calculator
    optimizer = FIRE(atoms=atoms,trajectory=trajectory_file,logfile=log_file)
    energy_history = []
    def warp(step):
        if step < 10:
            return str(step)+'  '
        elif step>=10 and step <100:
            return str(step)+' '
        else:return str(step)
    for step in range(max_steps):
        # 获取当前能量
        current_energy = atoms.get_potential_energy()
        energy_history.append(current_energy)
        print_out = f"Step {warp(step)}: Energy = {current_energy:.6f} eV"
        #print(f"步骤 {step}: 能量 = {current_energy:.6f} eV")
        # 检查能量收敛
        if len(energy_history) > 1:
            energy_diff = abs(energy_history[-1] - energy_history[-2])
            print_out += f"  Energy change: {energy_diff:.6f} eV"
            print(print_out)
            #print(f"  能量变化: {energy_diff:.6f} eV")
            if energy_diff < energy_threshold:
                #print(f"\n能量收敛于步骤 {step}!")
                #print(f"最终能量变化: {energy_diff:.6f} eV < 阈值 {energy_threshold} eV")
                break
        else:print(print_out)
        # 执行一步优化
        try:
            optimizer.step()
        except StopIteration:
            print("优化器自然收敛")
            break
    
    else:
        print(f"达到最大步数 {max_steps}")
    
    # 绘制能量收敛图
    #plot_energy_convergence(energy_history, energy_threshold)
    
    return atoms, energy_history
# 能量作图（optional)
def plot_energy_convergence(energy_history, threshold):
    """绘制能量收敛图"""
    plt.figure(figsize=(10, 6))
    
    # 计算能量变化
    energy_changes = []
    for i in range(1, len(energy_history)):
        energy_changes.append(abs(energy_history[i] - energy_history[i-1]))
    
    # 绘制能量历史
    plt.subplot(2, 1, 1)
    plt.plot(energy_history, 'b-o', markersize=3)
    plt.ylabel('能量 (eV)')
    plt.title('优化能量历史')
    plt.grid(True)
    
    # 绘制能量变化
    plt.subplot(2, 1, 2)
    plt.semilogy(range(1, len(energy_history)), energy_changes, 'r-o', markersize=3)
    plt.axhline(y=threshold, color='g', linestyle='--', label=f'收敛阈值: {threshold} eV')
    plt.xlabel('优化步骤')
    plt.ylabel('能量变化 (eV)')
    plt.title('能量变化历史 (对数坐标)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('energy_convergence.png', dpi=300)
    plt.show()
#Sigma Function
def sigma(delta_dIS,delta_dFS,Determination = ['IS','FS','Nondirectional']):
    """
    output=[dc,dnc]
    sigma Function:
    - If diff_IS < 0 and diff_FS > 0, sigma = IS
    - If diff_IS > 0 and diff_FS < 0, sigma = FS
    - If diff_IS and diff_FS have the same sign, sigma = Nondirectional
    """
    if delta_dIS < 0 and delta_dFS > 0:
        out = Determination[0]
        print(f"directionality sigma = {out}")
    elif delta_dIS > 0 and delta_dFS < 0:
        out = Determination[1]
        print(f"directionality sigma = {out}")
    elif delta_dIS*delta_dFS > 0:
        out = Determination[2]
        print(f"directionality sigma = {out}")
    else:
        ValueError("Error in sigma function calculation!")
    return out
# D_criteria
def D_criteria(delta_dIS,delta_dFS,limit=0.05):
    """
    计算D_criteria
    """
    if delta_dIS*delta_dFS > 0:
        print("diff_IS和diff_FS符号相同,满足D_criteria")
        return True
    elif np.abs(delta_dIS) < limit and np.abs(delta_dFS) < limit:
        print(f"diff_IS和diff_FS绝对值均小于{limit},满足D_criteria")   
        return True
    else:
        print("不满足D_criteria")
        return False 
def directionality_sigma(ris_rfs,rcopt_r1,Determination= ['IS','FS','Nondirectional']):
    """
    sigma_Ri,diff_IS_Ri,diff_FS_Ri = directionality_sigma(R1Copt,Rref,RiCopt,Ri)
    计算delta_dIS和delta_dFS
    计算方向性
    """
    [ris, rfs] = copy.deepcopy(ris_rfs)
    [rcopt, r1] = copy.deepcopy(rcopt_r1)
    diff_is = Euclidean_distance(rcopt,ris) - Euclidean_distance(r1,ris)
    diff_fs = Euclidean_distance(rcopt,rfs) - Euclidean_distance(r1,rfs)
    direction = sigma(diff_is,diff_fs,Determination)
    return direction, diff_is, diff_fs

class RDA_S():
    def __init__(self,ISfile,FSfile,path):
        self.IS = read(ISfile)
        self.FS = read(FSfile)
        self.path = path
    def initialize_floder(self):
        if not os.path.exists(f'{self.path}/IntermediateProcess'):
            os.makedirs(f'{self.path}/IntermediateProcess')
            os.mkdir(f'{self.path}/IntermediateProcess/step1')
            os.mkdir(f'{self.path}/IntermediateProcess/step2')
            os.mkdir(f'{self.path}/IntermediateProcess/step3')
            os.mkdir(f'{self.path}/IntermediateProcess/results')
            os.mkdir(f'{self.path}/IntermediateProcess/optimized_IS_FS')  
        else:
            def _del_folder(fp):
                folder_path = fp
                try:
                    shutil.rmtree(folder_path)
                    print(f"已删除文件夹及其所有内容: {folder_path}")
                except FileNotFoundError:
                    print("文件夹不存在")
                except PermissionError:
                    print("权限不足，无法删除")
                except OSError as e:
                    print(f"删除失败: {e}")  
            fl = ['step1','step2','step3','results']
            for fn in fl:
                _del_folder(f'{self.path}/IntermediateProcess/{fn}')
                os.mkdir(f'{self.path}/IntermediateProcess/{fn}')
    def skip_opt(self):
        self.optIS = self.IS
        self.optFS = self.FS
        return self.optIS, self.optFS
    def opt(self,mlp_path):
        def _check_file_exists(folder_path, filename):
            """
            检查指定文件夹中是否存在某个文件
            """
            file_path = os.path.join(folder_path, filename)
            return os.path.isfile(file_path)
        if _check_file_exists(f'{self.path}/IntermediateProcess/optimized_IS_FS', 'IS_opt.vasp') and _check_file_exists(f'{self.path}/IntermediateProcess/optimized_IS_FS', 'FS_opt.vasp'):
            print("检测到已优化的IS和FS，直接读取")
            self.optIS = read(f'{self.path}/IntermediateProcess/optimized_IS_FS/IS_opt.vasp')
            self.optFS = read(f'{self.path}/IntermediateProcess/optimized_IS_FS/FS_opt.vasp')
            return self.optIS, self.optFS
        else:
            calc=NequIPCalculator.from_deployed_model(mlp_path, device='cpu')
            print("未检测到已优化的IS和FS，开始优化")
            atoms1 = copy.deepcopy(self.IS)
            atoms2 = copy.deepcopy(self.FS)
            atoms1.calc = calc
            print('Optimizing IS...')
            BFGS(atoms1).run(fmax=0.01)
            write(f'{self.path}/IntermediateProcess/optimized_IS_FS/IS_opt.vasp', atoms1)
            atoms2.calc = calc
            print('Optimizing FS...')
            BFGS(atoms2).run(fmax=0.01)
            write(f'{self.path}/IntermediateProcess/optimized_IS_FS/FS_opt.vasp', atoms2)
            print('IS and FS optimization done!')
            self.optIS = atoms1
            self.optFS = atoms2
            return atoms1, atoms2
    def run_idpp(self, mlp_path,fix_indices=list(range(32))):
        # 线性插值生成中间结构
        RIS = self.optIS
        RFS = self.optFS
        path_IP = f'{self.path}/IntermediateProcess/'
        print("Step 1:生成中间结构R1   <R_alpha>")
        R1,_ = interpolate_structure_from_idpp(initial_file=RIS,final_file=RFS,output_file=f'{path_IP}step1/R1.vasp')
        # 设置计算器
        calc = NequIPCalculator.from_deployed_model(mlp_path, device='cpu')
        # 使用能量收敛准则优化中间结构
        print("条件优化R1")
        R1Copt, _ = optimize_with_energy_criterion(R1, calc, 
                                                    energy_threshold=0.01, 
                                                    max_steps=200, 
                                                    #trajectory_file= f'{path_IP}step1/R1Copt.traj',
                                                    #log_file=  f'{path_IP}step1/R1Copt.log'
                                                    )
        sigma_R1,diff_IS,diff_FS = directionality_sigma([RIS,RFS],[R1Copt,R1])
        write(f'{path_IP}step1/R1Copt.xyz', R1Copt)
        print(f"R1Copt的diff_IS: {diff_IS}; diff_FS: {diff_FS}")
        if D_criteria(diff_IS,diff_FS) == True:
            print("R1Copt满足D_criteria")
            print('Start Sella')
            QTS = copy.deepcopy(R1Copt)
            QTS.calc = calc
            Sella_Search = Sella(QTS, 
                                 logfile=f'{path_IP}results/R1Copt_sella.log', 
                                 trajectory=f'{path_IP}results/R1Copt_sella.traj',
                                 delta0=0.1,          # 更小的初始步长，谨慎起步
                                 #rho_inc=1.1,            # 提高放大阈值，更难放大步长
                                 #sigma_inc=1.05,         # 降低放大倍数，放也要小步放
                                 #rho_dec=3.0,            # 降低“不准”的阈值，对误差更敏感
                                 #sigma_dec=0.5,          # 加大缩减力度，出错时缩得更狠
                                 )
            for i, geom in enumerate(Sella_Search.irun(fmax=0.05)):
                _, fmax, cmax = Sella_Search.pes.converged(Sella_Search.fmax)
                if fmax < 0.1 and Sella_Search.delta > 5e-3:
                    Sella_Search.delta = 5e-3
            write(f'{path_IP}results/TS.xyz', QTS)
            print("过渡态搜索完成,结果保存在TS.xyz")
            return QTS
        else:
            print("R1Copt不满足D_criteria,继续调整IS和FS")
            print("Step 2:生成中间结构R2   <R_beta>")
            if sigma_R1 == 'IS':
                print("将RFS作为Rref")
                Rref = copy.deepcopy(RFS)
            elif sigma_R1 == 'FS':
                print("将RIS作为Rref")
                Rref = copy.deepcopy(RIS)
            else:
                raise ValueError("sigma_check==Nondirectional,程序终止!")
            write(f'{path_IP}step2/Rref.xyz', Rref)
            R2,_ = interpolate_structure_from_idpp(initial_file=R1Copt,final_file=Rref, output_file=f'{path_IP}step2/R2_50.vasp')
            print("条件优化R2")
            R2Copt, _ = optimize_with_energy_criterion(R2, calc, 
                                                        energy_threshold=0.05,
                                                        max_steps=200,
                                                        #trajectory_file= f'{path_IP}step2/R2_50_Copt.traj',
                                                        #log_file=  f'{path_IP}step2/R2_50_Copt.log'
                                                        )
            write(f'{path_IP}step2/R2_50_Copt.xyz', R2Copt)
            sigma_R2,diff_IS_R2,diff_FS_R2 = directionality_sigma([R1Copt,Rref],[R2Copt,R2],Determination=['alpha','ref','Nondirectional'])
            print(f"R2Copt的diff_IS: {diff_IS_R2}; diff_FS: {diff_FS_R2}")
            beta = 0.5
            BETAlist = []
            if sigma_R2 == 'ref' :
                BETAlist.append(beta)
                sigma_Ri = copy.deepcopy(sigma_R2)
                while sigma_Ri == 'ref' or sigma_Ri == 'Nondirectional':
                    beta = beta - 0.1
                    Ri,_ = interpolate_structure_from_idpp(initial_file=R1Copt,final_file=Rref,n_images=11,fraction=beta,output_file=f'{path_IP}/step2/R2_{int(beta*100)}.vasp')
                    print(f"当前beta: {beta}")
                    RiCopt, _ = optimize_with_energy_criterion(Ri, calc, 
                                                                energy_threshold=0.05,
                                                                max_steps=200,
                                                                #trajectory_file= f'{path_IP}/step2/R2_{int(beta*100)}_Copt.traj',
                                                                #log_file=  f'{path_IP}/step2/R2_{int(beta*100)}_Copt.log'
                                                                )
                    write(f'{path_IP}/step2/R2_{int(beta*100)}_Copt.xyz', RiCopt)
                    sigma_Ri,diff_IS_Ri,diff_FS_Ri = directionality_sigma([R1Copt,Rref],[RiCopt,Ri],Determination=['alpha','ref','Nondirectional'])
                    print(f"RiCopt的diff_IS: {diff_IS_Ri}; diff_FS: {diff_FS_Ri}, sigma_Ri: {sigma_Ri}")
                    BETAlist.append(beta)
                    assert beta >= 0 , "beta < 0 ,程序终止!"
                #Rdc_beta =  copy.deepcopy(BETAlist[-1])
                Rdnc = copy.deepcopy(BETAlist[-2])
            elif sigma_R2 == 'alpha':
                sigma_Ri = copy.deepcopy(sigma_R2)
                BETAlist.append(beta)
                while sigma_Ri == 'alpha'or sigma_Ri == 'Nondirectional':
                    beta = beta + 0.1
                    Ri,_ = interpolate_structure_from_idpp(initial_file=R1Copt,final_file=Rref,n_images=11,fraction=beta,output_file=f'{path_IP}/step2/R2_{int(beta*100)}.vasp')
                    print(f"当前beta: {beta}")
                    RiCopt, _ = optimize_with_energy_criterion(Ri, calc, 
                                                                energy_threshold=0.05,
                                                                max_steps=100,
                                                                #trajectory_file= f'{path_IP}/step2/R2_{int(beta*100)}_Copt.traj',
                                                                #log_file=  f'{path_IP}/step2/R2_{int(beta*100)}_Copt.log'
                                                                )
                    write(f'{path_IP}/step2/R2_{int(beta*100)}_Copt.xyz', RiCopt)
                    sigma_Ri,diff_IS_Ri,diff_FS_Ri = directionality_sigma([R1Copt,Rref],[RiCopt,Ri],Determination=['alpha','ref','Nondirectional'])
                    print(f"RiCopt的diff_IS: {diff_IS_Ri}; diff_FS: {diff_FS_Ri}, sigma_Ri: {sigma_Ri}")
                    BETAlist.append(beta)
                    assert  beta <= 1, "beta > 1,程序终止!"
                #Rdc_beta =  copy.deepcopy(BETAlist[-2])
                Rdnc = copy.deepcopy(BETAlist[-1])
            else:
                print("R2Copt满足D_criteria")
                print('Start Sella')
                QTS = copy.deepcopy(R2Copt)
                QTS.calc = calc
                Sella_Search = Sella(QTS, 
                                    logfile=f'{path_IP}results/R2Copt_sella.log', 
                                    trajectory=f'{path_IP}results/R2Copt_sella.traj',
                                    delta0=0.1,          # 更小的初始步长，谨慎起步
                                    #rho_inc=1.1,            # 提高放大阈值，更难放大步长
                                    #sigma_inc=1.05,         # 降低放大倍数，放也要小步放
                                    #rho_dec=3.0,            # 降低“不准”的阈值，对误差更敏感
                                    #sigma_dec=0.5,          # 加大缩减力度，出错时缩得更狠
                                    )
                for i, geom in enumerate(Sella_Search.irun(fmax=0.05)):
                    _, fmax, cmax = Sella_Search.pes.converged(Sella_Search.fmax)
                    if fmax < 0.1 and Sella_Search.delta > 5e-3:
                        Sella_Search.delta = 5e-3
                write(f'{path_IP}results/TS.xyz', QTS)
                print("过渡态搜索完成,结果保存在TS.xyz")
                return QTS
            print("Step 3:生成初猜过渡态R3   <R_gamma>")
            print(f"Rdnc_beta: {Rdnc}")
            #Rdc = interpolate_structure(R1Copt,Rref,fraction=Rdc_beta, output_file=f'{path_IP}step2/Rdc.vasp')
            Rdnc,_ = interpolate_structure_from_idpp(initial_file=R1Copt,final_file=Rref,n_images=11,fraction=Rdnc, output_file=f'{path_IP}step3/Rdnc.vasp')
            RdncCopt, _ = optimize_with_energy_criterion(Rdnc, calc, 
                                                            energy_threshold=0.05,
                                                            max_steps=100,
                                                            #trajectory_file= f'{path_IP}step3/RdncCopt.traj',
                                                            #log_file=  f'{path_IP}step3/RdncCopt.log'
                                                            )
            write(f'{path_IP}step3/RdncCopt.xyz',RdncCopt)        
            if sigma_R1 == 'IS':
                print("将RIS作为Rref")
                Rref = copy.deepcopy(RIS)
            elif sigma_R1 == 'FS':
                print("将RFS作为Rref")
                Rref = copy.deepcopy(RFS)
            else:
                raise ValueError("sigma_check==Nondirectional,程序终止!")
            write(f'{path_IP}step3/Rref.vasp', Rref)
            gamma = 0.1
            R3,_ = interpolate_structure_from_idpp(initial_file=RdncCopt,final_file=Rref,n_images=11,fraction=gamma,output_file=f'{path_IP}step3/R3_{int(gamma*100)}.vasp')
            Rgamma = copy.deepcopy(R3)
            while Euclidean_distance(Rgamma,Rref) >= Euclidean_distance(Rdnc,Rref):
                gamma = gamma + 0.1
                Rgamma,_ = interpolate_structure_from_idpp(initial_file=RdncCopt,final_file=Rref,n_images=11,fraction=gamma,output_file=f'{path_IP}step3/R3_{int(gamma*100)}.vasp')
                assert gamma < 1, "gamma  > 1,程序终止!"
            print(f"gamma: {gamma}")
            write(f'{path_IP}step3/R3final.vasp', Rgamma)
            print("过渡态搜索开始")
            QTS = copy.deepcopy(Rgamma)
            QTS.calc = calc
            Sella_Search = Sella(QTS, 
                                 logfile=f'{path_IP}results/R3_sella.log', 
                                 trajectory=f'{path_IP}results/R3_sella.traj',
                                 delta0=0.1,          # 更小的初始步长，谨慎起步
                                 #rho_inc=1.1,            # 提高放大阈值，更难放大步长
                                 #sigma_inc=1.05,         # 降低放大倍数，放也要小步放
                                 #rho_dec=3.0,            # 降低“不准”的阈值，对误差更敏感
                                 #sigma_dec=0.5,          # 加大缩减力度，出错时缩得更狠
                                 )
            for i, geom in enumerate(Sella_Search.irun(fmax=0.05)):
                _, fmax, cmax = Sella_Search.pes.converged(Sella_Search.fmax)
                if fmax < 0.1 and Sella_Search.delta > 5e-3:
                    Sella_Search.delta = 5e-3
            write(f'{path_IP}results/TS.xyz', QTS)
            print("过渡态搜索完成,结果保存在TS.xyz")
            return QTS
    def control_rtrust_by_fmax(opt,control_delta=5e-3):
        for i, geom in enumerate(opt.irun(fmax=0.05)):
                _, fmax, cmax = opt.pes.converged(opt.fmax)
                if fmax < 0.1 and opt.delta > control_delta:
                    opt.delta = control_delta