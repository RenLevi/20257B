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
import json

def check_file_exists(folder_path, filename):
    """
    检查指定文件夹中是否存在某个文件
    """
    file_path = os.path.join(folder_path, filename)
    return os.path.isfile(file_path)
# 线性插值生成中间结构
def interpolate_structure(initial_model, final_model, fraction=0.5, output_file=None):
    """
    线性插值生成任意比例的中间结构
    
    Parameters:
    - initial_file: 初始结构文件
    - final_file: 最终结构文件
    - fraction: 插值比例 (0=初始结构, 1=最终结构, 0.5=50%中间结构)
    - output_file: 输出文件路径
    """
    # 读取结构
    initial = initial_model
    final = final_model
    
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
                                   max_steps=100, trajectory_file='optimization.traj'):
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
    # 设置计算器
    atoms.calc = calculator
    # 初始化优化器
    optimizer = BFGS(atoms, 
                     trajectory=trajectory_file, 
                     #logfile='optimization.log'
                     )
    # 记录能量历史
    energy_history = []
    # 手动执行优化步骤
    for step in range(max_steps):
        # 获取当前能量
        current_energy = atoms.get_potential_energy()
        energy_history.append(current_energy)
        print(f"步骤 {step}: 能量 = {current_energy:.6f} eV")
        # 检查能量收敛
        if len(energy_history) > 1:
            energy_diff = abs(energy_history[-1] - energy_history[-2])
            print(f"  能量变化: {energy_diff:.6f} eV")
            if energy_diff < energy_threshold:
                print(f"\n能量收敛于步骤 {step}!")
                print(f"最终能量变化: {energy_diff:.6f} eV < 阈值 {energy_threshold} eV")
                break
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
#计算方向性
def sigmaCopt(delta_dIS,delta_dFS):
    """
    计算sigmaCopt
    """
    if delta_dIS < 0 and delta_dFS > 0:
        out = 'IS'
        print("directionality sigma = IS")
    elif delta_dIS > 0 and delta_dFS < 0:
        out = 'FS'
        print("directionality sigma = FS")
    elif delta_dIS*delta_dFS > 0:
        out = 'Nondirectional'
        print("directionality sigma = Nondirectional")
    else:
        ValueError("Error in sigmaCopt calculation!")
    return out
def directionality_sigma(ris,rfs,rcopt,r1):
    """
    计算delta_dIS和delta_dFS
    计算方向性
    """
    diff_is = Euclidean_distance(rcopt,ris) - Euclidean_distance(r1,ris)
    diff_fs = Euclidean_distance(rcopt,rfs) - Euclidean_distance(r1,rfs)
    output = sigmaCopt(diff_is,diff_fs)
    return output, diff_is, diff_fs
# D_criteria
def D_criteria(delta_dIS,delta_dFS):
    """
    计算D_criteria
    """
    if delta_dIS*delta_dFS > 0:
        print("diff_IS和diff_FS符号相同,满足D_criteria")
        return True
    elif np.abs(delta_dIS) < 0.05 and np.abs(delta_dFS) < 0.05:
        print("diff_IS和diff_FS绝对值均小于0.05,满足D_criteria")   
        return True
    else:
        print("不满足D_criteria")
        return False 
class RDA_S():
    def __init__(self,ISfile,FSfile,path):
        self.IS = read(ISfile)
        self.FS = read(FSfile)
        self.path = path
        if not os.path.exists(f'{self.path}/IntermediateProcess'):
            os.makedirs(f'{self.path}/IntermediateProcess')
            os.mkdir(f'{self.path}/IntermediateProcess/step1')
            os.mkdir(f'{self.path}/IntermediateProcess/step2')
            os.mkdir(f'{self.path}/IntermediateProcess/step3')
            os.mkdir(f'{self.path}/IntermediateProcess/results')
            os.mkdir(f'{self.path}/IntermediateProcess/optimized_IS_FS')    
    def opt(self,calculator):
        if check_file_exists(f'{self.path}/IntermediateProcess/optimized_IS_FS', 'IS_opt.vasp') and check_file_exists(f'{self.path}/IntermediateProcess/optimized_IS_FS', 'FS_opt.vasp'):
            print("检测到已优化的IS和FS，直接读取")
            self.optIS = read(f'{self.path}/IntermediateProcess/optimized_IS_FS/IS_opt.vasp')
            self.optFS = read(f'{self.path}/IntermediateProcess/optimized_IS_FS/FS_opt.vasp')
            return self.optIS, self.optFS
        else:
            print("未检测到已优化的IS和FS，开始优化")
            atoms1 = copy.deepcopy(self.IS)
            atoms2 = copy.deepcopy(self.FS)
            atoms1.calc = calculator
            print('Optimizing IS...')
            BFGS(atoms1).run(fmax=0.01)
            write(f'{self.path}/IntermediateProcess/optimized_IS_FS/IS_opt.vasp', atoms1)
            atoms2.calc = calculator
            print('Optimizing FS...')
            BFGS(atoms2).run(fmax=0.01)
            write(f'{self.path}/IntermediateProcess/optimized_IS_FS/FS_opt.vasp', atoms2)
            print('IS and FS optimization done!')
            self.optIS = atoms1
            self.optFS = atoms2
            return atoms1, atoms2
    def run(self, calculator):
        # 线性插值生成中间结构
        RIS = self.optIS
        RFS = self.optFS
        path_IP = f'{self.path}/IntermediateProcess/'
        print('-'*50)
        print("Step 1:生成中间结构R1   <R_alpha>")
        R1 = interpolate_structure(RIS,RFS, fraction=0.5, output_file=f'{path_IP}step1/R_alpha.vasp')
        # 设置计算器
        calc = calculator
        # 使用能量收敛准则优化中间结构
        print("条件优化R1")
        R1Copt, alpha_energy_history = optimize_with_energy_criterion(R1, calc, 
                                                                        energy_threshold=0.01, 
                                                                        max_steps=100, 
                                                                        trajectory_file=f'{path_IP}step1/R1Copt.traj')
        sigma_R1,diff_IS,diff_FS = directionality_sigma(RIS,RFS,R1Copt,R1)
        print(f"R1Copt的diff_IS: {diff_IS}; diff_FS: {diff_FS}")
        if D_criteria(diff_IS,diff_FS) == True:
            print("R1Copt满足D_criteria")
            print('Start Sella')
            write(f'{path_IP}step1/R1Copt.vasp', R1Copt)
            QTS = copy.deepcopy(R1Copt)
            Sella_Search = Sella(QTS, 
                                 logfile=f'{path_IP}step1/R1Copt_sella.log', 
                                 trajectory=f'{path_IP}step1/R1Copt_sella.traj')
            Sella_Search.run(fmax=0.05)
            write(f'{path_IP}/result/TS_RDA_D_combine_Sella.vasp', QTS)
            print("过渡态搜索完成,结果保存在TS_RDA_D_combine_Sella.vasp")
            return R1Copt,QTS
        else:
            print("R1Copt不满足D_criteria,继续调整IS和FS")
            print('-'*50)
            print("Step 2:生成中间结构R2   <R_beta>")
            if sigma_R1 == 'IS':
                print("将RFS作为Rref")
                Rref = copy.deepcopy(RFS)
            elif sigma_R1 == 'FS':
                print("将RIS作为Rref")
                Rref = copy.deepcopy(RIS)
            else:
                raise ValueError("sigma_check==Nondirectional,程序终止!")
            write(f'{path_IP}step2/Rref.vasp', Rref)
            R2 = interpolate_structure(R1Copt,Rref,fraction=0.5, output_file=f'{path_IP}step2/R_beta.vasp')
            print("条件优化R2")
            R2Copt, beta_energy_history = optimize_with_energy_criterion(R2, calc, 
                                                                         energy_threshold=0.05,
                                                                         max_steps=100,
                                                                         trajectory_file=f'{path_IP}step2/R2Copt.traj')
            write(f'{path_IP}step2/R2Copt.vasp', R2Copt)
            sigma_R2,diff_IS_R2,diff_FS_R2 = directionality_sigma(R1Copt,Rref,R2Copt,R2)
            print(f"R2Copt的diff_IS: {diff_IS_R2}; diff_FS: {diff_FS_R2}")
            beta = 0.5
            BETAlist = []
            if sigma_R2 != sigma_R1 :
                BETAlist.append(beta)
                sigma_Ri = copy.deepcopy(sigma_R2)
                while sigma_Ri != sigma_R1:
                    beta = beta - 0.1
                    assert beta >= 0 , "beta < 0 ,程序终止!"
                    Ri = interpolate_structure(R1Copt,Rref,fraction=beta)
                    print(f"当前beta: {beta}")
                    RiCopt, energy_history = optimize_with_energy_criterion(Ri, calc, 
                                                                            energy_threshold=0.05,
                                                                            max_steps=100,
                                                                            trajectory_file=None)
                    write(f'{path_IP}/step2/Ri_beta_{beta*10}.vasp', RiCopt)
                    sigma_Ri,diff_IS_Ri,diff_FS_Ri = directionality_sigma(R1Copt,Rref,RiCopt,Ri)
                    print(f"RiCopt的diff_IS: {diff_IS_Ri}; diff_FS: {diff_FS_Ri}")
                    BETAlist.append(beta)
                Rdc_beta =  copy.deepcopy(BETAlist[-1])
                Rdnc_beta = copy.deepcopy(BETAlist[-2])
            else:
                sigma_Ri = copy.deepcopy(sigma_R2)
                BETAlist.append(beta)
                while sigma_Ri == sigma_R1:
                    beta = beta + 0.1
                    assert  beta <= 1, "beta > 1,程序终止!"
                    Ri = interpolate_structure(R1Copt,Rref,fraction=beta)
                    print(f"当前beta: {beta}")
                    RiCopt, energy_history = optimize_with_energy_criterion(Ri, calc, 
                                                                            energy_threshold=0.05,
                                                                            max_steps=100,
                                                                            trajectory_file=None)
                    write(f'{path_IP}/step2/Ri_beta_{beta*10}.vasp', RiCopt)
                    sigma_Ri,diff_IS_Ri,diff_FS_Ri = directionality_sigma(R1Copt,Rref,RiCopt,Ri)
                    print(f"RiCopt的diff_IS: {diff_IS_Ri}; diff_FS: {diff_FS_Ri}")
                    BETAlist.append(beta)
                Rdc_beta =  copy.deepcopy(BETAlist[-2])
                Rdnc_beta = copy.deepcopy(BETAlist[-1])
            print('-'*50)
            print("Step 3:生成初猜过渡态R3   <R_gamma>")
            print(f"Rdc_beta: {Rdc_beta}, Rdnc_beta: {Rdnc_beta}")
            Rdc = interpolate_structure(R1Copt,Rref,fraction=Rdc_beta, output_file=f'{path_IP}step2/Rdc.vasp')
            Rdnc = interpolate_structure(R1Copt,Rref,fraction=Rdnc_beta, output_file=f'{path_IP}step2/Rdnc.vasp')
            RdncCopt, energy_history = optimize_with_energy_criterion(Rdc, calc, 
                                                                    energy_threshold=0.05,
                                                                    max_steps=100,
                                                                    trajectory_file=f'{path_IP}step3/RdcCopt.traj')
            sigma_R3,diffis3,difffs3 = directionality_sigma(RIS,RFS,RdncCopt,Rdnc)
            if sigma_R3 == 'IS':
                print("将RFS作为Rref")
                Rref = copy.deepcopy(RFS)
            elif sigma_R3 == 'FS':
                print("将RIS作为Rref")
                Rref = copy.deepcopy(RIS)
            else:
                raise ValueError("sigma_check==Nondirectional,程序终止!")
            write(f'{path_IP}step3/Rref.vasp', Rref)
            gamma = 0.1
            R3 = interpolate_structure(RdncCopt,Rref,fraction=0.1,output_file=f'{path_IP}step3/R_gamma_1.vasp')
            Rgamma = copy.deepcopy(R3)
            while Euclidean_distance(Rgamma,Rref) >= Euclidean_distance(Rdnc,Rref):
                gamma = gamma + 0.1
                Rgamma = interpolate_structure(RdncCopt,Rref,fraction=gamma,output_file=f'{path_IP}step3/R_gamma_{int(gamma*10)}.vasp')
                assert gamma <= 1, "gamma  > 1,程序终止!"
            print(f"gamma: {gamma}")
            write(f'{path_IP}step3/R_gamma.vasp', Rgamma)
            print("过渡态搜索开始")
            QTS = copy.deepcopy(Rgamma)
            Sella_Search = Sella(QTS, 
                                 logfile=f'{path_IP}step3/Rgamma_sella.log', 
                                 trajectory=f'{path_IP}step3/Rgamma_sella.traj')
            Sella_Search.run(fmax=0.05)
            write(f'{path_IP}results/TS_RDA_S_RdncCopt.vasp', QTS)
            '''------------------------------------------'''
            R4 = interpolate_structure(Rdnc,Rref,fraction=0.1,output_file=f'{path_IP}step3/R_gamma_1_Rdnc.vasp')
            gamma = 0.1
            Rgamma = copy.deepcopy(R4)
            while Euclidean_distance(Rgamma,Rref) >= Euclidean_distance(Rdnc,Rref):
                gamma = gamma + 0.1
                Rgamma = interpolate_structure(Rdnc,Rref,fraction=gamma,output_file=f'{path_IP}step3/R_gamma_{int(gamma*10)}_Rdnc.vasp')
                assert gamma <= 1, "gamma  > 1,程序终止!"
            print(f"gamma: {gamma}")
            write(f'{path_IP}step3/R_gamma_Rdnc.vasp', Rgamma)
            print("过渡态搜索开始")
            QTS = copy.deepcopy(Rgamma)
            Sella_Search = Sella(QTS, 
                                 logfile=f'{path_IP}step3/Rgamma_sella_Rdnc.log', 
                                 trajectory=f'{path_IP}step3/Rgamma_sella_Rdnc.traj')
            Sella_Search.run(fmax=0.05)
            write(f'{path_IP}results/TS_RDA_S_Rdnc.vasp', QTS)
            '''------------------------------------------'''
            print("过渡态搜索完成,结果保存在TS_RDA_S_****.vasp")
            #return Rgamma,QTS

'''------------------------------------------------------'''

model_path = '/work/home/ac877eihwp/renyq/prototypeModel.pth'
calc = NequIPCalculator.from_deployed_model(model_path, device='cpu')
with open('config.json','r') as j:
    data = json.load(j)
path = data['path']
folderpath=data['folderpath']
for name in folderpath:
    p0 = f'{path}/{name}'
    with open (f'{path}/foldername.json','r') as j:
        datadict4check = json.load(j)
    answerlist = datadict4check[name]#File name ：[Reaction,bond changed atoms(bid,eid),mainBodyIdx,subBodyIdx,bonded smiles,broken smiles]
    print(answerlist[0])#
    if os.path.exists(f'{p0}/IntermediateProcess'):
        SearchTS = RDA_S(ISfile=f'{path}/{name}/IntermediateProcess/optimized_IS_FS/IS_opt.vasp', FSfile=f'{path}/{name}/IntermediateProcess/optimized_IS_FS/FS_opt.vasp', path=p0)
        SearchTS.opt(calc)
        SearchTS.run(calc)
    else:
        pass
'''------------------------------------------------------'''

'''if (__name__ == "__main__"):
    model_path = '/work/home/ac877eihwp/renyq/LUNIX_all/mlp_opt/prototypeModel.pth'
    calculator = NequIPCalculator.from_deployed_model(model_path, device='cpu')
    p0 = '/work/home/ac877eihwp/renyq/20250828TT/[H]CO_[H]CO[H]'
    SearchTS = RDA_S(ISfile='IS.vasp', FSfile='FS.vasp', path=p0)
    SearchTS.opt(calculator)
    SearchTS.run(calculator)
    model_path = '/public/home/ac877eihwp/haoyl/potential/change/new.pth'
    calculator = NequIPCalculator.from_deployed_model(model_path, device='cpu')
    p0 = '/public/home/ac877eihwp/haoyl/yanjiusheng/pianduan2quanbu/2+H/2newmethod'
    SearchTS = RDA_D(ISfile='IS_opt.vasp', FSfile='FS_opt.vasp', path=p0)
    SearchTS.opt(calculator)
    SearchTS.run(calculator)'''

            


            
            
            
                


        
    

    