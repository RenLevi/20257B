#!/bin/bash
#SBATCH -J ne
#SBATCH -p dzagnormal           # GPU 分区
#SBATCH -N 1                    # 2 个节点
#SBATCH --ntasks-per-node=1     # 每节点 4 个任务
#SBATCH --cpus-per-task=8       # 每任务 8 个 CPU 线程
#SBATCH --gres=gpu:1            # 每节点 4 块 GPU
#SBATCH -t 20-00:00:00         # 最长运行时间
#SBATCH -o std.out.%j
#SBATCH -e std.err.%j

module load nvidia/cuda/11.6
# 1) 正确初始化 Conda（启用 conda activate）
source /work/home/ac877eihwp/Xingai/miniconda3/etc/profile.d/conda.sh   # :contentReference[oaicite:3]{index=3}



conda activate /work/home/ac877eihwp/Xingai/conda_envs/nequip


LD_LIBRARY_PATH=/work/home/ac877eihwp/Xingai/miniconda3/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH


export WANDB_MODE=offline      # 离线模式

export CUDA_VISIBLE_DEVICES=0,1   # 使用4个GPU

# 4) 启动训练
srun nequip-train nequipFull.yaml
