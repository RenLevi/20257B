#!/bin/bash
#SBATCH -J MLP_opt       # 作业名
#SBATCH -p ihicnormal           # 队列名
#SBATCH -N 1                    # 节点数
#SBATCH --ntasks-per-node=28    # 每节点进程数
#SBATCH --cpus-per-task=1       # 每进程占用核心数
#SBATCH -o %j.out               # 标准输出文件
#SBATCH -e %j.err               # 错误输出文件

source ~/.bashrc
conda activate MLPcal
python preTS.py | tee -a resLog.out
echo "All jobs completed"

    
