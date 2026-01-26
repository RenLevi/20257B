#!/bin/bash
#SBATCH -J MLP_RDA_S	             #作业名
#SBATCH -p ihicnormal            #队列名
#SBATCH -N 1                     #节点数
#SBATCH --ntasks-per-node=28     #每节点进程数
#SBATCH --cpus-per-task=1        #每进程占用核心数
##SBATCH --exclusive             ##独占节点，按节点计费
#SBATCH -o %j.out
#SBATCH -e %j.err

source ~/.bashrc
conda activate MLPcal

echo "Start searching TS" | tee -a resLog.out
python SearchTS.py | tee -a resLog.out
echo "" | tee -a resLog.out
echo "Evaluation finished"
echo "See resLog.out to check the results"
