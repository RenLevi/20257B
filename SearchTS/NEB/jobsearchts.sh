#!/bin/bash
#SBATCH -J MLP_neb_sella         #作业名
#SBATCH -p wzhcnormal            #队列名
#SBATCH -N 1                     #节点数
#SBATCH --ntasks-per-node=28     #每节点进程数
#SBATCH --cpus-per-task=1        #每进程占用核心数
##SBATCH --exclusive             ##独占节点，按节点计费
#SBATCH -o %j.out
#SBATCH -e %j.err

source ~/.bashrc
conda activate op

echo "Starting NEB_Sella to search TS" | tee -a resLog.out
python neb_searchTS.py | tee -a resLog.out
echo "" | tee -a resLog.out
echo "Evaluation finished"
echo "See resLog.out to check the results"


