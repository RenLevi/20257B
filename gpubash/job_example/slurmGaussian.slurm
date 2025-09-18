#!/bin/bash
#SBATCH -J g16                         # 作业名
#SBATCH -N 1                           # 节点数
#SBATCH --ntasks-per-node=32         # 每节点申请的核数
#SBATCH -p kshctest                    # 可用队列

export G16ROOT= #Gaussian install dir
export GAUSS_EXEDIR=${G16ROOT}
export GAUSS_SCRDIR=${G16ROOT}/scratch
export LD_LIBRARY_PATH=${G16ROOT}:$LD_LIBRARY_PATH
export PATH=${GAUSS_EXEDIR}:$PATH


for i in `ls *gjf` 
do
FILE_NAME=`echo $i|awk -F. '{print $1}'`
export GAUSS_SCRDIR=${PWD}/tmp${FILE_NAME}-${SLURM_JOBID}
mkdir $GAUSS_SCRDIR
g16 $i
if [ $? == 0  ];then
rm -rf $GAUSS_SCRDIR
rm -rf /public/home/user/log/${SLURM_JOBID}*
fi
done

