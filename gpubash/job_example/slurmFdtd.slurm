#!/bin/bash
#SBATCH -J test
#SBATCH -p test
#SBATCH -N 2
#SBATCH -n 60
#SBATCH --ntasks-per-node=30
#SBATCH -o %j.out
#SBATCH -e %j.err

WDIR=`pwd`
cd $WDIR

APP="/public/home/user/app/lumerical/fdtd/mpich2/nemesis/bin/mpiexec"
MY_PROG="/public/home/user/app/lumerical/fdtd/bin/fdtd-engine-mpich2nem"
INPUT_FILE=$WDIR/paralleltest.fsp

NP=$SLURM_NPROCS
echo "Starting run at: `date`"
echo "Running on $NP processors."

$APP -n $NP $MY_PROG -t 1 ${INPUT_FILE} 2>&1

echo "Job finished at: `date`"
exit
