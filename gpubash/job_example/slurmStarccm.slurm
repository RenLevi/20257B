#!/bin/bash
#SBATCH -J STAR
#SBATCH -p cpu
#SBATCH -N 32
#SBATCH --ntasks-per-node=30
#SBATCH -o %j.out
#SBATCH -e %j.err

date

WDIR=`pwd`
cd $WDIR

APP="/public/home/user/app/Siemens/14.02.010-R8/STAR-CCM+14.02.010-R8/star/bin/starccm+ -power -mesa"
INPUT_FILE=$WDIR/test.sim
SCRIPT_FILE=$WDIR/test.java

NP=$SLURM_NPROCS
NNODE=`srun hostname |sort |uniq | wc -l` 
LOG_FILE=$WDIR/job_${NP}c_${NNODE}n_$SLURM_JOB_ID.log 
HOSTFILE=$WDIR/hosts_$SLURM_JOB_ID
srun hostname |sort |uniq -c |awk '{printf "%s:%s\n",$2,$1}' > $HOSTFILE
#$APP $INPUT_FILE -batch $SCRIPT_FILE -machinefile $HOSTFILE -np $NP -rsh ssh  -mpidriver intel  2>&1

export I_MPI_FABRICS=shm:dapl
export I_MPI_DAPL_UD=enable
export I_MPI_FALLBACK_DEVICE=disable
export I_MPI_DEBUG=0
export I_MPI_PIN=disable
export I_MPI_ADJUST_REDUCE=2
export I_MPI_ADJUST_ALLREDUCE=2
export I_MPI_ADJUST_BCAST=1
export I_MPI_PLATFORM=auto
export I_MPI_DAPL_SCALABLE_PROGRESS=1
export I_MPI_DAPL_UD_PROVIDER=ofa-v2-mlx5_0-1u

$APP $INPUT_FILE -batch $SCRIPT_FILE -machinefile $HOSTFILE -np $NP -rsh ssh  -mpidriver intel -fabricverbose 2>&1  


