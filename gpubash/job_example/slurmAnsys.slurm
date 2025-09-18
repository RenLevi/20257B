#!/bin/bash 
#SBATCH -p test 
#SBATCH -N 2 
#SBATCH -n 64
#SBATCH --ntasks-per-node 32 
#SBATCH -e err 
#SBATCH -o out 
 
WORK_DIR=`pwd`

EXEC=/public/home/user/app/ANSYS/ansys2019/ansys_inc/v190/ansys/bin/ansys190
INPUT_FILE=${WORK_DIR}/test.dat  

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

$EXEC -b -dis -np 64 -mpi ibmmpi -i $INPUT_FILE -o output.log -dir ${WORK_DIR}