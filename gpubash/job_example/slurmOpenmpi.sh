#!/bin/bash
#SBATCH -J FIRE
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -o %j.loop
#SBATCH -e %j.loop
#SBATCH --comment=WRF


echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"

#module load mpi/openmpi/3.0.0/intel
export PATH=/public/software/mpi/openmpi/3.0.0/intel/bin:$PATH
export LD_LIBRARY_PATH=/public/software/mpi/openmpi/3.0.0/intel/lib:$LD_LIBRARY_PATH
export INCLUDE=/public/software/mpi/openmpi/3.0.0/intel/include:$INCLUDE

date

#export I_MPI_PMI_LIBRARY=/opt/gridview/slurm17/lib/libpmi.so
#export I_MPI_PMI_LIBRARY=/opt/gridview/slurm17/lib/libpmi2.so

time srun --mpi=pmi2 ./fire_openmpi_slurm 960000
date

