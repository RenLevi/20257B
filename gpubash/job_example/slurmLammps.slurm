#!/bin/bash
#SBATCH -J lammps
#SBATCH -N 10
#SBATCH --ntasks-per-node=32
#SBATCH -p test

module use  /public/software/modules
module rm   mpi/hpcx/2.4.1/gcc-7.3.1
module load compiler/intel/2017.5.239
module load mpi/hpcx/2.7.4/intel-2017.5.239
module load apps/lammps/7Aug19/hpcx-2.4.1-intel2017

srun --mpi=pmix_v3 lmp_cpu -i in.sue
