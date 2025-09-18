#!/bin/bash
#SBATCH -J cp2k
#SBATCH -N 2
#SBATCH --ntasks-per-node=32
#SBATCH -p test

module rm mpi/hpcx/2.7.4/gcc-7.3.1
module load compiler/intel/2017.5.239
module load mpi/hpcx/2.7.4/intel-2017.5.239
module load apps/cp2k/6.1.0/hpcx-2.7.4-intel2017

export LD_LIBRARY_PATH=/public/software/apps/cp2k/libs/libint-1.1.6/lib:/public/software/apps/cp2k/libs/elpa-2017.05.002/lib:/public/software/apps/cp2k/libs/libxsmm-1.8.3/lib:/public/software/apps/cp2k/libs/libxc-4.0.3/lib:$LD_LIBRARY_PATH

srun --mpi=pmix_v3 cp2k.popt -i H2O-256.inp -o output
