#!/bin/bash
#SBATCH -J gromacs
#SBATCH -p normal
#SBATCH -N 2
#SBATCH --ntasks-per-node=32

module use  /public/software/modules
module rm   mpi/hpcx/2.7.4/gcc-7.3.1
module load compiler/intel/2017.5.239
module load mpi/hpcx/2.7.4/intel-2017.5.239
module load apps/gromacs/2019.5/hpcx-2.4.1-intel2017

srun --mpi=pmix_v3  gmx_mpi mdrun -nsteps 6000 -ntomp 1 -v -s 2018.tpr -g log/test.log -c log/test.gro -e log/test.edr 
