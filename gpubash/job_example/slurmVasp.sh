#/bin/bash
#SBATCH --job-name=VaspTest
#SBATCH --mem-per-cpu=2gb
#SBATCH --ntasks=16
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --output=%j.log
#SBATCH --partition=debug

# load the environment
module purge
source /public/software/profile.d/compiler_intel-compiler-2017.5.239.sh
source /public/software/profile.d/mpi_intelmpi-2017.4.239.sh
export I_MPI_PMI_LIBRARY=/opt/gridview/slurm/lib/libpmi.so
export PATH=/public/software/apps/vasp/5.4.4/hpcx-2.4.1-intel2017:${PATH}

# run vasp on 2 nodes with 16 cores
srun vasp_std

# Output file is ...:         VaspTest.log

