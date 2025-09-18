#!/bin/bash
#SBATCH --job-name=PythonTest
#SBATCH --mem-per-cpu=2gb
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=%j.log
#SBATCH --partition=debug

# load the environment
module purge
module load apps/python/3.6.1

# run python
python --version

# Output file is ...:         PythonTest.log

