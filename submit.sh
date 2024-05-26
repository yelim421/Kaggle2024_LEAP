#!/bin/bash
#SBATCH -J leap_job
#SBATCH --time=24:00:00
#SBATCH -o leap_output.log
#SBATCH -e leap_error.log
#SBATCH -p cas_v100_4
#SBATCH --comment="python"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --exclusive

module load python/3.12.3

#source /home01/x2817a03/.conda/envs/leap/bin/activate


python3 main.py --config hyper.yaml
#python3 test.py
