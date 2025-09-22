#!/bin/bash
# vim: ft=slurm

#–––- JOB SPECIFICATIONS (unchanged) –––––
#SBATCH --job-name=MIM
#SBATCH --output=job_%j.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:1
#SBATCH --cpus-per-gpu=15
#SBATCH --mem-per-gpu=30G

#––––– ENVIRONMENT SETUP (now much faster) ––––––
echo "========================================================"
echo "Job is running on host: $(hostname)"
echo "Job started at: $(date)"
echo "---"

# Load the same modules used to build the environment
module purge
module load arch_specific
module load slurm/current
module load cuda/12.2
module load gcc/9.4.0
module load python/3.11
module list
echo "---"

echo "Python executable: $(which python)"
echo "---"
nvidia-smi
echo "---"

git branch

#––––– EXECUTION ––––––
#echo "Starting training..."
cd /cbica/home/gangarav/rsna25
uv run MIM_run.py 

echo "========================================================"
echo "Job finished at: $(date)"
echo "========================================================"
