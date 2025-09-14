#!/bin/bash
# vim: ft=slurm

#–––- JOB SPECIFICATIONS –––––
#SBATCH --job-name=MIMsweep
#SBATCH --output=mim_%j.log
#SBATCH --partition=ai
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=42
#SBATCH --mem-per-gpu=160G

echo "========================================================"
echo "Job is running on host: $(hostname)"
echo "Job started at: $(date)"
echo "---"

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
echo "---"


#––––– EXECUTION ––––––
echo "Starting W&B sweep agent..."
cd /cbica/home/gangarav/rsna25

uv run wandb agent gangaramstyle-university-of-pennsylvania/kai/f8etzrex 

echo "========================================================"
echo "Agent job finished at: $(date)"
echo "========================================================"
