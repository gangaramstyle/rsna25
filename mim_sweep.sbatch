#!/bin/bash
# vim: ft=slurm

#–––- JOB SPECIFICATIONS –––––
#SBATCH --job-name=MIMsweep
#SBATCH --output=mim_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=28
#SBATCH --mem-per-gpu=160G

echo "========================================================"
echo "Job is running on host: $(hostname)"
echo "Job started at: $(date)"
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

uv run wandb agent vineeth-gangaram-penn/rsna25/61z09o6e
echo "========================================================"
echo "Agent job finished at: $(date)"
echo "========================================================"
