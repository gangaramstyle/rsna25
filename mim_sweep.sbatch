#!/bin/bash
# vim: ft=slurm

#–––- JOB SPECIFICATIONS (unchanged) –––––
#SBATCH --job-name=MIMsweep
#SBATCH --output=mim_%j.log
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:1
#SBATCH --cpus-per-gpu=14
#SBATCH --mem-per-gpu=35G

#––––– ENVIRONMENT SETUP (unchanged) ––––––
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


#––––– EXECUTION ––––––
echo "Starting W&B sweep agent..."
cd /cbica/home/gangarav/rsna25

uv run wandb agent gangaramstyle-university-of-pennsylvania/full_data/kxbxd3vi 

echo "========================================================"
echo "Agent job finished at: $(date)"
echo "========================================================"

