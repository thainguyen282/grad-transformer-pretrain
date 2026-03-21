#!/bin/bash -l
#BATCH --job-name=pretrain
#SBATCH --output=logs/converge.out
#SBATCH --error=logs/converge.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --account=phan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00  # D-HH:MM:SS
#SBATCH --mem=250G
#SBATCH --mail-user=tqn@njit.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -e

module purge
module load wulver # load slurn, easybuild

source ~/.bashrc
conda activate train_env

python main.py
