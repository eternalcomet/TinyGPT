#!/bin/bash
#SBATCH --account=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8

export WANDB_MODE=offline
cmd="torchrun --standalone --nproc_per_node=8 train_ffn.py"

echo "RUNNING: $cmd"

$cmd
