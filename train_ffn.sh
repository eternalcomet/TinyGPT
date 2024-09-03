#!/bin/bash
#SBATCH --account=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8

run_name="vanilla"
# run_name="tie_kv"

ffn_tie_kv="1"
n_train_iters="50000"
n_decay_iters="5000"
n_warmup_iters="1000"

export WANDB_MODE=offline
cmd="torchrun --standalone --nproc_per_node=8 train_ffn.py"
cmd+=" --ffn_tie_kv ${ffn_tie_kv}"
cmd+=" --n_train_iters ${n_train_iters}"
cmd+=" --n_decay_iters ${n_decay_iters}"
cmd+=" --n_warmup_iters ${n_warmup_iters}"

echo "RUNNING: $cmd"

$cmd
