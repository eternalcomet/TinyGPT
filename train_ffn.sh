#!/bin/bash
#SBATCH --account=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8

proj_name="ffn"
run_name="vanilla_ng"
# run_name="tie_kv_ng"

ffn_is_gated="0"
ffn_tie_kv="0"
use_mhf="0"
mhf_n_heads="1"
d_model="256"
mhf_dim_k="256"
mhf_dim_v="256"
n_train_iters="50000"
n_decay_iters="5000"
n_warmup_iters="1000"

export WANDB_MODE=offline
cmd="torchrun --standalone --nproc_per_node=8 train_ffn.py"
cmd+=" --ffn_tie_kv ${ffn_tie_kv}"
cmd+=" --ffn_is_gated ${ffn_is_gated}"
cmd+=" --use_mhf ${use_mhf}"
cmd+=" --mhf_n_heads ${mhf_n_heads}"
cmd+=" --mhf_dim_k ${mhf_dim_k}"
cmd+=" --mhf_dim_v ${mhf_dim_v}"

cmd+=" --n_train_iters ${n_train_iters}"
cmd+=" --n_decay_iters ${n_decay_iters}"
cmd+=" --n_warmup_iters ${n_warmup_iters}"
cmd+=" --wandb_run_name ${run_name}"
cmd+=" --wandb_project ${proj_name}"

echo "RUNNING: $cmd"

$cmd
