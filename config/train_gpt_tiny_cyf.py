# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = 1
wandb_project = "ukv"
wandb_run_name = "gpt-tiny-31M"

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 64
block_size = 1024
gradient_accumulation_steps = 1 * 8

n_train_iters = 100000
lr_decay_iters = 10000

# model settings
n_layer = 6
n_head = 8
d_model = 256
dropout = 0.0
ffn_tie_kv = 1

# optimizer settings
lr = 5e-4
warmup_iters = 14300  # 0.01 * max_iters
min_lr = learning_rate / 10.0
grad_clip = 1.0

# eval stuff
eval_interval = 200
eval_iters = 100
log_interval = 10

# weight decay
weight_decay = 1e-1
