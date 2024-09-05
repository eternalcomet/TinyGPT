# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = "kv_assign"
wandb_run_name = "gpt-tiny-31M"

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 64
block_size = 1024
gradient_accumulation_steps = 1 * 8

# wsd scheduler
# max_iters = 50000
# lr_decay_iters = 5000
# warmup_iters = 2000  # 0.01 * max_iters
# learning_rate = 5e-4
max_iters = 100000
lr_decay_iters = 10000
warmup_iters = 4000  # 0.01 * max_iters
learning_rate = 1e-4


# model settings
# n_layer = 6
# n_head = 4
# n_embd = 256
n_layer = 12
n_head = 8
n_embd = 512
dropout = 0.0

# optimizer settings
min_lr = learning_rate / 10.0
grad_clip = 1.0

# eval stuff
eval_interval = 500
eval_iters = 50
log_interval = 10

# weight decay
weight_decay = 1e-1
