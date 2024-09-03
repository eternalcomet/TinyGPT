import torch
from tap import Tap


class Args(Tap):
    out_dir: str = 'out'
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: int = 0  # if not 0, script exits right after the first eval
    always_save_checkpoint: int = 1  # if not 0, always save a checkpoint after each eval
    init_from: str = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'

    # wandb logging
    wandb_log: int = 1  # disabled by default
    wandb_project: str = 'ffn'
    wandb_run_name: str = 'test'  # 'run' + str(time.time())

    # data
    dataset: str = 'openwebtext'
    grad_accum_steps: int = 1 * 8  # used to simulate larger batch sizes
    batch_size: int = 64  # if grad_accum_steps > 1, this is the micro-batch size
    block_size: int = 1024

    # model
    n_layer: int = 6
    n_head: int = 4
    d_model: int = 256
    dim_k: int = 64
    dim_v: int = 64
    dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias: int = 0  # 0 to turn off bias. Do we use bias inside LayerNorm and Linear layers?
    ffn_tie_kv: int = 0
    ffn_is_gated: int = 1
    ffn_d_mid: int = None
    use_mhf: int = 0
    mhf_n_heads: int = 1

    # adamw optimizer
    lr: float = 5e-4  # max learning rate
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    n_warmup_iters: int = 2000  # how many steps to warm up for
    n_train_iters: int = 100000
    n_decay_iters: int = 10000  # should be ~= n_train_iters per Chinchilla
    min_lr: float = 5e-5  # minimum learning rate, should be ~= lr/10 per Chinchilla

    # DDP settings
    backend: str = 'nccl'  # 'nccl', 'gloo', etc.

    # system
    device: str = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: int = 1  # use PyTorch 2.0 to compile the model to be faster
