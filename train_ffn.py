"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
from pathlib import Path
import time
import math
import pickle
from contextlib import nullcontext
from functools import partial

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from models.gpt import GPTConfig, GPT
from args import Args
from lr_scheduler import get_wsd_lr

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O

args = Args().parse_args()
# # -----------------------------------------------------------------------------
# config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read())  # overrides from command line or config file
# config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# # -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
device = args.device
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=args.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert args.grad_accum_steps % ddp_world_size == 0
    args.grad_accum_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    
if master_process:
    print('================ args ================')
    print(args)
    print('======================================')
tokens_per_iter = args.grad_accum_steps * ddp_world_size * args.batch_size * args.block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

out_dir = Path(args.out_dir)

if master_process:
    out_dir.mkdir(exist_ok=True, parents=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = Path('data', args.dataset)

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(data_dir / 'train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap(data_dir / 'val.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + args.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + args.block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a ckpt)
cur_step = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = data_dir / 'meta.pkl'
meta_vocab_size = None
if meta_path.exists():
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=args.n_layer,
    n_head=args.n_head,
    d_model=args.d_model,
    block_size=args.block_size,
    bias=bool(args.bias),
    vocab_size=None,
    dropout=args.dropout,
    dim_k=args.dim_k,
    dim_v=args.dim_v,
    ffn_is_gated=bool(args.ffn_is_gated),
    ffn_tie_kv=bool(args.ffn_tie_kv),
    ffn_d_mid=args.ffn_d_mid,
    use_mhf=bool(args.use_mhf),
    mhf_n_heads=args.mhf_n_heads,
    mhf_dim_k=args.mhf_dim_k,
    mhf_dim_v=args.mhf_dim_v,
)  # start with model_args from command line

if args.init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif args.init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a ckpt.
    ckpt_path = out_dir / 'ckpt.pt'
    ckpt = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = ckpt['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'd_model', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = ckpt['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    cur_step = ckpt['cur_step']
    best_val_loss = ckpt['best_val_loss']
elif args.init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {args.init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=args.dropout)
    model = GPT.from_pretrained(args.init_from, override_args)
    # read off the created config params, so we can store them into ckpt correctly
    for k in ['n_layer', 'n_head', 'd_model', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
        

# crop down the model block size if desired, using model surgery
if args.block_size < model.config.block_size:
    model.crop_block_size(args.block_size)
    model_args['block_size'] = args.block_size  # so that the ckpt will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(args.weight_decay, args.lr, (args.beta1, args.beta2), device_type)
if args.init_from == 'resume':
    optimizer.load_state_dict(ckpt['optimizer'])
ckpt = None  # free up memory

# compile the model
if bool(args.compile):
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


get_lr = partial(
    get_wsd_lr,
    lr=args.lr,
    min_lr=args.min_lr,
    n_decay_iters=args.n_decay_iters, 
    n_train_iters=args.n_train_iters, 
    n_warmup_iters=args.n_warmup_iters,
)


# logging
if args.wandb_log and master_process:
    import wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args.as_dict())

# training loop
X, Y = get_batch('train')  # fetch the very first batch
local_cur_step = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0

while True:
    iter_start_time = time.time()
    # determine and set the learning rate for this iteration
    lr = get_lr(cur_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if cur_step % args.eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {cur_step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if args.wandb_log:
            wandb.log({
                "iter": cur_step,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
            })
        if losses['val'] < best_val_loss or args.always_save_checkpoint:
            best_val_loss = losses['val']
            if cur_step > 0:
                ckpt = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'cur_step': cur_step,
                    'best_val_loss': best_val_loss,
                    'config': args.as_dict(),
                }
                print(f"saving ckpt to {out_dir}")
                ckpt_path = out_dir / 'ckpt.pt'
                torch.save(ckpt, ckpt_path)
    if cur_step == 0 and args.eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(args.grad_accum_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == args.grad_accum_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / args.grad_accum_steps  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if args.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    cur_time = time.time()
    iter_time = cur_time - iter_start_time
    if master_process and cur_step % args.log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * args.grad_accum_steps
        if local_cur_step >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(args.batch_size * args.grad_accum_steps, iter_time)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"[it {cur_step}] loss {lossf:.4f}, time {iter_time * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%, lr {lr:.3e}")
        if args.wandb_log:
            wandb.log(
                {
                    "iter": cur_step,
                    "train/loss": lossf,
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage,
                    "iter_time": iter_time,
                }
            )
    cur_step += 1
    local_cur_step += 1

    # termination conditions
    if cur_step > args.n_train_iters:
        break

if ddp:
    destroy_process_group()
