# Unified KV

## Environment

```bash
pip install -r requirements.txt
```

sometimes we need:

```bash
export WANDB_MODE=offline
```

## Dataset Processing

Run the following code to download openwebtext dataset.

```bash
python data/openwebtext/prepare.py
```

## How to run

Use following code to run the train and override settings in the config file specified. **DO NOT** forget `""` between string params.

```bash
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt_tiny.py --wandb_run_name="k48v48" --dim_k=48 --dim_v=48
```

Note: you can also modify the config file above or write your own config file. The command line param will **override** the config file.

sometimes we need:

```bash
export WANDB_MODE=offline
```

Then run the command above to sync.

```bash
wandb sync wandb/latest-run
```

## How to Run with SLURM?

Set the arguments in `train_ffn.sh`, and run:

```shell
sbatch -w g{gpu_node} train_ffn.sh
```
