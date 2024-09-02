# Updatable Long-Term Memory

Try to design language models whose long-term memory (FFN) can be updated.

## MQAR (Multi-Query Associative Recall)

Multi-Query Associative Recall (MQAR) is a synthetic tasks where the model is presented with a sequence of KV pairs, and is then prompted with a sequence of keys (with spaces in between) and is supposed to produce values.

Run GPT:

```shell
python train.py --model gpt --batch_size 32 --dim 64 --dk 32 --dv 32 --n_heads 2 --n_layers 4 --exp_group 29
# 参数量params: 65792
```

Run DeltaNet:

```shell
python train.py --model delta ...
```

## 参数量对齐的实验

### GPT

```shell
python train.py --model gpt --batch_size 32 --dim 64 --dk 48 --dv 48 --n_heads 2 --n_layers 4 --use_ffn=0 --n_epochs=400
```

#### params: 98560

```plaintext
GPT(
  (embed): Embedding(2048, 64)
  (layers): ModuleList(
    (0-3): 4 x GPTBlock(
      (att_norm): RMSNorm()
      (att): CausalSelfAttention(
        (proj_q): Linear(in_features=64, out_features=96, bias=False)
        (proj_k): Linear(in_features=64, out_features=96, bias=False)
        (proj_v): Linear(in_features=64, out_features=96, bias=False)
        (out_proj): Linear(in_features=96, out_features=64, bias=False)
      )
    )
  )
  (final_norm): RMSNorm()
  (lm_head): Linear(in_features=64, out_features=2048, bias=False)
)
```

#### args

```python
### LOGDIR: runs/exp6_gpt_L4-D64_FF0_OG0_H2_q-48-none-none-0_k-48-none-none-0_v-48-none_data-128-32-2048_lr0.001_1722585151
{'n_epochs': 400, 'ffn_expand_factor': 4, 'n_train_examples': 40000, 'input_seq_len': 128, 'q_gated': 0, 'num_kv_pairs': 32, 'v_kernel': 'none', 'vocab_size': 2048, 'init_std': 0.02, 'att_output_gate_act': 'silu', 'k_gated': 0, 'log_interval': 20, 'du': 64, 'lr_temp': 1.0, 'lr': 0.001, 'k_kernel': 'none', 'model': 'gpt', 'q_norm': 'none', 'lr_gamma': 0.95, 'use_ffn': 0, 'chunk_size': 16, 'n_layers': 4, 'use_att': 1, 'dk': 48, 'dv': 48, 'residual_mul': 0, 'log_dir': 'runs', 'ffn_act': 'silu', 'att_use_output_gate': 0, 'batch_size': 32, 'q_kernel': 'none', 'dim': 64, 'k_norm': 'none', 'random_non_queries': 0, 'n_test_examples': 3000, 'n_heads': 2, 'exp_group': '6', 'clip_grad': 1.0}
>>> n_train_steps = 1250, n_val_steps = 750
```

### Unified KV

```shell
python train.py --model unified_kv --batch_size 32 --dim 64 --du 64 --n_heads 2 --n_layer
s 4 --use_ffn=0 --n_epochs=400
```

#### params: 98560

```plaintext
UnifiedKV(
  (embed): Embedding(2048, 64)
  (layers): ModuleList(
    (0-3): 4 x UnifiedKVBlock(
      (att_norm): RMSNorm()
      (att): CausalSelfAttentionUnifiedKV(
        (proj_q): Linear(in_features=64, out_features=128, bias=False)
        (proj_kv): Linear(in_features=64, out_features=128, bias=False)
        (out_proj): Linear(in_features=128, out_features=64, bias=False)
      )
    )
  )
  (final_norm): RMSNorm()
  (lm_head): Linear(in_features=64, out_features=2048, bias=False)
)
```

#### args

```python
### LOGDIR: runs/exp6_unified_kv_L4-D64_FF0_OG0_H2_q-64-none-none-0_k-64-none-none-0_v-64-none_data-128-32-2048_lr0.001_1722585162
{'v_kernel': 'none', 'du': 64, 'k_gated': 0, 'lr_temp': 1.0, 'model': 'unified_kv', 'n_train_examples': 40000, 'chunk_size': 16, 'exp_group': '6', 'lr_gamma': 0.95, 'n_heads': 2, 'log_interval': 20, 'ffn_act': 'silu', 'num_kv_pairs': 32, 'residual_mul': 0, 'ffn_expand_factor': 4, 'log_dir': 'runs', 'k_kernel': 'none', 'att_output_gate_act': 'silu', 'q_kernel': 'none', 'input_seq_len': 128, 'dk': 64, 'clip_grad': 1.0, 'use_ffn': 0, 'k_norm': 'none', 'use_att': 1, 'att_use_output_gate': 0, 'init_std': 0.02, 'dim': 64, 'n_test_examples': 3000, 'n_epochs': 400, 'q_gated': 0, 'dv': 64, 'random_non_queries': 0, 'q_norm': 'none', 'vocab_size': 2048, 'batch_size': 32, 'lr': 0.001, 'n_layers': 4}
>>> n_train_steps = 1250, n_val_steps = 750
```
