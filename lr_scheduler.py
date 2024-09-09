import math


# WSD scheduler, linear warmup and cosine decay.
def get_wsd_lr(
    it: int,
    lr: float,
    min_lr: float,
    n_decay_iters: int,
    n_warmup_iters: int,
    n_train_iters: int,
) -> float:
    # 1) Warmup stage
    if it < n_warmup_iters:
        return lr * it / n_warmup_iters
    # 2) Stable stage: if it < n_train_iters - n_decay_iters: return max_lr
    if it < n_train_iters - n_decay_iters:
        return lr
    # 3) Annealing stage
    if it < n_train_iters:
        decayed_steps = it - n_train_iters + n_decay_iters
        decay_ratio = decayed_steps / n_decay_iters
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (lr - min_lr)
    # 4) After annealing
    return min_lr


# learning rate decay scheduler (cosine with warmup)
def get_cos_lr(
    it: int, lr: float, min_lr: float, n_warmup_iters: int, n_train_iters: int
) -> float:
    return get_wsd_lr(
        it,
        lr=lr,
        min_lr=min_lr,
        n_decay_iters=n_train_iters - n_warmup_iters,
        n_warmup_iters=n_warmup_iters,
        n_train_iters=n_train_iters,
    )


def get_constant_lr(it: int, lr: float) -> float:
    return lr
