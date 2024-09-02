# Unified KV

## 参数量对齐

**传统的KV Attention**的参数量为：
Hyperparameters: $d_Q = d_K$, $d_V$, $d_h$

- $W_Q: d_Q \times d_h = d_K \times d_h$
- $W_K: d_K \times d_h$
- $W_V: d_V \times d_h$
- $W_O: d_h \times d_V$

总参数量为：$d_Q \times d_h + d_K \times d_h + d_V \times d_h + d_h \times d_V = 2d_h \times (d_K + d_V)$

**Unified KV Attention**的参数量为：
Hyperparameters: $d_Q = d_K = d_V = d_U$, $d_h$

- $W_Q: d_Q \times d_h$ = $d_U \times d_h$
- $W_{K/V}: d_K \times d_h$ = $d_U \times d_h$
- $W_O: d_h \times d_V$ = $d_h \times d_U$

总参数量为：$d_U \times d_h + d_U \times d_h + d_h \times d_U = 3d_U \times d_h$

对齐参数量：
$$2d_h \times (d_K + d_V) = 3d_U \times d_h$$
即：
$$d_U = \frac{2}{3}(d_K + d_V)$$
