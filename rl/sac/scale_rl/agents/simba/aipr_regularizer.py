"""
AIPR: Adaptive Isometric Policy Regularization (JAX/Flax implementation)

将 AIPR 的 DfI 指标从"离散重初始化触发条件"转化为"连续在线正则化信号"。

核心公式:
    L_AIPR = lambda_t * sum_l DfI(W_l)
    DfI(W) = ||W^T W - I||_F^2
    lambda_t = lambda_0 * sigmoid((DfI_avg - tau) / alpha)

- DfI 小时 (< tau): sigmoid 输出接近 0, 几乎不干扰任务梯度 (解决 Parseval 收敛慢)
- DfI 大时 (>> tau): 强力纠正权重等距性
- 无需任务边界信息 (解决 AIPR 的核心局限)
"""

from typing import Dict, Tuple

import flax
import jax
import jax.numpy as jnp

Params = flax.core.FrozenDict


def compute_dfi_2d(W: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Deviation from Isometry (DfI) for a 2D weight matrix W.

    DfI(W) = ||W^T W - I||_F^2  (if W has more rows than cols)
           = ||W W^T - I||_F^2  (if W has more cols than rows)

    Args:
        W: 2D weight matrix of shape (m, n)
    Returns:
        Scalar DfI value
    """
    m, n = W.shape
    if m >= n:
        gram = W.T @ W
        identity = jnp.eye(n, dtype=W.dtype)
    else:
        gram = W @ W.T
        identity = jnp.eye(m, dtype=W.dtype)
    return jnp.sum((gram - identity) ** 2)


def compute_aipr_loss(
    params: Params,
    lambda_0: float = 0.01,
    tau: float = 1.0,
    alpha: float = 0.5,
) -> jnp.ndarray:
    """
    Compute AIPR regularization loss over all weight matrices in params.

    Traverses the params pytree, computes DfI for each 2D kernel matrix,
    and returns the adaptively-weighted sum. Compatible with JAX JIT and
    gradient computation (the tree structure is static at trace time).

    Args:
        params: Flax frozen dict of network parameters
        lambda_0: base regularization strength
        tau: DfI activation threshold (regularization activates above this)
        alpha: temperature for sigmoid sharpness
    Returns:
        Scalar AIPR loss (JAX array)
    """
    leaves = jax.tree_util.tree_leaves(params)

    dfi_list = []
    for leaf in leaves:
        if leaf.ndim == 2:
            dfi_list.append(compute_dfi_2d(leaf))
        elif leaf.ndim == 3:
            # Vmapped multi-head networks (e.g. ClippedDoubleCritic):
            # shape (num_heads, m, n) — compute per-head and average
            dfi_per_head = jax.vmap(compute_dfi_2d)(leaf)
            dfi_list.append(jnp.mean(dfi_per_head))

    if not dfi_list:
        return jnp.array(0.0, dtype=jnp.float32)

    # Sum all DfI values
    total_dfi = dfi_list[0]
    for dfi in dfi_list[1:]:
        total_dfi = total_dfi + dfi

    n_layers = len(dfi_list)
    avg_dfi = total_dfi / n_layers

    # Adaptive lambda: sigmoid gate — quiet when DfI is small, strong when large
    adaptive_lambda = jax.lax.stop_gradient(
        lambda_0 * jax.nn.sigmoid((avg_dfi - tau) / alpha)
    )

    return adaptive_lambda * total_dfi


def compute_dfi_metrics(
    params: Params,
    prefix: str = "",
) -> Dict[str, jnp.ndarray]:
    """
    Compute DfI diagnostic metrics for logging.

    Args:
        params: Flax frozen dict of network parameters
        prefix: string prefix for metric keys (e.g. "actor", "critic")
    Returns:
        Dict of metric name -> scalar JAX array
    """
    leaves = jax.tree_util.tree_leaves(params)

    dfi_list = []
    for leaf in leaves:
        if leaf.ndim == 2:
            dfi_list.append(compute_dfi_2d(leaf))
        elif leaf.ndim == 3:
            dfi_per_head = jax.vmap(compute_dfi_2d)(leaf)
            dfi_list.append(jnp.mean(dfi_per_head))

    if not dfi_list:
        return {}

    total_dfi = dfi_list[0]
    for dfi in dfi_list[1:]:
        total_dfi = total_dfi + dfi

    avg_dfi = total_dfi / len(dfi_list)
    key_prefix = f"{prefix}/" if prefix else ""

    return {
        f"{key_prefix}aipr/total_dfi": total_dfi,
        f"{key_prefix}aipr/avg_dfi": avg_dfi,
        f"{key_prefix}aipr/n_layers": len(dfi_list),
    }
