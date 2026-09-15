
# AIPR: Adaptive Isometric Plasticity Regularization for Reinforcement Learning

This repository contains the code for the paper:  
**"AIPR: Adaptive Isometric Plasticity Regularization for Reinforcement Learning"**  
Authors: 
[Jianghui Sang]
[Guangdi Jiang](https://github.com/jgdlyy123-arch) 
[Jun Huang]
[Yongli Wang]
[Hua Yang] 
[Anqi Huang]

<p align="left">
  <img src="assets/iclr2026fire.png" width="300">
</p>

For more information, please see our [project webpage](https://github.com/jgdlyy123-arch/AIPR/) and [paper](https://openreview.net/forum?id=ZSouL9NBLv)


## 📖 Codebase

As we conducted experiments in diverse domains (vision, language and RL), we used different settings for each of them. Please refer below to set up and run experiments:

#### Continual Visual Learning (Fig 2) > [vision/README.md](vision/README.md)

#### Continual Pretraining of LLMs (Fig 3) > [language/README.md](language/README.md)

#### Reinforcement Learning (Fig 4) > [rl/dqn/README.md](rl/dqn/README.md) and [rl/sac/README.md](rl/sac/README.md)

## 🔥FIRE implementation
Stop worrying about plasticity loss, just apply FIRE before training on new data.
```python
import torch
from torch import nn
import numpy as np

@torch.no_grad()
def fire(model, iteration=10):
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            param = m.weight
            weight_matrix = param.data.detach().clone()
            if weight_matrix.ndim == 4: # cnn
                ortho_weight_matrix = torch.zeros_like(weight_matrix)
                for i in range(weight_matrix.shape[2]):
                    for j in range(weight_matrix.shape[3]):
                        ortho_weight_matrix[:,:,i,j] = newton_schulz(weight_matrix[:,:,i,j], num_iters=iteration)
            else: # linear
                ortho_weight_matrix = newton_schulz(weight_matrix, num_iters=iteration)

            # scale = sqrt(d_out/d_in) / kernel_size
            kernel_size = weight_matrix.shape[2]*weight_matrix.shape[3] if weight_matrix.ndim==4 else 1.0
            scale = np.sqrt(weight_matrix.shape[0]/weight_matrix.shape[1]) / kernel_size
            ortho_weight_matrix *= scale
            param.data = ortho_weight_matrix

def newton_schulz(matrix, num_iters=10):
    a, b = (1.5, -0.5)
    assert matrix.ndim == 2
    do_transpose = matrix.size(1) > matrix.size(0)

    X = matrix
    if do_transpose:
        X = X.T

    X = X / X.norm()
    for _ in range(num_iters):
        A = X.T @ X
        X = a * X + b * X @ A

    if do_transpose:
        X = X.T
    return X
```
---

## � 统一基准测试 (run_benchmark.py)

为方便跨环境对比实验，我们提供了 `run_benchmark.py` 作为统一入口，支持所有环境和标准化结果保存。

### 环境要求

| 依赖项 | 版本 |
|--------|------|
| Python | ≥ 3.10 |
| CUDA | 12.4 |
| cuDNN | 9.1 |
| PyTorch | ≥ 2.3.0 |
| torchvision | ≥ 0.18.0 |
| gymnasium | ≥ 0.29.0 |
| openpyxl | ≥ 3.1.0 |

### 安装

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install gymnasium[mujoco,atari,accept-rom-license] dm_control metaworld robosuite ale-py openpyxl
pip install humanoid-bench carl-bench
```

### 运行命令

```bash
# MuJoCo
python run_benchmark.py --algo sac --env HalfCheetah-v4 --suite mujoco --seeds 0 1 2 3 4 --total_steps 1000000

# DMControl
python run_benchmark.py --algo sac --env cartpole-swingup --suite dmc --seeds 0 1 2 3 4 --total_steps 1000000

# Meta-World
python run_benchmark.py --algo sac --env reach-v2 --suite metaworld --seeds 0 1 2 3 4 --total_steps 1000000

# RoboSuite
python run_benchmark.py --algo sac --env Lift --suite robosuite --seeds 0 1 2 3 4 --total_steps 1000000

# Gridworld
python run_benchmark.py --algo ppo --env gridworld --suite gridworld --seeds 0 1 2 3 4 --total_steps 200000

# ALE / Atari
python run_benchmark.py --algo ppo --env ALE/Pong-v5 --suite ale --seeds 0 1 2 3 4 --total_steps 10000000

# CARL-DMCQuadruped
python run_benchmark.py --algo sac --env CARLDmcQuadrupedEnv --suite carl_dmcquadruped --seeds 0 1 2 3 4 --total_steps 1000000

# CARL-LunarLander
python run_benchmark.py --algo ppo --env CARLLunarLanderEnv --suite carl_lunarlander --seeds 0 1 2 3 4 --total_steps 1000000

# HumanoidBench
python run_benchmark.py --algo sac --env h1hand-walk-v0 --suite humanoidbench --seeds 0 1 2 3 4 --total_steps 1000000

# CIFAR-10
python run_benchmark.py --algo ppo --env CIFAR10 --suite cifar10 --seeds 0 1 2 3 4 --n_epochs 200

# CIFAR-100
python run_benchmark.py --algo ppo --env CIFAR100 --suite cifar100 --seeds 0 1 2 3 4 --n_epochs 200
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--algo` | 算法选择 (sac / ppo) | sac |
| `--env` | 环境名称 | HalfCheetah-v4 |
| `--suite` | 环境套件 | mujoco |
| `--seeds` | 随机种子列表 | 0 1 2 3 4 |
| `--total_steps` | RL 总训练步数 | 1000000 |
| `--n_epochs` | 视觉任务训练轮数 | 200 |
| `--reinit_interval` | FIRE 重初始化间隔步数 | 50000 |
| `--fire_iters` | Newton-Schulz 迭代次数 | 10 |
| `--hidden_dim` | 隐藏层维度 | 256 |
| `--lr` | 学习率 | 3e-4 |
| `--batch_size` | 批次大小 | 256 |
| `--gamma` | 折扣因子 | 0.99 |
| `--result_dir` | 结果保存目录 | result |

### 评估指标

**RL 任务**: Mean Return, Sample Efficiency, Learning Curves, 5-seed 标准差, Normalized Return, Sample Efficiency Curves, Success Rate (Meta-World/RoboSuite)

**视觉任务**: Test Accuracy, Validation Loss, Best Test Accuracy

### 输出格式

结果保存在 `result/` 目录，命名规则: `FIRE-{任务名}-{YYYYMMDD_HHMMSS}-seed{N}`
- `.txt`: 纯文本格式，包含所有指标
- `.xlsx`: Excel 格式，包含 Summary / EpisodeMetrics / VisionMetrics 多个工作表

### RTX 4090 性能参考

| 环境 | 步数 | 预计时间 |
|------|------|----------|
| HalfCheetah-v4 (SAC) | 1M | ~30 min |
| ALE/Pong-v5 (PPO) | 10M | ~3 h |
| CIFAR-10 (200 epochs) | - | ~20 min |

---

## �📄 Citation
If you find our work useful, please consider citing the paper as follows:
```

```

---

## AIPR-fire Extension: Adaptive Isometric Policy Regularization

This fork extends FIRE with **AIPR2** — a continuous, adaptive replacement for FIRE's discrete reinitialization.

### Core Idea

| | FIRE (original) | AIPR (this fork) |
|---|---|---|
| Mechanism | Discrete reinit via Newton-Schulz | Continuous adaptive loss term |
| Trigger | Fixed update steps | Automatic via DfI threshold |
| Task boundary needed | Yes | No |

**AIPR loss:**
```
DfI(W) = ||W^T W - I||_F^2
λ_t = λ_0 · sigmoid((DfI_avg − τ) / α)
L_AIPR = λ_t · Σ_l DfI(W_l)
```

### New Files
- `rl/sac/scale_rl/agents/simba/aipr_regularizer.py` — JAX AIPR regularizer
- `rl/dqn/cleanrl/aipr_regularizer_torch.py` — PyTorch AIPR regularizer

### Key Hyperparameters
- `aipr_lambda_0=0.01` — base regularization strength
- `aipr_tau=1.0` — DfI threshold
- `aipr_alpha=0.5` — sigmoid temperature

### Run AIPR-fire (SAC)
```bash
cd rl/sac && bash scripts/single_run/simba_rr2_aipr.sh
```

### Run AIPR-fire (DQN)
```bash
cd rl/dqn
python -m cleanrl.dqn_atari --env-id BreakoutNoFrameskip-v4 --seed 1 --use-aipr
```

