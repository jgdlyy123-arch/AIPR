# AIPR: Adaptive Isometric Plasticity Regularization for Reinforcement Learning

Official implementation of:

**Adaptive Isometric Plasticity Regularization for Reinforcement
Learning (AIPR)**

Submitted to **ICLR 2027 Conference**

**Authors**

-   Jianghui Sang`<sup>`{=html}1`</sup>`{=html}
-   Guangdi Jiang`<sup>`{=html}1`</sup>`{=html}
-   Yongli Wang`<sup>`{=html}2`</sup>`{=html}
-   Anqi Huang`<sup>`{=html}3`</sup>`{=html}
-   Hua Yang`<sup>`{=html}2`</sup>`{=html}
-   Jun Huang`<sup>`{=html}1`</sup>`{=html}

Affiliations:

1.  Anhui University of Technology\
2.  Nanjing University of Science and Technology\
3.  Shandong Technology and Business University

Keywords: Reinforcement Learning, Actor-Critic, Plasticity Loss,
Adaptive Regularization

------------------------------------------------------------------------

## 1. Overview

Deep reinforcement learning (DRL) agents often suffer from **plasticity
loss** during long-term training. Although the agent can initially learn
effective representations, continuous optimization may gradually reduce
the network's ability to adapt to new information.

Existing approaches based on **Deviation from Isometry (DfI)** usually
treat geometric degradation as a trigger condition. When the weight
matrix deviates from an approximately isometric structure, these methods
perform discrete operations such as parameter reinitialization.

However, hard-triggered reinitialization introduces several limitations:

-   It interrupts the continuous optimization trajectory.
-   Abrupt parameter changes may disturb the policy distribution.
-   The same fixed intervention strength cannot adapt to different
    training stages.
-   Actor and Critic networks may have different geometric degradation
    patterns.

To address these issues, we propose:

> **Adaptive Isometric Plasticity Regularization (AIPR)**

AIPR reformulates DfI from a discrete intervention criterion into a
differentiable geometric regularization objective, enabling continuous
preservation of network plasticity during reinforcement learning.

------------------------------------------------------------------------

# 2. Method: Adaptive Isometric Plasticity Regularization

## 2.1 DfI-based Geometric Regularization

For a weight matrix:

\[ W `\in `{=tex}R\^{m `\times `{=tex}n} \]

AIPR measures the deviation from an isometric structure through DfI:

\[ DfI(W) \]

Unlike previous methods that only check whether DfI exceeds a threshold,
AIPR directly incorporates DfI into gradient optimization.

The final objective becomes:

\[ L = L\_{RL}+`\lambda`{=tex}*t L*{AIPR} \]

where:

-   (L\_{RL}): original reinforcement learning objective
-   (L\_{AIPR}): geometric regularization term
-   (`\lambda`{=tex}\_t): adaptive regularization strength

------------------------------------------------------------------------

## 2.2 Adaptive DfI-aware Gating

AIPR introduces a differentiable gating mechanism:

\[ `\lambda`{=tex}\_t=f(DfI_t) \]

The regularization strength changes automatically according to the
current network geometry.

Advantages:

-   Stronger constraint when geometric degradation increases.
-   Weaker constraint when the network maintains healthy plasticity.
-   Avoids manually selecting fixed regularization coefficients.

------------------------------------------------------------------------

## 2.3 Actor-Critic Specific Regularization

Because Actor and Critic networks may experience different geometric
changes, AIPR separately estimates:

-   Actor geometric state
-   Critic geometric state

and generates independent adaptive regularization weights:

\[ `\lambda`{=tex}\_{`\pi`{=tex}},`\lambda`{=tex}\_v \]

This preserves the original Actor-Critic learning coupling while
allowing independent geometric adaptation.

------------------------------------------------------------------------

# 3. Repository Structure

    AIPR/
    │
    ├── run_benchmark.py          # Unified benchmark entry
    ├── main_aipr.sh              # AIPR experiment commands
    ├── main_fire.sh              # FIRE baseline experiments
    │
    ├── rl/
    │   ├── dqn/                  # Atari DQN experiments
    │   │   ├── cleanrl/
    │   │   │   ├── dqn_atari.py
    │   │   │   └── aipr_regularizer_torch.py
    │   │
    │   └── sac/                  # SAC experiments
    │       ├── configs/
    │       ├── scripts/
    │       └── run_online.py
    │
    ├── vision/                   # Vision continual learning experiments
    │
    └── language/                 # Language model continual learning experiments

------------------------------------------------------------------------

# 4. Supported Benchmarks

AIPR supports experiments across multiple reinforcement learning
environments.

## Continuous Control

### MuJoCo

Supported environments:

-   Ant
-   HalfCheetah
-   Hopper
-   Walker2d
-   Humanoid

Example:

``` bash
python run_benchmark.py \
--mode rl_ppo \
--env Ant-v5 \
--suite mujoco \
--seeds 0 1 2 3 4 \
--total_steps 5000000
```

------------------------------------------------------------------------

## DMControl

Supported environments:

-   Cartpole
-   Cheetah
-   Walker
-   Finger
-   Quadruped

Example:

``` bash
python run_benchmark.py \
--mode rl_ppo \
--env dm_control/cartpole-swingup-v0 \
--suite dmcontrol \
--seeds 0 1 2 3 4 \
--total_steps 1000000
```

------------------------------------------------------------------------

## Robosuite

Supported tasks:

-   Lift
-   Stack
-   Door
-   NutAssembly

------------------------------------------------------------------------

## CARL

Supported environments:

-   CARL Acrobot
-   CARL CartPole
-   CARL MountainCarContinuous
-   CARL Pendulum
-   CARL Quadruped

------------------------------------------------------------------------

## Atari

AIPR supports pixel-based reinforcement learning:

-   Pong
-   Breakout
-   Seaquest
-   Qbert
-   SpaceInvaders
-   BeamRider
-   Phoenix
-   Gravitar

Example:

``` bash
python run_benchmark.py \
--mode rl_ppo \
--env ALE/Pong-v5 \
--suite ale \
--seeds 0 1 2 3 4 \
--total_steps 100000000
```

------------------------------------------------------------------------

# 5. Installation

Recommended environment:

  Package     Version
  ----------- ---------
  Python      \>=3.10
  CUDA        12.4
  PyTorch     \>=2.3
  Gymnasium   \>=0.29

Install dependencies:

``` bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install \
gymnasium[mujoco,atari,accept-rom-license] \
dm_control \
robosuite \
metaworld \
carl-bench \
humanoid-bench \
openpyxl
```

------------------------------------------------------------------------

# 6. Running Experiments

## AIPR PPO

Example:

``` bash
python train.py \
--algo ppo \
--env HalfCheetah-v4 \
--suite mujoco \
--seeds 0 1 2 3 4 \
--total_steps 5000000
```

------------------------------------------------------------------------

## AIPR SAC

Example:

``` bash
python run_benchmark.py \
--mode rl_sac \
--env HalfCheetah-v4 \
--suite mujoco \
--seeds 0 1 2 3 4
```

------------------------------------------------------------------------

# 7. Implementation Details

## AIPR Regularizer

The main implementation is located at:

    rl/dqn/cleanrl/aipr_regularizer_torch.py

The module implements:

-   DfI computation
-   Adaptive regularization weight calculation
-   Differentiable geometric constraint
-   Actor/Critic independent regularization

------------------------------------------------------------------------

## Experiment Configuration

Important parameters:

  Parameter            Description
  -------------------- -----------------------------------
  DfI interval         Frequency of geometry measurement
  lambda_max           Maximum regularization strength
  gate function        Adaptive DfI-aware controller
  actor coefficient    Actor regularization weight
  critic coefficient   Critic regularization weight

------------------------------------------------------------------------

# 8. Experimental Goal

AIPR evaluates whether continuous geometric regulation can preserve
neural network plasticity without disrupting optimization.

The experiments compare AIPR with:

-   Vanilla training
-   FIRE
-   Parseval regularization
-   Other plasticity preservation methods

Evaluation metrics include:

-   Episodic return
-   Learning speed
-   Long-term optimization stability
-   Network geometric indicators

------------------------------------------------------------------------

# 9. Citation

If you use this repository, please cite:

``` bibtex
@inproceedings{
sang2026aipr,
title={Adaptive Isometric Plasticity Regularization for Reinforcement Learning},
author={Jianghui Sang and Guangdi Jiang and Yongli Wang and Anqi Huang and Hua Yang and Jun Huang},
booktitle={International Conference on Learning Representations},
year={2027}
}
```

------------------------------------------------------------------------

# 10. License

This project is released under the CC BY 4.0 license.
