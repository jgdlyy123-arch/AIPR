#!/usr/bin/env python3
"""
AIPR 统一基准测试脚本。
支持在所有环境套件上运行 AIPR 算法，并保存标准化实验结果。

用法:
  # RL 任务 (SAC)
  python run_benchmark.py --mode rl_sac --env HalfCheetah-v4 --suite mujoco --seed 42
  python run_benchmark.py --mode rl_sac --env dm_control/cheetah-run-v0 --suite dmcontrol --seed 0

  # RL 任务 (DQN, Atari)
  python run_benchmark.py --mode rl_dqn --env ALE/Pong-v5 --suite atari --seed 42

  # 视觉任务
  python run_benchmark.py --mode vision --env CIFAR10 --seed 0
  python run_benchmark.py --mode vision --env CIFAR100 --seed 0

  # 多 seed
  python run_benchmark.py --mode rl_sac --env HalfCheetah-v4 --suite mujoco --seeds 0 1 2 3 4
"""

import argparse
import os
import sys
import time
import random
from datetime import datetime
from collections import deque
import numpy as np
import torch

# ─── 通用环境工厂 ───────────────────────────────────────────────────────

try:
    import gymnasium as gym
except ImportError:
    import gym

ENV_SUITES = {
    "atari": ["ALE/Pong-v5", "ALE/Breakout-v5", "ALE/Seaquest-v5", "ALE/Qbert-v5",
              "ALE/SpaceInvaders-v5", "ALE/BeamRider-v5", "ALE/Asterix-v5", "ALE/DemonAttack-v5"],
    "ale": ["ALE/Pong-v5", "ALE/Breakout-v5", "ALE/Seaquest-v5", "ALE/Qbert-v5",
            "ALE/SpaceInvaders-v5", "ALE/BeamRider-v5", "ALE/Asterix-v5", "ALE/DemonAttack-v5"],
    "mujoco": ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4", "Humanoid-v4",
               "Swimmer-v4", "Reacher-v4", "InvertedPendulum-v4"],
    "dmcontrol": ["dm_control/cartpole-swingup-v0", "dm_control/cheetah-run-v0",
                  "dm_control/walker-walk-v0", "dm_control/finger-spin-v0",
                  "dm_control/quadruped-walk-v0", "dm_control/quadruped-run-v0"],
    "gridworld": ["GridWorld-v0"],
    "carl": ["CARLDmcQuadrupedEnv", "CARLLunarLanderEnv", "CARLMountainCarContinuous",
             "CARLPendulum", "CARLCartPole", "CARLAcrobot"],
    "carl_dmcquadruped": ["CARLDmcQuadrupedEnv"],
    "carl_lunarlander": ["CARLLunarLanderEnv"],
    "metaworld": ["reach-v2", "push-v2", "pick-place-v2", "drawer-open-v2", "door-open-v2"],
    "robosuite": ["Lift", "Stack", "Door", "NutAssembly"],
    "humanoidbench": ["h1hand-walk-v0", "h1hand-run-v0", "h1hand-stand-v0"],
    "cifar10": ["CIFAR10"],
    "cifar100": ["CIFAR100"],
}


def detect_suite(env_name):
    for suite, names in ENV_SUITES.items():
        if env_name in names:
            return suite
    name_lower = env_name.lower()
    if "ale/" in name_lower:
        return "atari"
    if "dm_control" in name_lower:
        return "dmcontrol"
    if "gridworld" in name_lower:
        return "gridworld"
    if env_name.startswith("CARL"):
        return "carl"
    if "h1hand" in name_lower:
        return "humanoidbench"
    if name_lower in ("cifar10", "cifar-10"):
        return "cifar10"
    if name_lower in ("cifar100", "cifar-100"):
        return "cifar100"
    return "mujoco"


def make_env(env_name, suite=None, seed=42):
    if suite is None:
        suite = detect_suite(env_name)

    if suite in ("cifar10", "cifar100"):
        return None, {"is_vision": True, "dataset": suite.upper(),
                       "obs_dim": (3, 32, 32), "act_dim": 10 if suite == "cifar10" else 100,
                       "continuous": False, "has_success": False, "use_cnn": True}

    env_info = {"has_success": False, "use_cnn": False, "is_vision": False}

    if suite in ("atari", "ale"):
        env = gym.make(env_name, render_mode=None)
        from gymnasium.wrappers import AtariPreprocessing, FrameStack
        env = AtariPreprocessing(env, frame_skip=1, terminal_on_life_loss=True)
        env = FrameStack(env, 4)
        env_info.update(obs_dim=env.observation_space.shape, act_dim=env.action_space.n,
                        continuous=False, use_cnn=True)
    elif suite == "mujoco":
        env = gym.make(env_name)
        env_info.update(obs_dim=env.observation_space.shape[0],
                        act_dim=env.action_space.shape[0], continuous=True)
    elif suite == "dmcontrol":
        env = gym.make(env_name)
        import numpy as np
        from gymnasium.wrappers import FlattenObservation
        env = FlattenObservation(env)
        obs_dim = env.observation_space.shape[0]
        env_info.update(obs_dim=obs_dim, act_dim=env.action_space.shape[0], continuous=True)
    elif suite == "gridworld":
        env = gym.make("CartPole-v1")  # Fallback
        env_info.update(obs_dim=env.observation_space.shape[0],
                        act_dim=env.action_space.n, continuous=False)
    elif suite in ("carl", "carl_dmcquadruped", "carl_lunarlander"):
        try:
            import carl.envs as carl_envs
            env = getattr(carl_envs, env_name)()
        except ImportError:
            env = gym.make("CartPole-v1")
        if isinstance(env.action_space, gym.spaces.Discrete):
            env_info.update(obs_dim=env.observation_space.shape[0],
                            act_dim=env.action_space.n, continuous=False)
        else:
            env_info.update(obs_dim=env.observation_space.shape[0],
                            act_dim=env.action_space.shape[0], continuous=True)
    elif suite == "metaworld":
        try:
            import metaworld
            mt1 = metaworld.MT1(env_name, seed=seed)
            env = mt1.train_classes[env_name]()
            env.set_task(mt1.train_tasks[0])
        except ImportError:
            env = gym.make("Reacher-v4")
        env_info.update(obs_dim=env.observation_space.shape[0],
                        act_dim=env.action_space.shape[0], continuous=True, has_success=True)
    elif suite == "robosuite":
        try:
            import robosuite
            from robosuite.wrappers import GymWrapper
            raw = robosuite.make(env_name, robots="Panda", has_renderer=False,
                                  has_offscreen_renderer=False, use_camera_obs=False,
                                  reward_shaping=True, control_freq=20, horizon=500)
            env = GymWrapper(raw)
        except ImportError:
            env = gym.make("Reacher-v4")
        env_info.update(obs_dim=env.observation_space.shape[0],
                        act_dim=env.action_space.shape[0], continuous=True, has_success=True)
    elif suite == "humanoidbench":
        try:
            import humanoid_bench
            env = gym.make(env_name)
        except ImportError:
            env = gym.make("Humanoid-v4")
        env_info.update(obs_dim=env.observation_space.shape[0],
                        act_dim=env.action_space.shape[0], continuous=True, has_success=True)
    else:
        env = gym.make(env_name)
        if isinstance(env.action_space, gym.spaces.Discrete):
            env_info.update(obs_dim=env.observation_space.shape[0],
                            act_dim=env.action_space.n, continuous=False)
        else:
            env_info.update(obs_dim=env.observation_space.shape[0],
                            act_dim=env.action_space.shape[0], continuous=True)
    return env, env_info


# ─── 结果保存 ────────────────────────────────────────────────────────────

def make_filename(algo_name, task_name, seed):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_task = task_name.replace("/", "-").replace("\\", "-")
    return f"{algo_name}-{safe_task}-{timestamp}-seed{seed}"

def save_results_txt(filepath, metrics, config=None, episodes=None, save_individual_files=True):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("AIPR Experiment Results\n")
        f.write("=" * 60 + "\n\n")
        if config:
            f.write("--- Configuration ---\n")
            for k, v in config.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
        f.write("--- Final Metrics ---\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"  {k}: {v:.4f}\n")
            else:
                f.write(f"  {k}: {v}\n")
        f.write("\n")
        if episodes:
            f.write("--- Episode Data ---\n")
            f.write(f"{'Episode':>8} {'Return':>12} {'Length':>8}\n")
            f.write("-" * 30 + "\n")
            for ep in episodes:
                f.write(f"{ep['episode']:>8d} {ep['return']:>12.2f} {ep['length']:>8d}\n")
    print(f"[ResultSaver] TXT saved: {filepath}")
    if save_individual_files:
        episode_rewards_file = filepath.replace(".txt", "_episode_rewards.txt")
        with open(episode_rewards_file, "w", encoding="utf-8") as f:
            for ep in episodes:
                f.write(f"{ep['return']}\n")
        print(f"[ResultSaver] Episode rewards saved: {episode_rewards_file}")

        mean_return_file = filepath.replace(".txt", "_mean_return.txt")
        with open(mean_return_file, "w", encoding="utf-8") as f:
            f.write(f"{metrics['mean_return']}\n")
        print(f"[ResultSaver] Mean return saved: {mean_return_file}")

        success_rate_file = filepath.replace(".txt", "_success_rate.txt")
        with open(success_rate_file, "w", encoding="utf-8") as f:
            f.write(f"{metrics['success_rate']}\n")
        print(f"[ResultSaver] Success rate saved: {success_rate_file}")


def save_results_excel(filepath, metrics, config=None, episodes=None, vision_metrics=None):
    try:
        import openpyxl
    except ImportError:
        print("[WARN] openpyxl 未安装. pip install openpyxl")
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["Metric", "Value"])
    for k, v in metrics.items():
        ws.append([k, float(v) if isinstance(v, (np.floating, np.integer)) else v])
    if config:
        ws2 = wb.create_sheet("Config")
        ws2.append(["Parameter", "Value"])
        for k, v in config.items():
            ws2.append([str(k), str(v)])
    if episodes:
        ws3 = wb.create_sheet("Episodes")
        ws3.append(["Episode", "Return", "Length"])
        for ep in episodes:
            ws3.append([ep["episode"], ep["return"], ep["length"]])
    if vision_metrics:
        ws4 = wb.create_sheet("VisionMetrics")
        ws4.append(["Epoch", "TrainLoss", "TrainAcc", "ValLoss", "ValAcc", "TestAcc"])
        for vm in vision_metrics:
            ws4.append([vm["epoch"], vm["train_loss"], vm["train_acc"],
                        vm["val_loss"], vm["val_acc"], vm["test_acc"]])
    wb.save(filepath)
    print(f"[ResultSaver] Excel saved: {filepath}")


def compute_final_metrics(returns, successes=None, normalize_factor=1000.0,
                          vision_metrics=None):
    metrics = {}
    if returns:
        last_n = min(100, len(returns))
        last_r = returns[-last_n:]
        metrics["mean_return"] = float(np.mean(last_r))
        metrics["std_return"] = float(np.std(last_r))
        metrics["max_return"] = float(np.max(returns))
        metrics["normalized_return"] = float(np.mean(last_r) / normalize_factor)
        metrics["total_episodes"] = len(returns)

        max_ret = np.max(returns)
        if max_ret > 0:
            for t in [0.5, 0.7, 0.9]:
                target = t * max_ret
                idx = next((i for i, r in enumerate(returns) if r >= target), len(returns))
                metrics[f"sample_eff_{int(t*100)}pct"] = idx

    if successes:
        metrics["success_rate"] = float(np.mean(successes))

    if vision_metrics:
        last = vision_metrics[-1]
        metrics["test_accuracy"] = last.get("test_acc", 0.0)
        metrics["validation_loss"] = last.get("val_loss", 0.0)
        metrics["best_test_accuracy"] = max(vm.get("test_acc", 0) for vm in vision_metrics)

    return metrics


# ─── RL 训练 (SAC 和 DQN 用 PPO 代替) ──────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleSACAgent:
    """轻量 SAC 实现，带 AIPR 正则化（Newton-Schulz 正交投影）。"""

    def __init__(self, obs_dim, act_dim, hidden=256, lr=3e-4, gamma=0.99,
                 tau=0.005, alpha=0.2, buffer_size=1000000, batch_size=256,
                 start_steps=10000,
                 aipr_interval=50000, aipr_iters=10, device="cuda"):
        import torch.nn as nn
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.aipr_interval = aipr_interval
        self.aipr_iters = aipr_iters
        self.start_steps = start_steps
        self.name = "AIPR_SAC"

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        ).to(device)
        self.actor_mean = nn.Linear(hidden, act_dim).to(device)
        self.actor_log_std = nn.Linear(hidden, act_dim).to(device)

        # Critic (Twin Q)
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(device)
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(device)
        self.q1_target = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(device)
        self.q2_target = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.actor_opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.actor_mean.parameters()) + list(self.actor_log_std.parameters()),
            lr=lr
        )
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )

        # Temperature
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device, dtype=torch.float32)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -act_dim

        # Buffer
        self.buffer_obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.buffer_act = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.buffer_rew = np.zeros(buffer_size, dtype=np.float32)
        self.buffer_next = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.buffer_done = np.zeros(buffer_size, dtype=np.float32)
        self.buffer_ptr = 0
        self.buffer_full = False
        self.buffer_max = buffer_size

        # Orthogonal init
        for m in [self.actor, self.q1, self.q2]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)

    def add(self, obs, act, rew, next_obs, done):
        i = self.buffer_ptr % self.buffer_max
        self.buffer_obs[i] = obs
        self.buffer_act[i] = act
        self.buffer_rew[i] = rew
        self.buffer_next[i] = next_obs
        self.buffer_done[i] = done
        self.buffer_ptr += 1
        if self.buffer_ptr >= self.buffer_max:
            self.buffer_full = True
            self.buffer_ptr = self.buffer_ptr % self.buffer_max

    def sample_batch(self):
        max_idx = self.buffer_max if self.buffer_full else self.buffer_ptr
        idx = np.random.randint(0, max_idx, size=self.batch_size)
        return {
            "obs": torch.as_tensor(self.buffer_obs[idx], device=self.device),
            "act": torch.as_tensor(self.buffer_act[idx], device=self.device),
            "rew": torch.as_tensor(self.buffer_rew[idx], device=self.device),
            "next": torch.as_tensor(self.buffer_next[idx], device=self.device),
            "done": torch.as_tensor(self.buffer_done[idx], device=self.device),
        }

    def _get_action(self, obs_t, deterministic=False):
        h = self.actor(obs_t)
        mean = self.actor_mean(h)
        log_std = self.actor_log_std(h).clamp(-20, 2)
        std = log_std.exp()
        if deterministic:
            return torch.tanh(mean), None
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)

    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _ = self._get_action(obs_t, deterministic)
        return action.cpu().numpy().flatten()

    def aipr_reinit(self):
        """AIPR: Newton-Schulz 正交重初始化。"""
        import torch.nn as nn
        for net in [self.actor, self.q1, self.q2]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    W = layer.weight.data
                    if W.shape[0] >= W.shape[1]:
                        W = self._newton_schulz(W, self.aipr_iters)
                        scale = np.sqrt(W.shape[0] / W.shape[1])
                        layer.weight.data = scale * W

    @staticmethod
    def _newton_schulz(W, n_iters=10):
        X = W / (W.norm() + 1e-7)
        for _ in range(n_iters):
            A = X.t() @ X
            X = 1.5 * X - 0.5 * X @ A
        return X

    def update(self, step):
        if step < self.start_steps:
            return {}
        if (self.buffer_ptr if not self.buffer_full else self.buffer_max) < self.batch_size:
            return {}
        batch = self.sample_batch()
        alpha = self.log_alpha.exp().detach()

        # Critic update
        with torch.no_grad():
            next_action, next_log_prob = self._get_action(batch["next"])
            sa_next = torch.cat([batch["next"], next_action], -1)
            target_q = torch.min(self.q1_target(sa_next), self.q2_target(sa_next))
            target = batch["rew"].unsqueeze(-1) + self.gamma * (1 - batch["done"].unsqueeze(-1)) * (target_q - alpha * next_log_prob)

        sa = torch.cat([batch["obs"], batch["act"]], -1)
        q1_loss = ((self.q1(sa) - target) ** 2).mean()
        q2_loss = ((self.q2(sa) - target) ** 2).mean()
        critic_loss = q1_loss + q2_loss

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        action, log_prob = self._get_action(batch["obs"])
        sa_new = torch.cat([batch["obs"], action], -1)
        q_val = torch.min(self.q1(sa_new), self.q2(sa_new))
        actor_loss = (alpha * log_prob - q_val).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Target update
        for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
            pt.data.mul_(1 - self.tau).add_(p.data * self.tau)
        for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
            pt.data.mul_(1 - self.tau).add_(p.data * self.tau)

        # AIPR reinit
        if self.aipr_interval > 0 and step % self.aipr_interval == 0 and step > 0:
            self.aipr_reinit()

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}


class SimplePPOAgent:
    """轻量 PPO 实现，带 AIPR 正则化。用于离散/连续动作空间。"""

    def __init__(self, obs_dim, act_dim, continuous=True, hidden=256, lr=3e-4,
                 gamma=0.99, gae_lambda=0.95, clip_eps=0.2, n_steps=2048,
                 n_epochs=10, mini_batch_size=64, ent_coef=0.01,
                 aipr_interval=50000, aipr_iters=10, use_cnn=False, device="cuda"):
        import torch.nn as nn
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.ent_coef = ent_coef
        self.continuous = continuous
        self.aipr_interval = aipr_interval
        self.aipr_iters = aipr_iters
        self.name = "AIPR_PPO"
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        if use_cnn and isinstance(obs_dim, tuple):
            in_ch = obs_dim[0]
            self.feature_net = nn.Sequential(
                nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
                nn.Flatten(),
            ).to(device)
            feat_dim = 3136
        else:
            self.feature_net = None
            feat_dim = obs_dim if isinstance(obs_dim, int) else int(np.prod(obs_dim))

        # Actor
        if continuous:
            self.actor = nn.Sequential(
                nn.Linear(feat_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, act_dim),
            ).to(device)
            self.log_std = nn.Parameter(torch.zeros(act_dim, device=device))
        else:
            self.actor = nn.Sequential(
                nn.Linear(feat_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, act_dim),
            ).to(device)
            self.log_std = None

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        ).to(device)

        all_params = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.feature_net:
            all_params += list(self.feature_net.parameters())
        if self.log_std is not None:
            all_params.append(self.log_std)
        self.optimizer = torch.optim.Adam(all_params, lr=lr)

        # Rollout buffer
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.done_buf = []
        self.val_buf = []
        self.logp_buf = []

    def _features(self, obs_t):
        if self.feature_net:
            return self.feature_net(obs_t.float() / 255.0)
        return obs_t.float()

    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            feat = self._features(obs_t)
            if self.continuous:
                mean = self.actor(feat)
                std = self.log_std.exp()
                if deterministic:
                    action = mean
                else:
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                log_prob = torch.distributions.Normal(mean, std).log_prob(action).sum(-1)
                value = self.critic(feat)
                return action.cpu().numpy().flatten(), log_prob.item(), value.item()
            else:
                logits = self.actor(feat)
                dist = torch.distributions.Categorical(logits=logits)
                if deterministic:
                    action = logits.argmax(-1)
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action)
                value = self.critic(feat)
                return action.item(), log_prob.item(), value.item()

    def store(self, obs, act, rew, done, val, logp):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.done_buf.append(done)
        self.val_buf.append(val)
        self.logp_buf.append(logp)

    def update(self, last_value):
        # GAE
        rewards = np.array(self.rew_buf)
        values = np.array(self.val_buf + [last_value])
        dones = np.array(self.done_buf)
        advs = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advs[t] = gae
        returns = advs + values[:-1]

        obs_t = torch.as_tensor(np.array(self.obs_buf), dtype=torch.float32, device=self.device)
        if self.continuous:
            act_t = torch.as_tensor(np.array(self.act_buf), dtype=torch.float32, device=self.device)
        else:
            act_t = torch.as_tensor(np.array(self.act_buf), dtype=torch.long, device=self.device)
        adv_t = torch.as_tensor(advs, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        old_logp_t = torch.as_tensor(np.array(self.logp_buf), dtype=torch.float32, device=self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(self.obs_buf)
        for _ in range(self.n_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n)
                mb = idx[start:end]
                feat = self._features(obs_t[mb])

                if self.continuous:
                    mean = self.actor(feat)
                    std = self.log_std.exp()
                    dist = torch.distributions.Normal(mean, std)
                    log_prob = dist.log_prob(act_t[mb]).sum(-1)
                    entropy = dist.entropy().sum(-1).mean()
                else:
                    logits = self.actor(feat)
                    dist = torch.distributions.Categorical(logits=logits)
                    log_prob = dist.log_prob(act_t[mb])
                    entropy = dist.entropy().mean()

                ratio = (log_prob - old_logp_t[mb]).exp()
                surr1 = ratio * adv_t[mb]
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                actor_loss = -torch.min(surr1, surr2).mean()

                value = self.critic(feat).squeeze(-1)
                value_loss = ((value - ret_t[mb]) ** 2).mean()

                loss = actor_loss + 0.5 * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                all_params = list(self.actor.parameters()) + list(self.critic.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, 0.5)
                self.optimizer.step()

        self.obs_buf.clear()
        self.act_buf.clear()
        self.rew_buf.clear()
        self.done_buf.clear()
        self.val_buf.clear()
        self.logp_buf.clear()

    def aipr_reinit(self):
        import torch.nn as nn
        for net in [self.actor, self.critic]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    W = layer.weight.data
                    if W.shape[0] >= W.shape[1]:
                        X = W / (W.norm() + 1e-7)
                        for _ in range(self.aipr_iters):
                            A = X.t() @ X
                            X = 1.5 * X - 0.5 * X @ A
                        scale = np.sqrt(W.shape[0] / W.shape[1])
                        layer.weight.data = scale * X


# ─── 视觉任务训练 ─────────────────────────────────────────────────────

def train_vision_aipr(args, seed):
    """AIPR 视觉训练 (CIFAR-10/100)。"""
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, random_split
    import torch.nn as nn

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = args.env.upper()

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    if dataset == "CIFAR10":
        train_full = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)
        n_cls = 10
    else:
        train_full = torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=transform_test)
        n_cls = 100

    val_size = int(len(train_full) * 0.1)
    train_set, val_set = random_split(train_full, [len(train_full) - val_size, val_size])
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    model = torchvision.models.resnet18(weights=None, num_classes=n_cls)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
    criterion = nn.CrossEntropyLoss()

    aipr_interval = getattr(args, "aipr_interval", 20)
    aipr_iters = getattr(args, "aipr_iters", 10)

    episodes = []
    vision_metrics = []
    returns = []
    successes = []

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
        scheduler.step()
        train_acc = 100.0 * correct / total
        train_loss /= total

        # AIPR reinit
        if aipr_interval > 0 and epoch % aipr_interval == 0:
            with torch.no_grad():
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        W = m.weight.data
                        if W.shape[0] >= W.shape[1]:
                            X = W / (W.norm() + 1e-7)
                            for _ in range(aipr_iters):
                                A = X.t() @ X
                                X = 1.5 * X - 0.5 * X @ A
                            scale = np.sqrt(W.shape[0] / W.shape[1])
                            m.weight.data = scale * X

        # Val
        model.eval()
        val_loss, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += criterion(out, y).item() * x.size(0)
                _, pred = out.max(1)
                vt += y.size(0)
                vc += pred.eq(y).sum().item()
        val_acc = 100.0 * vc / vt
        val_loss /= vt

        # Test
        tc, tt = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                _, pred = model(x).max(1)
                tt += y.size(0)
                tc += pred.eq(y).sum().item()
        test_acc = 100.0 * tc / tt

        returns.append(test_acc)
        ep_success = test_acc > 90.0 if n_cls == 10 else test_acc > 70.0
        successes.append(ep_success)
        episodes.append({"episode": epoch, "return": test_acc, "length": len(train_loader)})
        vision_metrics.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                                "val_loss": val_loss, "val_acc": val_acc, "test_acc": test_acc})

        if epoch % 10 == 0:
            print(f"[Epoch {epoch:>3d}/{args.n_epochs}] TrainAcc: {train_acc:.2f}% | "
                  f"ValLoss: {val_loss:.4f} | TestAcc: {test_acc:.2f}%")

    return returns, successes, episodes, vision_metrics


# ─── RL 训练循环 ──────────────────────────────────────────────────────

def train_rl(args, seed):
    """RL 训练 (SAC / PPO)，带 AIPR 正则化。"""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env, env_info = make_env(args.env, args.suite, seed)

    mean_queue = deque(maxlen=100)
    suite = detect_suite(args.env)
    result_dir = os.path.join(args.result_dir, suite, args.mode)
    os.makedirs(result_dir, exist_ok=True)
    base = make_filename("AIPR", args.env, seed)
    mean_rewards_path = os.path.join(result_dir, base + "_mean_rewards.txt")

    returns, successes, episodes = [], [], []
    ep_return, ep_length, ep_count = 0.0, 0, 0

    if env_info["continuous"]:
        agent = SimpleSACAgent(
            obs_dim=env_info["obs_dim"], act_dim=env_info["act_dim"],
            hidden=args.hidden_dim,
            lr=args.lr, gamma=args.gamma, batch_size=args.batch_size,
            start_steps=args.start_steps,
            aipr_interval=args.aipr_interval, aipr_iters=args.aipr_iters,
            device=device,
        )
    else:
        agent = SimplePPOAgent(
            obs_dim=env_info["obs_dim"], act_dim=env_info["act_dim"],
            continuous=False, hidden=args.hidden_dim,
            lr=args.lr, gamma=args.gamma,
            use_cnn=env_info.get("use_cnn", False),
            aipr_interval=args.aipr_interval, aipr_iters=args.aipr_iters,
            device=device,
        )

    obs, _ = env.reset()
    eval_returns = []

    if env_info["continuous"]:
        # SAC loop
        for step in range(1, args.total_steps + 1):
            if step < 10000:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.add(obs, action, reward, next_obs, float(terminated))
            agent.update(step)
            obs = next_obs
            ep_return += reward
            ep_length += 1

            if done:
                ep_count += 1
                returns.append(ep_return)
                successes.append(info.get("success", False))
                episodes.append({"episode": ep_count, "return": ep_return, "length": ep_length})
                mean_queue.append(ep_return)
                if ep_count % 100 == 0:
                    mean_val = np.mean(mean_queue)
                    with open(mean_rewards_path, "a", encoding="utf-8") as f:
                        f.write(f"{mean_val:.6f}\n")
                if step % args.log_interval < ep_length + 1:
                    print(f"[Step {step:>8d}] Ep {ep_count} | Return: {ep_return:.2f}")
                obs, _ = env.reset()
                ep_return, ep_length = 0.0, 0

            if step % args.eval_interval == 0:
                eval_r = _evaluate(agent, args.env, args.suite, seed)
                eval_returns.append(eval_r)
                print(f"[Eval @ {step:>8d}] Mean Return: {eval_r:.2f}")
    else:
        # PPO loop
        step = 0
        while step < args.total_steps:
            for _ in range(agent.n_steps):
                action, logp, val = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.store(obs, action, reward, float(done), val, logp)
                obs = next_obs
                ep_return += reward
                ep_length += 1
                step += 1

                if done:
                    ep_count += 1
                    returns.append(ep_return)
                    successes.append(info.get("success", False))
                    episodes.append({"episode": ep_count, "return": ep_return, "length": ep_length})
                    mean_queue.append(ep_return)
                    if ep_count % 100 == 0:
                        mean_val = np.mean(mean_queue)
                        with open(mean_rewards_path, "a", encoding="utf-8") as f:
                            f.write(f"{mean_val:.6f}\n")
                    if step % args.log_interval < ep_length + 5:
                        print(f"[Step {step:>8d}] Ep {ep_count} | Return: {ep_return:.2f}")
                    obs, _ = env.reset()
                    ep_return, ep_length = 0.0, 0

                if step >= args.total_steps:
                    break

            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                feat = agent._features(obs_t)
                lv = agent.critic(feat).item()
            agent.update(lv)

            if agent.aipr_interval > 0 and step % agent.aipr_interval == 0 and step > 0:
                agent.aipr_reinit()

            if step % args.eval_interval < agent.n_steps + 1:
                eval_r = _evaluate(agent, args.env, args.suite, seed)
                eval_returns.append(eval_r)
                print(f"[Eval @ {step:>8d}] Mean Return: {eval_r:.2f}")


    if len(mean_queue) > 0 and ep_count % 100 != 0:
        mean_val = np.mean(mean_queue)
        start_ep = ep_count - len(mean_queue) + 1
        end_ep = ep_count
        with open(mean_rewards_path, "a", encoding="utf-8") as f:
            f.write(f"{mean_val:.6f}\n")
    env.close()
    return returns, successes, episodes, None


def _evaluate(agent, env_name, suite, seed, n_episodes=10):
    eval_env, _ = make_env(env_name, suite, seed + 1000)
    rets = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done, ep_ret = False, 0.0
        while not done:
            if hasattr(agent, 'select_action'):
                action = agent.select_action(obs, deterministic=True)
                if isinstance(action, tuple):
                    action = action[0]
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_ret += reward
            done = terminated or truncated
        rets.append(ep_ret)
    eval_env.close()
    return float(np.mean(rets))


# ─── 主函数 ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="AIPR Unified Benchmark")
    parser.add_argument("--mode", type=str, default="rl_sac",
                        choices=["rl_sac", "rl_dqn", "rl_ppo", "vision"],
                        help="训练模式")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--suite", type=str, default=None,
                        choices=["atari", "ale", "mujoco", "dmcontrol", "gridworld",
                                 "carl", "carl_dmcquadruped", "carl_lunarlander",
                                 "metaworld", "robosuite", "humanoidbench",
                                 "cifar10", "cifar100"])
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=200, help="视觉任务 epoch 数")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="隐藏层维度 (与 AIPR 保持一致以公平对比)")
    parser.add_argument("--start_steps", type=int, default=10000,
                        help="SAC 随机探索步数 (与 AIPR 保持一致)")
    parser.add_argument("--aipr_interval", type=int, default=50000,
                        help="AIPR 重初始化间隔 (RL步数 / 视觉epoch)")
    parser.add_argument("--aipr_iters", type=int, default=10, help="Newton-Schulz 迭代次数")
    parser.add_argument("--eval_interval", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=5000)
    parser.add_argument("--result_dir", type=str, default="result")
    parser.add_argument("--normalize_factor", type=float, default=1000.0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def run_single_seed(args, seed):
    print("=" * 60)
    print(f"AIPR Benchmark | Mode: {args.mode} | Env: {args.env} | Seed: {seed}")
    print("=" * 60)

    t0 = time.time()

    suite = args.suite or detect_suite(args.env)
    is_vision = suite in ("cifar10", "cifar100") or args.mode == "vision"

    if is_vision:
        returns, successes, episodes, vision_metrics = train_vision_aipr(args, seed)
    else:
        returns, successes, episodes, vision_metrics = train_rl(args, seed)

    train_time = time.time() - t0
    print(f"\nTraining completed in {train_time:.1f}s")

    metrics = compute_final_metrics(returns, successes, args.normalize_factor, vision_metrics)
    config = vars(args).copy()
    config["seed"] = seed
    config["train_time_sec"] = train_time

    result_dir = os.path.join(args.result_dir, suite, args.mode)
    os.makedirs(result_dir, exist_ok=True)
    base = make_filename("AIPR", args.env, seed)
    txt_path = os.path.join(result_dir, base + ".txt")
    xlsx_path = os.path.join(result_dir, base + ".xlsx")

    save_results_txt(txt_path, metrics, config, episodes, save_individual_files=True)
    save_results_excel(xlsx_path, metrics, config, episodes, vision_metrics)

    print(f"\nResults: {txt_path}")
    print(f"         {xlsx_path}")
    return metrics


def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    seeds = args.seeds if args.seeds else [args.seed]

    all_metrics = []
    for seed in seeds:
        m = run_single_seed(args, seed)
        all_metrics.append(m)
        print()

    if len(seeds) > 1:
        print("=" * 60)
        print(f"Multi-seed Summary ({len(seeds)} seeds)")
        print("=" * 60)
        rets = [m.get("mean_return", 0) for m in all_metrics]
        print(f"  Mean Return: {np.mean(rets):.2f} ± {np.std(rets):.2f}")


if __name__ == "__main__":
    main()
