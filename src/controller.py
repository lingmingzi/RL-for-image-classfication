from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(states)
        logits = self.actor(feat)
        values = self.critic(feat).squeeze(-1)
        return logits, values

    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value


@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    logprob: torch.Tensor
    reward: float
    value: torch.Tensor
    done: bool


class PPOController:
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        lr: float,
        gamma: float,
        gae_lambda: float,
        clip_eps: float,
        value_coef: float,
        entropy_coef: float,
        ppo_epochs: int,
        mini_batch_size: int,
        device: torch.device,
    ):
        self.net = ActorCritic(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.device = device
        self.memory: List[Transition] = []

    def select_action(self, state_vec: torch.Tensor) -> Tuple[int, float, float]:
        state = state_vec.to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, logprob, value = self.net.act(state)
        return int(action.item()), float(logprob.item()), float(value.item())

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def _compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
            next_value = values[t]
        returns = [a + v for a, v in zip(advantages, values)]
        return advantages, returns

    def update(self) -> dict:
        if not self.memory:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        states = torch.stack([m.state for m in self.memory]).to(self.device)
        actions = torch.tensor([m.action.item() for m in self.memory], dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor([m.logprob.item() for m in self.memory], dtype=torch.float32, device=self.device)
        rewards = [m.reward for m in self.memory]
        values = [m.value.item() for m in self.memory]
        dones = [m.done for m in self.memory]

        advantages, returns = self._compute_gae(rewards, values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(self.memory)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        step_count = 0

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.mini_batch_size):
                idx = perm[start:start + self.mini_batch_size]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_logprobs = old_logprobs[idx]
                batch_adv = advantages[idx]
                batch_ret = returns[idx]

                logits, value_pred = self.net(batch_states)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value_pred, batch_ret)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.item())
                step_count += 1

        self.memory.clear()
        if step_count > 0:
            for k in stats:
                stats[k] /= step_count
        return stats
