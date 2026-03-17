from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    num_workers: int = 4
    batch_size: int = 128
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    max_epochs: int = 100
    seed: int = 42
    device: str = "cuda"


@dataclass
class RLConfig:
    state_dim: int = 6
    hidden_dim: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    ppo_epochs: int = 4
    mini_batch_size: int = 32
    update_every: int = 8
    episode_epochs: int = 1
    reward_baseline_decay: float = 0.99
    reward_w_acc: float = 1.0
    reward_w_loss: float = 0.5
    reward_w_complexity: float = 0.1
