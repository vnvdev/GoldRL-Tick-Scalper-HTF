"""
config.py - Centralized configuration for XAUUSD Tick Scalping Environment & PPO Training.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

# Default multi-timeframe list (m1, m5, m15, h1)
DEFAULT_TF_LIST: List[Tuple[str, int]] = [
    ("m1", 1),
    ("m5", 5),
    ("m15", 15),
    ("h1", 60),
]


@dataclass
class EnvConfig:
    csv_path: str = "zero.csv"
    tf_list: List[Tuple[str, int]] = field(default_factory=lambda: DEFAULT_TF_LIST.copy())
    sl_tp_distance: float = 5.0        # Fixed Take Profit and Stop Loss distance (e.g. 5.0 USD)
    allow_early_close: bool = False    # When False, trade must hit TP or SL (disciplined 1:1 R:R)
    risk_value: float = 0.03           # 3% risk per trade for dynamic lot sizing
    initial_balance: float = 1000.0    # Initial starting balance in USD
    window_size: int = 138             # Number of historical bars per timeframe window
    warmup_lookback: int = 3_000_000   # Number of ticks used to warm up H4 indicators


@dataclass
class TrainConfig:
    timesteps: int = 20_000_000        # Total timesteps for PPO training
    n_envs: int = 32                   # Number of parallel subprocess workers
    learning_rate: float = 3e-4        # Optimal learning rate from JonusNattapong
    n_steps: int = 512                 # Timesteps per environment per rollout
    batch_size: int = 64               # Minibatch size for gradient descent
    gamma: float = 0.99                # Discount factor
    gae_lambda: float = 0.95           # Generalized Advantage Estimation lambda
    clip_range: float = 0.2            # PPO clipping ratio
    ent_coef: float = 0.0005           # Entropy coefficient to maintain aggressive exploration
    vf_coef: float = 0.5               # Value function loss coefficient
    max_grad_norm: float = 0.5         # Maximum gradient clipping norm
    save_dir: str = "models"
