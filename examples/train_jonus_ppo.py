"""
train_jonus_ppo.py - Train an XAUUSD High-Frequency Tick Scalping RL Agent
using standard Stable-Baselines3 PPO (MlpPolicy) & JonusNattapong's Optimal Hyperparameters.
"""

import os
import sys
import warnings

# Add parent directory to sys.path so we can import xau_rl_scalp locally without installing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
from stable_baselines3 import PPO
from xau_rl_scalp import (
    XAUScalpEnv,
    make_parallel_envs,
    load_tick_cache,
    TrainingProgressCallback,
    EnvConfig,
    TrainConfig,
)

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
CSV_PATH = "zero.csv"
N_ENVS = 3                     # Number of parallel worker subprocesses (e.g., 3 to 32)
TOTAL_TIMESTEPS = 20_000_000   # Total training timesteps

# PPO Hyperparameters adopted from JonusNattapong's Gold Trading research
LEARNING_RATE = 3e-4
N_STEPS = 512                  # 512 steps per worker per update cycle for rapid adaptation
BATCH_SIZE = 64
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.0005              # 0.0005 ensures decisive trade execution without passive flat holding
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Standard 2-layer [64, 64] MLP architecture for separate Policy (pi) and Value (vf) networks
POLICY_KWARGS = dict(
    net_arch=dict(pi=[64, 64], vf=[64, 64])
)


def main():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Dataset '{CSV_PATH}' not found!")
        print("Please run `python get_data.py --synthetic` first to generate benchmark tick data.")
        return

    print("=" * 70)
    print("STARTING PPO TICK SCALPING TRAINING (JONUS NATTAPONG ARCHITECTURE)")
    print("=" * 70)

    print("Loading tick dataset into zero-copy shared memory cache...")
    times, _, _ = load_tick_cache(CSV_PATH)
    df_len = len(times)
    print(f"Total Ticks: {df_len:,} | Parallel Workers: {N_ENVS}")

    # Pass dummy zero labels to completely bypass auxiliary hindsight calculations for standard PPO
    dummy_labels = (np.zeros(df_len, dtype=np.int8), np.zeros(df_len, dtype=np.int8))

    env_config = EnvConfig(csv_path=CSV_PATH)
    env_kwargs = dict(
        tf_list=env_config.tf_list,
        sl_tp_distance=env_config.sl_tp_distance,
        allow_early_close=env_config.allow_early_close,
        risk_value=env_config.risk_value,
        initial_balance=env_config.initial_balance,
        window_size=env_config.window_size,
        warmup_lookback=env_config.warmup_lookback,
        precomputed_labels=dummy_labels,
    )

    print(f"\nInitializing {N_ENVS} parallel subprocess environments...")
    vec_env = make_parallel_envs(
        csv_path=CSV_PATH,
        n_envs=N_ENVS,
        env_kwargs=env_kwargs,
        use_subprocess=True,
    )

    print("\nBuilding PPO model (MlpPolicy)...")
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=POLICY_KWARGS,
        verbose=1,
    )

    print("\nStarting reinforcement learning training... (Progress log every 2000 steps)")
    callback = TrainingProgressCallback(print_freq=2000)

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
    finally:
        os.makedirs("models", exist_ok=True)
        save_path = "models/ppo_jonus_xauscalp.zip"
        model.save(save_path)
        print(f"\n[OK] Model successfully saved to: {save_path}")
        vec_env.close()


if __name__ == "__main__":
    main()
