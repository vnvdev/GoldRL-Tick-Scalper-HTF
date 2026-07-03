"""
evaluate_model.py - Backtest & Evaluate a Trained PPO Tick Scalping Model on XAUUSD Data.
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
from stable_baselines3 import PPO
from xau_rl_scalp import XAUScalpEnv, load_tick_cache, EnvConfig


def main():
    model_path = "models/ppo_jonus_xauscalp.zip"
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model '{model_path}' not found!")
        print("Please run `python examples/train_jonus_ppo.py` first to train a model.")
        return

    csv_path = "zero.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] Dataset '{csv_path}' not found!")
        return

    print("=" * 70)
    print("EVALUATING TRAINED PPO AGENT ON XAUUSD TICK SCALPING")
    print("=" * 70)

    print("Loading tick dataset...")
    times, _, _ = load_tick_cache(csv_path)
    df_len = len(times)
    
    # Use the last 20% of tick data as an out-of-sample evaluation test segment
    test_start = int(df_len * 0.8)
    test_end = df_len - 1
    print(f"Out-of-Sample Test Range: Tick {test_start:,} -> {test_end:,} ({test_end - test_start:,} ticks)")

    dummy_labels = (np.zeros(df_len, dtype=np.int8), np.zeros(df_len, dtype=np.int8))
    env_config = EnvConfig(csv_path=csv_path)

    env = XAUScalpEnv(
        csv_path=csv_path,
        tf_list=env_config.tf_list,
        sl_tp_distance=env_config.sl_tp_distance,
        allow_early_close=env_config.allow_early_close,
        risk_value=env_config.risk_value,
        initial_balance=env_config.initial_balance,
        window_size=env_config.window_size,
        tick_range=(test_start, test_end),
        warmup_lookback=env_config.warmup_lookback,
        precomputed_labels=dummy_labels,
    )

    print(f"\nLoading trained model from '{model_path}'...")
    model = PPO.load(model_path, env=env)

    obs, info = env.reset()
    done = False
    step_count = 0

    print("\nRunning tick-by-tick simulation...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        
        if step_count % 500_000 == 0:
            stats = env.progress_stats
            if stats:
                print(f"  [Simulating] Tick {stats['i']:,}/{stats['df_len']:,} | Trades: {stats['total']} | "
                      f"Winrate: {stats['cum_wr']:.1f}% | Balance: ${stats['balance']:.2f}")

    stats = env.progress_stats
    print("\n" + "=" * 70)
    print("FINAL OUT-OF-SAMPLE EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Total Ticks Processed : {step_count:,}")
    print(f"  Total Trades Taken    : {stats['total']}")
    print(f"  Win / Loss            : {env.account['win']} Wins / {env.account['lose']} Losses")
    print(f"  Cumulative Win Rate   : {stats['cum_wr']:.2f}%")
    print(f"  Initial Balance       : ${env.initial_balance:.2f}")
    print(f"  Final Balance         : ${stats['balance']:.2f}")
    print(f"  Total Profit / Loss   : ${stats['balance'] - env.initial_balance:+.2f} "
          f"({(stats['balance'] - env.initial_balance) / env.initial_balance * 100:+.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
