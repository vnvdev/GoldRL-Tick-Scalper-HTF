"""
utils.py - Utility functions for zero-copy tick caching and parallel multiprocessing environments.
"""

import os
import numpy as np
import polars as pl
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


def _tick_cache_paths(csv_path):
    base = str(csv_path)
    return {
        'time': base + '.time.npy',
        'bid': base + '.bid.npy',
        'ask': base + '.ask.npy'
    }


def build_tick_cache(csv_path, force=False):
    """
    Converts a raw tick CSV (time, bid, ask) into memory-mapped .npy files for zero-copy loading
    across multiple subprocess workers.
    """
    paths = _tick_cache_paths(csv_path)
    if not force and all(os.path.exists(p) for p in paths.values()):
        return paths
    
    print(f"[build_tick_cache] Converting {csv_path} to zero-copy memory-mapped .npy cache...")
    df = pl.scan_csv(
        csv_path,
        schema_overrides={'time': pl.Int64, 'bid': pl.Float32, 'ask': pl.Float32}
    ).select(['time', 'bid', 'ask']).collect()
    
    if not df['time'].is_sorted():
        df = df.sort('time')
        
    np.save(paths['time'], df['time'].to_numpy().astype(np.int64))
    np.save(paths['bid'],  df['bid'].to_numpy().astype(np.float32))
    np.save(paths['ask'],  df['ask'].to_numpy().astype(np.float32))
    return paths


def load_tick_cache(csv_path):
    """
    Loads zero-copy memory-mapped arrays (time, bid, ask) from disk.
    Multiple workers can read these arrays without multiplying memory usage.
    """
    paths = build_tick_cache(csv_path, force=False)
    times = np.load(paths['time'], mmap_mode='r')
    bids  = np.load(paths['bid'],  mmap_mode='r')
    asks  = np.load(paths['ask'],  mmap_mode='r')
    return times, bids, asks


def make_parallel_envs(csv_path, n_envs, env_kwargs=None, use_subprocess=True):
    """
    Splits the tick dataset into `n_envs` contiguous, non-overlapping segments.
    Each subprocess worker handles one segment independently using zero-copy memory mapping.
    """
    from xau_rl_scalp.env import XAUScalpEnv

    times, _, _ = load_tick_cache(csv_path)
    df_length = len(times)
    env_kwargs = dict(env_kwargs or {})
    warmup_lookback = env_kwargs.pop("warmup_lookback", 3_000_000)

    bounds = np.linspace(warmup_lookback, df_length - 1, n_envs + 1, dtype=np.int64)

    def _make(rank):
        start, end = int(bounds[rank]), int(bounds[rank + 1])

        def _init():
            return XAUScalpEnv(
                csv_path=csv_path,
                tick_range=(start, end),
                warmup_lookback=warmup_lookback,
                **env_kwargs,
            )
        return _init

    env_fns = [_make(r) for r in range(n_envs)]
    if use_subprocess and n_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)
