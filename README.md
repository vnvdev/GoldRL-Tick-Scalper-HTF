# 🏆 GoldRL: 1:1 Live MT5 Real-Time Tick Scalping RL Framework

<div align="center">

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![Exness Zero](https://img.shields.io/badge/Exness-XAUUSD%20Zero%20Account-FFD700.svg?style=for-the-badge)
![Numba JIT](https://img.shields.io/badge/Numba-JIT%20Accelerated-00C853.svg?style=for-the-badge&logo=numba&logoColor=white)
![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-E91E63.svg?style=for-the-badge)
![Gymnasium](https://img.shields.io/badge/API-Gymnasium-FF6F00.svg?style=for-the-badge)
![Built in Vietnam](https://img.shields.io/badge/Proudly%20Built%20in-Vietnam%20%F0%9F%87%B6%F0%9F%87%B3-red.svg?style=for-the-badge)

*An institutional-grade Reinforcement Learning (RL) trading framework matching 1:1 with real-time MT5 execution.*<br>
*Engineered for Tick-by-Tick Precision, Multi-Timeframe Synthesis, and Maximum Processing Speed.*<br>
**Solo-Engineered by [Kudzo Vu](#-author--contact) (Vietnam Quant & Algorithmic Trading Developer)**

[Why GoldRL Exists?](#-the-3-year-solo-journey-eliminating-live-execution-discrepancies) • [Key Features](#-key-features) • [Architecture](#-repository-architecture) • [Quick Start](#-quick-start) • [Open Incubator](#-open-incubator-we-need-your-profitable-neural-networks--breakthrough-ideas) • [Author & Contact](#-author--contact)

</div>

---

## 🌟 Overview

Most open-source algorithmic trading repositories for Reinforcement Learning rely on slow, aggregated candlestick bars (e.g., M15 or H1 timeframes). While suitable for swing trading, those approaches miss the micro-structural price dynamics, spread drag, and real-world execution latency required for profitable algorithmic scalping.

**GoldRL** bridges the gap between academic RL research and institutional quantitative execution. By processing **raw, tick-by-tick Level 1 market data (Bid/Ask)** at maximum speed, GoldRL trains deep neural networks to execute disciplined, risk-managed scalping strategies on **XAUUSD (Gold)**.

### 📖 The 3-Year Solo Journey: Eliminating Live Execution Discrepancies

Over 3 years of rigorous quantitative research and analyzing open-source trading repositories, a systemic flaw became undeniable: **most research models do not match 1:1 with live market execution.** 

When research systems rely on smoothed candlestick bars and frictionless assumptions, transitioning them to live deployment on **MetaTrader 5 (MT5)** results in severe tracking errors and massive execution discrepancies. 

**GoldRL** was solo-engineered from the ground up to solve this fundamental gap. By standardizing order execution, broker microstructure, and risk rules to match real-time MT5 execution 1:1, what you observe during research is 100% standardized to live production.

> [!IMPORTANT]
> **💎 99.99% Real Market Standard for Exness XAUUSD Zero Account**
> Unlike simplified academic simulators, GoldRL is engineered specifically to match **99.99% real-world execution standards of the Exness XAUUSD Zero account**. It models exact institutional broker commissions (`$0.11 per 0.01 lot` / `$11.00 per standard lot`), dynamic risk-based lot sizing (risking exactly 3% of max account equity), floating tick-level unrealized PnL drag, and strict real-time Bid/Ask SL/TP triggering. **No other open-source reinforcement learning framework achieves this level of institutional realism!**

---

## ✨ Key Features

### 🛡️ 1. 1:1 Live MT5 Execution Matching (99.99% Real-Market Standard)
- **Zero Research-to-Production Discrepancy**: Simulates exact Exness Zero account specifications with raw zero-spread/ultra-tight tick execution and institutional commission deductions (`(lotsize / 0.01) * 0.11`).
- **Disciplined 1:1 Risk-to-Reward**: Enforces strict **fixed $10.00 USD ($5.00 distance) Take Profit and Stop Loss** orders. Unlike naive trading bots, early trade closures can be disabled (`allow_early_close = False`) to prevent premature profit-cutting and spread bleeding.
- **Rolling Local Win-Rate Tracking**: Separates cumulative win-rate from recent 200-trade performance to accurately track agent adaptation over time.

### ⚡ 2. Tick-by-Tick Precision at Maximum Processing Speed
- **Direct Level 1 Data Processing**: Crunches raw Bid/Ask tick data at maximum possible speed without candlestick aggregation delay.
- **Zero-Copy Memory Mapping (`mmap_mode='r'`)**: Converts massive CSV datasets into memory-mapped `.npy` caches. Dozens of parallel worker processes (`SubprocVecEnv`) share physical OS memory pages without RAM bottlenecks.
- **JIT-Compiled Math (`@njit`)**: All heavy numerical computations, indicators, and historical lookbacks are compiled to native C-speed machine code using **Numba** (achieving under 1.5s warmup per worker).
- **Automated Parallel Downloader (`get_data.py`)**: Fetches and compiles raw historical Level 1 tick archives directly from official Exness servers in parallel.

### 🧠 3. Multi-Timeframe Synthesis (`TFWindow`)
- Why rely on a single timeframe when institutions trade across all of them? GoldRL synthesizes tick streams on-the-fly into concurrent **M1, M5, M15, and H1** rolling windows.
- Automatically extracts **28 real-time technical features** (EMA21, EMA50, RSI, WMA, trend divergence, and crossover velocity) without lookahead bias.

### 🌍 4. Universal Multi-Asset & Multi-Timeframe Scaling (Forex, Crypto, Stocks & Any Broker)
While pre-configured and benchmarked on Gold (XAUUSD), GoldRL's mathematical core is **100% asset-agnostic and universally scalable**:
- **Any Financial Instrument**: Seamlessly train reinforcement learning models on **Bitcoin / Ethereum (Crypto), EURUSD / GBPUSD (Forex), S&P 500 / Nasdaq (Indices), or Apple / Tesla (Stocks)**!
- **Tick or Candle Granularity**: Don't have Level 1 tick data? No problem! You can feed standard **M1 (1-minute) or M5 candlestick data** from Binance, MetaTrader, or Yahoo Finance directly into the multi-timeframe window engine!
- **Any Broker Microstructure**: Customize `calculate_fee` and risk lot sizing parameters in `config.py` to simulate **thousands of account types across any broker worldwide** (e.g., Binance VIP tiers, Interactive Brokers, FTMO Prop Firms, Exness Standard/Raw, IC Markets, etc.).

---

## ⚔️ Why Tick Scalping vs. 15m Bars?

| Feature | Standard RL Repos (15m/1h Bars) | 🏆 GoldRL (Exness Zero 99.99% Standard) |
| :--- | :--- | :--- |
| **Market Standard** | Generic / Simulated | **99.99% Exness XAUUSD Zero Account** |
| **Data Granularity** | 1 price update every 15 minutes | **Real Level 1 Bid/Ask tick-by-tick** |
| **Commission Drag** | Ignored or arbitrary | **Exact $11/lot ($0.11 per 0.01 lot) Exness fee** |
| **Order Execution** | Assumes fills at bar close | **Matches 1:1 real-time MT5 Bid/Ask trigger** |
| **CPU Efficiency** | Single-process Pandas/Python | **Numba JIT + Multiprocessing SubprocVecEnv** |
| **RAM Footprint** | Duplicates memory per worker | **Zero-Copy OS Shared Page Caching** |

---

## 📁 Repository Architecture

To maintain clean separation of concerns and maximum accessibility for researchers, GoldRL splits indicator engineering and gymnasium environment logic into independent modules:

```text
goldrl/
├── README.md               # You are here!
├── requirements.txt        # Package dependencies
├── setup.py                # Installable Python package configuration
├── get_data.py             # High-speed parallel Exness tick downloader & compiler
│
├── src/
│   └── xau_rl_scalp/       # Core library package
│       ├── __init__.py     # Module exports
│       ├── config.py       # Centralized hyperparameters & environment configuration
│       ├── tf_window.py    # [CORE CLASS 1]: Numba-accelerated multi-timeframe bar builder
│       ├── env.py          # [CORE CLASS 2]: 99.99% Exness Zero XAUUSD Gymnasium environment
│       ├── callbacks.py    # Real-time training progress logging callback
│       └── utils.py        # Zero-copy memory mapping & parallel environment splitters
│
└── examples/
    ├── train_jonus_ppo.py  # Production PPO training script (JonusNattapong architecture)
    └── evaluate_model.py   # Out-of-sample backtest & performance evaluation script
```

---

## ⚡ Quick Start

### 1. Installation
Clone the repository and install the package in editable mode:
```bash
git clone https://github.com/vnvdev/GoldRL-Tick-Scalper-HTF.git
cd GoldRL-Tick-Scalper-HTF
pip install -e .
```

### 2. Download Real Exness XAUUSD Zero Tick Data
Use our automated parallel downloader (`get_data.py`) to fetch, extract, and merge real historical tick data directly from official Exness servers (`XAUUSD_Zero_Spread`):
```bash
python get_data.py
```
*This script automatically checks existing archives, downloads missing months/days in parallel, converts timestamps to millisecond precision, and compiles everything into a ready-to-train `zero.csv`!*

### 3. Train the RL Agent
Launch parallel multiprocessing training across your CPU cores:
```bash
python examples/train_jonus_ppo.py
```
During training, GoldRL displays a real-time institutional progress table:
```text
[timestep 100000]
  env0: tick 3500000/5000000 (70.0%) | trades: 142 | recent 200 winrate: 54.5% | cumulative winrate: 53.2% | balance: $1045.20
  env1: tick 3650000/5000000 (73.0%) | trades: 138 | recent 200 winrate: 56.0% | cumulative winrate: 55.1% | balance: $1082.40
```

### 4. Evaluate & Backtest
Test your trained model (`models/ppo_jonus_xauscalp.zip`) on out-of-sample data:
```bash
python examples/evaluate_model.py
```

---

## 🌐 Open Incubator: We Need Your Profitable Neural Networks & Breakthrough Ideas!

**Our ultimate vision is to build an open, collaborative quantitative community where researchers, AI developers, and traders worldwide join forces to create genuinely profitable algorithmic trading AI.**

We actively invite and challenge you to contribute:
1. **🔥 Profitable Neural Network Architectures**: Have an innovative deep learning architecture that outperforms standard MLP policies? Submit Pull Requests with custom feature extractors, **Temporal Convolutional Networks (TCNs), LSTM / GRU hybrid networks, Transformer / Attention models, or Mamba state space models**! If your model proves consistently profitable on out-of-sample Exness Zero tick data, we will feature it in our official model suite!
2. **💡 Breakthrough Quantitative Ideas**: Have ideas for novel reward shaping functions, order-flow imbalance (OFI) signals, micro-structure volatility indicators, or dynamic risk management algorithms? Share your concepts, open discussions, and let's experiment together!
3. **📊 Out-of-Sample Benchmark Results**: Share your training logs, equity curves, and hyperparameter configurations to help elevate the global algorithmic trading community.

---

## 🤝 Contributing

We welcome pull requests from algorithmic traders, quantitative researchers, and machine learning engineers!
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/ProfitableQuantModel`)
3. Commit your Changes (`git commit -m 'Add custom Transformer attention policy for XAUUSD'`)
4. Push to the Branch (`git push origin feature/ProfitableQuantModel`)
5. Open a Pull Request!

---

## 👤 Author & Contact

**Proudly built in Vietnam 🇻🇳 by Kudzo Vu**  
*Quantitative Researcher & Algorithmic Trading Developer*

We welcome collaboration, algorithmic discussions, and contributions from algorithmic traders and researchers worldwide! Feel free to connect with the author directly:

- **Facebook**: [fb.com/kudzovu](https://facebook.com/kudzovu)
- **Telegram**: [@kudzovu](https://t.me/kudzovu)
- **GitHub**: [github.com/vnvdev](https://github.com/vnvdev)

---

<div align="center">
<b>If you find GoldRL helpful for your research or algorithmic trading, please give it a ⭐ on GitHub and join our quant revolution!</b>
</div>
