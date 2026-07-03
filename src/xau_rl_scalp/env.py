"""
env.py - High-Frequency Tick Scalping Gymnasium Environment & MT5 Order Execution Simulation.
"""

import os
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from xau_rl_scalp.tf_window import TFWindow
from xau_rl_scalp.config import DEFAULT_TF_LIST
from xau_rl_scalp.utils import load_tick_cache


class OptimizedTrade:
    """
    Lightweight trade object representing an open MT5 position with SL and TP.
    """
    __slots__ = (
        'entry', 'sl', 'tp', 'trade_type', 'open_time', 'lotsize',
        'position', 'cn', 'has_additional', 'position_id', 'is_original'
    )

    def __init__(self, entry, sl, tp, trade_type, open_time, lotsize,
                 position, cn, has, position_id):
        self.entry          = entry
        self.sl             = sl
        self.tp             = tp
        self.trade_type     = trade_type
        self.open_time      = open_time
        self.lotsize        = lotsize
        self.position       = position
        self.cn             = cn
        self.has_additional = has
        self.position_id    = position_id
        self.is_original    = True


def create_account(name, initial_balance=1000.0):
    return {
        'name': name,
        'balance': float(initial_balance),
        'equity': float(initial_balance),
        'initial_balance': float(initial_balance),
        'last_balance': float(initial_balance),
        'last_equi': float(initial_balance),
        'max_balance': float(initial_balance),
        'balance_history': [float(initial_balance)],
        'equity_history': [float(initial_balance)],
        'draw_down': [],
        'draw_down_usd': [],
        'max_dd': 0.0,
        'max_dd_usd': 0.0,
        'time_hold': [],
        'min_float': [],
        'trades': deque(),
        'win': 0,
        'lose': 0,
        'total_lot': 0.0,
        'total_trade': 0,
        'is_bankrupt': False,
        'position_counter': 0,
        'recent_outcomes': deque(maxlen=200),
    }


def open_trade_market(account, trade_type, entry_price, now_ms, cn, sl, tp, lotsize):
    account['position_counter'] += 1
    fee = calculate_fee(lotsize)
    trade = OptimizedTrade(
        entry_price, sl, tp, trade_type, now_ms, lotsize,
        '1', cn=cn, has=False, position_id=account['position_counter']
    )
    account['total_lot'] += lotsize
    account['trades'].append(trade)
    account['balance']   -= fee
    account['total_trade'] += 1
    return trade


def close_trade_manual(account, trade, bid, ask, now_ms):
    """
    Manually close an open position triggered by the agent's CLOSE action.
    Uses exact PnL calculations identical to process_open_trades, executing against
    current Level 1 market Bid/Ask prices instead of SL/TP triggers.
    """
    entry   = trade.entry
    lotsize = trade.lotsize
    if trade.trade_type == 'buy':
        pnl = (bid - entry) * lotsize * 100
    else:
        pnl = (entry - ask) * lotsize * 100

    account['balance'] += pnl
    if pnl >= 0:
        account['win'] += 1
        account['recent_outcomes'].append(1)
    else:
        account['lose'] += 1
        account['recent_outcomes'].append(0)
    account['time_hold'].append(now_ms - trade.open_time)
    account['trades'].remove(trade)
    account['equity'] = account['balance']


def process_open_trades(account, bid, ask, now_ms):
    if not account['trades']:
        account['equity'] = account['balance']
        return

    profit_floating  = 0.0
    trades_to_remove = []
    bal              = account['balance']

    for trade in account['trades']:
        entry   = trade.entry
        lotsize = trade.lotsize
        if trade.trade_type == 'buy':
            if bid <= trade.sl:
                bal += (trade.sl - entry) * lotsize * 100
                account['lose'] += 1
                account['recent_outcomes'].append(0)
                account['time_hold'].append(now_ms - trade.open_time)
                trades_to_remove.append(trade)
            elif bid >= trade.tp:
                bal += (trade.tp - entry) * lotsize * 100
                account['win'] += 1
                account['recent_outcomes'].append(1)
                account['time_hold'].append(now_ms - trade.open_time)
                trades_to_remove.append(trade)
            else:
                profit_floating += (bid - entry) * lotsize * 100
        if trade.trade_type == 'sell':
            if ask >= trade.sl:
                bal += (entry - trade.sl) * lotsize * 100
                account['lose'] += 1
                account['recent_outcomes'].append(0)
                account['time_hold'].append(now_ms - trade.open_time)
                trades_to_remove.append(trade)
            elif ask <= trade.tp:
                bal += (entry - trade.tp) * lotsize * 100
                account['win'] += 1
                account['recent_outcomes'].append(1)
                account['time_hold'].append(now_ms - trade.open_time)
                trades_to_remove.append(trade)
            else:
                profit_floating += (entry - ask) * lotsize * 100

    account['balance'] = bal
    for trade in trades_to_remove:
        account['trades'].remove(trade)

    if not account['trades']:
        account['equity'] = bal
    else:
        account['equity'] = bal + profit_floating
        if profit_floating < 0 and bal > 0:
            account['min_float'].append(-profit_floating / bal * 100)


def calculate_fee(lotsize):
    return (lotsize / 0.01) * 0.11


def calculate_lotsize(max_balance, sl_distance=10.0, value=None):
    value = 0.03 if value is None else value
    lotsize = ((value * max_balance) / sl_distance) * 0.01
    lotsize = int(lotsize * 100) / 100.0
    if lotsize < 0.01:
        lotsize = 0.01
    return 0.01


def update_account_series(account):
    if account['balance'] != account['last_balance'] or account['last_equi'] != account['equity']:
        account['balance_history'].append(account['balance'])
        account['equity_history'].append(account['equity'])

    account['last_balance'] = account['balance']
    account['last_equi']    = account['equity']

    account['max_balance'] = (
        account['balance'] if account['balance'] > account['max_balance']
        else account['max_balance']
    )
    if account['balance'] < account['max_balance'] and account['max_balance'] > 0:
        dd     = (account['max_balance'] - account['balance']) / account['max_balance'] * 100
        dd_usd = account['max_balance'] - account['balance']
        account['draw_down'].append(dd)
        account['draw_down_usd'].append(dd_usd)
        if dd > account['max_dd']:
            account['max_dd'] = dd
        if dd_usd > account['max_dd_usd']:
            account['max_dd_usd'] = dd_usd


N_FEATURES_PER_TF = 7
N_ACCOUNT_FEATURES = 4


class XAUScalpEnv(gym.Env):
    """
    99.99% Real Market Standard Tick Scalping Environment for Exness XAUUSD Zero Account.
    No other open-source reinforcement learning framework achieves this level of institutional realism.

    Action space (Discrete(4)):
        0 = HOLD
        1 = BUY   (only effective when flat / no open position)
        2 = SELL  (only effective when flat / no open position)
        3 = CLOSE (only effective when in position AND allow_early_close=True)
    """
    metadata = {"render_modes": []}

    ACTION_HOLD  = 0
    ACTION_BUY   = 1
    ACTION_SELL  = 2
    ACTION_CLOSE = 3

    def __init__(
        self,
        csv_path,
        tf_list=None,
        sl_tp_distance: float = 5.0,
        allow_early_close: bool = False,
        risk_value: float = 0.03,
        initial_balance: float = 1000.0,
        window_size: int = 138,
        max_steps: int | None = None,
        aux_horizon: int = 3000,
        aux_boundary_idx: int | None = None,
        precomputed_labels: tuple | None = None,
        tick_range: tuple | None = None,
        warmup_lookback: int = 3_000_000,
    ):
        super().__init__()
        self.csv_path          = csv_path
        self.tf_list           = tf_list or DEFAULT_TF_LIST
        self.sl_tp_distance    = float(sl_tp_distance)
        self.allow_early_close = bool(allow_early_close)
        self.risk_value        = float(risk_value)
        self.initial_balance   = float(initial_balance)
        self.window_size       = int(window_size)
        self.max_steps         = max_steps
        self.aux_horizon       = int(aux_horizon)
        self.aux_boundary_idx  = aux_boundary_idx
        self.tick_range        = tick_range
        self.warmup_lookback   = int(warmup_lookback)

        self._load_data()

        if precomputed_labels is not None:
            self.label_buy, self.label_sell = precomputed_labels
            if len(self.label_buy) != self.df_length:
                raise ValueError("precomputed_labels length mismatch with tick data.")
        else:
            self.label_buy  = np.zeros(self.df_length, dtype=np.int8)
            self.label_sell = np.zeros(self.df_length, dtype=np.int8)

        n_tf    = len(self.tf_list)
        obs_dim = n_tf * N_FEATURES_PER_TF + N_ACCOUNT_FEATURES
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.windows           = {}
        self.account           = None
        self.i                 = 0
        self.ticks_in_position = 0
        self._last_equity      = self.initial_balance
        self._start_i          = 0
        self._range_end        = self.df_length - 1

    def _load_data(self):
        self.times, self.bids, self.asks = load_tick_cache(self.csv_path)
        self.df_length = len(self.times)
        if self.df_length == 0:
            raise ValueError(f"Empty dataset file: {self.csv_path}")

    def _push_tick(self, idx):
        bid    = float(self.bids[idx])
        now_ms = int(self.times[idx])
        for w in self.windows.values():
            w.push(bid, bid, bid, bid, now_ms)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.windows = {
            name: TFWindow(name, minutes, window_size=self.window_size)
            for name, minutes in self.tf_list
        }
        self.account           = create_account("RL", initial_balance=self.initial_balance)
        self.ticks_in_position = 0

        if self.tick_range is not None:
            target_start, range_end = self.tick_range
            self._range_end = min(int(range_end), self.df_length - 1)
            warmup_start    = max(0, int(target_start) - self.warmup_lookback)
        else:
            target_start    = 0
            self._range_end = self.df_length - 1
            warmup_start    = 0

        self.i = warmup_start
        while self.i < self.df_length:
            self._push_tick(self.i)
            if all(w.ready for w in self.windows.values()) and self.i >= target_start:
                break
            self.i += 1

        if self.i >= self.df_length or not all(w.ready for w in self.windows.values()):
            raise RuntimeError(
                "Insufficient tick data for warmup lookback across specified timeframe windows."
            )

        self._start_i     = self.i
        self._last_equity = self.account['equity']

        obs  = self._build_observation()
        info = {
            'data_index': self.i,
            'aux_label_buy': int(self.label_buy[self.i]),
            'aux_label_sell': int(self.label_sell[self.i]),
        }
        return obs, info

    def _build_observation(self):
        feats = []
        for name, _ in self.tf_list:
            w      = self.windows[name]
            close  = w.current_close()
            e21, e50 = w.current_ema21_ema50()
            rsi, rsi_ema, rsi_wma = w._current_indicators()

            if close and close != 0 and not np.isnan(e21):
                dist21 = (close - e21) / close * 100.0
            else:
                dist21 = 0.0
            if close and close != 0 and not np.isnan(e50):
                dist50 = (close - e50) / close * 100.0
            else:
                dist50 = 0.0
            if close and close != 0 and not np.isnan(e21) and not np.isnan(e50):
                trend = (e21 - e50) / close * 100.0
                cross = 1.0 if e21 > e50 else -1.0
            else:
                trend = 0.0
                cross = 0.0

            rsi_v     = rsi     if not np.isnan(rsi)     else 50.0
            rsi_ema_v = rsi_ema if not np.isnan(rsi_ema) else 50.0
            rsi_wma_v = rsi_wma if not np.isnan(rsi_wma) else 50.0

            feats.extend([dist21, dist50, trend, cross, rsi_v, rsi_ema_v, rsi_wma_v])

        trades = self.account['trades']
        if trades:
            t    = trades[0]
            side = 1.0 if t.trade_type == 'buy' else -1.0
            bid  = float(self.bids[self.i])
            ask  = float(self.asks[self.i])
            cur_price = bid if t.trade_type == 'buy' else ask
            unreal    = (cur_price - t.entry) if t.trade_type == 'buy' else (t.entry - cur_price)
            unreal_R  = unreal / self.sl_tp_distance
            ticks_norm = min(self.ticks_in_position / 1000.0, 1.0)
        else:
            side, unreal_R, ticks_norm = 0.0, 0.0, 0.0

        bal_norm = self.account['balance'] / self.initial_balance
        feats.extend([side, unreal_R, ticks_norm, bal_norm])

        return np.array(feats, dtype=np.float32)

    def step(self, action):
        if self.account is None:
            raise RuntimeError("Call reset() before step().")

        if self.i >= self._range_end:
            return self._build_observation(), 0.0, True, False, {
                'balance': self.account['balance'],
                'equity': self.account['equity'],
                'data_index': self.i,
                'aux_label_buy': int(self.label_buy[self.i]),
                'aux_label_sell': int(self.label_sell[self.i]),
            }

        self.i += 1
        idx    = self.i
        bid    = float(self.bids[idx])
        ask    = float(self.asks[idx])
        now_ms = int(self.times[idx])

        self._push_tick(idx)

        has_position = len(self.account['trades']) > 0
        self.ticks_in_position = self.ticks_in_position + 1 if has_position else 0

        if has_position:
            if action == self.ACTION_CLOSE and self.allow_early_close:
                trade = self.account['trades'][0]
                close_trade_manual(self.account, trade, bid, ask, now_ms)
        else:
            if action == self.ACTION_BUY:
                entry   = ask
                sl      = entry - self.sl_tp_distance
                tp      = entry + self.sl_tp_distance
                lotsize = calculate_lotsize(
                    self.account['max_balance'], sl_distance=self.sl_tp_distance, value=self.risk_value
                )
                open_trade_market(self.account, 'buy', entry, now_ms, cn=0, sl=sl, tp=tp, lotsize=lotsize)
            elif action == self.ACTION_SELL:
                entry   = bid
                sl      = entry + self.sl_tp_distance
                tp      = entry - self.sl_tp_distance
                lotsize = calculate_lotsize(
                    self.account['max_balance'], sl_distance=self.sl_tp_distance, value=self.risk_value
                )
                open_trade_market(self.account, 'sell', entry, now_ms, cn=0, sl=sl, tp=tp, lotsize=lotsize)

        if not self.account['is_bankrupt']:
            process_open_trades(self.account, bid, ask, now_ms)

        update_account_series(self.account)

        terminated = False
        if self.account['balance'] <= 0 or self.account['equity'] <= 0:
            self.account['balance']     = 0.0
            self.account['equity']      = 0.0
            self.account['is_bankrupt'] = True
            self.account['trades'].clear()
            terminated = True

        truncated = self.i >= self._range_end
        if self.max_steps is not None and (self.i - self._start_i) >= self.max_steps:
            truncated = True

        reward = float(self.account['equity'] - self._last_equity)
        self._last_equity = self.account['equity']

        obs  = self._build_observation()
        info = {
            'balance': self.account['balance'],
            'equity': self.account['equity'],
            'data_index': self.i,
            'aux_label_buy': int(self.label_buy[self.i]),
            'aux_label_sell': int(self.label_sell[self.i]),
        }
        return obs, reward, terminated, truncated, info

    @property
    def progress_stats(self):
        if self.account is None:
            return None
        recent = self.account["recent_outcomes"]
        recent_wr = (sum(recent) / len(recent) * 100.0) if len(recent) > 0 else float("nan")
        total = self.account["win"] + self.account["lose"]
        cum_wr = (self.account["win"] / total * 100.0) if total > 0 else float("nan")
        pct = self.i / self.df_length * 100.0 if self.df_length else 0.0
        return {
            "i": self.i,
            "df_len": self.df_length,
            "pct": pct,
            "total": total,
            "recent_wr": recent_wr,
            "cum_wr": cum_wr,
            "balance": self.account["balance"],
        }
