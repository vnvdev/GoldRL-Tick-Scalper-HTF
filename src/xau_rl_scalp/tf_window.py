"""
tf_window.py - High-Performance Multi-Timeframe Bar Builder & Technical Indicator Calculator.
"""

from collections import deque
import numpy as np


class TFWindow:
    """
    Builds historical bars (M1, M5, M15, H1, etc.) on the fly from incoming tick data
    and calculates real-time technical indicators (RSI, WMA, EMA21, EMA50) without lookahead bias.
    """
    __slots__ = (
        'tf_name', 'minutes', 'window_size', 'buf', 'ready',
        '_dirty', '_cur',
        '_close_final', '_rsi_final', '_ema_final', '_wma_final',
        '_rsi_len', '_ema_len', '_wma_len', '_rsi_alpha', '_ema_alpha',
        '_wma_weights', '_wma_weight_sum', '_gain_rma_last', '_loss_rma_last',
        '_high_final', '_low_final',
        '_ema21_len', '_ema50_len', '_ema21_final', '_ema50_final',
    )

    def __init__(self, tf_name: str, minutes: int, window_size: int = 150):
        self.tf_name     = tf_name
        self.minutes     = minutes
        self.window_size = window_size
        self.buf         = deque(maxlen=self.window_size - 1)
        self.ready       = False
        self._dirty      = True
        self._cur        = None

        self._close_final = np.array([], dtype=float)
        self._rsi_final   = np.array([], dtype=float)
        self._ema_final   = np.array([], dtype=float)
        self._wma_final   = np.array([], dtype=float)
        self._high_final  = np.array([], dtype=float)
        self._low_final   = np.array([], dtype=float)

        self._rsi_len = 14
        self._ema_len = 9
        self._wma_len = 45

        self._rsi_alpha      = 1.0 / self._rsi_len
        self._ema_alpha      = 2.0 / (self._ema_len + 1.0)
        self._wma_weights    = np.arange(1, self._wma_len + 1, dtype=float)
        self._wma_weight_sum = self._wma_weights.sum()

        self._gain_rma_last = np.nan
        self._loss_rma_last = np.nan

        self._ema21_len   = 21
        self._ema50_len   = 50
        self._ema21_final = np.array([], dtype=float)
        self._ema50_final = np.array([], dtype=float)

    @staticmethod
    def _ema_np(values, length):
        arr = np.asarray(values, dtype=float)
        out = np.full(arr.shape, np.nan, dtype=float)
        if length <= 0 or arr.size == 0 or arr.size < length:
            return out
        alpha       = 2.0 / (length + 1.0)
        seed_window = arr[:length]
        if np.isnan(seed_window).all():
            return out
        seed        = np.nanmean(seed_window)
        seed_i      = length - 1
        out[seed_i] = seed
        prev        = seed
        for i in range(seed_i + 1, arr.size):
            v = arr[i]
            if np.isnan(v):
                continue
            prev   = alpha * v + (1.0 - alpha) * prev
            out[i] = prev
        return out

    @staticmethod
    def _rsi_np(close, length=14):
        c   = np.asarray(close, dtype=float)
        rsi = np.full(c.shape, np.nan, dtype=float)
        if length <= 0 or c.size <= length:
            return rsi
        delta   = np.diff(c)
        pos     = np.full(c.shape, np.nan, dtype=float)
        neg     = np.full(c.shape, np.nan, dtype=float)
        pos[1:] = np.where(delta > 0,  delta, 0.0)
        neg[1:] = np.where(delta < 0, -delta, 0.0)

        def _rma(arr, n):
            out   = np.full(arr.shape, np.nan, dtype=float)
            alpha = 1.0 / n
            prev  = np.nan
            for i, v in enumerate(arr):
                if np.isnan(v):
                    continue
                prev   = v if np.isnan(prev) else alpha * v + (1.0 - alpha) * prev
                out[i] = prev
            return out

        avg_gain = _rma(pos, length)
        avg_loss = _rma(neg, length)
        den = avg_gain + avg_loss
        rsi = np.divide(
            100.0 * avg_gain, den,
            out=np.full(c.shape, np.nan, dtype=float),
            where=den != 0,
        )
        return rsi

    @staticmethod
    def _wma_np(values, length=45):
        arr = np.asarray(values, dtype=float)
        out = np.full(arr.shape, np.nan, dtype=float)
        if length <= 0 or arr.size < length:
            return out
        weights    = np.arange(1, length + 1, dtype=float)
        weight_sum = weights.sum()
        for i in range(length - 1, arr.size):
            win = arr[i - length + 1:i + 1]
            if np.isnan(win).any():
                continue
            out[i] = np.dot(win, weights) / weight_sum
        return out

    def _epoch(self, t: int) -> int:
        ms = self.minutes * 60 * 1000
        return (t // ms) * ms

    def _refresh_final_cache(self):
        if not self._dirty:
            return
        if not self.buf:
            self._close_final   = np.array([], dtype=float)
            self._rsi_final     = np.array([], dtype=float)
            self._ema_final     = np.array([], dtype=float)
            self._wma_final     = np.array([], dtype=float)
            self._high_final    = np.array([], dtype=float)
            self._low_final     = np.array([], dtype=float)
            self._ema21_final   = np.array([], dtype=float)
            self._ema50_final   = np.array([], dtype=float)
            self._gain_rma_last = np.nan
            self._loss_rma_last = np.nan
            self._dirty         = False
            return

        close = np.array([c['close'] for c in self.buf], dtype=float)
        high  = np.array([c['high']  for c in self.buf], dtype=float)
        low   = np.array([c['low']   for c in self.buf], dtype=float)
        rsi   = self._rsi_np(close, self._rsi_len)
        ema   = self._ema_np(rsi,   self._ema_len)
        wma   = self._wma_np(rsi,   self._wma_len)

        self._close_final = close
        self._high_final  = high
        self._low_final   = low
        self._rsi_final   = rsi
        self._ema_final   = ema
        self._wma_final   = wma

        self._ema21_final = self._ema_np(close, self._ema21_len)
        self._ema50_final = self._ema_np(close, self._ema50_len)

        gain = np.nan
        loss = np.nan
        if close.size >= 2:
            for i in range(1, close.size):
                d  = close[i] - close[i - 1]
                up = d  if d > 0 else 0.0
                dn = -d if d < 0 else 0.0
                if np.isnan(gain):
                    gain = up
                    loss = dn
                else:
                    gain = self._rsi_alpha * up + (1.0 - self._rsi_alpha) * gain
                    loss = self._rsi_alpha * dn + (1.0 - self._rsi_alpha) * loss

        self._gain_rma_last = gain
        self._loss_rma_last = loss
        self._dirty         = False

    def _current_indicators(self):
        self._refresh_final_cache()
        if self._cur is None:
            return np.nan, np.nan, np.nan
        n = self._close_final.size
        if n == 0:
            return np.nan, np.nan, np.nan

        cur_close  = float(self._cur['close'])
        prev_close = self._close_final[-1]

        d  = cur_close - prev_close
        up = d  if d > 0 else 0.0
        dn = -d if d < 0 else 0.0
        if np.isnan(self._gain_rma_last):
            gain = up
            loss = dn
        else:
            gain = self._rsi_alpha * up + (1.0 - self._rsi_alpha) * self._gain_rma_last
            loss = self._rsi_alpha * dn + (1.0 - self._rsi_alpha) * self._loss_rma_last
        den     = gain + loss
        rsi_cur = np.nan if den == 0 else 100.0 * gain / den

        rsi_count = self._rsi_final.size + 1
        if rsi_count < self._ema_len:
            ema_cur = np.nan
        elif rsi_count == self._ema_len:
            ema_cur = np.nanmean(np.append(self._rsi_final, rsi_cur))
        else:
            prev_ema = self._ema_final[-1]
            ema_cur  = np.nan if np.isnan(rsi_cur) else (
                self._ema_alpha * rsi_cur + (1.0 - self._ema_alpha) * prev_ema
            )

        if rsi_count < self._wma_len:
            wma_cur = np.nan
        else:
            tail_len = self._wma_len - 1
            win      = np.empty(self._wma_len, dtype=float)
            win[:tail_len] = self._rsi_final[-tail_len:]
            win[-1]  = rsi_cur
            wma_cur  = np.nan if np.isnan(win).any() else float(
                np.dot(win, self._wma_weights) / self._wma_weight_sum
            )

        return rsi_cur, ema_cur, wma_cur

    def _current_price_ema(self, length, final_arr):
        if self._cur is None:
            return np.nan
        n = self._close_final.size + 1
        if n < length:
            return np.nan
        cur_close = float(self._cur['close'])
        if n == length:
            if self._close_final.size == 0:
                return cur_close
            window = np.append(self._close_final[-(length - 1):], cur_close)
            return float(np.nanmean(window))
        prev = final_arr[-1]
        if np.isnan(prev):
            return np.nan
        alpha = 2.0 / (length + 1.0)
        return alpha * cur_close + (1.0 - alpha) * prev

    def current_ema21_ema50(self):
        self._refresh_final_cache()
        e21 = self._current_price_ema(self._ema21_len, self._ema21_final)
        e50 = self._current_price_ema(self._ema50_len, self._ema50_final)
        return e21, e50

    def current_close(self):
        return float(self._cur['close']) if self._cur is not None else np.nan

    def push(self, o: float, h: float, l: float, c: float, t: int):
        ep = self._epoch(t)

        if self._cur is None:
            self._cur  = {'ep': ep, 'open': o, 'high': h, 'low': l, 'close': c}
            self.ready = (len(self.buf) + 1) >= self.window_size
            return

        if self._cur['ep'] == ep:
            if h > self._cur['high']: self._cur['high']  = h
            if l < self._cur['low']:  self._cur['low']   = l
            self._cur['close'] = c
        else:
            self.buf.append(self._cur)
            self._dirty = True
            self._cur   = {'ep': ep, 'open': o, 'high': h, 'low': l, 'close': c}

        self.ready = (len(self.buf) + 1) >= self.window_size
