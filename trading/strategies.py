"""
Trading strategies for the Kronos trading bot.

Each strategy implements a `decide` method that returns a list of trade signals.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import KronosPredictor

# ─── Trade Signal ─────────────────────────────────────────────────────────


class TradeSignal:
    """A single trade signal (BUY, SELL, or HOLD)."""

    def __init__(self, symbol: str, action: str, price: float,
                 confidence: float = 1.0, reason: str = ""):
        self.symbol = symbol
        self.action = action  # "BUY", "SELL", "HOLD"
        self.price = price
        self.confidence = confidence  # 0.0 ~ 1.0
        self.reason = reason
        self.timestamp = datetime.now()

    def __repr__(self):
        return (f"[{self.timestamp.strftime('%H:%M:%S')}] "
                f"{self.action:4s} {self.symbol:10s} @ ${self.price:.2f} "
                f"({self.confidence:.0%} confidence) — {self.reason}")


# ─── Abstract Base ────────────────────────────────────────────────────────


class Strategy(ABC):
    """Base class for trading strategies."""

    def __init__(self, config):
        self.config = config
        self.prices: Dict[str, pd.DataFrame] = {}  # cached price data

    @abstractmethod
    def name(self) -> str:
        ...

    def on_price_update(self, symbol: str, df: pd.DataFrame):
        """Called when new price data arrives."""
        self.prices[symbol] = df

    @abstractmethod
    def decide(self) -> List[TradeSignal]:
        """Return list of trade signals based on current data."""
        ...


# ─── Strategy 1: Prediction-based ─────────────────────────────────────────


class PredictionStrategy(Strategy):
    """Use Kronos model predictions to generate buy/sell signals.

    Logic:
        - Feed last N lookback candles into Kronos → predict next M candles
        - Compare predicted close to current close:
            - If predicted close >= current * (1 + buy_threshold) → BUY
            - If predicted close <= current * (1 + sell_threshold) → SELL
    """

    def __init__(self, config, predictor: 'KronosPredictor'):
        super().__init__(config)
        self.predictor = predictor
        self.last_prediction: Dict[str, pd.DataFrame] = {}
        self._last_pred_time: Dict[str, datetime] = {}

    def name(self) -> str:
        return "prediction"

    def _should_repredict(self, symbol: str) -> bool:
        """Check if we should re-predict (based on interval)."""
        if symbol not in self._last_pred_time:
            return True
        elapsed = (datetime.now() - self._last_pred_time[symbol]).total_seconds()
        return elapsed >= self.config.pred_interval_min * 60

    def predict(self, symbol: str):
        """Run Kronos prediction for a symbol."""
        if symbol not in self.prices:
            return None
        df = self.prices[symbol]
        lookback = self.config.lookback
        pred_len = self.config.pred_len

        if len(df) < lookback:
            return None

        # Use most recent data
        x_df = df.iloc[-lookback:][['open', 'high', 'low', 'close', 'volume', 'amount']]
        x_ts = df.iloc[-lookback:]['timestamps']
        # Generate future timestamps (infer from last known interval)
        if len(df) >= lookback + 2:
            delta = (df.iloc[-1]['timestamps'] - df.iloc[-2]['timestamps'])
        else:
            delta = pd.Timedelta(minutes=1)
        y_ts = pd.Series(
            [df.iloc[-1]['timestamps'] + delta * (i + 1) for i in range(pred_len)]
        )

        try:
            pred_df = self.predictor.predict(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=pred_len,
                T=self.config.temperature,
                top_p=self.config.top_p,
                sample_count=self.config.sample_count,
            )
            self.last_prediction[symbol] = pred_df
            self._last_pred_time[symbol] = datetime.now()
            return pred_df
        except Exception as e:
            print(f"  ⚠ Prediction failed for {symbol}: {e}")
            return None

    def decide(self) -> List[TradeSignal]:
        signals = []
        for symbol in self.config.symbols:
            if symbol not in self.prices:
                continue

            df = self.prices[symbol]
            if len(df) < self.config.lookback:
                continue

            current_close = df.iloc[-1]['close']

            # Predict if needed
            if self._should_repredict(symbol):
                self.predict(symbol)

            pred = self.last_prediction.get(symbol)
            if pred is None or len(pred) == 0:
                continue

            # Use last predicted close as target
            predicted_close = pred.iloc[-1]['close']
            change_pct = ((predicted_close - current_close) / current_close) * 100

            confidence = min(abs(change_pct) / 1.0, 1.0)

            if change_pct >= self.config.buy_threshold_pct:
                signals.append(TradeSignal(
                    symbol, "BUY", current_close,
                    confidence=confidence,
                    reason=f"Kronos predicts +{change_pct:.2f}% (target ${predicted_close:.2f})"
                ))
            elif change_pct <= self.config.sell_threshold_pct:
                signals.append(TradeSignal(
                    symbol, "SELL", current_close,
                    confidence=confidence,
                    reason=f"Kronos predicts {change_pct:.2f}% (target ${predicted_close:.2f})"
                ))

        return signals


# ─── Strategy 2: Simple Mean Reversion ────────────────────────────────────


class MeanReversionStrategy(Strategy):
    """Buy when price drops significantly below recent MA, sell when above.

    No ML required — useful as a fallback or comparison baseline.
    """

    def __init__(self, config, short_window=20, long_window=50):
        super().__init__(config)
        self.short_window = short_window
        self.long_window = long_window

    def name(self) -> str:
        return "mean_reversion"

    def decide(self) -> List[TradeSignal]:
        signals = []
        for symbol in self.config.symbols:
            if symbol not in self.prices:
                continue
            df = self.prices[symbol]
            if len(df) < self.long_window:
                continue

            close = df['close'].values
            short_ma = np.mean(close[-self.short_window:])
            long_ma = np.mean(close[-self.long_window:])
            current = close[-1]

            # Price significantly below long MA → buy signal (oversold)
            if current < long_ma * 0.97 and short_ma < long_ma:
                confidence = min(abs(current - long_ma) / long_ma * 10, 1.0)
                signals.append(TradeSignal(
                    symbol, "BUY", current,
                    confidence=confidence,
                    reason=f"Price ${current:.2f} below MA{self.long_window} (${long_ma:.2f})"
                ))
            # Price significantly above long MA → sell signal (overbought)
            elif current > long_ma * 1.03 and short_ma > long_ma:
                confidence = min(abs(current - long_ma) / long_ma * 10, 1.0)
                signals.append(TradeSignal(
                    symbol, "SELL", current,
                    confidence=confidence,
                    reason=f"Price ${current:.2f} above MA{self.long_window} (${long_ma:.2f})"
                ))

        return signals


# ─── Strategy 3: Grid Trading ──────────────────────────────────────────────


class GridStrategy(Strategy):
    """Grid trading: place limit buy orders at price levels below current,
    and limit sell orders at levels above current.

    This is a classic market-making strategy.
    """

    def __init__(self, config, grid_count=5, grid_spacing_pct=0.5,
                 position_size=100):
        super().__init__(config)
        self.grid_count = grid_count
        self.grid_spacing_pct = grid_spacing_pct
        self.position_size = position_size

    def name(self) -> str:
        return "grid"

    def decide(self) -> List[TradeSignal]:
        signals = []
        for symbol in self.config.symbols:
            if symbol not in self.prices:
                continue
            df = self.prices[symbol]
            if len(df) < 10:
                continue

            current = df.iloc[-1]['close']
            spacing = current * self.grid_spacing_pct / 100

            # Place buy orders below current price
            for i in range(1, self.grid_count + 1):
                buy_price = round(current - spacing * i, 2)
                signals.append(TradeSignal(
                    symbol, "BUY", buy_price,
                    confidence=0.5,
                    reason=f"Grid level {i} below: ${buy_price}"
                ))

            # Place sell orders above current price
            for i in range(1, self.grid_count + 1):
                sell_price = round(current + spacing * i, 2)
                signals.append(TradeSignal(
                    symbol, "SELL", sell_price,
                    confidence=0.5,
                    reason=f"Grid level {i} above: ${sell_price}"
                ))

        return signals


# ─── Strategy Factory ──────────────────────────────────────────────────────


def create_strategy(config, predictor: Optional['KronosPredictor'] = None) -> Strategy:
    """Factory to create strategy by name."""
    if config.strategy == "prediction":
        if predictor is None:
            raise ValueError("Prediction strategy requires a trained KronosPredictor")
        return PredictionStrategy(config, predictor)
    elif config.strategy == "mean_reversion":
        return MeanReversionStrategy(config)
    elif config.strategy == "grid":
        return GridStrategy(config)
    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")
