"""
Trading configuration for Kronos Trading Bot.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class TradingConfig:
    """Central configuration for the Kronos trading bot."""

    # ─── Accounts ──────────────────────────────────────────────────────────
    # .env file path for Longbridge credentials
    env_path: str = os.path.join(ROOT, 'examples', 'data', '.env')

    # ─── Stocks to Monitor ─────────────────────────────────────────────────
    symbols: List[str] = field(default_factory=lambda: [
        "AAPL.US", "MSFT.US", "GOOGL.US", "AMZN.US",
        "NVDA.US", "META.US", "TSLA.US", "BABA.US",
    ])

    # ─── Model ──────────────────────────────────────────────────────────────
    model_type: str = "pretrained"  # "pretrained" or "finetuned"
    exp_name: str = "AAPL_US_min1"
    model_cache: str = os.path.join(ROOT, 'examples', 'model_cache')
    finetune_base: str = os.path.join(ROOT, 'finetune_csv', 'finetuned')
    lookback: int = 400
    pred_len: int = 48          # Predict this many steps ahead
    max_context: int = 512
    sample_count: int = 3       # Average this many predictions
    temperature: float = 0.8
    top_p: float = 0.9

    # ─── Strategy ──────────────────────────────────────────────────────────
    strategy: str = "prediction"  # "prediction", "mean_reversion", "grid"
    # Prediction strategy: buy when predicted gain exceeds threshold
    buy_threshold_pct: float = 0.3   # Buy if predicted to rise >= 0.3%
    sell_threshold_pct: float = -0.3 # Sell if predicted to fall <= -0.3%
    # Position sizing
    max_position_pct: float = 30.0   # Max position per stock (% of total cap)
    order_quantity: int = 100        # Shares per order
    # Stop loss
    stop_loss_pct: float = -3.0      # Stop loss per trade
    take_profit_pct: float = 5.0     # Take profit per trade

    # ─── Timing ────────────────────────────────────────────────────────────
    check_interval_sec: int = 60      # Check prices every N seconds
    pred_interval_min: int = 15       # Re-run prediction every N minutes
    market_open: str = "09:30"        # US market open (EST)
    market_close: str = "16:00"       # US market close (EST)

    # ─── Risk Controls ─────────────────────────────────────────────────────
    max_daily_trades: int = 10        # Max trades per day
    max_drawdown_pct: float = -10.0   # Max daily drawdown before stopping
    cooldown_minutes: int = 5         # Wait between trades on same stock
    trade_on_weekends: bool = False
    verbose: bool = True

    # ─── Paths ──────────────────────────────────────────────────────────────
    log_dir: str = os.path.join(ROOT, 'trading', 'logs')
    trade_log: str = os.path.join(ROOT, 'trading', 'logs', 'trades.csv')
    pnl_log: str = os.path.join(ROOT, 'trading', 'logs', 'pnl.csv')
