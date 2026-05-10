"""
Kronos Trading Bot — Real-time Quantitative Trading with Longbridge Simulated Account.

This bot connects to Longbridge's simulated trading environment, fetches real-time
K-line data, runs Kronos model predictions, executes trades based on strategy signals,
and logs all activity.

Usage:
    conda activate kronos
    python trading/bot.py [--strategy prediction] [--mode paper]

Features:
    - Live K-line data via Longbridge QuoteContext
    - Kronos-powered price predictions
    - Multiple trading strategies (prediction, mean reversion, grid)
    - Simulated (paper) trading or real order execution
    - Trade logging + PnL tracking
    - Risk controls (max positions, stop loss, cooldown)
"""
import os
import sys
import time
import json
import csv
from datetime import datetime, date
from typing import Dict, List, Optional
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from longport.openapi import (
    TradeContext, QuoteContext, Config,
    OrderType, OrderSide, TimeInForceType,
    Period, AdjustType, OutsideRTH,
)

from trading.config import TradingConfig
from trading.strategies import Strategy, TradeSignal, create_strategy

# ─── Trading Bot ──────────────────────────────────────────────────────────


class KronosTradingBot:
    """
    Main trading bot orchestrator.

    Flow per cycle:
        1. Fetch latest K-line data for all tracked symbols
        2. Run strategy → get trade signals
        3. Filter signals against risk controls
        4. Execute trades via Longbridge API (or log for paper trading)
        5. Log everything + update PnL
    """

    def __init__(self, config: Optional[TradingConfig] = None,
                 mode: str = "paper"):
        """
        Args:
            config: Trading configuration
            mode: "paper" (simulated, no real orders) or "live" (real orders)
        """
        self.config = config or TradingConfig()
        self.mode = mode
        self.strategy: Optional[Strategy] = None
        self.predictor = None

        # ─── State ───────────────────────────────────────────────────────
        self.positions: Dict[str, float] = {}      # symbol → shares held
        self.cash_balance: float = 0.0
        self.daily_trades: int = 0
        self.daily_pnl: float = 0.0
        self.last_trade_time: Dict[str, datetime] = {}
        self.price_cache: Dict[str, pd.DataFrame] = {}
        self.total_trades: List[dict] = []
        self.running = False

        # ─── Longbridge clients ──────────────────────────────────────────
        self.quote_ctx: Optional[QuoteContext] = None
        self.trade_ctx: Optional[TradeContext] = None

        # Setup logging
        os.makedirs(self.config.log_dir, exist_ok=True)
        self._init_logs()

    def _init_logs(self):
        """Initialize CSV log files."""
        for path in [self.config.trade_log, self.config.pnl_log]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    if 'trade' in path:
                        f.write("timestamp,symbol,action,price,quantity,reason,status\n")
                    else:
                        f.write("timestamp,symbol,shares,avg_cost,current_price,pnl\n")

    def _load_credentials(self) -> Config:
        """Load Longbridge credentials from .env."""
        if os.path.exists(self.config.env_path):
            load_dotenv(self.config.env_path)
        import os as _os
        return Config(
            app_key=_os.environ.get('LONGBRIDGE_APP_KEY'),
            app_secret=_os.environ.get('LONGBRIDGE_APP_SECRET'),
            access_token=_os.environ.get('LONGBRIDGE_ACCESS_TOKEN'),
        )

    def _load_model(self):
        """Load Kronos model for prediction strategies."""
        from model import Kronos, KronosTokenizer, KronosPredictor

        if self.config.model_type == "finetuned":
            tok_path = os.path.join(
                self.config.finetune_base, self.config.exp_name,
                'tokenizer', 'best_model'
            )
            model_path = os.path.join(
                self.config.finetune_base, self.config.exp_name,
                'basemodel', 'best_model'
            )
        else:
            tok_path = "NeoQuasar/Kronos-Tokenizer-base"
            model_path = "NeoQuasar/Kronos-base"

        print(f"📦 Loading tokenizer: {tok_path}")
        tokenizer = KronosTokenizer.from_pretrained(
            tok_path, cache_dir=self.config.model_cache
        )
        print(f"📦 Loading model: {model_path}")
        model = Kronos.from_pretrained(
            model_path, cache_dir=self.config.model_cache
        )

        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        predictor = KronosPredictor(
            model, tokenizer, device=device,
            max_context=self.config.max_context
        )
        print(f"✅ KronosPredictor loaded (device={device})")
        return predictor

    def connect(self):
        """Connect to Longbridge API and load model."""
        print("🔌 Connecting to Longbridge...")
        cfg = self._load_credentials()

        self.quote_ctx = QuoteContext(cfg)
        self.trade_ctx = TradeContext(cfg)
        print("✅ QuoteContext & TradeContext connected")

        # Load model for prediction strategy
        if self.config.strategy == "prediction":
            self.predictor = self._load_model()

        # Create strategy
        self.strategy = create_strategy(self.config, self.predictor)
        print(f"📋 Strategy: {self.strategy.name()}")

        # Fetch initial account info
        self._refresh_account()

    def _refresh_account(self):
        """Refresh account balance and positions."""
        try:
            # Cash balance
            balances = self.trade_ctx.account_balance()
            if balances:
                # Use total_cash; for multi-currency, sum available cash per currency
                self.cash_balance = float(balances[0].total_cash)

            # Stock positions
            positions = self.trade_ctx.stock_positions()
            self.positions = {}
            for pos in positions:
                self.positions[pos.symbol] = float(pos.quantity)
                if self.config.verbose:
                    print(f"  📊 Position: {pos.symbol} × {pos.quantity}")

            if self.config.verbose:
                print(f"  💰 Cash: ${self.cash_balance:.2f}")
        except Exception as e:
            print(f"  ⚠ Account refresh error: {e}")

    def fetch_prices(self):
        """Fetch latest K-line data for all tracked symbols."""
        today = date.today()
        for symbol in self.config.symbols:
            try:
                resp = self.quote_ctx.history_candlesticks_by_date(
                    symbol, Period.Min_1, AdjustType.NoAdjust,
                    today, today
                )
                if not resp:
                    continue
                records = []
                for c in resp:
                    records.append({
                        "timestamps": pd.to_datetime(c.timestamp),
                        "open": float(c.open),
                        "high": float(c.high),
                        "low": float(c.low),
                        "close": float(c.close),
                        "volume": float(c.volume),
                        "amount": float(c.turnover),
                    })
                df = pd.DataFrame(records)
                df = df.sort_values('timestamps').reset_index(drop=True)
                self.price_cache[symbol] = df
                # Update strategy with new prices
                if self.strategy:
                    self.strategy.on_price_update(symbol, df)
            except Exception as e:
                print(f"  ⚠ Could not fetch {symbol}: {e}")

    def _check_risk_controls(self, signal: TradeSignal) -> bool:
        """Check if a trade signal passes all risk controls."""
        # Weekend trading
        if not self.config.trade_on_weekends:
            if datetime.now().weekday() >= 5:
                return False

        # Daily trade limit
        if self.daily_trades >= self.config.max_daily_trades:
            if self.config.verbose:
                print(f"  ⛔ Daily trade limit ({self.config.max_daily_trades}) reached")
            return False

        # Cooldown
        if signal.symbol in self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time[signal.symbol]).total_seconds()
            if elapsed < self.config.cooldown_minutes * 60:
                if self.config.verbose:
                    print(f"  ⏳ {signal.symbol} in cooldown ({elapsed:.0f}s < {self.config.cooldown_minutes*60}s)")
                return False

        # Min confidence
        if signal.confidence < 0.3:
            return False

        return True

    def _execute_trade(self, signal: TradeSignal):
        """Execute a trade signal via Longbridge API (or log for paper mode)."""
        side = OrderSide.Buy if signal.action == "BUY" else OrderSide.Sell
        qty = self.config.order_quantity

        if self.mode == "paper":
            # Paper trading: just log
            self._log_trade(signal, "PAPER")
            self._paper_execute(signal, qty)
            return True

        # Live trading: submit real order
        try:
            order = self.trade_ctx.submit_order(
                symbol=signal.symbol,
                order_type=OrderType.LO,
                side=side,
                submitted_quantity=qty,
                time_in_force=TimeInForceType.Day,
                submitted_price=round(signal.price, 2),
            )
            status = f"ORDER_ID={order.order_id}"
            self._log_trade(signal, status)
            print(f"  ✅ Order submitted: {side.name} {qty}×{signal.symbol} @ ${signal.price:.2f}")
            self.last_trade_time[signal.symbol] = datetime.now()
            self.daily_trades += 1
            return True
        except Exception as e:
            print(f"  ❌ Order failed: {e}")
            self._log_trade(signal, f"FAILED:{e}")
            return False

    def _paper_execute(self, signal: TradeSignal, qty: int):
        """Simulate trade execution for paper mode."""
        cost = signal.price * qty
        if signal.action == "BUY":
            if cost > self.cash_balance:
                # Adjust to max affordable
                qty = int(self.cash_balance / signal.price)
                cost = signal.price * qty
            self.cash_balance -= cost
            self.positions[signal.symbol] = self.positions.get(signal.symbol, 0) + qty
        else:
            held = self.positions.get(signal.symbol, 0)
            if qty > held:
                qty = held
                cost = signal.price * qty
            self.positions[signal.symbol] = held - qty
            self.cash_balance += cost

        self.last_trade_time[signal.symbol] = datetime.now()
        self.daily_trades += 1

        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": signal.symbol,
            "action": signal.action,
            "price": signal.price,
            "quantity": qty,
            "cost": cost,
            "reason": signal.reason,
        }
        self.total_trades.append(entry)

        print(f"  💼 [PAPER] {signal.action} {qty}×{signal.symbol} @ ${signal.price:.2f} | "
              f"Cash: ${self.cash_balance:.2f}")

    def _log_trade(self, signal: TradeSignal, status: str):
        """Write a trade to the CSV log."""
        with open(self.config.trade_log, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                datetime.now().isoformat(),
                signal.symbol, signal.action,
                f"{signal.price:.4f}", self.config.order_quantity,
                signal.reason, status,
            ])

    def _estimate_pnl(self) -> float:
        """Estimate current PnL based on last known prices."""
        total_pnl = 0.0
        details = []
        for symbol, shares in self.positions.items():
            if shares <= 0:
                continue
            if symbol not in self.price_cache:
                continue
            df = self.price_cache[symbol]
            if len(df) == 0:
                continue
            current_price = df.iloc[-1]['close']

            # Find average cost from trade history
            trades = [t for t in self.total_trades
                      if t['symbol'] == symbol and t['action'] == 'BUY']
            if not trades:
                continue
            total_cost = sum(t['cost'] for t in trades)
            total_shares = sum(t['quantity'] for t in trades)
            if total_shares == 0:
                continue
            avg_cost = total_cost / total_shares
            pnl = (current_price - avg_cost) * shares
            total_pnl += pnl
            details.append((symbol, shares, avg_cost, current_price, pnl))

        # Log PnL
        with open(self.config.pnl_log, 'a', newline='') as f:
            w = csv.writer(f)
            for symbol, shares, avg_cost, cur, pnl in details:
                w.writerow([
                    datetime.now().isoformat(),
                    symbol, shares, f"{avg_cost:.4f}",
                    f"{cur:.4f}", f"{pnl:.2f}",
                ])

        return total_pnl

    def run_once(self) -> List[TradeSignal]:
        """Run a single trading cycle: fetch → decide → execute → log."""
        print(f"\n{'='*50}")
        print(f"🔄 Trading cycle @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")

        # 1. Fetch latest prices
        self.fetch_prices()

        # 2. Run strategy
        signals = self.strategy.decide() if self.strategy else []
        if not signals:
            print("  No trade signals generated")
            return []

        print(f"\n📡 Strategy signals ({len(signals)}):")
        for s in signals:
            print(f"  {s}")

        # 3. Filter + execute
        executed = []
        for signal in signals:
            if self._check_risk_controls(signal):
                self._execute_trade(signal)
                executed.append(signal)
            else:
                print(f"  ⛔ Blocked: {signal} (risk control)")

        # 4. Refresh account + PnL
        self._refresh_account()
        pnl = self._estimate_pnl()
        print(f"\n📊 PnL: ${pnl:.2f} | Cash: ${self.cash_balance:.2f} | "
              f"Daily trades: {self.daily_trades}")

        return executed

    def run_loop(self):
        """Run the trading loop continuously during market hours."""
        self.running = True
        print(f"\n🚀 Kronos Trading Bot started (mode={self.mode})")
        print(f"   Strategy: {self.strategy.name() if self.strategy else 'N/A'}")
        print(f"   Monitoring: {', '.join(self.config.symbols)}")
        print(f"   Press Ctrl+C to stop\n")

        cycle_count = 0
        try:
            while self.running:
                cycle_count += 1
                self.run_once()

                time.sleep(self.config.check_interval_sec)

        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
        finally:
            self.running = False
            self._print_summary()

    def _print_summary(self):
        """Print trading summary."""
        print(f"\n{'='*50}")
        print("📊 Trading Summary")
        print(f"{'='*50}")
        print(f"Total trades: {len(self.total_trades)}")
        print(f"Daily trades: {self.daily_trades}")
        print(f"Cash balance: ${self.cash_balance:.2f}")
        print(f"Positions: {self.positions}")
        pnl = self._estimate_pnl()
        print(f"Estimated PnL: ${pnl:.2f}")
        print(f"{'='*50}\n")


# ─── CLI Entrypoint ────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Kronos Trading Bot")
    parser.add_argument('--mode', default='paper', choices=['paper', 'live'],
                        help='Trading mode (default: paper)')
    parser.add_argument('--strategy', default='prediction',
                        choices=['prediction', 'mean_reversion', 'grid'],
                        help='Trading strategy (default: prediction)')
    parser.add_argument('--model', default='pretrained',
                        choices=['pretrained', 'finetuned'],
                        help='Model to use (default: pretrained)')
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Stocks to trade (default: config)')
    parser.add_argument('--once', action='store_true',
                        help='Run a single cycle and exit')
    parser.add_argument('--pred-interval', type=int, default=None,
                        help='Minutes between Kronos predictions')

    args = parser.parse_args()

    # Build config
    cfg = TradingConfig()
    cfg.strategy = args.strategy
    cfg.model_type = args.model
    if args.symbols:
        cfg.symbols = args.symbols
    if args.pred_interval:
        cfg.pred_interval_min = args.pred_interval

    bot = KronosTradingBot(cfg, mode=args.mode)
    bot.connect()

    if args.once:
        bot.run_once()
    else:
        bot.run_loop()


if __name__ == '__main__':
    main()
