"""
Fetch recent K-line data from Longbridge API for evaluation.

Usage:
    conda activate kronos
    python fetch_eval_data.py

This fetches up to the last N trading days of 1-min data for the stock list.
"""
from datetime import datetime, date, timedelta
from longport.openapi import QuoteContext, Config, Period, AdjustType
from tqdm import tqdm
import json
import time
import decimal
import os
import pandas as pd
from dotenv import load_dotenv

# Load credentials
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# ─── Config ──────────────────────────────────────────────────────────────
STOCKS = [
    "AAPL.US", "BABA.US", "TSLA.US", "MSFT.US",
    "GOOGL.US", "AMZN.US", "NVDA.US", "META.US",
    "NFLX.US", "INTC.US", "AMD.US", "TSM.US",
    "ASML.US", "AVGO.US", "ORCL.US", "MU.US",
]
LOOKBACK_TRADING_DAYS = 60  # fetch ~3 months of recent data
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, 'finetune_csv', 'data', 'eval')


def parse_timestamp(ts):
    """Handle both datetime objects and string timestamps from API."""
    if isinstance(ts, datetime):
        return ts
    return pd.to_datetime(ts)


def get_recent_trading_days(end_date, count):
    """Get last N trading days (weekdays)."""
    days = []
    d = end_date
    while len(days) < count and d > end_date - timedelta(days=count * 2):
        if d.weekday() < 5:  # Mon-Fri
            days.append(d)
        d -= timedelta(days=1)
    return days


def main():
    # Use direct constructor instead of from_env() for reliability
    config = Config(
        app_key=os.environ.get('LONGBRIDGE_APP_KEY'),
        app_secret=os.environ.get('LONGBRIDGE_APP_SECRET'),
        access_token=os.environ.get('LONGBRIDGE_ACCESS_TOKEN'),
    )
    ctx = QuoteContext(config)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    end_date = date.today()  # today is 2026-05-10 (Sunday)
    trading_days = get_recent_trading_days(end_date, LOOKBACK_TRADING_DAYS)
    # Sort ascending for API calls
    trading_days = sorted(trading_days)

    print(f"📅 Fetching data for {len(trading_days)} recent trading days ({trading_days[0]} ~ {trading_days[-1]})")
    print(f"📊 Stocks: {', '.join(STOCKS)}")

    all_records = []
    for stock in STOCKS:
        stock_records = 0
        for single_date in tqdm(trading_days, desc=f"{stock}"):
            try:
                resp = ctx.history_candlesticks_by_date(
                    stock, Period.Min_1, AdjustType.NoAdjust,
                    single_date, single_date
                )
                for candle in resp:
                    ts = parse_timestamp(candle.timestamp)
                    all_records.append({
                        "stock": stock,
                        "timestamps": ts,
                        "open": float(candle.open),
                        "high": float(candle.high),
                        "low": float(candle.low),
                        "close": float(candle.close),
                        "volume": float(candle.volume),
                        "amount": float(candle.turnover),
                    })
                    stock_records += 1
            except Exception as e:
                print(f"  ⚠ {stock} on {single_date}: {e}")
            time.sleep(0.5)  # rate limit

        print(f"  {stock}: {stock_records} records")

    if not all_records:
        print("❌ No data fetched. Check your Longbridge credentials.")
        return

    df = pd.DataFrame(all_records)
    df = df.sort_values(['stock', 'timestamps']).reset_index(drop=True)

    # Save combined eval data
    combined_path = os.path.join(OUTPUT_DIR, 'eval_data_combined.csv')
    df.to_csv(combined_path, index=False)
    print(f"\n✅ Combined eval data: {len(df)} records -> {combined_path}")

    # Save per-stock CSV files (matching training format)
    for stock in STOCKS:
        stock_df = df[df['stock'] == stock].copy()
        stock_df = stock_df.drop(columns=['stock'])
        if len(stock_df) == 0:
            continue
        safe_name = stock.replace('.', '_').replace('^', '')
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}_min1_eval.csv")
        stock_df.to_csv(out_path, index=False)
        print(f"  ✅ {stock}: {len(stock_df)} records -> {os.path.basename(out_path)}")

    print(f"\n📊 Date range: {df['timestamps'].min()} ~ {df['timestamps'].max()}")


if __name__ == '__main__':
    main()
