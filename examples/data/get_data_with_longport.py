"""
Fetch K-line data from Longbridge API and save to JSONL.

IMPORTANT: Set your credentials via environment variables or .env file:
    LONGBRIDGE_APP_KEY, LONGBRIDGE_APP_SECRET, LONGBRIDGE_ACCESS_TOKEN

Copy .env.example to .env and fill in your credentials.
"""
from datetime import datetime, date, timedelta
from longport.openapi import QuoteContext, Config, Period, AdjustType
from tqdm import tqdm
import json
import time
import decimal
import os
from dotenv import load_dotenv

# Load credentials from .env file (safer than hardcoding)
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(f"Loaded credentials from {dotenv_path}")

config = Config(
    app_key=os.environ.get('LONGBRIDGE_APP_KEY'),
    app_secret=os.environ.get('LONGBRIDGE_APP_SECRET'),
    access_token=os.environ.get('LONGBRIDGE_ACCESS_TOKEN'),
)
ctx = QuoteContext(config)

stock_list = [
    "AAPL.US", "BABA.US", "TSLA.US", "MSFT.US",
    "GOOGL.US", "AMZN.US", "NVDA.US", "META.US",
    "NFLX.US", "INTC.US", "AMD.US", "TSM.US",
    "ASML.US", "AVGO.US", "ORCL.US", "MU.US",
]

class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        if isinstance(obj, (int, float)):
            return float(obj)
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        return super().default(obj)

def main():
    start_date = date(2023, 12, 4)
    end_date = date(2025, 11, 28)
    delta = end_date - start_date
    output_path = os.path.join(os.path.dirname(__file__), 'history_data.jsonl')

    with open(output_path, 'a+') as f:
        for stock in stock_list:
            for single_date in tqdm(
                (start_date + timedelta(n) for n in range(delta.days + 1)),
                desc=f"{stock}"
            ):
                resp = ctx.history_candlesticks_by_date(
                    stock, Period.Min_1, AdjustType.NoAdjust,
                    single_date, single_date
                )
                for candle in resp:
                    k_line = {
                        "stock": stock,
                        "timestamp": candle.timestamp,
                        "open": candle.open,
                        "high": candle.high,
                        "low": candle.low,
                        "close": candle.close,
                        "volume": candle.volume,
                        "turnover": candle.turnover,
                    }
                    f.write(json.dumps(k_line, cls=CustomJsonEncoder, ensure_ascii=False) + '\n')
                time.sleep(0.8)  # rate limit

if __name__ == "__main__":
    main()
