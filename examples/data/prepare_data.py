"""
Data processing pipeline: Convert raw JSONL K-line data to CSV format for Kronos finetuning.

Usage:
    conda activate kronos
    python prepare_data.py [--data_dir ./examples/data] [--output_dir ./finetune_csv/data]

This script:
    1. Reads history_data.jsonl (or fresh data from Longbridge)
    2. Groups by stock, sorts chronologically
    3. Renames turnover -> amount
    4. Saves as individual per-stock CSV files
    5. Saves a combined all-stock CSV for multi-stock training
"""
import os
import sys
import json
import argparse
import pandas as pd
from collections import defaultdict

# ─── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.dirname(SCRIPT_DIR)  # Kronos/examples/
ROOT = os.path.dirname(EXAMPLES_DIR)  # Kronos/
DEFAULT_DATA_DIR = EXAMPLES_DIR  # the JSONL is at Kronos/examples/history_data.jsonl
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, 'finetune_csv', 'data')


def read_jsonl(filepath):
    """Read JSONL file into list of dicts."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group_by_stock(records):
    """Group records by stock symbol."""
    groups = defaultdict(list)
    for rec in records:
        groups[rec['stock']].append(rec)
    print(f"Found {len(groups)} stocks in dataset")
    for stock, recs in sorted(groups.items()):
        print(f"  {stock}: {len(recs)} records")
    return groups


def records_to_df(records):
    """Convert records list to sorted DataFrame with required columns."""
    df = pd.DataFrame(records)
    df['timestamps'] = pd.to_datetime(df['timestamp'])
    df = df.drop(columns=['timestamp', 'stock'], errors='ignore')

    # Rename turnover -> amount
    if 'turnover' in df.columns:
        df = df.rename(columns={'turnover': 'amount'})

    # Ensure all required columns exist
    required = ['open', 'high', 'low', 'close', 'volume', 'amount']
    for col in required:
        if col not in df.columns:
            print(f"  ⚠ Missing column '{col}', filling with 0")
            df[col] = 0.0

    # Keep only needed columns, in order
    cols = ['timestamps'] + [c for c in required if c in df.columns]
    df = df[cols]

    # Sort chronologically
    df = df.sort_values('timestamps').reset_index(drop=True)
    return df


def save_csv(df, output_path):
    """Save DataFrame to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  ✅ Saved {len(df)} rows -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare Kronos training data')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR,
                        help='Directory with history_data.jsonl')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                        help='Directory for output CSV files')
    parser.add_argument('--input', default='history_data.jsonl',
                        help='Input JSONL filename')
    parser.add_argument('--stock', default=None,
                        help='Only process specific stock (e.g. AAPL.US)')
    parser.add_argument('--combine', action='store_true', default=True,
                        help='Also generate a combined CSV (default: True)')
    args = parser.parse_args()

    input_path = os.path.join(args.data_dir, args.input)
    if not os.path.exists(input_path):
        print(f"❌ Input not found: {input_path}")
        print("   Run get_data_with_longport.py first or check path.")
        sys.exit(1)

    print(f"📂 Reading data from {input_path} ...")
    records = read_jsonl(input_path)
    print(f"   Total: {len(records)} records")

    groups = group_by_stock(records)

    stock_dfs = {}
    for stock in sorted(groups.keys()):
        if args.stock and stock != args.stock:
            continue
        safe_name = stock.replace('.', '_').replace('^', '')
        output_path = os.path.join(args.output_dir, f"{safe_name}_min1_all.csv")
        df = records_to_df(groups[stock])
        save_csv(df, output_path)
        stock_dfs[stock] = df

    # Generate combined CSV for multi-stock training
    if args.combine and len(stock_dfs) > 1:
        combined_path = os.path.join(args.output_dir, 'all_stocks_min1_combined.csv')
        combined_dfs = []
        for stock, df in sorted(stock_dfs.items()):
            df_copy = df.copy()
            df_copy['stock'] = stock
            combined_dfs.append(df_copy)
        combined = pd.concat(combined_dfs, ignore_index=True)
        combined = combined.sort_values('timestamps').reset_index(drop=True)
        save_csv(combined, combined_path)

    # Print data stats
    print("\n📊 Data Summary:")
    for stock, df in sorted(stock_dfs.items()):
        print(f"  {stock}: {df['timestamps'].min()} ~ {df['timestamps'].max()}, "
              f"{len(df)} records, "
              f"OHLC range: [{df['low'].min():.2f}, {df['high'].max():.2f}]")


if __name__ == '__main__':
    main()
