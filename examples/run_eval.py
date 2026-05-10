"""
Evaluate Kronos predictions against recent market data.

This script:
  1. Loads the finetuned (or pretrained) Kronos model + tokenizer
  2. Loads recent eval data from Longbridge
  3. Runs predictions and compares against actual prices
  4. Plots and saves results

Usage:
    conda activate kronos
    python run_eval.py [--use-finetuned] [--stock AAPL.US] [--lookback 512]

By default, uses the pretrained model. Use --use-finetuned after training.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import Kronos, KronosTokenizer, KronosPredictor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINETUNE_SAVE_DIR = os.path.join(ROOT, 'finetune_csv', 'finetuned')
EVAL_DATA_DIR = os.path.join(ROOT, 'finetune_csv', 'data', 'eval')
CACHE_DIR = os.path.join(ROOT, 'examples', 'model_cache')


# ─── Stocks and their safe names ────────────────────────────────────────
STOCK_MAP = {
    'AAPL.US': 'AAPL_US', 'BABA.US': 'BABA_US', 'TSLA.US': 'TSLA_US',
    'MSFT.US': 'MSFT_US', 'GOOGL.US': 'GOOGL_US', 'AMZN.US': 'AMZN_US',
    'NVDA.US': 'NVDA_US', 'META.US': 'META_US', 'NFLX.US': 'NFLX_US',
    'INTC.US': 'INTC_US', 'AMD.US': 'AMD_US', 'TSM.US': 'TSM_US',
    'ASML.US': 'ASML_US', 'AVGO.US': 'AVGO_US', 'ORCL.US': 'ORCL_US',
    'MU.US': 'MU_US',
}


def load_models(use_finetuned, exp_name='AAPL_US_min1'):
    """Load tokenizer and predictor."""
    tokenizer_path = (
        os.path.join(FINETUNE_SAVE_DIR, exp_name, 'tokenizer', 'best_model')
        if use_finetuned
        else "NeoQuasar/Kronos-Tokenizer-base"
    )
    model_path = (
        os.path.join(FINETUNE_SAVE_DIR, exp_name, 'basemodel', 'best_model')
        if use_finetuned
        else "NeoQuasar/Kronos-base"
    )

    print(f"📦 Loading tokenizer from: {tokenizer_path}")
    tokenizer = KronosTokenizer.from_pretrained(
        tokenizer_path, cache_dir=CACHE_DIR
    )
    print(f"📦 Loading predictor from: {model_path}")
    model = Kronos.from_pretrained(
        model_path, cache_dir=CACHE_DIR
    )
    predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
    return predictor


def load_eval_data(stock_name):
    """Load eval CSV for a given stock."""
    safe = STOCK_MAP.get(stock_name, stock_name.replace('.', '_'))
    csv_path = os.path.join(EVAL_DATA_DIR, f"{safe}_min1_eval.csv")
    if not os.path.exists(csv_path):
        alt = os.path.join(EVAL_DATA_DIR, f"{stock_name.replace('.', '_')}_min1_eval.csv")
        if os.path.exists(alt):
            csv_path = alt
        else:
            raise FileNotFoundError(f"Eval data not found: {csv_path} or {alt}")
    df = pd.read_csv(csv_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)
    print(f"📊 Loaded {len(df)} records for {stock_name}")
    print(f"   Range: {df['timestamps'].min()} ~ {df['timestamps'].max()}")
    return df


def run_prediction(predictor, df, lookback=512, pred_len=48):
    """Run prediction on the last lookback window, predict pred_len steps ahead."""
    if len(df) < lookback + pred_len:
        raise ValueError(
            f"Need at least {lookback + pred_len} records, got {len(df)}"
        )

    # Use the last lookback + pred_len window for evaluation
    x_df = df.iloc[:lookback][['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.iloc[:lookback]['timestamps']
    # The ground truth for the prediction period
    y_timestamp = df.iloc[lookback:lookback + pred_len]['timestamps']
    y_actual = df.iloc[lookback:lookback + pred_len][['open', 'high', 'low', 'close', 'volume', 'amount']]

    print(f"🔮 Predicting next {pred_len} steps from {y_timestamp.iloc[0]} ...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=5,  # average 5 samples for smoother prediction
    )

    return pred_df, y_actual


def plot_results(stock_name, timestamps, y_actual, pred_df, save_dir):
    """Plot close price: predicted vs actual."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'{stock_name} — Kronos Prediction vs Actual', fontsize=14)

    pairs = [('close', 'Close Price'), ('open', 'Open Price'),
             ('high', 'High Price'), ('low', 'Low Price')]

    for ax, (col, title) in zip(axes.flatten(), pairs):
        # Actual
        ax.plot(timestamps, y_actual[col].values, 'b-', label='Actual', linewidth=1.5)
        # Predicted
        ax.plot(timestamps, pred_df[col].values, 'r--', label='Predicted', linewidth=1.5)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    out_path = os.path.join(save_dir, f'{stock_name.replace(".", "_")}_eval.png')
    plt.savefig(out_path, dpi=150)
    print(f"📈 Plot saved: {out_path}")
    plt.close()

    # Print metrics
    print(f"\n📊 Prediction Metrics for {stock_name}:")
    for col in ['open', 'high', 'low', 'close']:
        mape = np.mean(np.abs((y_actual[col].values - pred_df[col].values)
                              / (y_actual[col].values + 1e-8))) * 100
        print(f"   {col:>6s} MAPE: {mape:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Kronos predictions')
    parser.add_argument('--use-finetuned', action='store_true',
                        help='Use finetuned model instead of pretrained')
    parser.add_argument('--exp-name', default='AAPL_US_min1',
                        help='Experiment name for finetuned models')
    parser.add_argument('--stock', default='AAPL.US',
                        help='Stock to evaluate')
    parser.add_argument('--lookback', type=int, default=400,
                        help='Lookback window (default: 400)')
    parser.add_argument('--pred-len', type=int, default=120,
                        help='Prediction length (default: 120)')
    args = parser.parse_args()

    save_dir = os.path.join(ROOT, 'figures', 'eval_results')
    os.makedirs(save_dir, exist_ok=True)

    # Load models
    predictor = load_models(args.use_finetuned, args.exp_name)

    # Load eval data
    df = load_eval_data(args.stock)

    # Run prediction
    pred_df, y_actual = run_prediction(
        predictor, df, args.lookback, args.pred_len
    )

    # Plot and show metrics
    pred_timestamps = y_actual.index  # use index positions for plot
    plot_results(args.stock, y_actual.index, y_actual, pred_df, save_dir)

    # Save prediction CSV
    out_csv = os.path.join(save_dir, f'{args.stock.replace(".", "_")}_prediction.csv')
    results = y_actual.copy()
    for col in pred_df.columns:
        results[f'pred_{col}'] = pred_df[col].values
    results.to_csv(out_csv)
    print(f"📄 Predictions saved: {out_csv}")


if __name__ == '__main__':
    main()
