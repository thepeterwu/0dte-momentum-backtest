from src.data_loader import load_mbo_trades, get_or_convert_parquet_files
from src.signals import generate_micro_bars_and_signals
import scipy.stats as stats
import numpy as np
import polars as pl


def evaluate_signals(df):
    # Extract arrays
    imb = df["trade_imbalance"].to_numpy()
    zscore = df["z_score_breakout"].to_numpy()
    fwd_ret_30 = df["fwd_return_30s"].to_numpy()
    fwd_ret_10 = df["fwd_return_10s"].to_numpy()
    fwd_ret_5 = df["fwd_return_5s"].to_numpy()
    fwd_ret = fwd_ret_30
    fwd_shift = 30

    # Calculate Information Coefficients (Spearman Rank Correlation)
    ic_imb, p_imb = stats.spearmanr(imb, fwd_ret)
    ic_z, p_z = stats.spearmanr(zscore, fwd_ret)

    print("==================================================")
    print("           SIGNAL PREDICTIVE POWER (Time period)  ")
    print("==================================================")
    print(f"Total Sample Bars  : {len(df):,}")
    print(f"Trade Imbalance IC : {ic_imb:+.4f} (p-value: {p_imb:.2e})")
    print(f"Z-Score Breakout IC: {ic_z:+.4f} (p-value: {p_z:.2e})")
    print("==================================================")

    # Check Conditional Edge (in Basis Points)
    # Long threshold: Imbalance > 0.6 | Short threshold: Imbalance < -0.6
    long_mask = imb > 0.6
    short_mask = imb < -0.6

    avg_long_bps = np.mean(fwd_ret[long_mask]) * 10000 if np.any(long_mask) else 0.0
    avg_short_bps = np.mean(fwd_ret[short_mask]) * 10000 if np.any(short_mask) else 0.0

    print(f"\n--- Conditional Forward {fwd_shift}s Edge ---")
    print(f"Long  (Imbalance >  0.6) [{np.sum(long_mask):,} bars]: {avg_long_bps:+.2f} bps")
    print(f"Short (Imbalance < -0.6) [{np.sum(short_mask):,} bars]: {avg_short_bps:+.2f} bps")

    # Debugging --------------------------------------------------------------------------------------------------------
    # Check trade balance distribution
    print("\nTrade Imbalance")
    print(df.select([
        pl.col("trade_imbalance").quantile(q).alias(f"q_{int(q * 100)}")
        for q in [0.01, 0.05, 0.50, 0.95, 0.99]
    ]))
    # verify buy sell volume
    print(df.select([
        pl.col("buy_vol").sum().alias("total_buy_vol"),
        pl.col("sell_vol").sum().alias("total_sell_vol"),
    ]))


def main():
    data_dir = "data"
    parquet_paths = get_or_convert_parquet_files(data_dir="data", max_workers=4)
    daily_feature_dfs = []
    print("\nProcessing daily files...")

    # Extract signals per day
    for path in parquet_paths:
        trades = load_mbo_trades(path)
        day_signals = generate_micro_bars_and_signals(trades)
        if len(day_signals) > 0:
            daily_feature_dfs.append(day_signals)
            print(f"  {path.name}: {len(day_signals):,} bars")

    # Stack all 23 days
    full_dataset = pl.concat(daily_feature_dfs)
    print(f"\nAll days processed. Total aggregated bars: {len(full_dataset):,}")

    # Run statistical diagnostics
    median_vol = full_dataset["volume"].median()
    liquid_bars = full_dataset.filter(pl.col("volume") > median_vol)
    evaluate_signals(liquid_bars)


if __name__ == "__main__":
    main()

