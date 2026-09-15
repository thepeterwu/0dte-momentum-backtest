from src.data_loader import load_mbo_trades, get_or_convert_parquet_files
from src.signals import generate_micro_bars_and_signals
import scipy.stats as stats
import numpy as np
import polars as pl


def evaluate_signals(df):
    # Extract Arrays
    p_delta = df["persistent_delta_3m"].to_numpy()
    fwd_5m = df["fwd_return_5m"].to_numpy()
    fwd_15m = df["fwd_return_15m"].to_numpy()

    ic_5m, p_val_5m = stats.spearmanr(p_delta, fwd_5m)
    ic_15m, p_val_15m = stats.spearmanr(p_delta, fwd_15m)

    print("\n" + "=" * 60)
    print("      PERSISTENT ORDER FLOW PREDICTIVE POWER (1-MIN BARS)   ")
    print("=" * 60)
    print(f"Total Sample Bars        : {len(df):,}")
    print(f"Persistent Delta ->  5m IC: {ic_5m:+.4f} (p-value: {p_val_5m:.2e})")
    print(f"Persistent Delta -> 15m IC: {ic_15m:+.4f} (p-value: {p_val_15m:.2e})")
    print("=" * 60)

    # Momentum Conditional Edge: Strong multi-period buying vs selling
    strong_buyers = p_delta > 0.4
    strong_sellers = p_delta < -0.4

    print("\n--- Conditional Forward 15m Momentum Edge ---")
    print(f"Long  (Delta > +0.4) [{np.sum(strong_buyers):,} bars]: {np.mean(fwd_15m[strong_buyers]) * 10000:+.2f} bps")
    print(
        f"Short (Delta < -0.4) [{np.sum(strong_sellers):,} bars]: {np.mean(fwd_15m[strong_sellers]) * 10000:+.2f} bps")


    # # Debugging --------------------------------------------------------------------------------------------------------
    # # Check trade balance distribution
    # print("\nTrade Imbalance")
    # print(df.select([
    #     pl.col("trade_imbalance").quantile(q).alias(f"q_{int(q * 100)}")
    #     for q in [0.01, 0.05, 0.50, 0.95, 0.99]
    # ]))
    # # verify buy sell volume
    # print(df.select([
    #     pl.col("buy_vol").sum().alias("total_buy_vol"),
    #     pl.col("sell_vol").sum().alias("total_sell_vol"),
    # ]))

    # Evaluate the Contrarian Edge (Fading Exhaustion)
    fade_short_triggers = df.filter(pl.col("persistent_delta_3m") > 0.4)
    print(
        f"Fade Short (Sell after high buy flow) Return: {-fade_short_triggers['fwd_return_15m'].mean() * 10000:+.2f} bps")

    # Check if Confirmation Filters Flipped the Sign
    confirmed_longs = df.filter(pl.col("confirmed_long_breakout"))
    confirmed_shorts = df.filter(pl.col("confirmed_short_breakout"))

    print(f"\n--- Confirmed Breakouts (Delta + VWAP + Candle Direction) ---")
    print(
        f"Confirmed Longs  [{len(confirmed_longs):,} bars]: {confirmed_longs['fwd_return_15m'].mean() * 10000:+.2f} bps")
    print(
        f"Confirmed Shorts [{len(confirmed_shorts):,} bars]: {confirmed_shorts['fwd_return_15m'].mean() * 10000:+.2f} bps")


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
    momentum_hours = full_dataset.filter(
        (pl.col("ts_event").dt.hour() == 9) & (pl.col("ts_event").dt.minute() >= 45) |
        (pl.col("ts_event").dt.hour() == 10) |
        (pl.col("ts_event").dt.hour() == 15)
    )
    evaluate_signals(momentum_hours)


if __name__ == "__main__":
    main()

