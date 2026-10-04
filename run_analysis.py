from src.data_loader import load_mbo_trades, get_or_convert_parquet_files, filter_regular_trading_hours
from src.signals import generate_micro_bars_and_signals
from src.labeling import label_momentum_episodes
from src.visualization import plot_multi_event_momentum_sample
from src.models import train_momentum_classifier
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
        # Filter raw trades to 09:30 - 16:00 US Eastern
        trades_rth = filter_regular_trading_hours(trades, time_col="ts_event")
        day_signals = generate_micro_bars_and_signals(trades_rth)
        if len(day_signals) > 0:
            daily_feature_dfs.append(day_signals)
            print(f"  {path.name}: {len(day_signals):,} bars")

    # Stack all days
    full_dataset = pl.concat(daily_feature_dfs)
    print(f"\nAll days processed. Total aggregated bars: {len(full_dataset):,}")

    # Run statistical diagnostics
    momentum_hours = full_dataset.filter(
        (pl.col("ts_event").dt.hour() == 9) & (pl.col("ts_event").dt.minute() >= 45) |
        (pl.col("ts_event").dt.hour() == 10) |
        (pl.col("ts_event").dt.hour() == 15)
    )
    evaluate_signals(momentum_hours)

    # -------------------------------------------------------------
    # RUN LONG MOMENTUM PIPELINE
    # -------------------------------------------------------------
    print("\nLabeling long momentum episodes...")
    df_long = label_momentum_episodes(
        full_dataset,
        direction="long",
        k_baseline=15,
        z_thresh=2.0,
        trail_mult=2.0,
        min_stop_bps=5.0,
        min_run_bps=5.0,
        max_horizon=15
    )
    clf_long = train_momentum_classifier(df_long, direction="long", cooldown_bars=12)

    # -------------------------------------------------------------
    # RUN SHORT MOMENTUM PIPELINE
    # -------------------------------------------------------------
    print("\nLabeling short momentum episodes...")
    df_short = label_momentum_episodes(
        full_dataset,
        direction="short",
        k_baseline=15,
        z_thresh=1.8,
        trail_mult=2.0,
        min_stop_bps=4.5,
        min_run_bps=5.0,
        max_horizon=15
    )
    clf_short = train_momentum_classifier(df_short, direction="short", cooldown_bars=12)

    print("Rendering multi-event audit charts...")
    # Plots a window containing at least 2 to 3 momentum episodes (wins and stops)
    plot_multi_event_momentum_sample(
        df_long,
        direction="long",
        min_events=3,
        max_span_bars=100,  # Max 100 minutes between first and last event
        pad_bars=15,  # Context bars before/after
        require_positive_only=False,  # Set to True if you only want successful runners
    )
    plot_multi_event_momentum_sample(
        df_short,
        direction="short",
        min_events=3,
        max_span_bars=100,  # Max 100 minutes between first and last event
        pad_bars=15,  # Context bars before/after
        require_positive_only=False,  # Set to True if you only want successful runners
    )

if __name__ == "__main__":
    main()

