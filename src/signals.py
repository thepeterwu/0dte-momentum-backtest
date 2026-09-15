import polars as pl


def generate_micro_bars_and_signals(trades_df: pl.DataFrame) -> pl.DataFrame:
    # 1. Aggregate trades into 5-second dynamic intervals
    bars = (
        trades_df.group_by_dynamic("ts_event", every="5s")
        .agg([
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("size").sum().alias("volume"),
            # Buy volume: aggressor hit the Ask (side == 'A')
            pl.col("size").filter(pl.col("side") == "A").sum().fill_null(0).alias("buy_vol"),
            # Sell volume: aggressor hit the Bid (side == 'B')
            pl.col("size").filter(pl.col("side") == "B").sum().fill_null(0).alias("sell_vol"),
        ])
    )

    # 2. Compute signals and forward returns
    processed = (
        bars
        .with_columns([
            # Signal 1: Normalized Trade Flow Imbalance
            ((pl.col("buy_vol") - pl.col("sell_vol")) /
             (pl.col("buy_vol") + pl.col("sell_vol") + 1e-6)).alias("trade_imbalance"),

            # 20-period (100-second) rolling stats
            pl.col("close").rolling_mean(window_size=20).alias("ma_20"),
            pl.col("close").rolling_std(window_size=20).alias("std_20"),

            # Target: Forward 30-second return (6 five-second bars ahead)
            ((pl.col("close").shift(-6) - pl.col("close")) / pl.col("close")).alias("fwd_return_30s")
        ])
        .with_columns([
            # Signal 2: Z-Score Breakout
            ((pl.col("close") - pl.col("ma_20")) / (pl.col("std_20") + 1e-6)).alias("z_score_breakout")
        ])
        .drop_nulls()
    )

    return processed