import polars as pl


def generate_micro_bars_and_signals(trades_df: pl.DataFrame) -> pl.DataFrame:
    if trades_df.is_empty():
        return pl.DataFrame()

    # Aggregate trades into 1-min dynamic intervals
    bars = (
        trades_df.group_by_dynamic("ts_event", every="1m")
        .agg([
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("size").sum().alias("volume"),
            ((pl.col("price") * pl.col("size")).sum() / pl.col("size").sum()).alias("vwap"),
            # Buy volume: aggressor hit the Ask (side == 'A')
            pl.col("size").filter(pl.col("side") == "A").sum().fill_null(0).cast(pl.Float64).alias("sell_vol"),
            # Sell volume: aggressor hit the Bid (side == 'B')
            pl.col("size").filter(pl.col("side") == "B").sum().fill_null(0).cast(pl.Float64).alias("buy_vol"),
        ])
    )

    # Compute signals and forward returns
    processed = (
        bars
        .with_columns([
            # Single-bar Net Delta
            (pl.col("buy_vol") - pl.col("sell_vol")).alias("net_delta"),
            # Distance from Bar VWAP (in bps)
            ((pl.col("close") - pl.col("vwap")) / pl.col("vwap") * 10000).alias("dist_vwap_bps"),
            # Rolling 3-bar (3-minute) volume sums
            pl.col("buy_vol").rolling_sum(window_size=3).alias("roll_buy_3m"),
            pl.col("sell_vol").rolling_sum(window_size=3).alias("roll_sell_3m"),
            pl.col("volume").rolling_sum(window_size=3).alias("roll_vol_3m"),
            # Forward Targets: 5-minute (5 bars) and 15-minute (15 bars) returns
            ((pl.col("close").shift(-5) - pl.col("close")) / pl.col("close")).alias("fwd_return_5m"),
            ((pl.col("close").shift(-15) - pl.col("close")) / pl.col("close")).alias("fwd_return_15m"),
        ])
        .with_columns([
            # Normalized Persistent Delta Flow over 3 minutes [-1.0, 1.0]
            ((pl.col("roll_buy_3m") - pl.col("roll_sell_3m")) /
             (pl.col("roll_vol_3m") + 1e-6)).alias("persistent_delta_3m"),
            # Rolling 20-minute Z-Score of price
            ((pl.col("close") - pl.col("close").rolling_mean(window_size=20)) /
             (pl.col("close").rolling_std(window_size=20) + 1e-6)).alias("z_score_20m"),
        ])
        .with_columns([
            (
                    (pl.col("persistent_delta_3m") > 0.4) &
                    (pl.col("dist_vwap_bps") > 2.0) &
                    (pl.col("close") > pl.col("open"))
            ).alias("confirmed_long_breakout"),
            (
                    (pl.col("persistent_delta_3m") < -0.4) &
                    (pl.col("dist_vwap_bps") < -2.0) &
                    (pl.col("close") < pl.col("open"))
            ).alias("confirmed_short_breakout"),
        ])
        .drop_nulls()
    )

    return processed

