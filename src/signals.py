import polars as pl
import numpy as np

def generate_micro_bars_and_signals(trades_df: pl.DataFrame) -> pl.DataFrame:
    if trades_df.is_empty():
        return pl.DataFrame()

    # Aggregate trades into 1 sec and then 1-min dynamic intervals
    sec_bars = (
        trades_df.group_by_dynamic("ts_event", every="1s")
        .agg([
            pl.col("price").last().alias("close_1s"),
            pl.col("size").sum().alias("vol_1s"),
            pl.col("size").filter(pl.col("side") == "B").sum().fill_null(0).cast(pl.Float64).alias("buy_vol_1s"),
            pl.col("size").filter(pl.col("side") == "A").sum().fill_null(0).cast(pl.Float64).alias("sell_vol_1s"),
        ])
        .with_columns([
            (pl.col("close_1s") / pl.col("close_1s").shift(1)).log().fill_null(0).alias("log_ret_1s")
        ])
    )

    min_bars_from_sec = (
        sec_bars.group_by_dynamic("ts_event", every="1m")
        .agg([
            pl.col("close_1s").first().alias("open"),
            pl.col("close_1s").max().alias("high"),
            pl.col("close_1s").min().alias("low"),
            pl.col("close_1s").last().alias("close"),
            pl.col("vol_1s").sum().alias("volume"),
            pl.col("buy_vol_1s").sum().alias("buy_vol"),
            pl.col("sell_vol_1s").sum().alias("sell_vol"),
            pl.col("log_ret_1s").pow(2).sum().alias("realized_variance"),
            pl.col("vol_1s").filter(pl.col("vol_1s") > 0).count().alias("active_seconds"),
        ])
        .with_columns([
            pl.col("realized_variance").sqrt().alias("realized_vol_1m"),
            # Parkinson Volatility
            (((pl.col("high") / pl.col("low")).log().pow(2)) / (4 * np.log(2))).sqrt().alias("parkinson_vol"),
            (pl.col("buy_vol") - pl.col("sell_vol")).alias("net_delta"),
        ])
    )

    # Calculate VWAP directly from raw trades in a separate 1-minute aggregation
    raw_vwap_1m = (
        trades_df.group_by_dynamic("ts_event", every="1m")
        .agg([
            (((pl.col("price") * pl.col("size")).sum()) / (pl.col("size").sum() + 1e-6)).alias("vwap")
        ])
    )

    # Join VWAP onto the 1-minute bars
    minute_bars = (
        min_bars_from_sec
        .join(raw_vwap_1m, on="ts_event", how="left")
    )

    # Compute signals and forward returns
    processed = (
        minute_bars
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
            # Rolling 20-minute Z-Score of price (remove self-inclusion to capture stronger deviations)
            ((pl.col("close") - pl.col("close").shift(1).rolling_mean(window_size=20)) /
             (pl.col("close").shift(1).rolling_std(window_size=20) + 1e-6)).alias("z_score_20m_prior"),
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

