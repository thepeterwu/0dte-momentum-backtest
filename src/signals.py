import itertools
import numpy as np
import polars as pl
from scipy.stats import spearmanr


def generate_micro_bars_and_signals(trades_df: pl.DataFrame) -> pl.DataFrame:
    if trades_df.is_empty():
        return pl.DataFrame()

    # -------------------------------------------------------------------------
    # 1-Second Aggregation
    # -------------------------------------------------------------------------
    sec_bars = (
        trades_df.group_by_dynamic("ts_event", every="1s")
        .agg([
            pl.col("price").first().alias("open_1s"),
            pl.col("price").max().alias("high_1s"),
            pl.col("price").min().alias("low_1s"),
            pl.col("price").last().alias("close_1s"),
            pl.col("size").sum().alias("vol_1s"),
            pl.col("size").filter(pl.col("side") == "B").sum().fill_null(0).cast(pl.Float64).alias("buy_vol_1s"),
            pl.col("size").filter(pl.col("side") == "A").sum().fill_null(0).cast(pl.Float64).alias("sell_vol_1s"),
        ])
        .with_columns([
            (pl.col("close_1s") / pl.col("close_1s").shift(1)).log().fill_null(0).alias("log_ret_1s"),
        ])
    )

    # -------------------------------------------------------------------------
    # Sub-Minute Coiling, Burst, & Order Flow Signals
    # -------------------------------------------------------------------------
    sec_bars_enriched = (
        sec_bars.with_columns([
            (pl.col("close_1s") / pl.col("close_1s").shift(1)).log().fill_null(0.0).alias("log_ret_1s")
        ])
        .with_columns([
            # 30-second calm baseline (lagged 5s to avoid self-contamination), null estimate to be 1 bps
            # (30s and 5s calibrated via grid search in tune_micro_signals.py)
            (pl.col("log_ret_1s").shift(5).rolling_std(window_size=30, min_samples=15)
             .fill_null(0.0001).alias("vol_calm_30s")),
            # 3-minute macro baseline, null estimate to be 3 bps
            (pl.col("log_ret_1s").rolling_std(window_size=180, min_samples=30)
             .fill_null(0.0003).alias("vol_macro_3m")),
            # 5-second price velocity (5s calibrated via grid search in tune_micro_signals.py)
            (pl.col("close_1s") / pl.col("close_1s").shift(5) - 1.0).fill_null(0.0).alias("burst_ret_5s"),
            # 30-second consolidation channel bounds (lagged 5s)
            pl.col("high_1s").shift(5).rolling_max(window_size=30).alias("box_high_30s"),
            pl.col("low_1s").shift(5).rolling_min(window_size=30).alias("box_low_30s"),
            # 5-second aggressor flow thrust
            (pl.col("buy_vol_1s").rolling_sum(5) - pl.col("sell_vol_1s").rolling_sum(5)).alias("delta_5s"),
            (pl.col("vol_1s").rolling_sum(5) + 1e-6).alias("vol_total_5s"),
        ])
        .with_columns([
            # Micro-to-Macro Volatility Compression Ratio
            (pl.col("vol_calm_30s") / (pl.col("vol_macro_3m") + 1e-6)).alias("micro_compression_ratio"),
            # Micro-Normalized Z-Score Burst
            (pl.col("burst_ret_5s") / (pl.col("vol_calm_30s") * np.sqrt(10) + 1e-6)).alias("z_micro_burst"),
            # Consolidation box breakout ratios
            ((pl.col("close_1s") - pl.col("box_high_30s")) / (
                    pl.col("box_high_30s") - pl.col("box_low_30s") + 1e-4)).alias("box_expansion_long"),
            ((pl.col("box_low_30s") - pl.col("close_1s")) / (
                    pl.col("box_high_30s") - pl.col("box_low_30s") + 1e-4)).alias("box_expansion_short"),
            # Net flow aggressor thrust ratio [-1.0, 1.0]
            (pl.col("delta_5s") / pl.col("vol_total_5s")).alias("flow_thrust_5s"),
        ])
    )

    # -------------------------------------------------------------------------
    # 1-Minute Aggregation with Micro-Summary Features
    # -------------------------------------------------------------------------
    min_bars_from_sec = (
        sec_bars_enriched.group_by_dynamic("ts_event", every="1m")
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
            # --- Micro-signal summary metrics across the 1m bar ---
            pl.col("micro_compression_ratio").min().fill_null(1.0).alias("min_intra_compression_ratio"),
            pl.col("z_micro_burst").max().fill_null(0.0).alias("max_intra_z_burst_long"),
            pl.col("z_micro_burst").min().fill_null(0.0).alias("min_intra_z_burst_short"),
            pl.col("box_expansion_long").max().fill_null(0.0).alias("max_intra_box_expansion_long"),
            pl.col("box_expansion_short").max().fill_null(0.0).alias("max_intra_box_expansion_short"),
            pl.col("flow_thrust_5s").last().fill_null(0.0).alias("exit_flow_thrust_5s"),
        ])
        .with_columns([
            (pl.col("ts_event").dt.truncate("1d").alias("trade_date")),  # Extract trading session date
            pl.col("realized_variance").sqrt().alias("realized_vol_1m"),
            # Parkinson Volatility - more robust measure of volatility compared to volatility derived from closing price
            (((pl.col("high") / pl.col("low")).log().pow(2)) / (4 * np.log(2))).sqrt().alias("parkinson_vol"),
            (pl.col("buy_vol") - pl.col("sell_vol")).alias("net_delta"),
        ])
        # Zero out returns when bars cross between days
        .with_columns(
            pl.when(pl.col("trade_date") != pl.col("trade_date").shift(1))  # First bar of day
            .then(0.0)
            .otherwise((pl.col("close") / pl.col("close").shift(1)).log())
            .fill_null(0.0)
            .alias("ret_1m")
        )
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

    # -------------------------------------------------------------------------
    # Macro Intraday Features & Forward Returns (15m Strategy)
    # -------------------------------------------------------------------------
    processed = (
        minute_bars
        .with_columns([
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
        .with_columns([
            # 3-minute cumulative return
            (pl.col("close") / pl.col("close").shift(3) - 1.0).alias("cum_ret_3m"),
            # Donchian breakdown / breakout (distance to 15m low / high)
            (
                    (pl.col("close") - pl.col("low").rolling_min(15).shift(1))
                    / (pl.col("close") + 1e-6)
                    * 10000
            ).alias("dist_lowest_15m_bps"),
            (
                    (pl.col("close") - pl.col("high").rolling_max(15).shift(1))
                    / (pl.col("close") + 1e-6)
                    * 10000
            ).alias("dist_highest_15m_bps"),
        ])
        .drop_nulls()
    )

    return processed


def sweep_micro_parameters(
        second_bars: pl.DataFrame,
        coil_range: list[int] | None = None,
        burst_range: list[int] | None = None,
) -> pl.DataFrame:
    """ Sweeps W_coil and W_burst combinations on 1s bars and evaluates predictive power
        against forward 1m and 3m returns.
    """
    # Precompute 1s log returns and forward targets
    if burst_range is None:
        burst_range = [3, 5, 8, 10, 15]
    if coil_range is None:
        coil_range = [15, 20, 30, 45, 60]
    base_df = second_bars.with_columns([
        (pl.col("close_1s") / pl.col("close_1s").shift(1)).log().fill_null(0.0).alias("ret_1s"),
        ((pl.col("close_1s").shift(-60) - pl.col("close_1s")) / pl.col("close_1s")).fill_null(0.0).alias("fwd_ret_1m"),
        ((pl.col("close_1s").shift(-180) - pl.col("close_1s")) / pl.col("close_1s")).fill_null(0.0).alias("fwd_ret_3m"),
    ]).with_columns(
        pl.col("ret_1s").rolling_std(window_size=180, min_samples=30).fill_null(0.0003).alias("vol_macro_3m"),
    )

    results = []

    for w_coil, w_burst in itertools.product(coil_range, burst_range):
        # Compute test signal for this specific pair
        eval_df = base_df.with_columns([
            pl.col("ret_1s")
            .shift(w_burst)
            .rolling_std(window_size=w_coil, min_samples=max(5, w_coil // 2))
            .fill_null(0.0001)
            .alias("vol_coil"),
            (pl.col("close_1s") / pl.col("close_1s").shift(w_burst) - 1.0)
            .fill_null(0.0)
            .alias("burst_ret"),
            (
                    pl.col("buy_vol_1s").rolling_sum(w_burst)
                    - pl.col("sell_vol_1s").rolling_sum(w_burst)
            ).alias("delta_burst"),
            (pl.col("vol_1s").rolling_sum(w_burst) + 1e-6).alias("vol_burst"),
        ]).with_columns([
            (pl.col("vol_coil") / (pl.col("vol_macro_3m") + 1e-6)).alias("compression_ratio"),
            (pl.col("burst_ret") / (pl.col("vol_coil") * np.sqrt(w_burst) + 1e-6)).alias("z_burst"),
            (pl.col("delta_burst") / pl.col("vol_burst")).alias("flow_thrust"),
        ])

        # CONDITION 1: Prior window was genuinely coiled (volatility compressed)
        # CONDITION 2: Burst was accompanied by aggressive taker buying
        coiled_breakouts = eval_df.filter(
            (pl.col("compression_ratio") <= 0.45)
            & (pl.col("flow_thrust") > 0.30)
            & (pl.col("z_burst") > 1.5)
        )

        if len(coiled_breakouts) < 30:
            continue

        fwd_1m_bps = coiled_breakouts["fwd_ret_1m"].to_numpy() * 10000.0
        fwd_3m_bps = coiled_breakouts["fwd_ret_3m"].to_numpy() * 10000.0

        results.append({
            "w_coil_sec": w_coil,
            "w_burst_sec": w_burst,
            "trigger_count": len(coiled_breakouts),
            "avg_fwd_1m_bps": float(np.mean(fwd_1m_bps)),
            "avg_fwd_3m_bps": float(np.mean(fwd_3m_bps)),
            "win_rate_3m_pct": float(np.mean(fwd_3m_bps > 0) * 100),
        })

    if not results:
        print("No parameter combinations met the minimum threshold of 30 triggers.")
        return pl.DataFrame()

    return pl.DataFrame(results).sort("avg_fwd_3m_bps", descending=True)
