import polars as pl
import numpy as np


def label_momentum_episodes(
        df: pl.DataFrame,
        direction: str = "long",  # "long" or "short"
        k_baseline: int = 15,  # strictly lagged window of K bars
        z_thresh: float = 2.0,  # 2 std dev of Z-score of normalized log return
        trail_mult: float = 1.5,  # trailing stop multiplier: see vol_stop in code for logic
        min_stop_bps: float = 5.0,  # Minimum stop floor to prevent noise whipsaws
        min_run_bps: float = 5.0,
        max_horizon: int = 15,
) -> pl.DataFrame:
    """
        Labels path-dependent momentum episodes for either 'long' or 'short' directions.
    """

    if direction not in ["long", "short"]:
        raise ValueError("direction must be either 'long' or 'short'")

    # Calculate strictly lagged baseline volatility to determine z-impulse
    df = (
        df.with_columns([
            # Strictly lagged return WITHIN the same trading day
            pl.col("ret_1m").shift(1).over("trade_date").alias("ret_1m_lagged")
        ]).with_columns([
            (pl.col("ret_1m_lagged").rolling_mean(window_size=k_baseline, min_samples=1)
                                    .over("trade_date")
                                    .fill_null(0.0)     # expected zero drift for first bars of the day
                                    .alias("mu_base")
             ),
            (pl.col("ret_1m_lagged").rolling_std(window_size=k_baseline, min_samples=2)
                                    .over("trade_date")
                                    .fill_null(0.0003)  # expected variance (3 bps), empirical estimate w/ annual vol 9.4%
                                    .alias("std_base")
             ),
        ])
        .with_columns([
            ((pl.col("ret_1m") - pl.col("mu_base")) / (pl.col("std_base") + 1e-6)).alias("z_impulse")
        ])
        .drop("ret_1m_lagged")
    )

    # Convert arrays to numpy for path traversal
    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    z_scores = df["z_impulse"].to_numpy()
    std_bases = df["std_base"].to_numpy()
    trade_dates = df["trade_date"].to_numpy()
    n = len(df)

    labels = np.zeros(n, dtype=np.int32)
    mfe_bps = np.zeros(n, dtype=np.float64)  # MFE (Maximum Favorable Excursion)
    end_indices = np.full(n, -1, dtype=np.int32)

    # Pre-calculate day change boundaries
    day_changed = trade_dates[1:] != trade_dates[:-1]   # drop first vs drop last in slice and compare
    day_end_indices = np.where(day_changed)[0]          # returns indices of day boundaries
    # Create a lookup array mapping each bar index to the last bar of that day
    last_bar_of_day = np.zeros(n, dtype=np.int32)
    curr_end_ptr = 0
    for i in range(n):
        if (
                curr_end_ptr < len(day_end_indices)
                and i > day_end_indices[curr_end_ptr]
        ):
            curr_end_ptr += 1
        last_bar_of_day[i] = (
            day_end_indices[curr_end_ptr]
            if curr_end_ptr < len(day_end_indices)
            else n - 1
        )

    # Directional Path-dependent evaluation of momentum episodes
    for t in range(k_baseline + 1, n - max_horizon):
        p_start = closes[t]
        # Enforce minimum volatility stop buffer to avoid micro-chopping
        stop_pct = max(trail_mult * std_bases[t], min_stop_bps / 10000.0)
        vol_stop = stop_pct * p_start

        # Hard barrier: Cannot exceed horizon OR the end of the current trading day
        intraday_limit = last_bar_of_day[t]
        tau_max = min(t + max_horizon, intraday_limit)

        if tau_max <= t:
            continue  # Skip triggers firing on the final bar of the day

        if direction == "long" and z_scores[t] >= z_thresh:
            peak = highs[t]
            tau_end = tau_max

            for s in range(t + 1, tau_max + 1):
                if highs[s] > peak:
                    peak = highs[s]
                # Trailing pullback check against current bar's low
                if (peak - lows[s]) >= vol_stop:
                    tau_end = s
                    break

            realized_mfe = (peak - p_start) / p_start * 10000.0
            mfe_bps[t] = realized_mfe
            end_indices[t] = tau_end
            if realized_mfe >= min_run_bps:
                labels[t] = 1

        elif direction == "short" and z_scores[t] <= -z_thresh:
            trough = lows[t]
            tau_end = tau_max

            for s in range(t + 1, tau_max + 1):
                if lows[s] < trough:
                    trough = lows[s]
                # Trailing bounce check against current bar's high
                if (highs[s] - trough) >= vol_stop:
                    tau_end = s
                    break

            # Favorable excursion in short direction (gain as price falls)
            realized_mfe = (p_start - trough) / p_start * 10000.0
            mfe_bps[t] = realized_mfe
            end_indices[t] = tau_end
            if realized_mfe >= min_run_bps:
                labels[t] = 1

    return df.with_columns([
        pl.Series("target_label", labels),
        pl.Series("target_mfe_bps", mfe_bps),
        pl.Series("target_end_idx", end_indices),
        pl.lit(direction).alias("momentum_direction"),
    ])
