import polars as pl
import numpy as np


def label_momentum_episodes(
        df: pl.DataFrame,
        direction: str = "long",  # "long" or "short"
        k_baseline: int = 15,  # strictly lagged window of K bars
        z_thresh: float = 2.0,  # Z-score hurdle
        trail_mult: float = 1.5,  # trailing stop multiplier
        min_stop_bps: float = 5.0,  # Minimum stop floor in basis points
        min_run_bps: float = 5.0,  # Target expansion milestone in basis points
        max_horizon: int = 15,
        cooldown_bars: int = 12
) -> pl.DataFrame:
    """Labels path-dependent momentum episodes using Parkinson or close-to-close volatility."""
    if direction not in ["long", "short"]:
        raise ValueError("direction must be either 'long' or 'short'")

    # Compute strictly lagged baselines within each day
    df = (
        df.with_columns([
            # Strictly lagged 1-minute return within session
            pl.col("ret_1m").shift(1).over("trade_date").alias("ret_1m_lagged"),
        ])
        .with_columns([
            # Mean return drift (identical for both methods)
            (
                pl.col("ret_1m_lagged")
                .rolling_mean(window_size=k_baseline, min_samples=1)
                .over("trade_date")
                .fill_null(0.0)
                .alias("mu_base")
            ),
            # Close-to-close volatility baseline
            (
                pl.col("ret_1m_lagged")
                .rolling_std(window_size=k_baseline, min_samples=2)
                .over("trade_date")
                .fill_null(0.0003)
                .alias("std_dev_base_c2c")
            ),
        ])
        .with_columns([
            (
                    (pl.col("ret_1m") - pl.col("mu_base"))
                    / (pl.col("std_dev_base_c2c") + 1e-6)
            ).alias("z_impulse_c2c"),
        ])
        .drop(["ret_1m_lagged"])
    )

    active_z = df["z_impulse_c2c"].to_numpy()
    active_std = df["std_dev_base_c2c"].to_numpy()
    active_impulse_col = "z_impulse_c2c"
    box_exp_short = df["max_intra_box_expansion_short"].to_numpy()
    thrust_5s = df["exit_flow_thrust_5s"].to_numpy()
    dist_low_15m = df["dist_lowest_15m_bps"].to_numpy()

    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    trade_dates = df["trade_date"].to_numpy()
    n = len(df)

    labels = np.zeros(n, dtype=np.int32)
    mfe_bps = np.zeros(n, dtype=np.float64)
    end_indices = np.full(n, -1, dtype=np.int32)
    is_candidate = np.zeros(n, dtype=bool)

    # Pre-calculate day boundary indices for hard session stops
    day_changed = trade_dates[1:] != trade_dates[:-1]
    day_end_indices = np.where(day_changed)[0]

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

    # Path-dependent evaluation
    t = k_baseline + 1
    while t < n - max_horizon:
        p_start = closes[t]
        if p_start <= 0.0:
            t += 1
            continue

        # Stop distance in price terms
        stop_pct = max(trail_mult * active_std[t], min_stop_bps / 10000.0)
        vol_stop = stop_pct * p_start

        # Hard boundary: Cannot exceed horizon OR the end of the current trading day
        intraday_limit = last_bar_of_day[t]
        tau_max = min(t + max_horizon, intraday_limit)

        if tau_max <= t:
            t += 1
            continue

        triggered = False

        if direction == "long" and active_z[t] >= z_thresh:
            is_candidate[t] = True
            # Peak starts at entry price, not entry candle's high
            peak = p_start
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

            triggered = True
            t = max(tau_end + 1, t + cooldown_bars)

        elif direction == "short":
            is_short_impulse = active_z[t] <= -z_thresh
            is_box_break = box_exp_short[t] > 0.50
            is_absorption_trap = (thrust_5s[t] < -0.30) and (dist_low_15m[t] > 0.0)

            if is_short_impulse and is_box_break and (not is_absorption_trap):
                is_candidate[t] = True
                # Trough starts at entry price, not entry candle's low
                trough = p_start
                tau_end = tau_max

                for s in range(t + 1, tau_max + 1):
                    if lows[s] < trough:
                        trough = lows[s]
                    # Trailing bounce check against current bar's high
                    if (highs[s] - trough) >= vol_stop:
                        tau_end = s
                        break

                realized_mfe = (p_start - trough) / p_start * 10000.0
                mfe_bps[t] = realized_mfe
                end_indices[t] = tau_end
                if realized_mfe >= min_run_bps:
                    labels[t] = 1

                triggered = True
                t = max(tau_end + 1, t + cooldown_bars)

        if not triggered:
            t += 1

    # Alias active Z-score to 'z_impulse' for downstream compatibility
    return df.with_columns([
        pl.col(active_impulse_col).alias("z_impulse"),
        pl.Series("target_label", labels),
        pl.Series("target_mfe_bps", mfe_bps),
        pl.Series("target_end_idx", end_indices),
        pl.Series("is_candidate", is_candidate),
        pl.lit(direction).alias("momentum_direction"),
    ])
