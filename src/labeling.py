import polars as pl
import numpy as np

def label_momentum_episodes(
        df: pl.DataFrame,
        k_baseline: int = 15,
        z_thresh: float = 2.0,
        trail_mult: float = 1.5,
        min_run_bps: float = 8.0,
        max_horizon: int = 10,
) -> pl.DataFrame:

    # Calculate strictly lagged baseline
    df = (
        df.with_columns([
            (pl.col("close") / pl.col("close").shift(1)).log().alias("ret_1m")
        ])
        .with_columns([
            pl.col("ret_1m").shift(1).rolling_mean(k_baseline).alias("mu_base"),
            pl.col("ret_1m").shift(1).rolling_std(k_baseline).alias("std_base"),
        ])
        .with_columns([
            ((pl.col("ret_1m") - pl.col("mu_base")) / (pl.col("std_base") + 1e-6)).alias("z_impulse")
        ])
    )

    closes = df["close"].to_numpy()
    z_scores = df["z_impulse"].to_numpy()
    std_bases = df["std_base"].to_numpy()
    n = len(df)
    labels = np.zeros(n, dtype=np.int32)
    mfe_bps = np.zeros(n, dtype=np.float64)
    end_indices = np.full(n, -1, dtype=np.int32)

    # Path-dependent evaluation of momentum episodes
    for t in range(k_baseline + 1, n - max_horizon):
        if z_scores[t] >= z_thresh:  # Long impulse trigger
            p_start = closes[t]
            vol_stop = trail_mult * std_bases[t] * p_start
            peak = p_start
            tau_end = t + max_horizon

            for s in range(t + 1, min(t + max_horizon + 1, n)):
                if closes[s] > peak:
                    peak = closes[s]
                # Termination check: Trailing pullback from peak
                if (peak - closes[s]) >= vol_stop:
                    tau_end = s
                    break

            realized_mfe = (peak - p_start) / p_start * 10000
            mfe_bps[t] = realized_mfe
            end_indices[t] = tau_end

            if realized_mfe >= min_run_bps:
                labels[t] = 1

    return df.with_columns([
        pl.Series("target_label", labels),
        pl.Series("target_mfe_bps", mfe_bps),
        pl.Series("target_end_idx", end_indices),
    ])
