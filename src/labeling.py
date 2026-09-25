import polars as pl
import numpy as np

def label_momentum_episodes(
        df: pl.DataFrame,
        direction: str = "long",  # "long" or "short"
        k_baseline: int = 15,  # strictly lagged window of K bars
        z_thresh: float = 2.0,  # 2 std dev of Z-score of normalized log return
        trail_mult: float = 1.5,
        min_run_bps: float = 4.0,
        max_horizon: int = 10,
) -> pl.DataFrame:
    """
        Labels path-dependent momentum episodes for either 'long' or 'short' directions.
    """

    if direction not in ["long", "short"]:
        raise ValueError("direction must be either 'long' or 'short'")

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
    mfe_bps = np.zeros(n, dtype=np.float64) # MFE (Maximum Favorable Excursion)
    end_indices = np.full(n, -1, dtype=np.int32)

    # Directional Path-dependent evaluation of momentum episodes
    for t in range(k_baseline + 1, n - max_horizon):
        p_start = closes[t]
        vol_stop = trail_mult * std_bases[t] * p_start

        if direction == "long" and z_scores[t] >= z_thresh:
            peak = p_start
            tau_end = t + max_horizon

            for s in range(t + 1, min(t + max_horizon + 1, n)):
                if closes[s] > peak:
                    peak = closes[s]
                # Termination: Pullback from peak
                if (peak - closes[s]) >= vol_stop:
                    tau_end = s
                    break

            realized_mfe = (peak - p_start) / p_start * 10000.0
            mfe_bps[t] = realized_mfe
            end_indices[t] = tau_end
            if realized_mfe >= min_run_bps:
                labels[t] = 1

        elif direction == "short" and z_scores[t] <= -z_thresh:
            trough = p_start
            tau_end = t + max_horizon

            for s in range(t + 1, min(t + max_horizon + 1, n)):
                if closes[s] < trough:
                    trough = closes[s]
                # Termination: Bounce upward from trough
                if (closes[s] - trough) >= vol_stop:
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
