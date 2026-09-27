import lightgbm as lgb
import polars as pl
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve


def evaluate_trading_thresholds(y_test: np.ndarray, preds_prob: np.ndarray, candidate_test_df, direction: str = "long"):
    print("\n" + "=" * 70)
    print(f"   {direction.upper()} MOMENTUM PERFORMANCE ACROSS PROBABILITY THRESHOLDS   ")
    print("=" * 70)

    base_rate = np.mean(y_test) * 100
    print(f"Test Sample Size: {len(y_test)} bars | Unconditional Win Rate: {base_rate:.1f}%\n")

    mfe_arr = candidate_test_df["target_mfe_bps"].to_numpy()

    for thresh in [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        mask = preds_prob >= thresh
        trades = int(np.sum(mask))

        if trades > 0:
            win_rate = np.mean(y_test[mask] == 1) * 100
            avg_mfe = float(np.mean(mfe_arr[mask]))
            print(
                f"Cutoff >= {thresh:.2f} | Trades: {trades:3d} ({trades / len(y_test) * 100:4.1f}%) | "
                f"Win Rate: {win_rate:5.1f}% | Avg Expansion: +{avg_mfe / 100:.2f}% (+{avg_mfe:.1f} bps)"
            )
        else:
            print(f"Cutoff >= {thresh:.2f} | Trades:   0")
    print("=" * 70)


def filter_non_overlapping_candidates(
        candidate_df: pl.DataFrame, cooldown_bars: int = 12
) -> pl.DataFrame:
    """Suppresses overlapping impulse triggers. Once a candidate triggers, no other

    candidate can trigger within `cooldown_bars` (minutes) on the same trading
    day.
    """
    if candidate_df.is_empty():
        return candidate_df

    # Compute exact minute-of-day integer (09:30 ET -> 9*60 + 30 = 570)
    # This makes the cooldown immune to gaps or missing bars.
    df_with_time = candidate_df.with_columns(
        (
                pl.col("ts_event").dt.hour().cast(pl.Int32) * 60 + pl.col("ts_event").dt.minute().cast(pl.Int32)
        ).alias("minute_of_day")
    )

    trade_dates = df_with_time["trade_date"].to_numpy()
    minute_indices = df_with_time["minute_of_day"].to_numpy()

    # Fast single-pass loop to isolate non-overlapping setup entries
    keep_mask = np.zeros(len(df_with_time), dtype=bool)
    last_accepted_minute = -9999
    last_date = None

    for i, (date, minute_val) in enumerate(zip(trade_dates, minute_indices)):
        if date != last_date:
            # Fresh trading session: always accept the first valid trigger of the day
            keep_mask[i] = True
            last_accepted_minute = minute_val
            last_date = date
        elif minute_val >= (last_accepted_minute + cooldown_bars):
            # Cooldown has elapsed within the same trading session
            keep_mask[i] = True
            last_accepted_minute = minute_val

    # Return filtered DataFrame and drop the temporary helper column
    return df_with_time.filter(pl.Series(keep_mask)).drop("minute_of_day")


def train_momentum_classifier(df: pl.DataFrame, direction: str = "long", cooldown_bars: int = 12):
    feature_cols = [
        # Macro volatility & structure (15-min strategy)
        "realized_vol_1m",
        "parkinson_vol",
        "active_seconds",
        # Order Flow & Price Velocity
        "persistent_delta_3m",
        "cum_ret_3m",  # Captures ongoing cascades past single-bar variance spikes
        # Structural Location & Context
        "dist_vwap_bps",
        "z_score_20m_prior",
        "dist_lowest_15m_bps",  # Critical for short breakdowns: <= 0
        "dist_highest_15m_bps",  # Critical for long breakouts: >= 0

        # Sub-minute burst & coiling features (Calibrated 30s/5s strategy)
        "min_intra_compression_ratio",  # Identifies if price coiled in the last 30s
        "max_intra_z_burst_long",  # Captures explosive upward micro-moves
        "min_intra_z_burst_short",  # Captures explosive downward micro-moves
        "max_intra_box_expansion_long",  # Distance escaped above 30s consolidation box
        "max_intra_box_expansion_short",  # Distance escaped below 30s consolidation box
        "exit_flow_thrust_5s",  # Order book taker aggression at bar close
        "session_drift_bps",  # session trend to stop shorting into strong momentum
    ]

    # Filter candidate bars according to direction
    if direction == "long":
        raw_candidates = df.filter(pl.col("z_impulse") >= 1.5)  # positive vol impulse
    else:  # short
        raw_candidates = df.filter(pl.col("z_impulse") <= -1.5,  # negative vol impulse
                                   pl.col("max_intra_box_expansion_short") > 0.5,  # price below 30s consolidation
                                   # Exclude bars where heavy selling (< -0.30) fails to breach/threaten 15m support (> 0 bps)
                                   ~((pl.col("exit_flow_thrust_5s") < -0.30) & (pl.col("dist_lowest_15m_bps") > 0.0)),
                                   )

    # Apply the non-overlapping candidate filter
    candidate_pl = filter_non_overlapping_candidates(
        raw_candidates, cooldown_bars=cooldown_bars
    )
    print(f"\n[{direction.upper()}] Candidate Overlap Filter (Cooldown = {cooldown_bars}m):")
    print(
        f"  Raw triggers: {len(raw_candidates):,} -> Independent events:"
        f" {len(candidate_pl):,} (Suppressed {len(raw_candidates) - len(candidate_pl):,} duplicates)"
    )

    if len(candidate_pl) < 20:
        print(
            f"[{direction.upper()}] Insufficient candidate bars"
            f" ({len(candidate_pl)}). Ingest more days or relax threshold."
        )
        return None

    candidate_bars = candidate_pl.to_pandas()

    # Guard against missing feature columns
    missing_cols = [c for c in feature_cols if c not in candidate_bars.columns]
    if missing_cols:
        raise KeyError(
            f"Missing engineered features in DataFrame: {missing_cols}. "
            "Ensure signals.py has been run with updated features."
        )

    x = candidate_bars[feature_cols]
    y = candidate_bars["target_label"]

    # Chronological 75/25 split (Upgrade to rolling walk-forward validation with larger datasets)
    split_idx = int(len(candidate_bars) * 0.75)
    X_train, X_test = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_meta_df = candidate_bars.iloc[split_idx:].reset_index(drop=True)

    n_pos_train = int(np.sum(y_train == 1))
    n_neg_train = int(np.sum(y_train == 0))
    n_pos_test = int(np.sum(y_test == 1))
    n_neg_test = int(np.sum(y_test == 0))

    print(f"\n--- [{direction.upper()}] DATASET PARTITION CHECK ---")
    print(
        f"Train Pool: {len(y_train)} bars | Positives (Winners): {n_pos_train} |"
        f" Negatives (Stops): {n_neg_train}"
    )
    print(
        f"Test  Pool: {len(y_test)} bars | Positives (Winners): {n_pos_test} |"
        f" Negatives (Stops): {n_neg_test}"
    )

    if n_pos_train == 0:
        print(
            f"Cannot train: 0 positive momentum episodes in training set."
        )
        return None

    # Handle class weights safely
    pos_weight = n_neg_train / max(n_pos_train, 1)
    # Dynamic child samples: adapts cleanly from 23 days to 80 days
    min_child = max(10, min(30, int(len(X_train) * 0.05)))

    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.03,  # set to 0.02 for larger datasets
        max_depth=3,  # set to 4 with more data, increased depth to capture (volatility x location) interactions
        num_leaves=7,  # set to 15 for larger datasets, 2^depth - 1 capacity
        min_child_samples=min_child,  # dynamically scaled to stabilize splits on larger sample
        scale_pos_weight=pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,  # -1 = Fatal only (silences Info and Warnings)
        force_col_wise=True,  # Removes the threading evaluation overhead prompt
    )

    clf.fit(X_train, y_train)

    # ROC-AUC Metric calculation
    preds_prob = clf.predict_proba(X_test)[:, 1]
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, preds_prob)
        print(f"Test Out-Of-Sample ROC-AUC: {auc:.4f}")
        evaluate_trading_thresholds(
            y_test.to_numpy(), preds_prob, test_meta_df, direction=direction
        )
    else:
        print(
            f"\n[Warning] Only class '{y_test.iloc[0]}' present in test set."
            " ROC-AUC is undefined for this test window."
        )
        print(
            "Candidate bars in test window were either all stops or all runners."
        )

    importance = pl.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    }).sort("importance", descending=True)
    print(f"\nFeature Importance ({direction.upper()}):\n", importance)

    return clf
