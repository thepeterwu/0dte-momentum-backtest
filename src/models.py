import lightgbm as lgb
import polars as pl
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve


#   TODO: Implement Max drawdown
#       : Add walk forward testing


def evaluate_trading_thresholds(
        y_test: np.ndarray,
        preds_prob: np.ndarray,
        candidate_test_df,
        direction: str = "long",
):
    print("\n" + "=" * 88)
    print(
        f"   {direction.upper()} MOMENTUM PERFORMANCE ACROSS PROBABILITY"
        " THRESHOLDS   "
    )
    print("=" * 88)

    base_rate = np.mean(y_test) * 100
    print(
        f"Test Sample Size: {len(y_test)} bars | Unconditional Win Rate:"
        f" {base_rate:.1f}%\n"
    )

    mfe_arr = candidate_test_df["target_mfe_bps"].to_numpy()
    pnl_arr = (
        candidate_test_df["target_realized_pnl_bps"].to_numpy()
        if "target_realized_pnl_bps" in candidate_test_df.columns
        else np.zeros(len(candidate_test_df))
    )

    for thresh in [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        mask = preds_prob >= thresh
        trades = int(np.sum(mask))

        if trades > 0:
            win_rate = np.mean(y_test[mask] == 1) * 100
            avg_pnl = float(np.mean(pnl_arr[mask]))
            avg_mfe = float(np.mean(mfe_arr[mask]))
            print(
                f"Cutoff >= {thresh:.2f} | Trades: {trades:3d} ({trades / len(y_test) * 100:4.1f}%) | "
                f"Win Rate: {win_rate:5.1f}% | "
                f"Realized PnL: {avg_pnl / 100:+.2f}% ({avg_pnl:+.1f} bps) | "
                f"Avg MFE: +{avg_mfe / 100:.2f}% (+{avg_mfe:.1f} bps)"
            )
        else:
            print(f"Cutoff >= {thresh:.2f} | Trades:   0")
    print("=" * 88)


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

    # Apply the non-overlapping candidate filter
    candidate_pl = df.filter(
        pl.col("is_candidate") & (pl.col("momentum_direction") == direction)
    )

    # 2. Count raw unconstrained impulse triggers across the dataset
    z_col = "z_impulse"
    raw_condition = (
        (pl.col(z_col) >= 2.0)
        if direction == "long"
        else (pl.col(z_col) <= -2.0)
    )
    n_raw_triggers = df.filter(raw_condition).height
    n_independent = len(candidate_pl)
    n_suppressed = max(0, n_raw_triggers - n_independent)

    print(
        f"\n[{direction.upper()}] Candidate Overlap Filter (Cooldown ="
        f" {cooldown_bars}m):"
    )
    print(
        f"  Raw triggers: {n_raw_triggers:,} -> Independent events:"
        f" {n_independent:,} (Suppressed {n_suppressed:,} duplicates)"
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

    # # Handle class weights safely
    # pos_weight = n_neg_train / max(n_pos_train, 1)
    pos_weight = 1.0    # use true posterior probability
    # Dynamic child samples: adapts cleanly from 23 days to 80 days
    min_child = max(10, min(30, int(len(X_train) * 0.05)))

    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.03,  # set to 0.02 for larger datasets
        max_depth=3,  # set to 4 with more data, increased depth to capture (volatility x location) interactions
        num_leaves=7,  # set to 15 for larger datasets, 2^depth - 1 capacity
        min_child_samples=min_child,  # dynamically scaled to stabilize splits on larger sample
        scale_pos_weight=pos_weight,
        subsample_freq=1,
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
