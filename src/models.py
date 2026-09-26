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

    for thresh in [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        mask = preds_prob >= thresh
        trades = int(np.sum(mask))

        if trades > 0:
            win_rate = np.mean(y_test[mask] == 1) * 100
            avg_mfe = candidate_test_df.loc[mask, "target_mfe_bps"].mean()
            print(
                f"Cutoff >= {thresh:.2f} | Trades: {trades:3d} ({trades / len(y_test) * 100:4.1f}%) | "
                f"Win Rate: {win_rate:5.1f}% | Avg Expansion: +{avg_mfe / 100:.2f}% (+{avg_mfe:.1f} bps)"
            )
        else:
            print(f"Cutoff >= {thresh:.2f} | Trades:   0")
    print("=" * 70)


def train_momentum_classifier(df: pl.DataFrame, direction: str = "long"):
    feature_cols = [
        # Intra-bar Microstructure & Volatility
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
    ]

    # Filter candidate impulse bars according to direction
    if direction == "long":
        candidate_bars = df.filter(pl.col("z_impulse") >= 1.5).to_pandas()
    else:
        candidate_bars = df.filter(pl.col("z_impulse") <= -1.5).to_pandas()

    if len(candidate_bars) == 0:
        print(f"No {direction} impulse candidate bars found.")
    elif len(candidate_bars) < 20:
        print(
            f"[{direction.upper()}] Insufficient candidate bars"
            f" ({len(candidate_bars)}). Adjust threshold or ingest more days."
        )

    X = candidate_bars[feature_cols]
    y = candidate_bars["target_label"]

    # Chronological 75/25 split (Upgrade to rolling walk-forward validation with larger datasets)
    split_idx = int(len(candidate_bars) * 0.75)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
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

    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.03,     # set to 0.02 for larger datasets
        max_depth=3,            # set to 4, increased depth to capture (volatility x location) interactions
        num_leaves=7,           # set to 15 for larger datasets, 2^depth - 1 capacity
        min_child_samples=15,   # set to 30 to stabilize splits on larger sample
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





