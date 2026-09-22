import lightgbm as lgb
import polars as pl
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve


def evaluate_trading_thresholds(y_test, preds_prob, candidate_test_df):
    precisions, recalls, thresholds = precision_recall_curve(y_test, preds_prob)

    print("\n--- Model Performance Across Probability Thresholds ---")
    for thresh in [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        mask = preds_prob >= thresh
        trades = np.sum(mask)
        if trades > 0:
            win_rate = np.mean(y_test[mask] == 1) * 100
            # Average MFE (Maximum Favorable Excursion) captured
            avg_mfe = candidate_test_df.loc[mask, "target_mfe_bps"].mean()
            print(f"Threshold >= {thresh:.2f} | Trades: {trades:3d} ({trades / len(y_test) * 100:4.1f}%) | "
                  f"Win Rate: {win_rate:4.1f}% | Avg Expansion: {avg_mfe:+.2f} bps")
        else:
            print(f"Threshold >= {thresh:.2f} | Trades:   0")


def train_momentum_classifier(df: pl.DataFrame):
    # Select clean numeric features
    feature_cols = [
        "realized_vol_1m", "parkinson_vol", "active_seconds",
        "persistent_delta_3m", "dist_vwap_bps", "z_score_20m_prior"
    ]

    # Restrict to candidate setup bars (Z-score impulse triggered)
    candidate_bars = df.filter(pl.col("z_impulse") >= 1.5).to_pandas()

    if len(candidate_bars) == 0:
        print("No impulse candidate bars found.")
        return

    X = candidate_bars[feature_cols]
    y = candidate_bars["target_label"]

    # Time-series walk-forward split (e.g. 75% train, 25% test)
    split_idx = int(len(candidate_bars) * 0.75)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_meta_df = candidate_bars.iloc[split_idx:].reset_index(drop=True)

    # Set weight inversely proportional to class frequencies: 826 / 143 ≈ 5.77 (from the 1000 samples in our training set)
    pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-6)

    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,  # Shallow depth prevents overfitting on ~1,000 bars
        num_leaves=7,
        min_child_samples=15,  # Lower sample leaf limit for smaller candidate sets
        scale_pos_weight=pos_weight,  # Balances loss gradient
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    # Fit the model
    clf.fit(X_train, y_train)

    # Generate out-of-sample probabilities
    preds_prob = clf.predict_proba(X_test)[:, 1]
    print(f"Test ROC-AUC: {roc_auc_score(y_test, preds_prob):.4f}")
    print(classification_report(y_test, (preds_prob > 0.5).astype(int)))

    # Evaluate the edge across multiple thresholds
    evaluate_trading_thresholds(
            y_test.to_numpy(), preds_prob, candidate_test_df=test_meta_df
        )

    # Feature Importance
    importance = pl.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_
    }).sort("importance", descending=True)
    print("\nFeature Importance:\n", importance)





