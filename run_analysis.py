from src.data_loader import load_mbo_trades
from src.signals import generate_micro_bars_and_signals
import scipy.stats as stats
import numpy as np


def evaluate_signals(df):
    # Extract arrays
    imb = df["trade_imbalance"].to_numpy()
    zscore = df["z_score_breakout"].to_numpy()
    fwd_ret = df["fwd_return_30s"].to_numpy()

    # 1. Calculate Information Coefficients (Spearman Rank Correlation)
    ic_imb, p_imb = stats.spearmanr(imb, fwd_ret)
    ic_z, p_z = stats.spearmanr(zscore, fwd_ret)

    print("==================================================")
    print("           SIGNAL PREDICTIVE POWER (IC)           ")
    print("==================================================")
    print(f"Trade Imbalance IC : {ic_imb:+.4f} (p-value: {p_imb:.2e})")
    print(f"Z-Score Breakout IC: {ic_z:+.4f} (p-value: {p_z:.2e})")
    print("==================================================")

    # 2. Check Conditional Edge (in Basis Points)
    # Long threshold: Imbalance > 0.6 | Short threshold: Imbalance < -0.6
    long_mask = imb > 0.6
    short_mask = imb < -0.6

    avg_long_bps = np.mean(fwd_ret[long_mask]) * 10000 if np.any(long_mask) else 0.0
    avg_short_bps = np.mean(fwd_ret[short_mask]) * 10000 if np.any(short_mask) else 0.0

    print("\n--- Conditional Forward 30s Edge ---")
    print(f"Sample Count (Imbalance >  0.6): {np.sum(long_mask)}")
    print(f"Mean Return  (Imbalance >  0.6): {avg_long_bps:+.2f} bps")
    print(f"Sample Count (Imbalance < -0.6): {np.sum(short_mask)}")
    print(f"Mean Return  (Imbalance < -0.6): {avg_short_bps:+.2f} bps")


def main():
    parquet_path = "data/raw_mbo.parquet"
    print("Loading filtered executions from Parquet...")
    trades_df = load_mbo_trades(parquet_path)

    print("Constructing 5-second bars and signals...")
    feature_df = generate_micro_bars_and_signals(trades_df)

    print(f"Processed {len(feature_df):,} bars. Running diagnostics...")
    evaluate_signals(feature_df)


if __name__ == "__main__":
    main()