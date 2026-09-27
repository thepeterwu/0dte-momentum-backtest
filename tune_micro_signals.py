from pathlib import Path
import polars as pl
from src.data_loader import filter_regular_trading_hours, load_mbo_trades
from src.signals import sweep_micro_parameters


def main():
    data_dir = Path("data")
    # Use a representative sample of 5 to 10 trading days for calibration
    parquet_files = sorted(data_dir.glob("*.mbo.parquet"))[:10]

    all_sec_bars = []
    print(f"Aggregating 1s bars across {len(parquet_files)} days for tuning...")

    for path in parquet_files:
        trades = load_mbo_trades(path)
        trades_rth = filter_regular_trading_hours(trades, time_col="ts_event")

        # Aggregate to 1s bars
        sec_bars = trades_rth.group_by_dynamic("ts_event", every="1s").agg([
            pl.col("price").first().alias("open_1s"),
            pl.col("price").max().alias("high_1s"),
            pl.col("price").min().alias("low_1s"),
            pl.col("price").last().alias("close_1s"),
            pl.col("size").sum().alias("vol_1s"),
            pl.col("size")
            .filter(pl.col("side") == "B")
            .sum()
            .fill_null(0)
            .cast(pl.Float64)
            .alias("buy_vol_1s"),
            pl.col("size")
            .filter(pl.col("side") == "A")
            .sum()
            .fill_null(0)
            .cast(pl.Float64)
            .alias("sell_vol_1s"),
        ])
        all_sec_bars.append(sec_bars)

    combined_1s = pl.concat(all_sec_bars)
    print(f"Total 1-second rows evaluated: {len(combined_1s):,}")

    # Run the sweep
    results = sweep_micro_parameters(
        combined_1s,
        coil_range=[15, 20, 30, 45, 60],
        burst_range=[3, 5, 8, 10, 15],
    )

    print("\n" + "=" * 65)
    print("      TOP SUB-MINUTE PARAMETER COMBINATIONS BY SPREAD      ")
    print("=" * 65)
    print(results.head(10))
    print("=" * 65)


if __name__ == "__main__":
    main()
