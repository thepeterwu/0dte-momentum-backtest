import polars as pl

def load_mbo_trades(parquet_path: str) -> pl.DataFrame:
    """
    Lazily scans the Parquet file, extracts executions (trades/fills),
    scales price to floating-point dollars, and returns a sorted DataFrame.
    Databento schema: https://databento.com/docs/schemas-and-data-formats/mbo#fields-mbo?historical=python&live=python&reference=python
    """
    q = (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("action").is_in(["T", "F"])) # trades and fills
        .select([
            pl.col("ts_event").cast(pl.Datetime("ns")),
            pl.col("side"),
            (pl.col("price") / 1e9).alias("price"), # price scaling
            pl.col("size"),
        ])
        .sort("ts_event")
    )
    return q.collect()