import polars as pl

df = pl.read_parquet("data/xnas-itch-20260818.mbo.parquet", n_rows=50)
executions = df.filter(pl.col("action").is_in(["T", "F"]))
print(executions.select(["ts_event", "action", "side", "price", "size"]).head(10))