from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import databento as db
import polars as pl


def _convert_single_file(dbn_file: Path) -> Path:
    base_name = dbn_file.name.replace(".dbn.zst", "")
    parquet_file = dbn_file.parent / f"{base_name}.parquet"

    print(f"Converting {dbn_file.name} -> {parquet_file.name}...")
    store = db.DBNStore.from_file(dbn_file)
    store.to_parquet(parquet_file)
    return parquet_file


def get_or_convert_parquet_files(data_dir: str = "data", max_workers: int = 4) -> list[Path]:
    """
    Checks for missing .parquet files corresponding to available .dbn.zst files,
    converts missing ones in parallel, and returns all available .parquet files.
    """
    data_path = Path(data_dir)
    dbn_files = sorted(data_path.glob("*.mbo.dbn.zst"))

    # Identify which DBN files actually need conversion
    dbn_to_convert = []
    for dbn_file in dbn_files:
        base_name = dbn_file.name.replace(".dbn.zst", "")
        parquet_file = data_path / f"{base_name}.parquet"
        if not parquet_file.exists():
            dbn_to_convert.append(dbn_file)

    # Convert only the missing files
    if dbn_to_convert:
        print(f"Found {len(dbn_to_convert)} new DBN file(s) to convert (workers={max_workers})...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(_convert_single_file, dbn_to_convert))
    else:
        if dbn_files:
            print("All DBN files already have matching Parquet files.")

    # Collect all existing and newly converted Parquet files
    parquet_files = sorted(data_path.glob("*.mbo.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No processed '*.mbo.parquet' or raw '*.mbo.dbn.zst' files found in '{data_dir}'"
        )

    print(f"Ready with {len(parquet_files)} total Parquet files.")
    return parquet_files


def load_mbo_trades(parquet_path: Path) -> pl.DataFrame:
    """
    Lazily scans the Parquet file, extracts executions (trades only),
    scales price to floating-point dollars, and returns a sorted DataFrame.
    Databento schema: https://databento.com/docs/schemas-and-data-formats/mbo#fields-mbo?historical=python&live=python&reference=python
    """
    q = (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("action").is_in(["T"]))  # trades only (prevent double counting)
        .select([
            pl.col("ts_event").cast(pl.Datetime("ns")),
            pl.col("side"),
            pl.col("price"),
            pl.col("size"),
        ])
        .sort("ts_event")
    )
    return q.collect()


def filter_regular_trading_hours(
        df: pl.DataFrame, time_col: str = "ts_event"
) -> pl.DataFrame:
    """Filters trades or bars strictly to US Regular Trading Hours (09:30:00 to 16:00:00 Eastern).

  Handles daylight saving transitions automatically.
  """
    # 1. Ensure UTC timezone awareness
    if df[time_col].dtype.time_zone is None:
        ts_expr = pl.col(time_col).dt.replace_time_zone("UTC")
    else:
        ts_expr = pl.col(time_col)

    # 2. Convert to US/Eastern and filter by time of day
    return (
        df.with_columns(
            ts_expr.dt.convert_time_zone("America/New_York").alias("ts_ny")
        )
        .filter(
            (pl.col("ts_ny").dt.time() >= pl.time(9, 30, 0))
            & (pl.col("ts_ny").dt.time() <= pl.time(16, 0, 0))
        )
        .drop("ts_ny")
    )
