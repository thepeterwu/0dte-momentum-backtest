from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import databento as db
import polars as pl


def _convert_single_file(dbn_file: Path) -> Path:
    base_name = dbn_file.name.replace(".dbn.zst", "")
    parquet_file = dbn_file.parent / f"{base_name}.parquet"

    if not parquet_file.exists():
        print(f"Converting {dbn_file.name}...")
        store = db.DBNStore.from_file(dbn_file)
        store.to_parquet(parquet_file)
    return parquet_file


def get_or_convert_parquet_files(data_dir: str = "data", max_workers: int = 4) -> list[Path]:
    """
    Returns existing .parquet files. If none are found, searches for .dbn.zst
    files and converts them in parallel before returning the list.
    """
    data_path = Path(data_dir)

    # 1. Check if Parquet files already exist (e.g. copied from another machine)
    parquet_files = sorted(data_path.glob("*.mbo.parquet"))
    if parquet_files:
        print(f"Found {len(parquet_files)} existing Parquet files. Skipping conversion.")
        return parquet_files

    # 2. If no Parquet files, fall back to discovering and converting DBN files
    dbn_files = sorted(data_path.glob("*.mbo.dbn.zst"))
    if not dbn_files:
        raise FileNotFoundError(
            f"No processed '*.mbo.parquet' or raw '*.mbo.dbn.zst' files found in '{data_dir}'"
        )

    print(f"Found {len(dbn_files)} DBN files. Converting in parallel (workers={max_workers})...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        parquet_files = list(executor.map(_convert_single_file, dbn_files))

    return sorted(parquet_files)

def load_mbo_trades(parquet_path: Path) -> pl.DataFrame:
    """
    Lazily scans the Parquet file, extracts executions (trades/fills),
    scales price to floating-point dollars, and returns a sorted DataFrame.
    Databento schema: https://databento.com/docs/schemas-and-data-formats/mbo#fields-mbo?historical=python&live=python&reference=python
    """
    q = (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("action").is_in(["T", "F"]))  # trades and fills
        .select([
            pl.col("ts_event").cast(pl.Datetime("ns")),
            pl.col("side"),
            (pl.col("price") / 1e9).alias("price"),  # price scaling
            pl.col("size"),
        ])
        .sort("ts_event")
    )
    return q.collect()

