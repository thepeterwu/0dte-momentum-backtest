import databento as db

# Reads .dbn or .dbn.zst and writes a compressed, columnar Parquet file
dbn_store = db.DBNStore.from_file("data/raw_mbo.dbn.zst")
dbn_store.to_parquet("data/raw_mbo.parquet")