import os
import polars as pl

DATA_DIR = "data"

def load_data_file(file_name: str) -> pl.DataFrame:
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing dataset file: {file_path}")
    return pl.read_csv(file_path)

def scan_data_file(file_name: str) -> pl.LazyFrame:
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing dataset file: {file_path}")
    return pl.scan_csv(file_path)