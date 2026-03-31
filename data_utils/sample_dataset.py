"""
Script to sample the origianl dataset to allow for quciker prototyping.
Sample and extract only a small ammount of rows (30k for ~30 minute training for larger mode).
"""

import pandas as pd
import logging

MAX_ROWS = 2000
SRC_PQRT_FILE = "/Volumes/NVME_EXT/GitHub_Repos/Fin-FFB/data/fin_ffb_200k_ready.parquet"
DEST_PQRT_FILE = "/Volumes/NVME_EXT/GitHub_Repos/Fin-FFB/data/fin_ffb_2k_proto_ready.parquet"

if __name__ == "__main__":
    logging.info(f"Reading {SRC_PQRT_FILE}")
    df: pd.DataFrame = pd.read_parquet(SRC_PQRT_FILE)
    df = df.sample(n=MAX_ROWS, random_state=42)

    logging.info(f"Writing {DEST_PQRT_FILE}")
    df.to_parquet(DEST_PQRT_FILE)
    logging.info("Done.")
