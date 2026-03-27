import pandas as pd
import pyarrow.parquet as pq
import logging
import numpy as np

PRQT_IN_F_PATHS = [
    "/Volumes/NVME_EXT/GitHub_Repos/Fin-FFB/data/ready/edgar_corp_80k_ready.parquet",
    "/Volumes/NVME_EXT/GitHub_Repos/Fin-FFB/data/ready/nyt_100y_80k_ready.parquet",
    "/Volumes/NVME_EXT/GitHub_Repos/Fin-FFB/data/ready/wiki_en_40k_ready.parquet",
]

PRQT_OUT_F_PATH = "/Volumes/NVME_EXT/GitHub_Repos/Fin-FFB/data/fin_ffb_200k_ready.parquet"

if __name__ == "__main__":
    logging.info(f"Reading {PRQT_IN_F_PATHS}")
    dfs = [pd.read_parquet(f) for f in PRQT_IN_F_PATHS]
    new_df: pd.DataFrame = pd.DataFrame({"text": []})

    logging.info(f"Loaded {len(dfs)} files, concatenating...")
    # df = pd.concat(dfs, ignore_index=True)
    for partial_df in dfs:
        new_df = pd.concat([new_df, partial_df], ignore_index=True)
        del partial_df
    del dfs
    
    
    # Drop empty strings and nan
    new_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    new_df = new_df.dropna()

    # Strip strings
    new_df["text"] = new_df["text"].str.strip()
    
    # Shuffle
    new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)
    logging.info(f"DF size after clean: {len(new_df)}")
    
    logging.info(f"Saving to {PRQT_OUT_F_PATH}")
    new_df.to_parquet(PRQT_OUT_F_PATH)