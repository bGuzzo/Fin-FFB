import logging
import pandas as pd
import pyarrow.parquet as pq
import numpy as np


IN_PQT_FILE = "/Volumes/NVME_EXT/GitHub_Repos/Fin-FFB/data/edgar_corp_full.parquet"
OUT_PQT_FILE = "/Volumes/NVME_EXT/GitHub_Repos/Fin-FFB/data/edgar_corp_1m_ready.parquet"

MAX_ROWS = 1_000_000
BATCH_SIZE = 8096

COLUMS_TO_DROP = [
    "filename",
    "cik",
    "year"
]

if __name__ == "__main__":
    logging.info(f"Reading {IN_PQT_FILE}")
    prqt_file = pq.ParquetFile(IN_PQT_FILE)

    # Initalize DF
    final_df = pd.DataFrame({"text": []})
    for batch in prqt_file.iter_batches(batch_size=BATCH_SIZE):
        df: pd.DataFrame = batch.to_pandas()
        df = df.drop(columns=COLUMS_TO_DROP)
        
        df = df.melt(value_name="text", var_name="oring_col")
        df = df.drop(columns=["oring_col"])

        # Drop empty strings and nan
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        df = df.dropna()

        final_df = pd.concat([final_df, df], ignore_index=True)
        logging.info(f"Melted dataset, total len: {len(final_df)}")
        
        if len(final_df) > MAX_ROWS:
            logging.info(f"Current len {len(final_df)} exceed max len of {MAX_ROWS}. Saving the file")
            break
    
    # Shuffle and cut
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    final_df = final_df[0:MAX_ROWS]


    logging.info(str(final_df.head(n=10)))
    logging.info(str(final_df.tail(n=10)))
    
    logging.info(f"Saving to {OUT_PQT_FILE}")
    final_df.to_parquet(OUT_PQT_FILE)
    







    print(df.head())
    print(df.columns)
