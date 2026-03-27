import logging
import pandas as pd
import pyarrow.parquet as pq
import numpy as np


IN_PQT_FILE = "/Volumes/NVME_EXT/GitHub_Repos/Fin-FFB/data/bulk/edgar_corp_full.parquet"
OUT_PQT_FILE = "/Volumes/NVME_EXT/GitHub_Repos/Fin-FFB/data/ready/edgar_corp_80k_ready.parquet"

MAX_ROWS = 80_000
BATCH_SIZE = 1000

COLUMS_TO_DROP = [
    "filename",
    "cik",
    "year"
]

if __name__ == "__main__":
    """"
    Extract one section per row of the original dataset (220k rows).
    """

    logging.info(f"Reading {IN_PQT_FILE}")
    prqt_file = pq.ParquetFile(IN_PQT_FILE)

    # Initalize DF
    final_df = pd.DataFrame({"text": []})
    batch_idx = 0
    for batch in prqt_file.iter_batches(batch_size=BATCH_SIZE):
        df_batch: pd.DataFrame = batch.to_pandas()
        df_batch = df_batch.drop(columns=COLUMS_TO_DROP)
        
        logging.info(f"[{batch_idx}] Extracting one column value per row")
        new_df = pd.DataFrame({"text": []})
        cols = df_batch.columns
        rand_idxs = np.random.randint(0, len(cols), size=len(df_batch))
        new_df["text"] = df_batch[cols].values[np.arange(len(df_batch)), rand_idxs]
        logging.info(f"[{batch_idx}] Extracted batch DF of size {len(new_df)}")
        
        # Drop empty strings and nan
        new_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        new_df = new_df.dropna()
        logging.info(f"[{batch_idx}] DF size after clean {len(new_df)}")

        # Join data
        final_df = pd.concat([final_df, new_df], ignore_index=True)
        logging.info(f"[{batch_idx}] Joined data. Current len: {len(final_df)}")
        
        if len(final_df) > MAX_ROWS:
            logging.info(f"[{batch_idx}] Current len {len(final_df)} exceed max len of {MAX_ROWS}. Saving the file")
            break
            
        batch_idx += 1
    
    # Shuffle and cut
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    final_df = final_df[0:MAX_ROWS]


    logging.info(str(final_df.head(n=10)))
    logging.info(str(final_df.tail(n=10)))
    
    logging.info(f"Saving to {OUT_PQT_FILE}")
    final_df.to_parquet(OUT_PQT_FILE)
    







    print(df_batch.head())
    print(df_batch.columns)
