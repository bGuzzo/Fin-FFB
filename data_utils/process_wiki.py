import datasets
import pandas as pd
import logging
import numpy as np


HF_D_PATH = "/Volumes/NVME_EXT/Datasets/wikipedia"
PRQT_OUT_PATH = "/Volumes/NVME_EXT/GitHub_Repos/Fin-FFB/data/ready/wiki_en_40k_ready.parquet"
SEED = 42
ROW_COUNT = 40_000

if __name__ == "__main__":
    logging.info(f"Loading {HF_D_PATH}")
    hf_dataset = datasets.load_dataset(HF_D_PATH, split="all", streaming=False)
    hf_dataset = hf_dataset.shuffle(seed=SEED)
    hf_dataset = hf_dataset.select(range(ROW_COUNT))

    logging.info(f"Loading to pandas")
    df: pd.DataFrame = hf_dataset.to_pandas()
    df["text_new"] = df.apply(lambda row: f"{row['title']}\n{row['text']}", axis=1)
    df = df.drop(columns=["title", "text"])
    df = df.rename(columns={"text_new": "text"})
    df = df[["text"]]
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df = df.dropna()


    logging.info("DF Head")
    logging.info(df.head(10))
    logging.info("DF Tail")
    logging.info(df.tail(10))
    
    logging.info(f"Saving to {PRQT_OUT_PATH}")
    df.to_parquet(PRQT_OUT_PATH, compression="snappy", engine="pyarrow")
