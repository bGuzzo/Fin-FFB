import pandas as pd
import logging

IN_PQT_FILE = "/Volumes/NVME_EXT/GitHub_Repos/Fin-FFB/data/nyt_100y_05m.parquet"
OUT_PQT_FILE = "/Volumes/NVME_EXT/GitHub_Repos/Fin-FFB/data/nyt_100y_05m_ready.parquet"

if __name__ == "__main__":
    logging.info(f"Reading {IN_PQT_FILE}")
    df: pd.DataFrame = pd.read_parquet(IN_PQT_FILE)
    df = df.drop(columns=["date", "__index_level_0__"])

    logging.info(f"Combining 'headline' and 'abstract' into 'text'")
    df["text"] = df.apply(lambda row: f"{row['headline']}\n{row['abstract']}", axis=1)
    df = df.drop(columns=["headline", "abstract"])
    df = df.dropna(subset=["text"])


    logging.info(f"Saving to {OUT_PQT_FILE}")
    df.to_parquet(OUT_PQT_FILE)

