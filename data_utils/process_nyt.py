import pandas as pd
import logging

IN_PQT_FILE = "/Volumes/NVME_EXT/GitHub_Repos/Fin-FFB/data/bulk/nyt_100y_1m.parquet"
OUT_PQT_FILE = "/Volumes/NVME_EXT/GitHub_Repos/Fin-FFB/data/nyt_100y_80k_ready.parquet"

MAX_ROWS = 80_000


if __name__ == "__main__":
    logging.info(f"Reading {IN_PQT_FILE}")
    df: pd.DataFrame = pd.read_parquet(IN_PQT_FILE)
    df = df.drop(columns=["date", "__index_level_0__"])

    logging.info(f"Combining 'headline' and 'abstract' into 'text'")
    df["text"] = df.apply(lambda row: f"{row['headline']}\n{row['abstract']}", axis=1)
    df = df.drop(columns=["headline", "abstract"])
    df = df.dropna(subset=["text"])

    logging.info(f"DF size after clean: {len(df)}")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df[0:MAX_ROWS]
    logging.info(f"DF size after cut: {len(df)}")
    
    logging.info(str(df.head(n=10)))

    logging.info(f"Saving to {OUT_PQT_FILE}")
    df.to_parquet(OUT_PQT_FILE)

