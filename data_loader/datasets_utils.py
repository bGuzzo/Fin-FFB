import torch
import yaml
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
from typing import Optional, List, Dict
import pandas as pd
import logging


class MockDataset(Dataset):
    """
    A mock dataset that loads sentences from a YAML configuration file.
    Useful for testing the training pipeline without large external datasets.
    """

    def __init__(self, config_path: str = "config/mock_data.yaml"):
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        self.sentences = data.get("sentences", [])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {"text": self.sentences[idx]}

NYT_PRQT_FILE = "/Volumes/NVME_EXT/GitHub_Repos/Fin-FFB/data/nyt_100y_05m_ready.parquet"
class NYTDataset(Dataset):
    """
    Dataset for New York Times 100 Years of News Headlines.
    Read as pandas DF.
    """

    def __init__(self, prqt_file: str = NYT_PRQT_FILE):
        logging.info(f"Loading NYT dataset from {prqt_file}")
        self.df: pd.DataFrame = pd.read_parquet(prqt_file)
        logging.info(f"Loaded {len(self.df)} rows")

    def __len__(self):
        return len(self.df)

    # TODO fix it to be string only
    def __getitem__(self, idx) -> Dict[str, str]:
        item = self.df.iloc[idx]
        return {"text": item["text"]}

EDGAR_PRQT_FILE = "/Volumes/NVME_EXT/GitHub_Repos/Fin-FFB/data/edgar_corp_05m_ready.parquet"
class EDGARDataset(Dataset):
    """
    Dataset for EDGAR-CORPUS annual reports.
    Concatenates available sections (1-15).
    Read as pandas DF.
    """

    def __init__(self, prqt_file: str = EDGAR_PRQT_FILE):
        logging.info(f"Loading EDGAR dataset from {prqt_file}")
        self.df: pd.DataFrame = pd.read_parquet(prqt_file)
        logging.info(f"Loaded {len(self.df)} rows")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, str]:
        item = self.df.iloc[idx]
        return {"text": item["text"]}

# Revise to apply shaffle
class CombinedFinancialDataset(Dataset):
    """
    A simple wrapper to combine multiple datasets.
    """

    def __init__(self, datasets: list[Dataset]):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]  # pyright: ignore[reportArgumentType]
        self.total_len = sum(self.lengths)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_len:
            raise IndexError

        for i, length in enumerate(self.lengths):
            if idx < length:
                return self.datasets[i][idx]
            idx -= length
        raise IndexError


# Test only
if __name__ == "__main__":
    nyt_ds = NYTDataset()

    logging.info(len(nyt_ds))
    logging.info(nyt_ds[0])

    # edgar_ds = EDGARDataset()
    # logging.info(len(edgar_ds))
    # logging.info(edgar_ds[0])


