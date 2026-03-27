"""
Pandas-to-PyTorch Dataset Adapter.

This module provides an interface between persistent storage (Apache Parquet) 
and the PyTorch ecosystem, allowing pandas DataFrames to be consumed by 
DataLoaders without intermediate conversion steps.
"""

import pandas as pd
from torch.utils.data import Dataset
import logging
from typing import Dict


PRQT_FILE = "/Volumes/NVME_EXT/GitHub_Repos/Fin-FFB/data/fin_ffb_200k_ready.parquet"
TEXT_COL = "text"


class PdDataset(Dataset):
    """
    An adapter class that treats a Pandas DataFrame as a native PyTorch Dataset.

    This implementation loads a Parquet file into memory and provides indexed
    access to individual records. It is designed to work seamlessly with the
    JitCollator to provide raw text strings for on-the-fly tokenization.

    Attributes:
        df (pd.DataFrame): The underlying data structure containing the text corpus.
        text_col (str): The name of the column containing the raw text data.
    """

    def __init__(self, prqt_f: str = PRQT_FILE, text_col: str = TEXT_COL):
        """
        Loads the dataset from the specified Parquet file.

        Args:
            prqt_f (str): Absolute path to the .parquet file.
            text_col (str): Column name containing the text to be modeled.
        """
        logging.info(f"Initiating dataset load from: {prqt_f}")
        self.df = pd.read_parquet(prqt_f)
        self.text_col = text_col
        logging.info(f"Dataset successfully loaded. Total record count: {len(self.df)}")

    def __len__(self):
        """
        Returns the total number of records in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, str]:
        """
        Retrieves a single record from the DataFrame by index.

        Args:
            idx (int): The integer index of the record to retrieve.

        Returns:
            Dict[str, str]: A dictionary containing the raw text under the 'text' key.
        """
        item = self.df.iloc[idx]
        return {"text": item[self.text_col]}

# Test only
if __name__ == "__main__":
    dataset = PdDataset()
    logging.info(len(dataset))
    logging.info(dataset[0])
    logging.info(dataset[1])
    logging.info(dataset[2])
