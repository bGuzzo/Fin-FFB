import torch
import yaml
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
from typing import Optional, List, Dict


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


class NYTDataset(Dataset):
    """
    Dataset for New York Times 100 Years of News Headlines.
    Concatenates 'headline' and 'abstract'.
    """

    def __init__(self, split: str = "all"):
        self.ds = load_dataset(
            # "bguzzo2k/nyt_100y_news_headlines", 
            "/Volumes/NVME_EXT/Datasets/nyt_100y_news_headlines",
            split=split, streaming=False
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        headline = item.get("headline", "") or ""
        abstract = item.get("abstract", "") or ""
        # Combine headline and abstract for context
        text = f"{headline}. {abstract}"
        return {"text": text}


class EDGARDataset(Dataset):
    """
    Dataset for EDGAR-CORPUS annual reports.
    Concatenates available sections (1-15).
    """

    def __init__(self, split: str = "all"):
        # Loading 'full' config for EDGAR
        self.ds = load_dataset(
            # "eloukas/edgar-corpus", 
            "/Volumes/NVME_EXT/Datasets/eloukas_edgar-corpus",
            # config="full", 
            split=split, streaming=False
        )
        self.sections = [f"section_{i}" for i in range(1, 16)]
        # Add sub-sections commonly found
        self.sections.extend(
            ["section_1A", "section_1B", "section_7A", "section_9A", "section_9B"]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        texts = []
        for sec in self.sections:
            content = item.get(sec, "")
            if content and isinstance(content, str):
                texts.append(content)

        # Join sections with newlines
        full_text = "\n".join(texts)
        return {"text": full_text}


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
    # nyt_ds = NYTDataset(split="all")
    
    # logging.info(len(nyt_ds))
    # logging.info(nyt_ds[0])

    edgar_ds = EDGARDataset()
    logging.info(len(edgar_ds))
    logging.info(edgar_ds[0])