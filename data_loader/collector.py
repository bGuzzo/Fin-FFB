"""
Collector strategy for the Fin-FFB training pipeline.

This module implements the JIT (Just-In-Time) loading and tokenization strategy
required for training on consumer hardware. It handles the combination of
diverse financial datasets and applies Masked Language Modeling (MLM)
transformations during batch collation.
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from typing import List, Dict, Any
from .datasets import NYTDataset, EDGARDataset, CombinedFinancialDataset


class FinancialDataCollector:
    """
    A callable class that handles JIT tokenization and MLM masking.

    Instead of pre-processing the entire dataset (which would exceed RAM on
    consumer hardware), this collector performs tokenization and masking
    on-the-fly for each batch retrieved by the DataLoader.

    Attributes:
        tokenizer: The ALBERT tokenizer used for text encoding.
        mlm_collator: Transformers utility for applying MLM masks.
        max_length: Maximum sequence length for truncation and padding.
    """

    def __init__(
        self,
        tokenizer_name: str = "albert-base-v2",
        mlm_probability: float = 0.15,
        max_length: int = 512,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # We use the standard Transformers collator for the actual masking logic
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=mlm_probability
        )

    def __call__(self, examples: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """
        Transforms a list of raw text strings into a batch of masked tensors for MLM.

        This method acts as the 'collate_fn' for the PyTorch DataLoader. It performs
        the following transformation pipeline:

        1.  **Text Extraction**: Pulls the raw 'text' strings from the batch of
            examples provided by the Dataset.
        2.  **JIT Tokenization**: Converts raw strings into integer token IDs
            using the ALBERT tokenizer. It applies padding and truncation to
            'max_length' to ensure uniform tensor shapes.
        3.  **Feature Reformatting**: Converts the batch-centric output of the
            tokenizer back into a list of individual example dictionaries,
            which is the format required by the Transformers DataCollator.
        4.  **MLM Masking**: Randomly masks tokens in the 'input_ids' based on
            'mlm_probability' and generates the corresponding 'labels'.

        Args:
            examples: A list of dictionaries, each containing a "text" key
                     with raw string content.

        Returns:
            A dictionary of PyTorch tensors containing:
            - "input_ids": LongTensor [batch, seq_len] where ~15% of tokens are
              replaced by [MASK], a random token, or kept as-is.
            - "attention_mask": LongTensor [batch, seq_len] (1 for real tokens,
              0 for padding).
            - "labels": LongTensor [batch, seq_len] containing the original
              token IDs for masked positions, and -100 for non-masked positions
              (ignored by CrossEntropyLoss).
        """
        # 1. Extract strings from the dataset items
        texts = [example["text"] for example in examples]

        # 2. Tokenize on-the-fly.
        # This converts raw text into input_ids, attention_mask, etc.
        tokenized_batch = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # 3. Format for the internal mlm_collator.
        # It expects a list of dictionaries, where each dict is a single example.
        features = []
        batch_size = len(texts)
        for i in range(batch_size):
            feature = {k: v[i] for k, v in tokenized_batch.items()}
            features.append(feature)

        # 4. Apply MLM masking logic and return final tensors
        return self.mlm_collator(features)


def get_dataloader(
    batch_size: int = 8,
    max_length: int = 512,
    mlm_probability: float = 0.15,
    tokenizer_name: str = "albert-base-v2",
    num_workers: int = 4,
) -> DataLoader:
    """
    Orchestrates the data loading pipeline for Fin-FFB training.

    This function:
    1. Loads the NYT and EDGAR datasets using memory-mapping (low RAM).
    2. Combines them into a single stream.
    3. Initializes the FinancialDataCollector for JIT processing.
    4. Returns a DataLoader that yields batches of masked tokens.

    Args:
        batch_size: Number of sequences per batch.
        max_length: Maximum sequence length (important for ALiBi extrapolation).
        mlm_probability: Percentage of tokens to mask for training.
        tokenizer_name: Name of the pre-trained tokenizer (default: albert-base-v2).
        num_workers: Number of subprocesses for data loading.
    """

    # 1. Initialize Individual Datasets (Memory-mapped via HF Datasets)
    nyt_ds = NYTDataset(split="train")
    edgar_ds = EDGARDataset(split="train")

    # 2. Combine into a unified training stream
    combined_ds = CombinedFinancialDataset([nyt_ds, edgar_ds])

    # 3. Initialize the Collector (replaces nested collate_fn)
    collector = FinancialDataCollector(
        tokenizer_name=tokenizer_name,
        mlm_probability=mlm_probability,
        max_length=max_length,
    )

    # 4. Initialize DataLoader
    # pin_memory is enabled to speed up transfer to GPU.
    dataloader = DataLoader(
        combined_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collector,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
