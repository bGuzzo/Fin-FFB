"""
Just-In-Time (JIT) Data Collection and Masked Language Modeling (MLM) Strategy.

This module provides the core logic for efficient data ingestion in the Fin-FFB 
training pipeline. It addresses the memory constraints of consumer-grade hardware
by deferring tokenization and MLM masking until the moment a batch is requested
by the training loop, rather than pre-processing the entire corpus in memory.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from typing import List, Dict
from data_loader.pd_adpt import PdDataset
import logging
import os

# Determine optimal parallelism based on available hardware threads.
N_PROC = os.cpu_count() or 10


class JitCollator:
    """
    A callable transformation pipeline for on-the-fly tokenization and MLM masking.

    To maintain a low memory footprint, this class performs computationally intensive
    text processing during the batch collation phase. It leverages the HuggingFace
    Transformers library to handle complex ALBERT tokenization and the stochastic
    masking required for the Masked Language Modeling objective.

    Attributes:
        tokenizer (AutoTokenizer): The pre-trained ALBERT tokenizer used to encode
            raw financial text into discrete token identifiers.
        mlm_collator (DataCollatorForLanguageModeling): A specialized utility that
            applies the MLM masking logic (e.g., [MASK] replacement, random token
            substitution, or original token retention).
        max_length (int): The rigid sequence length used for padding and truncation
            to ensure uniform tensor dimensions within a batch.
    """

    def __init__(
        self,
        tokenizer_name: str = "albert-base-v2",
        mlm_probability: float = 0.15,
        max_length: int = 512,
    ):
        """
        Initializes the JIT Collator with specific model and masking parameters.

        Args:
            tokenizer_name (str): The identifier for the pre-trained tokenizer.
            mlm_probability (float): The probability with which tokens are selected
                for the MLM objective (default 15% as per BERT standards).
            max_length (int): Maximum allowable sequence length for the model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # Internal collator handles the stochastic selection of tokens for masking.
        # It follows the 80/10/10 rule: 80% [MASK], 10% random, 10% unchanged.
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=mlm_probability
        )
        logging.info(
            f"Initialized JitCollator: Tokenizer={tokenizer_name}, "
            f"MLM_Prob={mlm_probability}, Max_Len={max_length}"
        )

    def __call__(self, examples: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """
        Transforms a list of raw string examples into a tensorized, masked batch.

        This method implements the following four-stage transformation pipeline:

        1.  **Text Extraction**: Aggregates the 'text' fields from the input list 
            of dictionary items provided by the Dataset.
        2.  **JIT Tokenization**: Encodes the raw strings into integer sequences.
            Crucially, this step applies truncation to 'max_length' for long 
            documents and adds padding tokens to shorter sequences to maintain
            rectangular tensor shapes.
        3.  **Feature Re-alignment**: Converts the batch-oriented output of the
            tokenizer (a single dictionary of lists) back into a list of 
            individual dictionaries (features), satisfying the interface 
            requirements of the internal MLM collator.
        4.  **MLM Masking & Label Generation**: Applies the MLM logic to the 
            'input_ids'. It generates a 'labels' tensor where only the tokens
            selected for masking contain their original IDs, while all other
            positions are set to -100 (the standard CrossEntropyLoss ignore index).

        Args:
            examples (List[Dict[str, str]]): A batch-sized list of raw text records.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'input_ids': Masked token sequences.
                - 'attention_mask': Binary mask indicating valid tokens vs padding.
                - 'labels': Target tokens for the MLM prediction task.
        """
        # 1. Extract strings from the dataset items
        texts = [example["text"] for example in examples]

        # 2. Tokenize on-the-fly.
        # This converts raw text into input_ids, attention_mask, etc.
        tokenized_batch = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,  # Important, chunk the text if len(token) > max_length
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
    dataset: Dataset,
    batch_size: int = 8,
    max_length: int = 512,
    mlm_probability: float = 0.4,
    tokenizer_name: str = "albert-base-v2",
    num_workers: int = N_PROC,
) -> DataLoader:
    """
    Constructs an optimized PyTorch DataLoader for the Fin-FFB training process.

    This factory function encapsulates the complexity of the data pipeline by
    coupling a standard Dataset with the JitCollator. The resulting DataLoader
    is responsible for parallel data fetching, batching, and the real-time
    transformation of raw text into training-ready MLM tensors.

    Args:
        dataset (Dataset): The source PyTorch Dataset (e.g., PdDataset).
        batch_size (int): Number of training examples per forward/backward pass.
        max_length (int): Maximum token count per sequence.
        mlm_probability (float): Density of tokens to be masked for the objective.
        tokenizer_name (str): Identifier for the ALBERT-style tokenizer.
        num_workers (int): Number of background processes used for data loading.

    Returns:
        DataLoader: An iterator yielding masked and tensorized batches.
    """

    jit_collator = JitCollator(
        tokenizer_name=tokenizer_name,
        mlm_probability=mlm_probability,
        max_length=max_length,
    )

    # 4. Initialize DataLoader
    # pin_memory is enabled to speed up transfer to GPU.
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        # Data already shuffled with pandas
        # shuffle=True,
        collate_fn=jit_collator,
        num_workers=num_workers,
        # Apple MPS do not support it
        # pin_memory=True,
    )

    return dataloader


# Test
if __name__ == "__main__":
    dataset = PdDataset()
    dataloader = get_dataloader(dataset)
    batch = next(iter(dataloader))

    logging.info(batch.keys())
    logging.info(batch["input_ids"].shape)
    logging.info(batch["attention_mask"].shape)
    logging.info(batch["labels"].shape)
    logging.info(batch["input_ids"])
    logging.info(batch["attention_mask"])
    logging.info(batch["labels"])
