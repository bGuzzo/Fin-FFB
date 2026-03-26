"""
Sentence Embedding wrapper for Fin-FFB.

This module provides a wrapper to transform the sequence output of Fin-FFB
into a single semantic vector using mean pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from huggingface_hub import PyTorchModelHubMixin
from .fin_ffb import FinFFB

class FinFFBSentenceEncoder(nn.Module, PyTorchModelHubMixin):
    """
    Wrapper for Fin-FFB to produce sentence-level embeddings.
    
    Uses mean pooling on the sequence output, optionally followed by 
    L2 normalization for better cosine similarity performance.
    """

    def __init__(self, encoder: FinFFB, normalize: bool = True):
        super().__init__()
        self.fin_ffb = encoder
        self.normalize = normalize

    def mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on token embeddings, respecting the attention mask.
        
        This process ensures that padding tokens (0s in the mask) do not 
        distort the average semantic vector of the sentence.
        """
        # 1. Expand the mask to match the hidden state dimensions: [batch, seq_len] -> [batch, seq_len, d_model]
        # - unsqueeze(-1): Adds a new "empty" dimension at the end [batch, seq_len, 1].
        # - expand(...): Stretches the mask across the entire hidden dimension.
        # - float(): Converts 0/1 integers to decimals for multiplication.
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        
        # 2. Mask out padding tokens and sum across the sequence dimension (dim=1)
        # Multiplying by the mask explicitly zero-fills any noise in padding positions.
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        
        # 3. Count the actual number of non-padding tokens for each sequence.
        # We sum the mask values and clamp to a small epsilon to avoid division by zero.
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # 4. Return the true mean (Sum of real tokens / Number of real tokens)
        return sum_embeddings / sum_mask

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass to get sentence embeddings.
        
        Returns:
            torch.Tensor: Pooled (and optionally normalized) vector [batch, d_model].
        """
        # Get hidden states from the encoder
        outputs = self.fin_ffb(input_ids, attention_mask=attention_mask, **kwargs)

        # Handle potential tuple return (history)
        if isinstance(outputs, tuple):
            h_out = outputs[0]
        else:
            h_out = outputs

        # Apply Mean Pooling
        embeddings = self.mean_pooling(h_out, attention_mask)

        # Normalize to unit hypersphere (recommended for Cosine Similarity)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings
