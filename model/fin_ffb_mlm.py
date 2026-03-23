"""
Masked Language Modeling (MLM) extensions for Fin-FFB.

This module provides the MLM head and the wrapper class for Fin-FFB
to be used in MLM pre-training tasks.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .fin_ffb import FinFFB


class MLMHead(nn.Module):
    """
    Masked Language Modeling (MLM) Head for Fin-FFB.

    Transforms the final aggregated hidden states into logits over the vocabulary.
    Following the BERT-style MLM head with transformation and decoder layers.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        # Transformation layer
        self.dense = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.rms_norm = nn.RMSNorm(d_model)

        # Output layer (Logits)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, h_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_states: Hidden states from the encoder [batch, seq_len, d_model].
        Returns:
            Logits over vocabulary [batch, seq_len, vocab_size].
        """
        x = self.dense(h_states)
        x = self.act(x)
        x = self.rms_norm(x)
        logits = self.decoder(x)
        return logits


class FinFFBForMaskedLM(nn.Module, PyTorchModelHubMixin):
    """
    Fin-FFB Model with a Masked Language Modeling (MLM) head.

    This class wraps the FinFFB encoder and adds an MLMHead for pre-training.
    It supports automatic weight tying between input embeddings and the decoder.
    """

    def __init__(self, encoder: FinFFB):
        super().__init__()
        self.fin_ffb = encoder
        self.mlm_head = MLMHead(
            d_model=encoder.d_model, vocab_size=encoder.embeddings.num_embeddings
        )

        # Tie weights: The output layer (decoder) shares weights with input embeddings.
        # This is standard practice in models like BERT and ALBERT.
        # Use the same matrix for embeddig and de-embedding
        self.mlm_head.decoder.weight = self.fin_ffb.embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for MLM training or inference.

        Args:
            input_ids: Token IDs [batch, seq_len].
            attention_mask: Optional binary mask [batch, seq_len].
            labels: Optional original token IDs for masked positions [batch, seq_len].
                    Positions with value -100 are ignored by CrossEntropyLoss.
            **kwargs: Additional arguments for the encoder.

        Returns:
            If labels is None: Logits [batch, seq_len, vocab_size].
            If labels is provided: (Loss, Logits).
        """
        # Get hidden states from encoder
        outputs = self.fin_ffb(input_ids, attention_mask=attention_mask, **kwargs)

        # Handle tuple return (if return_history=True)
        if isinstance(outputs, tuple):
            h_out = outputs[0]
        else:
            h_out = outputs

        # Compute MLM Logits
        logits = self.mlm_head(h_out)

        if labels is not None:
            # Calculate Cross Entropy Loss
            loss_fct = nn.CrossEntropyLoss()
            # Reshape for loss calculation: [batch * seq_len, vocab_size] vs [batch * seq_len]
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits

        return logits
