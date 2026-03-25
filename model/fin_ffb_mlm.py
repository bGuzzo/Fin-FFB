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
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Use modern SwiGLU activation
        # Ref: https://arxiv.org/abs/2302.13971
        self.swiglu_proj_1 = nn.Linear(self.d_model, self.d_model)
        self.swiglu_proj_2 = nn.Linear(self.d_model, self.d_model)

        # Output layer (Logits)
        # Use only one dense linear lajer to project final output to vocab size.
        # Avoid useless information memorization in other head layers.
        # All 'core' information/weights must be held in the levels underneath.
        # Note: bias are kept seprate to learn the base frequency distribution of the tokens. 
        self.decoder = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden states from the encoder [batch, seq_len, d_model].
        Returns:
            Logits over vocabulary [batch, seq_len, vocab_size].
        """

        # Apply SwiGLU activation
        x_proj_1 = self.swiglu_proj_1(x)
        x_proj_2 = self.swiglu_proj_2(x)

        x_swish = x_proj_1 * torch.sigmoid(x_proj_1)
        x_swiglu = x_swish * x_proj_2

        # Compute logits
        logits = self.decoder(x_swiglu)
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
