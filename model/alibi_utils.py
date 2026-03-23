"""
ALiBi (Attention with Linear Biases) Utilities.

This module implements the ALiBi positioning method as described in the paper:
"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
(Press et al., 2021) - https://arxiv.org/abs/2108.12409

ALiBi eliminates traditional positional embeddings and instead biases the attention
scores by a value proportional to the distance between tokens. This allows models
to extrapolate to sequence lengths longer than those seen during training.
"""

import torch
from typing import List


def get_slopes(n_heads: int) -> List[float]:
    """
    Apply the general formula provided by the paper.
    m_i = (2^(-8)/n))^i
    """
    start = (2 ** ((-8)/n_heads))
    return [start * (start ** i) for i in range(n_heads)]


def generate_bidirectional_alibi_bias(
    seq_len: int, 
    slopes: torch.Tensor, 
    dtype: torch.dtype, 
    device: torch.device
) -> torch.Tensor:
    """
    Constructs the ALiBi bias matrix for bidirectional (non-causal) attention.
    
    Unlike causal ALiBi which only penalizes the 'past', bidirectional ALiBi
    penalizes distance in both directions symmetrically.

    Args:
        seq_len: The length of the input sequence.
        slopes: A tensor of shape (num_heads, 1, 1) containing the pre-calculated slopes.
        dtype: The desired torch.dtype for the output bias matrix.
        device: The target device (CPU/CUDA).

    Returns:
        A bias tensor of shape (num_heads, seq_len, seq_len).
    """
    # Create a 1D tensor of positions: [0, 1, 2, ..., seq_len-1]
    positions = torch.arange(seq_len, device=device, dtype=torch.long)

    # Compute absolute bidirectional distances using broadcasting.
    # Resulting 'distances' shape: (seq_len, seq_len)
    # entry (i, j) is |i - j|
    distances = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))

    # Apply the slopes and negate. 
    # ALiBi penalizes distance: score = (qK^T)/sqrt(d) - slope * distance
    # slopes shape: (num_heads, 1, 1)
    # distances shape: (seq_len, seq_len)
    # Result shape: (num_heads, seq_len, seq_len)
    alibi_bias = -1.0 * slopes * distances.to(dtype)

    return alibi_bias
