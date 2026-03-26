"""
Unit tests for AttentionLayer and AttentionResidualBlock.
Verifies dimensionality, projection logic, and depth-wise aggregation.
"""

import os
import sys

import torch
import torch.nn as nn

# Add the project root to sys.path to allow imports from the 'model' directory.
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.attn_core import AttentionLayer
from model.attn_res_block import AttnResBlock

import logging


def test_attention_layer():
    logging.info("Testing AttentionLayer...")
    batch_size = 2
    seq_len = 16
    d_model = 128
    num_heads = 4
    dropout = 0.0

    # Initialize Layer
    layer = AttentionLayer(
        layer=0, d_model=d_model, num_heads=num_heads, dropout=dropout
    )

    # Mock Input
    x = torch.randn(batch_size, seq_len, d_model)
    # Mock mask
    mask = torch.ones(batch_size, seq_len)
    mask[:, 8:] = 0  # Mask second half

    # Forward Pass
    output = layer(x, attention_mask=mask)

    # Check Dimensions: [Batch, SeqLen, d_model]
    assert output.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"

    logging.info("AttentionLayer dimensionality test passed.")


def test_attn_res_block():
    logging.info("Testing AttnResBlock (Full Attention Residuals)...")
    batch_size = 2
    seq_len = 16
    d_model = 128
    num_heads = 4
    dropout = 0.0

    # 1. Initialize First Block (Layer 0)
    # The first block receives only the token embedding in its history.
    block_0 = AttnResBlock(
        layer_idx=0, d_model=d_model, num_heads=num_heads, dropout=dropout
    )

    embedding = torch.randn(batch_size, seq_len, d_model)
    history = [embedding]  # b_0 = embedding

    # Mock mask
    mask = torch.ones(batch_size, seq_len)

    # Forward Pass Block 0
    v0, history = block_0(history, attention_mask=mask)

    # Verify outputs
    assert v0.shape == (batch_size, seq_len, d_model)
    assert len(history) == 2, "History should contain [embedding, v0]"
    assert torch.equal(history[0], embedding)
    assert torch.equal(history[1], v0)

    logging.info("Block 0 (First Layer) tests passed.")

    # 2. Initialize Second Block (Layer 1)
    # This layer should attend over [embedding, v0].
    block_1 = AttnResBlock(
        layer_idx=1, d_model=d_model, num_heads=num_heads, dropout=dropout
    )

    v1, history = block_1(history, attention_mask=mask)

    assert v1.shape == (batch_size, seq_len, d_model)
    assert len(history) == 3, "History should contain [embedding, v0, v1]"

    logging.info("Block 1 (Second Layer) tests passed.")

    # 3. Test Selective Aggregation Logic
    # Verify that the pseudo-query projection (wl) correctly produces
    # attention weights over the depth dimension.
    # Initially, wl is zeroed, so weights should be uniform.

    for i in range(2, 5):
        block = AttnResBlock(i, d_model, num_heads, dropout)
        vi, history = block(history, attention_mask=mask)
        assert vi.shape == (batch_size, seq_len, d_model)
        assert len(history) == i + 2

    logging.info(f"Stack of 5 layers passed. History size: {len(history)}")

    # 4. Test Final Output Layer Aggregation
    # The final layer should aggregate the full history and return a single tensor.
    final_block = AttnResBlock(
        layer_idx=5,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        final_layer=True,
    )

    h_out = final_block(history, attention_mask=mask)

    # Verify it returns a single tensor, not a tuple
    assert isinstance(h_out, torch.Tensor), "Final layer must return a single Tensor."
    assert h_out.shape == (batch_size, seq_len, d_model)

    # Since pseudo-query is zeroed, weights are uniform.
    # h_out should be the mean of all history tensors.
    expected_out = torch.stack(history).mean(dim=0)
    assert torch.allclose(
        h_out, expected_out, atol=1e-5
    ), "Final layer aggregation should be uniform average at initialization."

    logging.info("Final Output Layer (Aggregation Only) tests passed.")


def test_initialization_state():
    logging.info("Testing Weight Initialization...")
    d_model = 64
    block = AttnResBlock(0, d_model, 4, 0.0)

    # Per paper, wl should be initialized to zero.
    assert torch.all(
        block.attn_res_proj.weight == 0
    ), "Pseudo-query wl must be initialized to zero."
    logging.info("Initialization check passed.")


def test_fin_ffb_model():
    logging.info("Testing FinFFB Model...")
    vocab_size = 1000
    batch_size = 2
    seq_len = 32
    d_model = 256
    num_layers = 3
    num_heads = 8

    model = FinFFB(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1,
    )

    # Mock input_ids
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Mock mask
    mask = torch.ones(batch_size, seq_len)
    mask[:, 16:] = 0

    # Forward pass
    output = model(input_ids, attention_mask=mask)

    assert output.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"

    # Test with return_history=True
    output, history = model(input_ids, attention_mask=mask, return_history=True)

    # history should contain v0 (embedding) + v1, v2, v3 (layer outputs) = 4 tensors
    assert (
        len(history) == num_layers + 1
    ), f"Expected history length {num_layers + 1}, got {len(history)}"

    logging.info("FinFFB model tests passed.")


def test_fin_ffb_for_masked_lm():
    logging.info("Testing FinFFBForMaskedLM...")
    vocab_size = 1000
    batch_size = 2
    seq_len = 32
    d_model = 256
    num_layers = 3
    num_heads = 8

    encoder = FinFFB(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1,
    )

    model = FinFFBForMaskedLM(encoder)

    # Mock input_ids, mask and labels
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Mask some labels as -100 (ignored by CE)
    labels[0, :5] = -100

    # Forward pass without labels
    logits = model(input_ids, attention_mask=mask)
    assert logits.shape == (
        batch_size,
        seq_len,
        vocab_size,
    ), f"Expected logits shape {(batch_size, seq_len, vocab_size)}, got {logits.shape}"

    # Forward pass with labels
    loss, logits = model(input_ids, attention_mask=mask, labels=labels)
    assert loss.item() > 0, "Loss should be positive."
    assert logits.shape == (batch_size, seq_len, vocab_size)

    # Check weight tying
    assert torch.equal(
        model.mlm_head.decoder.weight, encoder.embeddings.weight
    ), "Weights of embeddings and MLM decoder should be tied."

    logging.info("FinFFBForMaskedLM model tests passed.")


if __name__ == "__main__":
    try:
        from model.fin_ffb import FinFFB
        from model.fin_ffb_mlm import FinFFBForMaskedLM

        test_attention_layer()
        test_attn_res_block()
        test_initialization_state()
        test_fin_ffb_model()
        test_fin_ffb_for_masked_lm()
        logging.info("All tests completed successfully.")
    except Exception as e:
        logging.info(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
