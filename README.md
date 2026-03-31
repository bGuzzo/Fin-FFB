# Financial Fast Fat BERT (Fin-FFB) Text Encoder

```
 _____ _             _____ _____ ____  
|  ___(_)_ __       |  ___|  ___| __ ) 
| |_  | | '_ \ _____| |_  | |_  |  _ \ 
|  _| | | | | |_____|  _| |  _| | |_) |
|_|   |_|_| |_|     |_|   |_|   |____/ 
```
A fast, shallow, and wide financial embedding model built for local usage on consumer hardware.

```
Add photo of model architecture & benchmark/loss.
```

## Motivation
Financial forecasting and analysis require encoding a massive volume of text-from news headlines and earnings reports to long-form SEC filings. Existing models are often too deep for efficient local inference or too general for niche financial nuances.

**Fin-FFB** is designed to be a "shallow-but-wide" specialist. By trading depth for width, we achieve high-dimensional representational capacity while keeping latency low enough to run on a standard laptop. Its primary application is extracting the **sentiment of the day** and encoding complex **financial reports** locally.

## Core Concepts
The architecture disrupts the standard serial BERT design in favor of parallelized blocks and selective information flow:

*   **Parallel Computation**: Attention and FFN blocks are computed concurrently on the same normalized input and then summed, following the PaLM/GPT-J approach. This maximizes hardware utilization.
*   **Full Attention Residuals (AttnRes)**: Instead of standard additive residuals ($h_l = h_{l-1} + f_{l-1}$), we use a depth-wise selective aggregation mechanism. A learned pseudo-query vector decides how much information to "pull" from every preceding layer (including the initial embedding).
*   **Amygdala Salience Filtering (Layer 0)**: A biological-inspired pre-processing layer that acts as a high-pass filter. It uses sharp, low-temperature latent attention to identify and highlight the most salient tokens (e.g., critical financial terms, "shocks") before the deeper transformer layers process the sequence.
*   **Gated Attention**: Implements a sigmoid gating mechanism on the attention output to improve non-linearity and filter noise.
*   **Bidirectional ALiBi**: Uses Attention with Linear Biases (ALiBi) instead of positional embeddings, allowing for sequence length extrapolation and better handling of long financial documents.
*   **Modern Primitives**: Uses **RMSNorm** for stability and **SwiGLU** activations in the FFN for better gradient flow.

## Model Architecture (Large/Fat Variant)
According to the `large.yaml` configuration, the primary "Fat" target is:

| Parameter | Value |
| :--- | :--- |
| **Vocabulary Size** | 30,000 (ALBERT-base-v2) |
| **Hidden Dimension ($d_{model}$)** | 1024 |
| **Layers** | 3 (+ Amygdala Layer 0) |
| **Attention Heads** | 16 ($d_{head}=64$) |
| **FFN Expansion** | 4x ($d_{ffn}=4096$) |
| **Context Window** | 1024 Tokens |
| **Activation** | SwiGLU |
| **Normalization** | RMSNorm |

### Detailed Component Breakdown

#### 1. Parallel Attention & FFN (PaLM-style)
Traditional Transformers compute Attention followed by an FFN in a serial bottleneck. Fin-FFB disrupts this by computing both concurrently on the same normalized input ($x_{norm} = \text{RMSNorm}(h_l)$). This approach reduces the sequential path length and improves hardware utilization during training:
$$f_l(h_l) = \text{Dropout}(\text{GatedAttn}(x_{norm}) + \text{SwiGLUFFN}(x_{norm}))$$

#### 2. Gated Attention Mechanism
To increase the non-linearity of the attention block and provide better noise filtering for dense financial text, we implement a sigmoid gating mechanism. The raw attention output is scaled by a learned gate projection derived from the normalized input:
*   **Gate Projection**: $\text{Gate} = \sigma(W_g \cdot x_{norm})$
*   **Selective Filtering**: $\text{Output} = W_o \cdot (\text{Gate} \odot \text{Attn}(x_{norm}))$

#### 3. Full Attention Residuals (AttnRes)
Instead of standard additive residuals ($h_l = h_{l-1} + v_{l-1}$), Fin-FFB uses a depth-wise selective aggregation mechanism. Each layer $l$ calculates its input $h_l$ by attending to the entire history of previous outputs ($v_0, v_1, \dots, v_{l-1}$):
*   **Pseudo-Query**: A learned vector $w_l$ computes scalar importance logits for each preceding layer.
*   **Depth-Wise Softmax**: Weights are normalized across the depth dimension, allowing the model to dynamically "choose" which preceding representations are most relevant for the current transformation.
*   **Selective Sum**: $h_l = \sum_{i=0}^{l-1} \alpha_i v_i$

#### 4. Amygdala Indexer (Layer 0)
The Amygdala Indexer mimics the biological amygdala's role as a primary filter for salient information. Positioned as "Layer 0" in the AttnRes framework, it scans for crucial tokens before deeper processing.
*   **Purpose**: It provides an "emotional" high-pass filter that identifies high-impact tokens.
*   **Dual-Path Initialization**: Creates a parallel path for gradients. Subsequent layers can choose between raw embeddings and these salient highlights, significantly accelerating convergence.
*   **Representation De-correlation**: Sharp, low-temperature attention helps break the "embedding cone" effect (representation collapse) by introducing high-entropy signals early.
*   **Latent Efficiency**: Operating in a reduced dimensional space ($d_{latent} \ll d_{model}$) allows for complex salience filtering with minimal computational overhead.

#### 5. Dropout Placement & Regularization
Dropout is strategically placed to maintain training stability across the shallow-but-wide architecture:
*   **Post-Embedding**: Applied to $v_0$ (initial embeddings) to regularize the massive $1024 \times 30000$ embedding matrix.
*   **Attention Weights**: Applied to the softmax scores before value aggregation to prevent head dominance.
*   **Post-Parallel Block**: Applied after the sum of Attention and FFN outputs, acting as the final regularizer before the history update.

## Training Strategy
The model is trained on a curated **200M token** corpus (200k articles) with a high **40% Masked Language Modeling (MLM)** probability to force the model to understand broader context:

*   **40% - EDGAR-CORPUS**: 80k sections from SEC filings.
*   **40% - NYT 100Y**: 80k headlines and abstracts spanning a century of news.
*   **20% - Wikipedia**: 40k articles for general linguistic grounding.

## Evaluation & Benchmarks
*Evaluation and comparison to other financial models (e.g., FinBERT, BloombergGPT-lite) are currently in progress and will be added here once validated.*

## Implementation Details
The project is implemented in **PyTorch** with several strategies to enable training on consumer hardware (e.g., MacBook Air M4 or RTX 4060 8GB):

*   **JIT Tokenization**: To save memory, tokenization and MLM masking are performed on-the-fly during batch collation using `data_loader/mlm_loader.py`.
*   **Gradient Accumulation**: We use a physical batch size of 4 but accumulate gradients over 32 steps to reach a **virtual batch size of 128**, matching the effective training dynamics of larger models.
*   **Mixed Precision (bf16)**: Specifically optimized for Apple Silicon (MPS) and modern NVIDIA GPUs to reduce VRAM footprint and speed up computation.
*   **Weight Tying**: The output MLM decoder shares weights with the input embedding matrix, significantly reducing the total parameter count.

## Repo Structure
```text
Fin-FFB/
├── config/             # YAML configs for model variants (Nano, Small, Med, Large, X-Large)
├── data/               # Parquet datasets (Raw bulk data and processed 'ready' files)
├── data_loader/        # PyTorch Dataset adapters and JIT MLM Collator
│   ├── mlm_loader.py   # Core logic for on-the-fly tokenization/masking
│   └── pd_adpt.py      # Pandas-to-PyTorch adapter for Parquet files
├── data_utils/         # Pre-processing pipeline for SEC/NYT/Wiki sources
├── model/              # Core Architecture Implementation
│   ├── amygdala.py     # Amygdala Indexer (Layer 0 salience filter)
│   ├── attn_core.py    # Parallel Attention + FFN blocks
│   ├── attn_res_block.py # Full Attention Residuals (selective aggregation)
│   ├── alibi_utils.py  # Bidirectional ALiBi bias generation
│   ├── fin_ffb.py      # Base Encoder
│   └── fin_ffb_mlm.py  # MLM Pre-training wrapper
├── utils/              # Training lifecycle, hardware management, and logging
├── train.py            # Main training entry point
└── requirements.txt    # Project dependencies
```

## Hardware Constraints & Performance
The project is a testament to what's possible with limited resources:
*   **Apple Silicon Optimization**: Utilizes `MPS` and `bf16` for maximum throughput on M-series chips.
*   **VRAM Efficiency**: Shallow depth (3 layers) allows for a massive 1024-dimension width even on 8GB cards.
*   **Training Time**: Optimized to complete a full 200M token pre-training pass in approximately ~30 hours on a consumer setup.

## Inspiration Papers
*   [BERT](https://arxiv.org/abs/1810.04805), [PaLM](https://arxiv.org/abs/2204.02311), [Attention Residuals](https://arxiv.org/abs/2603.15031), [Gated Attention](https://arxiv.org/abs/2505.06708), [ALiBi](https://arxiv.org/abs/2108.12409).

## Improvements (to be applied)
* [Query-Key Normalization for Transformers](https://arxiv.org/abs/2010.04245)
