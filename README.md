# Financial Fast Fat BERT Text Encoder

## Motivation
Financial forecasting and analysis tasks require encoding a large number of texts, from articles, reports, news, and earnings statements.
So, a faster embedding model is crucial, especially when dealing with hardware constraints, such as consumer hardware.

My idea is to create a shallow, highly-specialized embedding model for financial purposes. 
Its primary application is to embed news and financial reports for the [Fin-PT](https://github.com/bGuzzo/Fin-PT) project (next asset price prediction).

## Core Concepts
The core idea is to create a large (Fat) and shallow (Fast) model like BERT, but with a parallel optimization: FFN & Self-Attention are computed concurrently and then joined by summation.
Other optimizations, like gated attention and novel attention residuals (Kimi AI, [Attention Residuals](https://arxiv.org/abs/2603.15031)), are used to gain performance edges and stabilize training.

* Initially, the model will have a maximum of 3 layers, with a d_model of up to 2048.
* The tokenized choosen is `albert-base-v2` due to it's small vocabulaty size (30k).
* **ALiBi** will be use to help the model genelize on longer texts.  

## Inspiration Papers

* [Investigating the Role of Feed-Forward Networks in Transformers Using Parallel Attention and Feed-Forward Net Design](https://arxiv.org/abs/2305.13297)
* [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)
* [Attention Residuals](https://arxiv.org/abs/2603.15031)
* [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708)
* [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
* [EDGAR-CORPUS: Billions of Tokens Make The World Go Round](https://arxiv.org/abs/2109.14394)
* [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)
* [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)

## Training
The dataset is composed of news, reports, and specialized text:
* 100 years of **New York Times** news (title + abstract): [nyt_100y_news_headlines](https://huggingface.co/datasets/bguzzo2k/nyt_100y_news_headlines)
* Timeless asset reports (fully synthetic): [tkrs_timeless_report](https://huggingface.co/datasets/bguzzo2k/tkrs_timeless_report)
* Wikipedia (Financial Subsets)
* EDGAR-CORPUS (eloukas/edgar-corpus on Hugging Face)

In the first project phase, the model will only be trained with **MLM** (Masked Language Modeling).  


## Improvements (to be applied)
* [Query-Key Normalization for Transformers](https://arxiv.org/abs/2010.04245)


## Implementation
TBD


