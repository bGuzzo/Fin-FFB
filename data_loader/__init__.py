"""
Data Loading and Preprocessing Package for the Fin-FFB Training Pipeline.

This package provides utilities for efficient data ingestion, adapting pandas
DataFrames to PyTorch Dataset objects, and implementing Just-In-Time (JIT)
tokenization and Masked Language Modeling (MLM) strategies optimized for
consumer-grade hardware.
"""

from utils.logging_config import configure_logging

configure_logging()
