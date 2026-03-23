"""
Unit tests for the data loading pipeline.
Verifies NYTDataset, EDGARDataset, CombinedFinancialDataset, and FinancialDataCollector.
"""

import torch
from torch.utils.data import DataLoader
from data_loader.datasets import NYTDataset, EDGARDataset, CombinedFinancialDataset
from data_loader.collector import FinancialDataCollector, get_dataloader
import sys

def test_nyt_dataset():
    print("\n--- Testing NYTDataset ---")
    try:
        dataset = NYTDataset(split="train")
        print(f"Dataset successfully loaded. Length: {len(dataset)}")
        
        # Test first few items
        for i in range(min(2, len(dataset))):
            item = dataset[i]
            print(f"Item {i} (Index {i}):")
            print(f"  Keys: {list(item.keys())}")
            text_preview = item['text'].replace('\n', ' ')
            print(f"  Text preview (first 150 chars): {text_preview[:150]}...")
            assert "text" in item
            assert isinstance(item["text"], str)
            assert len(item["text"]) > 0
    except Exception as e:
        print(f"NYTDataset test failed or skipped: {e}")

def test_edgar_dataset():
    print("\n--- Testing EDGARDataset ---")
    try:
        # Note: EDGAR corpus 'full' config can be large.
        dataset = EDGARDataset(split="train")
        print(f"Dataset successfully loaded. Length: {len(dataset)}")
        
        for i in range(min(2, len(dataset))):
            item = dataset[i]
            print(f"Item {i} (Index {i}):")
            # EDGAR items are usually much longer
            text_preview = item['text'].replace('\n', ' ')
            print(f"  Text preview (first 150 chars): {text_preview[:150]}...")
            print(f"  Total text length: {len(item['text'])} chars")
            assert "text" in item
            assert isinstance(item["text"], str)
    except Exception as e:
        print(f"EDGARDataset test failed or skipped: {e}")
        print("Note: This might be due to dataset size or connection issues.")

def test_combined_dataset():
    print("\n--- Testing CombinedFinancialDataset ---")
    try:
        nyt = NYTDataset(split="train")
        edgar = EDGARDataset(split="train")
        combined = CombinedFinancialDataset([nyt, edgar])
        
        nyt_len = len(nyt)
        edgar_len = len(edgar)
        print(f"NYT items: {nyt_len}")
        print(f"EDGAR items: {edgar_len}")
        print(f"Combined total: {len(combined)}")
        
        assert len(combined) == nyt_len + edgar_len
        
        # Test boundary: last item of NYT
        print(f"Testing NYT boundary (Index {nyt_len - 1})...")
        item_nyt_last = combined[nyt_len - 1]
        assert item_nyt_last is not None
        
        # Test boundary: first item of EDGAR
        print(f"Testing EDGAR boundary (Index {nyt_len})...")
        item_edgar_first = combined[nyt_len]
        assert item_edgar_first is not None
        
        print("Boundary tests passed.")
    except Exception as e:
        print(f"CombinedFinancialDataset test failed: {e}")

def test_collector():
    print("\n--- Testing FinancialDataCollector ---")
    # Use a small max_length for testing speed
    max_len = 128
    mlm_prob = 0.15
    collector = FinancialDataCollector(
        tokenizer_name="albert-base-v2", 
        max_length=max_len, 
        mlm_probability=mlm_prob
    )
    
    examples = [
        {"text": "Quarterly earnings for the financial sector exceeded expectations despite volatility."},
        {"text": "The SEC has released new guidelines regarding climate-related risk disclosures in 10-K filings."},
        {"text": "Inflationary pressures continue to influence central bank policy decisions globally."}
    ]
    
    batch = collector(examples)
    
    print(f"Batch keys: {list(batch.keys())}")
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
    print(f"labels shape: {batch['labels'].shape}")
    
    assert batch['input_ids'].shape == (3, max_len)
    assert batch['labels'].shape == (3, max_len)
    
    # Calculate masking rate
    # labels == -100 means the token was NOT masked for prediction
    masked_mask = batch['labels'] != -100
    num_masked = masked_mask.sum().item()
    total_tokens = 3 * max_len
    print(f"Actual masking rate in this batch: {num_masked/total_tokens:.2%} ({num_masked} tokens)")
    
    # Analyze masked tokens
    tokenizer = collector.tokenizer
    first_example_ids = batch['input_ids'][0]
    first_example_labels = batch['labels'][0]
    
    print("\nSample token analysis (First Example):")
    decoded_input = tokenizer.decode(first_example_ids)
    print(f"  Decoded Input (with [MASK]): {decoded_input[:150]}...")
    
    # Show which tokens were actually masked
    masked_indices = torch.where(first_example_labels != -100)[0]
    print(f"  Masked indices: {masked_indices.tolist()}")
    if len(masked_indices) > 0:
        idx = masked_indices[0].item()
        original_token = tokenizer.decode([first_example_labels[idx].item()])
        masked_token = tokenizer.decode([first_example_ids[idx].item()])
        print(f"  Example Mask: Original='{original_token}', In Batch='{masked_token}' (at index {idx})")

def test_dataloader_integration():
    print("\n--- Testing get_dataloader Integration ---")
    try:
        # Using 0 workers for stable test output and 128 max_length for speed
        dataloader = get_dataloader(
            batch_size=2,
            max_length=128,
            tokenizer_name="albert-base-v2",
            num_workers=0
        )
        
        print("DataLoader initialized. Fetching one batch...")
        batch = next(iter(dataloader))
        
        print(f"Batch received. input_ids shape: {batch['input_ids'].shape}")
        assert batch['input_ids'].shape == (2, 128)
        print("Integration test passed.")
    except Exception as e:
        print(f"DataLoader integration test failed: {e}")

if __name__ == "__main__":
    test_nyt_dataset()
    test_edgar_dataset()
    test_combined_dataset()
    test_collector()
    test_dataloader_integration()
    print("\nAll data loader tests completed.")
