from datasets import load_dataset
import logging

HF_DATASET_PATH = "/Volumes/NVME_EXT/Datasets/eloukas_edgar-corpus"
PD_FILE_OUT = "/Volumes/NVME_EXT/GitHub_Repos/Fin-FFB/data/edgar_corp_05m.parquet"
NUM_SAMPLES = 500_000

def sample_and_save_to_parquet(
    hf_dataset_path: str, 
    output_path: str, 
    num_samples: int = NUM_SAMPLES, 
    split: str = "all", 
    seed: int = 42
):
    """
    Loads a HF dataset, randomly samples it, and saves it to a Parquet file.
    """
    try:
        logging.info(f"Loading dataset '{hf_dataset_path}' (split: {split})...")
        # Load the dataset
        dataset = load_dataset(hf_dataset_path, split=split, streaming=False)
        
        total_rows = len(dataset)
        logging.info(f"Original dataset size: {total_rows} rows.")

        # Handle the edge case where the requested samples exceed the dataset size
        if num_samples >= total_rows:
            logging.warning(
                f"Requested {num_samples} samples, but dataset only has {total_rows} rows. "
                "Saving the entire dataset without sampling."
            )
            sampled_dataset = dataset
        else:
            logging.info(f"Shuffling and sampling {num_samples} rows (seed: {seed})...")
            # Shuffle the dataset and select the first 'num_samples' rows
            sampled_dataset = dataset.shuffle(seed=seed).select(range(num_samples))

        # Save to Parquet
        logging.info(f"Saving sampled dataset to '{output_path}'...")
        sampled_dataset.to_parquet(output_path)
        
        logging.info("Done! Dataset saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    sample_and_save_to_parquet(
        hf_dataset_path=HF_DATASET_PATH,
        output_path=PD_FILE_OUT
    )