import argparse
import pandas as pd
import os
import sys


def analyze_parquet(filepath: str, num_rows: int = 5):
    """Loads a Parquet file, prints basic info, and shows the head and tail."""

    # Check if the file exists before trying to load it
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' does not exist.")
        sys.exit(1)

    try:
        print(f"Loading Parquet file from: {filepath}...\n")
        # Load the parquet file into a pandas DataFrame
        df = pd.read_parquet(filepath)

        # --- 1. Basic Analysis ---
        print("=" * 60)
        print("📊 DATASET OVERVIEW")
        print("=" * 60)
        print(f"Total Rows:    {df.shape[0]:,}")
        print(f"Total Columns: {df.shape[1]}")
        print("\nColumns and Data Types:")
        # Convert dtypes to a string representation for cleaner output
        print(df.dtypes.to_string())

        # Optional: Check for missing values quickly
        # print("\nMissing Values per Column:")
        # print(df.isna().sum().to_string())

        # --- 2. Head ---
        print("\n" + "=" * 60)
        print(f"🔼 HEAD (First {num_rows} rows)")
        print("=" * 60)
        print(df.head(num_rows))

        # --- 3. Tail ---
        print("\n" + "=" * 60)
        print(f"🔽 TAIL (Last {num_rows} rows)")
        print("=" * 60)
        print(df.tail(num_rows))
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"❌ An error occurred while reading the Parquet file: {e}")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="A simple CLI tool to briefly analyze and preview a Parquet file."
    )

    # Required positional argument for the file path
    parser.add_argument(
        "filepath", type=str, help="The path to the .parquet file you want to inspect."
    )

    # Optional argument to control how many rows are shown in head/tail
    parser.add_argument(
        "-n",
        "--num_rows",
        type=int,
        default=5,
        help="Number of rows to display for head and tail (default is 5).",
    )

    args = parser.parse_args()

    # Run the analysis
    analyze_parquet(args.filepath, args.num_rows)
