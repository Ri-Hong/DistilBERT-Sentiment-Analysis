"""
Script to download and prepare the IMDB dataset from Hugging Face.
"""
from datasets import load_dataset
import os

def fetch_imdb_dataset():
    # Load the IMDB dataset from Hugging Face
    dataset = load_dataset("imdb")
    
    # Create directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    
    # Save train, test splits
    dataset['train'].to_json("data/raw/imdb_train.json")
    dataset['test'].to_json("data/raw/imdb_test.json")
    
    # Print some basic statistics
    print(f"Downloaded IMDB dataset:")
    print(f"Training examples: {len(dataset['train'])}")
    print(f"Test examples: {len(dataset['test'])}")
    
    return dataset

if __name__ == "__main__":
    fetch_imdb_dataset()
