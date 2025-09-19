"""
Fetch and prepare IMDB dataset from Hugging Face.
"""

import json
import logging
from pathlib import Path

from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_json_dataset(data, output_file):
    """Save dataset split to JSON file."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to list of dictionaries
    json_data = []
    for item in data:
        json_data.append({
            "text": item["text"],
            "label": item["label"]
        })
    
    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        for item in json_data:
            # Write each item as a separate JSON line (jsonl format)
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    
    logger.info(f"Saved {len(json_data)} examples to {output_file}")

def main():
    # Load IMDB dataset from Hugging Face
    logger.info("Loading IMDB dataset from Hugging Face...")
    dataset = load_dataset("imdb")
    
    # Create output directory
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train and test splits
    save_json_dataset(dataset["train"], raw_dir / "imdb_train.jsonl")
    save_json_dataset(dataset["test"], raw_dir / "imdb_test.jsonl")
    
    logger.info("Dataset download and preparation completed!")

if __name__ == "__main__":
    main()
