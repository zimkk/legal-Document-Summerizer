import os
import json
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

# Define paths relative to current directory
DATASET_DIR = os.path.join(os.getcwd(), "datasets")
OUTPUT_DATASET_FILE = os.path.join(DATASET_DIR, "billsum_data.json")
OUTPUT_STATS_FILE = os.path.join(DATASET_DIR, "billsum_dataset_statistics.txt")

def download_and_process_billsum():
    """Download and process BillSum dataset for summarization"""
    print("Downloading BillSum dataset...")
    # Load the 'ca_test' split as it's smaller and suitable for demonstration/fine-tuning.
    # For full training, consider using the 'train' split (billsum[:train]), which is much larger.
    try:
        dataset = load_dataset("billsum", split="ca_test") 
        print(f"Loaded BillSum ca_test split with {len(dataset)} examples.")
        return dataset
    except Exception as e:
        print(f"Error downloading or loading BillSum dataset: {e}")
        raise

def create_dataset(billsum_dataset):
    """Create the final dataset structure from the BillSum dataset"""
    data = {
        "text": [],
        "summary": [],
    }
    
    print("Processing BillSum dataset...")
    # Ensure text and summary are not None and are strings
    processed_count = 0
    skipped_count = 0
    for item in tqdm(billsum_dataset):
        # Use .get() for safety, although 'text' and 'summary' should exist in billsum
        text = item.get("text")
        summary = item.get("summary")
        if text and summary:
             data["text"].append(str(text))
             data["summary"].append(str(summary))
             processed_count += 1
        else:
            skipped_count += 1
            
    print(f"Processed {processed_count} items, skipped {skipped_count} items due to missing text or summary.")
    return data

def save_dataset(data, output_file=OUTPUT_DATASET_FILE, stats_file=OUTPUT_STATS_FILE):
    """Save the processed dataset to a JSON file"""
    # Create datasets directory if it doesn't exist
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    print(f"Saving processed dataset to {output_file}...")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved dataset.")
    except Exception as e:
        print(f"Error saving dataset: {e}")
        raise
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total documents: {len(data['text'])}")
    
    # Save statistics to a separate file
    try:
        with open(stats_file, "w") as f:
            f.write("Dataset Statistics (BillSum - ca_test split):\n")
            f.write(f"Total documents processed: {len(data['text'])}\n")
        print(f"Successfully saved statistics to {stats_file}")
    except Exception as e:
        print(f"Error saving statistics: {e}")

def main():
    try:
        print("Starting dataset preparation using BillSum...")
        raw_dataset = download_and_process_billsum()
        processed_data = create_dataset(raw_dataset)
        save_dataset(processed_data)
        
        print("\nDataset preparation completed successfully!")
        print("Files saved to:")
        print(f"- {OUTPUT_DATASET_FILE}")
        print(f"- {OUTPUT_STATS_FILE}")
        
    except Exception as e:
        print(f"An error occurred during dataset preparation: {e}")
        raise

if __name__ == "__main__":
    main() 