# data_handler.py

import pandas as pd
from datasets import load_dataset
import os
import random
from config import OUTPUT_DIR, CSV_OUTPUT, NUM_REVIEWS, DATASET_NAME


def load_reviews(dataset_name=DATASET_NAME, num_reviews=NUM_REVIEWS, random_seed=None):
    """
    Loads movie reviews from HuggingFace datasets with random selection.
    
    Args:
        dataset_name: Either "imdb" or "rotten_tomatoes"
        num_reviews: Number of reviews to load
        random_seed: Optional seed for reproducibility (None = truly random)
        
    Returns:
        List of dictionaries containing review text and metadata
    """
    print(f"Loading {num_reviews} reviews from {dataset_name}...")
    
    if dataset_name == "imdb":
        dataset = load_dataset("stanfordnlp/imdb", split="test")
    elif dataset_name == "rotten_tomatoes":
        dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="test")
    else:
        raise ValueError("Dataset must be 'imdb' or 'rotten_tomatoes'")
    
    # Set random seed if provided (for reproducibility)
    if random_seed is not None:
        random.seed(random_seed)
    
    # Randomly select review indices
    total_reviews = len(dataset)
    selected_indices = random.sample(range(total_reviews), min(num_reviews, total_reviews))
    
    # Extract selected reviews
    reviews = []
    for idx in selected_indices:
        reviews.append({
            "review_id": idx,
            "text": dataset[idx]["text"],
            "original_label": dataset[idx]["label"]
        })
    
    print(f"Randomly selected {len(reviews)} reviews from {total_reviews} total")
    return reviews


def save_results(results_data):
    """
    Saves experimental results to CSV.
    
    Args:
        results_data: List of dictionaries containing trial results
        
    Returns:
        Path to saved CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(results_data)
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, CSV_OUTPUT)
    df.to_csv(output_path, index=False)
    
    print(f"Results saved to {output_path}")
    return output_path


def load_results(filepath=None):
    """
    Loads experimental results from CSV.
    
    Args:
        filepath: Path to CSV file (optional)
        
    Returns:
        DataFrame containing results
    """
    if filepath is None:
        filepath = os.path.join(OUTPUT_DIR, CSV_OUTPUT)
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} trials from {filepath}")
    return df