# extracting_features.py
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Configuration
INPUT_CSV = 'myntra_sample_clean.csv'  # Now contains all original columns
IMAGE_DIR = 'data/product_images/'
OUTPUT_CSV = 'myntra_features.csv'
BINS = (8, 8, 8)

def compute_hsv_histogram(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, BINS, [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def main():
    df = pd.read_csv(INPUT_CSV)
    
    # Validate required columns exist
    required_columns = ['image_filename', 'name', 'price']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"{INPUT_CSV} must contain '{col}' column")

    features = []
    valid_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        filename = os.path.basename(row['image_filename'])
        img_path = os.path.join(IMAGE_DIR, filename)
        
        hist = compute_hsv_histogram(img_path)
        
        if hist is not None:
            features.append(hist)
            # Preserve product metadata
            valid_data.append({
                'image_filename': filename,
                'name': row['name'],
                'price': row['price']
            })

    # Create output DataFrame
    features_df = pd.DataFrame(features, columns=[f'hsv_feature_{i}' for i in range(len(features[0]))])
    metadata_df = pd.DataFrame(valid_data)
    final_df = pd.concat([metadata_df, features_df], axis=1)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFeatures saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
