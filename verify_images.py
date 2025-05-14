# verify_images.py
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm

def verify_image(img_path):
    try:
        with Image.open(img_path) as img:
            img.verify()
            img.load()
        return True
    except (IOError, OSError, Image.DecompressionBombError):
        return False

def main():
    INPUT_CSV = 'myntra_sample_with_images.csv'
    OUTPUT_CSV = 'myntra_sample_clean.csv'
    IMAGE_DIR = 'data/product_images/'

    # Load original dataset with all columns
    df = pd.read_csv(INPUT_CSV)
    valid_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Verifying images"):
        filename = os.path.basename(row['image_filename'])
        img_path = os.path.join(IMAGE_DIR, filename)
        
        if os.path.exists(img_path) and verify_image(img_path):
            # Preserve ALL original columns for valid images
            valid_rows.append(row.to_dict())

    # Save cleaned data with original columns
    pd.DataFrame(valid_rows).to_csv(OUTPUT_CSV, index=False)
    print(f"\nCleaned dataset saved to {OUTPUT_CSV} ({len(valid_rows)} valid images)")

if __name__ == "__main__":
    main()
