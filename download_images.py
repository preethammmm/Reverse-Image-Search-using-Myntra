#Local copies of images are required for feature extractiona and similarity search.
#running this script after getting the myntra_samples.csv from sample-data.py
import pandas as pd
import requests
import os
import re


CSV_FILE = 'myntra_sample.csv'
IMG_URL_COLUMN = 'img'
IMG_SAVE_DIR = 'data/product_images'
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

def sanitize_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', str(name))[:80]

df = pd.read_csv(CSV_FILE)
failed = []

for idx, row in df.iterrows():
    url = str(row[IMG_URL_COLUMN])
    if not url.startswith('http'):
        failed.append(idx)
        continue
    filename = f"{sanitize_filename(row['name'])}_{idx}.jpg"
    save_path = os.path.join(IMG_SAVE_DIR, filename)
    df.at[idx, 'image_filename'] = filename 
    if os.path.isfile(save_path):
        df.at[idx, 'image_filename'] = save_path
        continue
    try:
        response = requests.get(url, stream=True, timeout=15)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
            if os.path.getsize(save_path) < 1024:
                os.remove(save_path)
                failed.append(idx)
                continue
            df.at[idx, 'image_filename'] = filename
        else:
            failed.append(idx)
    except Exception as e:
        failed.append(idx)
#creating a new 'myntra_samples_with_images.csv to store the images from the myntra_sample.csv file 
df.to_csv('myntra_sample_with_images.csv', index=False)
print(f"Downloaded images for sample. Failed: {len(failed)}") #using this statement too, to get the data if any of the images failed to download.