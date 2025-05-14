import pandas as pd

# Loading the full dataset
df = pd.read_csv('myntra202305041052.csv')

# Taking the first 5000 rows for sampling as the original dataset is much larger
sample_df = df.head(5000).copy()

# For each row, keep only the first image URL, as the dataset we took features multiple urls of each product, whic isn't needed. 
def get_first_img_url(img_cell):
    if pd.isna(img_cell):
        return ''
    return str(img_cell).split(';')[0].strip()

sample_df['img'] = sample_df['img'].apply(get_first_img_url)

# creating and saving the 5000 samples into a new .csv file
sample_df.to_csv('myntra_sample.csv', index=False)
print("Sampled 5000 rows with first image URL and saved to myntra_sample.csv")