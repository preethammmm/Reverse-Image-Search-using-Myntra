# HSV Pipeline Overview

This project uses the following files for HSV-based reverse image search:

| File Name               | Purpose                                              |
|-------------------------|------------------------------------------------------|
| sample_data.py          | Sample a manageable subset from the full Myntra CSV  |
| download_images.py      | Download product images and record local filenames   |
| verify_images.py        | Remove rows with missing or corrupted images         |
| extracting_features.py  | Extract HSV color histogram features for each image  |
| myntra_sample_clean.csv | Cleaned product metadata with valid images           |
| myntra_features.csv     | Extracted HSV features for each image                |
| data/product_images/    | Folder containing all product images                 |
| apphsv.py               | Streamlit app for HSV-based reverse image search  |

**Note:**  
- `apphsv.py` lets you upload a query image and find visually similar products using HSV color histograms.
- No deep learning or CNN is used in this pipeline.

---
the original dataset is taken from Kaggle, 
[text](https://www.kaggle.com/datasets/ronakbokaria/myntra-products-dataset)