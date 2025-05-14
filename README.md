# Reverse Image Search for Myntra Fashion Products

A dual-pipeline reverse image search engine for Myntra-style fashion products, using both traditional computer vision (HSV histogram) and deep learning (CNN/ResNet50) approaches. The system is designed to let users upload a product image and find visually similar products from a large catalog.

---

## 📖 Project Overview

This project implements a reverse image search solution for fashion e-commerce, inspired by Myntra’s image search feature. The goal is to allow users to upload an image of a fashion product (like a shirt, shoe, or dress) and retrieve similar products from a dataset of Myntra product images.

The project is organized into two main pipelines:

- **HSV Pipeline:** Uses color histograms for fast, color-based similarity search.
- **CNN Pipeline:** Uses deep features from a ResNet50 model for robust, semantic similarity.

Both pipelines are deployed via a user-friendly Streamlit web interface.

---

## 🛠️ Workflow & File Structure

**General Steps:**
1. **Sample and Clean Data:** Select a subset from the large Myntra dataset, download images, and remove missing/corrupt files.
2. **Feature Extraction:** Extract either HSV histograms or CNN (ResNet50) features from each image.
3. **Similarity Search:** Given a query image, extract its features and find the most similar images in the dataset.
4. **Web App:** Use Streamlit to provide an interactive search interface.

**Main Files:**
- `sample_data.py` – Sample a subset of the Myntra dataset.
- `download_images.py` – Download product images.
- `verify_images.py` – Remove missing/corrupt images.
- `fix_image_filenames.py` – Ensure filenames are clean for processing.
- `extracting_features.py` – Extract HSV features for each image.
- `cnn_feature_extraction.py` – Extract deep features using ResNet50.
- `app.py` – Streamlit app (HSV pipeline).
- `app_cnn.py` – Streamlit app (CNN pipeline).

---

## 🔍 Approaches Used

### 1. HSV Pipeline (Color Histogram)
- **How it works:**  
  - Extracts 3D HSV color histograms from each image.
  - Compares histograms using cosine similarity.
- **Strengths:**  
  - Fast, simple, interpretable.
- **Drawbacks:**  
  - Only considers color, not shape or texture.
  - Fails for products with similar colors but different styles.

### 2. CNN Pipeline (ResNet50 Deep Features)
- **How it works:**  
  - Uses a pretrained ResNet50 to extract 2048-dim feature vectors from each image.
  - Compares features using cosine similarity.
- **Strengths:**  
  - Captures both color and complex visual patterns.
  - Finds semantically similar products (e.g., similar style, not just color).
- **Drawbacks:**  
  - Requires more compute (slower feature extraction).
  - Needs more RAM for large datasets.

---

## 🧭 Why Switch to CNN?

- The HSV pipeline was limited to color similarity and often returned visually different products with similar colors.
- CNN-based features significantly improved the quality of matches by considering patterns, textures, and overall style, not just color.
- This was especially important for fashion, where style and cut matter as much as color.


## 🚀 How to Run

### 1. HSV Pipeline

### Use these files:
python sample_data.py
python download_images.py
python verify_images.py
python fix_image_filenames.py
python extracting_features.py
streamlit run app.py

### 2. CNN Pipeline

### use these files after using them in HSV
python cnn_feature_extraction.py
streamlit run app_cnn.py

## ⚠️ Dataset & Project Limitations

- **Broken Product URLs:** Many product URLs in the original Myntra dataset are no longer valid, so product detail pages cannot be linked.
- **Duplicates:** Some products/images appear multiple times, which can lead to repeated results in search.
- **No Category Filtering:** The dataset lacks reliable product categories, so cross-category matches can occur.
- **HSV Drawbacks:** Color-based retrieval sometimes fails for visually different products with similar color palettes.
- **CNN Drawbacks:** Feature extraction is slow on CPU and requires significant memory for large datasets.

---


## 🚧 Future Improvements

- **Deduplication:** Remove near-duplicate images using perceptual hashing or feature clustering.
- **Hybrid Features:** Combine HSV and CNN features for even better results.
- **Category Filtering:** Add product categories to improve relevance.
- **Efficient Search:** Use FAISS or Annoy for faster nearest neighbor search on large datasets.
- **Better Dataset:** Use or build a dataset with valid product URLs and richer metadata.

---

## 📚 References

- [Myntra Products Dataset (Kaggle)](https://www.kaggle.com/datasets/ronakbokaria/myntra-products-dataset)
- [PyTorch ResNet50](https://pytorch.org/hub/pytorch_vision_resnet/)
- [Streamlit Documentation](https://streamlit.io/)
- [imagededup (for deduplication)](https://github.com/idealo/imagededup)

---

*I have developed this project to intend as a demonstration of reverse image search techniques for fashion e-commerce, using both traditional and deep learning methods.*
