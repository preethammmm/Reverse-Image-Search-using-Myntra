# apphsv.py
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
from scipy.spatial.distance import cosine

# Configuration
FEATURE_CSV = 'myntra_features.csv'
IMAGE_DIR = 'data/product_images/'
BINS = (8, 8, 8)
TOP_K = 5

@st.cache_resource
def load_features():
    df = pd.read_csv(FEATURE_CSV)
    
    # Verify required columns
    required_columns = ['image_filename', 'name', 'price']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"'{FEATURE_CSV}' is missing required column: '{col}'")
            st.stop()
    
    # Get feature columns (all except metadata)
    feature_cols = [col for col in df.columns if col.startswith('hsv_feature')]
    return df[feature_cols].values, df['image_filename'].values, df

def compute_hsv_histogram(image):
    # image: PIL Image
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, BINS, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def find_similar(query_feature, features, filenames, top_k=TOP_K):
    similarities = []
    for idx, feat in enumerate(features):
        sim = 1 - cosine(query_feature, feat)
        similarities.append((idx, sim))
    sorted_matches = sorted(similarities, key=lambda x: -x[1])[:top_k]
    return sorted_matches

def display_results(matches, filenames, df):
    cols = st.columns(5)  # Always 5 columns for consistency
    for i, (idx, score) in enumerate(matches[:5]):  # Ensure max 5 matches
        product = df.iloc[idx]
        img_path = os.path.join(IMAGE_DIR, os.path.basename(filenames[idx]))
        with cols[i]:
            st.image(img_path, use_container_width=True, caption=f"Score: {score:.2f}")
            st.markdown(f"**{product['name']}**")
            st.markdown(f"**Price:** ₹{product['price']}")

# Streamlit UI
st.set_page_config(page_title="HSV Reverse Image Search", layout="wide")
st.title("🔍 Myntra HSV Reverse Image Search")

features, filenames, df = load_features()

uploaded_file = st.file_uploader("Upload a product image:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    st.image(uploaded_file, caption="Query Image", width=300)
    
    # Process image
    with st.spinner("Analyzing image..."):
        image = Image.open(uploaded_file).convert('RGB')
        query_feature = compute_hsv_histogram(image)
    
    if query_feature is not None:
        with st.spinner("Searching for similar products..."):
            matches = find_similar(query_feature, features, filenames)
        
        st.subheader("Top 5 Matching Products")
        display_results(matches, filenames, df)
else:
    st.info("Please upload a product image to start searching.")

st.markdown("---")
st.caption("Powered by HSV color histograms · Myntra dataset · Streamlit demo")
