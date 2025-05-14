import streamlit as st
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.distance import cosine
import os

# Configuration
FEATURE_FILE = 'cnn_features.npy'
CSV_FILE = 'myntra_processed_final.csv'
IMAGE_DIR = 'data/product_images/'

@st.cache_resource
def load_data():
    features = np.load(FEATURE_FILE)
    df = pd.read_csv(CSV_FILE)
    return features, df

@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

def extract_query_features(uploaded_file):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    try:
        image = Image.open(uploaded_file).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = load_model()(tensor)
        return features.squeeze().numpy().flatten()
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def find_similar(query_features, features, top_k=5):
    similarities = []
    for idx, feat in enumerate(features):
        sim = 1 - cosine(query_features, feat)
        similarities.append((idx, sim))
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

# Streamlit UI
st.set_page_config(page_title="CNN Image Search", layout="wide")
st.title("🔍 Myntra Reverse Image Search (CNN)")

features, df = load_data()

uploaded_file = st.file_uploader("Upload a product image:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Query Image", width=300)
    with st.spinner("Extracting CNN features..."):
        query_features = extract_query_features(uploaded_file)
    if query_features is not None:
        with st.spinner("Searching..."):
            matches = find_similar(query_features, features)
        st.subheader("Top Matches")
        cols = st.columns(5)
        for idx, (match_idx, score) in enumerate(matches):
            product = df.iloc[match_idx]
            filename = os.path.basename(product['image_filename'])
            img_path = os.path.join(IMAGE_DIR, filename)
            with cols[idx]:
                st.image(img_path, use_container_width=True, caption=f"Score: {score:.2f}")
                st.markdown(f"**{product['name']}**")
                st.markdown(f"**Price:** ₹{product['price']}")
else:
    st.info("Please upload an image to start searching.")
