import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

class ProductSearch:
    def __init__(self):
        self.features = np.load('hybrid_features.npy')
        self.df = pd.read_csv('myntra_hybrid_clean.csv')
    
    def find_similar(self, query_features, top_n=5):
        """Find similar products using hybrid features"""
        similarities = []
        for idx, feat in enumerate(self.features):
            try:
                similarities.append((idx, 1 - cosine(query_features, feat)))
            except:
                continue
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

# Example usage
if __name__ == "__main__":
    searcher = ProductSearch()
    test_features = np.random.randn(2560)  # Replace with actual query features
    matches = searcher.find_similar(test_features)
    print("Top matches:", matches)