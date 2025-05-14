import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class MyntraDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        
        # Validate and sanitize filenames
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            filename = os.path.basename(row['image_filename'])
            full_path = os.path.join(image_dir, filename)
            if os.path.exists(full_path):
                self.valid_indices.append(idx)
                
        self.df = self.df.iloc[self.valid_indices].reset_index(drop=True)
        print(f"Initialized dataset with {len(self)} valid images")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        filename = os.path.basename(self.df.iloc[idx]['image_filename'])
        img_path = os.path.join(self.image_dir, filename)
        
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                return self.transform(img)
            return img
        except Exception as e:
            print(f"\n[ERROR] Corrupted image skipped: {img_path}")
            print(f"Details: {str(e)}")
            return None

def collate_fn(batch):
    """Filter invalid items and stack tensors"""
    batch = [item for item in batch if item is not None]
    return torch.stack(batch)  # Convert list of tensors to batched tensor

def initialize_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-2])  # Remove last 2 layers
    return model.eval()

def extract_features(data_loader, model, device):
    features = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting features"):
            batch = batch.to(device)
            batch_features = model(batch)
            
            # Global Average Pooling
            batch_features = torch.nn.functional.adaptive_avg_pool2d(batch_features, (1, 1))
            batch_features = batch_features.view(batch.size(0), -1)
            features.append(batch_features.cpu().numpy())
    
    return np.concatenate(features, axis=0)

def main():
    # Configuration
    CSV_PATH = 'myntra_sample_clean.csv'
    IMAGE_DIR = 'data/product_images/'
    OUTPUT_FEATURES = 'cnn_features.npy'
    OUTPUT_CSV = 'myntra_processed_final.csv'
    BATCH_SIZE = 32
    IMG_SIZE = 224

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Initialize model
    model = initialize_model().to(device)
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Create dataset and data loader
    dataset = MyntraDataset(CSV_PATH, IMAGE_DIR, transform)
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,  # Required for Windows compatibility
        collate_fn=collate_fn,
        shuffle=False
    )
    
    # Extract features
    features = extract_features(data_loader, model, device)
    
    if features.size == 0:
        raise ValueError("No valid features extracted. Check input data.")
    
    # Normalize features
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    # Save outputs
    np.save(OUTPUT_FEATURES, features.astype(np.float32))
    dataset.df.to_csv(OUTPUT_CSV, index=False)
    
    print("\nFeature extraction completed successfully!")
    print(f"Saved features: {OUTPUT_FEATURES}")
    print(f"Feature shape: {features.shape}")
    print(f"Saved metadata: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()