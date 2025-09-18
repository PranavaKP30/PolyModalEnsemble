"""
EuroSAT/AllBands Dataset Preprocessing Script

This script preprocesses the EuroSAT/AllBands dataset into a standardized format
for multimodal learning. It creates two modalities:
1. Visual: RGB images extracted from multispectral bands
2. Spectral: Tabular features (vegetation indices, band ratios)

Based on the corrected documentation in PreprocessingDocumentation.md
"""

import os
import json
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import rasterio
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_rgb_from_tiff(tiff_path, output_path):
    """
    Extract RGB bands (B04, B03, B02) from multispectral TIFF and save as JPEG
    
    Args:
        tiff_path (str): Path to input multispectral TIFF file
        output_path (str): Path to save RGB JPEG file
    """
    try:
        with rasterio.open(tiff_path) as src:
            # Read RGB bands (B04=Red, B03=Green, B02=Blue)
            # Note: Band indices are 1-based in rasterio, but 0-based in array
            rgb_data = src.read([4, 3, 2])  # B04, B03, B02
            
            # Convert to 8-bit and transpose for PIL
            rgb_data = np.transpose(rgb_data, (1, 2, 0))
            rgb_data = np.clip(rgb_data / 10000.0 * 255, 0, 255).astype(np.uint8)  # Sentinel-2 scaling with clipping
            
            # Save as JPEG
            rgb_image = Image.fromarray(rgb_data)
            rgb_image.save(output_path, 'JPEG')
            
    except Exception as e:
        logger.error(f"Error processing {tiff_path}: {e}")
        raise

def preprocess_eurosat_simple():
    """
    Simple, correct preprocessing for EuroSAT/AllBands
    Creates only 2 modalities: RGB images + spectral indices as tabular data
    
    Returns:
        tuple: (labels_df, spectral_df) - processed dataframes
    """
    # Paths
    raw_data_path = "Data/EuroSAT/AllBands"
    processed_path = "ProcessedData/EuroSAT"
    
    # Check if raw data path exists
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data path not found: {raw_data_path}")
    
    logger.info(f"Starting preprocessing from {raw_data_path} to {processed_path}")
    
    # Create output directories
    os.makedirs(f"{processed_path}/image_features", exist_ok=True)
    
    # Load dataset splits
    logger.info("Loading dataset splits...")
    train_df = pd.read_csv(f"{raw_data_path}/train.csv")
    val_df = pd.read_csv(f"{raw_data_path}/validation.csv")
    test_df = pd.read_csv(f"{raw_data_path}/test.csv")
    
    logger.info(f"Loaded splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Combine all splits
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    logger.info(f"Total samples to process: {len(all_data)}")
    
    # Process images and extract spectral features
    labels_data = []
    spectral_data = []
    
    logger.info("Processing images and extracting spectral features...")
    
    for idx, row in all_data.iterrows():
        if idx % 1000 == 0:
            logger.info(f"Processing sample {idx}/{len(all_data)}")
            
        sample_id = f"sample_{idx:05d}"
        tiff_path = f"{raw_data_path}/{row['Filename']}"
        
        if os.path.exists(tiff_path):
            # Extract RGB image
            rgb_path = f"{processed_path}/image_features/{sample_id}.jpg"
            extract_rgb_from_tiff(tiff_path, rgb_path)
            
            # Calculate spectral indices as tabular features
            with rasterio.open(tiff_path) as src:
                bands = src.read().astype(np.float32) / 10000.0
                
                # Calculate mean values across the image patch
                # Band mapping: B01=bands[0], B02=bands[1], ..., B11=bands[10], B12=bands[11]
                ndvi_mean = np.mean((bands[7] - bands[3]) / (bands[7] + bands[3] + 1e-8))  # B08-B04
                ndwi_mean = np.mean((bands[2] - bands[7]) / (bands[2] + bands[7] + 1e-8))  # B03-B08
                ndbi_mean = np.mean((bands[10] - bands[7]) / (bands[10] + bands[7] + 1e-8))  # B11-B08
                evi_mean = np.mean(2.5 * (bands[7] - bands[3]) / (bands[7] + 6*bands[3] - 7.5*bands[1] + 1))  # B08-B04-B02
                
                # Band ratios
                ratio_red_nir = np.mean(bands[3] / (bands[7] + 1e-8))  # B04/B08
                ratio_green_nir = np.mean(bands[2] / (bands[7] + 1e-8))  # B03/B08
                ratio_swir_nir = np.mean(bands[10] / (bands[7] + 1e-8))  # B11/B08
                
                # Check for invalid values
                if np.any(np.isnan([ndvi_mean, ndwi_mean, ndbi_mean, evi_mean])):
                    logger.warning(f"NaN values detected in spectral features for {tiff_path}")
                    continue
                
                spectral_data.append({
                    'sample_id': sample_id,
                    'ndvi_mean': ndvi_mean,
                    'ndwi_mean': ndwi_mean,
                    'ndbi_mean': ndbi_mean,
                    'evi_mean': evi_mean,
                    'ratio_red_nir': ratio_red_nir,
                    'ratio_green_nir': ratio_green_nir,
                    'ratio_swir_nir': ratio_swir_nir
                })
            
            # Add to labels
            labels_data.append({
                'sample_id': sample_id,
                'class_name': row['ClassName'],
                'class_id': row['Label'],
                'split': 'train' if idx < len(train_df) else 
                        'validation' if idx < len(train_df) + len(val_df) else 'test'
            })
        else:
            logger.warning(f"TIFF file not found: {tiff_path}")
    
    # Save labels
    logger.info("Saving labels...")
    labels_df = pd.DataFrame(labels_data)
    labels_df.to_csv(f"{processed_path}/labels.csv", index=False)
    
    # Save spectral features as CSV (not separate image files)
    logger.info("Saving spectral features...")
    spectral_df = pd.DataFrame(spectral_data)
    spectral_df.to_csv(f"{processed_path}/spectral_features.csv", index=False)
    
    # Create metadata
    create_metadata(labels_df, spectral_df, len(all_data))
    
    logger.info(f"Preprocessing completed successfully!")
    logger.info(f"Processed {len(labels_df)} samples")
    logger.info(f"Created {len(spectral_df.columns)-1} spectral features per sample")
    
    return labels_df, spectral_df

def create_metadata(labels_df, spectral_df, total_samples):
    """
    Create metadata.json with dataset statistics and configuration
    
    Args:
        labels_df (pd.DataFrame): Labels dataframe
        spectral_df (pd.DataFrame): Spectral features dataframe
        total_samples (int): Total number of samples
    """
    # Calculate class distribution
    class_counts = labels_df['class_name'].value_counts().to_dict()
    
    # Calculate split distribution
    split_counts = labels_df['split'].value_counts().to_dict()
    
    metadata = {
        "dataset_name": "EuroSAT_AllBands",
        "total_samples": int(total_samples),
        "image_dimensions": [64, 64, 3],
        "image_format": "JPEG",
        "num_classes": 10,
        "class_names": list(labels_df['class_name'].unique()),
        "class_distribution": class_counts,
        "split_distribution": split_counts,
        "modalities": {
            "visual": {
                "type": "RGB_images",
                "format": "JPEG",
                "dimensions": [64, 64, 3],
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                }
            },
            "spectral": {
                "type": "spectral_indices",
                "format": "CSV",
                "features": ["ndvi_mean", "ndwi_mean", "ndbi_mean", "evi_mean", "ratio_red_nir", "ratio_green_nir", "ratio_swir_nir"]
            }
        },
        "preprocessing_info": {
            "source": "EuroSAT/AllBands",
            "original_structure": "class_folders",
            "processed_structure": "unified_image_features",
            "naming_convention": "sample_XXXXX.jpg"
        }
    }
    
    # Save metadata
    with open("ProcessedData/EuroSAT/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Metadata saved to ProcessedData/EuroSAT/metadata.json")

class EuroSATDataset(Dataset):
    """
    PyTorch Dataset for EuroSAT processed data
    """
    def __init__(self, data_path, split='train', transform=None):
        self.data_path = data_path
        self.labels_df = pd.read_csv(f"{data_path}/labels.csv")
        self.split_df = self.labels_df[self.labels_df['split'] == split]
        self.transform = transform
        
        # Load spectral features once
        self.spectral_df = pd.read_csv(f"{data_path}/spectral_features.csv")
    
    def __len__(self):
        return len(self.split_df)
    
    def __getitem__(self, idx):
        row = self.split_df.iloc[idx]
        
        # Load image
        img_path = f"{self.data_path}/image_features/{row['sample_id']}.jpg"
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Load spectral features
        spectral_row = self.spectral_df[self.spectral_df['sample_id'] == row['sample_id']].iloc[0]
        spectral_features = torch.tensor(spectral_row.iloc[1:].values.astype(np.float32), dtype=torch.float32)
        
        return {
            'visual': image,
            'spectral': spectral_features,
            'label': row['class_id'],
            'class_name': row['class_name'],
            'sample_id': row['sample_id']
        }

def get_transforms(split='train'):
    """
    Get appropriate transforms for training/validation
    
    Args:
        split (str): 'train' or 'validation'/'test'
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def validate_preprocessing():
    """
    Validate that preprocessing was completed successfully
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    processed_path = "ProcessedData/EuroSAT"
    
    logger.info("Validating preprocessing results...")
    
    # Check file structure
    required_files = ['labels.csv', 'spectral_features.csv', 'metadata.json', 'image_features/']
    for file in required_files:
        if not os.path.exists(f"{processed_path}/{file}"):
            logger.error(f"Missing: {file}")
            return False
    
    # Check image count
    labels_df = pd.read_csv(f"{processed_path}/labels.csv")
    image_count = len([f for f in os.listdir(f"{processed_path}/image_features") if f.endswith('.jpg')])
    
    if len(labels_df) != image_count:
        logger.error(f"Mismatch: {len(labels_df)} labels vs {image_count} images")
        return False
    
    # Check spectral features count
    spectral_df = pd.read_csv(f"{processed_path}/spectral_features.csv")
    if len(labels_df) != len(spectral_df):
        logger.error(f"Mismatch: {len(labels_df)} labels vs {len(spectral_df)} spectral features")
        return False
    
    # Check class distribution
    logger.info("Class Distribution:")
    logger.info(labels_df['class_name'].value_counts())
    
    # Check split distribution
    logger.info("Split Distribution:")
    logger.info(labels_df['split'].value_counts())
    
    logger.info("Preprocessing validation passed!")
    logger.info(f"Total samples: {len(labels_df)}")
    logger.info(f"Total images: {image_count}")
    logger.info(f"Total spectral features: {len(spectral_df)}")
    
    return True

def load_eurosat_for_mainmodel():
    """
    Load preprocessed EuroSAT data in MainModel format
    
    Returns:
        dict: Dictionary containing datasets and metadata
    """
    data_path = "ProcessedData/EuroSAT"
    
    # Load labels
    labels_df = pd.read_csv(f"{data_path}/labels.csv")
    
    # Load metadata
    with open(f"{data_path}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Create datasets
    train_dataset = EuroSATDataset(data_path, 'train', get_transforms('train'))
    val_dataset = EuroSATDataset(data_path, 'validation', get_transforms('validation'))
    test_dataset = EuroSATDataset(data_path, 'test', get_transforms('test'))
    
    return {
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset,
        'metadata': metadata,
        'labels': labels_df
    }

if __name__ == "__main__":
    # Run preprocessing
    logger.info("Starting EuroSAT/AllBands preprocessing...")
    
    try:
        # Run preprocessing
        labels_df, spectral_df = preprocess_eurosat_simple()
        
        # Validate results
        if validate_preprocessing():
            logger.info("✅ Preprocessing completed successfully!")
            
            # Test data loading
            logger.info("Testing data loading...")
            data = load_eurosat_for_mainmodel()
            logger.info(f"Train dataset size: {len(data['train'])}")
            logger.info(f"Validation dataset size: {len(data['validation'])}")
            logger.info(f"Test dataset size: {len(data['test'])}")
            
            # Test a single sample
            sample = data['train'][0]
            logger.info(f"Sample keys: {list(sample.keys())}")
            logger.info(f"Visual shape: {sample['visual'].shape}")
            logger.info(f"Spectral shape: {sample['spectral'].shape}")
            logger.info(f"Label: {sample['label']} ({sample['class_name']})")
            
        else:
            logger.error("❌ Preprocessing validation failed!")
            
    except Exception as e:
        logger.error(f"❌ Preprocessing failed with error: {e}")
        raise
