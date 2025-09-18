## EuroSAT/AllBands Dataset - Current Raw Data Structure

### Complete File Directory Structure
```
Data/EuroSAT/AllBands/
├── AnnualCrop/                    # 3,000 multispectral images
│   ├── AnnualCrop_1.tif
│   ├── AnnualCrop_2.tif
│   └── ... (3,000 total)
├── Forest/                        # 3,000 multispectral images
│   ├── Forest_1.tif
│   ├── Forest_2.tif
│   └── ... (3,000 total)
├── HerbaceousVegetation/          # 3,000 multispectral images
│   ├── HerbaceousVegetation_1.tif
│   └── ... (3,000 total)
├── Highway/                       # 2,500 multispectral images
│   ├── Highway_1.tif
│   └── ... (2,500 total)
├── Industrial/                    # 2,500 multispectral images
│   ├── Industrial_1.tif
│   └── ... (2,500 total)
├── Pasture/                       # 2,000 multispectral images
│   ├── Pasture_1.tif
│   └── ... (2,000 total)
├── PermanentCrop/                 # 2,500 multispectral images
│   ├── PermanentCrop_1.tif
│   └── ... (2,500 total)
├── Residential/                   # 3,000 multispectral images
│   ├── Residential_1.tif
│   └── ... (3,000 total)
├── River/                         # 2,500 multispectral images
│   ├── River_1.tif
│   └── ... (2,500 total)
├── SeaLake/                       # 3,597 multispectral images
│   ├── SeaLake_1.tif
│   └── ... (3,597 total)
├── label_map.json                 # Class name to ID mapping
├── train.csv                      # 19,317 training samples
├── validation.csv                 # 5,519 validation samples
└── test.csv                       # 2,759 test samples
```

### Dataset Statistics
- **Total Images**: 27,595 multispectral satellite images (referenced in CSV)
- **Total TIFF Files**: 27,597 files (2 extra files not in CSV)
- **Total Size**: 2.8 GB
- **Image Format**: TIFF (13-band multispectral, 64×64 pixels)
- **File Size Range**: ~100-110 KB per image
- **Class Distribution** (from CSV):
  - AnnualCrop: 3,000 images
  - Forest: 3,000 images  
  - HerbaceousVegetation: 3,000 images
  - Highway: 2,500 images
  - Industrial: 2,500 images
  - Pasture: 2,000 images
  - PermanentCrop: 2,500 images
  - Residential: 3,000 images
  - River: 2,500 images
  - SeaLake: 3,595 images (largest class)

### Available Data Files

#### 1. Multispectral Image Files
- **Location**: 10 class-specific folders
- **Format**: TIFF images (GeoTIFF)
- **Dimensions**: 64×64 pixels
- **Channels**: 13 spectral bands
- **Naming Convention**: `ClassName_####.tif`
- **Content**: Full Sentinel-2 multispectral satellite imagery patches
- **File Size**: ~100-110 KB per image
- **Data Type**: 16-bit unsigned integer

#### 2. Label Mapping File (`label_map.json`)
```json
{
    "AnnualCrop": 0,
    "Forest": 1,
    "HerbaceousVegetation": 2,
    "Highway": 3,
    "Industrial": 4,
    "Pasture": 5,
    "PermanentCrop": 6,
    "Residential": 7,
    "River": 8,
    "SeaLake": 9
}
```

#### 3. Dataset Split Files
- **train.csv**: 19,317 samples (70% of data)
- **validation.csv**: 5,519 samples (20% of data)  
- **test.csv**: 2,759 samples (10% of data)

**CSV Format**:
```csv
Filename,Label,ClassName
PermanentCrop/PermanentCrop_2401.tif,6,PermanentCrop
PermanentCrop/PermanentCrop_1006.tif,6,PermanentCrop
SeaLake/SeaLake_2010.tif,9,SeaLake
```

**CSV Columns**:
- **Filename**: Relative path to multispectral TIFF file
- **Label**: Numerical class ID (0-9)
- **ClassName**: Textual class name

### Modalities Present in Raw Data

#### **1. Visual Modalities**
- **Multispectral Images** (Primary): 13-channel TIFF images (64×64×13)
  - Full Sentinel-2 spectral bands (B01-B12 + B8A)
  - Includes visible, near-infrared, red-edge, and short-wave infrared bands
  - 16-bit unsigned integer data type
  - Can be processed to extract RGB, vegetation indices, and other spectral features

#### **2. Text/Categorical Modalities**
- **Class Names**: Textual land cover class names (e.g., "AnnualCrop", "Forest")
- **Numerical Labels**: Class IDs (0-9) for machine learning
- **File Metadata**: Original filenames and directory paths

#### **3. Geospatial Metadata**
- **Note**: The raw EuroSAT/AllBands dataset does NOT contain explicit geospatial metadata files
- **Available**: Only implicit geographic information through class-based organization
- **Missing**: No latitude/longitude coordinates, country info, or acquisition timestamps in raw data

#### **4. Spectral Band Information** (AllBands only)
- **B01**: Coastal aerosol (443nm) - atmospheric correction
- **B02**: Blue (490nm) - water body detection
- **B03**: Green (560nm) - vegetation health
- **B04**: Red (665nm) - chlorophyll absorption
- **B05**: Red-edge 1 (705nm) - vegetation stress
- **B06**: Red-edge 2 (740nm) - leaf area index
- **B07**: Red-edge 3 (783nm) - vegetation structure
- **B08**: Near-infrared (842nm) - biomass estimation
- **B8A**: Near-infrared narrow (865nm) - vegetation density
- **B09**: Water vapor (945nm) - atmospheric correction
- **B10**: Cirrus (1375nm) - cloud detection
- **B11**: Short-wave infrared 1 (1610nm) - soil moisture
- **B12**: Short-wave infrared 2 (2190nm) - mineral mapping

## Required Preprocessing Steps

### Overview
The EuroSAT/AllBands dataset requires comprehensive preprocessing to transform the raw multispectral data into the standardized format expected by the MainModel API. The preprocessing involves:

1. **Multispectral data processing** (13-band TIFF to multiple modalities)
2. **RGB extraction** from multispectral bands for standard computer vision
3. **Spectral feature extraction** (vegetation indices, band ratios)
4. **Data reorganization** and consistent naming
5. **Label standardization** across all modalities
6. **Metadata creation** for model configuration

### Step 1: Data Reorganization and Label Creation

**Objective**: Transform scattered class folders into unified structure with consistent naming

**Implementation**:
```python
import os
import json
import shutil
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
import rasterio  # For AllBands TIFF files

def preprocess_eurosat_simple():
    """
    Simple, correct preprocessing for EuroSAT/AllBands
    Creates only 2 modalities: RGB images + spectral indices as tabular data
    """
    # Paths
    raw_data_path = "Data/EuroSAT/AllBands"
    processed_path = "ProcessedData/EuroSAT"
    
    # Check if raw data path exists
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data path not found: {raw_data_path}")
    
    # Create output directories
    os.makedirs(f"{processed_path}/image_features", exist_ok=True)
    
    # Load dataset splits
    train_df = pd.read_csv(f"{raw_data_path}/train.csv")
    val_df = pd.read_csv(f"{raw_data_path}/validation.csv")
    test_df = pd.read_csv(f"{raw_data_path}/test.csv")
    
    # Combine all splits
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Process images and extract spectral features
    labels_data = []
    spectral_data = []
    
    for idx, row in all_data.iterrows():
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
                    print(f"Warning: NaN values detected in spectral features for {tiff_path}")
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
    
    # Save labels
    labels_df = pd.DataFrame(labels_data)
    labels_df.to_csv(f"{processed_path}/labels.csv", index=False)
    
    # Save spectral features as CSV (not separate image files)
    spectral_df = pd.DataFrame(spectral_data)
    spectral_df.to_csv(f"{processed_path}/spectral_features.csv", index=False)
    
    return labels_df, spectral_df

def extract_rgb_from_tiff(tiff_path, output_path):
    """
    Extract RGB bands (B04, B03, B02) from multispectral TIFF and save as JPEG
    """
    try:
        with rasterio.open(tiff_path) as src:
            # Read RGB bands (B04=Red, B03=Green, B02=Blue)
            # Note: Band indices are 1-based in rasterio
            rgb_data = src.read([4, 3, 2])  # B04, B03, B02
            
            # Convert to 8-bit and transpose for PIL
            rgb_data = np.transpose(rgb_data, (1, 2, 0))
            rgb_data = np.clip(rgb_data / 10000.0 * 255, 0, 255).astype(np.uint8)  # Sentinel-2 scaling with clipping
            
            # Save as JPEG
            rgb_image = Image.fromarray(rgb_data)
            rgb_image.save(output_path, 'JPEG')
    except Exception as e:
        print(f"Error processing {tiff_path}: {e}")

# Run simple preprocessing
print("Processing EuroSAT/AllBands dataset...")
labels_df, spectral_df = preprocess_eurosat_simple()
print(f"Processed {len(labels_df)} samples (RGB images + spectral features)")
print(f"Created {len(spectral_df.columns)-1} spectral features per sample")
```

### Step 2: Metadata Creation

**Objective**: Generate comprehensive dataset metadata for model configuration

**Implementation**:
```python
def create_metadata(labels_df, total_samples):
    """
    Create metadata.json with dataset statistics and configuration
    """
    # Calculate class distribution
    class_counts = labels_df['class_name'].value_counts().to_dict()
    
    # Calculate split distribution
    split_counts = labels_df['split'].value_counts().to_dict()
    
    metadata = {
        "dataset_name": "EuroSAT_LimitedBands",
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
    
    return metadata

# Create metadata
metadata = create_metadata(labels_df, total_samples)
```

### Step 3: Image Preprocessing Pipeline

**Objective**: Define transforms for model training and inference

**Implementation**:
```python
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

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
        spectral_features = torch.tensor(spectral_row.iloc[1:].values, dtype=torch.float32)
        
        return {
            'visual': image,
            'spectral': spectral_features,
            'label': row['class_id'],
            'class_name': row['class_name'],
            'sample_id': row['sample_id']
        }

# Define transforms
def get_transforms(split='train'):
    """
    Get appropriate transforms for training/validation
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

# Create datasets
train_dataset = EuroSATDataset("ProcessedData/EuroSAT", 'train', get_transforms('train'))
val_dataset = EuroSATDataset("ProcessedData/EuroSAT", 'validation', get_transforms('validation'))
test_dataset = EuroSATDataset("ProcessedData/EuroSAT", 'test', get_transforms('test'))

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### Step 4: Data Validation

**Objective**: Verify preprocessing quality and data integrity

**Implementation**:
```python
def validate_preprocessing():
    """
    Validate that preprocessing was completed successfully
    """
    processed_path = "ProcessedData/EuroSAT"
    
    # Check file structure
    required_files = ['labels.csv', 'metadata.json', 'image_features/']
    for file in required_files:
        if not os.path.exists(f"{processed_path}/{file}"):
            print(f"Missing: {file}")
            return False
    
    # Check image count
    labels_df = pd.read_csv(f"{processed_path}/labels.csv")
    image_count = len([f for f in os.listdir(f"{processed_path}/image_features") if f.endswith('.jpg')])
    
    if len(labels_df) != image_count:
        print(f"Mismatch: {len(labels_df)} labels vs {image_count} images")
        return False
    
    # Check class distribution
    print("Class Distribution:")
    print(labels_df['class_name'].value_counts())
    
    # Check split distribution
    print("\nSplit Distribution:")
    print(labels_df['split'].value_counts())
    
    print(f"\nPreprocessing validation passed!")
    print(f"Total samples: {len(labels_df)}")
    print(f"Total images: {image_count}")
    
    return True

# Run validation
validate_preprocessing()
```

### Target File Structure (After Preprocessing)

```
ProcessedData/EuroSAT/
├── labels.csv                    # [sample_id, class_name, class_id, split]
├── image_features/               # RGB images extracted from multispectral data
│   ├── sample_00001.jpg
│   ├── sample_00002.jpg
│   └── ... (27,595 images)
└── spectral_features.csv         # Calculated spectral indices as tabular data
```

### Expected Output Files

1. **labels.csv**: Standardized label file with sample IDs, class information, and split assignments
2. **image_features/**: RGB images extracted from multispectral data (27,595 JPEG files)
3. **spectral_features.csv**: Calculated spectral indices as tabular data
   - NDVI, NDWI, NDBI, EVI (mean values per image patch)
   - Band ratios and other spectral features
   - One row per sample, features as columns

### Usage with MainModel API

```python
# Load preprocessed data for MainModel
def load_eurosat_for_mainmodel():
    """
    Load preprocessed EuroSAT data in MainModel format
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

# Load data
eurosat_data = load_eurosat_for_mainmodel()
```
_________________________________________________________________________________________

