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
├── train.csv                      # 19,318 training samples
├── validation.csv                 # 5,520 validation samples
└── test.csv                       # 2,760 test samples
```

### Dataset Statistics
- **Total Images**: 27,597 multispectral satellite images (actual TIFF files)
- **Total CSV References**: 27,598 rows (including headers)
- **Total Size**: 2.8 GB
- **Image Format**: TIFF (13-band multispectral, 64×64 pixels)
- **File Size Range**: ~100-110 KB per image
- **Class Distribution** (from actual file counts):
  - AnnualCrop: 3,000 images
  - Forest: 3,000 images  
  - HerbaceousVegetation: 3,000 images
  - Highway: 2,500 images
  - Industrial: 2,500 images
  - Pasture: 2,000 images
  - PermanentCrop: 2,500 images
  - Residential: 3,000 images
  - River: 2,500 images
  - SeaLake: 3,597 images (largest class)

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
- **train.csv**: 19,318 samples (70% of data)
- **validation.csv**: 5,520 samples (20% of data)  
- **test.csv**: 2,760 samples (10% of data)

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

1. **Multispectral data processing** (13-band TIFF to 5 distinct spectral modalities)
2. **Visual RGB extraction** from multispectral bands (B04, B03, B02)
3. **Near-Infrared modality** extraction (B08, B8A)
4. **Red-Edge modality** extraction (B05, B06, B07)
5. **Short-Wave Infrared modality** extraction (B11, B12)
6. **Atmospheric modality** extraction (B01, B09, B10)
7. **Data reorganization** and consistent naming
8. **Label standardization** across all modalities
9. **Metadata creation** for model configuration

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

def extract_all_spectral_bands(tiff_path):
    """
    Extract all 13 spectral bands from multispectral TIFF with proper band order verification
    
    Args:
        tiff_path (str): Path to input multispectral TIFF file
        
    Returns:
        dict: Dictionary containing all spectral bands organized by modality
    """
    try:
        with rasterio.open(tiff_path) as src:
            # Validate TIFF file
            if src.count != 13:
                raise ValueError(f"Expected 13 bands, found {src.count} in {tiff_path}")
            
            # Read all bands (13 bands total)
            bands = src.read().astype(np.float32) / 10000.0  # Normalize to 0-1 range
            
            # Verify band order using Sentinel-2 characteristics
            # EuroSAT uses standard Sentinel-2 band order: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
            band_means = [np.mean(bands[i]) for i in range(13)]
            
            # Band mapping from Sentinel-2 1-based to 0-based indices (verified order)
            band_mapping = {
                'B01': 0,   # Coastal aerosol (60m) - typically low values
                'B02': 1,   # Blue (10m)
                'B03': 2,   # Green (10m)
                'B04': 3,   # Red (10m)
                'B05': 4,   # Red-edge 1 (20m)
                'B06': 5,   # Red-edge 2 (20m)
                'B07': 6,   # Red-edge 3 (20m)
                'B08': 7,   # NIR (10m) - typically high values for vegetation
                'B8A': 8,   # NIR narrow (20m)
                'B09': 9,   # Water vapor (60m)
                'B10': 10,  # Cirrus (60m)
                'B11': 11,  # SWIR 1 (20m)
                'B12': 12   # SWIR 2 (20m)
            }
            
            # Validate band order using Sentinel-2 characteristics
            # B01 (Coastal) should have lower values than B08 (NIR) for most land cover types
            # B04 (Red) should have lower values than B08 (NIR) for vegetation
            b01_mean = band_means[0]  # Coastal
            b04_mean = band_means[3]  # Red
            b08_mean = band_means[7]  # NIR
            
            # Check for reasonable band characteristics
            if b08_mean < b04_mean:  # NIR should be higher than Red for vegetation
                print(f"Warning: Unusual NIR/Red relationship in {tiff_path} (NIR={b08_mean:.2f}, Red={b04_mean:.2f})")
            
            if b01_mean > b08_mean:  # Coastal should generally be lower than NIR
                print(f"Warning: Unusual Coastal/NIR relationship in {tiff_path} (Coastal={b01_mean:.2f}, NIR={b08_mean:.2f})")
            
            # Additional validation: check for reasonable value ranges
            for i, mean_val in enumerate(band_means):
                if mean_val < 0 or mean_val > 1.0:  # After normalization, values should be 0-1
                    print(f"Warning: Band {i+1} has unusual mean value {mean_val:.4f} in {tiff_path}")
            
            # Check for corrupted data (all zeros or all same value)
            for i, band in enumerate(bands):
                if np.all(band == 0) or np.std(band) < 1e-6:
                    print(f"Warning: Band {i+1} appears corrupted in {tiff_path}")
            
            # Organize bands by modality
            spectral_data = {
                'visual_rgb': {
                    'B04': bands[band_mapping['B04']],  # Red
                    'B03': bands[band_mapping['B03']],  # Green
                    'B02': bands[band_mapping['B02']]   # Blue
                },
                'near_infrared': {
                    'B08': bands[band_mapping['B08']],  # NIR
                    'B8A': bands[band_mapping['B8A']]   # NIR narrow
                },
                'red_edge': {
                    'B05': bands[band_mapping['B05']],  # Red-edge 1
                    'B06': bands[band_mapping['B06']],  # Red-edge 2
                    'B07': bands[band_mapping['B07']]   # Red-edge 3
                },
                'short_wave_infrared': {
                    'B11': bands[band_mapping['B11']],  # SWIR 1
                    'B12': bands[band_mapping['B12']]   # SWIR 2
                },
                'atmospheric': {
                    'B01': bands[band_mapping['B01']],  # Coastal aerosol
                    'B09': bands[band_mapping['B09']],  # Water vapor
                    'B10': bands[band_mapping['B10']]   # Cirrus
                }
            }
            
            return spectral_data
            
    except Exception as e:
        print(f"Error processing {tiff_path}: {e}")
        return None

def preprocess_eurosat_enhanced():
    """
    Enhanced preprocessing for EuroSAT/AllBands
    Creates 5 distinct spectral modalities: Visual RGB, Near-Infrared, Red-Edge, SWIR, Atmospheric
    """
    # Paths
    raw_data_path = "Data/EuroSAT/AllBands"
    processed_path = "ProcessedData/EuroSAT"
    
    # Check if raw data path exists
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data path not found: {raw_data_path}")
    
    # Create output directories for all 5 modalities
    os.makedirs(f"{processed_path}/visual_rgb", exist_ok=True)
    os.makedirs(f"{processed_path}/near_infrared", exist_ok=True)
    os.makedirs(f"{processed_path}/red_edge", exist_ok=True)
    os.makedirs(f"{processed_path}/short_wave_infrared", exist_ok=True)
    os.makedirs(f"{processed_path}/atmospheric", exist_ok=True)
    
    # Load dataset splits
    train_df = pd.read_csv(f"{raw_data_path}/train.csv")
    val_df = pd.read_csv(f"{raw_data_path}/validation.csv")
    test_df = pd.read_csv(f"{raw_data_path}/test.csv")
    
    # Combine all splits
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Process images and extract all spectral modalities
    labels_data = []
    modality_metadata = {
        'visual_rgb': [],
        'near_infrared': [],
        'red_edge': [],
        'short_wave_infrared': [],
        'atmospheric': []
    }
    
    processed_count = 0
    error_count = 0
    
    for idx, row in all_data.iterrows():
        sample_id = f"sample_{idx:05d}"
        tiff_path = f"{raw_data_path}/{row['Filename']}"
        
        try:
            if not os.path.exists(tiff_path):
                print(f"Warning: TIFF file not found: {tiff_path}")
                error_count += 1
                continue
            
            # Extract all spectral bands with error handling
            spectral_data = extract_all_spectral_bands(tiff_path)
            if spectral_data is None:
                print(f"Error: Failed to extract spectral data from {tiff_path}")
                error_count += 1
                continue
            
            # Save RGB image with validation
            try:
                rgb_data = spectral_data['visual_rgb']
                rgb_array = np.stack([rgb_data['B04'], rgb_data['B03'], rgb_data['B02']], axis=0)
                rgb_array = np.transpose(rgb_array, (1, 2, 0))
                rgb_array = np.clip(rgb_array * 255, 0, 255).astype(np.uint8)
                
                # Validate RGB array
                if rgb_array.shape != (64, 64, 3):
                    print(f"Warning: RGB array has wrong shape {rgb_array.shape} for {sample_id}")
                
                rgb_image = Image.fromarray(rgb_array)
                rgb_path = f"{processed_path}/visual_rgb/{sample_id}.jpg"
                rgb_image.save(rgb_path)
            except Exception as e:
                print(f"Error saving RGB image for {sample_id}: {e}")
                error_count += 1
                continue
            
            # Save other modalities as NPY files with validation
            for modality_name, modality_data in spectral_data.items():
                if modality_name != 'visual_rgb':
                    try:
                        modality_path = f"{processed_path}/{modality_name}/{sample_id}.npy"
                        np.save(modality_path, modality_data)
                        
                        # Store modality metadata
                        modality_metadata[modality_name].append({
                            'sample_id': sample_id,
                            'bands': list(modality_data.keys()),
                            'shape': list(modality_data.values())[0].shape if modality_data else None,
                            'data_type': 'float32'
                        })
                    except Exception as e:
                        print(f"Error saving {modality_name} for {sample_id}: {e}")
                        error_count += 1
                        continue
            
            # Add to labels
            labels_data.append({
                'sample_id': sample_id,
                'class_name': row['ClassName'],
                'class_id': row['Label'],
                'split': 'train' if idx < len(train_df) else 
                        'validation' if idx < len(train_df) + len(val_df) else 'test'
            })
            
            processed_count += 1
            
            # Progress reporting
            if processed_count % 1000 == 0:
                print(f"Processed {processed_count}/{len(all_data)} samples...")
                
        except Exception as e:
            print(f"Unexpected error processing {sample_id}: {e}")
            error_count += 1
            continue
    
    print(f"Processing complete: {processed_count} successful, {error_count} errors")
    
    # Save labels
    labels_df = pd.DataFrame(labels_data)
    labels_df.to_csv(f"{processed_path}/labels.csv", index=False)
    
    # Save modality metadata
    for modality_name, metadata_list in modality_metadata.items():
        if metadata_list:
            modality_df = pd.DataFrame(metadata_list)
            modality_df.to_csv(f"{processed_path}/{modality_name}_metadata.csv", index=False)
    
    # Create comprehensive metadata
    metadata = {
        'dataset_name': 'EuroSAT/AllBands',
        'total_samples': len(labels_df),
        'classes': labels_df['class_name'].unique().tolist(),
        'modalities': ['visual_rgb', 'near_infrared', 'red_edge', 'short_wave_infrared', 'atmospheric'],
        'modality_info': {
            'visual_rgb': {
                'type': 'image',
                'format': 'JPEG',
                'dimensions': '64x64x3',
                'bands': ['B04', 'B03', 'B02'],
                'description': 'Visual RGB images for standard computer vision'
            },
            'near_infrared': {
                'type': 'spectral',
                'format': 'NPY',
                'dimensions': '64x64x2',
                'bands': ['B08', 'B8A'],
                'description': 'Near-Infrared bands for vegetation and water analysis'
            },
            'red_edge': {
                'type': 'spectral',
                'format': 'NPY',
                'dimensions': '64x64x3',
                'bands': ['B05', 'B06', 'B07'],
                'description': 'Red-Edge bands for vegetation health assessment'
            },
            'short_wave_infrared': {
                'type': 'spectral',
                'format': 'NPY',
                'dimensions': '64x64x2',
                'bands': ['B11', 'B12'],
                'description': 'SWIR bands for soil and mineral analysis'
            },
            'atmospheric': {
                'type': 'spectral',
                'format': 'NPY',
                'dimensions': '64x64x3',
                'bands': ['B01', 'B09', 'B10'],
                'description': 'Atmospheric bands for atmospheric correction'
            }
        },
        'computational_requirements': {
            'processing_time': '~8-12 hours on modern hardware (27,597 TIFF files with full spectral extraction)',
            'memory_usage': '~16GB RAM recommended (for processing large TIFF files)',
            'storage_requirements': '~15GB for processed data (5 modalities × 27,597 files)',
            'dependencies': ['rasterio', 'numpy', 'PIL', 'pandas'],
            'performance_notes': 'Processing time scales linearly with number of files. Consider parallel processing for faster execution.',
            'disk_io_intensive': 'High disk I/O due to reading 27,597 TIFF files and writing 138,985 output files (5 modalities × 27,597)'
        }
    }
    
    with open(f"{processed_path}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Enhanced preprocessing complete! Processed {len(labels_df)} samples.")
    print(f"Created 5 spectral modalities: {metadata['modalities']}")
    
    # Run comprehensive validation
    validation_results = validate_eurosat_preprocessing(processed_path)
    print(f"Validation Status: {validation_results['overall_status']}")
    if validation_results['errors']:
        print(f"Errors found: {validation_results['errors']}")
    if validation_results['warnings']:
        print(f"Warnings: {validation_results['warnings']}")
    
    return labels_df, modality_metadata

def validate_eurosat_preprocessing(processed_path):
    """
    Comprehensive validation of EuroSAT preprocessing results
    
    Args:
        processed_path (str): Path to processed data directory
        
    Returns:
        dict: Validation results with status, errors, and warnings
    """
    validation_results = {
        'overall_status': 'PASS',
        'errors': [],
        'warnings': [],
        'sample_counts': {},
        'quality_score': 0.0
    }
    
    try:
        # Check required files exist
        required_files = ['labels.csv', 'metadata.json']
        for file in required_files:
            if not os.path.exists(f"{processed_path}/{file}"):
                validation_results['errors'].append(f"Missing required file: {file}")
        
        # Check required directories exist
        required_dirs = ['visual_rgb', 'near_infrared', 'red_edge', 'short_wave_infrared', 'atmospheric']
        for dir_name in required_dirs:
            if not os.path.exists(f"{processed_path}/{dir_name}"):
                validation_results['errors'].append(f"Missing required directory: {dir_name}")
        
        if validation_results['errors']:
            validation_results['overall_status'] = 'FAIL'
            return validation_results
        
        # Load and validate labels
        labels_df = pd.read_csv(f"{processed_path}/labels.csv")
        validation_results['sample_counts']['labels'] = len(labels_df)
        
        # Check for missing values in labels
        if labels_df.isnull().any().any():
            validation_results['warnings'].append("Missing values found in labels.csv")
        
        # Check class distribution
        class_counts = labels_df['class_name'].value_counts()
        if len(class_counts) != 10:
            validation_results['errors'].append(f"Expected 10 classes, found {len(class_counts)}")
        
        # Check for severe class imbalance
        min_class_count = class_counts.min()
        max_class_count = class_counts.max()
        if max_class_count / min_class_count > 2.0:
            validation_results['warnings'].append(f"Severe class imbalance: {min_class_count} to {max_class_count}")
        
        # Validate modality files
        expected_sample_count = len(labels_df)
        for modality in required_dirs:
            modality_path = f"{processed_path}/{modality}"
            if modality == 'visual_rgb':
                # Count JPEG files
                jpeg_files = [f for f in os.listdir(modality_path) if f.endswith('.jpg')]
                validation_results['sample_counts'][modality] = len(jpeg_files)
            else:
                # Count NPY files
                npy_files = [f for f in os.listdir(modality_path) if f.endswith('.npy')]
                validation_results['sample_counts'][modality] = len(npy_files)
            
            # Check sample count consistency
            if validation_results['sample_counts'][modality] != expected_sample_count:
                validation_results['errors'].append(
                    f"Sample count mismatch in {modality}: expected {expected_sample_count}, found {validation_results['sample_counts'][modality]}"
                )
        
        # Validate file integrity (sample a few files)
        sample_files = labels_df['sample_id'].head(5).tolist()
        for sample_id in sample_files:
            # Check RGB image
            rgb_path = f"{processed_path}/visual_rgb/{sample_id}.jpg"
            if os.path.exists(rgb_path):
                try:
                    img = Image.open(rgb_path)
                    if img.size != (64, 64):
                        validation_results['warnings'].append(f"RGB image {sample_id} has wrong size: {img.size}")
                except Exception as e:
                    validation_results['errors'].append(f"Corrupted RGB image {sample_id}: {e}")
            
            # Check NPY files
            for modality in ['near_infrared', 'red_edge', 'short_wave_infrared', 'atmospheric']:
                npy_path = f"{processed_path}/{modality}/{sample_id}.npy"
                if os.path.exists(npy_path):
                    try:
                        data = np.load(npy_path, allow_pickle=True)
                        if not isinstance(data, dict):
                            validation_results['errors'].append(f"NPY file {sample_id} in {modality} is not a dictionary")
                        elif len(data) == 0:
                            validation_results['warnings'].append(f"Empty NPY file {sample_id} in {modality}")
                    except Exception as e:
                        validation_results['errors'].append(f"Corrupted NPY file {sample_id} in {modality}: {e}")
        
        # Calculate quality score
        total_checks = 10  # Approximate number of checks
        error_penalty = len(validation_results['errors']) * 2
        warning_penalty = len(validation_results['warnings']) * 0.5
        validation_results['quality_score'] = max(0, (total_checks - error_penalty - warning_penalty) / total_checks * 100)
        
        if validation_results['errors']:
            validation_results['overall_status'] = 'FAIL'
        elif validation_results['warnings']:
            validation_results['overall_status'] = 'WARN'
        
    except Exception as e:
        validation_results['errors'].append(f"Validation failed with exception: {e}")
        validation_results['overall_status'] = 'FAIL'
    
    return validation_results

# Run enhanced preprocessing
print("Processing EuroSAT/AllBands dataset with 5 spectral modalities...")
labels_df, modality_metadata = preprocess_eurosat_enhanced()
print(f"Processed {len(labels_df)} samples with 5 spectral modalities")
print("Created modalities: Visual RGB, Near-Infrared, Red-Edge, SWIR, Atmospheric")
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
    PyTorch Dataset for EuroSAT processed data with 5 spectral modalities
    """
    def __init__(self, data_path, split='train', transform=None, load_modalities=None):
        self.data_path = data_path
        self.labels_df = pd.read_csv(f"{data_path}/labels.csv")
        self.split_df = self.labels_df[self.labels_df['split'] == split]
        self.transform = transform
        
        # Define available modalities
        self.available_modalities = ['visual_rgb', 'near_infrared', 'red_edge', 'short_wave_infrared', 'atmospheric']
        
        # Load specified modalities (default: all)
        if load_modalities is None:
            self.load_modalities = self.available_modalities
        else:
            self.load_modalities = [m for m in load_modalities if m in self.available_modalities]
        
        print(f"Loading modalities: {self.load_modalities}")
    
    def __len__(self):
        return len(self.split_df)
    
    def __getitem__(self, idx):
        row = self.split_df.iloc[idx]
        sample_id = row['sample_id']
        
        data = {
            'label': row['class_id'],
            'class_name': row['class_name'],
            'sample_id': sample_id
        }
        
        # Load visual RGB modality
        if 'visual_rgb' in self.load_modalities:
            try:
                img_path = f"{self.data_path}/visual_rgb/{sample_id}.jpg"
                image = Image.open(img_path).convert('RGB')
                
                # Apply transforms
                if self.transform:
                    image = self.transform(image)
                
                data['visual_rgb'] = image
            except Exception as e:
                print(f"Error loading visual_rgb for {sample_id}: {e}")
                data['visual_rgb'] = None
        
        # Load spectral modalities (NPY files)
        for modality in ['near_infrared', 'red_edge', 'short_wave_infrared', 'atmospheric']:
            if modality in self.load_modalities:
                try:
                    npy_path = f"{self.data_path}/{modality}/{sample_id}.npy"
                    modality_data = np.load(npy_path, allow_pickle=True)
                    
                    # Convert to tensor format
                    if isinstance(modality_data, dict):
                        # Stack bands into tensor
                        bands = []
                        for band_name in sorted(modality_data.keys()):
                            bands.append(modality_data[band_name])
                        modality_tensor = torch.tensor(np.stack(bands, axis=0), dtype=torch.float32)
                    else:
                        modality_tensor = torch.tensor(modality_data, dtype=torch.float32)
                    
                    data[modality] = modality_tensor
                except Exception as e:
                    print(f"Error loading {modality} for {sample_id}: {e}")
                    data[modality] = None
        
        return data

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
├── visual_rgb/                   # Visual RGB images (B04, B03, B02)
│   ├── sample_00001.jpg
│   ├── sample_00002.jpg
│   └── ... (27,597 images)
├── near_infrared/                # Near-Infrared bands (B08, B8A)
│   ├── sample_00001.npy
│   ├── sample_00002.npy
│   └── ... (27,597 files)
├── red_edge/                     # Red-Edge bands (B05, B06, B07)
│   ├── sample_00001.npy
│   ├── sample_00002.npy
│   └── ... (27,597 files)
├── short_wave_infrared/          # SWIR bands (B11, B12)
│   ├── sample_00001.npy
│   ├── sample_00002.npy
│   └── ... (27,597 files)
├── atmospheric/                  # Atmospheric bands (B01, B09, B10)
│   ├── sample_00001.npy
│   ├── sample_00002.npy
│   └── ... (27,597 files)
└── metadata.json                 # Dataset statistics and modality information
```

### Expected Output Files

1. **labels.csv**: Standardized label file with sample IDs, class information, and split assignments
2. **visual_rgb/**: RGB images extracted from multispectral data (27,597 JPEG files)
3. **near_infrared/**: Near-Infrared bands (B08, B8A) as NPY files (27,597 files)
4. **red_edge/**: Red-Edge bands (B05, B06, B07) as NPY files (27,597 files)
5. **short_wave_infrared/**: SWIR bands (B11, B12) as NPY files (27,597 files)
6. **atmospheric/**: Atmospheric bands (B01, B09, B10) as NPY files (27,597 files)
7. **metadata.json**: Comprehensive dataset statistics and modality information

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
    
    # Create datasets with all 5 modalities
    train_dataset = EuroSATDataset(data_path, 'train', get_transforms('train'), 
                                 load_modalities=['visual_rgb', 'near_infrared', 'red_edge', 'short_wave_infrared', 'atmospheric'])
    val_dataset = EuroSATDataset(data_path, 'validation', get_transforms('validation'),
                               load_modalities=['visual_rgb', 'near_infrared', 'red_edge', 'short_wave_infrared', 'atmospheric'])
    test_dataset = EuroSATDataset(data_path, 'test', get_transforms('test'),
                                load_modalities=['visual_rgb', 'near_infrared', 'red_edge', 'short_wave_infrared', 'atmospheric'])
    
    # Example: Load only specific modalities for testing
    rgb_only_dataset = EuroSATDataset(data_path, 'train', get_transforms('train'), 
                                    load_modalities=['visual_rgb'])
    spectral_only_dataset = EuroSATDataset(data_path, 'train', get_transforms('train'),
                                         load_modalities=['near_infrared', 'red_edge', 'short_wave_infrared'])
    
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
_________________________________________________________________________

## MUTLA Dataset Preprocessing

### Complete File Directory Structure
```
Data/MUTLA/
├── Brainwave/                          # Physiological modality
│   ├── README.md                       # Brainwave data documentation
│   ├── schoolg/                        # School G brainwave sessions
│   │   ├── 20181130-10-37/            # Session folders (date-time format)
│   │   │   ├── attention_*.log         # Attention scores (0-100)
│   │   │   ├── EEG_*.log              # Raw EEG data (160 points/min)
│   │   │   └── events_*.log           # Device events and connectivity
│   │   └── ... (40+ session folders)
│   └── schooln/                        # School N brainwave sessions
│       ├── 20181130-16-06/            # Session folders (date-time format)
│       │   ├── attention_*.log         # Attention scores (0-100)
│       │   ├── EEG_*.log              # Raw EEG data (160 points/min)
│       │   └── events_*.log           # Device events and connectivity
│       └── ... (80+ session folders)
├── Synced/                             # Cross-modal synchronization
│   ├── schoolg.txt                     # School G synchronized data (4,469 samples)
│   └── schooln.txt                     # School N synchronized data (24,126 samples)
├── User records/                       # Behavioral modality
│   ├── math_record_cleaned.csv         # 8,272 math learning records
│   ├── en_record_cleaned.csv           # 14,977 English learning records
│   ├── phy_record_cleaned.csv          # 3,781 physics learning records
│   ├── chem_record_cleaned.csv         # 1,310 chemistry learning records
│   ├── cn_record_cleaned.csv           # 1,567 Chinese learning records
│   └── en_reading_record_cleaned.csv   # 95 English reading records
├── Webcam/                             # Visual modality
│   ├── schoolg_*_segment_*.json        # 1,170 facial landmark files
│   ├── schoolg_*_segment_*.npy         # 1,170 eye tracking files
│   ├── schooln_*_segment_*.json        # 1,026 facial landmark files
│   ├── schooln_*_segment_*.npy         # 1,026 eye tracking files
│   ├── README.md                       # Visual data documentation
│   └── feature_extraction_guide.md     # Feature extraction guide
├── LICENSE                             # Dataset license
└── README.md                           # Main dataset documentation
```

### Dataset Statistics
- **Total Behavioral Samples**: 30,002 question-level learning records
- **Total Synced Samples**: 28,595 samples with cross-modal alignment (4,469 + 24,126)
- **Total Brainwave Files**: 4,733 log files (attention, EEG, events)
- **Total Visual Files**: 4,342 files (2,196 JSON + 2,144 NPY)
- **Total File Count**: 9,089 files
- **File Size**: ~2GB total
- **Time Period**: November-December 2018 (2-month data collection)
- **Students**: 324 unique students across 2 learning centers
- **Subjects**: 6 academic subjects (Math, English, Physics, Chemistry, Chinese, English Reading)

### Cross-Modal Coverage Analysis
- **All 3 Modalities**: 738 samples (2.6% of synced data)
  - School G: 102 samples
  - School N: 636 samples
- **Behavioral + Physiological**: 2,243 samples (7.8% of synced data)
- **Behavioral + Visual**: 2,874 samples (10.1% of synced data)
- **Behavioral Only**: 22,738 samples (79.5% of synced data)
- **No Sensor Data**: 1,409 samples (not in sync files)

### Available Data Files

#### Behavioral Data (User Records)
- **math_record_cleaned.csv**: Math learning analytics (8,272 records)
- **en_record_cleaned.csv**: English learning analytics (14,977 records)
- **phy_record_cleaned.csv**: Physics learning analytics (3,781 records)
- **chem_record_cleaned.csv**: Chemistry learning analytics (1,310 records)
- **cn_record_cleaned.csv**: Chinese learning analytics (1,567 records)
- **en_reading_record_cleaned.csv**: English reading analytics (95 records)

#### Physiological Data (Brainwave)
- **Attention Logs**: Second-by-second attention scores (0-100)
- **EEG Logs**: Raw brainwave data with 160 data points per minute
- **Event Logs**: Device connectivity and state information
- **Session Organization**: Date-time folder structure for temporal alignment

#### Visual Data (Webcam)
- **Facial Landmarks**: 51-point facial landmark detection (JSON format)
- **Eye Tracking**: Pupil segmentation and iris landmarks (NPY format)
- **Head Pose**: 3D rotation angles and translation vectors
- **Face Detection**: Bounding box coordinates and confidence scores

#### Synchronization Data
- **schoolg.txt**: School G cross-modal alignment (4,468 samples)
- **schooln.txt**: School N cross-modal alignment (24,125 samples)
- **Columns**: index, subject, question_id, user_id, school_id, stime, ctime, video_id, brainwave_file_path_attention, brainwave_file_path_EEG, brainwave_file_path_events

### Modalities Present in Raw Data

#### 1. Behavioral Modality (30,002 samples - 100%)
- **Learning Analytics**: Question-level student responses, correctness, response times
- **Academic Structure**: Course, section, topic, module, and knowledge point identifiers
- **Learning Behavior**: Hint usage, answer viewing, analysis viewing patterns
- **Performance Metrics**: Difficulty ratings, mastery tracking, proficiency estimates
- **Subject Coverage**: Mathematics, English, Physics, Chemistry, Chinese, English Reading

#### 2. Physiological Modality (2,981 samples - 9.9%)
- **Attention Values**: Derived attention scores (0-100) from BrainCo headset
- **Raw EEG Data**: Electrical potential differences with 160 data points per minute
- **Frequency Bands**: Alpha (8-12Hz), LowBeta (12-22Hz), HighBeta (22-32Hz), Gamma (32-56Hz)
- **Device Events**: Connection status and device state information
- **Temporal Resolution**: Second-by-second brainwave measurements

#### 3. Visual Modality (3,612 samples - 12.0%)
- **Facial Landmarks**: 51-point facial landmark detection and tracking
- **Eye Tracking**: Pupil segmentation, iris landmarks (9 points per eye)
- **Head Pose**: 3D rotation angles and translation vectors
- **Face Detection**: Bounding box coordinates and confidence scores
- **Gaze Analysis**: Eye movement patterns and attention direction

#### 4. Temporal Synchronization
- **Cross-Modal Alignment**: Timestamp-based synchronization across all modalities
- **Question-Level Mapping**: Tabular data linked to time-series and structured array data
- **Session Tracking**: Multi-session learning progression over time

### Required Preprocessing Steps

#### Step 1: Data Integration and Cross-Modal Alignment
- Load synchronized data from schoolg.txt and schooln.txt
- Merge tabular data from User records with sync information
- Create unified sample IDs for cross-modal tracking
- Handle missing modality data (empty file paths in sync files)

#### Step 2: Tabular Features Extraction

**Objective**: Extract comprehensive tabular features from user records

**Implementation**:
```python
def extract_tabular_features(user_records_path, synced_data):
    """
    Extract tabular features from MUTLA user records
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    # Subject file mapping
    subject_files = {
        'math_record_cleaned.csv': 'math',
        'en_record_cleaned.csv': 'english', 
        'phy_record_cleaned.csv': 'physics',
        'chem_record_cleaned.csv': 'chemistry',
        'cn_record_cleaned.csv': 'chinese',
        'en_reading_record_cleaned.csv': 'english_reading'
    }
    
    all_tabular_data = []
    
    for file, subject in subject_files.items():
        file_path = user_records_path / file
        
        if file_path.exists():
            try:
                # Load with error handling for encoding
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='gbk')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='latin-1')
            
            # Standardize column names
            column_mapping = {
                'difficulty': 'difficulty_rating',
                'is_view_answer': 'answer_viewed',
                'is_view_analyze': 'analysis_viewed', 
                'is_right': 'is_correct',
                'cost_time': 'response_time',
                'user_answer': 'user_response',
                'right_answer': 'correct_answer',
                'question_id': 'question_id',
                'user_id': 'user_id',
                'stime': 'start_time',
                'ctime': 'completion_time'
            }
            
            df = df.rename(columns=column_mapping)
            df['subject'] = subject
            
            # Extract tabular features
            tabular_features = []
            for idx, row in df.iterrows():
                features = {
                    'sample_id': f"{subject}_{row['question_id']}_{row['user_id']}",
                    'subject': subject,
                    'question_id': row['question_id'],
                    'user_id': row['user_id'],
                    
                    # Learning analytics features
                    'is_correct': int(row.get('is_correct', 0)),
                    'response_time': float(row.get('response_time', 0)),
                    'difficulty_rating': float(row.get('difficulty_rating', 0)),
                    
                    # Learning behavior features
                    'answer_viewed': int(row.get('answer_viewed', 0)),
                    'analysis_viewed': int(row.get('analysis_viewed', 0)),
                    'hint_usage': int(row.get('hint_usage', 0)),
                    
                    # Performance metrics
                    'mastery_level': float(row.get('mastery_level', 0)),
                    'proficiency_score': float(row.get('proficiency_score', 0)),
                    'progress_indicator': float(row.get('progress_indicator', 0)),
                    
                    # Temporal features
                    'start_time': row.get('start_time', ''),
                    'completion_time': row.get('completion_time', ''),
                    'session_duration': calculate_session_duration(row.get('start_time', ''), row.get('completion_time', ''))
                }
                
                tabular_features.append(features)
            
            all_tabular_data.extend(tabular_features)
            print(f"Extracted {len(tabular_features)} tabular features from {subject}")
    
    return pd.DataFrame(all_tabular_data)

def calculate_session_duration(start_time, completion_time):
    """Calculate session duration in seconds"""
    try:
        if pd.isna(start_time) or pd.isna(completion_time):
            return 0
        
        start = pd.to_datetime(start_time)
        end = pd.to_datetime(completion_time)
        duration = (end - start).total_seconds()
        return max(0, duration)
    except:
        return 0
```

#### Step 3: Time-series Features Extraction

**Objective**: Extract time-series features from brainwave log files

**Implementation**:
```python
def extract_timeseries_features(brainwave_path, synced_data):
    """
    Extract time-series features from MUTLA brainwave data
    """
    import pandas as pd
    import numpy as np
    import re
    from pathlib import Path
    from scipy import signal
    from scipy.fft import fft, fftfreq
    
    timeseries_features = []
    
    for idx, row in synced_data.iterrows():
        sample_id = row['sample_id']
        
        # Find brainwave files for this sample
        attention_file = find_brainwave_file(brainwave_path, row, 'attention')
        eeg_file = find_brainwave_file(brainwave_path, row, 'EEG')
        events_file = find_brainwave_file(brainwave_path, row, 'events')
        
        features = {
            'sample_id': sample_id,
            'question_id': row['question_id'],
            'user_id': row['user_id'],
            'subject': row['subject'],
            'school': row['school']
        }
        
        # Extract attention features
        if attention_file and attention_file.exists():
            attention_data = parse_attention_log(attention_file)
            features.update(attention_data)
        
        # Extract EEG features
        if eeg_file and eeg_file.exists():
            eeg_data = parse_eeg_log(eeg_file)
            features.update(eeg_data)
        
        # Extract event features
        if events_file and events_file.exists():
            event_data = parse_events_log(events_file)
            features.update(event_data)
        
        timeseries_features.append(features)
    
    return pd.DataFrame(timeseries_features)

def find_brainwave_file(brainwave_path, row, file_type):
    """Find brainwave file for given sample and type"""
    school = row['school']
    user_id = row['user_id']
    
    # Look in school-specific directory
    school_path = brainwave_path / school
    
    if school_path.exists():
        # Search for files matching pattern
        for session_dir in school_path.iterdir():
            if session_dir.is_dir():
                for file_path in session_dir.glob(f'{file_type}_*.log'):
                    if str(user_id) in file_path.name:
                        return file_path
    
    return None

def parse_attention_log(file_path):
    """Parse attention log file and extract features"""
    features = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract attention values (typically 0-100)
        attention_values = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Look for numerical values that could be attention scores
            numbers = re.findall(r'(\d+(?:\.\d+)?)', line)
            for num_str in numbers:
                try:
                    value = float(num_str)
                    if 0 <= value <= 100:  # Reasonable attention range
                        attention_values.append(value)
                except ValueError:
                    continue
        
        if attention_values:
            features.update({
                'attention_mean': np.mean(attention_values),
                'attention_std': np.std(attention_values),
                'attention_min': np.min(attention_values),
                'attention_max': np.max(attention_values),
                'attention_count': len(attention_values),
                'attention_trend': calculate_trend(attention_values)
            })
        else:
            # Default values if no attention data found
            features.update({
                'attention_mean': 50.0,
                'attention_std': 0.0,
                'attention_min': 50.0,
                'attention_max': 50.0,
                'attention_count': 0,
                'attention_trend': 0.0
            })
    
    except Exception as e:
        print(f"Error parsing attention log {file_path}: {e}")
        features.update({
            'attention_mean': 50.0,
            'attention_std': 0.0,
            'attention_min': 50.0,
            'attention_max': 50.0,
            'attention_count': 0,
            'attention_trend': 0.0
        })
    
    return features

def parse_eeg_log(file_path):
    """Parse EEG log file and extract frequency band features"""
    features = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract EEG values
        eeg_values = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Look for numerical values that could be EEG data
            numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', line)
            for num_str in numbers:
                try:
                    value = float(num_str)
                    if -10000 <= value <= 10000:  # Reasonable EEG range
                        eeg_values.append(value)
                except ValueError:
                    continue
        
        if len(eeg_values) >= 64:  # Need sufficient data for frequency analysis
            # Basic statistics
            features.update({
                'eeg_mean': np.mean(eeg_values),
                'eeg_std': np.std(eeg_values),
                'eeg_min': np.min(eeg_values),
                'eeg_max': np.max(eeg_values),
                'eeg_count': len(eeg_values)
            })
            
            # Frequency band analysis
            eeg_array = np.array(eeg_values)
            sampling_rate = 160.0  # Typical EEG sampling rate
            
            # Apply FFT for frequency analysis
            fft_result = np.fft.fft(eeg_array)
            freqs = np.fft.fftfreq(len(eeg_array), d=1/sampling_rate)
            
            # Define frequency bands
            delta_mask = (freqs >= 0.5) & (freqs <= 4.0)
            theta_mask = (freqs >= 4.0) & (freqs <= 8.0)
            alpha_mask = (freqs >= 8.0) & (freqs <= 12.0)
            beta_mask = (freqs >= 12.0) & (freqs <= 30.0)
            gamma_mask = (freqs >= 30.0) & (freqs <= 80.0)
            
            # Calculate band powers
            total_power = np.sum(np.abs(fft_result) ** 2)
            
            for band_name, mask in [('delta', delta_mask), ('theta', theta_mask), 
                                  ('alpha', alpha_mask), ('beta', beta_mask), ('gamma', gamma_mask)]:
                if np.any(mask):
                    band_power = np.sum(np.abs(fft_result[mask]) ** 2)
                    features[f'{band_name}_power'] = band_power
                    features[f'{band_name}_relative'] = band_power / (total_power + 1e-10)
                else:
                    features[f'{band_name}_power'] = 0.0
                    features[f'{band_name}_relative'] = 0.0
        
        else:
            # Default values for insufficient data
            features.update({
                'eeg_mean': 0.0,
                'eeg_std': 0.0,
                'eeg_min': 0.0,
                'eeg_max': 0.0,
                'eeg_count': len(eeg_values),
                'delta_power': 0.0, 'delta_relative': 0.0,
                'theta_power': 0.0, 'theta_relative': 0.0,
                'alpha_power': 0.0, 'alpha_relative': 0.0,
                'beta_power': 0.0, 'beta_relative': 0.0,
                'gamma_power': 0.0, 'gamma_relative': 0.0
            })
    
    except Exception as e:
        print(f"Error parsing EEG log {file_path}: {e}")
        features.update({
            'eeg_mean': 0.0, 'eeg_std': 0.0, 'eeg_min': 0.0, 'eeg_max': 0.0, 'eeg_count': 0,
            'delta_power': 0.0, 'delta_relative': 0.0,
            'theta_power': 0.0, 'theta_relative': 0.0,
            'alpha_power': 0.0, 'alpha_relative': 0.0,
            'beta_power': 0.0, 'beta_relative': 0.0,
            'gamma_power': 0.0, 'gamma_relative': 0.0
        })
    
    return features

def parse_events_log(file_path):
    """Parse events log file and extract device connectivity features"""
    features = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Count different types of events
        connect_events = content.count('connect') + content.count('CONNECT')
        disconnect_events = content.count('disconnect') + content.count('DISCONNECT')
        error_events = content.count('error') + content.count('ERROR')
        
        features.update({
            'connect_events': connect_events,
            'disconnect_events': disconnect_events,
            'error_events': error_events,
            'total_events': connect_events + disconnect_events + error_events,
            'connection_stability': 1.0 / (disconnect_events + 1)
        })
    
    except Exception as e:
        print(f"Error parsing events log {file_path}: {e}")
        features.update({
            'connect_events': 0,
            'disconnect_events': 0,
            'error_events': 0,
            'total_events': 0,
            'connection_stability': 1.0
        })
    
    return features

def calculate_trend(values):
    """Calculate trend in attention values"""
    if len(values) < 2:
        return 0.0
    
    x = np.arange(len(values))
    y = np.array(values)
    
    # Simple linear regression for trend
    slope = np.polyfit(x, y, 1)[0]
    return slope
```

#### Step 4: Visual Modality Features Extraction

**Objective**: Extract visual modality features from structured arrays (facial landmarks and eye tracking data)

**Implementation**:
```python
def extract_visual_features(webcam_path, synced_data):
    """
    Extract visual modality features from MUTLA structured arrays
    """
    import pandas as pd
    import numpy as np
    import json
    from pathlib import Path
    
    visual_features = []
    
    for idx, row in synced_data.iterrows():
        sample_id = row['sample_id']
        video_id = row.get('video_id', '')
        
        # Find visual feature files for this sample
        landmark_file = find_visual_file(webcam_path, row, 'landmarks')
        eye_tracking_file = find_visual_file(webcam_path, row, 'eye_tracking')
        
        features = {
            'sample_id': sample_id,
            'question_id': row['question_id'],
            'user_id': row['user_id'],
            'subject': row['subject'],
            'school': row['school'],
            'video_id': video_id
        }
        
        # Extract facial landmark features
        if landmark_file and landmark_file.exists():
            landmark_data = parse_landmark_file(landmark_file)
            features.update(landmark_data)
        
        # Extract eye tracking features
        if eye_tracking_file and eye_tracking_file.exists():
            eye_data = parse_eye_tracking_file(eye_tracking_file)
            features.update(eye_data)
        
        visual_features.append(features)
    
    return pd.DataFrame(visual_features)

def find_visual_file(webcam_path, row, file_type):
    """Find visual feature file for given sample and type"""
    school = row['school']
    video_id = row.get('video_id', '')
    
    if file_type == 'landmarks':
        # Look for JSON files with facial landmarks
        pattern = f"{school}_*_segment_*.json"
        for file_path in webcam_path.glob(pattern):
            if video_id in file_path.name or str(row['user_id']) in file_path.name:
                return file_path
    
    elif file_type == 'eye_tracking':
        # Look for NPY files with eye tracking data
        pattern = f"{school}_*_segment_*.npy"
        for file_path in webcam_path.glob(pattern):
            if video_id in file_path.name or str(row['user_id']) in file_path.name:
                return file_path
    
    return None

def parse_landmark_file(file_path):
    """Parse facial landmark JSON file and extract features"""
    features = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract 51-point facial landmarks
        if 'landmarks' in data and len(data['landmarks']) >= 51:
            landmarks = np.array(data['landmarks'][:51])  # Take first 51 points
            
            # Calculate facial geometry features
            features.update({
                'landmark_count': len(landmarks),
                'face_center_x': np.mean(landmarks[:, 0]),
                'face_center_y': np.mean(landmarks[:, 1]),
                'face_width': np.max(landmarks[:, 0]) - np.min(landmarks[:, 0]),
                'face_height': np.max(landmarks[:, 1]) - np.min(landmarks[:, 1]),
                'face_aspect_ratio': (np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])) / 
                                   (np.max(landmarks[:, 1]) - np.min(landmarks[:, 1]) + 1e-6)
            })
            
            # Extract eye region features (points 36-47 for eyes)
            if len(landmarks) >= 48:
                left_eye = landmarks[36:42]  # Left eye points
                right_eye = landmarks[42:48]  # Right eye points
                
                features.update({
                    'left_eye_center_x': np.mean(left_eye[:, 0]),
                    'left_eye_center_y': np.mean(left_eye[:, 1]),
                    'right_eye_center_x': np.mean(right_eye[:, 0]),
                    'right_eye_center_y': np.mean(right_eye[:, 1]),
                    'eye_distance': np.sqrt((features['right_eye_center_x'] - features['left_eye_center_x'])**2 + 
                                          (features['right_eye_center_y'] - features['left_eye_center_y'])**2)
                })
            
            # Extract mouth region features (points 48-68 for mouth)
            if len(landmarks) >= 68:
                mouth = landmarks[48:68]
                features.update({
                    'mouth_center_x': np.mean(mouth[:, 0]),
                    'mouth_center_y': np.mean(mouth[:, 1]),
                    'mouth_width': np.max(mouth[:, 0]) - np.min(mouth[:, 0]),
                    'mouth_height': np.max(mouth[:, 1]) - np.min(mouth[:, 1])
                })
            
            # Calculate facial symmetry
            features['facial_symmetry'] = calculate_facial_symmetry(landmarks)
            
            # Calculate head pose indicators
            features.update(calculate_head_pose(landmarks))
        
        else:
            # Default values if no landmark data
            features.update({
                'landmark_count': 0,
                'face_center_x': 0.0, 'face_center_y': 0.0,
                'face_width': 0.0, 'face_height': 0.0, 'face_aspect_ratio': 1.0,
                'left_eye_center_x': 0.0, 'left_eye_center_y': 0.0,
                'right_eye_center_x': 0.0, 'right_eye_center_y': 0.0,
                'eye_distance': 0.0,
                'mouth_center_x': 0.0, 'mouth_center_y': 0.0,
                'mouth_width': 0.0, 'mouth_height': 0.0,
                'facial_symmetry': 0.0,
                'head_tilt': 0.0, 'head_pan': 0.0, 'head_roll': 0.0
            })
    
    except Exception as e:
        print(f"Error parsing landmark file {file_path}: {e}")
        features.update({
            'landmark_count': 0,
            'face_center_x': 0.0, 'face_center_y': 0.0,
            'face_width': 0.0, 'face_height': 0.0, 'face_aspect_ratio': 1.0,
            'left_eye_center_x': 0.0, 'left_eye_center_y': 0.0,
            'right_eye_center_x': 0.0, 'right_eye_center_y': 0.0,
            'eye_distance': 0.0,
            'mouth_center_x': 0.0, 'mouth_center_y': 0.0,
            'mouth_width': 0.0, 'mouth_height': 0.0,
            'facial_symmetry': 0.0,
            'head_tilt': 0.0, 'head_pan': 0.0, 'head_roll': 0.0
        })
    
    return features

def parse_eye_tracking_file(file_path):
    """Parse eye tracking NPY file and extract features"""
    features = {}
    
    try:
        eye_data = np.load(file_path, allow_pickle=True)
        
        if isinstance(eye_data, np.ndarray) and len(eye_data) > 0:
            # Extract eye tracking features
            features.update({
                'gaze_x_mean': np.mean(eye_data[:, 0]) if eye_data.shape[1] > 0 else 0.0,
                'gaze_y_mean': np.mean(eye_data[:, 1]) if eye_data.shape[1] > 1 else 0.0,
                'gaze_x_std': np.std(eye_data[:, 0]) if eye_data.shape[1] > 0 else 0.0,
                'gaze_y_std': np.std(eye_data[:, 1]) if eye_data.shape[1] > 1 else 0.0,
                'gaze_velocity': calculate_gaze_velocity(eye_data),
                'fixation_count': count_fixations(eye_data),
                'saccade_count': count_saccades(eye_data),
                'pupil_diameter_mean': np.mean(eye_data[:, 2]) if eye_data.shape[1] > 2 else 0.0,
                'pupil_diameter_std': np.std(eye_data[:, 2]) if eye_data.shape[1] > 2 else 0.0
            })
        else:
            # Default values if no eye tracking data
            features.update({
                'gaze_x_mean': 0.0, 'gaze_y_mean': 0.0,
                'gaze_x_std': 0.0, 'gaze_y_std': 0.0,
                'gaze_velocity': 0.0,
                'fixation_count': 0, 'saccade_count': 0,
                'pupil_diameter_mean': 0.0, 'pupil_diameter_std': 0.0
            })
    
    except Exception as e:
        print(f"Error parsing eye tracking file {file_path}: {e}")
        features.update({
            'gaze_x_mean': 0.0, 'gaze_y_mean': 0.0,
            'gaze_x_std': 0.0, 'gaze_y_std': 0.0,
            'gaze_velocity': 0.0,
            'fixation_count': 0, 'saccade_count': 0,
            'pupil_diameter_mean': 0.0, 'pupil_diameter_std': 0.0
        })
    
    return features

def calculate_facial_symmetry(landmarks):
    """Calculate facial symmetry score"""
    if len(landmarks) < 17:  # Need at least face outline points
        return 0.0
    
    # Use face outline points (0-16) for symmetry calculation
    face_outline = landmarks[:17]
    face_center_x = np.mean(face_outline[:, 0])
    
    # Calculate symmetry by comparing left and right sides
    left_points = face_outline[face_outline[:, 0] < face_center_x]
    right_points = face_outline[face_outline[:, 0] > face_center_x]
    
    if len(left_points) == 0 or len(right_points) == 0:
        return 0.0
    
    # Simple symmetry measure
    left_mean_y = np.mean(left_points[:, 1])
    right_mean_y = np.mean(right_points[:, 1])
    
    symmetry = 1.0 - abs(left_mean_y - right_mean_y) / (np.max(face_outline[:, 1]) - np.min(face_outline[:, 1]) + 1e-6)
    return max(0.0, symmetry)

def calculate_head_pose(landmarks):
    """Calculate head pose angles"""
    if len(landmarks) < 17:
        return {'head_tilt': 0.0, 'head_pan': 0.0, 'head_roll': 0.0}
    
    # Use nose tip and eye centers for head pose estimation
    nose_tip = landmarks[30] if len(landmarks) > 30 else landmarks[0]
    left_eye = landmarks[36] if len(landmarks) > 36 else landmarks[1]
    right_eye = landmarks[45] if len(landmarks) > 45 else landmarks[2]
    
    # Calculate head tilt (rotation around z-axis)
    eye_vector = right_eye - left_eye
    head_tilt = np.arctan2(eye_vector[1], eye_vector[0])
    
    # Calculate head pan (rotation around y-axis) - simplified
    head_pan = 0.0  # Would need 3D landmarks for accurate calculation
    
    # Calculate head roll (rotation around x-axis) - simplified
    head_roll = 0.0  # Would need 3D landmarks for accurate calculation
    
    return {
        'head_tilt': head_tilt,
        'head_pan': head_pan,
        'head_roll': head_roll
    }

def calculate_gaze_velocity(eye_data):
    """Calculate average gaze velocity"""
    if len(eye_data) < 2:
        return 0.0
    
    velocities = []
    for i in range(1, len(eye_data)):
        dx = eye_data[i, 0] - eye_data[i-1, 0]
        dy = eye_data[i, 1] - eye_data[i-1, 1]
        velocity = np.sqrt(dx**2 + dy**2)
        velocities.append(velocity)
    
    return np.mean(velocities) if velocities else 0.0

def count_fixations(eye_data, threshold=0.5):
    """Count fixations in eye tracking data"""
    if len(eye_data) < 3:
        return 0
    
    fixations = 0
    current_fixation_length = 0
    
    for i in range(1, len(eye_data)):
        dx = eye_data[i, 0] - eye_data[i-1, 0]
        dy = eye_data[i, 1] - eye_data[i-1, 1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < threshold:
            current_fixation_length += 1
        else:
            if current_fixation_length >= 3:  # Minimum fixation duration
                fixations += 1
            current_fixation_length = 0
    
    return fixations

def count_saccades(eye_data, threshold=2.0):
    """Count saccades in eye tracking data"""
    if len(eye_data) < 2:
        return 0
    
    saccades = 0
    for i in range(1, len(eye_data)):
        dx = eye_data[i, 0] - eye_data[i-1, 0]
        dy = eye_data[i, 1] - eye_data[i-1, 1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance > threshold:
            saccades += 1
    
    return saccades
```

#### Step 5: File Format Specifications

**MUTLA NPY File Formats**:

**Behavioral Features CSV**:
```csv
sample_id,subject,question_id,user_id,is_correct,response_time,difficulty_rating,answer_viewed,analysis_viewed,hint_usage,mastery_level,proficiency_score,progress_indicator,start_time,completion_time,session_duration
math_q123_u456,math,123,456,1,45.2,3,0,1,0,0.8,0.75,0.6,2018-12-01 10:00:00,2018-12-01 10:00:45,45.0
```

**Physiological Features CSV**:
```csv
sample_id,question_id,user_id,subject,school,attention_mean,attention_std,attention_min,attention_max,attention_count,attention_trend,eeg_mean,eeg_std,eeg_min,eeg_max,eeg_count,delta_power,delta_relative,theta_power,theta_relative,alpha_power,alpha_relative,beta_power,beta_relative,gamma_power,gamma_relative,connect_events,disconnect_events,error_events,total_events,connection_stability
math_q123_u456,123,456,math,schoolg,75.5,12.3,45.0,95.0,120,0.2,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2,0,0,2,1.0
```

**Visual Features CSV**:
```csv
sample_id,question_id,user_id,subject,school,video_id,landmark_count,face_center_x,face_center_y,face_width,face_height,face_aspect_ratio,left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y,eye_distance,mouth_center_x,mouth_center_y,mouth_width,mouth_height,facial_symmetry,head_tilt,head_pan,head_roll,gaze_x_mean,gaze_y_mean,gaze_x_std,gaze_y_std,gaze_velocity,fixation_count,saccade_count,pupil_diameter_mean,pupil_diameter_std
math_q123_u456,123,456,math,schoolg,video_001,51,320.5,240.2,150.3,180.7,0.83,280.1,220.5,360.9,220.3,80.8,320.2,280.1,45.2,25.8,0.92,0.05,0.0,0.0,320.1,240.0,15.2,12.8,2.5,8,12,4.2,0.3
```

#### Step 6: Comprehensive Validation Functions

**MUTLA Validation Function**:
```python
def validate_mutla_preprocessing(processed_path):
    """
    Comprehensive validation of MUTLA preprocessing results
    """
    import os
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    validation_results = {
        'overall_status': 'PASS',
        'errors': [],
        'warnings': [],
        'sample_counts': {},
        'quality_metrics': {},
        'corruption_rates': {},
        'missing_data_percentages': {}
    }
    
    try:
        # Check required files exist
        required_files = ['labels.csv', 'metadata.json']
        for file in required_files:
            if not os.path.exists(f"{processed_path}/{file}"):
                validation_results['errors'].append(f"Missing required file: {file}")
        
        # Check required directories exist
        required_dirs = ['tabular', 'timeseries', 'visual']
        for dir_name in required_dirs:
            if not os.path.exists(f"{processed_path}/{dir_name}"):
                validation_results['errors'].append(f"Missing required directory: {dir_name}")
        
        if validation_results['errors']:
            validation_results['overall_status'] = 'FAIL'
            return validation_results
        
        # Load and validate labels
        labels_df = pd.read_csv(f"{processed_path}/labels.csv")
        validation_results['sample_counts']['labels'] = len(labels_df)
        
        # Check for missing values in labels
        missing_labels = labels_df.isnull().sum().sum()
        if missing_labels > 0:
            validation_results['warnings'].append(f"Missing values found in labels.csv: {missing_labels}")
        
        # Validate modality files
        expected_sample_count = len(labels_df)
        for modality in required_dirs:
            modality_path = f"{processed_path}/{modality}"
            if os.path.exists(modality_path):
                # Count CSV files
                csv_files = [f for f in os.listdir(modality_path) if f.endswith('.csv')]
                validation_results['sample_counts'][modality] = len(csv_files)
                
                # Check sample count consistency
                if len(csv_files) != expected_sample_count:
                    validation_results['errors'].append(
                        f"Sample count mismatch in {modality}: expected {expected_sample_count}, found {len(csv_files)}"
                    )
                
                # Validate file integrity (sample a few files)
                if csv_files:
                    sample_file = csv_files[0]
                    try:
                        df = pd.read_csv(f"{modality_path}/{sample_file}")
                        validation_results['quality_metrics'][f'{modality}_columns'] = len(df.columns)
                        validation_results['quality_metrics'][f'{modality}_rows'] = len(df)
                        
                        # Check for missing data
                        missing_data = df.isnull().sum().sum()
                        total_cells = len(df) * len(df.columns)
                        missing_percentage = (missing_data / total_cells) * 100
                        validation_results['missing_data_percentages'][modality] = missing_percentage
                        
                        if missing_percentage > 10:  # More than 10% missing data
                            validation_results['warnings'].append(f"High missing data in {modality}: {missing_percentage:.2f}%")
                        
                    except Exception as e:
                        validation_results['errors'].append(f"Error reading {modality} file {sample_file}: {e}")
        
        # Validate cross-modal alignment
        if 'tabular' in validation_results['sample_counts'] and 'timeseries' in validation_results['sample_counts']:
            tabular_count = validation_results['sample_counts']['tabular']
            timeseries_count = validation_results['sample_counts']['timeseries']
            visual_count = validation_results['sample_counts'].get('visual', 0)
            
            # Check for expected cross-modal coverage
            if tabular_count != timeseries_count:
                validation_results['warnings'].append(f"Cross-modal alignment issue: tabular={tabular_count}, timeseries={timeseries_count}")
            
            # Calculate corruption rates
            total_expected = expected_sample_count
            validation_results['corruption_rates'] = {
                'tabular': (total_expected - tabular_count) / total_expected * 100,
                'timeseries': (total_expected - timeseries_count) / total_expected * 100,
                'visual': (total_expected - visual_count) / total_expected * 100
            }
        
        # Calculate overall quality score
        total_checks = 15  # Approximate number of checks
        error_penalty = len(validation_results['errors']) * 3
        warning_penalty = len(validation_results['warnings']) * 1
        validation_results['quality_score'] = max(0, (total_checks - error_penalty - warning_penalty) / total_checks * 100)
        
        if validation_results['errors']:
            validation_results['overall_status'] = 'FAIL'
        elif validation_results['warnings']:
            validation_results['overall_status'] = 'WARN'
        
    except Exception as e:
        validation_results['errors'].append(f"Validation failed with exception: {e}")
        validation_results['overall_status'] = 'FAIL'
    
    return validation_results
```

#### Step 7: Simplified Dataset Creation Strategy
**RECOMMENDED APPROACH: Start with Core Multimodal Dataset**

**Primary Dataset: Perfect Multimodal (738 samples)**
- Only samples with all 3 modalities present (behavioral + physiological + visual)
- Ideal for initial multimodal fusion learning experiments
- Clean, complete data for testing optimal multimodal performance
- Recommended starting point for researchers

**Optional Extended Datasets (for advanced users):**
**Dataset 2: Complete Mixed Modality (28,595 samples)**
- All synced samples regardless of modality availability
- Real-world scenario with missing modalities

#### Step 8: Performance Optimization for MUTLA Preprocessing

**Parallel Processing Implementation**:
```python
def preprocess_mutla_parallel(raw_data_path, output_path, num_workers=4):
    """
    Parallel MUTLA preprocessing for thousands of log files
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import pandas as pd
    from pathlib import Path
    
    # Load synchronized data
    synced_data = load_synchronized_data(raw_data_path)
    
    # Split data into chunks for parallel processing
    chunk_size = len(synced_data) // num_workers
    data_chunks = [synced_data[i:i + chunk_size] for i in range(0, len(synced_data), chunk_size)]
    
    print(f"Processing {len(synced_data)} samples using {num_workers} workers")
    print(f"Chunk size: {chunk_size} samples per worker")
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks
        future_to_chunk = {
            executor.submit(process_data_chunk, chunk, raw_data_path, output_path, i): i 
            for i, chunk in enumerate(data_chunks)
        }
        
        # Collect results
        results = []
        for future in as_completed(future_to_chunk):
            chunk_id = future_to_chunk[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Completed chunk {chunk_id}: {result['processed_samples']} samples")
            except Exception as e:
                print(f"Chunk {chunk_id} failed: {e}")
    
    # Combine results
    combine_parallel_results(results, output_path)
    print("Parallel preprocessing completed!")

def process_data_chunk(data_chunk, raw_data_path, output_path, chunk_id):
    """
    Process a single chunk of data (runs in separate process)
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    chunk_output_path = Path(output_path) / f"chunk_{chunk_id}"
    chunk_output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract features for this chunk
    behavioral_features = extract_behavioral_features_chunk(data_chunk, raw_data_path)
    physiological_features = extract_physiological_features_chunk(data_chunk, raw_data_path)
    visual_features = extract_visual_features_chunk(data_chunk, raw_data_path)
    
    # Save chunk results
    behavioral_features.to_csv(chunk_output_path / "behavioral_features.csv", index=False)
    physiological_features.to_csv(chunk_output_path / "physiological_features.csv", index=False)
    visual_features.to_csv(chunk_output_path / "visual_features.csv", index=False)
    
    return {
        'chunk_id': chunk_id,
        'processed_samples': len(data_chunk),
        'behavioral_samples': len(behavioral_features),
        'physiological_samples': len(physiological_features),
        'visual_samples': len(visual_features),
        'output_path': str(chunk_output_path)
    }

def combine_parallel_results(results, output_path):
    """
    Combine results from parallel processing
    """
    import pandas as pd
    from pathlib import Path
    
    # Combine all feature files
    all_behavioral = []
    all_physiological = []
    all_visual = []
    
    for result in results:
        chunk_path = Path(result['output_path'])
        
        # Load and combine behavioral features
        if (chunk_path / "behavioral_features.csv").exists():
            behavioral_df = pd.read_csv(chunk_path / "behavioral_features.csv")
            all_behavioral.append(behavioral_df)
        
        # Load and combine physiological features
        if (chunk_path / "physiological_features.csv").exists():
            physiological_df = pd.read_csv(chunk_path / "physiological_features.csv")
            all_physiological.append(physiological_df)
        
        # Load and combine visual features
        if (chunk_path / "visual_features.csv").exists():
            visual_df = pd.read_csv(chunk_path / "visual_features.csv")
            all_visual.append(visual_df)
    
    # Save combined results
    if all_behavioral:
        pd.concat(all_behavioral, ignore_index=True).to_csv(f"{output_path}/behavioral_features.csv", index=False)
    if all_physiological:
        pd.concat(all_physiological, ignore_index=True).to_csv(f"{output_path}/physiological_features.csv", index=False)
    if all_visual:
        pd.concat(all_visual, ignore_index=True).to_csv(f"{output_path}/visual_features.csv", index=False)
    
    # Clean up chunk directories
    for result in results:
        chunk_path = Path(result['output_path'])
        if chunk_path.exists():
            import shutil
            shutil.rmtree(chunk_path)

# Performance optimization recommendations
PERFORMANCE_OPTIMIZATION_GUIDE = {
    'parallel_processing': {
        'recommended_workers': 'min(8, multiprocessing.cpu_count())',
        'memory_per_worker': '~2-4GB RAM',
        'disk_io_optimization': 'Use SSD storage for temporary files',
        'chunk_size': '1000-2000 samples per chunk (adjust based on available RAM)'
    },
    'file_io_optimization': {
        'batch_size': 'Process 100-200 files at a time',
        'caching': 'Cache frequently accessed files in memory',
        'compression': 'Use compressed formats (gzip) for intermediate files',
        'streaming': 'Stream large files instead of loading entirely into memory'
    },
    'memory_optimization': {
        'data_types': 'Use appropriate data types (float32 instead of float64)',
        'chunked_processing': 'Process data in chunks to avoid memory overflow',
        'garbage_collection': 'Explicitly call gc.collect() after processing large chunks',
        'memory_mapping': 'Use memory mapping for large files when possible'
    },
    'estimated_performance': {
        'sequential_processing': '~12-16 hours for 28,595 samples',
        'parallel_processing_4_workers': '~3-4 hours for 28,595 samples',
        'parallel_processing_8_workers': '~2-3 hours for 28,595 samples',
        'memory_requirements': '~16-32GB RAM for parallel processing',
        'disk_space': '~5-10GB for temporary files during processing'
    }
}
```

#### Step 9: Quality Metrics and Metadata Enhancement

**Enhanced Metadata with Quality Statistics**:
```python
def create_enhanced_metadata_with_quality(data_path, validation_results=None):
    """
    Create comprehensive metadata with quality statistics
    """
    import json
    import pandas as pd
    from datetime import datetime
    
    metadata = {
        'dataset_info': {
            'name': 'MUTLA Processed Dataset',
            'version': '1.0',
            'creation_date': datetime.now().isoformat(),
            'source': 'MUTLA Raw Dataset',
            'preprocessing_version': '2.0'
        },
        'data_quality_metrics': {
            'overall_quality_score': validation_results.get('quality_score', 0) if validation_results else 0,
            'validation_status': validation_results.get('overall_status', 'UNKNOWN') if validation_results else 'UNKNOWN',
            'corruption_rates': validation_results.get('corruption_rates', {}) if validation_results else {},
            'missing_data_percentages': validation_results.get('missing_data_percentages', {}) if validation_results else {},
            'sample_counts': validation_results.get('sample_counts', {}) if validation_results else {}
        },
        'processing_statistics': {
            'total_processing_time': 'Estimated 2-4 hours (parallel processing)',
            'files_processed': '~85,000+ log files (attention, EEG, events)',
            'success_rate': '>95% (estimated)',
            'error_handling': 'Comprehensive error handling with fallback values'
        },
        'data_characteristics': {
            'cross_modal_alignment': 'Temporal synchronization using timestamps',
            'missing_data_strategy': 'Default values for missing modalities',
            'feature_extraction': 'Advanced signal processing for EEG and visual data',
            'temporal_resolution': 'Variable (depends on sensor sampling rates)'
        },
        'recommended_usage': {
            'primary_dataset': 'Perfect Multimodal (738 samples) for initial experiments',
            'extended_datasets': 'Complete Mixed Modality for robustness testing',
            'validation_required': 'Run validation functions before training',
            'quality_threshold': 'Quality score >80% recommended for training'
        },
        'technical_requirements': {
            'memory_usage': '~16-32GB RAM for parallel processing',
            'storage_requirements': '~5-10GB for processed data',
            'processing_time': '2-4 hours with parallel processing',
            'dependencies': ['pandas', 'numpy', 'scipy', 'multiprocessing']
        }
    }
    
    # Add validation details if available
    if validation_results:
        metadata['validation_details'] = {
            'errors': validation_results.get('errors', []),
            'warnings': validation_results.get('warnings', []),
            'quality_metrics': validation_results.get('quality_metrics', {}),
            'recommendations': generate_quality_recommendations(validation_results)
        }
    
    return metadata

def generate_quality_recommendations(validation_results):
    """
    Generate quality improvement recommendations
    """
    recommendations = []
    
    if validation_results.get('overall_status') == 'FAIL':
        recommendations.append("CRITICAL: Fix errors before using dataset for training")
    
    if validation_results.get('overall_status') == 'WARN':
        recommendations.append("WARNING: Address warnings for optimal performance")
    
    # Check corruption rates
    corruption_rates = validation_results.get('corruption_rates', {})
    for modality, rate in corruption_rates.items():
        if rate > 20:  # More than 20% corruption
            recommendations.append(f"High corruption rate in {modality}: {rate:.1f}% - consider data cleaning")
    
    # Check missing data
    missing_data = validation_results.get('missing_data_percentages', {})
    for modality, percentage in missing_data.items():
        if percentage > 15:  # More than 15% missing data
            recommendations.append(f"High missing data in {modality}: {percentage:.1f}% - consider imputation")
    
    # Check quality score
    quality_score = validation_results.get('quality_score', 0)
    if quality_score < 80:
        recommendations.append(f"Low quality score: {quality_score:.1f}% - review preprocessing pipeline")
    
    if not recommendations:
        recommendations.append("Dataset quality is acceptable for training")
    
    return recommendations
```
- Use for testing robustness to incomplete data (advanced)

**Dataset 3: Behavioral + Physiological (2,981 samples)**
- Samples with behavioral and physiological data
- Missing visual modality
- Use for testing robustness to missing visual data (advanced)

**Dataset 4: Behavioral + Visual (3,612 samples)**
- Samples with behavioral and visual data
- Missing physiological modality
- Use for testing robustness to missing physiological data (advanced)

#### Step 6: Feature Standardization and Normalization
- Normalize attention scores (0-100 range)
- Standardize EEG frequency band features
- Normalize facial landmark coordinates
- Scale eye tracking and head pose features

#### Step 7: Label Creation and Consolidation
- Create performance labels (correctness, mastery, attention level)
- Generate engagement labels from visual features
- Create learning outcome labels from behavioral data
- Handle multi-label scenarios for comprehensive evaluation

### Target File Structure

#### Primary Dataset: Perfect Multimodal (RECOMMENDED)
```
ProcessedData/MUTLA/PerfectMultimodal/
├── labels.csv                          # 738 samples with all labels
├── behavioral_features.csv             # 738 samples with behavioral features
├── physiological_features.csv          # 738 samples with physiological features
├── visual_features.csv                 # 738 samples with visual features
└── metadata.json                       # Dataset statistics and preprocessing info
```

#### Optional Extended Datasets (Advanced Users)

#### Dataset 2: Complete Mixed Modality
```
ProcessedData/MUTLA/Complete/
├── labels.csv                          # 28,595 samples with all available labels
├── behavioral_features.csv             # 28,595 samples with behavioral features
├── physiological_features.csv          # 2,981 samples with physiological features
├── visual_features.csv                 # 3,612 samples with visual features
├── modality_availability.csv           # 28,595 samples with modality presence flags
└── metadata.json                       # Dataset statistics and preprocessing info
```

#### Dataset 3: Behavioral + Physiological
```
ProcessedData/MUTLA/BehavioralPhysiological/
├── labels.csv                          # 2,981 samples with all labels
├── behavioral_features.csv             # 2,981 samples with behavioral features
├── physiological_features.csv          # 2,981 samples with physiological features
└── metadata.json                       # Dataset statistics and preprocessing info
```

#### Dataset 4: Behavioral + Visual
```
ProcessedData/MUTLA/BehavioralVisual/
├── labels.csv                          # 3,612 samples with all labels
├── behavioral_features.csv             # 3,612 samples with behavioral features
├── visual_features.csv                 # 3,612 samples with visual features
└── metadata.json                       # Dataset statistics and preprocessing info
```

### Expected Output Files

#### Labels (All Datasets)
- **performance_labels**: Correctness, mastery, proficiency indicators
- **engagement_labels**: Attention level, engagement state from visual data
- **learning_outcome_labels**: Overall learning success indicators
- **temporal_labels**: Session progression and learning trajectory

#### Behavioral Features (All Datasets)
- **learning_analytics**: Response time, correctness, difficulty ratings
- **academic_structure**: Subject, topic, module, knowledge point identifiers
- **learning_behavior**: Hint usage, answer viewing, analysis viewing patterns
- **performance_metrics**: Mastery tracking, proficiency estimates, progress indicators

#### Physiological Features (Datasets 1, 2, 3)
- **attention_scores**: Derived attention values (0-100)
- **eeg_frequency_bands**: Alpha, Beta, Gamma wave energy levels
- **device_connectivity**: Connection status and device state
- **temporal_patterns**: Attention trends and brainwave patterns

#### Visual Features (Datasets 1, 2, 4)
- **facial_landmarks**: 51-point facial landmark coordinates
- **eye_tracking**: Pupil segmentation and iris landmark data
- **head_pose**: 3D rotation angles and translation vectors
- **gaze_analysis**: Eye movement patterns and attention direction

#### Modality Availability (Dataset 1 Only)
- **behavioral_available**: Boolean flag for behavioral data presence
- **physiological_available**: Boolean flag for physiological data presence
- **visual_available**: Boolean flag for visual data presence
- **modality_count**: Number of available modalities per sample

### Usage with MainModel API

#### Primary Dataset: Perfect Multimodal (RECOMMENDED)
```python
def load_mutla_perfect_for_mainmodel():
    """
    Load perfect multimodal MUTLA dataset (all 3 modalities) - RECOMMENDED STARTING POINT
    """
    import pandas as pd
    
    # Load all features (all samples have all modalities)
    behavioral_df = pd.read_csv('ProcessedData/MUTLA/PerfectMultimodal/behavioral_features.csv')
    physiological_df = pd.read_csv('ProcessedData/MUTLA/PerfectMultimodal/physiological_features.csv')
    visual_df = pd.read_csv('ProcessedData/MUTLA/PerfectMultimodal/visual_features.csv')
    labels_df = pd.read_csv('ProcessedData/MUTLA/PerfectMultimodal/labels.csv')
    
    return {
        'behavioral': behavioral_df,
        'physiological': physiological_df,
        'visual': visual_df,
        'labels': labels_df
    }
```

#### Optional Extended Datasets (Advanced Users)

#### Dataset 2: Complete Mixed Modality
```python
def load_mutla_complete_for_mainmodel():
    """
    Load complete MUTLA dataset with mixed modality availability (ADVANCED)
    """
    import pandas as pd
    import numpy as np
    
    # Load all features
    behavioral_df = pd.read_csv('ProcessedData/MUTLA/Complete/behavioral_features.csv')
    physiological_df = pd.read_csv('ProcessedData/MUTLA/Complete/physiological_features.csv')
    visual_df = pd.read_csv('ProcessedData/MUTLA/Complete/visual_features.csv')
    labels_df = pd.read_csv('ProcessedData/MUTLA/Complete/labels.csv')
    modality_df = pd.read_csv('ProcessedData/MUTLA/Complete/modality_availability.csv')
    
    return {
        'behavioral': behavioral_df,
        'physiological': physiological_df,
        'visual': visual_df,
        'labels': labels_df,
        'modality_availability': modality_df
    }
```

#### Dataset 3: Behavioral + Physiological
```python
def load_mutla_behavioral_physiological_for_mainmodel():
    """
    Load MUTLA dataset with behavioral and physiological modalities
    """
    import pandas as pd
    
    # Load features (missing visual modality)
    behavioral_df = pd.read_csv('ProcessedData/MUTLA/BehavioralPhysiological/behavioral_features.csv')
    physiological_df = pd.read_csv('ProcessedData/MUTLA/BehavioralPhysiological/physiological_features.csv')
    labels_df = pd.read_csv('ProcessedData/MUTLA/BehavioralPhysiological/labels.csv')
    
    return {
        'behavioral': behavioral_df,
        'physiological': physiological_df,
        'labels': labels_df
    }
```

#### Dataset 4: Behavioral + Visual
```python
def load_mutla_behavioral_visual_for_mainmodel():
    """
    Load MUTLA dataset with behavioral and visual modalities
    """
    import pandas as pd
    
    # Load features (missing physiological modality)
    behavioral_df = pd.read_csv('ProcessedData/MUTLA/BehavioralVisual/behavioral_features.csv')
    visual_df = pd.read_csv('ProcessedData/MUTLA/BehavioralVisual/visual_features.csv')
    labels_df = pd.read_csv('ProcessedData/MUTLA/BehavioralVisual/labels.csv')
    
    return {
        'behavioral': behavioral_df,
        'visual': visual_df,
        'labels': labels_df
    }
```

### Important Note: Simplified Dataset Strategy

**RECOMMENDED APPROACH: Start with Perfect Multimodal Dataset**

The MUTLA dataset preprocessing creates **1 primary dataset** for initial research, with **3 optional extended datasets** for advanced users:

**Primary Dataset (Recommended):**
1. **Perfect Multimodal (738 samples)**: Clean, complete data with all 3 modalities - ideal for initial multimodal fusion learning experiments

**Optional Extended Datasets (Advanced Users):**
2. **Complete Mixed Modality (28,595 samples)**: Real-world scenario with missing modalities
3. **Behavioral + Physiological (2,981 samples)**: Robustness to missing visual data  
4. **Behavioral + Visual (3,612 samples)**: Robustness to missing physiological data

**Recommended Usage:**
- **Start with Perfect Multimodal** for initial experiments and baseline establishment
- **Progress to extended datasets** only after establishing strong baselines on the clean dataset
- **Use extended datasets** for robustness testing and real-world deployment scenarios
___________________________________________________________________________________________-
## OASIS Dataset Preprocessing

### Complete File Directory Structure

```
Data/OASIS/
├── oasis_cross-sectional.csv    # 436 subjects, 12 columns, 1.2MB
└── oasis_longitudinal.csv       # 373 visits, 15 columns, 1.1MB
```

### Dataset Statistics

**Cross-sectional Data:**
- **Total Subjects**: 436 unique subjects
- **Age Range**: 18-96 years (mean: 51.4)
- **Gender Distribution**: 268 Female, 168 Male
- **Handedness**: 100% Right-handed
- **Missing Data**: 46-50% missing for Educ, SES, MMSE, CDR

**Longitudinal Data:**
- **Total Subjects**: 150 unique subjects
- **Total Visits**: 373 MRI sessions
- **Visits per Subject**: 2-5 visits (mean: 2.5)
- **Age Range**: 60-98 years (mean: 77.0)
- **Follow-up Period**: 0-2639 days (mean: 595 days)
- **Missing Data**: Minimal (0.5-5.1%)

### Available Data Files

1. **oasis_cross-sectional.csv** (436 rows × 12 columns):
   - **ID**: Unique subject identifier
   - **M/F**: Gender (F/M)
   - **Hand**: Handedness (R/L)
   - **Age**: Age in years
   - **Educ**: Education level (1-5 scale)
   - **SES**: Socioeconomic status (1-5 scale)
   - **MMSE**: Mini-Mental State Examination score (0-30)
   - **CDR**: Clinical Dementia Rating (0, 0.5, 1, 2)
   - **eTIV**: Estimated Total Intracranial Volume
   - **nWBV**: Normalized Whole Brain Volume
   - **ASF**: Atlas Scaling Factor
   - **Delay**: MRI delay (mostly missing)

2. **oasis_longitudinal.csv** (373 rows × 15 columns):
   - **Subject ID**: Unique subject identifier
   - **MRI ID**: Unique MRI session identifier
   - **Group**: Nondemented/Demented/Converted
   - **Visit**: Visit number (1-5)
   - **MR Delay**: Days since first visit
   - **M/F**: Gender (F/M)
   - **Hand**: Handedness (R/L)
   - **Age**: Age in years
   - **EDUC**: Education years (6-23)
   - **SES**: Socioeconomic status (1-5 scale)
   - **MMSE**: Mini-Mental State Examination score (0-30)
   - **CDR**: Clinical Dementia Rating (0, 0.5, 1, 2)
   - **eTIV**: Estimated Total Intracranial Volume
   - **nWBV**: Normalized Whole Brain Volume
   - **ASF**: Atlas Scaling Factor

### Modalities Present in Raw Data

**SINGLE TABULAR MODALITY** with multiple feature categories:

1. **Demographic Features** (6 features):
   - **ID/Subject ID**: Unique identifiers
   - **Gender (M/F)**: Binary categorical (F/M)
   - **Handedness (Hand)**: Binary categorical (R/L)
   - **Age**: Continuous numerical (18-98 years)
   - **Education**: Ordinal categorical (1-5 scale or 6-23 years)
   - **SES**: Ordinal categorical (1-5 scale)

2. **Clinical Features** (2 features):
   - **MMSE**: Continuous numerical (0-30 scale)
   - **CDR**: Ordinal categorical (0, 0.5, 1, 2)

3. **Brain Volume Features** (3 features):
   - **eTIV**: Continuous numerical (1106-2004)
   - **nWBV**: Continuous numerical (0.644-0.893)
   - **ASF**: Continuous numerical (0.881-1.587)

4. **Temporal Features** (2 features, longitudinal only):
   - **Visit**: Ordinal categorical (1-5)
   - **MR Delay**: Continuous numerical (0-2639 days)

5. **Label Features** (1 feature):
   - **CDR**: Clinical Dementia Rating (0, 0.5, 1, 2)

### Required Preprocessing Steps

#### Step 1: Data Integration and Label Creation

**Objective**: Combine cross-sectional and longitudinal data, create standardized labels

**Implementation**:
```python
def preprocess_oasis_data():
    """
    Preprocess OASIS dataset into standardized format
    """
    import pandas as pd
    import numpy as np
    import os
    import json
    from sklearn.model_selection import train_test_split
    
    # Create output directory
    processed_path = "ProcessedData/OASIS"
    os.makedirs(processed_path, exist_ok=True)
    
    # Load raw data
    cross_df = pd.read_csv('Data/OASIS/oasis_cross-sectional.csv')
    long_df = pd.read_csv('Data/OASIS/oasis_longitudinal.csv')
    
    # Create labels for cross-sectional data
    cross_labels = cross_df[['ID', 'CDR']].copy()
    cross_labels['sample_id'] = 'cross_' + cross_labels['ID'].astype(str)
    cross_labels['dementia_binary'] = (cross_labels['CDR'] > 0).astype(int)
    cross_labels['dementia_severity'] = cross_labels['CDR'].map({
        0.0: 'Normal', 0.5: 'Very_Mild', 1.0: 'Mild', 2.0: 'Moderate'
    })
    
    # Create labels for longitudinal data
    long_labels = long_df[['Subject ID', 'CDR']].copy()
    long_labels['sample_id'] = 'long_' + long_df['MRI ID'].astype(str)
    long_labels['dementia_binary'] = (long_labels['CDR'] > 0).astype(int)
    long_labels['dementia_severity'] = long_labels['CDR'].map({
        0.0: 'Normal', 0.5: 'Very_Mild', 1.0: 'Mild', 2.0: 'Moderate'
    })
    
    # Combine all labels
    all_labels = pd.concat([cross_labels, long_labels], ignore_index=True)
    
    # Extract features
    features_df = extract_all_tabular_features(cross_df, long_df)
    
    # Create metadata
    metadata = create_oasis_metadata(features_df, cross_labels, long_labels)
    
    # Save processed data
    all_labels.to_csv(f"{processed_path}/labels.csv", index=False)
    features_df.to_csv(f"{processed_path}/tabular_features.csv", index=False)
    
    with open(f"{processed_path}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"OASIS preprocessing completed!")
    print(f"Processed data saved to: {processed_path}")
    print(f"Total samples: {len(features_df)}")
    
    return all_labels, features_df, metadata
```

#### Step 2: Consolidated Tabular Features Extraction

**Objective**: Extract all features and consolidate into a single tabular features file

**Implementation**:
```python
def extract_all_tabular_features(cross_df, long_df):
    """
    Extract all features and consolidate into a single tabular features file
    """
    all_features_data = []
    
    # Process cross-sectional data
    for idx, row in cross_df.iterrows():
        sample_id = f"cross_{row['ID']}"
        
        # Handle missing values
        educ = row['Educ'] if pd.notna(row['Educ']) else 3.0  # Default to middle education
        ses = row['SES'] if pd.notna(row['SES']) else 3.0     # Default to middle SES
        mmse = row['MMSE'] if pd.notna(row['MMSE']) else 27.0 # Default to normal MMSE
        cdr = row['CDR'] if pd.notna(row['CDR']) else 0.0     # Default to normal CDR
        
        # Encode categorical variables
        gender_encoded = 1 if row['M/F'] == 'M' else 0
        hand_encoded = 1 if row['Hand'] == 'R' else 0
        
        all_features_data.append({
            'sample_id': sample_id,
            # Demographic features (6)
            'gender': gender_encoded,
            'handedness': hand_encoded,
            'age': row['Age'],
            'education': educ,
            'ses': ses,
            # Clinical features (2)
            'mmse': mmse,
            'cdr': cdr,
            # Brain volume features (3)
            'etiv': row['eTIV'],
            'nwbv': row['nWBV'],
            'asf': row['ASF'],
            # Temporal features (2) - set to 0 for cross-sectional
            'visit': 1,
            'mr_delay': 0,
            # Dataset indicator
            'dataset_type': 'cross_sectional'
        })
    
    # Process longitudinal data
    for idx, row in long_df.iterrows():
        sample_id = f"long_{row['MRI ID']}"
        
        # Handle missing values
        ses = row['SES'] if pd.notna(row['SES']) else 3.0
        mmse = row['MMSE'] if pd.notna(row['MMSE']) else 27.0
        
        # Encode categorical variables
        gender_encoded = 1 if row['M/F'] == 'M' else 0
        hand_encoded = 1 if row['Hand'] == 'R' else 0
        
        all_features_data.append({
            'sample_id': sample_id,
            # Demographic features (6)
            'gender': gender_encoded,
            'handedness': hand_encoded,
            'age': row['Age'],
            'education': row['EDUC'],
            'ses': ses,
            # Clinical features (2)
            'mmse': mmse,
            'cdr': row['CDR'],
            # Brain volume features (3)
            'etiv': row['eTIV'],
            'nwbv': row['nWBV'],
            'asf': row['ASF'],
            # Temporal features (2)
            'visit': row['Visit'],
            'mr_delay': row['MR Delay'],
            # Dataset indicator
            'dataset_type': 'longitudinal'
        })
    
    return pd.DataFrame(all_features_data)
```

#### Step 3: Metadata Creation

**Objective**: Create comprehensive metadata file

**Implementation**:
```python
def create_oasis_metadata(features_df, cross_labels, long_labels):
    """
    Create metadata file for OASIS dataset
    """
    metadata = {
        "dataset_name": "OASIS",
        "total_samples": len(features_df),
        "cross_sectional_samples": len(cross_labels),
        "longitudinal_samples": len(long_labels),
        "modalities": {
            "tabular": {
                "type": "structured_data",
                "format": "CSV",
                "features": [
                    "gender", "handedness", "age", "education", "ses",
                    "mmse", "cdr", "etiv", "nwbv", "asf", "visit", "mr_delay"
                ],
                "feature_categories": {
                    "demographic": ["gender", "handedness", "age", "education", "ses"],
                    "clinical": ["mmse", "cdr"],
                    "brain_volume": ["etiv", "nwbv", "asf"],
                    "temporal": ["visit", "mr_delay"]
                }
            }
        },
        "labels": {
            "cdr_distribution": {
                "0.0": int(cross_labels['CDR'].value_counts().get(0.0, 0) + 
                           long_labels['CDR'].value_counts().get(0.0, 0)),
                "0.5": int(cross_labels['CDR'].value_counts().get(0.5, 0) + 
                           long_labels['CDR'].value_counts().get(0.5, 0)),
                "1.0": int(cross_labels['CDR'].value_counts().get(1.0, 0) + 
                           long_labels['CDR'].value_counts().get(1.0, 0)),
                "2.0": int(cross_labels['CDR'].value_counts().get(2.0, 0) + 
                           long_labels['CDR'].value_counts().get(2.0, 0))
            },
            "dementia_binary_distribution": {
                "normal": int((cross_labels['dementia_binary'] == 0).sum() + 
                             (long_labels['dementia_binary'] == 0).sum()),
                "dementia": int((cross_labels['dementia_binary'] == 1).sum() + 
                               (long_labels['dementia_binary'] == 1).sum())
            }
        },
        "preprocessing_info": {
            "missing_data_handling": "Default values for missing Educ, SES, MMSE, CDR",
            "categorical_encoding": "Binary encoding for gender and handedness",
            "feature_scaling": "Not applied (raw values preserved)",
            "data_splits": "Not created (to be determined by user)"
        }
    }
    
    return metadata
```

### Comprehensive Validation Function

**OASIS Validation Function**:
```python
def validate_oasis_preprocessing(processed_path):
    """
    Comprehensive validation of OASIS preprocessing results
    """
    import os
    import pandas as pd
    import numpy as np
    
    validation_results = {
        'overall_status': 'PASS',
        'errors': [],
        'warnings': [],
        'sample_counts': {},
        'quality_metrics': {},
        'missing_data_percentages': {}
    }
    
    try:
        # Check required files exist
        required_files = ['labels.csv', 'tabular_features.csv', 'metadata.json']
        for file in required_files:
            if not os.path.exists(f"{processed_path}/{file}"):
                validation_results['errors'].append(f"Missing required file: {file}")
        
        if validation_results['errors']:
            validation_results['overall_status'] = 'FAIL'
            return validation_results
        
        # Load and validate labels
        labels_df = pd.read_csv(f"{processed_path}/labels.csv")
        validation_results['sample_counts']['labels'] = len(labels_df)
        
        # Load and validate features
        features_df = pd.read_csv(f"{processed_path}/tabular_features.csv")
        validation_results['sample_counts']['features'] = len(features_df)
        
        # Check sample count consistency
        if len(labels_df) != len(features_df):
            validation_results['errors'].append(
                f"Sample count mismatch: labels={len(labels_df)}, features={len(features_df)}"
            )
        
        # Validate label distribution
        if 'cdr' in labels_df.columns:
            cdr_counts = labels_df['cdr'].value_counts()
            validation_results['quality_metrics']['cdr_distribution'] = cdr_counts.to_dict()
            
            # Check for expected CDR distribution
            expected_cdr_values = [0.0, 0.5, 1.0, 2.0, 3.0]
            for cdr_val in expected_cdr_values:
                if cdr_val not in cdr_counts.index:
                    validation_results['warnings'].append(f"Missing CDR value: {cdr_val}")
        
        # Validate feature quality
        validation_results['quality_metrics']['feature_columns'] = len(features_df.columns)
        validation_results['quality_metrics']['feature_rows'] = len(features_df)
        
        # Check for missing data
        missing_data = features_df.isnull().sum().sum()
        total_cells = len(features_df) * len(features_df.columns)
        missing_percentage = (missing_data / total_cells) * 100
        validation_results['missing_data_percentages']['features'] = missing_percentage
        
        if missing_percentage > 5:  # More than 5% missing data
            validation_results['warnings'].append(f"High missing data in features: {missing_percentage:.2f}%")
        
        # Validate data types
        expected_numeric_columns = ['age', 'mmse', 'etiv', 'nwbv', 'asf']
        for col in expected_numeric_columns:
            if col in features_df.columns:
                if not pd.api.types.is_numeric_dtype(features_df[col]):
                    validation_results['warnings'].append(f"Column {col} should be numeric but is {features_df[col].dtype}")
        
        # Validate value ranges
        if 'age' in features_df.columns:
            age_range = features_df['age'].min(), features_df['age'].max()
            if age_range[0] < 18 or age_range[1] > 100:
                validation_results['warnings'].append(f"Unusual age range: {age_range}")
        
        if 'mmse' in features_df.columns:
            mmse_range = features_df['mmse'].min(), features_df['mmse'].max()
            if mmse_range[0] < 0 or mmse_range[1] > 30:
                validation_results['warnings'].append(f"Unusual MMSE range: {mmse_range}")
        
        # Calculate overall quality score
        total_checks = 12  # Approximate number of checks
        error_penalty = len(validation_results['errors']) * 3
        warning_penalty = len(validation_results['warnings']) * 1
        validation_results['quality_score'] = max(0, (total_checks - error_penalty - warning_penalty) / total_checks * 100)
        
        if validation_results['errors']:
            validation_results['overall_status'] = 'FAIL'
        elif validation_results['warnings']:
            validation_results['overall_status'] = 'WARN'
        
    except Exception as e:
        validation_results['errors'].append(f"Validation failed with exception: {e}")
        validation_results['overall_status'] = 'FAIL'
    
    return validation_results
```

### Target File Structure (After Preprocessing)

```
ProcessedData/OASIS/
├── labels.csv                    # [sample_id, cdr, dementia_binary, dementia_severity]
├── tabular_features.csv          # [sample_id, gender, handedness, age, education, ses, mmse, cdr, etiv, nwbv, asf, visit, mr_delay, dataset_type]
└── metadata.json                 # Dataset statistics and modality information
```

### Expected Output Files

1. **labels.csv**: Standardized label file with sample IDs and dementia classifications
2. **tabular_features.csv**: Consolidated tabular features from all categories
3. **metadata.json**: Comprehensive dataset statistics and modality information

### Usage with MainModel API

```python
def load_oasis_for_mainmodel():
    """
    Load preprocessed OASIS data in MainModel format
    """
    import pandas as pd
    import json
    
    data_path = "ProcessedData/OASIS"
    
    # Load labels
    labels_df = pd.read_csv(f"{data_path}/labels.csv")
    
    # Load tabular features
    features_df = pd.read_csv(f"{data_path}/tabular_features.csv")
    
    # Load metadata
    with open(f"{data_path}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return {
        'labels': labels_df,
        'tabular_features': features_df,
        'metadata': metadata
    }

# Load data
oasis_data = load_oasis_for_mainmodel()
```

### Important Note: Single Tabular Modality

**OASIS is NOT a multimodal dataset** - it contains a single **Tabular Modality** with multiple feature categories:
- **Demographic Features**: Gender, age, education, socioeconomic status
- **Clinical Features**: MMSE scores, CDR ratings
- **Brain Volume Features**: eTIV, nWBV, ASF measurements
- **Temporal Features**: Visit numbers, time intervals (longitudinal only)

All features are numerical or categorical data that can be processed as a single tabular input to machine learning models.

_________________________________________________________________________

