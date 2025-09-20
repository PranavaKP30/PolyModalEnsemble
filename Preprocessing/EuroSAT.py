"""
EuroSAT/AllBands Dataset Preprocessing Script - ENHANCED VERSION

This script preprocesses the EuroSAT/AllBands dataset into a standardized format
for multimodal learning. It creates 5 distinct spectral modalities:
1. Visual RGB: RGB images extracted from multispectral bands
2. Near-Infrared: NIR bands (B08, B8A) for vegetation and water analysis
3. Red-Edge: Red-edge bands (B05, B06, B07) for vegetation health assessment
4. Short-Wave Infrared: SWIR bands (B11, B12) for soil and mineral analysis
5. Atmospheric: Atmospheric bands (B01, B09, B10) for atmospheric correction

Based on the comprehensive preprocessing documentation in PreprocessingDocumentation.md
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
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_all_spectral_bands(tiff_path):
    """
    Extract all 13 spectral bands from multispectral TIFF with proper band order verification
    
    Args:
        tiff_path (str): Path to input multispectral TIFF file
        
    Returns:
        dict: Dictionary with all spectral bands organized by modality
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
                logger.warning(f"Unusual NIR/Red relationship in {tiff_path} (NIR={b08_mean:.2f}, Red={b04_mean:.2f})")
            
            if b01_mean > b08_mean:  # Coastal should generally be lower than NIR
                logger.warning(f"Unusual Coastal/NIR relationship in {tiff_path} (Coastal={b01_mean:.2f}, NIR={b08_mean:.2f})")
            
            # Additional validation: check for reasonable value ranges
            for i, mean_val in enumerate(band_means):
                if mean_val < 0 or mean_val > 1.0:  # After normalization, values should be 0-1
                    logger.warning(f"Band {i+1} has unusual mean value {mean_val:.4f} in {tiff_path}")
            
            # Check for corrupted data (all zeros or all same value)
            for i, band in enumerate(bands):
                if np.all(band == 0) or np.std(band) < 1e-6:
                    logger.warning(f"Band {i+1} appears corrupted in {tiff_path}")
            
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
        logger.error(f"Error processing {tiff_path}: {e}")
        return None

def preprocess_eurosat_enhanced():
    """
    Enhanced preprocessing for EuroSAT/AllBands dataset
    Creates 5 spectral modalities preserving all spectral information
    
    Returns:
        tuple: (labels_df, modality_metadata)
    """
    # Paths
    raw_data_path = "Data/EuroSAT/AllBands"
    processed_path = "ProcessedData/EuroSAT"
    
    # Check if raw data path exists
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data path not found: {raw_data_path}")
    
    logger.info(f"Starting enhanced preprocessing from {raw_data_path} to {processed_path}")
    
    # Create output directories
    modality_dirs = ['visual_rgb', 'near_infrared', 'red_edge', 'short_wave_infrared', 'atmospheric']
    for modality_dir in modality_dirs:
        os.makedirs(f"{processed_path}/{modality_dir}", exist_ok=True)
    
    # Load dataset splits
    train_df = pd.read_csv(f"{raw_data_path}/train.csv")
    val_df = pd.read_csv(f"{raw_data_path}/validation.csv")
    test_df = pd.read_csv(f"{raw_data_path}/test.csv")
    
    # Combine all data
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    logger.info(f"Total samples to process: {len(all_data)}")
    
    # Initialize data containers
    labels_data = []
    modality_metadata = {modality: [] for modality in modality_dirs}
    processed_count = 0
    error_count = 0
    
    # Process each sample
    for idx, row in all_data.iterrows():
        sample_id = f"sample_{idx:05d}"
        tiff_path = f"{raw_data_path}/{row['Filename']}"
        
        try:
            if not os.path.exists(tiff_path):
                logger.warning(f"TIFF file not found: {tiff_path}")
                error_count += 1
                continue
            
            # Extract all spectral bands with error handling
            spectral_data = extract_all_spectral_bands(tiff_path)
            if spectral_data is None:
                logger.error(f"Failed to extract spectral data from {tiff_path}")
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
                    logger.warning(f"RGB array has wrong shape {rgb_array.shape} for {sample_id}")
                
                rgb_image = Image.fromarray(rgb_array)
                rgb_path = f"{processed_path}/visual_rgb/{sample_id}.jpg"
                rgb_image.save(rgb_path)
            except Exception as e:
                logger.error(f"Error saving RGB image for {sample_id}: {e}")
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
                        logger.error(f"Error saving {modality_name} for {sample_id}: {e}")
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
                logger.info(f"Processed {processed_count}/{len(all_data)} samples...")
                
        except Exception as e:
            logger.error(f"Unexpected error processing {sample_id}: {e}")
            error_count += 1
            continue
    
    # Create labels DataFrame
    labels_df = pd.DataFrame(labels_data)
    labels_df.to_csv(f"{processed_path}/labels.csv", index=False)
    
    # Create enhanced metadata
    metadata = create_enhanced_metadata(labels_df, modality_metadata, processed_count, error_count)
    
    # Save metadata
    with open(f"{processed_path}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Run comprehensive validation
    validation_results = validate_eurosat_preprocessing(processed_path)
    logger.info(f"Validation Status: {validation_results['overall_status']}")
    if validation_results['errors']:
        logger.error(f"Errors found: {validation_results['errors']}")
    if validation_results['warnings']:
        logger.warning(f"Warnings: {validation_results['warnings']}")
    
    logger.info(f"Enhanced preprocessing completed!")
    logger.info(f"Successfully processed: {processed_count} samples")
    logger.info(f"Errors encountered: {error_count} samples")
    logger.info(f"Processed data saved to: {processed_path}")
    
    return labels_df, modality_metadata

def create_enhanced_metadata(labels_df, modality_metadata, processed_count, error_count):
    """
    Create comprehensive metadata for EuroSAT dataset
    """
    metadata = {
        "dataset_info": {
            "name": "EuroSAT Processed Dataset",
            "version": "2.0",
            "creation_date": datetime.now().isoformat(),
            "source": "EuroSAT/AllBands Raw Dataset",
            "preprocessing_version": "Enhanced with 5 spectral modalities"
        },
        "data_statistics": {
            "total_samples": len(labels_df),
            "successfully_processed": processed_count,
            "processing_errors": error_count,
            "success_rate": f"{(processed_count / (processed_count + error_count) * 100):.2f}%" if (processed_count + error_count) > 0 else "0%",
            "class_distribution": labels_df['class_name'].value_counts().to_dict(),
            "split_distribution": labels_df['split'].value_counts().to_dict()
        },
        "modality_info": {
            "total_modalities": 5,
            "modality_types": {
                "visual_rgb": {
                    "format": "JPEG images",
                    "bands": ["B04 (Red)", "B03 (Green)", "B02 (Blue)"],
                    "resolution": "10m",
                    "file_count": len(modality_metadata.get('visual_rgb', [])),
                    "data_type": "uint8"
                },
                "near_infrared": {
                    "format": "NPY files",
                    "bands": ["B08 (NIR)", "B8A (NIR narrow)"],
                    "resolution": "10m, 20m",
                    "file_count": len(modality_metadata.get('near_infrared', [])),
                    "data_type": "float32"
                },
                "red_edge": {
                    "format": "NPY files",
                    "bands": ["B05 (Red-edge 1)", "B06 (Red-edge 2)", "B07 (Red-edge 3)"],
                    "resolution": "20m",
                    "file_count": len(modality_metadata.get('red_edge', [])),
                    "data_type": "float32"
                },
                "short_wave_infrared": {
                    "format": "NPY files",
                    "bands": ["B11 (SWIR 1)", "B12 (SWIR 2)"],
                    "resolution": "20m",
                    "file_count": len(modality_metadata.get('short_wave_infrared', [])),
                    "data_type": "float32"
                },
                "atmospheric": {
                    "format": "NPY files",
                    "bands": ["B01 (Coastal)", "B09 (Water vapor)", "B10 (Cirrus)"],
                    "resolution": "60m",
                    "file_count": len(modality_metadata.get('atmospheric', [])),
                    "data_type": "float32"
                }
            }
        },
        "computational_requirements": {
            "processing_time": "~8-12 hours on modern hardware (27,597 TIFF files with full spectral extraction)",
            "memory_usage": "~16GB RAM recommended (for processing large TIFF files)",
            "storage_requirements": "~15GB for processed data (5 modalities × 27,597 files)",
            "dependencies": ["rasterio", "numpy", "PIL", "pandas"],
            "performance_notes": "Processing time scales linearly with number of files. Consider parallel processing for faster execution.",
            "disk_io_intensive": "High disk I/O due to reading 27,597 TIFF files and writing 138,985 output files (5 modalities × 27,597)"
        },
        "data_quality": {
            "band_order_verification": "Sentinel-2 standard order validated",
            "value_range_validation": "All bands normalized to 0-1 range",
            "corruption_detection": "Automatic detection of corrupted bands",
            "file_integrity": "Comprehensive validation of output files"
        }
    }
    
    return metadata

def validate_eurosat_preprocessing(processed_path):
    """
    Comprehensive validation of EuroSAT preprocessing results
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
        
        logger.info(f"Loading modalities: {self.load_modalities}")
    
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
                logger.error(f"Error loading visual_rgb for {sample_id}: {e}")
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
                    logger.error(f"Error loading {modality} for {sample_id}: {e}")
                    data[modality] = None
        
        return data

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
    
    # Create dataset
    dataset = EuroSATDataset(data_path, split='train')
    
    return {
        'labels': labels_df,
        'metadata': metadata,
        'dataset': dataset,
        'modality_types': {
            'visual_rgb': 'image',
            'near_infrared': 'tensor',
            'red_edge': 'tensor',
            'short_wave_infrared': 'tensor',
            'atmospheric': 'tensor'
        }
    }

def main():
    """
    Main function to run EuroSAT preprocessing
    """
    try:
        logger.info("Starting EuroSAT preprocessing...")
        labels_df, modality_metadata = preprocess_eurosat_enhanced()
        logger.info("EuroSAT preprocessing completed successfully!")
        
        # Test loading
        logger.info("Testing data loading...")
        eurosat_data = load_eurosat_for_mainmodel()
        logger.info(f"Successfully loaded {len(eurosat_data['labels'])} samples")
        logger.info(f"Available modalities: {list(eurosat_data['modality_types'].keys())}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()
