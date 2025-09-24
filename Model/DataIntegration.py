#!/usr/bin/env python3
"""
Stage 1: DataIntegration - Simplified Production-Ready Version

This module provides a clean, robust data integration pipeline with:
- Simple, reliable data loading
- Clear data contracts between stages
- Proper error handling and validation
- No fake lazy loading or complex consistency logic
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Constants
class DataIntegrationConfig:
    """Configuration constants for DataIntegration."""
    MAX_TRUNCATION_RATIO = 0.2  # Warn if truncating more than 20% of data
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_TARGET_SIZE = (224, 224)
    DEFAULT_TEST_SIZE = 0.2
    DEFAULT_MAX_SAMPLES = 1000
    EPSILON = 1e-8  # Small value to avoid division by zero
    SUPPORTED_IMAGE_FORMATS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    SUPPORTED_CSV_ENCODINGS = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    # Truncation warning thresholds (configurable based on dataset characteristics)
    TRUNCATION_WARNING_THRESHOLDS = {
        'small_dataset': 0.1,    # 10% for datasets < 100 samples
        'medium_dataset': 0.2,   # 20% for datasets 100-1000 samples  
        'large_dataset': 0.3     # 30% for datasets > 1000 samples
    }

class SimpleDataLoader:
    """
    Simplified multimodal data loader with clear contracts and robust error handling.
    """
    
    # ============================================================================
    # STEP 1: INITIALIZE DATA LOADER
    # ============================================================================
    
    def __init__(self, cache_dir: Optional[str] = None, device: str = 'cpu', 
                 handle_missing_modalities: bool = True, missing_modality_strategy: str = "zero_fill",
                 handle_class_imbalance: bool = True, class_imbalance_strategy: str = "report",
                 fast_mode: bool = True, max_samples: int = DataIntegrationConfig.DEFAULT_MAX_SAMPLES,
                 normalize: bool = True, normalization_method: str = "standard",
                 test_size: float = DataIntegrationConfig.DEFAULT_TEST_SIZE, stratify: bool = True, 
                 random_state: int = DataIntegrationConfig.DEFAULT_RANDOM_STATE,
                 target_size: Tuple[int, int] = DataIntegrationConfig.DEFAULT_TARGET_SIZE, 
                 channels_first: bool = True, alignment_strategy: str = "min_count"):
        """
        Initialize the data loader with clear, simple parameters.
        """
        # Input validation
        self._validate_config_parameters(
            device, missing_modality_strategy, class_imbalance_strategy,
            max_samples, normalization_method, test_size, random_state, target_size, alignment_strategy
        )
        
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device
        self.handle_missing_modalities = handle_missing_modalities
        self.missing_modality_strategy = missing_modality_strategy
        self.handle_class_imbalance = handle_class_imbalance
        self.class_imbalance_strategy = class_imbalance_strategy
        self.fast_mode = fast_mode
        self.max_samples = max_samples
        self.normalize = normalize
        self.normalization_method = normalization_method
        self.test_size = test_size
        self.stratify = stratify
        self.random_state = random_state
        self.target_size = target_size
        self.channels_first = channels_first
        self.alignment_strategy = alignment_strategy
        
        # Set random seeds for reproducibility
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Data storage
        self.data = {}
        self._modality_sample_counts = {}
        
        logger.info(f"Initialized SimpleDataLoader with device: {device}, fast_mode: {fast_mode}")
    
    def _validate_config_parameters(self, device: str, missing_modality_strategy: str, 
                                  class_imbalance_strategy: str, max_samples: int, 
                                  normalization_method: str, test_size: float, 
                                  random_state: int, target_size: Tuple[int, int], 
                                  alignment_strategy: str):
        """Validate configuration parameters."""
        # Device validation
        if device not in ['cpu', 'cuda']:
            raise ValueError(f"Invalid device: {device}. Must be 'cpu' or 'cuda'")
        
        # Strategy validation
        valid_missing_strategies = ['zero_fill', 'mean_fill', 'drop_samples']
        if missing_modality_strategy not in valid_missing_strategies:
            raise ValueError(f"Invalid missing_modality_strategy: {missing_modality_strategy}. "
                           f"Must be one of {valid_missing_strategies}")
        
        valid_imbalance_strategies = ['report', 'balance', 'weight']
        if class_imbalance_strategy not in valid_imbalance_strategies:
            raise ValueError(f"Invalid class_imbalance_strategy: {class_imbalance_strategy}. "
                           f"Must be one of {valid_imbalance_strategies}")
        
        # Numeric validation
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")
        
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        
        if random_state < 0:
            raise ValueError(f"random_state must be non-negative, got {random_state}")
        
        # Target size validation
        if len(target_size) != 2 or any(dim <= 0 for dim in target_size):
            raise ValueError(f"target_size must be a tuple of 2 positive integers, got {target_size}")
        
        # Normalization method validation
        valid_norm_methods = ['standard', 'minmax']
        if normalization_method not in valid_norm_methods:
            raise ValueError(f"Invalid normalization_method: {normalization_method}. "
                           f"Must be one of {valid_norm_methods}")
        
        # Alignment strategy validation
        valid_alignment_strategies = ['min_count', 'max_count', 'majority_modality']
        if alignment_strategy not in valid_alignment_strategies:
            raise ValueError(f"Invalid alignment_strategy: {alignment_strategy}. "
                           f"Must be one of {valid_alignment_strategies}")
        
        logger.info("Configuration parameters validated successfully")
    
    # ============================================================================
    # STEP 2: LOAD RAW DATA
    # ============================================================================
    
    def _load_labels(self, label_file: str) -> np.ndarray:
        """Load labels from CSV file."""
        try:
            # Try multiple encodings for robust CSV loading
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(label_file, encoding=encoding)
                    logger.info(f"Successfully loaded {label_file} with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError(f"Could not load {label_file} with any supported encoding")
            
            # Check for common label column names in order of preference
            label_columns = ['label', 'class', 'class_id', 'dementia_binary', 'correctness', 'target']
            
            labels = None
            for col in label_columns:
                if col in df.columns:
                    labels = df[col].values
                    logger.info(f"Using label column: {col}")
                    break
            
            if labels is None:
                # Safe column selection with proper validation
                if 'id' in df.columns and len(df.columns) > 1:
                    label_col = df.columns[1]
                elif len(df.columns) > 0:
                    label_col = df.columns[0]
                else:
                    raise ValueError(f"CSV file {label_file} has no columns")
                labels = df[label_col].values
                logger.info(f"Using fallback label column: {label_col}")
            
            # Handle non-numeric labels by converting to numeric
            if labels.dtype == 'object':
                # Try to convert to numeric first
                try:
                    labels = pd.to_numeric(labels, errors='coerce')
                    # If there are NaN values, use label encoding
                    if pd.isna(labels).any():
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        labels = le.fit_transform(df[col] if col in df.columns else df[label_col])
                        logger.info(f"Applied label encoding to non-numeric labels")
                except:
                    # Use label encoding for non-numeric data
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    labels = le.fit_transform(df[col] if col in df.columns else df[label_col])
                    logger.info(f"Applied label encoding to non-numeric labels")
            
            return labels.astype(np.int64)
        except Exception as e:
            logger.error(f"Error loading labels from {label_file}: {e}")
            raise
    
    def _load_modality_data(self, file_path: str, modality_type: str) -> np.ndarray:
        """Load modality data based on type."""
        try:
            if modality_type == 'image':
                return self._load_image_data(file_path)
            elif modality_type == 'spectral':
                return self._load_spectral_data(file_path)
            elif modality_type == 'tabular':
                return self._load_tabular_data(file_path)
            elif modality_type == 'time_series':
                return self._load_time_series_data(file_path)
            elif modality_type == 'visual':
                return self._load_visual_data(file_path)
            else:
                raise ValueError(f"Unknown modality type: {modality_type}")
        except Exception as e:
            logger.error(f"Error loading {modality_type} data from {file_path}: {e}")
            raise
    
    def _load_image_data(self, file_path: str) -> np.ndarray:
        """Load image data from directory with memory-efficient processing and proper resource cleanup."""
        image_dir = Path(file_path)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {file_path}")
        
        # Support multiple image formats
        image_files = []
        for ext in DataIntegrationConfig.SUPPORTED_IMAGE_FORMATS:
            image_files.extend(sorted(image_dir.glob(ext)))
        
        if not image_files:
            raise ValueError(f"No valid image files found in {file_path}")
        
        logger.info(f"Found {len(image_files)} image files in {file_path}")
        
        # Pre-allocate array for memory efficiency
        if self.channels_first:
            target_shape = (len(image_files), 3, *self.target_size)  # (N, C, H, W)
        else:
            target_shape = (len(image_files), *self.target_size, 3)  # (N, H, W, C)
        
        images = np.empty(target_shape, dtype=np.float32)
        failed_files = []
        successful_count = 0
        
        for i, img_file in enumerate(image_files):
            try:
                # Use context manager for proper resource cleanup
                with Image.open(img_file) as img:
                    img = img.convert('RGB')
                    img = img.resize(self.target_size)
                    img_array = np.array(img, dtype=np.float32)
                
                # Validate image dimensions
                if img_array.shape[:2] != self.target_size:
                    logger.warning(f"Image {img_file} resized from {img_array.shape[:2]} to {self.target_size}")
                
                if self.channels_first:
                    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
                
                # Store directly in pre-allocated array
                images[successful_count] = img_array / 255.0
                successful_count += 1
                
            except (OSError, IOError, ValueError) as e:
                logger.warning(f"Error loading image {img_file}: {e}")
                failed_files.append(img_file)
                continue
            except Exception as e:
                logger.error(f"Unexpected error loading image {img_file}: {e}")
                failed_files.append(img_file)
                continue
        
        if successful_count == 0:
            raise ValueError(f"No valid images could be loaded from {file_path}")
        
        # Trim array to actual loaded size
        if successful_count < len(image_files):
            images = images[:successful_count]
            logger.warning(f"Failed to load {len(failed_files)} images out of {len(image_files)} total")
        
        logger.info(f"Successfully loaded {successful_count} images with memory-efficient processing")
        return images
    
    def _load_spectral_data(self, file_path: str) -> np.ndarray:
        """Load spectral data from NPY file or directory of NPY files."""
        try:
            file_path = Path(file_path)
            
            if file_path.is_file():
                # Single NPY file
                try:
                    data = np.load(file_path)
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                    return data.astype(np.float32)
                except (OSError, IOError) as e:
                    raise FileNotFoundError(f"Cannot read NPY file {file_path}: {e}")
                except (ValueError, np.core._exceptions._ArrayMemoryError) as e:
                    raise ValueError(f"Invalid or corrupted NPY file {file_path}: {e}")
            
            elif file_path.is_dir():
                # Directory of NPY files (common for EuroSAT spectral data)
                npy_files = sorted(file_path.glob("*.npy"))
                if not npy_files:
                    raise ValueError(f"No NPY files found in directory: {file_path}")
                
                logger.info(f"Loading {len(npy_files)} spectral files from {file_path}")
                
                spectral_arrays = []
                for npy_file in npy_files:
                    try:
                        data = np.load(npy_file)
                        if data.ndim == 1:
                            data = data.reshape(-1, 1)
                        spectral_arrays.append(data)
                    except (OSError, IOError) as e:
                        logger.warning(f"Cannot read NPY file {npy_file}: {e}")
                        continue
                    except (ValueError, np.core._exceptions._ArrayMemoryError) as e:
                        logger.warning(f"Invalid or corrupted NPY file {npy_file}: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected error loading {npy_file}: {e}")
                        continue
                
                if not spectral_arrays:
                    raise ValueError(f"No valid spectral data could be loaded from {file_path}")
                
                # Concatenate along feature dimension
                try:
                    combined_data = np.concatenate(spectral_arrays, axis=1)
                    logger.info(f"Combined {len(spectral_arrays)} spectral files into shape {combined_data.shape}")
                    return combined_data.astype(np.float32)
                except ValueError as e:
                    raise ValueError(f"Cannot concatenate spectral arrays: {e}")
            
            else:
                raise FileNotFoundError(f"Spectral data path not found: {file_path}")
                
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error loading spectral data from {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading spectral data from {file_path}: {e}")
            raise
    
    def _load_tabular_data(self, file_path: str) -> np.ndarray:
        """Load tabular data from CSV file with robust encoding handling."""
        try:
            # Try different encodings for robust CSV loading
            for encoding in DataIntegrationConfig.SUPPORTED_CSV_ENCODINGS:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"Cannot read CSV file {file_path} with any supported encoding")
            
            # Remove non-numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                raise ValueError(f"No numeric columns found in CSV file: {file_path}")
            
            return numeric_df.values.astype(np.float32)
            
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            raise FileNotFoundError(f"Cannot read CSV file {file_path}: {e}")
        except (ValueError, pd.errors.ParserError) as e:
            raise ValueError(f"Invalid CSV file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading tabular data from {file_path}: {e}")
            raise
    
    def _load_time_series_data(self, file_path: str) -> np.ndarray:
        """Load time series data with proper parsing for different formats."""
        try:
            if file_path.endswith('.csv'):
                # Handle CSV time series data
                df = pd.read_csv(file_path)
                numeric_data = df.select_dtypes(include=[np.number])
                if numeric_data.empty:
                    raise ValueError(f"No numeric data found in CSV file: {file_path}")
                return numeric_data.values.astype(np.float32)
            
            elif file_path.endswith('.log'):
                # Handle structured log files properly
                data = []
                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        
                        # Skip headers, comments, and empty lines
                        if line.startswith('#') or line.startswith('//') or not line:
                            continue
                        
                        try:
                            # Handle tab-separated or space-separated values
                            if '\t' in line:
                                values = [float(x.strip()) for x in line.split('\t') if x.strip()]
                            else:
                                values = [float(x.strip()) for x in line.split() if x.strip()]
                            
                            if values:  # Only add non-empty rows
                                data.append(values)
                        except ValueError as e:
                            logger.warning(f"Skipping line {line_num} in {file_path}: {e}")
                            continue
                
                if not data:
                    raise ValueError(f"No valid time series data found in {file_path}")
                
                # Pad to same length
                max_len = max(len(row) for row in data)
                padded_data = []
                for row in data:
                    padded_row = row + [0.0] * (max_len - len(row))
                    padded_data.append(padded_row)
                
                return np.array(padded_data, dtype=np.float32)
            
            else:
                raise ValueError(f"Unsupported time series file format: {file_path}")
                
        except Exception as e:
            logger.error(f"Error loading time series data from {file_path}: {e}")
            raise
    
    def _load_visual_data(self, file_path: str) -> np.ndarray:
        """Load visual data from NPY file."""
        data = np.load(file_path)
        
        # Handle structured arrays (like MUTLA webcam data)
        if data.dtype.names is not None:
            logger.info(f"Detected structured array with fields: {data.dtype.names}")
            # Extract numeric features from structured array
            features = []
            for field_name in data.dtype.names:
                field_data = data[field_name]
                if field_data.dtype.names is not None:
                    # Nested structured array - flatten recursively
                    for nested_field in field_data.dtype.names:
                        nested_data = field_data[nested_field]
                        if nested_data.ndim > 1:
                            # Flatten multi-dimensional data
                            features.append(nested_data.reshape(len(nested_data), -1))
                        else:
                            features.append(nested_data.reshape(-1, 1))
                else:
                    # Simple field
                    if field_data.ndim > 1:
                        features.append(field_data.reshape(len(field_data), -1))
                    else:
                        features.append(field_data.reshape(-1, 1))
            
            # Concatenate all features (handle mixed data types)
            if features:
                # Convert all features to float32 to ensure compatibility
                float_features = []
                for feature in features:
                    if feature.dtype != np.float32:
                        try:
                            float_features.append(feature.astype(np.float32))
                        except (ValueError, TypeError):
                            # If conversion fails, skip this feature
                            logger.warning(f"Skipping feature with incompatible dtype: {feature.dtype}")
                            continue
                    else:
                        float_features.append(feature)
                
                if float_features:
                    data = np.concatenate(float_features, axis=1)
                    logger.info(f"Extracted {data.shape[1]} features from structured array")
                else:
                    # Fallback: use first field
                    first_field = data.dtype.names[0]
                    data = data[first_field]
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                    logger.info(f"Using first field '{first_field}' as fallback")
            else:
                # Fallback: use first field
                first_field = data.dtype.names[0]
                data = data[first_field]
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                logger.info(f"Using first field '{first_field}' as fallback")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        return data.astype(np.float32)
    
    # ============================================================================
    # STEP 4: DATA VALIDATION AND ALIGNMENT
    # ============================================================================
    
    def _validate_and_align_data(self, modality_data: Dict[str, np.ndarray], labels: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        SIMPLIFIED: Validate and align data with clear contracts and truncation warnings.
        """
        # Get sample counts
        sample_counts = {name: len(data) for name, data in modality_data.items()}
        sample_counts['labels'] = len(labels)
        
        logger.info(f"Sample counts: {sample_counts}")
        
        # Find minimum and maximum counts
        min_count = min(sample_counts.values())
        max_count = max(sample_counts.values())
        
        if len(set(sample_counts.values())) > 1:
            # Determine target count based on alignment strategy
            if self.alignment_strategy == "min_count":
                target_count = min_count
                strategy_desc = "minimum count"
            elif self.alignment_strategy == "max_count":
                target_count = max_count
                strategy_desc = "maximum count"
            elif self.alignment_strategy == "majority_modality":
                # Find the modality with the most samples
                majority_modality = max(sample_counts.items(), key=lambda x: x[1] if x[0] != 'labels' else 0)
                target_count = majority_modality[1]
                strategy_desc = f"majority modality ({majority_modality[0]})"
            
            # Warn if truncating too much data
            truncation_ratio = 1.0 - (min_count / max_count)
            
            # Determine appropriate threshold based on dataset size
            if min_count < 100:
                threshold = DataIntegrationConfig.TRUNCATION_WARNING_THRESHOLDS['small_dataset']
            elif min_count < 1000:
                threshold = DataIntegrationConfig.TRUNCATION_WARNING_THRESHOLDS['medium_dataset']
            else:
                threshold = DataIntegrationConfig.TRUNCATION_WARNING_THRESHOLDS['large_dataset']
            
            if truncation_ratio > threshold:
                logger.warning(f"⚠️  Truncating {truncation_ratio:.1%} of data (threshold: {threshold:.1%}) - consider investigating data alignment")
                logger.warning(f"   Min count: {min_count}, Max count: {max_count}")
                logger.warning(f"   This may indicate data loading or preprocessing issues")
            else:
                logger.info(f"Truncating {truncation_ratio:.1%} of data (acceptable level, threshold: {threshold:.1%})")
            
            logger.info(f"Inconsistent sample counts. Aligning all to {strategy_desc}: {target_count}")
            
            # Align all modalities to target count
            aligned_modality_data = {}
            for name, data in modality_data.items():
                if len(data) == target_count:
                    aligned_modality_data[name] = data
                elif len(data) > target_count:
                    # Truncate to target count
                    aligned_modality_data[name] = data[:target_count]
                    logger.info(f"Truncated {name} from {len(data)} to {target_count} samples")
                else:
                    # Pad to target count using the missing modality strategy
                    if self.missing_modality_strategy == "zero_fill":
                        if data.ndim == 1:
                            padding = np.zeros(target_count - len(data), dtype=data.dtype)
                        else:
                            padding = np.zeros((target_count - len(data), *data.shape[1:]), dtype=data.dtype)
                        aligned_modality_data[name] = np.concatenate([data, padding], axis=0)
                        logger.info(f"Zero-padded {name} from {len(data)} to {target_count} samples")
                    elif self.missing_modality_strategy == "mean_fill":
                        if data.ndim == 1:
                            mean_val = np.mean(data)
                            padding = np.full(target_count - len(data), mean_val, dtype=data.dtype)
                        else:
                            mean_vals = np.mean(data, axis=0)
                            padding = np.tile(mean_vals, (target_count - len(data), 1))
                        aligned_modality_data[name] = np.concatenate([data, padding], axis=0)
                        logger.info(f"Mean-padded {name} from {len(data)} to {target_count} samples")
                    else:
                        # For drop_samples, we can't pad, so truncate others to this size
                        logger.warning(f"Cannot pad {name} with drop_samples strategy, truncating others to {len(data)}")
                        target_count = len(data)
                        aligned_modality_data[name] = data
            
            # Align labels
            if len(labels) == target_count:
                aligned_labels = labels
            elif len(labels) > target_count:
                aligned_labels = labels[:target_count]
                logger.info(f"Truncated labels from {len(labels)} to {target_count} samples")
            else:
                # For labels, we need to handle this differently - duplicate or generate labels
                if self.alignment_strategy == "majority_modality":
                    # Duplicate existing labels to reach target count
                    n_duplicates = target_count - len(labels)
                    duplicate_indices = np.random.choice(len(labels), n_duplicates, replace=True)
                    aligned_labels = np.concatenate([labels, labels[duplicate_indices]])
                    logger.info(f"Duplicated {n_duplicates} labels to reach {target_count} samples")
                else:
                    aligned_labels = labels[:target_count]
                    logger.info(f"Truncated labels from {len(labels)} to {target_count} samples")
            
            # Store final counts
            self._modality_sample_counts = {name: target_count for name in modality_data.keys()}
            self._modality_sample_counts['labels'] = target_count
            
            logger.info(f"Data consistency fixed. All modalities now have {target_count} samples")
            return aligned_modality_data, aligned_labels
        else:
            # All counts are the same
            logger.info("All modalities have consistent sample counts")
            self._modality_sample_counts = sample_counts
            return modality_data, labels
    
    # ============================================================================
    # STEP 5: FAST MODE SAMPLING (if enabled)
    # ============================================================================
    
    def _apply_fast_mode_sampling(self, modality_data: Dict[str, np.ndarray], labels: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Apply fast mode sampling if enabled with proper random sampling."""
        if not self.fast_mode:
            return modality_data, labels
        
        total_samples = len(labels)
        if total_samples <= self.max_samples:
            logger.info(f"Fast mode: dataset size ({total_samples}) <= max_samples ({self.max_samples}), no sampling needed")
            return modality_data, labels
        
        # Use random sampling with fixed seed for reproducibility
        np.random.seed(self.random_state)
        indices = np.random.choice(total_samples, self.max_samples, replace=False)
        indices.sort()  # Maintain order for consistency
        
        logger.info(f"Fast mode: randomly sampled {self.max_samples} from {total_samples} samples")
        
        # Apply sampling to all modalities
        sampled_data = {}
        for name, data in modality_data.items():
            sampled_data[name] = data[indices]
        
        sampled_labels = labels[indices]
        
        return sampled_data, sampled_labels
    
    def _handle_missing_modalities(self, modality_data: Dict[str, np.ndarray], labels: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict]:
        """Handle missing modalities with configurable strategies."""
        if not self.handle_missing_modalities:
            return modality_data, labels, {}
        
        logger.info(f"Handling missing modalities with strategy: {self.missing_modality_strategy}")
        
        missing_info = {
            'strategy': self.missing_modality_strategy,
            'missing_modalities': [],
            'handled_modalities': []
        }
        
        # Get the target sample count (from labels)
        target_count = len(labels)
        
        # Check each modality for missing data
        for modality_name, data in modality_data.items():
            if len(data) < target_count:
                missing_count = target_count - len(data)
                missing_info['missing_modalities'].append({
                    'name': modality_name,
                    'missing_samples': missing_count,
                    'total_samples': len(data)
                })
                
                # Apply the configured strategy
                if self.missing_modality_strategy == "zero_fill":
                    # Pad with zeros
                    if data.ndim == 1:
                        padding = np.zeros(missing_count, dtype=data.dtype)
                    else:
                        padding = np.zeros((missing_count, *data.shape[1:]), dtype=data.dtype)
                    
                    modality_data[modality_name] = np.concatenate([data, padding], axis=0)
                    logger.info(f"Zero-filled {missing_count} missing samples for {modality_name}")
                    
                elif self.missing_modality_strategy == "mean_fill":
                    # Pad with mean values
                    if data.ndim == 1:
                        mean_val = np.mean(data)
                        padding = np.full(missing_count, mean_val, dtype=data.dtype)
                    else:
                        mean_vals = np.mean(data, axis=0)
                        padding = np.tile(mean_vals, (missing_count, 1))
                    
                    modality_data[modality_name] = np.concatenate([data, padding], axis=0)
                    logger.info(f"Mean-filled {missing_count} missing samples for {modality_name}")
                    
                elif self.missing_modality_strategy == "drop_samples":
                    # Remove corresponding samples from labels and other modalities
                    logger.warning(f"Dropping {missing_count} samples due to missing {modality_name} data")
                    # This would require more complex logic to maintain consistency
                    # For now, we'll use zero_fill as fallback
                    if data.ndim == 1:
                        padding = np.zeros(missing_count, dtype=data.dtype)
                    else:
                        padding = np.zeros((missing_count, *data.shape[1:]), dtype=data.dtype)
                    modality_data[modality_name] = np.concatenate([data, padding], axis=0)
                
                missing_info['handled_modalities'].append(modality_name)
        
        if missing_info['missing_modalities']:
            logger.info(f"Handled missing modalities: {[m['name'] for m in missing_info['missing_modalities']]}")
        else:
            logger.info("No missing modalities detected")
        
        return modality_data, labels, missing_info
    
    # ============================================================================
    # STEP 6: NORMALIZATION
    # ============================================================================
    
    def _apply_normalization(self, modality_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply normalization to modality data while preserving spatial relationships."""
        if not self.normalize:
            return modality_data
        
        logger.info(f"Applying {self.normalization_method} normalization")
        
        normalized_data = {}
        for name, data in modality_data.items():
            # Determine if this is spatial data (images) or feature data
            # Images have shape (N, C, H, W) or (N, H, W, C) - 4D
            # Spectral data has shape (N, H, W) - 3D, but should be treated as feature data
            is_spatial = len(data.shape) == 4  # Only 4D arrays are spatial (images)
            
            if is_spatial:
                # For spatial data, normalize per-channel to preserve spatial relationships
                if self.normalization_method == 'standard':
                    # Standard normalization per channel
                    if self.channels_first:  # (N, C, H, W)
                        for c in range(data.shape[1]):
                            channel_data = data[:, c, :, :]
                            mean = channel_data.mean()
                            std = channel_data.std()
                            if std > DataIntegrationConfig.EPSILON:
                                data[:, c, :, :] = (channel_data - mean) / std
                    else:  # (N, H, W, C)
                        for c in range(data.shape[3]):
                            channel_data = data[:, :, :, c]
                            mean = channel_data.mean()
                            std = channel_data.std()
                            if std > DataIntegrationConfig.EPSILON:
                                data[:, :, :, c] = (channel_data - mean) / std
                else:
                    # Min-max normalization per channel
                    if self.channels_first:  # (N, C, H, W)
                        for c in range(data.shape[1]):
                            channel_data = data[:, c, :, :]
                            data_min = channel_data.min()
                            data_max = channel_data.max()
                            if data_max > data_min + DataIntegrationConfig.EPSILON:
                                data[:, c, :, :] = (channel_data - data_min) / (data_max - data_min)
                    else:  # (N, H, W, C)
                        for c in range(data.shape[3]):
                            channel_data = data[:, :, :, c]
                            data_min = channel_data.min()
                            data_max = channel_data.max()
                            if data_max > data_min + DataIntegrationConfig.EPSILON:
                                data[:, :, :, c] = (channel_data - data_min) / (data_max - data_min)
                
                logger.info(f"Applied {self.normalization_method} normalization to spatial data {name} (preserved spatial structure)")
            else:
                # For non-spatial data, use standard feature normalization
                if self.normalization_method == 'standard':
                    if data.ndim > 1:
                        # Reshape for normalization
                        original_shape = data.shape
                        data_flat = data.reshape(data.shape[0], -1)
                        scaler = StandardScaler()
                        data_normalized = scaler.fit_transform(data_flat)
                        data = data_normalized.reshape(original_shape)
                    else:
                        scaler = StandardScaler()
                        data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
                else:
                    # Min-max normalization
                    data_min = data.min()
                    data_max = data.max()
                    if data_max > data_min + DataIntegrationConfig.EPSILON:
                        data = (data - data_min) / (data_max - data_min)
                
                logger.info(f"Applied {self.normalization_method} normalization to feature data {name}")
            
            normalized_data[name] = data.astype(np.float32)
        
        return normalized_data
    
    # ============================================================================
    # STEP 7: TRAIN/TEST SPLIT
    # ============================================================================
    
    def _create_train_test_split(self, modality_data: Dict[str, np.ndarray], labels: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Create train/test split with proper stratification and fallback."""
        try:
            # Check if stratification is possible
            can_stratify = self.stratify and len(labels) > 0
            
            if can_stratify:
                # Check if we have enough samples per class for stratification
                unique_labels, counts = np.unique(labels, return_counts=True)
                min_samples_per_class = counts.min()
                test_size_samples = int(self.test_size * len(labels))
                
                # Need at least 2 samples per class for stratification
                if min_samples_per_class < 2 or test_size_samples < len(unique_labels):
                    logger.warning(f"Cannot stratify: min_samples_per_class={min_samples_per_class}, "
                                 f"test_size_samples={test_size_samples}, num_classes={len(unique_labels)}")
                    can_stratify = False
            
            # Create train/test split
            if can_stratify:
                train_indices, test_indices = train_test_split(
                    range(len(labels)), 
                    test_size=self.test_size, 
                    stratify=labels, 
                    random_state=self.random_state
                )
                logger.info("Created stratified train/test split")
            else:
                train_indices, test_indices = train_test_split(
                    range(len(labels)), 
                    test_size=self.test_size, 
                    random_state=self.random_state
                )
                logger.info("Created non-stratified train/test split (stratification not possible)")
            
            # Split data
            train_data = {}
            test_data = {}
            
            for name, data in modality_data.items():
                train_data[name] = data[train_indices]
                test_data[name] = data[test_indices]
            
            train_labels = labels[train_indices]
            test_labels = labels[test_indices]
            
            logger.info(f"Created train/test split: {len(train_labels)} train, {len(test_labels)} test")
            
            return train_data, test_data, train_labels, test_labels
            
        except Exception as e:
            logger.error(f"Error creating train/test split: {e}")
            raise
    
    # ============================================================================
    # STEP 8: DATA STORAGE AND ACCESS
    # ============================================================================
    
    def _get_class_distribution(self, labels: np.ndarray) -> Dict[int, int]:
        """Get class distribution statistics."""
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))
    
    def get_train_data(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Get training data and labels."""
        if 'train_data' not in self.data:
            raise ValueError("No training data available. Call load_multimodal_data first.")
        return self.data['train_data'], self.data['train_labels']
    
    def get_test_data(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Get test data and labels."""
        if 'test_data' not in self.data:
            raise ValueError("No test data available. Call load_multimodal_data first.")
        return self.data['test_data'], self.data['test_labels']
    
    # ============================================================================
    # MAIN ORCHESTRATION METHOD
    # ============================================================================
    
    def load_multimodal_data(self, label_file: str, modality_files: Dict[str, str], 
                           modality_types: Dict[str, str]) -> Dict[str, Any]:
        """
        Load multimodal data with clear contracts and robust validation.
        
        Parameters
        ----------
        label_file : str
            Path to labels CSV file
        modality_files : dict
            Dictionary mapping modality names to file paths
        modality_types : dict
            Dictionary mapping modality names to data types
            
        Returns
        -------
        dict
            Processed data dictionary with clear structure
        """
        try:
            logger.info("Starting multimodal data loading")
            
            # Step 2: Load labels
            labels = self._load_labels(label_file)
            logger.info(f"Loaded {len(labels)} labels")
            
            # Step 2: Load modality data
            modality_data = {}
            for modality_name, file_path in modality_files.items():
                modality_type = modality_types.get(modality_name, 'unknown')
                data = self._load_modality_data(file_path, modality_type)
                modality_data[modality_name] = data
                logger.info(f"Loaded {modality_name} ({modality_type}): {data.shape}")
            
            # Step 4: Validate and align data
            modality_data, labels = self._validate_and_align_data(modality_data, labels)
            
            # Step 5: Apply fast mode sampling if enabled
            modality_data, labels = self._apply_fast_mode_sampling(modality_data, labels)
            
            # Step 5: Handle missing modalities if needed
            if self.handle_missing_modalities:
                modality_data, labels, missing_info = self._handle_missing_modalities(modality_data, labels)
            
            # Step 6: Apply normalization
            if self.normalize:
                modality_data = self._apply_normalization(modality_data)
            
            # Step 7: Create train/test split
            train_data, test_data, train_labels, test_labels = self._create_train_test_split(modality_data, labels)
            
            # Step 8: Store results
            self.data = {
                'train_data': train_data,
                'train_labels': train_labels,
                'test_data': test_data,
                'test_labels': test_labels,
                'modality_info': {name: {'shape': data.shape, 'type': modality_types.get(name, 'unknown')} 
                                for name, data in train_data.items()},
                'class_distribution': self._get_class_distribution(train_labels),
                'preprocessing_info': {
                    'normalization_method': self.normalization_method,
                    'target_size': self.target_size,
                    'channels_first': self.channels_first
                }
            }
            
            logger.info("✅ Multimodal data loading completed successfully")
            return self.data
            
        except Exception as e:
            logger.error(f"❌ Error in multimodal data loading: {e}")
            raise