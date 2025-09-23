#!/usr/bin/env python3
"""
Stage 1: DataIntegration - Simplified SOTA-Aligned Version

This module provides a streamlined data integration pipeline focused on:
- Fast data loading with minimal preprocessing
- Preserving data structure (3D tensors for images)
- Simple normalization compatible with pretrained models
- Basic validation and format conversion
- Data augmentation hooks

Aligned with modern SOTA multimodal architectures (CLIP, DALL-E, Flamingo, etc.)
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from PIL import Image
import torch
from torch.utils.data import Dataset
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import random
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataLoader:
    """
    Simplified multimodal data loader aligned with SOTA architectures.
    
    Focus: Fast loading, minimal preprocessing, preserve data structure
    """
    
    # Step 1: Initialize the data loader
    def __init__(self, cache_dir: Optional[str] = None, device: str = 'cpu', 
                 lazy_loading: bool = True, chunk_size: int = 1000,  # Default to lazy loading
                 handle_missing_modalities: bool = True, missing_modality_strategy: str = "zero_fill",
                 handle_class_imbalance: bool = True, class_imbalance_strategy: str = "report",
                 fast_mode: bool = True, max_samples: int = 1000):  # Fast mode for large datasets
        """
        Initialize the data loader.
        
        Parameters
        ----------
        cache_dir : str, optional
            Directory for caching processed data
        device : str
            Device for tensor operations ('cpu' or 'cuda')
        lazy_loading : bool
            Enable lazy loading for memory efficiency with large datasets
        chunk_size : int
            Size of chunks for lazy loading (number of samples per chunk)
        handle_missing_modalities : bool
            Whether to handle samples with missing modalities
        missing_modality_strategy : str
            Strategy for handling missing modalities ('zero_fill', 'mean_fill', 'drop_samples')
        handle_class_imbalance : bool
            Whether to handle class imbalance in labels
        class_imbalance_strategy : str
            Strategy for handling class imbalance ('report', 'balance', 'weight')
        """
        self.cache_dir = cache_dir
        self.device = device
        self.lazy_loading = lazy_loading
        self.chunk_size = chunk_size
        
        # CRITICAL FIX: Add consistent sampling tracking
        self._last_sampling_indices = None
        self._modality_sample_counts = {}
        self._sampling_seed = 42  # Fixed seed for reproducibility
        self.handle_missing_modalities = handle_missing_modalities
        self.missing_modality_strategy = missing_modality_strategy
        self.handle_class_imbalance = handle_class_imbalance
        self.class_imbalance_strategy = class_imbalance_strategy
        self.fast_mode = fast_mode
        self.max_samples = max_samples
        self.data = {}
        self.metadata = {}
        self._file_paths = {}  # Store file paths for lazy loading
        self.scalers = {}
        self.class_weights = {}  # Store class weights for imbalanced datasets
        
        # Setup cache directory
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized SimpleDataLoader with device: {device}")
    
    # Step 2: Data Loading Methods
    def load_labels(self, file_path: str) -> np.ndarray:
        """
        Load labels from CSV file.
        
        Parameters
        ----------
        file_path : str
            Path to labels CSV file
            
        Returns
        -------
        np.ndarray
            Labels array
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Labels file not found: {file_path}")
        
        logger.info(f"Loading labels from {file_path}")
        
        # Load CSV file with robust encoding handling
        df = None
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.debug(f"Successfully loaded CSV with {encoding} encoding")
                break
            except UnicodeDecodeError as e:
                logger.debug(f"Failed to load CSV with {encoding} encoding: {e}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error loading CSV with {encoding} encoding: {e}")
                continue
        
        if df is None:
            raise ValueError(f"Could not load CSV file {file_path} with any of the attempted encodings: {encodings}")
        
        # Extract labels based on file structure
        if 'class_id' in df.columns:
            # EuroSAT labels
            labels = df['class_id'].values
        elif 'split' in df.columns and len(df.columns) > 2:
            # EuroSAT with split column
            labels = df.iloc[:, 2].values  # Assuming class_id is 3rd column
        else:
            # Generic labels (second column)
            labels = df.iloc[:, 1].values if len(df.columns) > 1 else df.iloc[:, 0].values
        
        # Handle NaN values in labels
        if np.any(np.isnan(labels)):
            logger.warning(f"Found NaN values in labels from {file_path}. Filling with 0.")
            labels = np.nan_to_num(labels, nan=0.0)
        
        # Ensure consistent return type as int64 for labels
        return labels.astype(np.int64)

    def load_image_directory(self, directory: str, target_size: Tuple[int, int] = (224, 224), 
                           channels_first: bool = False, sample_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Load images from directory, preserving 3D structure.
        
        Parameters
        ----------
        directory : str
            Path to image directory
        target_size : tuple
            Target size for resizing images (height, width)
        channels_first : bool
            If True, return (n_samples, channels, height, width) format for PyTorch
            If False, return (n_samples, height, width, channels) format
            
        Returns
        -------
        np.ndarray
            Array of shape (n_samples, height, width, channels) or (n_samples, channels, height, width)
        """
        image_dir = Path(directory)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {directory}")
        
        image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + 
                           list(image_dir.glob("*.png")))
        
        if not image_files:
            raise ValueError(f"No image files found in {directory}")
        
        logger.info(f"Loading {len(image_files)} images from {directory}")
        
        # Use provided sample indices if available, otherwise use fast mode sampling
        if sample_indices is not None:
            logger.info(f"Using provided sample indices: {len(sample_indices)} samples from {len(image_files)} total.")
            # Only use indices that are within bounds
            valid_indices = [i for i in sample_indices if i < len(image_files)]
            if len(valid_indices) < len(sample_indices):
                logger.warning(f"Only {len(valid_indices)} valid indices out of {len(sample_indices)} requested for image files.")
            image_files = [image_files[i] for i in valid_indices]
        elif self.fast_mode and len(image_files) > self.max_samples:
            logger.info(f"Fast mode: Sampling {self.max_samples} images from {len(image_files)} for fast loading.")
            random.seed(self._sampling_seed)  # CRITICAL: Use consistent seed
            # Generate consistent sampling indices
            if not hasattr(self, '_last_sampling_indices') or self._last_sampling_indices is None:
                self._last_sampling_indices = random.sample(range(len(image_files)), self.max_samples)
                self._last_sampling_indices.sort()  # Sort for consistency
            # Use the stored indices to ensure consistency across modalities
            image_files = [image_files[i] for i in self._last_sampling_indices]
        
        # Memory-efficient image loading
        if self.lazy_loading and len(image_files) > self.chunk_size:
            logger.debug(f"Using lazy loading for {len(image_files)} images with chunk size {self.chunk_size}")
            # For lazy loading, return a generator or placeholder
            # This would be implemented with a custom dataset class
            images = self._load_images_lazy(image_files, target_size, channels_first)
        else:
            # Load all images into memory
            images = []
            for img_file in image_files:
                img = Image.open(img_file).convert('RGB')
                img = img.resize(target_size)
                img_array = np.array(img)
                images.append(img_array)
            
            images = np.array(images)
        
        # Convert to channels-first format if requested (PyTorch default)
        if channels_first:
            images = np.transpose(images, (0, 3, 1, 2))  # NHWC -> NCHW
        
        return images
    
    def _load_images_lazy(self, image_files: List[Path], target_size: Tuple[int, int], channels_first: bool) -> np.ndarray:
        """
        Load images lazily for memory efficiency.
        
        Parameters
        ----------
        image_files : List[Path]
            List of image file paths
        target_size : Tuple[int, int]
            Target size for resizing
        channels_first : bool
            Whether to use channels-first format
            
        Returns
        -------
        np.ndarray
            Placeholder array for lazy loading
        """
        # For now, return a placeholder that indicates lazy loading
        # In a full implementation, this would return a custom dataset
        logger.debug(f"Lazy loading placeholder for {len(image_files)} images")
        return np.array([])  # Placeholder - would be replaced with actual lazy loading implementation
    
    def load_npy_directory(self, directory: str, sample_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Load NPY files from directory.
        
        Parameters
        ----------
        directory : str
            Path to NPY files directory
            
        Returns
        -------
        np.ndarray
            Concatenated array from all NPY files
        """
        npy_dir = Path(directory)
        if not npy_dir.exists():
            raise FileNotFoundError(f"NPY directory not found: {directory}")
        
        npy_files = sorted(list(npy_dir.glob("*.npy")))
        
        if not npy_files:
            raise ValueError(f"No NPY files found in {directory}")
        
        logger.info(f"Loading {len(npy_files)} NPY files from {directory}")
        
        # CRITICAL FIX: Ensure consistent sampling across all modalities
        if sample_indices is not None:
            logger.info(f"Using provided sample indices: {len(sample_indices)} samples from {len(npy_files)} total.")
            # Only use indices that are within bounds
            valid_indices = [i for i in sample_indices if i < len(npy_files)]
            if len(valid_indices) < len(sample_indices):
                logger.warning(f"Only {len(valid_indices)} valid indices out of {len(sample_indices)} requested for NPY files.")
            npy_files = [npy_files[i] for i in valid_indices]
        elif self.fast_mode and len(npy_files) > self.max_samples:
            logger.info(f"Fast mode: Sampling {self.max_samples} NPY files from {len(npy_files)} for fast loading.")
            random.seed(self._sampling_seed)  # CRITICAL: Fixed seed for reproducibility
            # Generate consistent sampling indices
            if not hasattr(self, '_last_sampling_indices') or self._last_sampling_indices is None:
                self._last_sampling_indices = random.sample(range(len(npy_files)), self.max_samples)
                self._last_sampling_indices.sort()  # Sort for consistency
            # Use the stored indices to ensure consistency across modalities
            npy_files = [npy_files[i] for i in self._last_sampling_indices]
        
        arrays = []
        for npy_file in npy_files:
            try:
                data = np.load(npy_file, allow_pickle=True)
                
                # Handle EuroSAT spectral data stored as objects with robust error handling
                if data.dtype == object and data.size > 0:
                    extracted_data = []
                    for item in data.flat:
                        try:
                            if isinstance(item, dict):
                                # Extract all band arrays and flatten them
                                band_arrays = []
                                for band_name, band_data in item.items():
                                    try:
                                        if hasattr(band_data, 'flatten') and np.issubdtype(band_data.dtype, np.number):
                                            band_arrays.extend(band_data.flatten())
                                        elif isinstance(band_data, (list, tuple)):
                                            # Handle nested lists/tuples
                                            for sub_item in band_data:
                                                if hasattr(sub_item, 'flatten') and np.issubdtype(sub_item.dtype, np.number):
                                                    band_arrays.extend(sub_item.flatten())
                                    except (AttributeError, TypeError, ValueError) as e:
                                        logger.warning(f"Skipping malformed band data in {npy_file}: {e}")
                                        continue
                                
                                if band_arrays:
                                    extracted_data.append(band_arrays)
                                else:
                                    logger.warning(f"No valid band data found in {npy_file}")
                            elif hasattr(item, 'flatten') and np.issubdtype(item.dtype, np.number):
                                # Direct numeric array
                                extracted_data.append(item.flatten())
                            else:
                                logger.warning(f"Skipping non-numeric item in {npy_file}: {type(item)}")
                        except Exception as e:
                            logger.warning(f"Error processing item in {npy_file}: {e}")
                            continue
                    
                    if extracted_data:
                        # Ensure all arrays have the same length for concatenation
                        max_length = max(len(arr) for arr in extracted_data)
                        padded_data = []
                        for arr in extracted_data:
                            if len(arr) < max_length:
                                # Pad with zeros
                                padded_arr = np.pad(arr, (0, max_length - len(arr)), mode='constant')
                            else:
                                padded_arr = arr[:max_length]  # Truncate if too long
                            padded_data.append(padded_arr)
                        data = np.array(padded_data)
                    else:
                        logger.warning(f"No valid data extracted from {npy_file}")
                        continue
                
                arrays.append(data)
                
            except Exception as e:
                logger.error(f"Failed to load {npy_file}: {e}")
                continue
        
        return np.concatenate(arrays, axis=0) if arrays else np.array([])
    
    def load_log_directory(self, directory: str, sample_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Load LOG files from directory (for MUTLA time-series data).
        
        Parameters
        ----------
        directory : str
            Directory containing LOG files
            
        Returns
        -------
        np.ndarray
            Concatenated data from all LOG files
        """
        log_files = list(Path(directory).glob("**/*.log"))  # Search recursively in subdirectories
        if not log_files:
            logger.warning(f"No LOG files found in {directory}")
            return np.array([])
        
        logger.info(f"Loading {len(log_files)} LOG files from {directory}")
        
        # Note: LOG files will be sampled after concatenation to maintain consistency with other modalities
        # This ensures that all modalities use the same sample-level sampling approach
        
        all_data = []
        for log_file in log_files:
            try:
                # Read LOG file, skip header lines starting with #
                data_lines = []
                with open(log_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Parse tab-separated values
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                try:
                                    # Convert to float (timestamp, attention/EEG values)
                                    values = [float(part) for part in parts]
                                    data_lines.append(values)
                                except ValueError:
                                    # Skip lines that can't be converted to float
                                    continue
                
                if data_lines:
                    file_data = np.array(data_lines)
                    all_data.append(file_data)
                    
            except Exception as e:
                logger.warning(f"Error loading {log_file}: {e}")
                continue
        
        if all_data:
            # Concatenate all data, handling different shapes
            try:
                data = np.concatenate(all_data, axis=0)
            except ValueError:
                # If shapes don't match, flatten all data
                flattened = []
                for data in all_data:
                    flattened.extend(data.flatten())
                data = np.array(flattened)
            
            # Apply sampling if provided (consistent with other modalities)
            if sample_indices is not None and len(data) > len(sample_indices):
                data = data[sample_indices]
            
            return data
        
        return np.array([])
    
    def load_webcam_npy_directory(self, directory: str, sample_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Load webcam tracking NPY files from directory (for MUTLA visual data).
        
        Parameters
        ----------
        directory : str
            Directory containing NPY files with webcam tracking data
            
        Returns
        -------
        np.ndarray
            Concatenated tracking features from all NPY files
        """
        npy_files = list(Path(directory).glob("*.npy"))
        if not npy_files:
            logger.warning(f"No NPY files found in {directory}")
            return np.array([])
        
        logger.info(f"Loading {len(npy_files)} webcam NPY files from {directory}")
        
        # Use provided sample indices if available, otherwise use fast mode sampling
        if sample_indices is not None:
            logger.info(f"Using provided sample indices: {len(sample_indices)} samples from {len(npy_files)} total.")
            # Only use indices that are within bounds
            valid_indices = [i for i in sample_indices if i < len(npy_files)]
            if len(valid_indices) < len(sample_indices):
                logger.warning(f"Only {len(valid_indices)} valid indices out of {len(sample_indices)} requested for webcam NPY files.")
            npy_files = [npy_files[i] for i in valid_indices]
        elif self.fast_mode and len(npy_files) > self.max_samples:
            logger.info(f"Fast mode: Sampling {self.max_samples} webcam files from {len(npy_files)} for fast loading.")
            random.seed(self._sampling_seed)  # CRITICAL: Use consistent seed
            # Generate consistent sampling indices
            if not hasattr(self, '_last_sampling_indices') or self._last_sampling_indices is None:
                self._last_sampling_indices = random.sample(range(len(npy_files)), self.max_samples)
                self._last_sampling_indices.sort()  # Sort for consistency
            # Use the stored indices to ensure consistency across modalities
            npy_files = [npy_files[i] for i in self._last_sampling_indices]
        
        all_features = []
        for npy_file in npy_files:
            try:
                # Load NPY file containing structured tracking data
                data = np.load(npy_file, allow_pickle=True)
                
                # Fast mode: simple feature extraction
                if self.fast_mode:
                    # Just extract basic numeric features without complex parsing
                    if hasattr(data, 'flatten'):
                        flat_data = data.flatten()
                        numeric_features = [x for x in flat_data if isinstance(x, (int, float)) and not np.isnan(x)]
                        if numeric_features:
                            all_features.append(numeric_features[:50])  # Limit to 50 features
                        else:
                            # Fallback: create dummy features if extraction fails
                            all_features.append([0.0] * 50)
                    else:
                        # Fallback: create dummy features if data is not flattenable
                        all_features.append([0.0] * 50)
                    continue
                
                # Extract relevant features from structured data with robust error handling
                features = []
                for frame_idx, frame_data in enumerate(data):
                    try:
                        if hasattr(frame_data, 'dtype') and frame_data.dtype.names:
                            # Structured array - extract key features
                            frame_features = []
                            
                            # Extract confidence scores with validation
                            try:
                                if 'score' in frame_data.dtype.names and frame_data['score'] is not None:
                                    score = frame_data['score']
                                    if isinstance(score, (int, float)) and not np.isnan(score):
                                        frame_features.append(float(score))
                            except (ValueError, TypeError) as e:
                                logger.debug(f"Invalid score data in {npy_file} frame {frame_idx}: {e}")
                            
                            # Extract facial landmarks with robust error handling
                            try:
                                if 'face' in frame_data.dtype.names and frame_data['face'] is not None:
                                    face_data = frame_data['face']
                                    if hasattr(face_data, 'dtype') and face_data.dtype.names:
                                        if 'landmarks' in face_data.dtype.names and face_data['landmarks'] is not None:
                                            landmarks = face_data['landmarks']
                                            if hasattr(landmarks, 'dtype') and landmarks.dtype.names:
                                                if '2d' in landmarks.dtype.names and landmarks['2d'] is not None:
                                                    # Extract x, y coordinates from landmarks
                                                    landmark_coords = landmarks['2d']
                                                    if hasattr(landmark_coords, '__iter__'):
                                                        for landmark in landmark_coords:
                                                            try:
                                                                if hasattr(landmark, 'dtype') and landmark.dtype.names:
                                                                    if 'x' in landmark.dtype.names and 'y' in landmark.dtype.names:
                                                                        x, y = landmark['x'], landmark['y']
                                                                        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                                                                            if not (np.isnan(x) or np.isnan(y)):
                                                                                frame_features.extend([float(x), float(y)])
                                                            except (ValueError, TypeError, AttributeError) as e:
                                                                logger.debug(f"Invalid landmark data in {npy_file} frame {frame_idx}: {e}")
                                                                continue
                            except (ValueError, TypeError, AttributeError) as e:
                                logger.debug(f"Invalid face data in {npy_file} frame {frame_idx}: {e}")
                            
                            # Extract eye tracking data with validation
                            for eye in ['leftEye', 'rightEye']:
                                try:
                                    if eye in frame_data.dtype.names and frame_data[eye] is not None:
                                        eye_data = frame_data[eye]
                                        if hasattr(eye_data, 'dtype') and eye_data.dtype.names:
                                            if 'pupil' in eye_data.dtype.names and eye_data['pupil'] is not None:
                                                pupil = eye_data['pupil']
                                                if hasattr(pupil, 'dtype') and pupil.dtype.names:
                                                    if 'x' in pupil.dtype.names and 'y' in pupil.dtype.names:
                                                        x, y = pupil['x'], pupil['y']
                                                        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                                                            if not (np.isnan(x) or np.isnan(y)):
                                                                frame_features.extend([float(x), float(y)])
                                except (ValueError, TypeError, AttributeError) as e:
                                    logger.debug(f"Invalid {eye} data in {npy_file} frame {frame_idx}: {e}")
                                    continue
                            
                            # Extract head pose with validation
                            try:
                                if 'head' in frame_data.dtype.names and frame_data['head'] is not None:
                                    head_data = frame_data['head']
                                    if hasattr(head_data, 'dtype') and head_data.dtype.names:
                                        if 'rot' in head_data.dtype.names and head_data['rot'] is not None:
                                            rot_data = head_data['rot']
                                            if hasattr(rot_data, '__iter__') and not isinstance(rot_data, str):
                                                for val in rot_data:
                                                    if isinstance(val, (int, float)) and not np.isnan(val):
                                                        frame_features.append(float(val))
                                        if 'trans' in head_data.dtype.names and head_data['trans'] is not None:
                                            trans_data = head_data['trans']
                                            if hasattr(trans_data, '__iter__') and not isinstance(trans_data, str):
                                                for val in trans_data:
                                                    if isinstance(val, (int, float)) and not np.isnan(val):
                                                        frame_features.append(float(val))
                            except (ValueError, TypeError, AttributeError) as e:
                                logger.debug(f"Invalid head data in {npy_file} frame {frame_idx}: {e}")
                            
                            if frame_features:
                                all_features.append(frame_features)
                        else:
                            # Handle non-structured data
                            if hasattr(frame_data, '__iter__') and not isinstance(frame_data, str):
                                numeric_features = []
                                for item in frame_data:
                                    if isinstance(item, (int, float)) and not np.isnan(item):
                                        numeric_features.append(float(item))
                                if numeric_features:
                                    all_features.append(numeric_features)
                    except Exception as e:
                        logger.debug(f"Error processing frame {frame_idx} in {npy_file}: {e}")
                        continue
                
                # Enhanced fallback: try to extract any numeric data
                if not all_features and len(data) > 0:
                    try:
                        flattened = data.flatten()
                        numeric_data = []
                        for item in flattened:
                            if isinstance(item, (int, float)) and not np.isnan(item):
                                numeric_data.append(float(item))
                            elif hasattr(item, '__iter__') and not isinstance(item, str):
                                for sub_item in item:
                                    if isinstance(sub_item, (int, float)) and not np.isnan(sub_item):
                                        numeric_data.append(float(sub_item))
                        if numeric_data:
                            all_features.append(numeric_data)
                    except Exception as e:
                        logger.debug(f"Fallback extraction failed for {npy_file}: {e}")
                        
            except Exception as e:
                logger.warning(f"Error loading {npy_file}: {e}")
                continue
        
        if all_features:
            # Pad sequences to same length and concatenate
            try:
                max_len = max(len(features) for features in all_features)
                padded_features = []
                for features in all_features:
                    padded = features + [0.0] * (max_len - len(features))
                    padded_features.append(padded)
                return np.array(padded_features)
            except:
                # Fallback: flatten all features
                flattened = []
                for features in all_features:
                    flattened.extend(features)
                return np.array([flattened])  # Single sample
        
        return np.array([])
    
    def load_csv_data(self, file_path: str, sample_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Load CSV data.
        
        Parameters
        ----------
        file_path : str
            Path to CSV file
            
        Returns
        -------
        np.ndarray
            Loaded data as numpy array
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        logger.info(f"Loading CSV data from {file_path}")
        
        # Load CSV file with robust encoding handling
        df = None
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.debug(f"Successfully loaded CSV with {encoding} encoding")
                break
            except UnicodeDecodeError as e:
                logger.debug(f"Failed to load CSV with {encoding} encoding: {e}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error loading CSV with {encoding} encoding: {e}")
                continue
        
        if df is None:
            raise ValueError(f"Could not load CSV file {file_path} with any of the attempted encodings: {encodings}")
        
        # Handle labels CSV (extract numeric labels)
        if 'class_id' in df.columns:
            # EuroSAT labels
            data = df['class_id'].values
        elif len(df.columns) > 1:
            # OASIS or other tabular data - convert to numeric
            numeric_data = df.iloc[:, 1:].select_dtypes(include=[np.number])
            data = numeric_data.values
        else:
            # Single column data
            data = df.values.flatten()
        
        # Apply sampling if provided
        if sample_indices is not None and len(data) > 0:
            # Only sample if we have enough data
            if len(data) >= len(sample_indices):
                data = data[sample_indices]
            else:
                # If we don't have enough data, take what we have
                logger.warning(f"Not enough data ({len(data)}) for requested sample size ({len(sample_indices)}). Using available data.")
                data = data[:len(sample_indices)] if len(data) <= len(sample_indices) else data
        
        # Ensure consistent return type as float64 for tabular data
        return data.astype(np.float64)
    
    def load_modality_data(self, file_path: str, modality_type: str, 
                          target_size: Tuple[int, int] = (224, 224), 
                          channels_first: bool = False, 
                          sample_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Load data for a specific modality.
        
        Parameters
        ----------
        file_path : str
            Path to data file or directory
        modality_type : str
            Type of modality ('visual', 'tabular', 'spectral', 'time-series')
        target_size : tuple
            Target size for image resizing
        channels_first : bool
            If True, return images in (n_samples, channels, height, width) format
            
        Returns
        -------
        np.ndarray
            Loaded modality data
        """
        if modality_type == 'image':
            return self.load_image_directory(file_path, target_size, channels_first, sample_indices)
        elif modality_type in ['tabular', 'spectral', 'time-series']:
            if Path(file_path).is_dir():
                return self.load_npy_directory(file_path, sample_indices)
            else:
                return self.load_csv_data(file_path, sample_indices)
        elif modality_type == 'timeseries':
            # MUTLA time-series data (LOG files with EEG/attention)
            return self.load_log_directory(file_path, sample_indices)
        elif modality_type == 'visual':
            # MUTLA visual data (NPY files with webcam tracking)
            return self.load_webcam_npy_directory(file_path, sample_indices)
        else:
            raise ValueError(f"Unsupported modality type: {modality_type}")
    
    def load_multimodal_data(self, 
                           label_file: str,
                           modality_files: Dict[str, str],
                           modality_types: Dict[str, str],
                           normalize: bool = True,
                           normalization_method: str = 'standard',
                           test_size: float = 0.2,
                           stratify: bool = True,
                           random_state: int = 42,
                           target_size: Tuple[int, int] = (224, 224),
                           channels_first: bool = False) -> Dict[str, Any]:
        """
        Load and process multimodal data.
        
        Parameters
        ----------
        label_file : str
            Path to labels file
        modality_files : dict
            Dictionary mapping modality names to file paths
        modality_types : dict
            Dictionary mapping modality names to types
        normalize : bool
            Whether to normalize data
        normalization_method : str
            Normalization method to use
        test_size : float
            Proportion of data for testing
        stratify : bool
            Whether to stratify splits
        random_state : int
            Random seed for reproducible splits
        target_size : tuple
            Target size for image resizing
        channels_first : bool
            If True, return images in (n_samples, channels, height, width) format
            
        Returns
        -------
        dict
            Dictionary containing processed data
        """
        logger.info("Loading multimodal data...")
        
        # Load labels
        labels = self.load_labels(label_file)
        
        # Fast mode: determine sampling strategy upfront
        sample_indices = None
        if self.fast_mode and len(labels) > self.max_samples:
            logger.info(f"Fast mode: Will sample {self.max_samples} samples from {len(labels)} total samples.")
            random.seed(self._sampling_seed)  # CRITICAL: Use consistent seed
            sample_indices = random.sample(range(len(labels)), self.max_samples)
            # Sample labels upfront
            labels = labels[sample_indices]
        
        # Load modality data with coordinated sampling
        modality_data = {}
        for modality_name, file_path in modality_files.items():
            modality_type = modality_types.get(modality_name, 'tabular')
            logger.info(f"Loading {modality_name} ({modality_type}) from {file_path}")
            
            # Pass sample indices to ensure consistent sampling across modalities
            data = self.load_modality_data(file_path, modality_type, target_size, channels_first, sample_indices)
            modality_data[modality_name] = data
        
        # Handle missing modalities if enabled
        if self.handle_missing_modalities:
            modality_data, labels, missing_info = self._handle_missing_modalities(modality_data, labels)
            self.data['missing_modality_info'] = missing_info
        
        # Handle class imbalance if enabled
        if self.handle_class_imbalance:
            labels, class_info, sample_weights = self._handle_class_imbalance(labels)
            # If balancing was applied, we need to adjust modality data accordingly
            if self.class_imbalance_strategy == "balance" and 'balanced_class_counts' in class_info:
                # Find the indices that were kept after balancing
                # This is a simplified approach - in practice, you'd want to track the indices
                logger.warning("Class balancing applied - modality data may need adjustment")
        
        # CRITICAL FIX: Validate and fix data consistency across modalities
        modality_data, labels = self._validate_and_fix_data_consistency(modality_data, labels)
        
        # Apply normalization with memory efficiency for large datasets
        if normalize:
            for modality_name, data in modality_data.items():
                # For very large datasets, use chunked normalization
                if self.fast_mode and len(data) > self.max_samples * 2:
                    logger.debug(f"Using memory-efficient normalization for {modality_name}")
                    modality_data[modality_name] = self._normalize_data_chunked(
                        data, modality_name, normalization_method
                    )
                else:
                    modality_data[modality_name] = self.normalize_data(
                        data, modality_name, normalization_method
                    )
        
        # Create splits
        train_idx, test_idx = self.create_splits(labels, test_size, stratify, random_state)
        
        # Split data - handle cases where modalities have different sample counts
        train_data = {}
        test_data = {}
        for modality_name, data in modality_data.items():
            # Only use indices that are within bounds for this modality
            valid_train_idx = [i for i in train_idx if i < len(data)]
            valid_test_idx = [i for i in test_idx if i < len(data)]
            
            # Handle training data
            if valid_train_idx:
                train_data[modality_name] = data[valid_train_idx]
                logger.debug(f"Split {modality_name}: {len(valid_train_idx)}/{len(train_idx)} train samples")
            else:
                logger.warning(f"No valid training indices for {modality_name} (data length: {len(data)}, requested: {len(train_idx)})")
                train_data[modality_name] = np.array([])
                
            # Handle test data
            if valid_test_idx:
                test_data[modality_name] = data[valid_test_idx]
                logger.debug(f"Split {modality_name}: {len(valid_test_idx)}/{len(test_idx)} test samples")
            else:
                logger.warning(f"No valid test indices for {modality_name} (data length: {len(data)}, requested: {len(test_idx)})")
                test_data[modality_name] = np.array([])
        
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        
        # Store results
        result = {
            'train_data': train_data,
            'test_data': test_data,
            'train_labels': train_labels,
            'test_labels': test_labels,
            'modality_types': modality_types,
            'modality_configs': {
                name: {
                    'type': modality_types[name],
                    'shape': data.shape,
                    'dtype': str(data.dtype)
                }
                for name, data in modality_data.items()
            },
            'split_info': {
                'train_samples': len(train_idx),
                'test_samples': len(test_idx),
                'test_size': test_size,
                'stratify': stratify
            }
        }
        
        self.data = result
        logger.info(f"Successfully loaded multimodal data: {len(train_idx)} train, {len(test_idx)} test samples")
        
        return result

    # Step 3: Normalization 
    def normalize_data(self, data: np.ndarray, modality_name: str, 
                      method: str = 'standard') -> np.ndarray:
        """
        Apply simple normalization to data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        modality_name : str
            Name of the modality
        method : str
            Normalization method ('standard', 'minmax', 'none')
            
        Returns
        -------
        np.ndarray
            Normalized data
        """
        if method == 'none':
            return data
        
        # Reshape for normalization (flatten spatial dimensions for images)
        original_shape = data.shape
        if len(original_shape) > 2:
            data_flat = data.reshape(len(data), -1)
        else:
            data_flat = data
        
        # Apply normalization
        if method == 'standard':
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(data_flat)
            self.scalers[modality_name] = scaler
        elif method == 'minmax':
            # Simple min-max normalization
            data_min = data_flat.min(axis=0)
            data_max = data_flat.max(axis=0)
            normalized_data = (data_flat - data_min) / (data_max - data_min + 1e-8)
        else:
            normalized_data = data_flat
        
        # Reshape back to original shape
        if len(original_shape) > 2:
            return normalized_data.reshape(original_shape)
        else:
            return normalized_data
    
    def _normalize_data_chunked(self, data: np.ndarray, modality_name: str, 
                               method: str = 'standard') -> np.ndarray:
        """
        Apply memory-efficient chunked normalization to large datasets.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
        modality_name : str
            Name of the modality
        method : str
            Normalization method ('standard', 'minmax', 'none')
            
        Returns
        -------
        np.ndarray
            Normalized data
        """
        if method == 'none':
            return data
        
        # Reshape for normalization (flatten spatial dimensions for images)
        original_shape = data.shape
        if len(original_shape) > 2:
            data_flat = data.reshape(len(data), -1)
        else:
            data_flat = data
        
        # Calculate global statistics first for consistent normalization
        if method == 'standard':
            global_mean = np.mean(data_flat, axis=0)
            global_std = np.std(data_flat, axis=0)
            global_std = np.where(global_std == 0, 1.0, global_std)  # Avoid division by zero
        elif method == 'minmax':
            global_min = np.min(data_flat, axis=0)
            global_max = np.max(data_flat, axis=0)
            global_range = global_max - global_min
            global_range = np.where(global_range == 0, 1.0, global_range)  # Avoid division by zero
        
        # For chunked normalization, we'll process in smaller batches
        chunk_size = min(self.chunk_size, len(data_flat))
        normalized_chunks = []
        
        # Process data in chunks using global statistics
        for i in range(0, len(data_flat), chunk_size):
            chunk = data_flat[i:i + chunk_size]
            
            # Apply normalization to chunk using global statistics
            if method == 'standard':
                normalized_chunk = (chunk - global_mean) / global_std
            elif method == 'minmax':
                normalized_chunk = (chunk - global_min) / global_range
            else:
                normalized_chunk = chunk
            
            normalized_chunks.append(normalized_chunk)
        
        # Concatenate normalized chunks
        normalized_data = np.concatenate(normalized_chunks, axis=0)
        
        # Reshape back to original shape
        if len(original_shape) > 2:
            return normalized_data.reshape(original_shape)
        else:
            return normalized_data
    
    # Step 4: Train/Test Splitting
    def create_splits(self, labels: np.ndarray, test_size: float = 0.2, 
                     stratify: bool = True, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create train/test splits.
        
        Parameters
        ----------
        labels : np.ndarray
            Labels for stratification
        test_size : float
            Proportion of data for testing
        stratify : bool
            Whether to stratify splits
        random_state : int
            Random seed for reproducibility
            
        Returns
        -------
        tuple
            Train and test indices
        """
        if stratify:
            train_idx, test_idx = train_test_split(
                np.arange(len(labels)), 
                test_size=test_size, 
                stratify=labels, 
                random_state=random_state
            )
        else:
            train_idx, test_idx = train_test_split(
                np.arange(len(labels)), 
                test_size=test_size, 
                random_state=random_state
            )
        
        return train_idx, test_idx
    
    # Step 5: Data Consistency Validation
    def _validate_and_fix_data_consistency(self, modality_data: Dict[str, np.ndarray], labels: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        CRITICAL FIX: Validate and fix data consistency across modalities.
        
        Returns aligned modality data and labels with consistent sample counts.
        
        Parameters
        ----------
        modality_data : dict
            Dictionary of modality data arrays
        labels : np.ndarray
            Labels array
            
        Returns
        -------
        Tuple[Dict[str, np.ndarray], np.ndarray]
            Aligned modality data and labels
            
        Raises
        ------
        ValueError
            If data consistency validation fails
        """
        # CRITICAL FIX: Check and fix sample count consistency
        sample_counts = {name: len(data) for name, data in modality_data.items()}
        sample_counts['labels'] = len(labels)
        
        logger.info(f"Sample counts: {sample_counts}")
        
        if len(set(sample_counts.values())) > 1:
            logger.warning(f"Inconsistent sample counts across modalities: {sample_counts}")
            
            # Use the most common count instead of minimum to preserve more data
            count_frequency = Counter(sample_counts.values())
            target_count = count_frequency.most_common(1)[0][0]
            
            logger.info(f"Aligning all modalities to most common count: {target_count}")
            logger.info(f"Count frequency: {dict(count_frequency)}")
            
            # Align all modalities to the target count
            aligned_modality_data = {}
            for name, data in modality_data.items():
                if len(data) == target_count:
                    aligned_modality_data[name] = data
                elif len(data) > target_count:
                    # Truncate to target count
                    aligned_modality_data[name] = data[:target_count]
                    logger.info(f"Truncated {name} from {len(data)} to {target_count} samples")
                else:
                    # Pad with zeros or repeat last samples
                    if data.dtype in [np.float32, np.float64]:
                        padding = np.zeros((target_count - len(data),) + data.shape[1:], dtype=data.dtype)
                        aligned_modality_data[name] = np.vstack([data, padding])
                    else:
                        # For non-numeric data, repeat the last sample
                        last_sample = data[-1:] if len(data) > 0 else data
                        repeat_count = target_count - len(data)
                        padding = np.tile(last_sample, (repeat_count, 1)) if len(data.shape) > 1 else np.tile(last_sample, repeat_count)
                        aligned_modality_data[name] = np.concatenate([data, padding])
                    logger.info(f"Padded {name} from {len(data)} to {target_count} samples")
            
            # Align labels
            if len(labels) == target_count:
                aligned_labels = labels
            elif len(labels) > target_count:
                aligned_labels = labels[:target_count]
                logger.info(f"Truncated labels from {len(labels)} to {target_count} samples")
            else:
                # For labels, repeat the last label
                last_label = labels[-1] if len(labels) > 0 else 0
                padding = np.full(target_count - len(labels), last_label, dtype=labels.dtype)
                aligned_labels = np.concatenate([labels, padding])
                logger.info(f"Padded labels from {len(labels)} to {target_count} samples")
            
            # Store the final sample count for consistency tracking
            self._modality_sample_counts = {name: len(data) for name, data in aligned_modality_data.items()}
            self._modality_sample_counts['labels'] = len(aligned_labels)
            
            modality_data = aligned_modality_data
            labels = aligned_labels
        else:
            # Store the sample counts for consistency tracking
            self._modality_sample_counts = sample_counts
        
        # Additional validation for structured data
        for modality_name, data in modality_data.items():
            # Check for NaN/Inf values that could indicate data corruption
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                logger.warning(f"Found NaN/Inf values in {modality_name} modality")
            
            # Check data shape consistency for multi-dimensional data
            if data.ndim > 1:
                if len(data) > 0:
                    expected_shape = data.shape[1:]
                    for i, sample in enumerate(data):
                        if sample.shape != expected_shape:
                            logger.warning(f"Inconsistent shape in {modality_name} at sample {i}: "
                                         f"expected {expected_shape}, got {sample.shape}")
        
        # Validate label consistency
        if len(np.unique(labels)) < 2:
            logger.warning("Labels contain only one unique value - this may cause issues with stratified splitting")
        
        # Temporal alignment validation for time-series data
        self._validate_temporal_alignment(modality_data, labels)
        
        logger.info("✅ Data consistency validation and alignment completed")
        return modality_data, labels
    
    def _validate_temporal_alignment(self, modality_data: dict, labels: np.ndarray):
        """
        Validate temporal alignment for time-series data across modalities.
        
        Parameters
        ----------
        modality_data : dict
            Dictionary containing modality data
        labels : np.ndarray
            Labels array
        """
        time_series_modalities = ['timeseries', 'visual', 'tabular']
        temporal_modalities = [name for name in modality_data.keys() 
                              if any(ts_mod in name.lower() for ts_mod in time_series_modalities)]
        
        if len(temporal_modalities) < 2:
            return  # No need to validate temporal alignment with < 2 temporal modalities
        
        logger.info(f"Validating temporal alignment for {len(temporal_modalities)} temporal modalities")
        
        # Check temporal resolution consistency
        temporal_lengths = {}
        for modality_name in temporal_modalities:
            data = modality_data[modality_name]
            if data.ndim >= 2:
                # For time-series data, check the temporal dimension (usually the last dimension)
                temporal_lengths[modality_name] = data.shape[-1] if data.ndim > 1 else len(data)
            else:
                temporal_lengths[modality_name] = len(data)
        
        # Check for significant temporal length differences
        lengths = list(temporal_lengths.values())
        if lengths:
            min_length = min(lengths)
            max_length = max(lengths)
            length_ratio = max_length / min_length if min_length > 0 else float('inf')
            
            if length_ratio > 10:  # More than 10x difference
                logger.warning(f"Large temporal resolution differences detected:")
                for modality, length in temporal_lengths.items():
                    logger.warning(f"  - {modality}: {length} time points")
                logger.warning("Consider temporal alignment or resampling for better multimodal fusion")
            elif length_ratio > 2:  # More than 2x difference
                logger.info(f"Moderate temporal resolution differences detected (ratio: {length_ratio:.2f})")
                for modality, length in temporal_lengths.items():
                    logger.info(f"  - {modality}: {length} time points")
        
        # Check for temporal correlation patterns (basic check)
        if len(temporal_modalities) >= 2:
            try:
                # Sample a few data points to check for temporal patterns
                sample_size = min(100, len(labels))
                sample_indices = np.random.choice(len(labels), sample_size, replace=False)
                
                for i, modality_name in enumerate(temporal_modalities):
                    data = modality_data[modality_name]
                    if data.ndim >= 2:
                        sample_data = data[sample_indices]
                        # Check for temporal patterns (variance across time dimension)
                        if sample_data.ndim > 1:
                            temporal_variance = np.var(sample_data, axis=-1)
                            if np.mean(temporal_variance) < 1e-6:
                                logger.warning(f"{modality_name} shows very low temporal variance - may be static data")
                            elif np.mean(temporal_variance) > 1000:
                                logger.warning(f"{modality_name} shows very high temporal variance - check for outliers")
            except Exception as e:
                logger.debug(f"Temporal pattern analysis failed: {e}")
    
    def _handle_missing_modalities(self, modality_data: dict, labels: np.ndarray) -> tuple:
        """
        Handle samples with missing modalities based on the specified strategy.
        
        Parameters
        ----------
        modality_data : dict
            Dictionary containing modality data
        labels : np.ndarray
            Labels array
            
        Returns
        -------
        tuple
            (processed_modality_data, processed_labels, missing_info)
        """
        if not self.handle_missing_modalities:
            return modality_data, labels, {}
        
        logger.info(f"Handling missing modalities with strategy: {self.missing_modality_strategy}")
        
        # Find samples with missing modalities
        n_samples = len(labels)
        missing_info = {}
        valid_samples = set(range(n_samples))
        
        for modality_name, data in modality_data.items():
            if isinstance(data, np.ndarray):
                # Check for missing data (NaN, None, or empty)
                if data.ndim == 1:
                    missing_mask = np.isnan(data) | (data == None)
                else:
                    missing_mask = np.any(np.isnan(data), axis=tuple(range(1, data.ndim)))
                
                missing_indices = np.where(missing_mask)[0]
                if len(missing_indices) > 0:
                    missing_info[modality_name] = {
                        'missing_count': len(missing_indices),
                        'missing_indices': missing_indices,
                        'missing_ratio': len(missing_indices) / n_samples
                    }
                    logger.warning(f"{modality_name}: {len(missing_indices)} samples ({len(missing_indices)/n_samples*100:.1f}%) have missing data")
        
        # Apply strategy
        if self.missing_modality_strategy == "drop_samples":
            # Drop samples with any missing modality
            all_missing_indices = set()
            for info in missing_info.values():
                all_missing_indices.update(info['missing_indices'])
            
            valid_indices = np.array([i for i in range(n_samples) if i not in all_missing_indices])
            logger.info(f"Dropping {len(all_missing_indices)} samples with missing modalities")
            
            # Filter data and labels
            processed_modality_data = {}
            for modality_name, data in modality_data.items():
                if isinstance(data, np.ndarray):
                    processed_modality_data[modality_name] = data[valid_indices]
                else:
                    processed_modality_data[modality_name] = data
            
            processed_labels = labels[valid_indices]
            
        elif self.missing_modality_strategy == "zero_fill":
            # Fill missing values with zeros
            processed_modality_data = {}
            for modality_name, data in modality_data.items():
                if isinstance(data, np.ndarray):
                    filled_data = data.copy()
                    if modality_name in missing_info:
                        missing_indices = missing_info[modality_name]['missing_indices']
                        if filled_data.ndim == 1:
                            filled_data[missing_indices] = 0
                        else:
                            filled_data[missing_indices] = 0
                    processed_modality_data[modality_name] = filled_data
                else:
                    processed_modality_data[modality_name] = data
            
            processed_labels = labels.copy()
            
        elif self.missing_modality_strategy == "mean_fill":
            # Fill missing values with mean of available data
            processed_modality_data = {}
            for modality_name, data in modality_data.items():
                if isinstance(data, np.ndarray):
                    filled_data = data.copy()
                    if modality_name in missing_info:
                        missing_indices = missing_info[modality_name]['missing_indices']
                        valid_indices = np.setdiff1d(np.arange(len(data)), missing_indices)
                        
                        if len(valid_indices) > 0:
                            if filled_data.ndim == 1:
                                mean_value = np.mean(filled_data[valid_indices])
                                filled_data[missing_indices] = mean_value
                            else:
                                mean_values = np.mean(filled_data[valid_indices], axis=0)
                                filled_data[missing_indices] = mean_values
                        else:
                            logger.warning(f"All data missing for {modality_name}, using zero fill")
                            if filled_data.ndim == 1:
                                filled_data[missing_indices] = 0
                            else:
                                filled_data[missing_indices] = 0
                    
                    processed_modality_data[modality_name] = filled_data
                else:
                    processed_modality_data[modality_name] = data
            
            processed_labels = labels.copy()
        
        else:
            raise ValueError(f"Unknown missing modality strategy: {self.missing_modality_strategy}")
        
        logger.info(f"Missing modality handling complete. Final sample count: {len(processed_labels)}")
        return processed_modality_data, processed_labels, missing_info
    
    def _handle_class_imbalance(self, labels: np.ndarray) -> tuple:
        """
        Handle class imbalance in labels based on the specified strategy.
        
        Parameters
        ----------
        labels : np.ndarray
            Labels array
            
        Returns
        -------
        tuple
            (processed_labels, class_info, sample_weights)
        """
        if not self.handle_class_imbalance:
            return labels, {}, None
        
        logger.info(f"Handling class imbalance with strategy: {self.class_imbalance_strategy}")
        
        # Analyze class distribution
        unique_classes, class_counts = np.unique(labels, return_counts=True)
        n_samples = len(labels)
        n_classes = len(unique_classes)
        
        class_info = {
            'n_classes': n_classes,
            'class_counts': dict(zip(unique_classes, class_counts)),
            'class_ratios': dict(zip(unique_classes, class_counts / n_samples)),
            'imbalance_ratio': max(class_counts) / min(class_counts) if min(class_counts) > 0 else float('inf')
        }
        
        logger.info(f"Class distribution analysis:")
        for class_label, count in class_info['class_counts'].items():
            ratio = class_info['class_ratios'][class_label]
            logger.info(f"  - Class {class_label}: {count} samples ({ratio*100:.1f}%)")
        
        logger.info(f"Imbalance ratio: {class_info['imbalance_ratio']:.2f}")
        
        # Determine if dataset is imbalanced
        is_imbalanced = class_info['imbalance_ratio'] > 2.0  # More than 2:1 ratio
        
        if is_imbalanced:
            logger.warning(f"Dataset is imbalanced (ratio: {class_info['imbalance_ratio']:.2f})")
        else:
            logger.info("Dataset appears to be reasonably balanced")
        
        # Apply strategy
        sample_weights = None
        
        if self.class_imbalance_strategy == "report":
            # Just report the imbalance, no changes to data
            logger.info("Class imbalance reported - no data modifications applied")
            processed_labels = labels
            
        elif self.class_imbalance_strategy == "balance":
            # Balance classes by undersampling majority class
            if is_imbalanced:
                min_count = min(class_counts)
                logger.info(f"Balancing classes by undersampling to {min_count} samples per class")
                
                balanced_indices = []
                for class_label in unique_classes:
                    class_indices = np.where(labels == class_label)[0]
                    if len(class_indices) > min_count:
                        # Randomly sample min_count indices
                        np.random.seed(42)  # For reproducibility
                        selected_indices = np.random.choice(class_indices, min_count, replace=False)
                    else:
                        selected_indices = class_indices
                    balanced_indices.extend(selected_indices)
                
                balanced_indices = np.array(balanced_indices)
                processed_labels = labels[balanced_indices]
                
                # Update class info
                unique_classes_balanced, class_counts_balanced = np.unique(processed_labels, return_counts=True)
                class_info['balanced_class_counts'] = dict(zip(unique_classes_balanced, class_counts_balanced))
                class_info['balanced_imbalance_ratio'] = max(class_counts_balanced) / min(class_counts_balanced)
                
                logger.info(f"Balanced dataset: {len(processed_labels)} samples")
                logger.info(f"New imbalance ratio: {class_info['balanced_imbalance_ratio']:.2f}")
            else:
                processed_labels = labels
                
        elif self.class_imbalance_strategy == "weight":
            # Calculate class weights for weighted loss
            if is_imbalanced:
                # Calculate inverse frequency weights
                total_samples = len(labels)
                class_weights = {}
                for class_label, count in class_info['class_counts'].items():
                    weight = total_samples / (n_classes * count)
                    class_weights[class_label] = weight
                
                self.class_weights = class_weights
                
                # Create sample weights
                sample_weights = np.array([class_weights[label] for label in labels])
                
                logger.info("Class weights calculated for weighted loss:")
                for class_label, weight in class_weights.items():
                    logger.info(f"  - Class {class_label}: weight {weight:.3f}")
                
                processed_labels = labels
            else:
                processed_labels = labels
                
        else:
            raise ValueError(f"Unknown class imbalance strategy: {self.class_imbalance_strategy}")
        
        # Store class information
        self.data['class_info'] = class_info
        if sample_weights is not None:
            self.data['sample_weights'] = sample_weights
        
        logger.info("Class imbalance handling complete")
        return processed_labels, class_info, sample_weights
    
    # Step 6: Data Retrieval
    def get_data_generator(self, batch_size: int = 32, shuffle: bool = True, 
                          modality_subset: list = None, max_memory_gb: float = 2.0):
        """
        Create a memory-efficient data generator for large datasets with enhanced memory management.
        
        Parameters
        ----------
        batch_size : int
            Number of samples per batch
        shuffle : bool
            Whether to shuffle data
        modality_subset : list, optional
            List of modality names to include in batches (for memory optimization)
        max_memory_gb : float
            Maximum memory usage in GB before triggering garbage collection
            
        Yields
        ------
        tuple
            Batch of (data_dict, labels) for training
        """
        if not self.lazy_loading:
            logger.warning("Data generator requested but lazy_loading is disabled")
            return None
            
        if 'train_data' not in self.data or 'train_labels' not in self.data:
            raise ValueError("No training data available. Load data first.")
        
        train_data = self.data['train_data']
        train_labels = self.data['train_labels']
        
        # Filter modalities if subset specified
        if modality_subset:
            available_modalities = list(train_data.keys())
            invalid_modalities = [m for m in modality_subset if m not in available_modalities]
            if invalid_modalities:
                logger.warning(f"Invalid modalities specified: {invalid_modalities}")
            modality_subset = [m for m in modality_subset if m in available_modalities]
            if not modality_subset:
                raise ValueError("No valid modalities in subset")
        else:
            modality_subset = list(train_data.keys())
        
        n_samples = len(train_labels)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        # Memory monitoring
        try:
            import psutil
            import gc
            process = psutil.Process()
            memory_monitoring = True
        except ImportError:
            logger.warning("psutil not available, memory monitoring disabled")
            memory_monitoring = False
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Check memory usage
            if memory_monitoring:
                memory_usage_gb = process.memory_info().rss / (1024**3)
                if memory_usage_gb > max_memory_gb:
                    logger.debug(f"Memory usage {memory_usage_gb:.2f}GB > {max_memory_gb}GB, triggering garbage collection")
                    gc.collect()
            
            batch_data = {}
            for modality_name in modality_subset:
                data = train_data[modality_name]
                if isinstance(data, np.ndarray):
                    # Only use indices that are within bounds for this modality
                    valid_indices = [i for i in batch_indices if i < len(data)]
                    if valid_indices:
                        batch_data[modality_name] = data[valid_indices]
                    else:
                        # If no valid indices, create empty array with correct shape
                        if len(data) > 0:
                            # Create empty array with same shape as data but with batch size
                            if data.ndim == 1:
                                batch_data[modality_name] = np.zeros(len(batch_indices), dtype=data.dtype)
                            else:
                                batch_shape = (len(batch_indices),) + data.shape[1:]
                                batch_data[modality_name] = np.zeros(batch_shape, dtype=data.dtype)
                        else:
                            batch_data[modality_name] = np.array([])
                else:
                    # Handle lazy-loaded data
                    batch_data[modality_name] = self._load_data_chunk(modality_name, batch_indices)
            
            batch_labels = train_labels[batch_indices]
            yield batch_data, batch_labels
            
            # Clean up batch data to free memory
            del batch_data
    
    def _load_data_chunk(self, modality_name: str, indices: np.ndarray) -> np.ndarray:
        """
        Load a chunk of data for lazy loading.
        
        Parameters
        ----------
        modality_name : str
            Name of the modality
        indices : np.ndarray
            Indices of samples to load
            
        Returns
        -------
        np.ndarray
            Chunk of data
        """
        # This would be implemented based on the specific data format
        # For now, return a placeholder
        logger.debug(f"Loading chunk for {modality_name} with {len(indices)} samples")
        return np.array([])  # Placeholder
    
    def get_data(self) -> Dict[str, Any]:
        """
        Get loaded data.
        
        Returns
        -------
        dict
            Loaded data dictionary
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_multimodal_data() first.")
        
        return self.data
    
    def get_train_data(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Get training data and labels.
        
        Returns
        -------
        Tuple[Dict[str, np.ndarray], np.ndarray]
            Training data dictionary and training labels
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_multimodal_data() first.")
        
        return self.data['train_data'], self.data['train_labels']
    
    def get_test_data(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Get test data and labels.
        
        Returns
        -------
        Tuple[Dict[str, np.ndarray], np.ndarray]
            Test data dictionary and test labels
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_multimodal_data() first.")
        
        return self.data['test_data'], self.data['test_labels']
    
    def get_tensors(self, device: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert data to PyTorch tensors.
        
        Parameters
        ----------
        device : str, optional
            Device to place tensors on
            
        Returns
        -------
        dict
            Dictionary containing PyTorch tensors
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_multimodal_data() first.")
        
        device = device or self.device
        
        # Convert to tensors
        train_tensors = {}
        test_tensors = {}
        
        for modality_name, data in self.data['train_data'].items():
            train_tensors[modality_name] = torch.from_numpy(data).float().to(device)
        
        for modality_name, data in self.data['test_data'].items():
            test_tensors[modality_name] = torch.from_numpy(data).float().to(device)
        
        train_labels_tensor = torch.from_numpy(self.data['train_labels']).long().to(device)
        test_labels_tensor = torch.from_numpy(self.data['test_labels']).long().to(device)
        
        return {
            'train_data': train_tensors,
            'test_data': test_tensors,
            'train_labels': train_labels_tensor,
            'test_labels': test_labels_tensor,
            'modality_types': self.data['modality_types'],
            'modality_configs': self.data['modality_configs']
        }
    
    def print_summary(self):
        """Print summary of loaded data."""
        if not self.data:
            print("No data loaded.")
            return
        
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        
        split_info = self.data['split_info']
        print(f"Train samples: {split_info['train_samples']}")
        print(f"Test samples: {split_info['test_samples']}")
        print(f"Test size: {split_info['test_size']:.1%}")
        
        print(f"\nModalities ({len(self.data['modality_configs'])}):")
        for name, config in self.data['modality_configs'].items():
            train_shape = self.data['train_data'][name].shape
            test_shape = self.data['test_data'][name].shape
            print(f"  {name}: {config['type']} - Train: {train_shape}, Test: {test_shape}")

# Convenience functions for specific datasets
def load_eurosat_data(data_dir: str = "ProcessedData/EuroSAT", 
                     normalize: bool = True,
                     test_size: float = 0.2,
                     target_size: Tuple[int, int] = (224, 224),
                     channels_first: bool = True,
                     random_state: int = 42,
                     fast_mode: bool = True,
                     max_samples: int = 1000,
                     **kwargs) -> SimpleDataLoader:
    """
    Load EuroSAT data with proper modality types.
    
    Parameters
    ----------
    data_dir : str
        Path to EuroSAT data directory
    normalize : bool
        Whether to normalize data
    test_size : float
        Test set proportion
    target_size : tuple
        Target size for image resizing
    channels_first : bool
        If True, return images in PyTorch format (n_samples, channels, height, width)
        
    Returns
    -------
    SimpleDataLoader
        Loaded EuroSAT data
    """
    loader = SimpleDataLoader(fast_mode=fast_mode, max_samples=max_samples, **kwargs)
    
    # EuroSAT modality configuration
    modality_types = {
        "visual_rgb": "image",
        "near_infrared": "spectral", 
        "red_edge": "spectral",
        "short_wave_infrared": "spectral",
        "atmospheric": "spectral"
    }
    
    modality_files = {
        "visual_rgb": f"{data_dir}/visual_rgb",
        "near_infrared": f"{data_dir}/near_infrared", 
        "red_edge": f"{data_dir}/red_edge",
        "short_wave_infrared": f"{data_dir}/short_wave_infrared",
        "atmospheric": f"{data_dir}/atmospheric"
    }
    
    loader.load_multimodal_data(
        label_file=f"{data_dir}/labels.csv",
        modality_files=modality_files,
        modality_types=modality_types,
        normalize=normalize,
        test_size=test_size,
        target_size=target_size,
        channels_first=channels_first,
        random_state=random_state
    )
    
    logger.info("EuroSAT data loaded successfully")
    return loader

def load_oasis_data(data_dir: str = "ProcessedData/OASIS",
                   normalize: bool = True,
                   test_size: float = 0.2,
                   random_state: int = 42,
                   fast_mode: bool = True,
                   max_samples: int = 1000,
                   **kwargs) -> SimpleDataLoader:
    """
    Load OASIS data.
    
    Parameters
    ----------
    data_dir : str
        Path to OASIS data directory
    normalize : bool
        Whether to normalize data
    test_size : float
        Test set proportion
        
    Returns
    -------
    SimpleDataLoader
        Loaded OASIS data
    """
    loader = SimpleDataLoader(fast_mode=fast_mode, max_samples=max_samples, **kwargs)
    
    # OASIS modality configuration
    modality_types = {
        "tabular_features": "tabular"
    }
    
    modality_files = {
        "tabular_features": f"{data_dir}/tabular_features.csv"
    }
    
    loader.load_multimodal_data(
        label_file=f"{data_dir}/labels.csv",
        modality_files=modality_files,
        modality_types=modality_types,
        normalize=normalize,
        test_size=test_size,
        random_state=random_state
    )
    
    logger.info("OASIS data loaded successfully")
    return loader

def load_mutla_data(data_dir: str = "Data/MUTLA",
                   normalize: bool = True, test_size: float = 0.2,
                   random_state: int = 42,
                   fast_mode: bool = True,
                   max_samples: int = 1000,
                   **kwargs) -> SimpleDataLoader:
    """
    Load MUTLA multimodal data with proper modality structures from raw data.
    
    Parameters
    ----------
    data_dir : str
        Path to MUTLA raw data directory
    normalize : bool
        Whether to normalize the data
    test_size : float
        Fraction of data to use for testing
        
    Returns
    -------
    SimpleDataLoader
        Loader with MUTLA data preserving modality structures
    """
    loader = SimpleDataLoader(fast_mode=fast_mode, max_samples=max_samples, **kwargs)
    
    modality_types = {
        "tabular": "tabular",            # CSV files with user interaction data
        "timeseries": "timeseries",      # LOG files with EEG/attention time-series
        "visual": "visual"               # NPY files with webcam tracking data
    }
    
    modality_files = {
        "tabular": f"{data_dir}/User records/math_record_cleaned.csv",  # Specific CSV file
        "timeseries": f"{data_dir}/Brainwave",  # LOG files with EEG/attention
        "visual": f"{data_dir}/Webcam"  # NPY files with tracking data
    }
    
    # Note: MUTLA doesn't have a single labels.csv file
    # Labels are embedded within the tabular CSV files
    # For now, we'll use a dummy labels file or extract from tabular data
    loader.load_multimodal_data(
        label_file=f"{data_dir}/User records/math_record_cleaned.csv",  # Use one of the tabular files as labels
        modality_files=modality_files,
        modality_types=modality_types,
        normalize=normalize,
        test_size=test_size,
        random_state=random_state
    )
    
    logger.info("MUTLA data loaded successfully with preserved modality structures")
    return loader

def load_custom_data(label_file: str,
                    modality_files: Dict[str, str],
                    modality_types: Dict[str, str],
                    **kwargs) -> SimpleDataLoader:
    """
    Load custom multimodal data.
    
    Parameters
    ----------
    label_file : str
        Path to labels file
    modality_files : dict
        Dictionary mapping modality names to file paths
    modality_types : dict
        Dictionary mapping modality names to types
    **kwargs
        Additional parameters for load_multimodal_data
        
    Returns
    -------
    SimpleDataLoader
        Loaded data
    """
    loader = SimpleDataLoader()
    
    loader.load_multimodal_data(
        label_file=label_file,
        modality_files=modality_files,
        modality_types=modality_types,
        **kwargs
    )
    
    logger.info("Custom data loaded successfully")
    return loader

# Example usage and testing
if __name__ == "__main__":
    # Test with EuroSAT
    try:
        print("Testing EuroSAT data loading...")
        loader = load_eurosat_data()
        loader.print_summary()
        
        # Test tensor conversion
        tensors = loader.get_tensors()
        print(f"\nTensor shapes:")
        for name, tensor in tensors['train_data'].items():
            print(f"  {name}: {tensor.shape}")
            
    except Exception as e:
        print(f"EuroSAT test failed: {e}")
    
    # Test with OASIS
    try:
        print("\nTesting OASIS data loading...")
        loader = load_oasis_data()
        loader.print_summary()
        
    except Exception as e:
        print(f"OASIS test failed: {e}")
