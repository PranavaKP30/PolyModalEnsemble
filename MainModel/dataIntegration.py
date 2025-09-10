"""
Stage 1: Data Integration
Simple multimodal data integration for file-based loading, processing, and saving.

Implements:
- File-based data loading (separate files for each modality and labels)
- Data processing and cleaning
- Data saving for next stage
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger("data_integration")

class SimpleDataLoader:
    """
    Simple data loader for file-based multimodal data.
    
    Handles:
    - Loading separate files for each modality and labels
    - Data processing and cleaning
    - Saving processed data for next stage
    """
    
    def __init__(self):
        # Data storage
        self.train_data = {}
        self.test_data = {}
        self.train_labels = None
        self.test_labels = None
        self.modality_configs = {}
        self.verbose = True
        
    def load_from_files(self, 
                       train_label_file: str,
                       test_label_file: str,
                       train_modality_files: Dict[str, str],
                       test_modality_files: Dict[str, str],
                       modality_types: Dict[str, str]):
        """
        Load data from separate files.
        
        Parameters
        ----------
        train_label_file : str
            Path to training labels file
        test_label_file : str
            Path to testing labels file
        train_modality_files : dict
            Dict mapping modality names to training data file paths
        test_modality_files : dict
            Dict mapping modality names to testing data file paths
        modality_types : dict
            Dict mapping modality names to data types ('text', 'image', 'tabular', 'audio')
        """
        if self.verbose:
            print("Loading data from files...")
        
        # Load labels
        self.train_labels = self._load_file(train_label_file)
        self.test_labels = self._load_file(test_label_file)
        
        # Load modality data
        for modality_name in train_modality_files.keys():
            if modality_name not in test_modality_files:
                raise ValueError(f"Modality '{modality_name}' missing from test files")
            
            train_file = train_modality_files[modality_name]
            test_file = test_modality_files[modality_name]
            data_type = modality_types.get(modality_name, 'tabular')
            
            # Load data
            train_data = self._load_file(train_file)
            test_data = self._load_file(test_file)
            
            # Store data
            self.train_data[modality_name] = train_data
            self.test_data[modality_name] = test_data
            
            # Store config
            self.modality_configs[modality_name] = {
                'data_type': data_type,
                'feature_dim': train_data.shape[1],
                'train_samples': train_data.shape[0],
                'test_samples': test_data.shape[0]
            }
            
            if self.verbose:
                print(f"  {modality_name}: {data_type} - Train: {train_data.shape}, Test: {test_data.shape}")
    
    def _load_file(self, file_path: str) -> np.ndarray:
        """Load data from file (supports .npy, .csv, .txt)."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix == '.npy':
            return np.load(file_path)
        elif file_path.suffix == '.csv':
            return pd.read_csv(file_path).values
        elif file_path.suffix == '.txt':
            return np.loadtxt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def process_data(self, 
                    handle_nan: str = 'fill_mean',
                    handle_inf: str = 'fill_max',
                    normalize: bool = False,
                    remove_outliers: bool = False,
                    outlier_std: float = 3.0):
        """
        Process and clean the loaded data.
        
        Parameters
        ----------
        handle_nan : str
            How to handle NaN values: 'fill_mean', 'fill_zero', 'drop'
        handle_inf : str
            How to handle Inf values: 'fill_max', 'fill_zero', 'drop'
        normalize : bool
            Whether to normalize data (z-score scaling)
        remove_outliers : bool
            Whether to remove outliers
        outlier_std : float
            Standard deviation threshold for outlier removal
        """
        if self.verbose:
            print("Processing data...")
        
        # Process training data
        for modality_name, data in self.train_data.items():
            self.train_data[modality_name] = self._process_modality_data(
                data, handle_nan, handle_inf, normalize, remove_outliers, outlier_std
            )
        
        # Process testing data
        for modality_name, data in self.test_data.items():
            self.test_data[modality_name] = self._process_modality_data(
                data, handle_nan, handle_inf, normalize, remove_outliers, outlier_std
            )
        
        # Process labels
        self.train_labels = self._process_labels(self.train_labels, handle_nan, handle_inf)
        self.test_labels = self._process_labels(self.test_labels, handle_nan, handle_inf)
        
        if self.verbose:
            print("  Data processing completed")
    
    def _process_modality_data(self, data: np.ndarray, handle_nan: str, handle_inf: str, 
                              normalize: bool, remove_outliers: bool, outlier_std: float) -> np.ndarray:
        """Process a single modality's data."""
        processed_data = data.copy()
        
        # Handle NaN values
        if handle_nan == 'fill_mean':
            if np.issubdtype(processed_data.dtype, np.floating):
                means = np.nanmean(processed_data, axis=0)
                inds = np.where(np.isnan(processed_data))
                if len(inds) == 2:  # 2D array
                    processed_data[inds] = np.take(means, inds[1])
        elif handle_nan == 'fill_zero':
            processed_data = np.nan_to_num(processed_data, nan=0.0)
        elif handle_nan == 'drop':
            if np.issubdtype(processed_data.dtype, np.floating):
                mask = ~np.isnan(processed_data).any(axis=1)
                processed_data = processed_data[mask]
        
        # Handle Inf values
        if handle_inf == 'fill_max':
            if np.issubdtype(processed_data.dtype, np.floating):
                maxs = np.nanmax(processed_data, axis=0)
                maxs = np.where(np.isinf(maxs), 1e6, maxs)
                inds = np.where(np.isinf(processed_data))
                if len(inds) == 2:  # 2D array
                    processed_data[inds] = np.take(maxs, inds[1])
        elif handle_inf == 'fill_zero':
            processed_data = np.where(np.isinf(processed_data), 0.0, processed_data)
        elif handle_inf == 'drop':
            if np.issubdtype(processed_data.dtype, np.floating):
                mask = ~np.isinf(processed_data).any(axis=1)
                processed_data = processed_data[mask]
        
        # Remove outliers
        if remove_outliers and np.issubdtype(processed_data.dtype, np.floating):
            mean = np.mean(processed_data, axis=0)
            std = np.std(processed_data, axis=0)
            mask = np.all(np.abs(processed_data - mean) < outlier_std * std, axis=1)
            processed_data = processed_data[mask]
        
        # Normalize
        if normalize and np.issubdtype(processed_data.dtype, np.floating):
            mean = np.mean(processed_data, axis=0)
            std = np.std(processed_data, axis=0)
            std[std == 0] = 1.0
            processed_data = (processed_data - mean) / std
        
        return processed_data
    
    def _process_labels(self, labels: np.ndarray, handle_nan: str, handle_inf: str) -> np.ndarray:
        """Process labels (simpler than modality data)."""
        processed_labels = labels.copy()
        
        # Handle NaN values
        if handle_nan == 'fill_zero':
            processed_labels = np.nan_to_num(processed_labels, nan=0.0)
        elif handle_nan == 'drop':
            if np.issubdtype(processed_labels.dtype, np.floating):
                mask = ~np.isnan(processed_labels)
                processed_labels = processed_labels[mask]
        
        # Handle Inf values
        if handle_inf == 'fill_zero':
            processed_labels = np.where(np.isinf(processed_labels), 0.0, processed_labels)
        elif handle_inf == 'drop':
            if np.issubdtype(processed_labels.dtype, np.floating):
                mask = ~np.isinf(processed_labels)
                processed_labels = processed_labels[mask]
        
        return processed_labels
    
    def get_processed_data(self):
        """
        Get processed data for use by next stage (memory only, no file saving).
        
        Returns
        -------
        dict
            Dictionary containing all processed data and metadata
        """
        if self.verbose:
            print("Preparing processed data for next stage...")
        
        # Create data package for Stage 2
        processed_data = {
            'train_data': self.train_data,
            'test_data': self.test_data,
            'train_labels': self.train_labels,
            'test_labels': self.test_labels,
            'modality_configs': self.modality_configs,
            'metadata': {
                'train_samples': len(self.train_labels),
                'test_samples': len(self.test_labels),
                'n_modalities': len(self.modality_configs)
            }
        }
        
        if self.verbose:
            print(f"  Prepared {len(self.modality_configs)} modalities")
            print(f"  Train samples: {len(self.train_labels)}")
            print(f"  Test samples: {len(self.test_labels)}")
        
        return processed_data
    
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data."""
        summary = {
            'modalities': {},
            'overall': {
                'train_samples': len(self.train_labels) if self.train_labels is not None else 0,
                'test_samples': len(self.test_labels) if self.test_labels is not None else 0,
                'n_modalities': len(self.modality_configs)
            }
        }
        
        for modality_name, config in self.modality_configs.items():
            summary['modalities'][modality_name] = {
                'data_type': config['data_type'],
                'feature_dim': config['feature_dim'],
                'train_shape': self.train_data[modality_name].shape if modality_name in self.train_data else None,
                'test_shape': self.test_data[modality_name].shape if modality_name in self.test_data else None
            }
        
        return summary
    
    def print_summary(self):
        """Print data summary."""
        summary = self.get_data_summary()
        
        print("Data Summary:")
        print(f"  Train samples: {summary['overall']['train_samples']}")
        print(f"  Test samples: {summary['overall']['test_samples']}")
        print(f"  Modalities: {summary['overall']['n_modalities']}")
        
        for modality_name, info in summary['modalities'].items():
            print(f"  {modality_name}: {info['data_type']} - {info['feature_dim']} features")
            print(f"    Train: {info['train_shape']}, Test: {info['test_shape']}")

# Convenience function for quick data loading
def load_and_process_data(train_label_file: str,
                         test_label_file: str,
                         train_modality_files: Dict[str, str],
                         test_modality_files: Dict[str, str],
                         modality_types: Dict[str, str],
                         **processing_kwargs) -> SimpleDataLoader:
    """
    Convenience function to load and process data in one step (memory only).
    
    Returns
    -------
    SimpleDataLoader
        Loaded and processed data loader
    """
    loader = SimpleDataLoader()
    loader.verbose = True
    
    # Load data
    loader.load_from_files(
        train_label_file, test_label_file,
        train_modality_files, test_modality_files,
        modality_types
    )
    
    # Process data
    loader.process_data(**processing_kwargs)
    
    return loader
