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
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

logger = logging.getLogger("data_integration")

class SimpleDataLoader:
    """
    Simple data loader for file-based multimodal data.
    
    Handles:
    - Loading separate files for each modality and labels
    - Data processing and cleaning
    - Saving processed data for next stage
    """
    
    def __init__(self, cache_dir: str = None, n_jobs: int = -1):
        # Data storage
        self.train_data = {}
        self.test_data = {}
        self.train_labels = None
        self.test_labels = None
        self.modality_configs = {}
        self.verbose = True
        
        # Feature reduction and preprocessing
        self.text_svd = None
        self.feature_reduction_applied = False
        self.scalers = {}
        self.feature_selectors = {}
        self.outlier_detectors = {}
        
        # Caching
        self.cache_dir = cache_dir or "cache"
        self.cache_enabled = cache_dir is not None
        
        # Parallel processing
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        
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
                    outlier_std: float = 3.0,
                    # Model-specific features only (universal features moved to preprocessing)
                    feature_selection: bool = False,
                    selection_method: str = 'mutual_info',  # 'mutual_info', 'f_classif', 'variance', 'model_based'
                    n_features_select: int = None,
                    cross_modal_features: bool = False,
                    polynomial_features: bool = False,
                    polynomial_degree: int = 2,
                    parallel_processing: bool = True):
        """
        Process and clean the loaded data with advanced preprocessing options.
        
        Parameters
        ----------
        handle_nan : str
            How to handle NaN values: 'fill_mean', 'fill_zero', 'drop'
        handle_inf : str
            How to handle Inf values: 'fill_max', 'fill_zero', 'drop'
        normalize : bool
            Whether to normalize data
        scaler_type : str
            Type of scaler: 'standard', 'robust', 'minmax', 'power', 'quantile'
        remove_outliers : bool
            Whether to remove outliers
        outlier_method : str
            Outlier detection method: 'isolation_forest', 'lof', 'std'
        outlier_std : float
            Standard deviation threshold for outlier removal (std method only)
        reduce_text_features : bool
            Whether to apply dimensionality reduction to text features
        reduction_method : str
            Dimensionality reduction method: 'svd', 'pca', 'fastica', 'random_projection'
        text_svd_components : int
            Number of components for dimensionality reduction
        feature_selection : bool
            Whether to apply feature selection
        selection_method : str
            Feature selection method: 'mutual_info', 'f_classif', 'variance', 'model_based'
        n_features_select : int
            Number of features to select (if None, auto-select)
        cross_modal_features : bool
            Whether to create cross-modal interaction features
        polynomial_features : bool
            Whether to create polynomial features
        polynomial_degree : int
            Degree of polynomial features
        parallel_processing : bool
            Whether to use parallel processing
        """
        if self.verbose:
            print("Processing data...")
        
        # Check cache first
        cache_key = self._get_cache_key(handle_nan, handle_inf, normalize, remove_outliers, outlier_std, 
                                      feature_selection, selection_method, n_features_select, cross_modal_features, 
                                      polynomial_features, polynomial_degree, parallel_processing)
        
        if self.cache_enabled and self._load_from_cache(cache_key):
            if self.verbose:
                print("  Data loaded from cache")
            return
        
        # Process training data (with parallel processing if enabled)
        if parallel_processing and self.n_jobs > 1:
            self._process_data_parallel(handle_nan, handle_inf, normalize, remove_outliers, outlier_std)
        else:
            self._process_data_sequential(handle_nan, handle_inf, normalize, remove_outliers, outlier_std)
        
        # Create cross-modal features if requested
        if cross_modal_features and len(self.train_data) > 1:
            self._create_cross_modal_features()
        
        # Create polynomial features if requested
        if polynomial_features:
            self._create_polynomial_features(polynomial_degree)
        
        # Apply feature selection if requested
        if feature_selection:
            self._apply_advanced_feature_selection(selection_method, n_features_select)
        
        # Process labels
        self.train_labels = self._process_labels(self.train_labels, handle_nan, handle_inf)
        self.test_labels = self._process_labels(self.test_labels, handle_nan, handle_inf)
        
        # Save to cache
        if self.cache_enabled:
            self._save_to_cache(cache_key)
        
        if self.verbose:
            print("  Data processing completed")
    
    # REMOVED: _apply_text_feature_reduction - moved to preprocessing pipeline
    
    def _process_modality_data(self, data: np.ndarray, handle_nan: str, handle_inf: str, 
                              normalize: bool, remove_outliers: bool, outlier_std: float, modality_name: str) -> np.ndarray:
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
        
        # Remove outliers using basic method only
        if remove_outliers and np.issubdtype(processed_data.dtype, np.floating):
            mean = np.mean(processed_data, axis=0)
            std = np.std(processed_data, axis=0)
            mask = np.all(np.abs(processed_data - mean) < outlier_std * std, axis=1)
            processed_data = processed_data[mask]
        
        # Normalize with basic StandardScaler only
        if normalize and np.issubdtype(processed_data.dtype, np.floating):
            scaler = StandardScaler()
            processed_data = scaler.fit_transform(processed_data)
            self.scalers[modality_name] = scaler
        
        return processed_data
                else:
                    # Fallback if no scaler found
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
    
    def _get_cache_key(self, *args) -> str:
        """Generate cache key from processing parameters."""
        key_string = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> bool:
        """Load processed data from cache."""
        cache_file = Path(self.cache_dir) / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                self.train_data = cached_data['train_data']
                self.test_data = cached_data['test_data']
                self.train_labels = cached_data['train_labels']
                self.test_labels = cached_data['test_labels']
                self.modality_configs = cached_data['modality_configs']
                return True
            except Exception as e:
                if self.verbose:
                    print(f"  Cache load failed: {e}")
        return False
    
    def _save_to_cache(self, cache_key: str):
        """Save processed data to cache."""
        cache_file = Path(self.cache_dir) / f"{cache_key}.pkl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            cached_data = {
                'train_data': self.train_data,
                'test_data': self.test_data,
                'train_labels': self.train_labels,
                'test_labels': self.test_labels,
                'modality_configs': self.modality_configs
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
        except Exception as e:
            if self.verbose:
                print(f"  Cache save failed: {e}")
    
    def _create_cross_modal_features(self):
        """Create cross-modal interaction features."""
        if self.verbose:
            print("  Creating cross-modal features...")
        
        # Get modality names
        modalities = list(self.train_data.keys())
        if len(modalities) < 2:
            return
        
        # Create interaction features for each pair
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities[i+1:], i+1):
                # Get data shapes
                train1 = self.train_data[mod1]
                train2 = self.train_data[mod2]
                test1 = self.test_data[mod1]
                test2 = self.test_data[mod2]
                
                # Handle different dimensions by taking minimum
                min_features = min(train1.shape[1], train2.shape[1])
                
                # Truncate to same dimensions
                train1_trunc = train1[:, :min_features]
                train2_trunc = train2[:, :min_features]
                test1_trunc = test1[:, :min_features]
                test2_trunc = test2[:, :min_features]
                
                # Element-wise multiplication
                train_interaction = train1_trunc * train2_trunc
                test_interaction = test1_trunc * test2_trunc
                
                # Add to data
                interaction_name = f"{mod1}_{mod2}_interaction"
                self.train_data[interaction_name] = train_interaction
                self.test_data[interaction_name] = test_interaction
                
                # Update config
                self.modality_configs[interaction_name] = {
                    'data_type': 'interaction',
                    'feature_dim': min_features,
                    'train_samples': train_interaction.shape[0],
                    'test_samples': test_interaction.shape[0]
                }
        
        if self.verbose:
            print(f"  Created {len(modalities) * (len(modalities) - 1) // 2} cross-modal features")
    
    def _apply_feature_selection(self, n_features_select: int = None):
        """Apply feature selection to reduce dimensionality."""
        if self.verbose:
            print("  Applying feature selection...")
        
        # Auto-select number of features if not specified
        if n_features_select is None:
            total_features = sum(data.shape[1] for data in self.train_data.values())
            n_features_select = min(500, total_features // 2)  # Conservative selection
        
        # Combine all features for selection
        train_features = np.hstack([self.train_data[mod] for mod in self.train_data.keys()])
        test_features = np.hstack([self.test_data[mod] for mod in self.test_data.keys()])
        
        # Apply feature selection
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features_select)
        train_selected = selector.fit_transform(train_features, self.train_labels)
        test_selected = selector.transform(test_features)
        
        # Update data (simplified - replace first modality with selected features)
        first_modality = list(self.train_data.keys())[0]
        self.train_data[first_modality] = train_selected
        self.test_data[first_modality] = test_selected
        
        # Update config
        self.modality_configs[first_modality]['feature_dim'] = n_features_select
        
        if self.verbose:
            print(f"  Selected {n_features_select} features from {train_features.shape[1]} total")
    
    def _process_data_parallel(self, handle_nan: str, handle_inf: str, normalize: bool, 
                              scaler_type: str, remove_outliers: bool, outlier_method: str, outlier_std: float):
        """Process data using parallel processing."""
        if self.verbose:
            print(f"  Using parallel processing with {self.n_jobs} cores...")
        
        # Process training data in parallel
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            train_futures = {}
            for modality_name, data in self.train_data.items():
                future = executor.submit(
                    self._process_modality_data, data, handle_nan, handle_inf, 
                    normalize, scaler_type, remove_outliers, outlier_method, outlier_std, modality_name
                )
                train_futures[modality_name] = future
            
            # Process test data in parallel
            test_futures = {}
            for modality_name, data in self.test_data.items():
                future = executor.submit(
                    self._process_modality_data, data, handle_nan, handle_inf, 
                    normalize, scaler_type, remove_outliers, outlier_method, outlier_std, modality_name
                )
                test_futures[modality_name] = future
            
            # Collect results
            for modality_name, future in train_futures.items():
                self.train_data[modality_name] = future.result()
            
            for modality_name, future in test_futures.items():
                self.test_data[modality_name] = future.result()
    
    def _process_data_sequential(self, handle_nan: str, handle_inf: str, normalize: bool, 
                               scaler_type: str, remove_outliers: bool, outlier_method: str, outlier_std: float):
        """Process data sequentially."""
        # Process training data
        for modality_name, data in self.train_data.items():
            self.train_data[modality_name] = self._process_modality_data(
                data, handle_nan, handle_inf, normalize, scaler_type, 
                remove_outliers, outlier_method, outlier_std, modality_name
            )
        
        # Process testing data
        for modality_name, data in self.test_data.items():
            self.test_data[modality_name] = self._process_modality_data(
                data, handle_nan, handle_inf, normalize, scaler_type, 
                remove_outliers, outlier_method, outlier_std, modality_name
            )
    
    def _create_polynomial_features(self, degree: int = 2):
        """Create polynomial features for enhanced feature engineering."""
        if self.verbose:
            print(f"  Creating polynomial features (degree {degree})...")
        
        # Apply to each modality separately
        for modality_name in list(self.train_data.keys()):
            if modality_name.endswith('_interaction'):  # Skip interaction features
                continue
                
            train_data = self.train_data[modality_name]
            test_data = self.test_data[modality_name]
            
            # Limit features to avoid explosion
            max_features = min(50, train_data.shape[1])
            train_subset = train_data[:, :max_features]
            test_subset = test_data[:, :max_features]
            
            # Create polynomial features
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
            train_poly = poly.fit_transform(train_subset)
            test_poly = poly.transform(test_subset)
            
            # Add polynomial features as new modality
            poly_name = f"{modality_name}_polynomial"
            self.train_data[poly_name] = train_poly
            self.test_data[poly_name] = test_poly
            
            # Update config
            self.modality_configs[poly_name] = {
                'data_type': 'polynomial',
                'feature_dim': train_poly.shape[1],
                'train_samples': train_poly.shape[0],
                'test_samples': test_poly.shape[0]
            }
        
        if self.verbose:
            print(f"  Created polynomial features for {len([k for k in self.train_data.keys() if k.endswith('_polynomial')])} modalities")
    
    def _apply_advanced_feature_selection(self, method: str, n_features_select: int = None):
        """Apply advanced feature selection methods."""
        if self.verbose:
            print(f"  Applying {method} feature selection...")
        
        # Auto-select number of features if not specified
        if n_features_select is None:
            total_features = sum(data.shape[1] for data in self.train_data.values())
            n_features_select = min(500, total_features // 2)
        
        # Combine all features for selection
        train_features = np.hstack([self.train_data[mod] for mod in self.train_data.keys()])
        test_features = np.hstack([self.test_data[mod] for mod in self.test_data.keys()])
        
        # Choose selection method
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features_select)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=n_features_select)
        elif method == 'variance':
            # First remove low variance features
            var_selector = VarianceThreshold(threshold=0.01)
            train_var = var_selector.fit_transform(train_features)
            test_var = var_selector.transform(test_features)
            
            # Then select top features
            selector = SelectKBest(score_func=f_classif, k=n_features_select)
            train_selected = selector.fit_transform(train_var, self.train_labels)
            test_selected = selector.transform(test_var)
        elif method == 'model_based':
            # Use Random Forest feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=self.n_jobs)
            rf.fit(train_features, self.train_labels)
            selector = SelectFromModel(rf, max_features=n_features_select)
            train_selected = selector.fit_transform(train_features, self.train_labels)
            test_selected = selector.transform(test_features)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Apply selection (if not already applied in variance method)
        if method != 'variance':
            train_selected = selector.fit_transform(train_features, self.train_labels)
            test_selected = selector.transform(test_features)
        
        # Update data (replace first modality with selected features)
        first_modality = list(self.train_data.keys())[0]
        self.train_data[first_modality] = train_selected
        self.test_data[first_modality] = test_selected
        
        # Update config
        self.modality_configs[first_modality]['feature_dim'] = n_features_select
        
        if self.verbose:
            print(f"  Selected {n_features_select} features from {train_features.shape[1]} total using {method}")

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
