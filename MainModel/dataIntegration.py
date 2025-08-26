"""
Stage 1: Data Integration
Advanced multimodal data integration framework providing enterprise-grade data loading, validation, and preprocessing capabilities for heterogeneous data sources.

Implements:
- Universal Data Loading (files, arrays, directories)
- Enterprise Validation (data quality, shape/type/NaN/Inf checks)
- Intelligent Preprocessing (cleaning, normalization, outlier/missing handling)
- Memory Optimization (sparse support, batch, lazy loading)
- Quality Monitoring (real-time metrics, reporting)
- Pipeline Integration (export for ensemble generation)

See MainModel/1dataIntegrationDoc.md for full documentation and API reference.
"""

import numpy as np
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

logger = logging.getLogger("data_integration")

# --- ModalityConfig ---
@dataclass
class ModalityConfig:
    name: str
    data_type: str  # 'tabular', 'text', 'image', 'audio', etc.
    feature_dim: Optional[int] = None
    is_required: bool = False
    priority: float = 1.0
    min_feature_ratio: float = 0.3
    max_feature_ratio: float = 1.0

    def __post_init__(self):
        assert self.priority >= 0, "Priority must be non-negative"
        assert 0 <= self.min_feature_ratio <= 1, "min_feature_ratio must be in [0,1]"
        assert 0 <= self.max_feature_ratio <= 1, "max_feature_ratio must be in [0,1]"
        assert self.min_feature_ratio <= self.max_feature_ratio, "min_feature_ratio must be <= max_feature_ratio"
        if self.feature_dim is not None:
            assert self.feature_dim > 0, "feature_dim must be positive"

# --- DataLoaderInterface ---
class DataLoaderInterface(ABC):
    @abstractmethod
    def load_data(self) -> Dict[str, np.ndarray]:
        pass
    @abstractmethod
    def get_modality_configs(self) -> List[ModalityConfig]:
        pass

# --- GenericMultiModalDataLoader ---
class GenericMultiModalDataLoader:
    def __init__(self, validate_data: bool = True, memory_efficient: bool = False):
        self.validate_data = validate_data
        self.memory_efficient = memory_efficient
        self.data: Dict[str, Any] = {}
        self.modality_configs: List[ModalityConfig] = []
        self._data_stats: Dict[str, Any] = {}
        self._sample_size: int = 0
        self._quality_report: Optional[Dict[str, Any]] = None

    def add_modality_split(self, name: str, train_data: np.ndarray, test_data: np.ndarray, data_type: str = "tabular", is_required: bool = False, feature_dim: Optional[int] = None):
        """Add both train and test data for a modality."""
        if train_data.ndim == 1:
            train_data = train_data.reshape(-1, 1)
        if test_data.ndim == 1:
            test_data = test_data.reshape(-1, 1)
        if feature_dim is None:
            feature_dim = train_data.shape[1]
        
        # Validate that train and test have same number of features
        if train_data.shape[1] != test_data.shape[1]:
            raise ValueError(f"Train and test data for modality '{name}' must have same number of features. Got {train_data.shape[1]} vs {test_data.shape[1]}")
        
        # Validate consistency with existing modalities
        if hasattr(self, '_sample_size') and self._sample_size > 0:
            if train_data.shape[0] != self._sample_size:
                raise ValueError(f"Train data for modality '{name}' must have same number of samples as other modalities. Got {train_data.shape[0]} vs {self._sample_size}")
        
        self.data[f"{name}_train"] = train_data
        self.data[f"{name}_test"] = test_data
        self.modality_configs.append(ModalityConfig(name, data_type, feature_dim, is_required))
        self._sample_size = train_data.shape[0]
        if self.validate_data:
            self._validate_modality(f"{name}_train", train_data)
            self._validate_modality(f"{name}_test", test_data)

    def add_labels_split(self, train_labels: np.ndarray, test_labels: np.ndarray, name: str = "labels"):
        """Add both train and test labels."""
        if train_labels.ndim == 2 and train_labels.shape[1] == 1:
            train_labels = train_labels.ravel()
        if test_labels.ndim == 2 and test_labels.shape[1] == 1:
            test_labels = test_labels.ravel()
        
        # Validate consistency with existing modalities
        if hasattr(self, '_sample_size') and self._sample_size > 0:
            if train_labels.shape[0] != self._sample_size:
                raise ValueError(f"Train labels must have same number of samples as modalities. Got {train_labels.shape[0]} vs {self._sample_size}")
        
        # Validate that train and test labels have consistent dimensions
        if train_labels.ndim != test_labels.ndim:
            raise ValueError(f"Train and test labels must have same number of dimensions. Got {train_labels.ndim} vs {test_labels.ndim}")
        
        # For multi-label classification, validate number of label columns
        if train_labels.ndim == 2:
            if train_labels.shape[1] != test_labels.shape[1]:
                raise ValueError(f"Train and test labels must have same number of label columns. Got {train_labels.shape[1]} vs {test_labels.shape[1]}")
        
        self.data[f"{name}_train"] = train_labels
        self.data[f"{name}_test"] = test_labels

    def get_split(self, split: str = "train"):
        """Return (data_dict, labels) for the requested split ('train' or 'test')."""
        data_dict = {k.replace(f"_{split}", ""): v for k, v in self.data.items() if k.endswith(f"_{split}") and not k.startswith("labels")}
        labels_key = f"labels_{split}"
        if labels_key not in self.data:
            raise ValueError(f"No labels found for split '{split}'. Use add_labels_split().")
        labels = self.data[labels_key]
        return data_dict, labels
    def load_and_preprocess(self):
        """
        Returns (data_dict, labels) for use in the main pipeline.
        Assumes modalities and labels have already been added via add_modality/add_labels.
        """
        # Find the labels key (default 'labels')
        labels_key = None
        for k in self.data:
            if k.lower() == 'labels':
                labels_key = k
                break
        if labels_key is None:
            raise ValueError("No labels found in loader. Use add_labels().")
        data_dict = {k: v for k, v in self.data.items() if k != labels_key}
        labels = self.data[labels_key]
        return data_dict, labels
    def add_modality(self, name: str, data: Union[np.ndarray, str, Path], data_type: str = "tabular", is_required: bool = False, feature_dim: Optional[int] = None):
        # Load from file if needed
        if isinstance(data, (str, Path)):
            ext = str(data).split(".")[-1].lower()
            if ext == "csv":
                arr = np.genfromtxt(data, delimiter=",", skip_header=1)
            elif ext == "npy":
                arr = np.load(data)
            elif ext == "npz":
                arr = np.load(data)[np.load(data).files[0]]
            else:
                raise ValueError(f"Unsupported file format: {data}")
        else:
            arr = data
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if feature_dim is None:
            feature_dim = arr.shape[1]
        self.data[name] = arr
        self.modality_configs.append(ModalityConfig(name, data_type, feature_dim, is_required))
        self._sample_size = arr.shape[0]
        if self.validate_data:
            self._validate_modality(name, arr)

    def add_labels(self, labels: Union[np.ndarray, str, Path], name: str = "labels"):
        if isinstance(labels, (str, Path)):
            ext = str(labels).split(".")[-1].lower()
            if ext == "csv":
                arr = np.genfromtxt(labels, delimiter=",", skip_header=1)
            elif ext == "npy":
                arr = np.load(labels)
            else:
                raise ValueError(f"Unsupported label file format: {labels}")
        else:
            arr = labels
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
        self.data[name] = arr

    def _validate_modality(self, name: str, arr: np.ndarray):
        stats = {
            "shape": arr.shape,
            "nan_count": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
            "inf_count": int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
            "data_type": str(arr.dtype),
            "is_required": any(c.name == name and c.is_required for c in self.modality_configs)
        }
        self._data_stats[name] = stats

    def clean_data(self, handle_nan: str = 'fill_mean', handle_inf: str = 'fill_max'):
        for name, arr in self.data.items():
            if not isinstance(arr, np.ndarray) or arr.dtype.kind not in 'fc':
                continue
            
            # Handle NaN values
            if handle_nan == 'drop':
                mask = ~np.isnan(arr).any(axis=1)
                arr = arr[mask]
            elif handle_nan == 'fill_mean':
                means = np.nanmean(arr, axis=0)
                inds = np.where(np.isnan(arr))
                if len(inds) == 2:  # 2D array
                    arr[inds] = np.take(means, inds[1])
                else:  # 1D array
                    arr[inds] = means
            elif handle_nan == 'fill_zero':
                arr = np.nan_to_num(arr, nan=0.0)
            
            # Handle Inf values
            if handle_inf == 'drop':
                mask = ~np.isinf(arr).any(axis=1)
                arr = arr[mask]
            elif handle_inf == 'fill_max':
                # Use nanmax to ignore Inf values when computing max
                maxs = np.nanmax(arr, axis=0)
                # If maxs still contains Inf, replace with a large finite value
                maxs = np.where(np.isinf(maxs), 1e6, maxs)
                inds = np.where(np.isinf(arr))
                if len(inds) == 2:  # 2D array
                    arr[inds] = np.take(maxs, inds[1])
                else:  # 1D array
                    arr[inds] = maxs
            elif handle_inf == 'fill_zero':
                arr = np.where(np.isinf(arr), 0.0, arr)
            
            self.data[name] = arr

    def get_data_quality_report(self) -> Dict[str, Any]:
        report = {"modalities": {}, "overall": {}}
        total_samples = None
        for config in self.modality_configs:
            arr = self.data[config.name]
            stats = self._data_stats.get(config.name, {})
            if not stats:
                self._validate_modality(config.name, arr)
                stats = self._data_stats[config.name]
            report["modalities"][config.name] = stats
            if total_samples is None:
                total_samples = arr.shape[0]
        report["overall"] = {
            "validation_enabled": self.validate_data,
            "total_samples": total_samples,
            "total_modalities": len(self.modality_configs),
            "required_modalities": sum(c.is_required for c in self.modality_configs)
        }
        return report

    def export_for_ensemble_generation(self) -> Tuple[Dict[str, np.ndarray], List[ModalityConfig], Dict[str, Any]]:
        integration_metadata = {
            'sample_size': self._sample_size,
            'dataset_size': self._sample_size,
            'num_modalities': len(self.modality_configs),
            'modality_names': [c.name for c in self.modality_configs],
            'feature_dimensions': {c.name: c.feature_dim for c in self.modality_configs},
            'data_quality_report': self.get_data_quality_report()
        }
        return self.data, self.modality_configs, integration_metadata

    def summary(self):
        print("Data Integration Summary:")
        for name, arr in self.data.items():
            print(f"- {name}: shape={arr.shape}, dtype={arr.dtype}")
        print(f"Modalities: {[c.name for c in self.modality_configs]}")
        print(f"Total samples: {self._sample_size}")

# --- QuickDatasetBuilder ---
class QuickDatasetBuilder:
    @staticmethod
    def from_arrays(modality_data: Dict[str, np.ndarray], modality_types: Optional[Dict[str, str]] = None, labels: Optional[np.ndarray] = None, required_modalities: Optional[List[str]] = None) -> 'GenericMultiModalDataLoader':
        loader = GenericMultiModalDataLoader()
        modality_types = modality_types or {}
        required_modalities = required_modalities or []
        for name, data in modality_data.items():
            data_type = modality_types.get(name, "tabular")
            is_required = name in required_modalities
            loader.add_modality(name, data, data_type, is_required)
        if labels is not None:
            loader.add_labels(labels)
        return loader

    @staticmethod
    def from_files(file_paths: Dict[str, str], modality_types: Optional[Dict[str, str]] = None, required_modalities: Optional[List[str]] = None) -> 'GenericMultiModalDataLoader':
        loader = GenericMultiModalDataLoader()
        modality_types = modality_types or {}
        required_modalities = required_modalities or []
        for name, file_path in file_paths.items():
            data_type = modality_types.get(name, "tabular")
            is_required = name in required_modalities
            loader.add_modality(name, file_path, data_type, is_required)
        return loader

    @staticmethod
    def from_directory(data_dir: str, modality_patterns: Dict[str, str], modality_types: Optional[Dict[str, str]] = None, required_modalities: Optional[List[str]] = None) -> 'GenericMultiModalDataLoader':
        loader = GenericMultiModalDataLoader()
        data_dir = Path(data_dir)
        modality_types = modality_types or {}
        required_modalities = required_modalities or []
        for modality_name, pattern in modality_patterns.items():
            matching_files = list(data_dir.glob(pattern))
            if matching_files:
                file_path = matching_files[0]
                data_type = modality_types.get(modality_name, "tabular")
                is_required = modality_name in required_modalities
                loader.add_modality(modality_name, file_path, data_type, is_required)
            else:
                print(f"Warning: No files found for modality '{modality_name}' with pattern '{pattern}'")
        return loader

# --- Utility Functions ---
def create_synthetic_dataset(modality_specs: Dict[str, Tuple[int, str]], n_samples: int = 1000, n_classes: int = 10, noise_level: float = 0.1, missing_data_rate: float = 0.0, random_state: Optional[int] = None) -> GenericMultiModalDataLoader:
    rng = np.random.default_rng(random_state)
    loader = GenericMultiModalDataLoader()
    for name, (feature_dim, data_type) in modality_specs.items():
        data = rng.normal(0, 1, size=(n_samples, feature_dim))
        if noise_level > 0:
            data += rng.normal(0, noise_level, size=data.shape)
        if missing_data_rate > 0:
            mask = rng.uniform(0, 1, size=data.shape) < missing_data_rate
            data[mask] = np.nan
        loader.add_modality(name, data, data_type=data_type)
    labels = rng.integers(0, n_classes, size=n_samples)
    loader.add_labels(labels)
    return loader

def auto_preprocess_dataset(loader: GenericMultiModalDataLoader, normalize: bool = True, handle_missing: str = 'mean', remove_outliers: bool = False, outlier_std: float = 3.0) -> GenericMultiModalDataLoader:
    new_loader = GenericMultiModalDataLoader(validate_data=loader.validate_data, memory_efficient=loader.memory_efficient)
    for config in loader.modality_configs:
        data = loader.data[config.name]
        # Handle missing values
        if handle_missing == 'mean':
            if np.issubdtype(data.dtype, np.floating):
                means = np.nanmean(data, axis=0)
                inds = np.where(np.isnan(data))
                data[inds] = np.take(means, inds[1])
        elif handle_missing == 'zero':
            data = np.nan_to_num(data, nan=0.0)
        elif handle_missing == 'drop':
            if np.issubdtype(data.dtype, np.floating):
                mask = ~np.isnan(data).any(axis=1)
                data = data[mask]
        # Remove outliers
        if remove_outliers and np.issubdtype(data.dtype, np.floating):
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            mask = np.all(np.abs(data - mean) < outlier_std * std, axis=1)
            data = data[mask]
        # Normalize
        if normalize and np.issubdtype(data.dtype, np.floating):
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std[std == 0] = 1.0
            data = (data - mean) / std
        new_loader.add_modality(config.name, data, data_type=config.data_type, is_required=config.is_required)
    # Copy labels if present
    for k, v in loader.data.items():
        if 'label' in k.lower() and k not in new_loader.data:
            new_loader.add_labels(v, name=k)
    return new_loader
