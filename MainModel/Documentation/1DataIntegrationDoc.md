# Stage 1: Data Integration Documentation

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com)
[![Status](https://img.shields.io/badge/status-production_ready-brightgreen.svg)](https://github.com)
[![Component](https://img.shields.io/badge/component-data_integration-purple.svg)](https://github.com)

**Advanced multimodal data integration framework providing enterprise-grade data loading, validation, and preprocessing capabilities for heterogeneous data sources.**

## 🎯 Overview

The `dataIntegration.py` module is the **data integration engine** of the multimodal pipeline, responsible for loading, validating, and preprocessing heterogeneous data from multiple modalities. This component serves as the foundation for the entire ensemble pipeline, ensuring data quality, consistency, and optimal memory management for large-scale multimodal datasets.

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Architecture Overview](#️-architecture-overview)
3. [Core Classes & Components](#-core-classes--components)
4. [Quick Start Guide](#-quick-start-guide)
5. [Data Loading Deep Dive](#-data-loading-deep-dive)
6. [Advanced Features Overview](#-advanced-features-overview)
7. [Pipeline Integration](#-pipeline-integration)
8. [Configuration Reference](#️-configuration-reference)
9. [Comprehensive Analytics](#-comprehensive-analytics)
10. [Real-World Applications](#-real-world-applications)
11. [Troubleshooting Guide](#-troubleshooting-guide)
12. [Best Practices](#-best-practices)
13. [API Reference](#-api-reference)
14. [Summary](#-summary)

### Core Value Proposition
- 🔄 **Universal Data Loading**: Support for files, arrays, directories, and multiple formats
- 🛡️ **Enterprise Validation**: Comprehensive data quality, shape, type, NaN, and Inf checks
- 🧹 **Intelligent Preprocessing**: Advanced cleaning, normalization, outlier and missing value handling
- 💾 **Memory Optimization**: Sparse support, batch processing, and lazy loading capabilities
- 📊 **Quality Monitoring**: Real-time metrics, comprehensive reporting, and data health tracking
- 🔗 **Pipeline Integration**: Seamless export for ensemble generation and downstream processing

**Universal Data Loading Implementation**:
```python
def add_modality(self, name: str, data: Union[np.ndarray, str, Path], 
                data_type: str = "tabular", is_required: bool = False, 
                feature_dim: Optional[int] = None):
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
```

**Data Validation Characteristics**:
- **Multi-format Support**: CSV, NPY, NPZ, and direct array inputs
- **Automatic Reshaping**: 1D arrays converted to 2D feature matrices
- **Type Safety**: Comprehensive type checking and conversion
- **Memory Efficient**: Lazy loading and sparse matrix support
- **Error Handling**: Robust error detection and reporting

## 🏗️ Architecture Overview

The data integration system implements a **5-layer architecture** designed for maximum flexibility and data quality:

```
┌─────────────────────────────────────────────────────────────────┐
│                   Data Integration Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Data Source Interface                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │File Loading │  │Array Input  │  │Directory    │             │
│  │CSV/NPY/NPZ  │  │Direct Arrays│  │Scanning     │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Validation & Quality Control                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Shape        │  │Type         │  │NaN/Inf      │             │
│  │Validation   │  │Checking     │  │Detection    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Preprocessing & Cleaning                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Missing      │  │Outlier      │  │Normalization│             │
│  │Value        │  │Detection    │  │& Scaling    │             │
│  │Handling     │  │& Removal    │  │             │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Modality Configuration                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Modality     │  │Feature      │  │Priority     │             │
│  │Definition   │  │Dimension    │  │Assignment   │             │
│  │& Types      │  │Management   │  │& Weights    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 5: Export & Integration                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Quality      │  │Metadata     │  │Pipeline     │             │
│  │Reporting    │  │Generation   │  │Export API   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
           ↓                    ↓                    ↓
    Validated Data        Quality Metrics     Integration Metadata
```

### Core Components

#### 1. **GenericMultiModalDataLoader** - Primary Integration Engine
Advanced data loader with comprehensive validation, preprocessing, and quality monitoring.

#### 2. **ModalityConfig** - Intelligent Configuration Management
Dataclass-based modality configuration with metadata tracking and validation.

#### 3. **QuickDatasetBuilder** - Rapid Dataset Construction
Static utility class for fast dataset creation from various data sources.

#### 4. **DataLoaderInterface** - Abstract Interface
ABC-based interface for custom data loader implementations.

## 🔧 Core Classes & Components

### **ModalityConfig** - Configuration Management
```python
@dataclass
class ModalityConfig:
    name: str
    data_type: str  # 'tabular', 'text', 'image', 'audio', etc.
    feature_dim: Optional[int] = None
    is_required: bool = False
    priority: float = 1.0
    min_feature_ratio: float = 0.3
    max_feature_ratio: float = 1.0
```

**Key Features**:
- **Modality Identification**: Unique name and data type specification
- **Feature Management**: Automatic or manual feature dimension handling
- **Priority System**: Configurable importance weights for ensemble selection
- **Ratio Constraints**: Min/max feature ratios for adaptive processing
- **Validation**: Automatic parameter validation and error checking

### **GenericMultiModalDataLoader** - Main Integration Engine
```python
class GenericMultiModalDataLoader:
    def __init__(self, validate_data: bool = True, memory_efficient: bool = False):
        self.validate_data = validate_data
        self.memory_efficient = memory_efficient
        self.data: Dict[str, Any] = {}
        self.modality_configs: List[ModalityConfig] = []
        self._data_stats: Dict[str, Any] = {}
        self._sample_size: int = 0
        self._quality_report: Optional[Dict[str, Any]] = None
```

**Core Methods**:
- `add_modality_split()`: Add train/test data for a modality
- `add_labels_split()`: Add train/test labels with validation
- `get_split()`: Retrieve data for specific split (train/test)
- `clean_data()`: Advanced data cleaning and preprocessing
- `get_data_quality_report()`: Comprehensive quality metrics
- `export_for_ensemble_generation()`: Pipeline-ready export

### **QuickDatasetBuilder** - Rapid Construction Utilities
```python
class QuickDatasetBuilder:
    @staticmethod
    def from_arrays(modality_data: Dict[str, np.ndarray], 
                   modality_types: Optional[Dict[str, str]] = None, 
                   labels: Optional[np.ndarray] = None, 
                   required_modalities: Optional[List[str]] = None) -> 'GenericMultiModalDataLoader'
    
    @staticmethod
    def from_files(file_paths: Dict[str, str], 
                  modality_types: Optional[Dict[str, str]] = None, 
                  required_modalities: Optional[List[str]] = None) -> 'GenericMultiModalDataLoader'
    
    @staticmethod
    def from_directory(data_dir: str, 
                      modality_patterns: Dict[str, str], 
                      modality_types: Optional[Dict[str, str]] = None, 
                      required_modalities: Optional[List[str]] = None) -> 'GenericMultiModalDataLoader'
```

## 🚀 Quick Start Guide

### Basic Usage - Array-Based Data
```python
from MainModel.dataIntegration import GenericMultiModalDataLoader, QuickDatasetBuilder
import numpy as np

# Method 1: Direct construction
loader = GenericMultiModalDataLoader()
loader.add_modality_split("text", train_text, test_text, "text", is_required=True)
loader.add_modality_split("image", train_images, test_images, "image")
loader.add_labels_split(train_labels, test_labels)

# Method 2: Quick builder
modality_data = {
    "text": text_features,
    "image": image_features,
    "metadata": metadata_features
}
modality_types = {"text": "text", "image": "image", "metadata": "tabular"}
loader = QuickDatasetBuilder.from_arrays(modality_data, modality_types, labels)
```

### File-Based Loading
```python
# Load from files
file_paths = {
    "text": "data/text_features.npy",
    "image": "data/image_features.npy",
    "metadata": "data/metadata.csv"
}
loader = QuickDatasetBuilder.from_files(file_paths, modality_types)

# Load from directory with patterns
modality_patterns = {
    "text": "text_*.npy",
    "image": "image_*.npy",
    "metadata": "meta_*.csv"
}
loader = QuickDatasetBuilder.from_directory("data/", modality_patterns)
```

### Advanced Configuration
```python
# Custom modality configuration
loader = GenericMultiModalDataLoader(validate_data=True, memory_efficient=True)

# Add modalities with specific configurations
loader.add_modality_split(
    name="text",
    train_data=train_text,
    test_data=test_text,
    data_type="text",
    is_required=True,
    feature_dim=1000
)

# Add labels with validation
loader.add_labels_split(train_labels, test_labels, name="labels")

# Get data for specific split
train_data, train_labels = loader.get_split("train")
test_data, test_labels = loader.get_split("test")
```

## 📊 Data Loading Deep Dive

### **Multi-Format Support**
The system supports various data formats with automatic detection and loading:

```python
# CSV files
arr = np.genfromtxt(data, delimiter=",", skip_header=1)

# NumPy arrays
arr = np.load(data)

# Compressed NumPy arrays
arr = np.load(data)[np.load(data).files[0]]

# Direct arrays
arr = data  # Already numpy array
```

### **Automatic Reshaping**
```python
# 1D arrays automatically reshaped to 2D
if arr.ndim == 1:
    arr = arr.reshape(-1, 1)

# Labels handled specially
if arr.ndim == 2 and arr.shape[1] == 1:
    arr = arr.ravel()
```

### **Validation Pipeline**
```python
def _validate_modality(self, name: str, arr: np.ndarray):
    stats = {
        "shape": arr.shape,
        "nan_count": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
        "inf_count": int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
        "data_type": str(arr.dtype),
        "is_required": any(c.name == name and c.is_required for c in self.modality_configs)
    }
    self._data_stats[name] = stats
```

## 🔧 Advanced Features Overview

### **Data Cleaning & Preprocessing**
```python
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
            arr[inds] = np.take(means, inds[1])
        elif handle_nan == 'fill_zero':
            arr = np.nan_to_num(arr, nan=0.0)
        
        # Handle Inf values
        if handle_inf == 'drop':
            mask = ~np.isinf(arr).any(axis=1)
            arr = arr[mask]
        elif handle_inf == 'fill_max':
            maxs = np.nanmax(arr, axis=0)
            inds = np.where(np.isinf(arr))
            arr[inds] = np.take(maxs, inds[1])
        elif handle_inf == 'fill_zero':
            arr = np.where(np.isinf(arr), 0.0, arr)
        
        self.data[name] = arr
```

### **Quality Monitoring**
```python
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
```

## 🔗 Pipeline Integration

### **Export for Ensemble Generation**
```python
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
```

### **Integration with Stage 2 (Ensemble Generation)**
```python
# Export data for ensemble generation
data, modality_configs, metadata = loader.export_for_ensemble_generation()

# Pass to ensemble generation
from MainModel.modalityDropoutBagger import ModalityDropoutBagger
bagger = ModalityDropoutBagger(n_bags=10, sample_ratio=0.8)
bags = bagger.generate_bags(data, modality_configs, metadata)
```

## ⚙️ Configuration Reference

### **ModalityConfig Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Unique modality identifier |
| `data_type` | str | Required | Data type ('tabular', 'text', 'image', 'audio') |
| `feature_dim` | Optional[int] | None | Number of features (auto-detected if None) |
| `is_required` | bool | False | Whether modality is required for ensemble |
| `priority` | float | 1.0 | Priority weight for ensemble selection |
| `min_feature_ratio` | float | 0.3 | Minimum feature ratio for adaptive processing |
| `max_feature_ratio` | float | 1.0 | Maximum feature ratio for adaptive processing |

### **Data Cleaning Options**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `handle_nan` | 'drop' | Remove rows with NaN values |
| `handle_nan` | 'fill_mean' | Replace NaN with column means |
| `handle_nan` | 'fill_zero' | Replace NaN with zeros |
| `handle_inf` | 'drop' | Remove rows with Inf values |
| `handle_inf` | 'fill_max' | Replace Inf with column maximums |
| `handle_inf` | 'fill_zero' | Replace Inf with zeros |

## 📈 Comprehensive Analytics

### **Quality Metrics Tracking**
```python
# Get comprehensive quality report
quality_report = loader.get_data_quality_report()

# Example output structure
{
    "modalities": {
        "text": {
            "shape": (1000, 1000),
            "nan_count": 0,
            "inf_count": 0,
            "data_type": "float64",
            "is_required": True
        },
        "image": {
            "shape": (1000, 2048),
            "nan_count": 0,
            "inf_count": 0,
            "data_type": "float32",
            "is_required": False
        }
    },
    "overall": {
        "validation_enabled": True,
        "total_samples": 1000,
        "total_modalities": 2,
        "required_modalities": 1
    }
}
```

### **Data Summary**
```python
def summary(self):
    print("Data Integration Summary:")
    for name, arr in self.data.items():
        print(f"- {name}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"Modalities: {[c.name for c in self.modality_configs]}")
    print(f"Total samples: {self._sample_size}")
```

## 🌍 Real-World Applications

### **Multimodal E-commerce**
```python
# Load product review data
loader = GenericMultiModalDataLoader()

# Add text features (TF-IDF vectors)
loader.add_modality_split("text", train_text, test_text, "text", is_required=True)

# Add image features (CNN embeddings)
loader.add_modality_split("image", train_images, test_images, "image")

# Add metadata (product attributes)
loader.add_modality_split("metadata", train_meta, test_meta, "tabular")

# Add labels (rating predictions)
loader.add_labels_split(train_labels, test_labels)

# Export for ensemble training
data, configs, metadata = loader.export_for_ensemble_generation()
```

### **Medical Imaging with Clinical Data**
```python
# Load medical imaging dataset
loader = GenericMultiModalDataLoader(validate_data=True)

# Add imaging features
loader.add_modality_split("xray", train_xray, test_xray, "image", is_required=True)

# Add clinical metadata
loader.add_modality_split("clinical", train_clinical, test_clinical, "tabular")

# Add lab results
loader.add_modality_split("lab", train_lab, test_lab, "tabular")

# Add diagnosis labels
loader.add_labels_split(train_diagnosis, test_diagnosis)

# Clean and validate data
loader.clean_data(handle_nan='fill_mean', handle_inf='fill_max')
```

## 🔧 Troubleshooting Guide

### **Common Issues & Solutions**

#### Issue 1: Shape Mismatch Error
```
ValueError: Train and test data for modality 'text' must have same number of features
```
**Solution**: Ensure train and test data have identical feature dimensions.

#### Issue 2: Sample Size Inconsistency
```
ValueError: Train data for modality 'image' must have same number of samples as other modalities
```
**Solution**: Verify all modalities have the same number of samples.

#### Issue 3: File Format Not Supported
```
ValueError: Unsupported file format: .json
```
**Solution**: Use supported formats (CSV, NPY, NPZ) or convert data to numpy arrays.

#### Issue 4: Memory Issues with Large Datasets
**Solution**: Enable memory-efficient mode:
```python
loader = GenericMultiModalDataLoader(memory_efficient=True)
```

### **Debugging Tips**
```python
# Check data shapes
loader.summary()

# Validate specific modality
loader._validate_modality("text", text_data)

# Get quality report
report = loader.get_data_quality_report()
print(report)

# Check modality configurations
for config in loader.modality_configs:
    print(f"Modality: {config.name}, Type: {config.data_type}, Required: {config.is_required}")
```

## 📚 Best Practices

### **1. Data Validation**
```python
# Always enable validation for production
loader = GenericMultiModalDataLoader(validate_data=True)

# Check quality before processing
quality_report = loader.get_data_quality_report()
if quality_report["overall"]["total_samples"] == 0:
    raise ValueError("No valid data found")
```

### **2. Memory Management**
```python
# Use memory-efficient mode for large datasets
loader = GenericMultiModalDataLoader(memory_efficient=True)

# Clean data to reduce memory usage
loader.clean_data(handle_nan='fill_mean', handle_inf='fill_max')
```

### **3. Modality Configuration**
```python
# Set required modalities appropriately
loader.add_modality_split("text", train_text, test_text, "text", is_required=True)
loader.add_modality_split("image", train_images, test_images, "image", is_required=False)

# Use meaningful modality names
loader.add_modality_split("product_reviews", train_reviews, test_reviews, "text")
```

### **4. Error Handling**
```python
try:
    loader.add_modality_split("text", train_text, test_text, "text")
except ValueError as e:
    print(f"Validation error: {e}")
    # Handle error appropriately
```

## 📖 API Reference

### **GenericMultiModalDataLoader**

#### Constructor
```python
__init__(self, validate_data: bool = True, memory_efficient: bool = False)
```

#### Core Methods
```python
add_modality_split(name: str, train_data: np.ndarray, test_data: np.ndarray, 
                  data_type: str = "tabular", is_required: bool = False, 
                  feature_dim: Optional[int] = None)

add_labels_split(train_labels: np.ndarray, test_labels: np.ndarray, name: str = "labels")

get_split(split: str = "train") -> Tuple[Dict[str, np.ndarray], np.ndarray]

clean_data(handle_nan: str = 'fill_mean', handle_inf: str = 'fill_max')

get_data_quality_report() -> Dict[str, Any]

export_for_ensemble_generation() -> Tuple[Dict[str, np.ndarray], List[ModalityConfig], Dict[str, Any]]

summary()
```

### **ModalityConfig**

#### Constructor
```python
__init__(self, name: str, data_type: str, feature_dim: Optional[int] = None, 
         is_required: bool = False, priority: float = 1.0, 
         min_feature_ratio: float = 0.3, max_feature_ratio: float = 1.0)
```

#### Attributes
- `name`: Unique modality identifier
- `data_type`: Data type specification
- `feature_dim`: Number of features
- `is_required`: Required modality flag
- `priority`: Priority weight
- `min_feature_ratio`: Minimum feature ratio
- `max_feature_ratio`: Maximum feature ratio

### **QuickDatasetBuilder**

#### Static Methods
```python
from_arrays(modality_data: Dict[str, np.ndarray], 
           modality_types: Optional[Dict[str, str]] = None, 
           labels: Optional[np.ndarray] = None, 
           required_modalities: Optional[List[str]] = None) -> GenericMultiModalDataLoader

from_files(file_paths: Dict[str, str], 
          modality_types: Optional[Dict[str, str]] = None, 
          required_modalities: Optional[List[str]] = None) -> GenericMultiModalDataLoader

from_directory(data_dir: str, 
              modality_patterns: Dict[str, str], 
              modality_types: Optional[Dict[str, str]] = None, 
              required_modalities: Optional[List[str]] = None) -> GenericMultiModalDataLoader
```

### **Utility Functions**
```python
create_synthetic_dataset(modality_specs: Dict[str, Tuple[int, str]], 
                        n_samples: int = 1000, 
                        n_classes: int = 10, 
                        noise_level: float = 0.1, 
                        missing_data_rate: float = 0.0, 
                        random_state: Optional[int] = None) -> GenericMultiModalDataLoader

auto_preprocess_dataset(loader: GenericMultiModalDataLoader, 
                       normalize: bool = True, 
                       handle_missing: str = 'mean', 
                       remove_outliers: bool = False, 
                       outlier_std: float = 3.0) -> GenericMultiModalDataLoader
```

## 📝 Summary

The **Data Integration** module provides a comprehensive, enterprise-grade foundation for multimodal data processing in the ensemble pipeline. Key achievements include:

### **🎯 Core Capabilities**
- **Universal Data Loading**: Support for multiple formats and data sources
- **Enterprise Validation**: Comprehensive quality checks and error handling
- **Intelligent Preprocessing**: Advanced cleaning and normalization
- **Memory Optimization**: Efficient handling of large-scale datasets
- **Quality Monitoring**: Real-time metrics and comprehensive reporting

### **🔧 Technical Innovations**
- **ModalityConfig System**: Flexible configuration management with validation
- **Split Management**: Seamless train/test data handling
- **QuickDatasetBuilder**: Rapid dataset construction utilities
- **Synthetic Data Generation**: Testing and development support
- **Pipeline Integration**: Seamless export for ensemble generation

### **🚀 Performance Features**
- **Memory Efficient**: Optimized for large-scale datasets
- **Validation Enabled**: Production-grade error detection
- **Flexible Configuration**: Adaptable to various use cases
- **Quality Assurance**: Comprehensive monitoring and reporting

### **🔗 Integration Benefits**
- **Stage 1 Foundation**: Serves as the data foundation for the entire pipeline
- **Ensemble Ready**: Direct export to ensemble generation
- **Quality Guaranteed**: Validated data ensures robust ensemble performance
- **Scalable Architecture**: Designed for enterprise-scale deployments

The Data Integration module successfully bridges the gap between raw multimodal data and the sophisticated ensemble generation pipeline, ensuring data quality, consistency, and optimal performance for the entire multimodal ensemble system.

The data integration system implements a **5-layer architecture** designed for maximum flexibility and data quality:

```
┌─────────────────────────────────────────────────────────────────┐
│                   Data Integration Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Data Source Interface                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │File Loading │  │Array Input  │  │Directory    │             │
│  │CSV/NPY/NPZ  │  │Direct Arrays│  │Scanning     │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Validation & Quality Control                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Shape        │  │Type         │  │NaN/Inf      │             │
│  │Validation   │  │Checking     │  │Detection    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Preprocessing & Cleaning                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Missing      │  │Outlier      │  │Normalization│             │
│  │Value        │  │Detection    │  │& Scaling    │             │
│  │Handling     │  │& Removal    │  │             │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Modality Configuration                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Modality     │  │Feature      │  │Priority     │             │
│  │Definition   │  │Dimension    │  │Assignment   │             │
│  │& Types      │  │Management   │  │& Weights    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 5: Export & Integration                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Quality      │  │Metadata     │  │Pipeline     │             │
│  │Reporting    │  │Generation   │  │Export API   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
           ↓                    ↓                    ↓
    Validated Data        Quality Metrics     Integration Metadata
```

## 🔧 Core Classes & Components

### **ModalityConfig** - Configuration Management
```python
@dataclass
class ModalityConfig:
    name: str
    data_type: str  # 'tabular', 'text', 'image', 'audio', etc.
    feature_dim: Optional[int] = None
    is_required: bool = False
    priority: float = 1.0
    min_feature_ratio: float = 0.3
    max_feature_ratio: float = 1.0
```

**Key Features**:
- **Modality Identification**: Unique name and data type specification
- **Feature Management**: Automatic or manual feature dimension handling
- **Priority System**: Configurable importance weights for ensemble selection
- **Ratio Constraints**: Min/max feature ratios for adaptive processing
- **Validation**: Automatic parameter validation and error checking

### **GenericMultiModalDataLoader** - Main Integration Engine
```python
class GenericMultiModalDataLoader:
    def __init__(self, validate_data: bool = True, memory_efficient: bool = False):
        self.validate_data = validate_data
        self.memory_efficient = memory_efficient
        self.data: Dict[str, Any] = {}
        self.modality_configs: List[ModalityConfig] = []
        self._data_stats: Dict[str, Any] = {}
        self._sample_size: int = 0
        self._quality_report: Optional[Dict[str, Any]] = None
```

**Core Methods**:
- `add_modality_split()`: Add train/test data for a modality
- `add_labels_split()`: Add train/test labels with validation
- `get_split()`: Retrieve data for specific split (train/test)
- `clean_data()`: Advanced data cleaning and preprocessing
- `get_data_quality_report()`: Comprehensive quality metrics
- `export_for_ensemble_generation()`: Pipeline-ready export

### **QuickDatasetBuilder** - Rapid Construction Utilities
```python
class QuickDatasetBuilder:
    @staticmethod
    def from_arrays(modality_data: Dict[str, np.ndarray], 
                   modality_types: Optional[Dict[str, str]] = None, 
                   labels: Optional[np.ndarray] = None, 
                   required_modalities: Optional[List[str]] = None) -> 'GenericMultiModalDataLoader'
    
    @staticmethod
    def from_files(file_paths: Dict[str, str], 
                  modality_types: Optional[Dict[str, str]] = None, 
                  required_modalities: Optional[List[str]] = None) -> 'GenericMultiModalDataLoader'
    
    @staticmethod
    def from_directory(data_dir: str, 
                      modality_patterns: Dict[str, str], 
                      modality_types: Optional[Dict[str, str]] = None, 
                      required_modalities: Optional[List[str]] = None) -> 'GenericMultiModalDataLoader'
```

## 🚀 Quick Start Guide

### Basic Usage - Array-Based Data
```python
from MainModel.dataIntegration import GenericMultiModalDataLoader, QuickDatasetBuilder
import numpy as np

# Method 1: Direct construction
loader = GenericMultiModalDataLoader()
loader.add_modality_split("text", train_text, test_text, "text", is_required=True)
loader.add_modality_split("image", train_images, test_images, "image")
loader.add_labels_split(train_labels, test_labels)

# Method 2: Quick builder
modality_data = {
    "text": text_features,
    "image": image_features,
    "metadata": metadata_features
}
modality_types = {"text": "text", "image": "image", "metadata": "tabular"}
loader = QuickDatasetBuilder.from_arrays(modality_data, modality_types, labels)
```

### File-Based Loading
```python
# Load from files
file_paths = {
    "text": "data/text_features.npy",
    "image": "data/image_features.npy",
    "metadata": "data/metadata.csv"
}
loader = QuickDatasetBuilder.from_files(file_paths, modality_types)

# Load from directory with patterns
modality_patterns = {
    "text": "text_*.npy",
    "image": "image_*.npy",
    "metadata": "meta_*.csv"
}
loader = QuickDatasetBuilder.from_directory("data/", modality_patterns)
```

## 📊 Data Loading Deep Dive

### **Multi-Format Support**
The system supports various data formats with automatic detection and loading:

```python
# CSV files
arr = np.genfromtxt(data, delimiter=",", skip_header=1)

# NumPy arrays
arr = np.load(data)

# Compressed NumPy arrays
arr = np.load(data)[np.load(data).files[0]]

# Direct arrays
arr = data  # Already numpy array
```

### **Automatic Reshaping**
```python
# 1D arrays automatically reshaped to 2D
if arr.ndim == 1:
    arr = arr.reshape(-1, 1)

# Labels handled specially
if arr.ndim == 2 and arr.shape[1] == 1:
    arr = arr.ravel()
```

### **Validation Pipeline**
```python
def _validate_modality(self, name: str, arr: np.ndarray):
    stats = {
        "shape": arr.shape,
        "nan_count": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
        "inf_count": int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
        "data_type": str(arr.dtype),
        "is_required": any(c.name == name and c.is_required for c in self.modality_configs)
    }
    self._data_stats[name] = stats
```

## 🔧 Advanced Features Overview

### **Data Cleaning & Preprocessing**
```python
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
            arr[inds] = np.take(means, inds[1])
        elif handle_nan == 'fill_zero':
            arr = np.nan_to_num(arr, nan=0.0)
        
        # Handle Inf values
        if handle_inf == 'drop':
            mask = ~np.isinf(arr).any(axis=1)
            arr = arr[mask]
        elif handle_inf == 'fill_max':
            maxs = np.nanmax(arr, axis=0)
            inds = np.where(np.isinf(arr))
            arr[inds] = np.take(maxs, inds[1])
        elif handle_inf == 'fill_zero':
            arr = np.where(np.isinf(arr), 0.0, arr)
        
        self.data[name] = arr
```

### **Quality Monitoring**
```python
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
```

## 🔗 Pipeline Integration

### **Export for Ensemble Generation**
```python
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
```

### **Integration with Stage 2 (Ensemble Generation)**
```python
# Export data for ensemble generation
data, modality_configs, metadata = loader.export_for_ensemble_generation()

# Pass to ensemble generation
from MainModel.modalityDropoutBagger import ModalityDropoutBagger
bagger = ModalityDropoutBagger(n_bags=10, sample_ratio=0.8)
bags = bagger.generate_bags(data, modality_configs, metadata)
```

## ⚙️ Configuration Reference

### **ModalityConfig Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Unique modality identifier |
| `data_type` | str | Required | Data type ('tabular', 'text', 'image', 'audio') |
| `feature_dim` | Optional[int] | None | Number of features (auto-detected if None) |
| `is_required` | bool | False | Whether modality is required for ensemble |
| `priority` | float | 1.0 | Priority weight for ensemble selection |
| `min_feature_ratio` | float | 0.3 | Minimum feature ratio for adaptive processing |
| `max_feature_ratio` | float | 1.0 | Maximum feature ratio for adaptive processing |

### **Data Cleaning Options**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `handle_nan` | 'drop' | Remove rows with NaN values |
| `handle_nan` | 'fill_mean' | Replace NaN with column means |
| `handle_nan` | 'fill_zero' | Replace NaN with zeros |
| `handle_inf` | 'drop' | Remove rows with Inf values |
| `handle_inf` | 'fill_max' | Replace Inf with column maximums |
| `handle_inf` | 'fill_zero' | Replace Inf with zeros |

## 📈 Comprehensive Analytics

### **Quality Metrics Tracking**
```python
# Get comprehensive quality report
quality_report = loader.get_data_quality_report()

# Example output structure
{
    "modalities": {
        "text": {
            "shape": (1000, 1000),
            "nan_count": 0,
            "inf_count": 0,
            "data_type": "float64",
            "is_required": True
        },
        "image": {
            "shape": (1000, 2048),
            "nan_count": 0,
            "inf_count": 0,
            "data_type": "float32",
            "is_required": False
        }
    },
    "overall": {
        "validation_enabled": True,
        "total_samples": 1000,
        "total_modalities": 2,
        "required_modalities": 1
    }
}
```

### **Data Summary**
```python
def summary(self):
    print("Data Integration Summary:")
    for name, arr in self.data.items():
        print(f"- {name}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"Modalities: {[c.name for c in self.modality_configs]}")
    print(f"Total samples: {self._sample_size}")
```

## 🌍 Real-World Applications

### **Multimodal E-commerce**
```python
# Load product review data
loader = GenericMultiModalDataLoader()

# Add text features (TF-IDF vectors)
loader.add_modality_split("text", train_text, test_text, "text", is_required=True)

# Add image features (CNN embeddings)
loader.add_modality_split("image", train_images, test_images, "image")

# Add metadata (product attributes)
loader.add_modality_split("metadata", train_meta, test_meta, "tabular")

# Add labels (rating predictions)
loader.add_labels_split(train_labels, test_labels)

# Export for ensemble training
data, configs, metadata = loader.export_for_ensemble_generation()
```

### **Medical Imaging with Clinical Data**
```python
# Load medical imaging dataset
loader = GenericMultiModalDataLoader(validate_data=True)

# Add imaging features
loader.add_modality_split("xray", train_xray, test_xray, "image", is_required=True)

# Add clinical metadata
loader.add_modality_split("clinical", train_clinical, test_clinical, "tabular")

# Add lab results
loader.add_modality_split("lab", train_lab, test_lab, "tabular")

# Add diagnosis labels
loader.add_labels_split(train_diagnosis, test_diagnosis)

# Clean and validate data
loader.clean_data(handle_nan='fill_mean', handle_inf='fill_max')
```

## 🔧 Troubleshooting Guide

### **Common Issues & Solutions**

#### Issue 1: Shape Mismatch Error
```
ValueError: Train and test data for modality 'text' must have same number of features
```
**Solution**: Ensure train and test data have identical feature dimensions.

#### Issue 2: Sample Size Inconsistency
```
ValueError: Train data for modality 'image' must have same number of samples as other modalities
```
**Solution**: Verify all modalities have the same number of samples.

#### Issue 3: File Format Not Supported
```
ValueError: Unsupported file format: .json
```
**Solution**: Use supported formats (CSV, NPY, NPZ) or convert data to numpy arrays.

#### Issue 4: Memory Issues with Large Datasets
**Solution**: Enable memory-efficient mode:
```python
loader = GenericMultiModalDataLoader(memory_efficient=True)
```

### **Debugging Tips**
```python
# Check data shapes
loader.summary()

# Validate specific modality
loader._validate_modality("text", text_data)

# Get quality report
report = loader.get_data_quality_report()
print(report)

# Check modality configurations
for config in loader.modality_configs:
    print(f"Modality: {config.name}, Type: {config.data_type}, Required: {config.is_required}")
```

## 📚 Best Practices

### **1. Data Validation**
```python
# Always enable validation for production
loader = GenericMultiModalDataLoader(validate_data=True)

# Check quality before processing
quality_report = loader.get_data_quality_report()
if quality_report["overall"]["total_samples"] == 0:
    raise ValueError("No valid data found")
```

### **2. Memory Management**
```python
# Use memory-efficient mode for large datasets
loader = GenericMultiModalDataLoader(memory_efficient=True)

# Clean data to reduce memory usage
loader.clean_data(handle_nan='fill_mean', handle_inf='fill_max')
```

### **3. Modality Configuration**
```python
# Set required modalities appropriately
loader.add_modality_split("text", train_text, test_text, "text", is_required=True)
loader.add_modality_split("image", train_images, test_images, "image", is_required=False)

# Use meaningful modality names
loader.add_modality_split("product_reviews", train_reviews, test_reviews, "text")
```

### **4. Error Handling**
```python
try:
    loader.add_modality_split("text", train_text, test_text, "text")
except ValueError as e:
    print(f"Validation error: {e}")
    # Handle error appropriately
```

## 📖 API Reference

### **GenericMultiModalDataLoader**

#### Constructor
```python
__init__(self, validate_data: bool = True, memory_efficient: bool = False)
```

#### Core Methods
```python
add_modality_split(name: str, train_data: np.ndarray, test_data: np.ndarray, 
                  data_type: str = "tabular", is_required: bool = False, 
                  feature_dim: Optional[int] = None)

add_labels_split(train_labels: np.ndarray, test_labels: np.ndarray, name: str = "labels")

get_split(split: str = "train") -> Tuple[Dict[str, np.ndarray], np.ndarray]

clean_data(handle_nan: str = 'fill_mean', handle_inf: str = 'fill_max')

get_data_quality_report() -> Dict[str, Any]

export_for_ensemble_generation() -> Tuple[Dict[str, np.ndarray], List[ModalityConfig], Dict[str, Any]]

summary()
```

### **ModalityConfig**

#### Constructor
```python
__init__(self, name: str, data_type: str, feature_dim: Optional[int] = None, 
         is_required: bool = False, priority: float = 1.0, 
         min_feature_ratio: float = 0.3, max_feature_ratio: float = 1.0)
```

#### Attributes
- `name`: Unique modality identifier
- `data_type`: Data type specification
- `feature_dim`: Number of features
- `is_required`: Required modality flag
- `priority`: Priority weight
- `min_feature_ratio`: Minimum feature ratio
- `max_feature_ratio`: Maximum feature ratio

### **QuickDatasetBuilder**

#### Static Methods
```python
from_arrays(modality_data: Dict[str, np.ndarray], 
           modality_types: Optional[Dict[str, str]] = None, 
           labels: Optional[np.ndarray] = None, 
           required_modalities: Optional[List[str]] = None) -> GenericMultiModalDataLoader

from_files(file_paths: Dict[str, str], 
          modality_types: Optional[Dict[str, str]] = None, 
          required_modalities: Optional[List[str]] = None) -> GenericMultiModalDataLoader

from_directory(data_dir: str, 
              modality_patterns: Dict[str, str], 
              modality_types: Optional[Dict[str, str]] = None, 
              required_modalities: Optional[List[str]] = None) -> GenericMultiModalDataLoader
```

### **Utility Functions**
```python
create_synthetic_dataset(modality_specs: Dict[str, Tuple[int, str]], 
                        n_samples: int = 1000, 
                        n_classes: int = 10, 
                        noise_level: float = 0.1, 
                        missing_data_rate: float = 0.0, 
                        random_state: Optional[int] = None) -> GenericMultiModalDataLoader

auto_preprocess_dataset(loader: GenericMultiModalDataLoader, 
                       normalize: bool = True, 
                       handle_missing: str = 'mean', 
                       remove_outliers: bool = False, 
                       outlier_std: float = 3.0) -> GenericMultiModalDataLoader
```

## 📝 Summary

The **Data Integration** module provides a comprehensive, enterprise-grade foundation for multimodal data processing in the ensemble pipeline. Key achievements include:

### **🎯 Core Capabilities**
- **Universal Data Loading**: Support for multiple formats and data sources
- **Enterprise Validation**: Comprehensive quality checks and error handling
- **Intelligent Preprocessing**: Advanced cleaning and normalization
- **Memory Optimization**: Efficient handling of large-scale datasets
- **Quality Monitoring**: Real-time metrics and comprehensive reporting

### **🔧 Technical Innovations**
- **ModalityConfig System**: Flexible configuration management with validation
- **Split Management**: Seamless train/test data handling
- **QuickDatasetBuilder**: Rapid dataset construction utilities
- **Synthetic Data Generation**: Testing and development support
- **Pipeline Integration**: Seamless export for ensemble generation

### **🚀 Performance Features**
- **Memory Efficient**: Optimized for large-scale datasets
- **Validation Enabled**: Production-grade error detection
- **Flexible Configuration**: Adaptable to various use cases
- **Quality Assurance**: Comprehensive monitoring and reporting

### **🔗 Integration Benefits**
- **Stage 1 Foundation**: Serves as the data foundation for the entire pipeline
- **Ensemble Ready**: Direct export to ensemble generation
- **Quality Guaranteed**: Validated data ensures robust ensemble performance
- **Scalable Architecture**: Designed for enterprise-scale deployments

The Data Integration module successfully bridges the gap between raw multimodal data and the sophisticated ensemble generation pipeline, ensuring data quality, consistency, and optimal performance for the entire multimodal ensemble system.
