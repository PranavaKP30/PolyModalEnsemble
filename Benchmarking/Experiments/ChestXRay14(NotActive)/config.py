#!/usr/bin/env python3
"""
Configuration module for ChestX-ray14 experiments
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class ExperimentConfig:
    """Configuration for ChestX-ray14 experiments"""
    
    # Basic configuration
    use_small_sample: bool = True
    random_seed: int = 42
    
    # Dataset configuration
    dataset_name: str = "ChestX-ray14"
    task_type: str = "14-class multi-label pathology classification"
    n_classes: int = 14
    
    # Class names (pathologies)
    class_names: List[str] = None
    pathologies: List[str] = None  # Alias for class_names
    n_pathologies: int = 14  # Alias for n_classes
    
    # Sampling configuration (for quick testing)
    small_sample_train_size: int = 500
    small_sample_test_size: int = 100
    
    # Backward compatibility aliases
    @property 
    def train_sample_size(self) -> int:
        return self.get_train_size() or self.small_sample_train_size
        
    @property
    def test_sample_size(self) -> int:
        return self.get_test_size() or self.small_sample_test_size
    
    # Full dataset configuration (for production runs)
    full_dataset_train_size: int = None  # None = use entire available dataset
    full_dataset_test_size: int = None   # None = use entire available dataset
    
    # Model configuration (for quick testing)
    default_n_bags: int = 3
    default_epochs: int = 3
    default_batch_size: int = 256
    
    # Full model configuration (for production runs)
    full_n_bags: int = 10
    full_epochs: int = 20
    full_batch_size: int = 256
    
    # Experiment configuration
    cv_folds: int = 5  # Number of cross-validation folds
    enable_cv: bool = True  # Whether to enable cross-validation
    multi_seed_seeds: List[int] = None
    hyperparameter_search_trials: int = 12  # Quick mode: 2×2×3 = 12 trials (half of full)
    full_hyperparameter_search_trials: int = 54  # Full mode: 6×3×3 = 54 trials
    enhanced_quick_hp_trials: int = 27  # Legacy: Enhanced quick mode
    
    def __post_init__(self):
        """Initialize default values after instantiation"""
        if self.class_names is None:
            self.class_names = [
                'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 
                'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 
                'Pneumonia', 'Pneumothorax'
            ]
        
        # Set aliases for compatibility
        if self.pathologies is None:
            self.pathologies = self.class_names
        self.n_pathologies = len(self.class_names)
        
        if self.multi_seed_seeds is None:
            self.multi_seed_seeds = [42, 123, 456, 789, 999]
        
        # Set numpy random seed
        np.random.seed(self.random_seed)
    
    def get_train_size(self) -> Optional[int]:
        """Get appropriate train size based on use_small_sample flag"""
        return self.small_sample_train_size if self.use_small_sample else self.full_dataset_train_size
    
    def get_test_size(self) -> Optional[int]:
        """Get appropriate test size based on use_small_sample flag"""
        return self.small_sample_test_size if self.use_small_sample else self.full_dataset_test_size
    
    @property
    def num_seeds(self) -> int:
        """Get number of seeds for multi-seed experiments"""
        return len(self.multi_seed_seeds) if self.multi_seed_seeds else 1
    
    def get_n_bags(self) -> int:
        """Get appropriate number of bags based on use_small_sample flag"""
        return self.default_n_bags if self.use_small_sample else self.full_n_bags
    
    def get_epochs(self) -> int:
        """Get appropriate number of epochs based on use_small_sample flag"""
        return self.default_epochs if self.use_small_sample else self.full_epochs
    
    def get_batch_size(self) -> int:
        """Get appropriate batch size based on use_small_sample flag"""
        return self.default_batch_size if self.use_small_sample else self.full_batch_size
    
    def get_hp_trials(self) -> int:
        """Get appropriate number of hyperparameter trials based on use_small_sample flag"""
        return self.hyperparameter_search_trials if self.use_small_sample else self.full_hyperparameter_search_trials


@dataclass
class PathConfig:
    """Path configuration for experiments"""
    
    project_root: Path
    data_path: Path = None
    output_path: Path = None
    metrics_path: Path = None
    
    def __post_init__(self):
        """Initialize paths after instantiation"""
        # Use the full ChestXray14 dataset by default (not the fixed/500 version)
        if self.data_path is None:
            self.data_path = self.project_root / "Benchmarking" / "PreprocessedData" / "ChestXray14"
        if self.output_path is None:
            self.output_path = self.project_root / "Benchmarking" / "Experiments" / "ChestXRay14" / "results"
        if self.metrics_path is None:
            self.metrics_path = self.project_root / "Benchmarking" / "Experiments" / "ChestXRay14" / "metrics"
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.metrics_path.mkdir(parents=True, exist_ok=True)


def get_default_config(project_root: Path, **kwargs) -> tuple[ExperimentConfig, PathConfig]:
    """Get default configuration for experiments"""
    exp_config = ExperimentConfig(**kwargs)
    path_config = PathConfig(project_root=project_root)
    
    return exp_config, path_config


def print_config_summary(exp_config: ExperimentConfig, path_config: PathConfig):
    """Print configuration summary"""
    print(f"🏥 ChestX-ray14 Experiment Configuration")
    print("=" * 60)
    print(f"📁 Data path: {path_config.data_path}")
    print(f"📊 Output path: {path_config.output_path}")
    print(f"📈 Metrics path: {path_config.metrics_path}")
    print(f"🎯 Task: {exp_config.n_classes}-class multi-label pathology classification")
    print(f"🔬 Random seed: {exp_config.random_seed}")
    
    if exp_config.use_small_sample:
        print("⚡ Mode: SMALL SAMPLE (fast testing)")
        print(f"   📊 Train samples: {exp_config.get_train_size():,}")
        print(f"   📊 Test samples: {exp_config.get_test_size():,}")
        print(f"   🎯 Bags: {exp_config.get_n_bags()}")
        print(f"   📈 Epochs: {exp_config.get_epochs()}")
        print(f"   🔬 HP trials: {exp_config.get_hp_trials()}")
    else:
        print("🎯 Mode: FULL DATASET (production)")
        print(f"   📊 Train samples: {'All available' if exp_config.get_train_size() is None else f'{exp_config.get_train_size():,}'}")
        print(f"   📊 Test samples: {'All available' if exp_config.get_test_size() is None else f'{exp_config.get_test_size():,}'}")
        print(f"   🎯 Bags: {exp_config.get_n_bags()}")
        print(f"   📈 Epochs: {exp_config.get_epochs()}")
        print(f"   🔬 HP trials: {exp_config.get_hp_trials()}")
    print(f"🎲 Multi-seed evaluation: {exp_config.multi_seed_seeds}")
