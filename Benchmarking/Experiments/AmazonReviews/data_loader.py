#!/usr/bin/env python3
"""
Data handling module for Amazon Reviews experiments
"""

import numpy as np
from typing import Tuple, Dict, Any
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from config import ExperimentConfig, PathConfig


class AmazonReviewsDataLoader:
    """Data loader for Amazon Reviews dataset"""
    
    def __init__(self, exp_config: ExperimentConfig, path_config: PathConfig):
        self.exp_config = exp_config
        self.path_config = path_config
        
        # Data containers
        self.train_text = None
        self.train_metadata = None
        self.train_labels = None
        
        self.test_text = None
        self.test_metadata = None
        self.test_labels = None
        
        # Scalers for normalization
        self.text_scaler = StandardScaler()
        self.metadata_scaler = StandardScaler()
    
    def load_raw_data(self) -> None:
        """Load Amazon Reviews preprocessed data"""
        print("📊 Loading Amazon Reviews dataset...")
        
        try:
            # Load training data
            train_path = self.path_config.data_path / "train"
            self.train_text = np.load(train_path / "text_features.npy")
            self.train_metadata = np.load(train_path / "metadata_features.npy")
            train_labels_raw = np.load(train_path / "labels.npy")
            
            # Load test data
            test_path = self.path_config.data_path / "test"
            self.test_text = np.load(test_path / "text_features.npy")
            self.test_metadata = np.load(test_path / "metadata_features.npy")
            test_labels_raw = np.load(test_path / "labels.npy")
            
            # Transform labels from 1-5 to 0-4 for classification
            self.train_labels = train_labels_raw.astype(int) - 1
            self.test_labels = test_labels_raw.astype(int) - 1
            
            print(f"✅ Dataset loaded successfully:")
            print(f"   📊 Training: {len(self.train_labels):,} samples")
            print(f"   📊 Test: {len(self.test_labels):,} samples")
            print(f"   📝 Text features: {self.train_text.shape[1]:,} dimensions")
            print(f"   🏷️ Metadata features: {self.train_metadata.shape[1]} dimensions")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            print("   Please ensure Amazon Reviews data is preprocessed and available")
            raise
        except Exception as e:
            print(f"❌ Unexpected error loading data: {e}")
            raise
    
    def apply_sampling(self) -> None:
        """Apply dataset sampling strategy"""
        if not self.exp_config.use_small_sample:
            print("📊 Using full dataset (no sampling)")
            return
        
        print(f"🔬 Applying small sample strategy...")
        print(f"   📊 Target train size: {self.exp_config.small_sample_train_size:,}")
        print(f"   📊 Target test size: {self.exp_config.small_sample_test_size:,}")
        
        # Sample training data
        if len(self.train_labels) > self.exp_config.small_sample_train_size:
            rng = np.random.RandomState(self.exp_config.random_seed)
            train_indices = rng.choice(
                len(self.train_labels), 
                self.exp_config.small_sample_train_size, 
                replace=False
            )
            
            self.train_text = self.train_text[train_indices]
            self.train_metadata = self.train_metadata[train_indices]
            self.train_labels = self.train_labels[train_indices]
        
        # Sample test data
        if len(self.test_labels) > self.exp_config.small_sample_test_size:
            rng = np.random.RandomState(self.exp_config.random_seed)
            test_indices = rng.choice(
                len(self.test_labels), 
                self.exp_config.small_sample_test_size, 
                replace=False
            )
            
            self.test_text = self.test_text[test_indices]
            self.test_metadata = self.test_metadata[test_indices]
            self.test_labels = self.test_labels[test_indices]
        
        print(f"✅ Sampling completed:")
        print(f"   📊 Final train size: {len(self.train_labels):,}")
        print(f"   📊 Final test size: {len(self.test_labels):,}")
    
    def analyze_distribution(self) -> Dict[str, Any]:
        """Analyze dataset distribution and characteristics"""
        print("📈 Analyzing dataset distribution...")
        
        # Class distribution analysis
        unique, counts = np.unique(self.train_labels, return_counts=True)
        distribution_stats = {
            'total_samples': {
                'train': len(self.train_labels),
                'test': len(self.test_labels)
            },
            'feature_dimensions': {
                'text': self.train_text.shape[1],
                'metadata': self.train_metadata.shape[1]
            },
            'class_distribution': {},
            'class_balance': {}
        }
        
        print(f"📊 Class distribution analysis:")
        for i, (label, count) in enumerate(zip(unique, counts)):
            percentage = 100 * count / len(self.train_labels)
            class_name = self.exp_config.class_names[i]
            
            distribution_stats['class_distribution'][class_name] = {
                'count': int(count),
                'percentage': float(percentage)
            }
            
            print(f"   {class_name}: {count:,} ({percentage:.1f}%)")
        
        # Balance analysis
        max_count = max(counts)
        min_count = min(counts)
        balance_ratio = min_count / max_count
        
        distribution_stats['class_balance'] = {
            'balance_ratio': float(balance_ratio),
            'is_balanced': balance_ratio > 0.8,
            'imbalance_severity': 'low' if balance_ratio > 0.8 else 'moderate' if balance_ratio > 0.5 else 'high'
        }
        
        print(f"⚖️ Class balance analysis:")
        print(f"   Balance ratio: {balance_ratio:.3f}")
        print(f"   Imbalance severity: {distribution_stats['class_balance']['imbalance_severity']}")
        
        # Feature statistics
        print(f"📊 Feature statistics:")
        print(f"   Text features - Mean: {self.train_text.mean():.3f}, Std: {self.train_text.std():.3f}")
        print(f"   Metadata features - Mean: {self.train_metadata.mean():.3f}, Std: {self.train_metadata.std():.3f}")
        
        return distribution_stats
    
    def prepare_modality_data(self, normalize: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Prepare individual modality datasets with optional normalization"""
        print("🔧 Preparing modality-specific data...")
        
        # Copy data to avoid modifying originals
        text_train = self.train_text.copy()
        text_test = self.test_text.copy()
        meta_train = self.train_metadata.copy()
        meta_test = self.test_metadata.copy()
        
        # Apply normalization if requested
        if normalize:
            print("   📊 Applying StandardScaler normalization...")
            
            # Normalize text features
            text_train = self.text_scaler.fit_transform(text_train)
            text_test = self.text_scaler.transform(text_test)
            
            # Normalize metadata features
            meta_train = self.metadata_scaler.fit_transform(meta_train)
            meta_test = self.metadata_scaler.transform(meta_test)
        
        # Create combined features (early fusion)
        combined_train = np.hstack([text_train, meta_train])
        combined_test = np.hstack([text_test, meta_test])
        
        modality_data = {
            'text': (text_train, text_test),
            'metadata': (meta_train, meta_test),
            'combined': (combined_train, combined_test)
        }
        
        print(f"✅ Modality data prepared:")
        for modality, (train_data, test_data) in modality_data.items():
            print(f"   {modality}: {train_data.shape[1]} features")
        
        return modality_data
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        if self.train_labels is None:
            return {"error": "Data not loaded yet"}
        
        return {
            'dataset_info': {
                'name': self.exp_config.dataset_name,
                'task_type': self.exp_config.task_type,
                'n_classes': self.exp_config.n_classes,
                'class_names': self.exp_config.class_names
            },
            'data_shapes': {
                'train_samples': len(self.train_labels),
                'test_samples': len(self.test_labels),
                'text_features': self.train_text.shape[1],
                'metadata_features': self.train_metadata.shape[1]
            },
            'sampling_info': {
                'use_small_sample': self.exp_config.use_small_sample,
                'train_target_size': self.exp_config.small_sample_train_size,
                'test_target_size': self.exp_config.small_sample_test_size
            }
        }
    
    # Backward compatibility methods for advanced evaluator
    def get_combined_features(self) -> np.ndarray:
        """Get combined (text + metadata) features for training - backward compatibility"""
        modality_data = self.prepare_modality_data(normalize=True)
        return modality_data['combined'][0]  # Return training data
    
    def get_combined_text_features(self) -> np.ndarray:
        """Get text features for training - backward compatibility"""
        modality_data = self.prepare_modality_data(normalize=True)
        return modality_data['text'][0]  # Return training data
    
    def get_combined_metadata_features(self) -> np.ndarray:
        """Get metadata features for training - backward compatibility"""
        modality_data = self.prepare_modality_data(normalize=True)
        return modality_data['metadata'][0]  # Return training data
    
    def get_combined_labels(self) -> np.ndarray:
        """Get training labels - backward compatibility"""
        return self.train_labels
