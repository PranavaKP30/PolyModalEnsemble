#!/usr/bin/env python3
"""
Data handling module for ChestX-ray14 experiments
"""

import numpy as np
from typing import Tuple, Dict, Any
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from config import ExperimentConfig, PathConfig


class ChestXRayDataLoader:
    def get_subset_for_pathology(self, pathology_idx: int, max_samples: int = None):
        """
        Return train/test splits for a single pathology (binary classification for one column).
        Returns:
            (train_image, train_text, train_metadata, single_train_labels,
             test_image, test_text, test_metadata, single_test_labels)
        """
        # Select binary labels for the given pathology
        single_train_labels = self.train_labels[:, pathology_idx]
        single_test_labels = self.test_labels[:, pathology_idx]

        # Debug: Check original data
        print(f"     [DEBUG] Original data check:")
        print(f"       train_image non-zero: {np.count_nonzero(self.train_image)}")
        print(f"       train_text non-zero: {np.count_nonzero(self.train_text)}")
        print(f"       test_image non-zero: {np.count_nonzero(self.test_image)}")
        print(f"       test_text non-zero: {np.count_nonzero(self.test_text)}")

        # Optionally subsample for speed
        if max_samples is not None:
            # Subsample train
            train_pos = np.where(single_train_labels == 1)[0]
            train_neg = np.where(single_train_labels == 0)[0]
            n_train = min(max_samples, len(single_train_labels))
            rng = np.random.default_rng(42)
            n_pos = min(len(train_pos), n_train // 2)
            n_neg = n_train - n_pos
            sel_train = np.concatenate([
                rng.choice(train_pos, n_pos, replace=False) if n_pos > 0 else np.array([], int),
                rng.choice(train_neg, n_neg, replace=False) if n_neg > 0 else np.array([], int)
            ])
            rng.shuffle(sel_train)
            train_image = self.train_image[sel_train]
            train_text = self.train_text[sel_train]
            train_metadata = self.train_metadata[sel_train]
            single_train_labels = single_train_labels[sel_train]
            
            # Debug: Check subsampled train data
            print(f"     [DEBUG] After train subsampling:")
            print(f"       train_image non-zero: {np.count_nonzero(train_image)}")
            print(f"       train_text non-zero: {np.count_nonzero(train_text)}")
            
            # Subsample test
            test_pos = np.where(single_test_labels == 1)[0]
            test_neg = np.where(single_test_labels == 0)[0]
            n_test = min(max_samples, len(single_test_labels))
            n_pos = min(len(test_pos), n_test // 2)
            n_neg = n_test - n_pos
            sel_test = np.concatenate([
                rng.choice(test_pos, n_pos, replace=False) if n_pos > 0 else np.array([], int),
                rng.choice(test_neg, n_neg, replace=False) if n_neg > 0 else np.array([], int)
            ])
            rng.shuffle(sel_test)
            test_image = self.test_image[sel_test]
            test_text = self.test_text[sel_test]
            test_metadata = self.test_metadata[sel_test]
            single_test_labels = single_test_labels[sel_test]
            
            # Debug: Check subsampled test data
            print(f"     [DEBUG] After test subsampling:")
            print(f"       test_image non-zero: {np.count_nonzero(test_image)}")
            print(f"       test_text non-zero: {np.count_nonzero(test_text)}")
        else:
            train_image = self.train_image
            train_text = self.train_text
            train_metadata = self.train_metadata
            test_image = self.test_image
            test_text = self.test_text
            test_metadata = self.test_metadata
            
            # Debug: Check full data
            print(f"     [DEBUG] Using full dataset:")
            print(f"       train_image non-zero: {np.count_nonzero(train_image)}")
            print(f"       train_text non-zero: {np.count_nonzero(train_text)}")
            print(f"       test_image non-zero: {np.count_nonzero(test_image)}")
            print(f"       test_text non-zero: {np.count_nonzero(test_text)}")
        
        return (train_image, train_text, train_metadata, single_train_labels,
                test_image, test_text, test_metadata, single_test_labels)
    """Data loader for ChestX-ray14 dataset"""
    
    def __init__(self, exp_config: ExperimentConfig, path_config: PathConfig):
        self.exp_config = exp_config
        self.path_config = path_config
        
        # Data containers
        self.train_image = None
        self.train_text = None
        self.train_metadata = None
        self.train_labels = None
        
        self.test_image = None
        self.test_text = None
        self.test_metadata = None
        self.test_labels = None
        
        # Scalers for normalization
        self.image_scaler = StandardScaler()
        self.text_scaler = StandardScaler()
        self.metadata_scaler = StandardScaler()
    
    def load_raw_data(self) -> None:
        """Load ChestX-ray14 preprocessed data"""
        print("📊 Loading ChestX-ray14 dataset...")
        try:
            # Check if data directory exists
            if not self.path_config.data_path.exists():
                raise FileNotFoundError(f"Data directory does not exist: {self.path_config.data_path}")
            
            # Load training data
            train_path = self.path_config.data_path / "train"
            if not train_path.exists():
                raise FileNotFoundError(f"Training data directory does not exist: {train_path}")
            
            # Check each training file
            train_files = {
                'image_features.npy': train_path / "image_features.npy",
                'text_features.npy': train_path / "text_features.npy", 
                'metadata_features.npy': train_path / "metadata_features.npy",
                'labels.npy': train_path / "labels.npy"
            }
            
            for file_name, file_path in train_files.items():
                if not file_path.exists():
                    raise FileNotFoundError(f"Training file missing: {file_path}")
            
            self.train_image = np.load(train_files['image_features.npy'])
            self.train_text = np.load(train_files['text_features.npy'])
            self.train_metadata = np.load(train_files['metadata_features.npy'])
            self.train_labels = np.load(train_files['labels.npy'])

            # Load test data
            test_path = self.path_config.data_path / "test"
            if not test_path.exists():
                raise FileNotFoundError(f"Test data directory does not exist: {test_path}")
            
            # Check each test file
            test_files = {
                'image_features.npy': test_path / "image_features.npy",
                'text_features.npy': test_path / "text_features.npy",
                'metadata_features.npy': test_path / "metadata_features.npy", 
                'labels.npy': test_path / "labels.npy"
            }
            
            for file_name, file_path in test_files.items():
                if not file_path.exists():
                    raise FileNotFoundError(f"Test file missing: {file_path}")
            
            self.test_image = np.load(test_files['image_features.npy'])
            self.test_text = np.load(test_files['text_features.npy'])
            self.test_metadata = np.load(test_files['metadata_features.npy'])
            self.test_labels = np.load(test_files['labels.npy'])

            # Validate data shapes
            if len(self.train_image) != len(self.train_text) or len(self.train_image) != len(self.train_metadata) or len(self.train_image) != len(self.train_labels):
                raise ValueError(f"Training data shape mismatch: image={len(self.train_image)}, text={len(self.train_text)}, metadata={len(self.train_metadata)}, labels={len(self.train_labels)}")
            
            if len(self.test_image) != len(self.test_text) or len(self.test_image) != len(self.test_metadata) or len(self.test_image) != len(self.test_labels):
                raise ValueError(f"Test data shape mismatch: image={len(self.test_image)}, text={len(self.test_text)}, metadata={len(self.test_metadata)}, labels={len(self.test_labels)}")

            # Data leakage prevention: zero out test set text features ONLY (not train)
            if self.test_text is not None:
                print("   ⚠️ Zeroing out test set text features to prevent data leakage.")
                self.test_text[:] = 0

            print(f"✅ Dataset loaded successfully:")
            print(f"   📊 Training: {len(self.train_labels):,} samples")
            print(f"   📊 Test: {len(self.test_labels):,} samples")
            print(f"   🖼️ Image features: {self.train_image.shape[1]:,} dimensions")
            print(f"   📝 Text features: {self.train_text.shape[1]:,} dimensions")
            print(f"   🏷️ Metadata features: {self.train_metadata.shape[1]} dimensions")

            # Debug: Print means/stds to verify data is not all zeros
            print(f"   [DEBUG] train_image mean: {self.train_image.mean():.4f}, std: {self.train_image.std():.4f}")
            print(f"   [DEBUG] train_text mean: {self.train_text.mean():.4f}, std: {self.train_text.std():.4f}")
            print(f"   [DEBUG] train_metadata mean: {self.train_metadata.mean():.4f}, std: {self.train_metadata.std():.4f}")
            print(f"   [DEBUG] test_image mean: {self.test_image.mean():.4f}, std: {self.test_image.std():.4f}")
            print(f"   [DEBUG] test_text mean: {self.test_text.mean():.4f}, std: {self.test_text.std():.4f}")
            print(f"   [DEBUG] test_metadata mean: {self.test_metadata.mean():.4f}, std: {self.test_metadata.std():.4f}")

        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            print("   Please ensure ChestX-ray14 data is preprocessed and available")
            print(f"   Expected data path: {self.path_config.data_path}")
            print("   Available directories:")
            if self.path_config.data_path.parent.exists():
                for item in self.path_config.data_path.parent.iterdir():
                    print(f"     - {item.name}")
            raise
        except Exception as e:
            print(f"❌ Unexpected error loading data: {e}")
            import traceback
            traceback.print_exc()
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
            
            self.train_image = self.train_image[train_indices]
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
            
            self.test_image = self.test_image[test_indices]
            self.test_text = self.test_text[test_indices]
            self.test_metadata = self.test_metadata[test_indices]
            self.test_labels = self.test_labels[test_indices]
        
        print(f"✅ Sampling completed:")
        print(f"   📊 Final train size: {len(self.train_labels):,}")
        print(f"   📊 Final test size: {len(self.test_labels):,}")
    
    def analyze_distribution(self) -> Dict[str, Any]:
        """Analyze dataset distribution and characteristics"""
        print("📈 Analyzing dataset distribution...")
        
        # Multi-label distribution analysis
        n_samples, n_pathologies = self.train_labels.shape
        pathology_counts = self.train_labels.sum(axis=0)
        
        distribution_stats = {
            'total_samples': {
                'train': len(self.train_labels),
                'test': len(self.test_labels)
            },
            'feature_dimensions': {
                'image': self.train_image.shape[1],
                'text': self.train_text.shape[1],
                'metadata': self.train_metadata.shape[1]
            },
            'pathology_distribution': {},
            'pathology_statistics': {}
        }
        
        print(f"📊 Pathology distribution analysis:")
        for i, count in enumerate(pathology_counts):
            percentage = 100 * count / n_samples
            pathology_name = self.exp_config.class_names[i]
            
            distribution_stats['pathology_distribution'][pathology_name] = {
                'count': int(count),
                'percentage': float(percentage)
            }
            
            print(f"   {pathology_name}: {count:,} ({percentage:.1f}%)")
        
        # Multi-label statistics
        labels_per_sample = self.train_labels.sum(axis=1)
        avg_labels_per_sample = labels_per_sample.mean()
        
        distribution_stats['pathology_statistics'] = {
            'avg_pathologies_per_sample': float(avg_labels_per_sample),
            'min_pathologies_per_sample': int(labels_per_sample.min()),
            'max_pathologies_per_sample': int(labels_per_sample.max()),
            'total_pathologies': int(n_pathologies)
        }
        
        print(f"⚖️ Multi-label statistics:")
        print(f"   Avg pathologies per sample: {avg_labels_per_sample:.2f}")
        print(f"   Min pathologies per sample: {labels_per_sample.min()}")
        print(f"   Max pathologies per sample: {labels_per_sample.max()}")
        
        # Feature statistics
        print(f"📊 Feature statistics:")
        print(f"   Image features - Mean: {self.train_image.mean():.3f}, Std: {self.train_image.std():.3f}")
        print(f"   Text features - Mean: {self.train_text.mean():.3f}, Std: {self.train_text.std():.3f}")
        print(f"   Metadata features - Mean: {self.train_metadata.mean():.3f}, Std: {self.train_metadata.std():.3f}")
        
        return distribution_stats
    
    def prepare_modality_data(self, normalize: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Prepare individual modality datasets with optional normalization"""
        print("🔧 Preparing modality-specific data...")
        
        # Copy data to avoid modifying originals
        image_train = self.train_image.copy()
        image_test = self.test_image.copy()
        text_train = self.train_text.copy()
        text_test = self.test_text.copy()
        meta_train = self.train_metadata.copy()
        meta_test = self.test_metadata.copy()
        
        # Apply normalization if requested
        if normalize:
            print("   📊 Applying StandardScaler normalization...")
            
            # Normalize image features
            image_train = self.image_scaler.fit_transform(image_train)
            image_test = self.image_scaler.transform(image_test)
            
            # Normalize text features
            text_train = self.text_scaler.fit_transform(text_train)
            text_test = self.text_scaler.transform(text_test)
            
            # Normalize metadata features
            meta_train = self.metadata_scaler.fit_transform(meta_train)
            meta_test = self.metadata_scaler.transform(meta_test)
        
        # Create combined features (early fusion)
        combined_train = np.hstack([image_train, text_train, meta_train])
        combined_test = np.hstack([image_test, text_test, meta_test])
        
        modality_data = {
            'image': (image_train, image_test),
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
                'image_features': self.train_image.shape[1],
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
        """Get combined (image + text + metadata) features for training - backward compatibility"""
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
