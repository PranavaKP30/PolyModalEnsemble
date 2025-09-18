#!/usr/bin/env python3
"""
Phase 1: Data Validation & Preprocessing
Validates data quality and consistency for AmazonReviews dataset.
Handles both quick mode (10k samples) and full mode (300k samples) with proper stratified sampling.
"""

import json
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# MainModel imports not needed for Phase 1 data validation

logger = logging.getLogger(__name__)

class DataValidator:
    """Data validation and preprocessing for AmazonReviews dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data validator.
        
        Args:
            config: Experiment configuration dictionary
        """
        self.config = config
        self.dataset_path = Path(config["dataset_path"])
        self.phase_dir = Path(config["phase_dir"])
        self.seed = config["seed"]
        self.test_mode = config["test_mode"]
        # dataset_subset_ratio is no longer used - sampling is determined by test_mode
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        
        logger.info(f"Initialized DataValidator for {self.test_mode} mode")
        logger.info(f"Dataset path: {self.dataset_path}")
        logger.info(f"Test mode: {self.test_mode}")
    
    def load_dataset(self) -> Dict[str, Any]:
        """
        Load AmazonReviews dataset with proper stratified sampling.
        Quick mode: 10k samples (8k train, 2k test)
        Full mode: 300k samples (240k train, 60k test)
        
        Returns:
            Dictionary containing loaded data with proper class balance
        """
        logger.info("Loading AmazonReviews dataset with stratified sampling...")
        start_time = time.time()
        
        try:
            # Check if dataset directory exists
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
            
            # Always process the real dataset from Benchmarking/ProcessedData/AmazonReviews
            # Don't use existing sampled data - we want to validate the real processed data
            sampled_data_path = self.dataset_path / f"sampled_{self.test_mode}_{self.seed}"
            
            # Define file paths
            data_files = {
                'train': {
                    'text_features': self.dataset_path / 'train' / 'text_features.npy',
                    'metadata_features': self.dataset_path / 'train' / 'metadata_features.npy',
                    'labels': self.dataset_path / 'train' / 'labels.npy'
                },
                'test': {
                    'text_features': self.dataset_path / 'test' / 'text_features.npy',
                    'metadata_features': self.dataset_path / 'test' / 'metadata_features.npy',
                    'labels': self.dataset_path / 'test' / 'labels.npy'
                }
            }
            
            # Verify all files exist
            for split_name, files in data_files.items():
                for file_type, file_path in files.items():
                    if not file_path.exists():
                        raise FileNotFoundError(f"Required file not found: {file_path}")
            
            # Load full dataset using memory mapping
            logger.info("Loading full dataset for stratified sampling...")
            
            # Load train data
            train_labels = np.load(data_files['train']['labels'], mmap_mode='r')
            train_text = np.load(data_files['train']['text_features'], mmap_mode='r')
            train_metadata = np.load(data_files['train']['metadata_features'], mmap_mode='r')
            
            # Load test data
            test_labels = np.load(data_files['test']['labels'], mmap_mode='r')
            test_text = np.load(data_files['test']['text_features'], mmap_mode='r')
            test_metadata = np.load(data_files['test']['metadata_features'], mmap_mode='r')
            
            logger.info(f"Full dataset loaded:")
            logger.info(f"  Train: {len(train_labels):,} samples")
            logger.info(f"  Test: {len(test_labels):,} samples")
            logger.info(f"  Total: {len(train_labels) + len(test_labels):,} samples")
            
            # Calculate target sample sizes
            if self.test_mode == "quick":
                target_total = 10000
                target_train = int(target_total * 0.8)  # 80% for train
                target_test = target_total - target_train  # 20% for test
            else:  # full mode
                target_total = 300000
                target_train = int(target_total * 0.8)  # 80% for train
                target_test = target_total - target_train  # 20% for test
            
            logger.info(f"Target sample sizes for {self.test_mode} mode:")
            logger.info(f"  Train: {target_train:,} samples")
            logger.info(f"  Test: {target_test:,} samples")
            logger.info(f"  Total: {target_total:,} samples")
            
            # Perform stratified sampling
            dataset, sampling_info = self._perform_stratified_sampling(
                train_labels, train_text, train_metadata,
                test_labels, test_text, test_metadata,
                target_train, target_test
            )
            
            # Save sampled dataset for reuse
            self._save_sampled_dataset(dataset, sampling_info, sampled_data_path)
            self._last_sampling_info = sampling_info
            
            load_time = time.time() - start_time
            logger.info(f"Dataset loaded and sampled successfully in {load_time:.2f}s")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def _perform_stratified_sampling(self, train_labels, train_text, train_metadata,
                                   test_labels, test_text, test_metadata,
                                   target_train, target_test) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform stratified sampling to maintain class balance across modalities.
        
        Args:
            train_labels, train_text, train_metadata: Training data arrays
            test_labels, test_text, test_metadata: Test data arrays
            target_train, target_test: Target sample sizes
            
        Returns:
            Tuple of (sampled_dataset, sampling_info)
        """
        logger.info("Performing stratified sampling to maintain class balance...")
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        
        # Convert labels to integers for stratification
        train_labels_int = train_labels.astype(int)
        test_labels_int = test_labels.astype(int)
        
        # Get unique labels and their counts
        train_unique_labels, train_label_counts = np.unique(train_labels_int, return_counts=True)
        test_unique_labels, test_label_counts = np.unique(test_labels_int, return_counts=True)
        
        logger.info(f"Original label distribution:")
        logger.info(f"  Train: {dict(zip(train_unique_labels, train_label_counts))}")
        logger.info(f"  Test: {dict(zip(test_unique_labels, test_label_counts))}")
        
        # Calculate sampling ratios for each split
        train_ratio = target_train / len(train_labels)
        test_ratio = target_test / len(test_labels)
        
        logger.info(f"Sampling ratios: Train={train_ratio:.4f}, Test={test_ratio:.4f}")
        
        # Sample train data with stratification
        if train_ratio < 1.0:
            train_indices, _ = train_test_split(
                np.arange(len(train_labels)),
                test_size=1 - train_ratio,
                stratify=train_labels_int,
                                random_state=self.seed
                            )
        else:
            train_indices = np.arange(len(train_labels))
        
        # Sample test data with stratification
        if test_ratio < 1.0:
            test_indices, _ = train_test_split(
                np.arange(len(test_labels)),
                test_size=1 - test_ratio,
                stratify=test_labels_int,
                random_state=self.seed
            )
        else:
            test_indices = np.arange(len(test_labels))
        
        # Sort indices to maintain some order
        train_indices = np.sort(train_indices)
        test_indices = np.sort(test_indices)
        
        logger.info(f"Sampled indices: Train={len(train_indices):,}, Test={len(test_indices):,}")
        
        # Extract sampled data
        sampled_train_labels = train_labels[train_indices]
        sampled_train_text = train_text[train_indices]
        sampled_train_metadata = train_metadata[train_indices]
        
        sampled_test_labels = test_labels[test_indices]
        sampled_test_text = test_text[test_indices]
        sampled_test_metadata = test_metadata[test_indices]
        
        # Verify class balance is maintained
        sampled_train_unique, sampled_train_counts = np.unique(sampled_train_labels.astype(int), return_counts=True)
        sampled_test_unique, sampled_test_counts = np.unique(sampled_test_labels.astype(int), return_counts=True)
        
        logger.info(f"Sampled label distribution:")
        logger.info(f"  Train: {dict(zip(sampled_train_unique, sampled_train_counts))}")
        logger.info(f"  Test: {dict(zip(sampled_test_unique, sampled_test_counts))}")
        
        # Create separate DataFrames for each modality (NOT combined!)
        dataset = {}
        
        # Create train DataFrames - KEEP MODALITIES SEPARATE
        train_text_data = {}
        for i in range(sampled_train_text.shape[1]):
            train_text_data[f'text_feature_{i}'] = sampled_train_text[:, i]
        train_text_data['label'] = sampled_train_labels.astype(int) - 1
        dataset['train_text'] = pd.DataFrame(train_text_data)
        
        train_metadata_data = {}
        for i in range(sampled_train_metadata.shape[1]):
            train_metadata_data[f'metadata_feature_{i}'] = sampled_train_metadata[:, i]
        train_metadata_data['label'] = sampled_train_labels.astype(int) - 1
        dataset['train_metadata'] = pd.DataFrame(train_metadata_data)
        
        # Create test DataFrames - KEEP MODALITIES SEPARATE
        test_text_data = {}
        for i in range(sampled_test_text.shape[1]):
            test_text_data[f'text_feature_{i}'] = sampled_test_text[:, i]
        test_text_data['label'] = sampled_test_labels.astype(int) - 1
        dataset['test_text'] = pd.DataFrame(test_text_data)
        
        test_metadata_data = {}
        for i in range(sampled_test_metadata.shape[1]):
            test_metadata_data[f'metadata_feature_{i}'] = sampled_test_metadata[:, i]
        test_metadata_data['label'] = sampled_test_labels.astype(int) - 1
        dataset['test_metadata'] = pd.DataFrame(test_metadata_data)
        
        # Verify label conversion
        train_labels_final = dataset['train_text']['label'].unique()
        test_labels_final = dataset['test_text']['label'].unique()
        logger.info(f"Final label ranges: Train={sorted(train_labels_final)}, Test={sorted(test_labels_final)}")
        
        # Get final label distributions (after 0-indexing conversion)
        final_train_unique, final_train_counts = np.unique(dataset['train_text']['label'], return_counts=True)
        final_test_unique, final_test_counts = np.unique(dataset['test_text']['label'], return_counts=True)
        
        # Create sampling info
        sampling_info = {
            'test_mode': self.test_mode,
            'seed': self.seed,
            'target_sizes': {
                'train': target_train,
                'test': target_test,
                'total': target_train + target_test
            },
            'actual_sizes': {
                'train': len(dataset['train_text']),
                'test': len(dataset['test_text']),
                'total': len(dataset['train_text']) + len(dataset['test_text'])
            },
            'sampling_ratios': {
                'train': train_ratio,
                'test': test_ratio
            },
            'original_sizes': {
                'train': len(train_labels),
                'test': len(test_labels),
                'total': len(train_labels) + len(test_labels)
            },
            'label_distributions': {
                'original_train': {int(k): int(v) for k, v in zip(train_unique_labels, train_label_counts)},
                'original_test': {int(k): int(v) for k, v in zip(test_unique_labels, test_label_counts)},
                'sampled_train': {int(k): int(v) for k, v in zip(final_train_unique, final_train_counts)},
                'sampled_test': {int(k): int(v) for k, v in zip(final_test_unique, final_test_counts)}
            },
            'indices': {
                'train': train_indices.tolist(),
                'test': test_indices.tolist()
            }
        }
        
        logger.info("Stratified sampling completed successfully")
        return dataset, sampling_info
    
    def _apply_sampling(self, dataset: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Apply sampling to dataset for quick mode.
        
        Args:
            dataset: Full dataset dictionary
            
        Returns:
            Tuple of (sampled_dataset, sampling_info)
        """
        logger.info(f"Applying sampling for {self.test_mode} mode")
        
        sampling_info = {
            "test_mode": self.test_mode,
            "seed": self.seed,
            "original_sizes": {},
            "sampled_sizes": {}
        }
        
        sampled_dataset = {}
        
        for split_name, df in dataset.items():
            original_size = len(df)
            sampling_info["original_sizes"][split_name] = original_size
            
            # Calculate sample size based on test mode
            if self.test_mode == "quick":
                target_total = 10000
            else:  # full mode
                target_total = 300000
            
            if split_name == "train":
                sample_size = int(target_total * 0.8)  # 80% for train
            else:  # test
                sample_size = int(target_total * 0.2)  # 20% for test
            sampling_info["sampled_sizes"][split_name] = sample_size
            
            # Perform STRATIFIED sampling instead of random sampling
            if sample_size < original_size:
                # Get the label column (last column)
                labels = df.iloc[:, -1]
                
                # Use train_test_split for stratified sampling
                from sklearn.model_selection import train_test_split
                
                sampled_df, _ = train_test_split(
                    df, 
                    train_size=sample_size, 
                    random_state=self.seed,
                    stratify=labels
                )
                
                logger.info(f"Stratified sampled {split_name}: {original_size} -> {len(sampled_df)} samples")
                
                # Log class distribution after sampling
                sampled_labels = sampled_df.iloc[:, -1]
                class_counts = sampled_labels.value_counts().sort_index()
                logger.info(f"Class distribution after {split_name} sampling:")
                for class_val, count in class_counts.items():
                    percentage = count / len(sampled_df) * 100
                    logger.info(f"  Class {class_val}: {count} samples ({percentage:.1f}%)")
            else:
                sampled_df = df.copy()
                logger.info(f"No sampling needed for {split_name}: {original_size} samples")
            
            sampled_dataset[split_name] = sampled_df
        
        return sampled_dataset, sampling_info
    
    def _save_sampled_dataset(self, dataset: Dict[str, Any], sampling_info: Dict[str, Any], sampled_data_path: Path) -> None:
        """
        Save sampled dataset for reuse in subsequent phases.
        
        Args:
            dataset: Loaded dataset dictionary
            sampling_info: Information about sampling process
            sampled_data_path: Path to save sampled data
        """
        logger.info(f"Saving sampled dataset for {self.test_mode} mode at {sampled_data_path}")
        
        try:
            # Create directory
            sampled_data_path.mkdir(parents=True, exist_ok=True)
            
            # Save each split
            for split_name, data in dataset.items():
                split_path = sampled_data_path / split_name
                split_path.mkdir(exist_ok=True)
                
                # Save as CSV for easy loading
                csv_path = split_path / f"{split_name}_data.csv"
                data.to_csv(csv_path, index=False)
                logger.info(f"Saved {split_name} data to {csv_path} ({len(data):,} samples, {len(data.columns)} features)")
            
            # Save sampling info
            sampling_info_path = sampled_data_path / "sampling_info.json"
            with open(sampling_info_path, 'w') as f:
                json.dump(sampling_info, f, indent=2, default=str)
            logger.info(f"Saved sampling info to {sampling_info_path}")
            
            # Save metadata
            metadata = {
                "test_mode": self.test_mode,
                "seed": self.seed,
                "creation_timestamp": pd.Timestamp.now().isoformat(),
                "total_splits": len(dataset),
                "split_info": {split: {"samples": len(data), "features": len(data.columns)} 
                              for split, data in dataset.items()},
                "sampling_summary": {
                    "target_total": sampling_info.get('target_sizes', {}).get('total', 'unknown'),
                    "actual_total": sampling_info.get('actual_sizes', {}).get('total', 'unknown'),
                    "train_samples": sampling_info.get('actual_sizes', {}).get('train', 'unknown'),
                    "test_samples": sampling_info.get('actual_sizes', {}).get('test', 'unknown')
                }
            }
            metadata_path = sampled_data_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Saved metadata to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving sampled dataset: {e}")
            raise
    
    def _load_sampled_dataset(self, sampled_data_path: Path) -> Dict[str, Any]:
        """
        Load previously saved sampled dataset for quick mode.
        
        Args:
            sampled_data_path: Path to saved sampled data
            
        Returns:
            Dictionary containing loaded sampled data
        """
        logger.info(f"Loading sampled dataset from {sampled_data_path}")
        
        try:
            dataset = {}
            
            # Load each split
            for split_path in sampled_data_path.iterdir():
                if split_path.is_dir() and split_path.name in ['train', 'test']:
                    split_name = split_path.name
                    csv_path = split_path / f"{split_name}_data.csv"
                    
                    if csv_path.exists():
                        data = pd.read_csv(csv_path)
                        dataset[split_name] = data
                        logger.info(f"Loaded {split_name} data: {data.shape}")
                    else:
                        logger.warning(f"CSV file not found for {split_name}: {csv_path}")
            
            # Load and verify metadata
            metadata_path = sampled_data_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata: {metadata}")
                
                # Verify compatibility
                if metadata.get("seed") != self.seed:
                    logger.warning(f"Seed mismatch: expected {self.seed}, got {metadata.get('seed')}")
                if metadata.get("test_mode") != self.test_mode:
                    logger.warning(f"Test mode mismatch: expected {self.test_mode}, got {metadata.get('test_mode')}")
            else:
                logger.warning(f"Metadata file not found: {metadata_path}")
            
            logger.info(f"Successfully loaded sampled dataset with {len(dataset)} splits")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading sampled dataset: {e}")
            raise
    
    def validate_data_quality(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive data quality assessment.
        
        Args:
            dataset: Loaded dataset dictionary
            
        Returns:
            Data quality report
        """
        logger.info("Performing data quality assessment...")
        start_time = time.time()
        
        quality_report = {
            "dataset_info": {},
            "sample_counts": {},
            "feature_analysis": {},
            "missing_value_analysis": {},
            "data_type_analysis": {},
            "modality_analysis": {},
            "memory_usage": {},
            "validation_checks": {},
            "quality_score": 0.0
        }
        
        try:
            # Dataset info
            quality_report["dataset_info"] = {
                "dataset_name": "AmazonReviews",
                "test_mode": self.test_mode,
                "seed": self.seed,
                "available_splits": list(dataset.keys())
            }
            
            # Sample count verification
            sample_counts = {}
            for split_name, data in dataset.items():
                if split_name != 'metadata':
                    sample_counts[split_name] = len(data)
            quality_report["sample_counts"] = sample_counts
            
            # Feature analysis
            feature_analysis = {}
            for split_name, data in dataset.items():
                if split_name != 'metadata' and hasattr(data, 'columns'):
                    feature_analysis[split_name] = {
                        "n_features": len(data.columns),
                        "feature_names": list(data.columns),
                        "feature_types": {str(k): str(v) for k, v in data.dtypes.to_dict().items()}
                    }
            quality_report["feature_analysis"] = feature_analysis
            
            # Missing value analysis
            missing_analysis = {}
            for split_name, data in dataset.items():
                if split_name != 'metadata' and hasattr(data, 'isnull'):
                    missing_counts = data.isnull().sum()
                    missing_percentages = (missing_counts / len(data)) * 100
                    missing_analysis[split_name] = {
                        "missing_counts": missing_counts.to_dict(),
                        "missing_percentages": missing_percentages.to_dict(),
                        "total_missing": missing_counts.sum(),
                        "total_missing_percentage": (missing_counts.sum() / (len(data) * len(data.columns))) * 100
                    }
            quality_report["missing_value_analysis"] = missing_analysis
            
            # Data type analysis
            dtype_analysis = {}
            for split_name, data in dataset.items():
                if split_name != 'metadata' and hasattr(data, 'dtypes'):
                    dtype_counts = data.dtypes.value_counts()
                    dtype_analysis[split_name] = {
                        "dtype_counts": {str(k): int(v) for k, v in dtype_counts.to_dict().items()},
                        "text_columns": list(data.select_dtypes(include=['object', 'string']).columns),
                        "numeric_columns": list(data.select_dtypes(include=['number']).columns),
                        "categorical_columns": list(data.select_dtypes(include=['category']).columns)
                    }
            quality_report["data_type_analysis"] = dtype_analysis
            
            # Modality analysis
            modality_analysis = self._analyze_modalities(dataset)
            quality_report["modality_analysis"] = modality_analysis
            
            # Detailed modality-specific validation
            modality_validation = self._perform_modality_validation(dataset)
            quality_report["modality_validation"] = modality_validation
            
            # Memory usage
            memory_usage = {}
            total_memory = 0
            for split_name, data in dataset.items():
                if hasattr(data, 'memory_usage'):
                    memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
                    memory_usage[split_name] = memory_mb
                    total_memory += memory_mb
            memory_usage["total_memory_mb"] = total_memory
            quality_report["memory_usage"] = memory_usage
            
            # Validation checks
            validation_checks = self._perform_validation_checks(dataset, quality_report)
            quality_report["validation_checks"] = validation_checks
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(quality_report)
            quality_report["quality_score"] = quality_score
            
            validation_time = time.time() - start_time
            logger.info(f"Data quality assessment completed in {validation_time:.2f}s")
            logger.info(f"Overall quality score: {quality_score:.2f}/100")
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Error in data quality assessment: {e}")
            raise
    
    def _analyze_modalities(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze modalities present in the dataset."""
        modality_analysis = {
            "detected_modalities": [],
            "modality_distribution": {},
            "cross_modal_alignment": {}
        }
        
        try:
            # Analyze first split to determine modalities
            first_split = None
            for split_name, data in dataset.items():
                if split_name != 'metadata':
                    first_split = data
                    break
            
            if first_split is None:
                return modality_analysis
            
            # Detect modalities based on column patterns
            modalities = []
            
            # Text modality detection
            text_columns = []
            for col in first_split.columns:
                if any(keyword in col.lower() for keyword in ['text', 'review', 'comment', 'description', 'content']):
                    text_columns.append(col)
            
            if text_columns:
                modalities.append('text')
                modality_analysis["modality_distribution"]["text"] = {
                    "columns": text_columns,
                    "sample_count": len(first_split)
                }
            
            # Metadata modality detection
            metadata_columns = []
            for col in first_split.columns:
                if any(keyword in col.lower() for keyword in ['rating', 'score', 'price', 'category', 'user', 'product']):
                    if col not in text_columns:
                        metadata_columns.append(col)
            
            if metadata_columns:
                modalities.append('metadata')
                modality_analysis["modality_distribution"]["metadata"] = {
                    "columns": metadata_columns,
                    "sample_count": len(first_split)
                }
            
            modality_analysis["detected_modalities"] = modalities
            
            # Cross-modal alignment check
            for split_name, data in dataset.items():
                if split_name != 'metadata':
                    modality_analysis["cross_modal_alignment"][split_name] = {
                        "sample_count": len(data),
                        "modality_presence": {mod: len(data) for mod in modalities}
                    }
            
            return modality_analysis
            
        except Exception as e:
            logger.error(f"Error in modality analysis: {e}")
            return modality_analysis
    
    def _perform_modality_validation(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed modality-specific validation."""
        modality_validation = {
            "text_validation": {},
            "metadata_validation": {},
            "cross_modal_validation": {}
        }
        
        try:
            # Get first split for analysis
            first_split = None
            for split_name, data in dataset.items():
                if split_name != 'metadata':
                    first_split = data
                    break
            
            if first_split is None:
                return modality_validation
            
            # Text modality validation
            text_columns = [col for col in first_split.columns if 'text_feature' in col]
            if text_columns:
                text_data = first_split[text_columns]
                
                # Feature correlation analysis
                correlation_matrix = text_data.corr()
                high_corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_val = correlation_matrix.iloc[i, j]
                        if abs(corr_val) > 0.8:
                            high_corr_pairs.append({
                                'feature1': correlation_matrix.columns[i],
                                'feature2': correlation_matrix.columns[j],
                                'correlation': float(corr_val)
                            })
                
                # Statistical analysis
                text_stats = {
                    'mean': text_data.mean().to_dict(),
                    'std': text_data.std().to_dict(),
                    'min': text_data.min().to_dict(),
                    'max': text_data.max().to_dict(),
                    'high_correlation_pairs': high_corr_pairs[:10]  # Limit to top 10
                }
                
                modality_validation["text_validation"] = {
                    "feature_count": len(text_columns),
                    "statistics": text_stats,
                    "correlation_analysis": {
                        "high_correlation_pairs": len(high_corr_pairs),
                        "avg_correlation": float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean())
                    }
                }
            
            # Metadata modality validation
            metadata_columns = [col for col in first_split.columns if 'metadata_feature' in col]
            if metadata_columns:
                metadata_data = first_split[metadata_columns]
                
                # Outlier detection using IQR method
                outlier_analysis = {}
                for col in metadata_columns:
                    Q1 = metadata_data[col].quantile(0.25)
                    Q3 = metadata_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = metadata_data[(metadata_data[col] < lower_bound) | (metadata_data[col] > upper_bound)]
                    outlier_analysis[col] = {
                        'outlier_count': len(outliers),
                        'outlier_percentage': float(len(outliers) / len(metadata_data) * 100),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    }
                
                # Feature correlation analysis
                metadata_corr = metadata_data.corr()
                metadata_high_corr = []
                for i in range(len(metadata_corr.columns)):
                    for j in range(i+1, len(metadata_corr.columns)):
                        corr_val = metadata_corr.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            metadata_high_corr.append({
                                'feature1': metadata_corr.columns[i],
                                'feature2': metadata_corr.columns[j],
                                'correlation': float(corr_val)
                            })
                
                modality_validation["metadata_validation"] = {
                    "feature_count": len(metadata_columns),
                    "outlier_analysis": outlier_analysis,
                    "correlation_analysis": {
                        "high_correlation_pairs": len(metadata_high_corr),
                        "avg_correlation": float(metadata_corr.values[np.triu_indices_from(metadata_corr.values, k=1)].mean())
                    }
                }
            
            # Cross-modal validation
            if 'label' in first_split.columns:
                label_data = first_split['label']
                label_analysis = {
                    'unique_values': int(label_data.nunique()),
                    'value_counts': label_data.value_counts().to_dict(),
                    'missing_labels': int(label_data.isnull().sum()),
                    'label_distribution': {
                        'mean': float(label_data.mean()),
                        'std': float(label_data.std()),
                        'min': float(label_data.min()),
                        'max': float(label_data.max())
                    }
                }
                
                modality_validation["cross_modal_validation"] = {
                    "label_analysis": label_analysis,
                    "modality_alignment": {
                        "text_features_present": len(text_columns) > 0,
                        "metadata_features_present": len(metadata_columns) > 0,
                        "labels_present": 'label' in first_split.columns
                    }
                }
            
            return modality_validation
            
        except Exception as e:
            logger.error(f"Error in modality validation: {e}")
            return modality_validation
    
    def _perform_validation_checks(self, dataset: Dict[str, Any], quality_report: Dict[str, Any]) -> Dict[str, str]:
        """Perform specific validation checks."""
        validation_checks = {}
        
        try:
            # Sample count verification
            sample_counts = quality_report["sample_counts"]
            if len(sample_counts) >= 2:
                validation_checks["sample_count_verification"] = "passed"
            else:
                validation_checks["sample_count_verification"] = "failed"
            
            # Feature dimension consistency
            feature_analysis = quality_report["feature_analysis"]
            if len(feature_analysis) >= 2:
                feature_counts = [info["n_features"] for info in feature_analysis.values()]
                if len(set(feature_counts)) == 1:
                    validation_checks["feature_dimension_consistency"] = "passed"
                else:
                    validation_checks["feature_dimension_consistency"] = "failed"
            else:
                validation_checks["feature_dimension_consistency"] = "passed"
            
            # Missing value analysis
            missing_analysis = quality_report["missing_value_analysis"]
            high_missing_splits = []
            for split_name, info in missing_analysis.items():
                if info["total_missing_percentage"] > 50:
                    high_missing_splits.append(split_name)
            
            if not high_missing_splits:
                validation_checks["missing_value_analysis"] = "passed"
            else:
                validation_checks["missing_value_analysis"] = "warning"
            
            # Data type validation
            dtype_analysis = quality_report["data_type_analysis"]
            if dtype_analysis:
                validation_checks["data_type_validation"] = "passed"
            else:
                validation_checks["data_type_validation"] = "failed"
            
            # Cross-modal alignment
            modality_analysis = quality_report["modality_analysis"]
            if modality_analysis["detected_modalities"]:
                validation_checks["cross_modal_alignment"] = "passed"
            else:
                validation_checks["cross_modal_alignment"] = "warning"
            
            return validation_checks
            
        except Exception as e:
            logger.error(f"Error in validation checks: {e}")
            return {"error": str(e)}
    
    def _calculate_quality_score(self, quality_report: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)."""
        try:
            score = 100.0
            
            # Deduct points for missing values
            missing_analysis = quality_report["missing_value_analysis"]
            for split_info in missing_analysis.values():
                missing_pct = split_info["total_missing_percentage"]
                if missing_pct > 10:
                    score -= min(20, missing_pct)
            
            # Deduct points for validation failures
            validation_checks = quality_report["validation_checks"]
            for check_result in validation_checks.values():
                if check_result == "failed":
                    score -= 15
                elif check_result == "warning":
                    score -= 5
            
            # Deduct points for inconsistent feature dimensions
            feature_analysis = quality_report["feature_analysis"]
            if len(feature_analysis) >= 2:
                feature_counts = [info["n_features"] for info in feature_analysis.values()]
                if len(set(feature_counts)) > 1:
                    score -= 20
            
            return max(0.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def preprocess_data(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply preprocessing based on configuration.
        
        Args:
            dataset: Loaded dataset dictionary
            
        Returns:
            Preprocessed dataset
        """
        logger.info("Applying data preprocessing...")
        start_time = time.time()
        
        preprocessing_summary = {
            "preprocessing_steps": [],
            "preprocessing_time": 0.0,
            "memory_usage_before": {},
            "memory_usage_after": {},
            "preprocessing_config": {}
        }
        
        try:
            # Record memory usage before preprocessing
            for split_name, data in dataset.items():
                if hasattr(data, 'memory_usage'):
                    preprocessing_summary["memory_usage_before"][split_name] = data.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Apply preprocessing steps based on configuration
            preprocessing_steps = []
            
            # Handle NaN values
            if "handle_nan" in self.config:
                handle_nan_method = self.config["handle_nan"]
                preprocessing_steps.append(f"handle_nan_{handle_nan_method}")
                
                for split_name, data in dataset.items():
                    if split_name != 'metadata' and hasattr(data, 'fillna'):
                        if handle_nan_method == "fill_mean":
                            data.fillna(data.mean(), inplace=True)
                        elif handle_nan_method == "fill_zero":
                            data.fillna(0, inplace=True)
                        elif handle_nan_method == "drop":
                            data.dropna(inplace=True)
            
            # Handle Inf values
            if "handle_inf" in self.config:
                handle_inf_method = self.config["handle_inf"]
                preprocessing_steps.append(f"handle_inf_{handle_inf_method}")
                
                for split_name, data in dataset.items():
                    if split_name != 'metadata':
                        numeric_cols = data.select_dtypes(include=[np.number]).columns
                        if handle_inf_method == "fill_max":
                            for col in numeric_cols:
                                data[col] = data[col].replace([np.inf, -np.inf], data[col].max())
                        elif handle_inf_method == "fill_zero":
                            for col in numeric_cols:
                                data[col] = data[col].replace([np.inf, -np.inf], 0)
                        elif handle_inf_method == "drop":
                            for col in numeric_cols:
                                data = data[~np.isinf(data[col])]
            
            # Normalize data (exclude target column)
            if self.config.get("normalize_data", False):
                preprocessing_steps.append("normalize_data")
                
                for split_name, data in dataset.items():
                    if split_name != 'metadata':
                        numeric_cols = data.select_dtypes(include=[np.number]).columns
                        # Exclude target column from normalization
                        target_cols = ['label', 'target', 'class', 'rating', 'score', 'sentiment']
                        feature_cols = [col for col in numeric_cols if col not in target_cols]
                        
                        for col in feature_cols:
                            if data[col].std() > 0:
                                data[col] = (data[col] - data[col].mean()) / data[col].std()
            
            # Remove outliers (exclude target column)
            if self.config.get("remove_outliers", False):
                outlier_std = self.config.get("outlier_std", 3.0)
                preprocessing_steps.append(f"remove_outliers_std_{outlier_std}")
                
                for split_name, data in dataset.items():
                    if split_name != 'metadata':
                        numeric_cols = data.select_dtypes(include=[np.number]).columns
                        # Exclude target column from outlier removal
                        target_cols = ['label', 'target', 'class', 'rating', 'score', 'sentiment']
                        feature_cols = [col for col in numeric_cols if col not in target_cols]
                        
                        for col in feature_cols:
                            mean_val = data[col].mean()
                            std_val = data[col].std()
                            lower_bound = mean_val - outlier_std * std_val
                            upper_bound = mean_val + outlier_std * std_val
                            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
            
            # Record memory usage after preprocessing
            for split_name, data in dataset.items():
                if hasattr(data, 'memory_usage'):
                    preprocessing_summary["memory_usage_after"][split_name] = data.memory_usage(deep=True).sum() / 1024 / 1024
            
            preprocessing_time = time.time() - start_time
            preprocessing_summary.update({
                "preprocessing_steps": preprocessing_steps,
                "preprocessing_time": preprocessing_time,
                "preprocessing_config": {
                    "handle_nan": self.config.get("handle_nan"),
                    "handle_inf": self.config.get("handle_inf"),
                    "normalize_data": self.config.get("normalize_data", False),
                    "remove_outliers": self.config.get("remove_outliers", False),
                    "outlier_std": self.config.get("outlier_std", 3.0)
                }
            })
            
            logger.info(f"Preprocessing completed in {preprocessing_time:.2f}s")
            logger.info(f"Applied steps: {preprocessing_steps}")
            
            return preprocessing_summary
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise

def run_phase_1_data_validation(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Phase 1: Data Validation & Preprocessing.
    
    Args:
        config: Experiment configuration dictionary
        
    Returns:
        Dict containing validation results and statistics
    """
    logger.info("Starting Phase 1: Data Validation & Preprocessing")
    
    # Extract configuration
    dataset_path = Path(config["dataset_path"])
    phase_dir = Path(config["phase_dir"])
    seed = config["seed"]
    test_mode = config["test_mode"]
    
    results = {
        "phase": "phase_1_data_validation",
        "seed": seed,
        "test_mode": test_mode,
        "status": "completed",
        "timestamp": None,
        "data_quality_report": {},
        "preprocessing_summary": {},
        "memory_usage": {},
        "validation_checks": {},
        "execution_time": 0.0
    }
    
    start_time = time.time()
    
    try:
        # Initialize data validator
        validator = DataValidator(config)
        
        # Load dataset
        dataset = validator.load_dataset()
        
        # Get sampling info if available
        sampling_info = getattr(validator, '_last_sampling_info', None)
        
        # Get sampled data path from validator
        sampled_data_path = validator.dataset_path / f"sampled_{validator.test_mode}_{validator.seed}"
        
        # Perform data quality assessment
        quality_report = validator.validate_data_quality(dataset)
        
        # Apply preprocessing
        preprocessing_summary = validator.preprocess_data(dataset)
        
        # Verify label conversion before saving
        for split_name, data in dataset.items():
            if "label" in data.columns:
                unique_labels = data["label"].unique()
                logger.info(f"Final labels in {split_name}: {unique_labels}")
                if not np.array_equal(np.sort(unique_labels), [0, 1, 2, 3, 4]):
                    logger.warning(f"Label conversion verification failed for {split_name}. Expected [0,1,2,3,4], got {np.sort(unique_labels)}")
                    # Force conversion if needed
                    data["label"] = data["label"] - 1
                    logger.info(f"Force converted labels in {split_name}: {data['label'].unique()}")
        
        # Update results
        execution_time = time.time() - start_time
        results.update({
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_quality_report": quality_report,
            "preprocessing_summary": preprocessing_summary,
            "sampling_info": sampling_info,
            "memory_usage": quality_report["memory_usage"],
            "validation_checks": quality_report["validation_checks"],
            "execution_time": execution_time,
            "processed_data": {
                "train_text": str(sampled_data_path / "train_text" / "train_text_data.csv") if "train_text" in dataset else None,
                "train_metadata": str(sampled_data_path / "train_metadata" / "train_metadata_data.csv") if "train_metadata" in dataset else None,
                "test_text": str(sampled_data_path / "test_text" / "test_text_data.csv") if "test_text" in dataset else None,
                "test_metadata": str(sampled_data_path / "test_metadata" / "test_metadata_data.csv") if "test_metadata" in dataset else None,
                "task_type": "classification",
                "num_classes": 5,  # Amazon Reviews: 5-star rating (converted to 0-4)
                "class_names": ["1-star", "2-star", "3-star", "4-star", "5-star"]
            },
            "label_analysis": {
                "train_labels": dataset["train_text"]["label"].unique().tolist() if "train_text" in dataset and "label" in dataset["train_text"].columns else None,
                "test_labels": dataset["test_text"]["label"].unique().tolist() if "test_text" in dataset and "label" in dataset["test_text"].columns else None,
                "label_range": {
                    "min": int(dataset["train_text"]["label"].min()) if "train_text" in dataset and "label" in dataset["train_text"].columns else None,
                    "max": int(dataset["train_text"]["label"].max()) if "train_text" in dataset and "label" in dataset["train_text"].columns else None
                }
            }
        })
        
        # Create output directory if it doesn't exist
        phase_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_file = phase_dir / "phase_1_data_validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save detailed reports
        quality_report_file = phase_dir / "data_quality_report.json"
        with open(quality_report_file, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        preprocessing_file = phase_dir / "preprocessing_summary.json"
        with open(preprocessing_file, 'w') as f:
            json.dump(preprocessing_summary, f, indent=2, default=str)
        
        # Log summary
        logger.info(f"Phase 1 completed successfully in {execution_time:.2f}s")
        logger.info(f"Quality score: {quality_report['quality_score']:.2f}/100")
        logger.info(f"Validation checks: {quality_report['validation_checks']}")
        logger.info(f"Results saved to {phase_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Phase 1 failed: {str(e)}")
        results.update({
            "status": "failed",
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat(),
            "execution_time": time.time() - start_time
        })
        
        # Save error results
        results_file = phase_dir / "phase_1_data_validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results

if __name__ == "__main__":
    # Test the phase script independently
    test_config = {
        "dataset_path": "/Users/pranav/Coding/Projects/PolyModalEnsemble/Benchmarking/ProcessedData/AmazonReviews",
        "phase_dir": "./test_output",
        "seed": 42,
        "test_mode": "quick",  # Will sample 10k samples (8k train, 2k test). Use "full" for 300k samples (240k train, 60k test)
        # dataset_subset_ratio is no longer used - sampling determined by test_mode
        "handle_nan": "fill_mean",
        "handle_inf": "fill_max",
        "normalize_data": False,
        "remove_outliers": False,
        "outlier_std": 3.0
    }
    
    # Create test output directory
    Path(test_config["phase_dir"]).mkdir(exist_ok=True)
    
    # Run the phase
    results = run_phase_1_data_validation(test_config)
    print(f"Phase 1 Results: {json.dumps(results, indent=2, default=str)}")
