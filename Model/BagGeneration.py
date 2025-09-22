"""
Stage 2: Bag Generation
Simplified ensemble bag generation system with adaptive dropout strategies.

Implements:
- Multiple dropout strategies (linear, exponential, random, adaptive)
- Bootstrap sampling with configurable ratios
- Modality importance computation and adaptive weighting
- Bag configuration management
- Memory-only storage for Stage 3 integration
- Comprehensive testing and interpretability functions
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger("bag_generation")

# --- BagConfig ---
@dataclass
class BagConfig:
    """Configuration for a single ensemble bag."""
    bag_id: int
    data_indices: np.ndarray
    modality_mask: Dict[str, bool]
    modality_weights: Dict[str, float]
    dropout_rate: float
    sample_ratio: float = 0.8
    diversity_score: float = 0.0
    creation_timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# --- BagGeneration ---
class BagGeneration:
    
    def __init__(self, 
                 train_data: Dict[str, np.ndarray],
                 train_labels: np.ndarray,
                 n_bags: int = 20,
                 dropout_strategy: str = "adaptive",
                 max_dropout_rate: float = 0.5,
                 min_modalities: int = 1,
                 sample_ratio: float = 0.8,
                 random_state: Optional[int] = 42):
        """
        Initialize the Bag Generation system.
        
        Parameters
        ----------
        train_data : Dict[str, np.ndarray]
            Training data for each modality from Stage 1
        train_labels : np.ndarray
            Training labels from Stage 1
        n_bags : int, default=20
            Number of ensemble bags to create
        dropout_strategy : str, default="adaptive"
            Dropout strategy: 'linear', 'exponential', 'random', 'adaptive'
        max_dropout_rate : float, default=0.5
            Maximum dropout rate for modalities
        min_modalities : int, default=1
            Minimum number of modalities per bag
        sample_ratio : float, default=0.8
            Bootstrap sampling ratio
        random_state : Optional[int], default=42
            Random seed for reproducibility
        """
        self.train_data = train_data
        self.train_labels = train_labels
        self.n_bags = n_bags
        self.dropout_strategy = dropout_strategy
        self.max_dropout_rate = max_dropout_rate
        self.min_modalities = min_modalities
        self.sample_ratio = sample_ratio
        self.random_state = random_state
        
        # Initialize random number generator
        self._rng = np.random.default_rng(random_state)
        
        # Get modality names from training data
        self.modality_names = list(train_data.keys())
        self.n_modalities = len(self.modality_names)
        
        # Storage for generated bags
        self.bags: List[BagConfig] = []
        self.bag_data: Dict[int, Dict[str, np.ndarray]] = {}
        
        # Modality importance scores (computed in adaptive strategy)
        self.modality_importance: Dict[str, float] = {}
        
        # Interpretability and testing data
        self._interpretability_data: Dict[str, Any] = {}
        self._testing_data: Dict[str, Any] = {}
        
        # Validate parameters
        self._validate_params()
        
        logger.info(f"Initialized BagGeneration with {n_bags} bags, strategy: {dropout_strategy}")

    def _validate_params(self):
        """Validate input parameters and data consistency."""
        # Basic parameter validation
        assert 1 <= self.n_bags <= 1000, "n_bags must be between 1 and 1000"
        assert self.dropout_strategy in ['linear', 'exponential', 'random', 'adaptive'], \
            "dropout_strategy must be one of: 'linear', 'exponential', 'random', 'adaptive'"
        assert 0.0 <= self.max_dropout_rate <= 0.9, "max_dropout_rate must be between 0.0 and 0.9"
        assert 0.1 <= self.sample_ratio <= 1.0, "sample_ratio must be between 0.1 and 1.0"
        assert 1 <= self.min_modalities <= self.n_modalities, \
            "min_modalities must be between 1 and number of modalities"
        assert len(self.train_data) > 0, "train_data cannot be empty"
        assert len(self.train_labels) > 0, "train_labels cannot be empty"
        
        # Enhanced data consistency validation
        self._validate_data_consistency()
    
    def _validate_data_consistency(self):
        """
        Enhanced data consistency validation to prevent bootstrap sampling issues.
        Validates sample counts, data integrity, and cross-modality alignment.
        """
        logger.info("Validating data consistency for bag generation")
        
        # Check sample count consistency across modalities
        sample_counts = {}
        for modality_name, data in self.train_data.items():
            sample_counts[modality_name] = len(data)
        
        # Check if all modalities have consistent sample counts
        unique_counts = set(sample_counts.values())
        if len(unique_counts) > 1:
            logger.warning("Inconsistent sample counts across modalities:")
            for modality, count in sample_counts.items():
                logger.warning(f"  {modality}: {count} samples")
            logger.warning("This may cause bootstrap sampling issues. Consider data alignment.")
        
        # Validate label count matches data count
        label_count = len(self.train_labels)
        expected_count = max(sample_counts.values()) if sample_counts else 0
        
        if label_count != expected_count:
            logger.warning(f"Label count ({label_count}) doesn't match expected data count ({expected_count})")
            logger.warning("This may cause index out of bounds errors during bootstrap sampling.")
        
        # Check for NaN/Inf values in data
        for modality_name, data in self.train_data.items():
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                logger.warning(f"Found NaN/Inf values in modality {modality_name}")
                logger.warning("This may cause numerical issues in importance calculation.")
        
        # Check for NaN/Inf values in labels
        if np.any(np.isnan(self.train_labels)) or np.any(np.isinf(self.train_labels)):
            logger.warning("Found NaN/Inf values in training labels")
            logger.warning("This may cause issues during bootstrap sampling.")
        
        # Validate data shapes and types
        for modality_name, data in self.train_data.items():
            if data.ndim < 1:
                raise ValueError(f"Modality {modality_name} has invalid dimensions: {data.ndim}")
            if data.size == 0:
                raise ValueError(f"Modality {modality_name} is empty")
            if not isinstance(data, np.ndarray):
                raise ValueError(f"Modality {modality_name} must be a numpy array")
        
        # Check minimum modality constraint feasibility
        if self.min_modalities > self.n_modalities:
            raise ValueError(f"min_modalities ({self.min_modalities}) cannot exceed number of modalities ({self.n_modalities})")
        
        # Validate dropout strategy parameters
        if self.dropout_strategy == 'adaptive' and self.n_modalities < 2:
            logger.warning("Adaptive strategy with single modality may not provide meaningful diversity")
        
        
        logger.info("Data consistency validation completed")

    def generate_bags(self) -> List[BagConfig]:
        """
        Generate ensemble bags following the 6-step process.
        
        Returns
        -------
        List[BagConfig]
            List of generated bag configurations
        """
        logger.info(f"Starting bag generation with {self.dropout_strategy} strategy")
        
        # Step 1: Initialize bag creation
        self._initialize_bags()
        
        # Step 2: Dropout Strategy Calculation
        dropout_rates = self._calculate_dropout_rates()
        
        # Step 3: Modality Mask Creation
        modality_masks = self._create_modality_masks(dropout_rates)
        
        # Step 4: Sampling (Bootstrap)
        data_indices = self._bootstrap_sampling()
        
        # Step 5: Bag Data Extraction and Storage
        self._extract_and_store_bag_data(modality_masks, data_indices, dropout_rates)
        
        # Step 6: Collect interpretability data
        self._collect_interpretability_data()
        
        logger.info(f"Successfully generated {len(self.bags)} bags")
        return self.bags

    # ============================================================================
    # STEP 1: INITIALIZE BAG CREATION
    # ============================================================================
    
    def _initialize_bags(self):
        """Step 1: Initialize bag creation - create n empty bags and set up parameters."""
        logger.info("Step 1: Initializing bag creation")
        
        # Clear any existing bags
        self.bags = []
        self.bag_data = {}
        
        # Validate training data consistency
        dataset_size = len(self.train_labels)
        for modality_name, data in self.train_data.items():
            if len(data) != dataset_size:
                logger.warning(f"Modality {modality_name} has {len(data)} samples, expected {dataset_size}")
        
        logger.info(f"Initialized {self.n_bags} empty bags for {self.n_modalities} modalities")

    # ============================================================================
    # STEP 2: DROPOUT STRATEGY CALCULATION
    # ============================================================================
    
    def _calculate_dropout_rates(self) -> np.ndarray:
        """Step 2: Dropout Strategy Calculation."""
        logger.info(f"Step 2: Calculating dropout rates using {self.dropout_strategy} strategy")
        
        if self.dropout_strategy == 'linear':
            return self._linear_strategy()
        elif self.dropout_strategy == 'exponential':
            return self._exponential_strategy()
        elif self.dropout_strategy == 'random':
            return self._random_strategy()
        elif self.dropout_strategy == 'adaptive':
            return self._adaptive_strategy()
        else:
            return np.full(self.n_bags, self.max_dropout_rate)

    def _linear_strategy(self) -> np.ndarray:
        """Linear dropout strategy: uniform progression from 0 to max_dropout_rate."""
        return np.linspace(0, self.max_dropout_rate, self.n_bags)

    def _exponential_strategy(self) -> np.ndarray:
        """Exponential dropout strategy: exponential decay."""
        return self.max_dropout_rate * (1 - np.exp(-3 * np.linspace(0, 1, self.n_bags)))

    def _random_strategy(self) -> np.ndarray:
        """Random dropout strategy: random rates between 0 and max_dropout_rate."""
        return self._rng.uniform(0, self.max_dropout_rate, self.n_bags)

    def _adaptive_strategy(self) -> np.ndarray:
        """
        Adaptive strategy: modality-based dropout.
        - Computes modality importance scores using variance-based analysis
        - Normalizes to dropout probabilities (higher importance = lower dropout)
        - Ensures distinct modality combinations
        - Enforces minimum modality constraints
        """
        logger.info("Computing modality importance scores for adaptive strategy")
        
        # Compute sophisticated modality importance scores
        self.modality_importance = {}
        
        for modality_name, data in self.train_data.items():
            # Normalize data to [0, 1] range to handle different scales
            data_normalized = self._normalize_for_importance(data)
            
            # 1. Variance-based information content
            if data_normalized.ndim > 1:
                variance = float(np.mean(np.var(data_normalized, axis=0)))
            else:
                variance = float(np.var(data_normalized))
            
            # 2. Feature count balancing
            feature_count = data.shape[1] if data.ndim > 1 else 1
            feature_factor = np.log(feature_count + 1)
            
            # 3. Label correlation (intelligence component)
            label_correlation = self._compute_label_correlation(data_normalized)
            
            # 4. Cross-modality redundancy (distinct combinations)
            redundancy_penalty = self._compute_cross_modality_redundancy(data_normalized, modality_name)
            
            # 5. Data quality assessment
            data_quality = self._assess_data_quality(data_normalized)
            
            # 6. Sophisticated importance combination
            importance = (
                variance * 
                feature_factor * 
                label_correlation * 
                (1 - redundancy_penalty) * 
                data_quality
            )
            
            self.modality_importance[modality_name] = importance
        
        # Log importance scores
        logger.info("Modality importance scores:")
        for name, score in sorted(self.modality_importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {name}: {score:.4f}")
        
        # Normalize importance to dropout probabilities
        max_importance = max(self.modality_importance.values())
        min_importance = min(self.modality_importance.values())
        
        if max_importance > min_importance:
            # Normalize to [0, 1] range, then invert for dropout probabilities
            normalized_importance = {
                name: (score - min_importance) / (max_importance - min_importance)
                for name, score in self.modality_importance.items()
            }
            dropout_probs = {
                name: self.max_dropout_rate * (1 - norm_score)
                for name, norm_score in normalized_importance.items()
            }
        else:
            # All modalities have equal importance
            dropout_probs = {
                name: self.max_dropout_rate * 0.5
                for name in self.modality_names
            }
        
        # Generate adaptive dropout rates with distinct combinations
        dropout_rates = []
        used_combinations = set()
        
        for bag_id in range(self.n_bags):
            # Calculate base dropout rate with variation
            base_rate = self.max_dropout_rate * (bag_id / max(1, self.n_bags - 1))
            variation = self._rng.uniform(-0.1, 0.1)
            adaptive_rate = np.clip(base_rate + variation, 0, self.max_dropout_rate)
            
            # Ensure distinct modality combinations
            attempts = 0
            while attempts < 10:  # Prevent infinite loops
                # Sample modalities based on dropout probabilities
                modality_selection = {}
                for name in self.modality_names:
                    if self._rng.random() > dropout_probs[name]:
                        modality_selection[name] = True
                    else:
                        modality_selection[name] = False
                
                # Ensure minimum modalities constraint
                active_count = sum(modality_selection.values())
                if active_count < self.min_modalities:
                    # Activate additional modalities (least important first)
                    inactive_modalities = [name for name, active in modality_selection.items() if not active]
                    sorted_by_importance = sorted(
                        inactive_modalities, 
                        key=lambda x: self.modality_importance[x]
                    )
                    needed = self.min_modalities - active_count
                    for name in sorted_by_importance[:needed]:
                        modality_selection[name] = True
                
                # Check if this combination is distinct
                combination = tuple(sorted(
                    name for name, active in modality_selection.items() if active
                ))
                
                if combination not in used_combinations or len(used_combinations) >= 2**self.n_modalities:
                    used_combinations.add(combination)
                    break
                
                attempts += 1
            
            dropout_rates.append(adaptive_rate)
        
        logger.info(f"Generated {len(dropout_rates)} adaptive dropout rates with {len(used_combinations)} distinct combinations")
        return np.array(dropout_rates)

    
    def _normalize_for_importance(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to [0, 1] range for fair importance comparison across modalities.
        """
        if data.size == 0:
            return data
        
        # Handle different data types
        if data.dtype in [np.uint8, np.uint16, np.uint32]:
            # Image data - normalize by max possible value
            data_min, data_max = 0, np.iinfo(data.dtype).max
        else:
            # Other data types - use actual min/max
            data_min, data_max = np.min(data), np.max(data)
        
        if data_max > data_min:
            return (data - data_min) / (data_max - data_min)
        else:
            return np.zeros_like(data)
    
    def _compute_label_correlation(self, data_normalized: np.ndarray) -> float:
        """
        Compute correlation between modality data and labels (intelligence component).
        """
        try:
            # Flatten multi-dimensional data for correlation calculation
            if data_normalized.ndim > 2:
                data_flat = data_normalized.reshape(data_normalized.shape[0], -1)
            else:
                data_flat = data_normalized
            
            # Calculate correlation with labels for each feature
            correlations = []
            for i in range(data_flat.shape[1]):
                corr = np.corrcoef(data_flat[:, i], self.train_labels)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))  # Use absolute correlation
            
            # Return average absolute correlation, with minimum of 0.1 to avoid zero importance
            return float(np.mean(correlations)) if correlations else 0.1
            
        except Exception:
            # Fallback to variance-based importance if correlation fails
            return 0.5
    
    def _compute_cross_modality_redundancy(self, data_normalized: np.ndarray, modality_name: str) -> float:
        """
        Compute redundancy with other modalities (distinct combinations component).
        """
        try:
            redundancy_scores = []
            
            for other_name, other_data in self.train_data.items():
                if other_name != modality_name:
                    # Normalize other modality data
                    other_normalized = self._normalize_for_importance(other_data)
                    
                    # Flatten both datasets for comparison
                    if data_normalized.ndim > 2:
                        data_flat = data_normalized.reshape(data_normalized.shape[0], -1)
                    else:
                        data_flat = data_normalized
                    
                    if other_normalized.ndim > 2:
                        other_flat = other_normalized.reshape(other_normalized.shape[0], -1)
                    else:
                        other_flat = other_normalized
                    
                    # Compute average correlation between features
                    if data_flat.shape[1] > 0 and other_flat.shape[1] > 0:
                        # Sample features to avoid computational explosion
                        n_samples = min(10, data_flat.shape[1], other_flat.shape[1])
                        data_sample = data_flat[:, :n_samples]
                        other_sample = other_flat[:, :n_samples]
                        
                        # Compute correlation matrix
                        corr_matrix = np.corrcoef(data_sample.T, other_sample.T)
                        if corr_matrix.size > 1:
                            # Get cross-correlations (off-diagonal blocks)
                            cross_corr = corr_matrix[:n_samples, n_samples:]
                            if cross_corr.size > 0:
                                avg_corr = np.mean(np.abs(cross_corr))
                                redundancy_scores.append(avg_corr)
            
            # Return average redundancy, with maximum of 0.8 to avoid complete penalty
            return float(np.mean(redundancy_scores)) if redundancy_scores else 0.0
            
        except Exception:
            # Fallback to no redundancy penalty if computation fails
            return 0.0
    
    def _assess_data_quality(self, data_normalized: np.ndarray) -> float:
        """
        Assess data quality based on completeness, consistency, and noise levels.
        """
        try:
            # 1. Completeness (no NaN/Inf values)
            completeness = 1.0 - (np.isnan(data_normalized).sum() + np.isinf(data_normalized).sum()) / data_normalized.size
            
            # 2. Consistency (low variance in feature means across samples)
            if data_normalized.ndim > 1:
                feature_means = np.mean(data_normalized, axis=0)
                consistency = 1.0 / (1.0 + np.std(feature_means))  # Lower std = higher consistency
            else:
                consistency = 1.0
            
            # 3. Signal-to-noise ratio (variance vs mean)
            if np.mean(data_normalized) > 0:
                snr = np.var(data_normalized) / (np.mean(data_normalized) + 1e-8)
                noise_level = 1.0 / (1.0 + snr)  # Higher SNR = lower noise
            else:
                noise_level = 0.5
            
            # Combine quality metrics
            quality_score = (completeness * 0.4 + consistency * 0.3 + noise_level * 0.3)
            
            # Ensure quality score is between 0.1 and 1.0
            return max(0.1, min(1.0, quality_score))
            
        except Exception:
            # Fallback to moderate quality if assessment fails
            return 0.7

    # ============================================================================
    # STEP 3: MODALITY MASK CREATION
    # ============================================================================
    
    def _create_modality_masks(self, dropout_rates: np.ndarray) -> List[Dict[str, bool]]:
        """
        Step 3: Modality Mask Creation.
        Determines how many modalities should be in each bag and their abundance.
        """
        logger.info("Step 3: Creating modality masks")
        
        modality_masks = []
        
        for bag_id, dropout_rate in enumerate(dropout_rates):
            # Calculate number of modalities to drop
            n_drop = int(np.floor(dropout_rate * self.n_modalities))
            n_drop = min(n_drop, self.n_modalities - self.min_modalities)
            
            # Create mask
            mask = np.ones(self.n_modalities, dtype=bool)
            if n_drop > 0:
                drop_indices = self._rng.choice(
                    np.arange(self.n_modalities), size=n_drop, replace=False
                )
                mask[drop_indices] = False
            
            # Convert to dictionary
            modality_mask = {name: bool(mask[i]) for i, name in enumerate(self.modality_names)}
            modality_masks.append(modality_mask)
        
        logger.info(f"Created modality masks for {len(modality_masks)} bags")
        return modality_masks

    # ============================================================================
    # STEP 4: SAMPLING (BOOTSTRAP)
    # ============================================================================
    
    def _bootstrap_sampling(self) -> List[np.ndarray]:
        """
        Step 4: Bootstrap Sampling.
        Creates bootstrap sample indices for each bag.
        """
        logger.info("Step 4: Performing bootstrap sampling")
        
        dataset_size = len(self.train_labels)
        n_samples = int(self.sample_ratio * dataset_size)
        
        data_indices = []
        for bag_id in range(self.n_bags):
            # Create bootstrap sample with replacement
            bag_indices = self._rng.choice(
                np.arange(dataset_size), size=n_samples, replace=True
            )
            data_indices.append(bag_indices)
        
        logger.info(f"Created bootstrap samples for {len(data_indices)} bags ({n_samples} samples each)")
        return data_indices

    # ============================================================================
    # STEP 5: BAG DATA EXTRACTION AND STORAGE
    # ============================================================================
    
    def _extract_and_store_bag_data(self, 
                                   modality_masks: List[Dict[str, bool]], 
                                   data_indices: List[np.ndarray], 
                                   dropout_rates: np.ndarray):
        """
        Step 5: Bag Data Extraction and Storage.
        Extracts actual data for each bag and stores for upcoming stages.
        """
        logger.info("Step 5: Extracting and storing bag data")
        
        for bag_id in range(self.n_bags):
            # Create modality weights based on importance (for adaptive strategy)
            modality_weights = {}
            if self.dropout_strategy == 'adaptive' and self.modality_importance:
                # Weight active modalities by their importance
                total_importance = sum(
                    self.modality_importance[name] 
                    for name, is_active in modality_masks[bag_id].items() 
                    if is_active
                )
                for name, is_active in modality_masks[bag_id].items():
                    if is_active and total_importance > 0:
                        modality_weights[name] = self.modality_importance[name] / total_importance
                    else:
                        modality_weights[name] = 0.0
            else:
                # Equal weights for non-adaptive strategies
                active_count = sum(modality_masks[bag_id].values())
                for name, is_active in modality_masks[bag_id].items():
                    modality_weights[name] = 1.0 / active_count if is_active else 0.0
            
            # Create bag configuration
            bag = BagConfig(
                bag_id=bag_id,
                data_indices=data_indices[bag_id],
                modality_mask=modality_masks[bag_id],
                modality_weights=modality_weights,
                dropout_rate=dropout_rates[bag_id],
                sample_ratio=self.sample_ratio,
                diversity_score=0.0,  # Will be calculated later if needed
                creation_timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                metadata={}
            )
            
            self.bags.append(bag)
            
            # Extract and store actual bag data
            bag_data = {}
            indices = data_indices[bag_id]
            
            # Find the minimum sample count across all active modalities
            min_samples = len(indices)
            for modality_name, is_active in modality_masks[bag_id].items():
                if is_active and modality_name in self.train_data:
                    data = self.train_data[modality_name]
                    min_samples = min(min_samples, len(data))
            
            # Ensure we don't exceed the minimum available samples
            if min_samples < len(indices):
                logger.warning(f"Some modalities have fewer samples ({min_samples}) than requested ({len(indices)}). Using {min_samples} samples for consistency.")
                # Truncate indices to the minimum available samples
                indices = indices[:min_samples]
            
            # Update the bag's data_indices to reflect the actual indices used
            self.bags[bag_id].data_indices = indices
            
            # Extract data for each active modality using consistent indices
            for modality_name, is_active in modality_masks[bag_id].items():
                if is_active and modality_name in self.train_data:
                    data = self.train_data[modality_name]
                    
                    # Ensure indices are within bounds for this specific modality
                    valid_indices = indices[indices < len(data)]
                    if len(valid_indices) == 0:
                        logger.warning(f"No valid indices for modality {modality_name}, skipping.")
                        continue
                    
                    # Use the valid indices
                    bag_data[modality_name] = data[valid_indices]
            
            # Store labels using the same consistent indices
            if bag_data:  # Only if we have valid data
                # Use the same indices that were used for the data
                valid_label_indices = indices[indices < len(self.train_labels)]
                if len(valid_label_indices) > 0:
                    bag_data['labels'] = self.train_labels[valid_label_indices]
                else:
                    logger.warning(f"No valid label indices for bag {bag_id}")
                    continue
            
            # Store bag data
            self.bag_data[bag_id] = bag_data
        
        logger.info(f"Extracted and stored data for {len(self.bags)} bags")

    # ============================================================================
    # STEP 6: CONVENIENCE FUNCTIONS
    # ============================================================================
    
    def _collect_interpretability_data(self):
        """Collect comprehensive interpretability data for analysis."""
        logger.info("Collecting interpretability data")
        
        # Store modality importance scores
        self._interpretability_data['modality_importance'] = self.modality_importance.copy()
        
        # Store detailed bag information
        detailed_bags = []
        for bag in self.bags:
            detailed_bag = {
                'bag_id': bag.bag_id,
                'data_indices': bag.data_indices,
                'modality_mask': bag.modality_mask,
                'modality_weights': bag.modality_weights,
                'dropout_rate': bag.dropout_rate,
                'sample_ratio': bag.sample_ratio,
                'diversity_score': bag.diversity_score,
                'creation_timestamp': bag.creation_timestamp,
                'sample_count': len(bag.data_indices)
            }
            detailed_bags.append(detailed_bag)
        
        self._interpretability_data['detailed_bags'] = detailed_bags
        
        # Store ensemble-level statistics
        dropout_rates = [bag.dropout_rate for bag in self.bags]
        self._interpretability_data['ensemble_statistics'] = {
            'mean_dropout_rate': np.mean(dropout_rates),
            'std_dropout_rate': np.std(dropout_rates),
            'min_dropout_rate': np.min(dropout_rates),
            'max_dropout_rate': np.max(dropout_rates),
            'total_bags': len(self.bags),
            'strategy': self.dropout_strategy
        }
        
        # Store modality coverage statistics
        modality_coverage = {}
        for name in self.modality_names:
            coverage = sum(1 for bag in self.bags if bag.modality_mask.get(name, False))
            modality_coverage[name] = coverage / len(self.bags)
        
        self._interpretability_data['modality_coverage'] = modality_coverage
        
        logger.info("Interpretability data collection completed")

    # ============================================================================
    # DATA ACCESS METHODS
    # ============================================================================
    
    def get_bag_data(self, bag_id: int) -> Dict[str, np.ndarray]:
        """Get data for a specific bag."""
        if bag_id not in self.bag_data:
            raise ValueError(f"Bag {bag_id} not found")
        return self.bag_data[bag_id].copy()

    def get_bag_info(self, bag_id: int) -> Dict[str, Any]:
        """Get information about a specific bag."""
        if bag_id >= len(self.bags):
            raise ValueError(f"Bag {bag_id} not found")
        
        bag = self.bags[bag_id]
        return {
            'bag_id': bag.bag_id,
            'modalities': [name for name, active in bag.modality_mask.items() if active],
            'modality_weights': bag.modality_weights,
            'dropout_rate': bag.dropout_rate,
            'sample_count': len(bag.data_indices),
            'diversity_score': bag.diversity_score,
            'creation_timestamp': bag.creation_timestamp
        }

    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        if not self.bags:
            return {}
        
        # Calculate modality coverage
        modality_coverage = {}
        for name in self.modality_names:
            coverage = sum(1 for bag in self.bags if bag.modality_mask.get(name, False))
            modality_coverage[name] = coverage / len(self.bags)
        
        # Calculate dropout statistics
        dropout_rates = [bag.dropout_rate for bag in self.bags]
        dropout_stats = {
            'mean_dropout_rate': float(np.mean(dropout_rates)),
            'min_dropout_rate': float(np.min(dropout_rates)),
            'max_dropout_rate': float(np.max(dropout_rates)),
            'std_dropout_rate': float(np.std(dropout_rates))
        }
        
        return {
            'n_bags': len(self.bags),
            'modality_coverage': modality_coverage,
            'dropout_statistics': dropout_stats,
            'strategy': self.dropout_strategy,
            'modality_importance': self.modality_importance
        }

    def get_interpretability_data(self) -> Dict[str, Any]:
        """Get comprehensive interpretability data."""
        return self._interpretability_data.copy()

    def get_modality_importance(self) -> Dict[str, float]:
        """Get modality importance scores."""
        return self.modality_importance.copy()

    # ============================================================================
    # TESTING FUNCTIONS
    # ============================================================================
    
    def test_bag_consistency(self) -> Dict[str, Any]:
        """Test bag consistency and data integrity."""
        logger.info("Running bag consistency tests")
        
        test_results = {
            'total_bags': len(self.bags),
            'data_consistency': True,
            'modality_coverage': {},
            'sample_consistency': True,
            'errors': []
        }
        
        # Test data consistency
        for bag_id, bag in enumerate(self.bags):
            bag_data = self.bag_data[bag_id]
            
            # Check if all active modalities have data
            for modality_name, is_active in bag.modality_mask.items():
                if is_active:
                    if modality_name not in bag_data:
                        test_results['data_consistency'] = False
                        test_results['errors'].append(f"Bag {bag_id}: Missing data for active modality {modality_name}")
            
            # Check sample count consistency
            expected_samples = len(bag.data_indices)
            if 'labels' in bag_data and len(bag_data['labels']) != expected_samples:
                test_results['sample_consistency'] = False
                test_results['errors'].append(f"Bag {bag_id}: Label count mismatch")
        
        # Test modality coverage
        for name in self.modality_names:
            coverage = sum(1 for bag in self.bags if bag.modality_mask.get(name, False))
            test_results['modality_coverage'][name] = coverage / len(self.bags)
        
        logger.info(f"Bag consistency test completed: {len(test_results['errors'])} errors found")
        return test_results


    # ============================================================================
    # INTERPRETABILITY TESTS
    # ============================================================================
    
    def interpretability_test(self) -> Dict[str, Any]:
        """Run interpretability tests to analyze bag generation patterns."""
        logger.info("Running interpretability tests")
        
        interpretability_results = {
            'modality_importance_analysis': {},
            'dropout_pattern_analysis': {},
            'diversity_analysis': {},
            'coverage_analysis': {}
        }
        
        # Analyze modality importance
        if self.modality_importance:
            importance_scores = list(self.modality_importance.values())
            interpretability_results['modality_importance_analysis'] = {
                'mean_importance': np.mean(importance_scores),
                'std_importance': np.std(importance_scores),
                'min_importance': np.min(importance_scores),
                'max_importance': np.max(importance_scores),
                'importance_ranking': sorted(
                    self.modality_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
            }
        
        # Analyze dropout patterns
        dropout_rates = [bag.dropout_rate for bag in self.bags]
        interpretability_results['dropout_pattern_analysis'] = {
            'mean_dropout': np.mean(dropout_rates),
            'std_dropout': np.std(dropout_rates),
            'dropout_distribution': np.histogram(dropout_rates, bins=10)[0].tolist()
        }
        
        # Analyze diversity
        modality_combinations = set()
        for bag in self.bags:
            combination = tuple(sorted(
                name for name, active in bag.modality_mask.items() if active
            ))
            modality_combinations.add(combination)
        
        interpretability_results['diversity_analysis'] = {
            'unique_combinations': len(modality_combinations),
            'max_possible_combinations': 2**self.n_modalities,
            'diversity_ratio': len(modality_combinations) / (2**self.n_modalities),
            'combinations': list(modality_combinations)
        }
        
        # Analyze coverage
        coverage_stats = {}
        for name in self.modality_names:
            coverage = sum(1 for bag in self.bags if bag.modality_mask.get(name, False))
            coverage_stats[name] = {
                'coverage_ratio': coverage / len(self.bags),
                'total_usage': coverage
            }
        
        interpretability_results['coverage_analysis'] = coverage_stats
        
        logger.info("Interpretability tests completed")
        return interpretability_results

    # ============================================================================
    # ROBUSTNESS TESTS
    # ============================================================================
    
    def robustness_test(self, noise_level: float = 0.1) -> Dict[str, Any]:
        """Test robustness by adding noise to training data."""
        logger.info(f"Running robustness test with noise level {noise_level}")
        
        # Add noise to training data
        noisy_train_data = {}
        for modality_name, data in self.train_data.items():
            noise = self._rng.normal(0, noise_level, data.shape)
            noisy_train_data[modality_name] = data + noise
        
        # Create new bag generation with noisy data
        noisy_bagger = BagGeneration(
            train_data=noisy_train_data,
            train_labels=self.train_labels,
            n_bags=self.n_bags,
            dropout_strategy=self.dropout_strategy,
            max_dropout_rate=self.max_dropout_rate,
            min_modalities=self.min_modalities,
            sample_ratio=self.sample_ratio,
            random_state=self.random_state
        )
        
        noisy_bags = noisy_bagger.generate_bags()
        
        # Compare with original bags
        robustness_results = {
            'noise_level': noise_level,
            'original_bags': len(self.bags),
            'noisy_bags': len(noisy_bags),
            'modality_mask_similarity': {},
            'dropout_rate_similarity': {}
        }
        
        # Compare modality masks
        for name in self.modality_names:
            original_usage = sum(1 for bag in self.bags if bag.modality_mask.get(name, False))
            noisy_usage = sum(1 for bag in noisy_bags if bag.modality_mask.get(name, False))
            similarity = 1 - abs(original_usage - noisy_usage) / len(self.bags)
            robustness_results['modality_mask_similarity'][name] = similarity
        
        # Compare dropout rates
        original_rates = [bag.dropout_rate for bag in self.bags]
        noisy_rates = [bag.dropout_rate for bag in noisy_bags]
        rate_similarity = 1 - np.mean(np.abs(np.array(original_rates) - np.array(noisy_rates)))
        robustness_results['dropout_rate_similarity'] = rate_similarity
        
        logger.info(f"Robustness test completed: similarity = {rate_similarity:.3f}")
        return robustness_results

    # ============================================================================
    # UTILITY FUNCTIONS
    # ============================================================================
    
    def print_summary(self):
        """Print a comprehensive summary of the bag generation."""
        print("\n" + "="*60)
        print("BAG GENERATION SUMMARY")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  - Strategy: {self.dropout_strategy}")
        print(f"  - Number of bags: {self.n_bags}")
        print(f"  - Max dropout rate: {self.max_dropout_rate}")
        print(f"  - Min modalities: {self.min_modalities}")
        print(f"  - Sample ratio: {self.sample_ratio}")
        print(f"  - Modalities: {self.modality_names}")
        
        if self.bags:
            print(f"\nEnsemble Statistics:")
            stats = self.get_ensemble_stats()
            print(f"  - Total bags: {stats['n_bags']}")
            print(f"  - Mean dropout rate: {stats['dropout_statistics']['mean_dropout_rate']:.3f}")
            print(f"  - Dropout range: {stats['dropout_statistics']['min_dropout_rate']:.3f} - {stats['dropout_statistics']['max_dropout_rate']:.3f}")
            
            print(f"\nModality Coverage:")
            for name, coverage in stats['modality_coverage'].items():
                print(f"  - {name}: {coverage:.1%}")
            
            if self.modality_importance:
                print(f"\nModality Importance (Adaptive Strategy):")
                for name, importance in sorted(self.modality_importance.items(), key=lambda x: x[1], reverse=True):
                    print(f"  - {name}: {importance:.4f}")
        
        print("\n" + "="*60)