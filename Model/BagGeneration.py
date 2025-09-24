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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("bag_generation")

# Configuration Constants
class BagGenerationConfig:
    """Configuration constants for BagGeneration."""
    MAX_DROPOUT_RATE = 0.8
    MIN_MODALITIES = 1
    DEFAULT_SAMPLE_RATIO = 0.8
    DEFAULT_N_BAGS = 20
    DEFAULT_RANDOM_STATE = 42
    
    # Memory optimization constants
    MAX_SAMPLES_FOR_CV = 1000  # Limit samples for cross-validation
    MAX_FEATURES_FOR_REDUNDANCY = 100  # Limit features for redundancy calculation
    MEMORY_WARNING_THRESHOLD = 500 * 1024 * 1024  # 500MB warning threshold

# --- BagConfig ---
@dataclass
class BagConfig:
    """Configuration for a single ensemble bag."""
    bag_id: int
    data_indices: np.ndarray  # dtype: np.int64
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
        
        # Step 2: Calculate Modality Importance
        self._calculate_modality_importance()
        
        # Step 3: Generate Bag Configurations
        dropout_rates = self._calculate_dropout_rates()
        modality_masks = self._create_modality_masks(dropout_rates)
        
        # Step 4: Sampling (Bootstrap)
        data_indices = self._bootstrap_sampling()
        
        # Step 5: Bag Data Extraction and Storage
        self._extract_and_store_bag_data(modality_masks, data_indices, dropout_rates)
        
        # Step 6: Initialize interpretability data storage
        self._interpretability_data = {}
        
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
    # STEP 2: CALCULATE MODALITY IMPORTANCE
    # ============================================================================
    
    def _calculate_modality_importance(self):
        """Step 2: Calculate Modality Importance using 6-factor analysis."""
        logger.info("Step 2: Calculating modality importance using 6-factor analysis")
        
        if self.dropout_strategy == 'adaptive':
            # For adaptive strategy, calculate 6-factor importance
            self.modality_importance = {}
            
            for modality_name, data in self.train_data.items():
                try:
                    # Calculate 6-factor importance for this modality
                    importance = self._calculate_6_factor_importance(data, modality_name)
                    self.modality_importance[modality_name] = importance
                    
                except Exception as e:
                    logger.warning(f"Error calculating 6-factor importance for {modality_name}: {e}")
                    # Fallback to simple variance-based importance
                    data_normalized = self._normalize_for_importance(data)
                    
                    if data_normalized.ndim > 1:
                        variance = float(np.mean(np.var(data_normalized, axis=0)))
                    else:
                        variance = float(np.var(data_normalized))
                    
                    feature_count = data.shape[1] if data.ndim > 1 else 1
                    feature_factor = np.log(feature_count + 1)
                    
                    importance = variance * feature_factor
                    self.modality_importance[modality_name] = importance
            
            # Log 6-factor importance scores
            logger.info("6-Factor importance scores:")
            for name, score in sorted(self.modality_importance.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {name}: {score:.4f}")
        else:
            # For non-adaptive strategies, set equal importance
            self.modality_importance = {name: 1.0 for name in self.modality_names}
            logger.info("Non-adaptive strategy: Using equal importance for all modalities")

    # ============================================================================
    # STEP 3: GENERATE BAG CONFIGURATIONS
    # ============================================================================
    
    def _calculate_dropout_rates(self) -> np.ndarray:
        """Step 3: Calculate dropout rates based on strategy."""
        logger.info(f"Step 3: Calculating dropout rates using {self.dropout_strategy} strategy")
        
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
        Adaptive strategy: modality-based dropout using pre-calculated importance scores.
        - Uses pre-calculated 6-factor importance scores from Step 2
        - Normalizes to dropout probabilities (higher importance = lower dropout)
        - Ensures distinct modality combinations
        - Enforces minimum modality constraints
        """
        logger.info("Using pre-calculated 6-factor importance scores for adaptive strategy")
        
        # Use already calculated importance from Step 2
        if not self.modality_importance:
            raise ValueError("Modality importance must be calculated first in Step 2")
        
        # Log predictive power scores
        logger.info("Using 6-factor importance scores from Step 2:")
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

    def _calculate_6_factor_importance(self, modality_data: np.ndarray, modality_name: str) -> float:
        """
        Calculate sophisticated 6-factor importance score for a modality.
        
        The 6 factors are:
        1. Data Variance - Informativeness of the data
        2. Feature Count - Number of features (log-scaled)
        3. Label Correlation - Correlation with target labels
        4. Cross-Modality Redundancy - Uniqueness compared to other modalities
        5. Data Quality - Completeness, consistency, and noise levels
        6. Predictive Power - Cross-validation performance
        
        Parameters
        ----------
        modality_data : np.ndarray
            Data for the modality
        modality_name : str
            Name of the modality
            
        Returns
        -------
        float
            Combined 6-factor importance score (higher = more important)
        """
        try:
            # Optimization for large datasets: Use sampling for expensive computations
            n_samples = len(modality_data)
            max_samples_for_cv = BagGenerationConfig.MAX_SAMPLES_FOR_CV
            
            if n_samples > max_samples_for_cv:
                # Sample data for expensive computations
                sample_indices = self._rng.choice(n_samples, max_samples_for_cv, replace=False)
                sampled_data = modality_data[sample_indices]
                sampled_labels = self.train_labels[sample_indices]
                logger.debug(f"Using {max_samples_for_cv} samples for 6-factor calculation on {modality_name} (total: {n_samples})")
            else:
                sampled_data = modality_data
                sampled_labels = self.train_labels
            
            # Normalize data for fair comparison
            data_normalized = self._normalize_for_importance(sampled_data)
            
            # Factor 1: Data Variance (informativeness)
            variance_score = self._compute_data_variance(modality_data)
            
            # Factor 2: Feature Count (log-scaled)
            feature_count = modality_data.shape[1] if modality_data.ndim > 1 else 1
            feature_factor = np.log(feature_count + 1) / 10.0  # Scale down
            
            # Factor 3: Label Correlation (intelligence component)
            correlation_score = self._compute_label_correlation(data_normalized, sampled_labels)
            
            # Factor 4: Cross-Modality Redundancy (distinctiveness)
            redundancy_score = self._compute_cross_modality_redundancy(data_normalized, modality_name)
            
            # Factor 5: Data Quality (completeness, consistency, noise)
            quality_score = self._assess_data_quality(data_normalized)
            
            # Factor 6: Predictive Power (CV performance) - use sampled data for efficiency
            predictive_power = self._calculate_predictive_power(sampled_data, modality_name, sampled_labels)
            
            # Combine all 6 factors with weighted importance
            # Weights: variance(0.25), features(0.15), correlation(0.20), 
            #         redundancy(0.15), quality(0.10), predictive(0.15)
            combined_score = (
                0.25 * variance_score +
                0.15 * feature_factor +
                0.20 * correlation_score +
                0.15 * (1.0 - redundancy_score) +  # Invert redundancy (less redundant = more important)
                0.10 * quality_score +
                0.15 * predictive_power
            )
            
            # Ensure score is in reasonable range [0.01, 1.0]
            combined_score = max(0.01, min(1.0, combined_score))
            
            logger.debug(f"6-Factor scores for {modality_name}: "
                        f"variance={variance_score:.3f}, features={feature_factor:.3f}, "
                        f"correlation={correlation_score:.3f}, redundancy={redundancy_score:.3f}, "
                        f"quality={quality_score:.3f}, predictive={predictive_power:.3f}, "
                        f"combined={combined_score:.3f}")
            
            return float(combined_score)
            
        except Exception as e:
            logger.warning(f"Error in 6-factor importance calculation for {modality_name}: {e}")
            # Fallback to simple variance-based importance
            data_normalized = self._normalize_for_importance(modality_data)
            if data_normalized.ndim > 1:
                variance = float(np.mean(np.var(data_normalized, axis=0)))
            else:
                variance = float(np.var(data_normalized))
            feature_count = modality_data.shape[1] if modality_data.ndim > 1 else 1
            feature_factor = np.log(feature_count + 1)
            return max(0.01, variance * feature_factor / 10.0)

    # adaptive strategy functions
    def _normalize_for_importance(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to [0, 1] range for fair importance comparison across modalities.
        """
        if data.size == 0:
            return data
        
        # Runtime check: Validate array shape
        if data.ndim == 0:
            logger.warning("Received scalar data, converting to 1D array")
            data = np.array([data])
        
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
    
    def _compute_label_correlation(self, data_normalized: np.ndarray, labels: np.ndarray = None) -> float:
        """
        Compute correlation between modality data and labels (intelligence component).
        """
        try:
            if labels is None:
                labels = self.train_labels
                
            # Flatten multi-dimensional data for correlation calculation
            if data_normalized.ndim > 2:
                # Runtime check: Ensure first dimension is samples
                if data_normalized.shape[0] == 0:
                    logger.warning("Empty data array, returning zero correlation")
                    return 0.0
                data_flat = data_normalized.reshape(data_normalized.shape[0], -1)
            else:
                data_flat = data_normalized
            
            # Calculate correlation with labels for each feature
            correlations = []
            for i in range(data_flat.shape[1]):
                corr = np.corrcoef(data_flat[:, i], labels)[0, 1]
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
                        # Runtime check: Ensure first dimension is samples
                        if data_normalized.shape[0] == 0:
                            continue  # Skip empty data
                        data_flat = data_normalized.reshape(data_normalized.shape[0], -1)
                    else:
                        data_flat = data_normalized
                    
                    if other_normalized.ndim > 2:
                        # Runtime check: Ensure first dimension is samples
                        if other_normalized.shape[0] == 0:
                            continue  # Skip empty data
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
    
    def _calculate_predictive_power(self, modality_data: np.ndarray, modality_name: str, labels: np.ndarray = None) -> float:
        """
        Calculate predictive power of a modality using simplified assessment.
        
        Parameters
        ----------
        modality_data : np.ndarray
            Data for the modality
        modality_name : str
            Name of the modality
            
        Returns
        -------
        float
            Predictive power score (higher = more predictive)
        """
        try:
            # Use provided labels or fall back to train labels
            if labels is None:
                labels = self.train_labels
            
            # Ensure we have valid data
            if modality_data.size == 0 or len(labels) == 0:
                return 0.01
            
            # Ensure data and labels have same number of samples
            min_samples = min(len(modality_data), len(labels))
            if min_samples < 10:  # Need minimum samples for reliable assessment
                return 0.01
            
            modality_data = modality_data[:min_samples]
            labels = labels[:min_samples]
            
            # Simplified approach: Use CV score with variance fallback
            cv_score = self._quick_cv_score(modality_data, labels)
            
            if cv_score > 0.1:  # If CV score is meaningful
                return max(cv_score, 0.01)
            else:
                # Fallback to variance-based importance
                data_variance = self._compute_data_variance(modality_data)
                return max(data_variance, 0.01)
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Data error in predictive power calculation for {modality_name}: {e}")
            return 0.01
        except Exception as e:
            logger.warning(f"Unexpected error in predictive power calculation for {modality_name}: {e}")
            return 0.01
    
    def _quick_cv_score(self, modality_data: np.ndarray, labels: np.ndarray) -> float:
        """Quick cross-validation score using a simple classifier."""
        try:
            
            # Handle different data shapes
            if modality_data.ndim > 2:
                # Flatten high-dimensional data
                modality_data = modality_data.reshape(len(modality_data), -1)
            
            # Standardize features
            scaler = StandardScaler()
            modality_data_scaled = scaler.fit_transform(modality_data)
            
            # Quick Random Forest for assessment
            rf = RandomForestClassifier(
                n_estimators=10,  # Small for speed
                max_depth=5,
                random_state=42,
                n_jobs=1
            )
            
            # 3-fold CV for speed
            cv_scores = cross_val_score(rf, modality_data_scaled, labels, cv=3, scoring='accuracy')
            
            return float(np.mean(cv_scores))
            
        except Exception as e:
            logger.debug(f"Error in quick CV score: {e}")
            return 0.5  # Neutral score
    
    def _compute_feature_importance(self, modality_data: np.ndarray, labels: np.ndarray) -> float:
        """Compute feature importance using a quick model."""
        try:
            
            # Handle different data shapes
            if modality_data.ndim > 2:
                modality_data = modality_data.reshape(len(modality_data), -1)
            
            # Standardize features
            scaler = StandardScaler()
            modality_data_scaled = scaler.fit_transform(modality_data)
            
            # Quick Random Forest
            rf = RandomForestClassifier(
                n_estimators=10,
                max_depth=5,
                random_state=42,
                n_jobs=1
            )
            
            rf.fit(modality_data_scaled, labels)
            
            # Average feature importance
            avg_importance = float(np.mean(rf.feature_importances_))
            
            return avg_importance
            
        except Exception as e:
            logger.debug(f"Error in feature importance: {e}")
            return 0.1  # Low importance fallback
    
    def _compute_importance_stability(self, modality_data: np.ndarray, labels: np.ndarray) -> float:
        """Compute stability of importance across different samples."""
        try:
            
            # Handle different data shapes
            if modality_data.ndim > 2:
                modality_data = modality_data.reshape(len(modality_data), -1)
            
            # Standardize features
            scaler = StandardScaler()
            modality_data_scaled = scaler.fit_transform(modality_data)
            
            # Bootstrap sampling for stability assessment
            n_samples = len(modality_data_scaled)
            n_bootstrap = 5  # Small number for speed
            
            importance_scores = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_boot = modality_data_scaled[indices]
                y_boot = labels[indices]
                
                # Quick Random Forest
                rf = RandomForestClassifier(
                    n_estimators=5,  # Very small for speed
                    max_depth=3,
                    random_state=42,
                    n_jobs=1
                )
                
                rf.fit(X_boot, y_boot)
                importance_scores.append(np.mean(rf.feature_importances_))
            
            # Stability = 1 - coefficient of variation
            if len(importance_scores) > 1:
                mean_importance = np.mean(importance_scores)
                std_importance = np.std(importance_scores)
                stability = 1.0 - (std_importance / (mean_importance + 1e-8))
                stability = max(0.0, min(1.0, stability))  # Clamp to [0, 1]
            else:
                stability = 1.0
            
            return float(stability)
            
        except Exception as e:
            logger.debug(f"Error in stability computation: {e}")
            return 0.8  # High stability fallback
    
    def _compute_data_variance(self, modality_data: np.ndarray) -> float:
        """Compute normalized data variance as a measure of informativeness."""
        try:
            # Handle different data shapes
            if modality_data.ndim > 2:
                modality_data = modality_data.reshape(len(modality_data), -1)
            
            # Compute variance across all features
            feature_variances = np.var(modality_data, axis=0)
            mean_variance = np.mean(feature_variances)
            
            # Normalize variance to [0, 1] range
            # Use log scaling to handle large variance differences
            normalized_variance = np.log(1 + mean_variance) / 10.0  # Scale down
            normalized_variance = min(normalized_variance, 1.0)  # Cap at 1.0
            
            return normalized_variance
            
        except Exception as e:
            logger.debug(f"Error in data variance computation: {e}")
            return 0.1  # Small default variance

    # ============================================================================
    # STEP 4: SAMPLING (BOOTSTRAP)
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
            
            # Runtime check: Ensure n_drop is within valid bounds
            n_drop = max(0, min(n_drop, self.n_modalities - 1))
            
            # Create mask
            mask = np.ones(self.n_modalities, dtype=bool)
            if n_drop > 0 and n_drop < self.n_modalities:
                drop_indices = self._rng.choice(
                    np.arange(self.n_modalities), size=n_drop, replace=False
                )
                mask[drop_indices] = False
            
            # Convert to dictionary
            modality_mask = {name: bool(mask[i]) for i, name in enumerate(self.modality_names)}
            modality_masks.append(modality_mask)
        
        logger.info(f"Created modality masks for {len(modality_masks)} bags")
        return modality_masks

    def _bootstrap_sampling(self) -> List[np.ndarray]:
        """
        Step 4: Robust Bootstrap Sampling with Guaranteed Consistency.
        Creates bootstrap sample indices for each bag with guaranteed data consistency.
        """
        logger.info("Step 4: Performing robust bootstrap sampling with guaranteed consistency")
        
        # ENHANCED FIX: Get consistent sample count across all modalities and labels
        modality_sample_counts = {name: len(data) for name, data in self.train_data.items()}
        label_count = len(self.train_labels)
        
        # Defensive programming: Check for empty data
        if not modality_sample_counts:
            raise ValueError("No modality data available for bootstrap sampling")
        if label_count == 0:
            raise ValueError("No training labels available for bootstrap sampling")
        
        # Find the minimum sample count across all data sources
        all_counts = list(modality_sample_counts.values()) + [label_count]
        dataset_size = min(all_counts)
        
        # Defensive programming: Ensure we have valid data
        if dataset_size <= 0:
            raise ValueError(f"Invalid dataset size: {dataset_size}. All modalities and labels must have positive sample counts.")
        
        if len(set(all_counts)) > 1:
            logger.warning(f"Inconsistent sample counts: {modality_sample_counts}, labels: {label_count}")
            logger.info(f"Using minimum sample count: {dataset_size} for guaranteed consistency")
        
        # Calculate target sample size
        n_samples = int(self.sample_ratio * dataset_size)
        
        # Ensure we don't request more samples than available
        n_samples = min(n_samples, dataset_size)
        
        data_indices = []
        for bag_id in range(self.n_bags):
            # Create bootstrap sample with replacement
            # Use only valid indices (0 to dataset_size-1)
            bag_indices = self._rng.choice(
                np.arange(dataset_size), size=n_samples, replace=True
            )
            data_indices.append(bag_indices)
        
        logger.info(f"Created bootstrap samples for {len(data_indices)} bags ({n_samples} samples each)")
        logger.info(f"All indices guaranteed to be in range [0, {dataset_size-1}]")
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
            
            # Calculate diversity score for this bag
            diversity_score = self._calculate_bag_diversity(modality_masks[bag_id], bag_id)
            
            # Create bag configuration
            bag = BagConfig(
                bag_id=bag_id,
                data_indices=data_indices[bag_id],
                modality_mask=modality_masks[bag_id],
                modality_weights=modality_weights,
                dropout_rate=dropout_rates[bag_id],
                sample_ratio=self.sample_ratio,
                diversity_score=diversity_score,
                creation_timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                metadata={}
            )
            
            self.bags.append(bag)
            
            # ENHANCED FIX: Simple and robust bag data extraction
            bag_data = {}
            indices = data_indices[bag_id]  # These indices are already guaranteed to be valid
            
            # Extract data for each active modality using the same indices
            for modality_name, is_active in modality_masks[bag_id].items():
                if is_active and modality_name in self.train_data:
                    data = self.train_data[modality_name]
                    # Since indices are guaranteed to be valid, we can directly use them
                    bag_data[modality_name] = data[indices]
            
            # Store labels using the same indices (guaranteed to be valid)
            bag_data['labels'] = self.train_labels[indices]
            
            # Store bag data
            self.bag_data[bag_id] = bag_data
        
        logger.info(f"Extracted and stored data for {len(self.bags)} bags")

    def _calculate_bag_diversity(self, modality_mask: Dict[str, bool], bag_id: int) -> float:
        """
        Calculate diversity score for a bag based on modality combination uniqueness.
        
        Parameters
        ----------
        modality_mask : Dict[str, bool]
            Boolean mask indicating active modalities
        bag_id : int
            ID of the bag
            
        Returns
        -------
        float
            Diversity score between 0.0 and 1.0
        """
        try:
            # Get active modalities
            active_modalities = [name for name, active in modality_mask.items() if active]
            n_active = len(active_modalities)
            
            if n_active == 0:
                return 0.0
            
            # Base diversity from number of active modalities
            modality_diversity = n_active / self.n_modalities
            
            # Calculate combination uniqueness (how rare this combination is)
            combination = tuple(sorted(active_modalities))
            
            # Count how many other bags have the same combination
            same_combination_count = 0
            for other_bag in self.bags:
                if other_bag.bag_id != bag_id:
                    other_active = [name for name, active in other_bag.modality_mask.items() if active]
                    other_combination = tuple(sorted(other_active))
                    if other_combination == combination:
                        same_combination_count += 1
            
            # Uniqueness score (higher is more unique)
            uniqueness_score = 1.0 / (1.0 + same_combination_count)
            
            # Combine modality diversity and uniqueness
            diversity_score = (modality_diversity * 0.6 + uniqueness_score * 0.4)
            
            return float(np.clip(diversity_score, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Error calculating diversity score for bag {bag_id}: {e}")
            return 0.5  # Default moderate diversity

    # ============================================================================
    # STEP 6: CONVENIENCE FUNCTIONS
    # ============================================================================
    
    

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
    # ROBUSTNESS and INTERPRETABILITY TESTS
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

    def interpretability_test(self, save_to_file: bool = False, output_dir: str = None) -> Dict[str, Any]:
        """
        Comprehensive interpretability test - collects data, runs analysis, and optionally saves to file.
        
        Parameters
        ----------
        save_to_file : bool, default=False
            Whether to save results to a JSON file
        output_dir : str, optional
            Directory to save the file. If None, saves to current directory.
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive interpretability results including data and analysis
        """
        logger.info("Running comprehensive interpretability test")
        
        # Collect comprehensive interpretability data
        interpretability_data = {
            'modality_importance': self.modality_importance.copy(),
            'bag_configurations': [],
            'ensemble_statistics': {},
            'modality_usage_patterns': {},
            'dropout_patterns': {},
            'diversity_scores': [],
            'coverage_analysis': {},
            'analysis': {}
        }
        
        # Collect bag configuration data
        for bag in self.bags:
            bag_info = {
                'bag_id': bag.bag_id,
                'modality_mask': bag.modality_mask.copy(),
                'dropout_rate': bag.dropout_rate,
                'diversity_score': bag.diversity_score,
                'sample_count': len(bag.data_indices),
                'creation_timestamp': bag.creation_timestamp
            }
            interpretability_data['bag_configurations'].append(bag_info)
        
        # Collect ensemble statistics
        dropout_rates = [bag.dropout_rate for bag in self.bags]
        diversity_scores = [bag.diversity_score for bag in self.bags]
        
        interpretability_data['ensemble_statistics'] = {
            'total_bags': len(self.bags),
            'total_modalities': self.n_modalities,
            'strategy_used': self.dropout_strategy,
            'average_dropout_rate': np.mean(dropout_rates),
            'std_dropout_rate': np.std(dropout_rates),
            'min_dropout_rate': np.min(dropout_rates),
            'max_dropout_rate': np.max(dropout_rates),
            'average_diversity_score': np.mean(diversity_scores),
            'std_diversity_score': np.std(diversity_scores)
        }
        
        # Collect modality usage patterns
        for modality_name in self.modality_names:
            usage_count = sum(1 for bag in self.bags if bag.modality_mask.get(modality_name, False))
            interpretability_data['modality_usage_patterns'][modality_name] = {
                'usage_count': usage_count,
                'usage_percentage': (usage_count / len(self.bags)) * 100,
                'importance_score': self.modality_importance.get(modality_name, 0.0)
            }
        
        # Collect dropout patterns
        interpretability_data['dropout_patterns'] = {
            'unique_combinations': len(set(tuple(sorted(bag.modality_mask.items())) for bag in self.bags)),
            'dropout_rate_distribution': {
                'mean': np.mean(dropout_rates),
                'std': np.std(dropout_rates),
                'min': np.min(dropout_rates),
                'max': np.max(dropout_rates),
                'median': np.median(dropout_rates)
            }
        }
        
        # Collect diversity scores
        interpretability_data['diversity_scores'] = diversity_scores
        
        # Coverage analysis
        total_modality_combinations = 2 ** self.n_modalities - 1  # All possible non-empty combinations
        unique_combinations = len(set(tuple(sorted(bag.modality_mask.items())) for bag in self.bags))
        interpretability_data['coverage_analysis'] = {
            'total_possible_combinations': total_modality_combinations,
            'unique_combinations_used': unique_combinations,
            'coverage_percentage': (unique_combinations / total_modality_combinations) * 100,
            'combination_efficiency': unique_combinations / len(self.bags)
        }
        
        # Run comprehensive analysis
        analysis_results = {
            'modality_importance_analysis': {},
            'dropout_pattern_analysis': {},
            'diversity_analysis': {},
            'coverage_analysis': {},
            'strategy_effectiveness': {}
        }
        
        # Modality importance analysis
        importance_scores = list(self.modality_importance.values())
        analysis_results['modality_importance_analysis'] = {
            'most_important_modality': max(self.modality_importance.items(), key=lambda x: x[1]),
            'least_important_modality': min(self.modality_importance.items(), key=lambda x: x[1]),
            'importance_variance': np.var(importance_scores),
            'importance_range': max(importance_scores) - min(importance_scores)
        }
        
        # Dropout pattern analysis
        analysis_results['dropout_pattern_analysis'] = {
            'dropout_rate_variance': np.var(dropout_rates),
            'dropout_rate_range': np.max(dropout_rates) - np.min(dropout_rates),
            'dropout_consistency': 1 - np.std(dropout_rates) / np.mean(dropout_rates) if np.mean(dropout_rates) > 0 else 0
        }
        
        # Diversity analysis
        analysis_results['diversity_analysis'] = {
            'diversity_variance': np.var(diversity_scores),
            'diversity_range': np.max(diversity_scores) - np.min(diversity_scores),
            'diversity_consistency': 1 - np.std(diversity_scores) / np.mean(diversity_scores) if np.mean(diversity_scores) > 0 else 0
        }
        
        # Coverage analysis
        coverage_percentage = interpretability_data['coverage_analysis']['coverage_percentage']
        analysis_results['coverage_analysis'] = {
            'coverage_efficiency': coverage_percentage,
            'coverage_adequacy': 'Good' if coverage_percentage > 50 else 'Poor' if coverage_percentage < 20 else 'Moderate'
        }
        
        # Strategy effectiveness analysis
        if self.dropout_strategy == 'adaptive':
            importance_variance = analysis_results['modality_importance_analysis']['importance_variance']
            analysis_results['strategy_effectiveness'] = {
                'adaptive_effectiveness': 'High' if importance_variance > 0.1 else 'Low',
                'importance_differentiation': importance_variance
            }
        else:
            analysis_results['strategy_effectiveness'] = {
                'strategy_type': self.dropout_strategy,
                'uniform_application': True
            }
        
        interpretability_data['analysis'] = analysis_results
        
        # Save to file if requested
        if save_to_file:
            import json
            import os
            from datetime import datetime
            
            if output_dir is None:
                output_dir = "."
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"interpretability_results_{self.dropout_strategy}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            serializable_data = convert_numpy_types(interpretability_data)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            logger.info(f"Interpretability results saved to: {filepath}")
            interpretability_data['saved_to_file'] = filepath
        
        logger.info("Interpretability test completed successfully")
        return interpretability_data


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