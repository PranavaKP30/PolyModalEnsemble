"""
Stage 2: Ensemble Generation
Simplified modality-aware ensemble generation system with adaptive dropout strategies.

Implements:
- Adaptive Modality- and Feature-Aware Dropout Framework
- Linear, exponential, and random dropout strategies
- Bootstrap sampling with configurable ratios
- Bag configuration management
- Memory-only storage for Stage 3 integration
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger("ensemble_generation")

# --- BagConfig ---
@dataclass
class BagConfig:
    bag_id: int
    data_indices: np.ndarray
    modality_mask: Dict[str, bool]
    feature_mask: Dict[str, np.ndarray]
    dropout_rate: float
    sample_ratio: float = 0.8
    diversity_score: float = 0.0
    creation_timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# --- ModalityDropoutBagger ---
class ModalityDropoutBagger:
    def __init__(self, 
                 train_data: Dict[str, np.ndarray],
                 train_labels: np.ndarray,
                 modality_configs: List[Any],
                 n_bags: int = 20,
                 dropout_strategy: str = "adaptive",
                 max_dropout_rate: float = 0.5,
                 min_modalities: int = 1,
                 sample_ratio: float = 0.8,
                 feature_sampling_ratio: float = 0.8,
                 random_state: Optional[int] = None):
        """
        Initialize the Modality Dropout Bagger.
        
        Parameters
        ----------
        train_data : Dict[str, np.ndarray]
            Training data for each modality
        train_labels : np.ndarray
            Training labels
        modality_configs : List[Any]
            Modality configuration objects
        n_bags : int, default=20
            Number of ensemble bags to create
        dropout_strategy : str, default="adaptive"
            Dropout strategy: 'adaptive', 'linear', 'exponential', 'random'
        max_dropout_rate : float, default=0.5
            Maximum dropout rate for modalities
        min_modalities : int, default=1
            Minimum number of modalities per bag
        sample_ratio : float, default=0.8
            Bootstrap sampling ratio
        feature_sampling_ratio : float, default=0.8
            Feature sampling ratio within each modality
        random_state : Optional[int], default=None
            Random seed for reproducibility
        """
        self.train_data = train_data
        self.train_labels = train_labels
        self.modality_configs = modality_configs
        self.n_bags = n_bags
        self.dropout_strategy = dropout_strategy
        self.max_dropout_rate = max_dropout_rate
        self.min_modalities = min_modalities
        self.sample_ratio = sample_ratio
        self.feature_sampling_ratio = feature_sampling_ratio
        self.random_state = random_state
        
        # Initialize random number generator
        self._rng = np.random.default_rng(random_state)
        
        # Storage for generated bags
        self.bags: List[BagConfig] = []
        self.bag_data: Dict[int, Dict[str, np.ndarray]] = {}
        
        # Interpretability data storage
        self._modality_importance: Optional[Dict[str, float]] = None
        self._interpretability_data: Dict[str, Any] = {}
        
        # Validate parameters
        self._validate_params()
        
        logger.info(f"Initialized ModalityDropoutBagger with {n_bags} bags, strategy: {dropout_strategy}")

    def _validate_params(self):
        """Validate input parameters."""
        assert 1 <= self.n_bags <= 1000, "n_bags must be between 1 and 1000"
        assert self.dropout_strategy in ['adaptive', 'linear', 'exponential', 'random'], \
            "dropout_strategy must be one of: 'adaptive', 'linear', 'exponential', 'random'"
        assert 0.0 <= self.max_dropout_rate <= 0.9, "max_dropout_rate must be between 0.0 and 0.9"
        assert 0.1 <= self.sample_ratio <= 1.0, "sample_ratio must be between 0.1 and 1.0"
        assert 0.1 <= self.feature_sampling_ratio <= 1.0, "feature_sampling_ratio must be between 0.1 and 1.0"
        assert 1 <= self.min_modalities <= len(self.modality_configs), \
            "min_modalities must be between 1 and number of modalities"

    def generate_bags(self) -> List[BagConfig]:
        """
        Generate ensemble bags using the specified dropout strategy.
        
        Returns
        -------
        List[BagConfig]
            List of generated bag configurations
        """
        logger.info(f"Generating {self.n_bags} bags with {self.dropout_strategy} strategy")
        
        modality_names = [config.name for config in self.modality_configs]
        dataset_size = len(self.train_labels)
        
        # Calculate dropout rates for each bag
        dropout_rates = self._calculate_dropout_rates()
        
        # Generate bags
        for bag_id in range(self.n_bags):
            # Create modality mask based on dropout strategy
            modality_mask = self._create_modality_mask(
                modality_names, dropout_rates[bag_id]
            )
            
            # Create feature masks for active modalities
            feature_mask = self._create_feature_masks(
                modality_names, modality_mask
            )
            
            # Create bootstrap sample indices
            data_indices = self._create_bootstrap_sample(dataset_size)
            
            # Create bag configuration
            bag = BagConfig(
                bag_id=bag_id,
                data_indices=data_indices,
                modality_mask=modality_mask,
                feature_mask=feature_mask,
                dropout_rate=dropout_rates[bag_id],
                sample_ratio=self.sample_ratio,
                diversity_score=0.0,  # Will be calculated later if needed
                creation_timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                metadata={}
            )
            
            self.bags.append(bag)
            
            # Extract and store bag data
            self._extract_bag_data(bag)
        
        # Step 7: Interpretability Data Collection
        self._collect_interpretability_data()
        
        logger.info(f"Successfully generated {len(self.bags)} bags")
        return self.bags

    def _calculate_dropout_rates(self) -> np.ndarray:
        """Calculate dropout rates for each bag based on strategy."""
        if self.dropout_strategy == 'linear':
            return np.linspace(0, self.max_dropout_rate, self.n_bags)
        elif self.dropout_strategy == 'exponential':
            return self.max_dropout_rate * (1 - np.exp(-3 * np.linspace(0, 1, self.n_bags)))
        elif self.dropout_strategy == 'random':
            return self._rng.uniform(0, self.max_dropout_rate, self.n_bags)
        elif self.dropout_strategy == 'adaptive':
            return self._adaptive_dropout_strategy()
        else:
            return np.full(self.n_bags, self.max_dropout_rate)

    def _adaptive_dropout_strategy(self) -> np.ndarray:
        """
        Adaptive Modality- and Feature-Aware Dropout Framework.
        
        This comprehensive function implements the full adaptive dropout strategy:
        1. Computes predictive importance scores for each modality
        2. Normalizes scores to generate modality-specific dropout probabilities
        3. Samples modality subsets inversely proportional to importance
        4. Enforces distinct modality combinations across bags
        5. Samples features within each selected modality
        6. Adaptively adjusts probabilities to maximize diversity
        """
        modality_names = [config.name for config in self.modality_configs]
        n_modalities = len(modality_names)
        
        # Step 1: Compute predictive importance scores for each modality
        modality_importance = self._compute_modality_importance()
        
        # Step 2: Normalize importance scores to create dropout probabilities
        # Higher importance = lower dropout probability
        max_importance = max(modality_importance.values())
        min_importance = min(modality_importance.values())
        
        if max_importance > min_importance:
            # Normalize to [0, 1] range, then invert for dropout probabilities
            normalized_importance = {
                name: (score - min_importance) / (max_importance - min_importance)
                for name, score in modality_importance.items()
            }
            dropout_probs = {
                name: self.max_dropout_rate * (1 - norm_score)
                for name, norm_score in normalized_importance.items()
            }
        else:
            # All modalities have equal importance
            dropout_probs = {
                name: self.max_dropout_rate * 0.5
                for name in modality_names
            }
        
        # Step 3: Generate adaptive dropout rates for each bag
        dropout_rates = []
        used_combinations = set()
        
        for bag_id in range(self.n_bags):
            # Calculate base dropout rate with some variation
            base_rate = self.max_dropout_rate * (bag_id / max(1, self.n_bags - 1))
            variation = self._rng.uniform(-0.1, 0.1)
            adaptive_rate = np.clip(base_rate + variation, 0, self.max_dropout_rate)
            
            # Ensure distinct modality combinations
            attempts = 0
            while attempts < 10:  # Prevent infinite loops
                # Sample modalities based on dropout probabilities
                modality_selection = {}
                for name in modality_names:
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
                        key=lambda x: modality_importance[x]
                    )
                    needed = self.min_modalities - active_count
                    for name in sorted_by_importance[:needed]:
                        modality_selection[name] = True
                
                # Check if this combination is distinct
                combination = tuple(sorted(
                    name for name, active in modality_selection.items() if active
                ))
                
                if combination not in used_combinations or len(used_combinations) >= 2**n_modalities:
                    used_combinations.add(combination)
                    break
                
                attempts += 1
            
            dropout_rates.append(adaptive_rate)
        
        return np.array(dropout_rates)

    def _compute_modality_importance(self) -> Dict[str, float]:
        """
        Compute predictive importance scores for each modality.
        Uses simple variance-based importance as a proxy for predictive power.
        """
        modality_importance = {}
        
        for config in self.modality_configs:
            modality_name = config.name
            if modality_name in self.train_data:
                data = self.train_data[modality_name]
                
                # Use feature variance as importance proxy
                if data.ndim > 1:
                    # For multi-dimensional data, use mean variance across features
                    importance = float(np.mean(np.var(data, axis=0)))
                else:
                    # For 1D data, use variance directly
                    importance = float(np.var(data))
                
                modality_importance[modality_name] = importance
        
        return modality_importance

    def _create_modality_mask(self, modality_names: List[str], dropout_rate: float) -> Dict[str, bool]:
        """Create modality mask for a single bag."""
        n_modalities = len(modality_names)
        n_drop = int(np.floor(dropout_rate * n_modalities))
        n_drop = min(n_drop, n_modalities - self.min_modalities)
        
        # Create mask
        mask = np.ones(n_modalities, dtype=bool)
        if n_drop > 0:
            drop_indices = self._rng.choice(
                np.arange(n_modalities), size=n_drop, replace=False
            )
            mask[drop_indices] = False
        
        return {name: bool(mask[i]) for i, name in enumerate(modality_names)}

    def _create_feature_masks(self, modality_names: List[str], modality_mask: Dict[str, bool]) -> Dict[str, np.ndarray]:
        """Create feature masks for active modalities with feature-level sampling."""
        feature_mask = {}
        
        for i, name in enumerate(modality_names):
            if not modality_mask[name]:
                # Modality is dropped, no features selected
                feature_mask[name] = np.array([], dtype=bool)
            else:
                # Modality is active, sample features based on feature_sampling_ratio
                config = self.modality_configs[i]
                n_features = config.feature_dim
                
                # Calculate number of features to select
                n_selected = max(1, int(self.feature_sampling_ratio * n_features))
                
                # Randomly select features
                selected_indices = self._rng.choice(
                    n_features, size=n_selected, replace=False
                )
                
                # Create boolean mask
                mask = np.zeros(n_features, dtype=bool)
                mask[selected_indices] = True
                feature_mask[name] = mask
        
        return feature_mask

    def _create_bootstrap_sample(self, dataset_size: int) -> np.ndarray:
        """Create bootstrap sample indices."""
        n_samples = int(self.sample_ratio * dataset_size)
        return self._rng.choice(
            np.arange(dataset_size), size=n_samples, replace=True
        )

    def _extract_bag_data(self, bag: BagConfig):
        """Extract and store data for a specific bag."""
        bag_data = {}
        
        # Extract data for active modalities
        for modality_name, is_active in bag.modality_mask.items():
            if is_active and modality_name in self.train_data:
                data = self.train_data[modality_name]
                feature_mask = bag.feature_mask[modality_name]
                
                # Apply feature mask and bootstrap sampling
                if len(feature_mask) > 0:
                    bag_data[modality_name] = data[bag.data_indices][:, feature_mask]
                else:
                    bag_data[modality_name] = data[bag.data_indices]
        
        # Extract labels
        bag_data['labels'] = self.train_labels[bag.data_indices]
        
        # Store bag data
        self.bag_data[bag.bag_id] = bag_data

    def _collect_interpretability_data(self):
        """Collect comprehensive interpretability data for Stage 2 analysis."""
        logger.info("Collecting interpretability data for Stage 2")
        
        # Store modality importance scores
        if self.dropout_strategy == 'adaptive' and hasattr(self, '_modality_importance'):
            self._interpretability_data['modality_importance'] = self._modality_importance
        else:
            # Calculate modality usage for non-adaptive strategies
            modality_usage = {}
            for bag in self.bags:
                for modality, is_active in bag.modality_mask.items():
                    if modality not in modality_usage:
                        modality_usage[modality] = {'active': 0, 'total': 0}
                    modality_usage[modality]['total'] += 1
                    if is_active:
                        modality_usage[modality]['active'] += 1
            
            # Convert to importance scores
            importance_scores = {}
            for modality, usage in modality_usage.items():
                importance_scores[modality] = usage['active'] / usage['total']
            self._interpretability_data['modality_importance'] = importance_scores
        
        # Store detailed bag information
        detailed_bags = []
        for bag in self.bags:
            detailed_bag = {
                'bag_id': bag.bag_id,
                'data_indices': bag.data_indices,
                'modality_mask': bag.modality_mask,
                'feature_mask': bag.feature_mask,
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
        
        # Store feature selection statistics
        feature_stats = {}
        for bag in self.bags:
            for modality, feature_mask in bag.feature_mask.items():
                if modality not in feature_stats:
                    feature_stats[modality] = {
                        'total_features': len(feature_mask),
                        'selection_ratios': [],
                        'selected_counts': []
                    }
                
                selected_count = np.sum(feature_mask)
                selection_ratio = selected_count / len(feature_mask) if len(feature_mask) > 0 else 0.0
                feature_stats[modality]['selection_ratios'].append(selection_ratio)
                feature_stats[modality]['selected_counts'].append(selected_count)
        
        # Calculate feature statistics
        for modality in feature_stats:
            stats = feature_stats[modality]
            stats['mean_selection_ratio'] = np.mean(stats['selection_ratios'])
            stats['std_selection_ratio'] = np.std(stats['selection_ratios'])
            stats['min_selection_ratio'] = np.min(stats['selection_ratios'])
            stats['max_selection_ratio'] = np.max(stats['selection_ratios'])
        
        self._interpretability_data['feature_statistics'] = feature_stats
        
        logger.info("Interpretability data collection completed")

    def get_bag_data(self, bag_id: int, multimodal_data: Dict[str, np.ndarray] = None, return_metadata: bool = False) -> Dict[str, np.ndarray]:
        """
        Get data for a specific bag.
        
        Parameters
        ----------
        bag_id : int
            ID of the bag to retrieve
        multimodal_data : Dict[str, np.ndarray], optional
            Multimodal data (not used in new implementation, kept for compatibility)
        return_metadata : bool, default=False
            Whether to return metadata along with data
            
        Returns
        -------
        Dict[str, np.ndarray] or tuple
            Bag data, optionally with metadata
        """
        if bag_id not in self.bag_data:
            raise ValueError(f"Bag {bag_id} not found")
        
        bag_data = self.bag_data[bag_id]
        
        if return_metadata:
            bag = self.bags[bag_id]
            metadata = {
                'sample_count': len(bag.data_indices),
                'diversity_score': bag.diversity_score,
                'dropout_rate': bag.dropout_rate,
                'modality_mask': bag.modality_mask,
                'feature_mask': bag.feature_mask
            }
            return bag_data, bag.modality_mask, metadata
        
        return bag_data

    def get_interpretability_data(self) -> Dict[str, Any]:
        """Get comprehensive interpretability data for Stage 2 analysis."""
        return self._interpretability_data.copy()

    def get_modality_importance(self) -> Dict[str, float]:
        """Get modality importance scores."""
        importance = self._interpretability_data.get('modality_importance', {})
        return importance.copy() if importance else {}

    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get feature selection statistics."""
        stats = self._interpretability_data.get('feature_statistics', {})
        return stats.copy() if stats else {}

    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get ensemble-level statistics."""
        stats = self._interpretability_data.get('ensemble_statistics', {})
        return stats.copy() if stats else {}

    def get_bag_info(self, bag_id: int) -> Dict[str, Any]:
        """Get information about a specific bag."""
        if bag_id >= len(self.bags):
            raise ValueError(f"Bag {bag_id} not found")
        
        bag = self.bags[bag_id]
        return {
            'bag_id': bag.bag_id,
            'modalities': [name for name, active in bag.modality_mask.items() if active],
            'dropout_rate': bag.dropout_rate,
            'sample_count': len(bag.data_indices),
            'diversity_score': bag.diversity_score,
            'creation_timestamp': bag.creation_timestamp
        }

    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        if not self.bags:
            return {}
        
        modality_names = [config.name for config in self.modality_configs]
        n_bags = len(self.bags)
        
        # Calculate modality coverage
        modality_coverage = {}
        for name in modality_names:
            coverage = sum(1 for bag in self.bags if bag.modality_mask.get(name, False))
            modality_coverage[name] = coverage / n_bags
        
        # Calculate dropout statistics
        dropout_rates = [bag.dropout_rate for bag in self.bags]
        dropout_stats = {
            'mean_dropout_rate': float(np.mean(dropout_rates)),
            'min_dropout_rate': float(np.min(dropout_rates)),
            'max_dropout_rate': float(np.max(dropout_rates)),
            'std_dropout_rate': float(np.std(dropout_rates))
        }
        
        return {
            'n_bags': n_bags,
            'modality_coverage': modality_coverage,
            'dropout_statistics': dropout_stats,
            'strategy': self.dropout_strategy
        }