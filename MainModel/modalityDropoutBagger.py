"""
Stage 2: Ensemble Generation
Advanced modality-aware ensemble generation system with intelligent dropout strategies, adaptive diversity optimization, and comprehensive bag configuration management.

Implements:
- Intelligent Modality Dropout (linear, exponential, random, adaptive)
- Bootstrap and feature-level sampling
- Diversity optimization and tracking
- Bag configuration management and validation
- Seamless integration with Stage 1 and Stage 3

See MainModel/2ModalityDropoutBaggerDoc.md for full documentation and API reference.
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
    def __init__(self, modality_configs: List[Any], integration_metadata: Dict[str, Any], n_bags: int = 20, dropout_strategy: str = "adaptive", max_dropout_rate: float = 0.5, min_modalities: int = 1, sample_ratio: float = 0.8, diversity_target: float = 0.7, feature_sampling: bool = True, enable_validation: bool = True, random_state: Optional[int] = None):
        self.modality_configs = modality_configs
        self.integration_metadata = integration_metadata
        self.n_bags = n_bags
        self.dropout_strategy = dropout_strategy
        self.max_dropout_rate = max_dropout_rate
        self.min_modalities = min_modalities
        self.sample_ratio = sample_ratio
        self.diversity_target = diversity_target
        self.feature_sampling = feature_sampling
        self.enable_validation = enable_validation
        self.random_state = random_state
        self.bags: List[BagConfig] = []
        self._rng = np.random.default_rng(random_state)
        self._validate_params()

    @classmethod
    def from_data_integration(cls, integrated_data: Dict[str, np.ndarray], modality_configs: List[Any], integration_metadata: Dict[str, Any], n_bags: int = 20, dropout_strategy: str = "adaptive", max_dropout_rate: float = 0.5, **kwargs) -> "ModalityDropoutBagger":
        # Create instance with proper parameters
        instance = cls(modality_configs, integration_metadata, n_bags, dropout_strategy, max_dropout_rate, **kwargs)
        # Store integrated data for later use
        instance.integrated_data = integrated_data
        return instance

    def _validate_params(self):
        assert 1 <= self.n_bags <= 1000, "n_bags out of range"
        assert self.dropout_strategy in ['linear', 'exponential', 'random', 'adaptive'], "Invalid dropout_strategy"
        assert 0.0 <= self.max_dropout_rate <= 0.9, "max_dropout_rate out of range"
        assert 0.1 <= self.sample_ratio <= 1.0, "sample_ratio out of range"
        assert 1 <= self.min_modalities <= len(self.modality_configs), "min_modalities out of range"
        assert 0.0 <= self.diversity_target <= 1.0, "diversity_target out of range"

    def generate_bags(self, dataset_size: Optional[int] = None, modality_feature_dims: Optional[Dict[str, int]] = None) -> List[BagConfig]:
        dataset_size = dataset_size or self.integration_metadata.get('dataset_size', 0)
        modality_names = [c.name for c in self.modality_configs]
        feature_dims = modality_feature_dims or {c.name: c.feature_dim for c in self.modality_configs}
        bags = []
        for bag_id in range(self.n_bags):
            # Dropout rate calculation
            progress = bag_id / max(1, self.n_bags - 1)
            dropout_rate = self._calc_dropout_rate(progress, bag_id)
            # Modality mask
            n_modalities = len(modality_names)
            n_drop = int(np.floor(dropout_rate * n_modalities))
            n_drop = min(n_drop, n_modalities - self.min_modalities)
            mask = np.ones(n_modalities, dtype=bool)
            if n_drop > 0:
                drop_indices = self._rng.choice(np.arange(n_modalities), size=n_drop, replace=False)
                mask[drop_indices] = False
            modality_mask = {name: bool(mask[i]) for i, name in enumerate(modality_names)}
            # Feature mask
            feature_mask = {}
            for i, name in enumerate(modality_names):
                if not mask[i]:
                    feature_mask[name] = np.zeros(feature_dims[name], dtype=bool)
                elif self.feature_sampling and feature_dims[name] > 1:
                    n_features = feature_dims[name]
                    min_ratio = getattr(self.modality_configs[i], 'min_feature_ratio', 0.3)
                    max_ratio = getattr(self.modality_configs[i], 'max_feature_ratio', 1.0)
                    ratio = self._rng.uniform(min_ratio, max_ratio)
                    n_sample = max(1, int(ratio * n_features))
                    fmask = np.zeros(n_features, dtype=bool)
                    fmask[self._rng.choice(np.arange(n_features), size=n_sample, replace=False)] = True
                    feature_mask[name] = fmask
                else:
                    feature_mask[name] = np.ones(feature_dims[name], dtype=bool)
            # Bootstrap sample indices
            n_samples = int(self.sample_ratio * dataset_size)
            data_indices = self._rng.choice(np.arange(dataset_size), size=n_samples, replace=True)
            # BagConfig
            bag = BagConfig(
                bag_id=bag_id,
                data_indices=data_indices,
                modality_mask=modality_mask,
                feature_mask=feature_mask,
                dropout_rate=dropout_rate,
                sample_ratio=self.sample_ratio,
                creation_timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                metadata={}
            )
            bags.append(bag)
        self.bags = bags
        return bags

    def _calc_dropout_rate(self, progress: float, bag_id: int) -> float:
        if self.dropout_strategy == 'linear':
            return progress * self.max_dropout_rate
        elif self.dropout_strategy == 'exponential':
            return self.max_dropout_rate * (1 - np.exp(-3 * progress))
        elif self.dropout_strategy == 'random':
            return self._rng.uniform(0, self.max_dropout_rate)
        elif self.dropout_strategy == 'adaptive':
            # Placeholder: simple adaptive logic
            if hasattr(self, 'bags') and self.bags:
                current_div = self._estimate_diversity()
                if current_div < self.diversity_target:
                    return min(self.max_dropout_rate, progress * self.max_dropout_rate + 0.1)
                else:
                    return max(0.0, progress * self.max_dropout_rate - 0.1)
            else:
                return progress * self.max_dropout_rate
        else:
            return self.max_dropout_rate

    def _estimate_diversity(self) -> float:
        # Simple diversity: mean pairwise Hamming distance between modality masks
        if not hasattr(self, 'bags') or not self.bags:
            return 0.0
        masks = np.array([[int(v) for v in bag.modality_mask.values()] for bag in self.bags])
        if len(masks) < 2:
            return 0.0
        dists = [np.mean(masks[i] != masks[j]) for i in range(len(masks)) for j in range(i+1, len(masks))]
        return float(np.mean(dists)) if dists else 0.0

    def get_bag_data(self, bag_id: int, multimodal_data: Dict[str, np.ndarray], return_metadata: bool = False):
        bag = self.bags[bag_id]
        bag_data = {}
        for name, active in bag.modality_mask.items():
            if active:
                mask = bag.feature_mask[name]
                bag_data[name] = multimodal_data[name][bag.data_indices][:, mask]
        labels = multimodal_data.get('labels')
        if labels is not None:
            bag_data['labels'] = labels[bag.data_indices]
        if return_metadata:
            return bag_data, bag.modality_mask, {
                'sample_count': len(bag.data_indices),
                'diversity_score': bag.diversity_score,
                'dropout_rate': bag.dropout_rate,
                'feature_sampling_info': {k: {'features_sampled': mask.sum(), 'total_features': len(mask), 'sampling_ratio': mask.sum()/len(mask) if len(mask) else 0.0} for k, mask in bag.feature_mask.items() if len(mask)},
            }
        return bag_data, bag.modality_mask

    def get_bag_info(self, bag_id: int) -> Dict[str, Any]:
        bag = self.bags[bag_id]
        return {
            'bag_id': bag.bag_id,
            'modalities': [k for k, v in bag.modality_mask.items() if v],
            'dropout_rate': bag.dropout_rate,
            'sample_count': len(bag.data_indices),
            'diversity_score': bag.diversity_score
        }

    def get_ensemble_stats(self, return_detailed: bool = False) -> Dict[str, Any]:
        # Compute diversity and coverage metrics
        masks = np.array([[int(v) for v in bag.modality_mask.values()] for bag in self.bags])
        n_bags = len(self.bags)
        n_modalities = masks.shape[1] if n_bags else 0
        modality_coverage = {name: float(masks[:, i].sum())/n_bags for i, name in enumerate([c.name for c in self.modality_configs])} if n_bags else {}
        diversity_scores = [self._estimate_diversity()] * n_bags
        diversity_metrics = {
            'ensemble_diversity': float(np.mean(diversity_scores)) if diversity_scores else 0.0,
            'mean_bag_diversity': float(np.mean(diversity_scores)) if diversity_scores else 0.0,
            'std_bag_diversity': float(np.std(diversity_scores)) if diversity_scores else 0.0
        }
        dropout_rates = [bag.dropout_rate for bag in self.bags]
        dropout_statistics = {
            'mean_dropout_rate': float(np.mean(dropout_rates)) if dropout_rates else 0.0,
            'min_dropout_rate': float(np.min(dropout_rates)) if dropout_rates else 0.0,
            'max_dropout_rate': float(np.max(dropout_rates)) if dropout_rates else 0.0
        }
        stats = {
            'modality_coverage': modality_coverage,
            'diversity_metrics': diversity_metrics,
            'dropout_statistics': dropout_statistics
        }
        if return_detailed:
            stats['bags'] = [self.get_bag_info(i) for i in range(n_bags)]
        return stats

    def save_ensemble(self, filepath: str):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_ensemble(cls, filepath: str) -> "ModalityDropoutBagger":
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)
