"""
Stage 3: Base Learner Selection
Simplified implementation that:
1. Takes saved bags with sampled training data and labels
2. Analyzes modalities present and their weightage in each bag
3. Assigns learner types based on bag characteristics
4. Considers task type (regression/classification) and optimization mode
5. Saves bags with their assigned weak learners for Stage 4
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger("base_learner_selection")

def train_pytorch_model(model, X, y, epochs=10, batch_size=32, lr=0.001):
    """Helper function to train PyTorch models."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    # Training loop
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model.forward(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    return model

@dataclass
class BagLearnerConfig:
    """Configuration for a bag with its assigned learner."""
    bag_id: int
    bag_data: Dict[str, np.ndarray]  # Sampled data for this bag
    bag_labels: np.ndarray  # Labels for this bag
    modality_mask: Dict[str, bool]  # Which modalities are active
    feature_mask: Dict[str, np.ndarray]  # Which features are selected
    learner_type: str  # Type of learner assigned
    learner_config: Dict[str, Any]  # Learner-specific configuration
    modality_weights: Dict[str, float]  # Weightage of each modality
    optimization_mode: str  # accuracy, performance, efficiency
    task_type: str  # classification or regression
    expected_performance: float  # Predicted performance score
    learner_instance: Any = None  # The actual learner instance

class BaseLearnerSelector:
    """Simplified base learner selector for Stage 3."""
    
    def __init__(self, 
                 task_type: str = 'classification',
                 optimization_mode: str = 'accuracy',
                 n_classes: int = 2,
                 random_state: int = 42,
                 # New ablation study parameters
                 modality_aware: bool = True,
                 bag_learner_pairing: bool = True,
                 metadata_level: str = 'complete',
                 pairing_focus: str = 'performance',
                 # Modality weightage analysis parameters
                 feature_ratio_weight: float = 0.4,
                 variance_weight: float = 0.3,
                 dimensionality_weight: float = 0.3,
                 # Performance prediction parameters
                 base_performance: float = 0.6,
                 diversity_bonus: float = 0.1,
                 weightage_bonus: float = 0.1,
                 dropout_penalty: float = 0.1,
                 # Learner type selection
                 learner_type: str = 'sklearn'):
        """
        Initialize the base learner selector.
        
        Args:
            task_type: 'classification' or 'regression'
            optimization_mode: 'accuracy', 'performance', or 'efficiency'
            n_classes: Number of classes for classification
            random_state: Random seed for reproducibility
            modality_aware: Enable modality-aware learner selection
            bag_learner_pairing: Enable complete bag-learner pairing storage
            metadata_level: Level of metadata storage ('minimal', 'complete', 'enhanced')
            pairing_focus: Focus for pairing optimization ('performance', 'diversity', 'efficiency')
            feature_ratio_weight: Weight for feature selection ratio in modality importance
            variance_weight: Weight for data variance in modality importance
            dimensionality_weight: Weight for dimensionality in modality importance
            base_performance: Base performance score for prediction
            diversity_bonus: Bonus multiplier for modality diversity
            weightage_bonus: Bonus multiplier for modality weightage
            dropout_penalty: Penalty multiplier for dropout rate
                learner_type: 'sklearn', 'pytorch', 'deep_learning', or 'transformer' - choose base learner implementation
        """
        self.task_type = task_type
        self.optimization_mode = optimization_mode
        self.n_classes = n_classes
        self.random_state = random_state
        
        # New ablation study parameters
        self.modality_aware = modality_aware
        self.bag_learner_pairing = bag_learner_pairing
        self.metadata_level = metadata_level
        self.pairing_focus = pairing_focus
        
        # Modality weightage analysis parameters
        self.feature_ratio_weight = feature_ratio_weight
        self.variance_weight = variance_weight
        self.dimensionality_weight = dimensionality_weight
        
        # Performance prediction parameters
        self.base_performance = base_performance
        self.diversity_bonus = diversity_bonus
        self.weightage_bonus = weightage_bonus
        self.dropout_penalty = dropout_penalty
        
        # Learner type selection
        self.learner_type = learner_type
        
        # Storage for bag-learner configurations
        self.bag_learner_configs: List[BagLearnerConfig] = []
        
        # Storage for ablation study data
        self.pairing_statistics: Dict[str, Any] = {}
        self.metadata_completeness: Dict[str, Any] = {}
        self.ensemble_coherence: Dict[str, Any] = {}
        
        logger.info(f"Initialized BaseLearnerSelector: task={task_type}, mode={optimization_mode}, modality_aware={modality_aware}")
    
    def select_learners_for_bags(self, 
                                bags: List[Any], 
                                bag_data: Dict[int, Dict[str, np.ndarray]]) -> List[BagLearnerConfig]:
        """
        Main method: Select learners for all bags.
        
        Args:
            bags: List of BagConfig objects from Stage 2
            bag_data: Dictionary mapping bag_id to bag data
            
        Returns:
            List of BagLearnerConfig objects
        """
        logger.info(f"Selecting learners for {len(bags)} bags")
        
        self.bag_learner_configs = []
        
        for bag in bags:
            # Get bag data
            bag_id = bag.bag_id
            if bag_id not in bag_data:
                logger.warning(f"No data found for bag {bag_id}, skipping")
                continue
            
            # Analyze bag characteristics
            modality_weights = self._analyze_modality_weightage(bag, bag_data[bag_id])
            
            # Select learner type based on bag characteristics
            learner_type = self._select_learner_type(bag, modality_weights)
            
            # Configure learner based on optimization mode
            learner_config = self._configure_learner(learner_type, bag, modality_weights)
            
            # Predict expected performance
            expected_performance = self._predict_performance(bag, modality_weights, learner_type)
            
            # Create the actual learner instance
            learner_instance = self._create_learner_instance(
                bag_data[bag_id], learner_type, learner_config, bag_data[bag_id]['labels']
            )
            
            # Create bag-learner configuration
            bag_learner_config = BagLearnerConfig(
                bag_id=bag_id,
                bag_data=bag_data[bag_id],
                bag_labels=bag_data[bag_id]['labels'],
                modality_mask=bag.modality_mask,
                feature_mask=bag.feature_mask,
                learner_type=learner_type,
                learner_config=learner_config,
                modality_weights=modality_weights,
                optimization_mode=self.optimization_mode,
                task_type=self.task_type,
                expected_performance=expected_performance,
                learner_instance=learner_instance
            )
            
            self.bag_learner_configs.append(bag_learner_config)
        
        logger.info(f"Selected learners for {len(self.bag_learner_configs)} bags")
        return self.bag_learner_configs
    
    def _analyze_modality_weightage(self, bag: Any, bag_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze the weightage/importance of each modality in the bag.
        
        Args:
            bag: BagConfig object
            bag_data: Data for this specific bag
            
        Returns:
            Dictionary mapping modality names to their weightage scores
        """
        modality_weights = {}
        
        for modality, is_active in bag.modality_mask.items():
            if not is_active:
                modality_weights[modality] = 0.0
                continue
            
            if modality not in bag_data or modality == 'labels':
                continue
            
            # Calculate modality weightage based on:
            # 1. Feature dimensionality
            # 2. Data variance
            # 3. Feature selection ratio
            
            data = bag_data[modality]
            feature_mask = bag.feature_mask.get(modality, np.ones(data.shape[1], dtype=bool))
            
            # Weightage factors
            feature_ratio = np.sum(feature_mask) / len(feature_mask) if len(feature_mask) > 0 else 0
            data_variance = np.var(data) if data.size > 0 else 0
            dimensionality = data.shape[1] if len(data.shape) > 1 else 1
            
            # Combined weightage score (0-1 range) using configurable weights
            weightage = (
                self.feature_ratio_weight * feature_ratio +  # Feature selection importance
                self.variance_weight * min(1.0, data_variance) +  # Data variance importance
                self.dimensionality_weight * min(1.0, dimensionality / 100)  # Dimensionality importance
            )
            
            modality_weights[modality] = min(1.0, max(0.0, weightage))
        
        # Normalize weights to sum to 1
        total_weight = sum(modality_weights.values())
        if total_weight > 0:
            modality_weights = {k: v / total_weight for k, v in modality_weights.items()}
        
        return modality_weights
    
    def _select_learner_type(self, bag: Any, modality_weights: Dict[str, float]) -> str:
        """
        Select learner type based on bag characteristics and modality weights.
        
        Args:
            bag: BagConfig object
            modality_weights: Weightage of each modality
            
        Returns:
            Learner type string
        """
        # Choose learner implementation based on learner_type parameter
        if self.learner_type == 'pytorch':
            return self._select_pytorch_learner_type(bag, modality_weights)
        elif self.learner_type == 'deep_learning':
            return self._select_deep_learning_learner_type(bag, modality_weights)
        elif self.learner_type == 'transformer':
            return self._select_transformer_learner_type(bag, modality_weights)
        else:  # sklearn (default)
            return self._select_sklearn_learner_type(bag, modality_weights)
    
    def _select_pytorch_learner_type(self, bag: Any, modality_weights: Dict[str, float]) -> str:
        """Select PyTorch learner type based on bag characteristics."""
        if not self.modality_aware:
            # Fixed learner selection (baseline for ablation studies)
            if self.optimization_mode == 'accuracy':
                return 'advanced_fusion_learner'
            elif self.optimization_mode == 'performance':
                return 'simple_fusion_learner'
            else:  # efficiency
                return 'tabular_learner'
        
        # Modality-aware selection (novel feature)
        active_modalities = [k for k, v in bag.modality_mask.items() if v]
        n_modalities = len(active_modalities)
        
        if n_modalities == 1:
            # Single modality - use modality-specific learner
            modality = active_modalities[0]
            if modality in ['text', 'clinical_text']:
                return 'text_learner'  # PyTorch learner for text features
            elif modality in ['image', 'visual']:
                return 'image_learner'  # PyTorch learner for image features
            else:
                return 'tabular_learner'  # PyTorch learner for tabular features
        
        elif n_modalities == 2:
            # Two modalities - Use modality-aware selection based on dominant modality
            text_weight = modality_weights.get('text', 0.5)
            metadata_weight = modality_weights.get('metadata', 0.5)
            
            # Select learner based on which modality is more dominant
            if text_weight > metadata_weight:
                return 'text_learner'  # PyTorch learner for text features
            elif metadata_weight > text_weight:
                return 'tabular_learner'  # PyTorch learner for structured metadata
            else:
                return 'simple_fusion_learner'  # PyTorch learner for mixed features
        
        else:
            # Multiple modalities - use advanced fusion
            return 'advanced_fusion_learner'
    
    def _select_sklearn_learner_type(self, bag: Any, modality_weights: Dict[str, float]) -> str:
        """Select sklearn learner type based on bag characteristics."""
        if not self.modality_aware:
            # Fixed learner selection (baseline for ablation studies)
            if self.optimization_mode == 'accuracy':
                return 'sklearn_random_forest'
            elif self.optimization_mode == 'performance':
                return 'sklearn_logistic_regression'
            else:  # efficiency
                return 'sklearn_gradient_boosting'
        
        # Modality-aware selection (novel feature)
        active_modalities = [k for k, v in bag.modality_mask.items() if v]
        n_modalities = len(active_modalities)
        
        if n_modalities == 1:
            # Single modality - use modality-specific learner
            modality = active_modalities[0]
            if modality in ['text', 'clinical_text']:
                return 'sklearn_logistic_regression'  # Best for text features (78.0%)
            elif modality in ['image', 'visual']:
                return 'sklearn_random_forest'  # Good for image features
            else:
                return 'sklearn_gradient_boosting'  # Good for tabular features
        
        elif n_modalities == 2:
            # Two modalities - Use modality-aware selection based on dominant modality
            text_weight = modality_weights.get('text', 0.5)
            metadata_weight = modality_weights.get('metadata', 0.5)
            
            # Select learner based on which modality is more dominant
            if text_weight > metadata_weight:
                return 'sklearn_logistic_regression'  # Best for text features (78.0%)
            elif metadata_weight > text_weight:
                return 'sklearn_random_forest'  # Good for structured metadata (68.0%)
            else:
                return 'sklearn_gradient_boosting'  # Good for mixed features
        
        else:
            # Multiple modalities - use advanced fusion
            return 'sklearn_random_forest'
    
    def _select_deep_learning_learner_type(self, bag: Any, modality_weights: Dict[str, float]) -> str:
        """Select deep learning learner type based on bag characteristics."""
        if not self.modality_aware:
            # Fixed learner selection (baseline for ablation studies)
            if self.optimization_mode == 'accuracy':
                return 'deep_multimodal_fusion'
            elif self.optimization_mode == 'performance':
                return 'deep_simple_fusion'
            else:  # efficiency
                return 'deep_tabular_learner'
        
        # Modality-aware selection (novel feature)
        active_modalities = [k for k, v in bag.modality_mask.items() if v]
        n_modalities = len(active_modalities)
        
        if n_modalities == 1:
            # Single modality - use modality-specific deep learning model
            modality = active_modalities[0]
            if modality in ['text', 'clinical_text']:
                return 'deep_text_lstm'  # LSTM for text sequences
            elif modality in ['image', 'visual']:
                return 'deep_image_cnn'  # CNN for image features
            else:
                return 'deep_metadata_mlp'  # Advanced MLP for tabular features
        
        elif n_modalities == 2:
            # Two modalities - Use modality-aware selection based on dominant modality
            text_weight = modality_weights.get('text', 0.5)
            metadata_weight = modality_weights.get('metadata', 0.5)
            
            # Select learner based on which modality is more dominant
            if text_weight > metadata_weight:
                return 'deep_text_dominant'  # LSTM + attention to metadata
            elif metadata_weight > text_weight:
                return 'deep_metadata_dominant'  # MLP + text embeddings
            else:
                return 'deep_simple_fusion'  # Basic fusion network
        
        else:
            # Multiple modalities - use advanced fusion
            return 'deep_multimodal_fusion'
    
    def _select_transformer_learner_type(self, bag: Any, modality_weights: Dict[str, float]) -> str:
        """Select transformer learner type based on bag characteristics."""
        if not self.modality_aware:
            # Fixed learner selection (baseline for ablation studies)
            if self.optimization_mode == 'accuracy':
                return 'transformer_multimodal'
            elif self.optimization_mode == 'performance':
                return 'transformer_cross_modal'
            else:  # efficiency
                return 'transformer_tabular'
        
        # Modality-aware selection (novel feature)
        active_modalities = [k for k, v in bag.modality_mask.items() if v]
        n_modalities = len(active_modalities)
        
        if n_modalities == 1:
            # Single modality - use modality-specific transformer
            modality = active_modalities[0]
            if modality in ['text', 'clinical_text']:
                return 'transformer_bert'  # Fine-tuned BERT for text
            elif modality in ['image', 'visual']:
                return 'transformer_vision'  # Vision Transformer for images
            else:
                return 'transformer_tabular'  # Transformer for structured data
        
        elif n_modalities == 2:
            # Two modalities - Use cross-modal transformer
            return 'transformer_cross_modal'  # Cross-modal attention
        
        else:
            # Multiple modalities - use multimodal transformer
            return 'transformer_multimodal'
    
    def _configure_learner(self, learner_type: str, bag: Any, modality_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Configure learner parameters based on optimization mode and bag characteristics.
        
        Args:
            learner_type: Type of learner to configure
            bag: BagConfig object
            modality_weights: Weightage of each modality
            
        Returns:
            Learner configuration dictionary
        """
        config = {
            'learner_type': learner_type,
            'task_type': self.task_type,
            'n_classes': self.n_classes,
            'random_state': self.random_state
        }
        
        # Configure based on optimization mode
        if self.optimization_mode == 'accuracy':
            # Prioritize accuracy - use more complex models
            config.update({
                'complexity': 'high',
                'n_estimators': 200,
                'max_depth': 20,
                'learning_rate': 0.05,
                'early_stopping': True
            })
        
        elif self.optimization_mode == 'performance':
            # Balance accuracy and speed
            config.update({
                'complexity': 'medium',
                'n_estimators': 100,
                'max_depth': 10,
                'learning_rate': 0.1,
                'early_stopping': False
            })
        
        elif self.optimization_mode == 'efficiency':
            # Prioritize speed - use simpler models
            config.update({
                'complexity': 'low',
                'n_estimators': 50,
                'max_depth': 5,
                'learning_rate': 0.2,
                'early_stopping': False
            })
        
        # Add modality-specific configurations
        if learner_type == 'text_learner':
            config.update({
                'model_type': 'mlp',  # Simple MLP for text features
                'hidden_layers': (128, 64),
                'dropout': 0.2
            })
        
        elif learner_type == 'image_learner':
            config.update({
                'model_type': 'random_forest',  # RF for image features
                'max_features': 'sqrt',
                'min_samples_split': 5
            })
        
        elif learner_type == 'tabular_learner':
            config.update({
                'model_type': 'gradient_boosting',
                'subsample': 0.8,
                'min_samples_leaf': 2
            })
        
        elif 'fusion' in learner_type:
            config.update({
                'fusion_type': 'weighted' if learner_type == 'simple_fusion_learner' else 'attention',
                'modality_weights': modality_weights,
                'cross_modal_learning': True
            })
        
        return config
    
    def _predict_performance(self, bag: Any, modality_weights: Dict[str, float], learner_type: str) -> float:
        """
        Predict expected performance for the bag-learner combination.
        
        Args:
            bag: BagConfig object
            modality_weights: Weightage of each modality
            learner_type: Type of learner
            
        Returns:
            Expected performance score (0-1)
        """
        # Base performance (configurable)
        base_performance = self.base_performance
        
        # Modality diversity bonus (configurable)
        n_active_modalities = sum(1 for v in bag.modality_mask.values() if v)
        diversity_bonus = self.diversity_bonus * (n_active_modalities - 1)
        
        # Modality weightage bonus (configurable)
        max_weight = max(modality_weights.values()) if modality_weights else 0
        weightage_bonus = self.weightage_bonus * max_weight
        
        # Dropout penalty (configurable)
        dropout_penalty = self.dropout_penalty * getattr(bag, 'dropout_rate', 0)
        
        # Learner type bonus
        learner_bonus = {
            'text_learner': 0.05,
            'image_learner': 0.05,
            'tabular_learner': 0.05,
            'simple_fusion_learner': 0.1,
            'advanced_fusion_learner': 0.15
        }.get(learner_type, 0)
        
        # Optimization mode bonus
        mode_bonus = {
            'accuracy': 0.1,
            'performance': 0.05,
            'efficiency': 0.0
        }.get(self.optimization_mode, 0)
        
        # Calculate final performance
        performance = (
            base_performance + 
            diversity_bonus + 
            weightage_bonus - 
            dropout_penalty + 
            learner_bonus + 
            mode_bonus
        )
        
        return min(1.0, max(0.0, performance))
    
    def get_bag_learner_summary(self) -> Dict[str, Any]:
        """Get summary of bag-learner assignments."""
        if not self.bag_learner_configs:
            return {}
        
        # Count learner types
        learner_type_counts = {}
        optimization_modes = {}
        expected_performances = []
        
        for config in self.bag_learner_configs:
            learner_type_counts[config.learner_type] = learner_type_counts.get(config.learner_type, 0) + 1
            optimization_modes[config.optimization_mode] = optimization_modes.get(config.optimization_mode, 0) + 1
            expected_performances.append(config.expected_performance)
        
        return {
            'total_bags': len(self.bag_learner_configs),
            'learner_type_distribution': learner_type_counts,
            'optimization_mode': self.optimization_mode,
            'task_type': self.task_type,
            'average_expected_performance': np.mean(expected_performances),
            'performance_range': (min(expected_performances), max(expected_performances)),
            'bag_learner_configs': self.bag_learner_configs
        }
    
    def get_bag_learner_config(self, bag_id: int) -> Optional[BagLearnerConfig]:
        """Get configuration for a specific bag."""
        for config in self.bag_learner_configs:
            if config.bag_id == bag_id:
                return config
        return None

    # Stage 3 Interpretability Test Methods

    def run_stage3_interpretability_test(self, test_type: str, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run specific Stage 3 interpretability tests.
        
        Parameters
        ----------
        test_type : str
            Type of interpretability test to run
        test_scenarios : dict
            Dictionary of test scenarios with their configurations
            
        Returns
        -------
        dict
            Results of the interpretability test
        """
        if test_type == 'modality_importance':
            return self.test_modality_importance_interpretability(test_scenarios)
        elif test_type == 'learner_selection':
            return self.test_learner_selection_interpretability(test_scenarios)
        elif test_type == 'performance_prediction':
            return self.test_performance_prediction_interpretability(test_scenarios)
        elif test_type == 'bag_learner_pairing':
            return self.test_bag_learner_pairing_interpretability(test_scenarios)
        elif test_type == 'ensemble_coherence':
            return self.test_ensemble_coherence_interpretability(test_scenarios)
        elif test_type == 'optimization_mode':
            return self.test_optimization_mode_interpretability(test_scenarios)
        else:
            raise ValueError(f"Unknown test_type: {test_type}. Must be one of: modality_importance, learner_selection, performance_prediction, bag_learner_pairing, ensemble_coherence, optimization_mode")

    def test_modality_importance_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test interpretability of modality importance in learner selection."""
        results = {}
        
        for scenario_name, scenario_config in test_scenarios.items():
            try:
                # Create test selector with specific weightage parameters
                test_selector = BaseLearnerSelector(
                    task_type=self.task_type,
                    optimization_mode=self.optimization_mode,
                    n_classes=self.n_classes,
                    random_state=self.random_state,
                    modality_aware=self.modality_aware,
                    bag_learner_pairing=self.bag_learner_pairing,
                    metadata_level=self.metadata_level,
                    pairing_focus=self.pairing_focus,
                    feature_ratio_weight=scenario_config.get('feature_ratio_weight', self.feature_ratio_weight),
                    variance_weight=scenario_config.get('variance_weight', self.variance_weight),
                    dimensionality_weight=scenario_config.get('dimensionality_weight', self.dimensionality_weight),
                    base_performance=self.base_performance,
                    diversity_bonus=self.diversity_bonus,
                    weightage_bonus=self.weightage_bonus,
                    dropout_penalty=self.dropout_penalty
                )
                
                # Use existing bag data for testing
                if hasattr(self, '_test_bags') and hasattr(self, '_test_bag_data'):
                    test_selector.select_learners_for_bags(self._test_bags, self._test_bag_data)
                else:
                    # Create dummy data for testing
                    dummy_bags = [type('Bag', (), {'bag_id': i, 'modality_mask': {'text': True, 'image': True}, 'feature_mask': {'text': np.ones(10, dtype=bool), 'image': np.ones(5, dtype=bool)}}) for i in range(3)]
                    dummy_bag_data = {i: {'text': np.random.randn(10, 10), 'image': np.random.randn(10, 5), 'labels': np.random.randint(0, 3, 10)} for i in range(3)}
                    dummy_labels = np.random.randint(0, 3, 10)
                    test_selector.select_learners_for_bags(dummy_bags, dummy_bag_data)
                
                # Analyze modality importance
                bag_learner_summary = test_selector.get_bag_learner_summary()
                stage3_interpretability = test_selector.get_stage3_interpretability_data()
                
                # Calculate modality importance interpretability metrics
                modality_importance = self._calculate_modality_importance_interpretability(
                    bag_learner_summary, stage3_interpretability
                )
                
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'modality_importance': modality_importance,
                    'interpretability_score': self._calculate_modality_importance_interpretability_score(modality_importance),
                    'success': True
                }
                
            except Exception as e:
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'error': str(e),
                    'success': False
                }
        
        return results

    def test_learner_selection_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test interpretability of learner selection decisions."""
        results = {}
        
        for scenario_name, scenario_config in test_scenarios.items():
            try:
                # Create test selector with specific selection parameters
                test_selector = BaseLearnerSelector(
                    task_type=self.task_type,
                    optimization_mode=scenario_config.get('optimization_mode', self.optimization_mode),
                    n_classes=self.n_classes,
                    random_state=self.random_state,
                    modality_aware=scenario_config.get('modality_aware', self.modality_aware),
                    bag_learner_pairing=scenario_config.get('bag_learner_pairing', self.bag_learner_pairing),
                    metadata_level=self.metadata_level,
                    pairing_focus=self.pairing_focus,
                    feature_ratio_weight=self.feature_ratio_weight,
                    variance_weight=self.variance_weight,
                    dimensionality_weight=self.dimensionality_weight,
                    base_performance=self.base_performance,
                    diversity_bonus=self.diversity_bonus,
                    weightage_bonus=self.weightage_bonus,
                    dropout_penalty=self.dropout_penalty
                )
                
                # Use existing bag data for testing
                if hasattr(self, '_test_bags') and hasattr(self, '_test_bag_data'):
                    test_selector.select_learners_for_bags(self._test_bags, self._test_bag_data)
                else:
                    # Create dummy data for testing
                    dummy_bags = [type('Bag', (), {'bag_id': i, 'modality_mask': {'text': True, 'image': True}, 'feature_mask': {'text': np.ones(10, dtype=bool), 'image': np.ones(5, dtype=bool)}}) for i in range(3)]
                    dummy_bag_data = {i: {'text': np.random.randn(10, 10), 'image': np.random.randn(10, 5), 'labels': np.random.randint(0, 3, 10)} for i in range(3)}
                    dummy_labels = np.random.randint(0, 3, 10)
                    test_selector.select_learners_for_bags(dummy_bags, dummy_bag_data)
                
                # Analyze learner selection decisions
                bag_learner_summary = test_selector.get_bag_learner_summary()
                pairing_stats = test_selector.get_pairing_statistics()
                ensemble_coherence = test_selector.get_ensemble_coherence()
                
                # Calculate selection decision interpretability
                selection_decisions = self._calculate_learner_selection_interpretability(
                    bag_learner_summary, pairing_stats, ensemble_coherence
                )
                
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'selection_decisions': selection_decisions,
                    'interpretability_score': self._calculate_learner_selection_interpretability_score(selection_decisions),
                    'success': True
                }
                
            except Exception as e:
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'error': str(e),
                    'success': False
                }
        
        return results

    def test_performance_prediction_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test interpretability of performance prediction system."""
        results = {}
        
        for scenario_name, scenario_config in test_scenarios.items():
            try:
                # Create test selector with specific prediction parameters
                test_selector = BaseLearnerSelector(
                    task_type=self.task_type,
                    optimization_mode=self.optimization_mode,
                    n_classes=self.n_classes,
                    random_state=self.random_state,
                    modality_aware=self.modality_aware,
                    bag_learner_pairing=self.bag_learner_pairing,
                    metadata_level=self.metadata_level,
                    pairing_focus=self.pairing_focus,
                    feature_ratio_weight=self.feature_ratio_weight,
                    variance_weight=self.variance_weight,
                    dimensionality_weight=self.dimensionality_weight,
                    base_performance=scenario_config.get('base_performance', self.base_performance),
                    diversity_bonus=scenario_config.get('diversity_bonus', self.diversity_bonus),
                    weightage_bonus=scenario_config.get('weightage_bonus', self.weightage_bonus),
                    dropout_penalty=scenario_config.get('dropout_penalty', self.dropout_penalty)
                )
                
                # Use existing bag data for testing
                if hasattr(self, '_test_bags') and hasattr(self, '_test_bag_data'):
                    test_selector.select_learners_for_bags(self._test_bags, self._test_bag_data)
                else:
                    # Create dummy data for testing
                    dummy_bags = [type('Bag', (), {'bag_id': i, 'modality_mask': {'text': True, 'image': True}, 'feature_mask': {'text': np.ones(10, dtype=bool), 'image': np.ones(5, dtype=bool)}}) for i in range(3)]
                    dummy_bag_data = {i: {'text': np.random.randn(10, 10), 'image': np.random.randn(10, 5), 'labels': np.random.randint(0, 3, 10)} for i in range(3)}
                    dummy_labels = np.random.randint(0, 3, 10)
                    test_selector.select_learners_for_bags(dummy_bags, dummy_bag_data)
                
                # Analyze performance prediction system
                bag_learner_summary = test_selector.get_bag_learner_summary()
                stage3_interpretability = test_selector.get_stage3_interpretability_data()
                
                # Calculate performance prediction interpretability
                prediction_interpretability = self._calculate_performance_prediction_interpretability(
                    bag_learner_summary, stage3_interpretability
                )
                
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'prediction_interpretability': prediction_interpretability,
                    'interpretability_score': self._calculate_performance_prediction_interpretability_score(prediction_interpretability),
                    'success': True
                }
                
            except Exception as e:
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'error': str(e),
                    'success': False
                }
        
        return results

    def test_bag_learner_pairing_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test interpretability of bag-learner pairing system."""
        results = {}
        
        for scenario_name, scenario_config in test_scenarios.items():
            try:
                # Create test selector with specific pairing configuration
                test_selector = BaseLearnerSelector(
                    task_type=self.task_type,
                    optimization_mode=self.optimization_mode,
                    n_classes=self.n_classes,
                    random_state=self.random_state,
                    modality_aware=self.modality_aware,
                    bag_learner_pairing=scenario_config.get('bag_learner_pairing', self.bag_learner_pairing),
                    metadata_level=scenario_config.get('metadata_level', self.metadata_level),
                    pairing_focus=scenario_config.get('pairing_focus', self.pairing_focus),
                    feature_ratio_weight=self.feature_ratio_weight,
                    variance_weight=self.variance_weight,
                    dimensionality_weight=self.dimensionality_weight,
                    base_performance=self.base_performance,
                    diversity_bonus=self.diversity_bonus,
                    weightage_bonus=self.weightage_bonus,
                    dropout_penalty=self.dropout_penalty
                )
                
                # Use existing bag data for testing
                if hasattr(self, '_test_bags') and hasattr(self, '_test_bag_data'):
                    test_selector.select_learners_for_bags(self._test_bags, self._test_bag_data)
                else:
                    # Create dummy data for testing
                    dummy_bags = [type('Bag', (), {'bag_id': i, 'modality_mask': {'text': True, 'image': True}, 'feature_mask': {'text': np.ones(10, dtype=bool), 'image': np.ones(5, dtype=bool)}}) for i in range(3)]
                    dummy_bag_data = {i: {'text': np.random.randn(10, 10), 'image': np.random.randn(10, 5), 'labels': np.random.randint(0, 3, 10)} for i in range(3)}
                    dummy_labels = np.random.randint(0, 3, 10)
                    test_selector.select_learners_for_bags(dummy_bags, dummy_bag_data)
                
                # Analyze bag-learner pairing system
                pairing_stats = test_selector.get_pairing_statistics()
                metadata_completeness = test_selector.get_metadata_completeness()
                ensemble_coherence = test_selector.get_ensemble_coherence()
                
                # Calculate pairing interpretability
                pairing_interpretability = self._calculate_bag_learner_pairing_interpretability(
                    pairing_stats, metadata_completeness, ensemble_coherence
                )
                
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'pairing_interpretability': pairing_interpretability,
                    'interpretability_score': self._calculate_bag_learner_pairing_interpretability_score(pairing_interpretability),
                    'success': True
                }
                
            except Exception as e:
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'error': str(e),
                    'success': False
                }
        
        return results

    def test_ensemble_coherence_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test interpretability of ensemble coherence and consistency."""
        results = {}
        
        for scenario_name, scenario_config in test_scenarios.items():
            try:
                # Create test selector with specific ensemble configuration
                test_selector = BaseLearnerSelector(
                    task_type=self.task_type,
                    optimization_mode=self.optimization_mode,
                    n_classes=self.n_classes,
                    random_state=self.random_state,
                    modality_aware=self.modality_aware,
                    bag_learner_pairing=self.bag_learner_pairing,
                    metadata_level=self.metadata_level,
                    pairing_focus=self.pairing_focus,
                    feature_ratio_weight=self.feature_ratio_weight,
                    variance_weight=self.variance_weight,
                    dimensionality_weight=self.dimensionality_weight,
                    base_performance=self.base_performance,
                    diversity_bonus=self.diversity_bonus,
                    weightage_bonus=self.weightage_bonus,
                    dropout_penalty=self.dropout_penalty
                )
                
                # Use existing bag data for testing
                if hasattr(self, '_test_bags') and hasattr(self, '_test_bag_data'):
                    test_selector.select_learners_for_bags(self._test_bags, self._test_bag_data)
                else:
                    # Create dummy data for testing
                    dummy_bags = [type('Bag', (), {'bag_id': i, 'modality_mask': {'text': True, 'image': True}, 'feature_mask': {'text': np.ones(10, dtype=bool), 'image': np.ones(5, dtype=bool)}}) for i in range(3)]
                    dummy_bag_data = {i: {'text': np.random.randn(10, 10), 'image': np.random.randn(10, 5), 'labels': np.random.randint(0, 3, 10)} for i in range(3)}
                    dummy_labels = np.random.randint(0, 3, 10)
                    test_selector.select_learners_for_bags(dummy_bags, dummy_bag_data)
                
                # Analyze ensemble coherence
                bag_learner_summary = test_selector.get_bag_learner_summary()
                pairing_stats = test_selector.get_pairing_statistics()
                ensemble_coherence = test_selector.get_ensemble_coherence()
                
                # Calculate ensemble coherence interpretability
                coherence_interpretability = self._calculate_ensemble_coherence_interpretability(
                    bag_learner_summary, pairing_stats, ensemble_coherence
                )
                
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'coherence_interpretability': coherence_interpretability,
                    'interpretability_score': self._calculate_ensemble_coherence_interpretability_score(coherence_interpretability),
                    'success': True
                }
                
            except Exception as e:
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'error': str(e),
                    'success': False
                }
        
        return results

    def test_optimization_mode_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test interpretability of optimization mode selection."""
        results = {}
        
        for scenario_name, scenario_config in test_scenarios.items():
            try:
                # Create test selector with specific optimization mode
                test_selector = BaseLearnerSelector(
                    task_type=self.task_type,
                    optimization_mode=scenario_config.get('optimization_mode', 'accuracy'),
                    n_classes=self.n_classes,
                    random_state=self.random_state,
                    modality_aware=self.modality_aware,
                    bag_learner_pairing=self.bag_learner_pairing,
                    metadata_level=self.metadata_level,
                    pairing_focus=self.pairing_focus,
                    feature_ratio_weight=self.feature_ratio_weight,
                    variance_weight=self.variance_weight,
                    dimensionality_weight=self.dimensionality_weight,
                    base_performance=self.base_performance,
                    diversity_bonus=self.diversity_bonus,
                    weightage_bonus=self.weightage_bonus,
                    dropout_penalty=self.dropout_penalty
                )
                
                # Use existing bag data for testing
                if hasattr(self, '_test_bags') and hasattr(self, '_test_bag_data'):
                    test_selector.select_learners_for_bags(self._test_bags, self._test_bag_data)
                else:
                    # Create dummy data for testing
                    dummy_bags = [type('Bag', (), {'bag_id': i, 'modality_mask': {'text': True, 'image': True}, 'feature_mask': {'text': np.ones(10, dtype=bool), 'image': np.ones(5, dtype=bool)}}) for i in range(3)]
                    dummy_bag_data = {i: {'text': np.random.randn(10, 10), 'image': np.random.randn(10, 5), 'labels': np.random.randint(0, 3, 10)} for i in range(3)}
                    dummy_labels = np.random.randint(0, 3, 10)
                    test_selector.select_learners_for_bags(dummy_bags, dummy_bag_data)
                
                # Analyze optimization mode impact
                bag_learner_summary = test_selector.get_bag_learner_summary()
                ensemble_coherence = test_selector.get_ensemble_coherence()
                stage3_interpretability = test_selector.get_stage3_interpretability_data()
                
                # Calculate optimization mode interpretability
                optimization_interpretability = self._calculate_optimization_mode_interpretability(
                    bag_learner_summary, ensemble_coherence, stage3_interpretability
                )
                
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'optimization_interpretability': optimization_interpretability,
                    'interpretability_score': self._calculate_optimization_mode_interpretability_score(optimization_interpretability),
                    'success': True
                }
                
            except Exception as e:
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'error': str(e),
                    'success': False
                }
        
        return results

    # Stage 3 Robustness Test Methods

    def test_optimization_mode_robustness(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test robustness of optimization mode selection."""
        results = {}
        
        for scenario_name, scenario_config in test_scenarios.items():
            try:
                # Create test selector with different optimization mode
                test_selector = BaseLearnerSelector(
                    task_type=self.task_type,
                    optimization_mode=scenario_config.get('optimization_mode', 'accuracy'),
                    n_classes=self.n_classes,
                    random_state=self.random_state,
                    modality_aware=self.modality_aware,
                    bag_learner_pairing=self.bag_learner_pairing,
                    metadata_level=self.metadata_level,
                    pairing_focus=self.pairing_focus,
                    feature_ratio_weight=self.feature_ratio_weight,
                    variance_weight=self.variance_weight,
                    dimensionality_weight=self.dimensionality_weight,
                    base_performance=self.base_performance,
                    diversity_bonus=self.diversity_bonus,
                    weightage_bonus=self.weightage_bonus,
                    dropout_penalty=self.dropout_penalty
                )
                
                # Test with same bag data - get from bag_data parameter if available
                bag_data = scenario_config.get('bag_data', None)
                if bag_data is None and hasattr(self, 'bag_data') and self.bag_data:
                    bag_data = self.bag_data
                
                if bag_data:
                    # Get bags from scenario config or create default bags
                    bags = scenario_config.get('bags', None)
                    if bags is None:
                        # Create default bag objects if not provided
                        from dataclasses import dataclass
                        @dataclass
                        class DefaultBag:
                            bag_id: int
                        
                        bags = [DefaultBag(i) for i in range(len(bag_data))]
                    
                    # Convert bag_data to the expected format
                    bag_data_dict = {i: bag for i, bag in enumerate(bag_data)}
                    test_selector.select_learners_for_bags(bags, bag_data_dict)
                    
                    bag_learner_summary = test_selector.get_bag_learner_summary()
                    pairing_stats = test_selector.get_pairing_statistics()
                    ensemble_coherence = test_selector.get_ensemble_coherence()
                    
                    results[scenario_name] = {
                        'scenario_config': scenario_config,
                        'bag_learner_summary': bag_learner_summary,
                        'pairing_statistics': pairing_stats,
                        'ensemble_coherence': ensemble_coherence,
                        'robustness_score': np.mean([
                            pairing_stats.get('modality_learner_match_rate', 0),
                            pairing_stats.get('pairing_consistency', 0),
                            ensemble_coherence.get('overall_coherence', 0)
                        ]),
                        'success': True
                    }
                else:
                    results[scenario_name] = {
                        'scenario_config': scenario_config,
                        'error': 'No bag data available for testing',
                        'success': False
                    }
                
            except Exception as e:
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'error': str(e),
                    'success': False
                }
        
        return results

    def test_modality_weightage_robustness(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test robustness of modality weightage parameters."""
        results = {}
        
        for scenario_name, scenario_config in test_scenarios.items():
            try:
                # Create test selector with different weightage parameters
                test_selector = BaseLearnerSelector(
                    task_type=self.task_type,
                    optimization_mode=self.optimization_mode,
                    n_classes=self.n_classes,
                    random_state=self.random_state,
                    modality_aware=self.modality_aware,
                    bag_learner_pairing=self.bag_learner_pairing,
                    metadata_level=self.metadata_level,
                    pairing_focus=self.pairing_focus,
                    feature_ratio_weight=scenario_config.get('feature_ratio_weight', self.feature_ratio_weight),
                    variance_weight=scenario_config.get('variance_weight', self.variance_weight),
                    dimensionality_weight=scenario_config.get('dimensionality_weight', self.dimensionality_weight),
                    base_performance=self.base_performance,
                    diversity_bonus=self.diversity_bonus,
                    weightage_bonus=self.weightage_bonus,
                    dropout_penalty=self.dropout_penalty
                )
                
                # Test with same bag data - get from bag_data parameter if available
                bag_data = scenario_config.get('bag_data', None)
                if bag_data is None and hasattr(self, 'bag_data') and self.bag_data:
                    bag_data = self.bag_data
                
                if bag_data:
                    # Get bags from scenario config or create default bags
                    bags = scenario_config.get('bags', None)
                    if bags is None:
                        # Create default bag objects if not provided
                        from dataclasses import dataclass
                        @dataclass
                        class DefaultBag:
                            bag_id: int
                        
                        bags = [DefaultBag(i) for i in range(len(bag_data))]
                    
                    # Convert bag_data to the expected format
                    bag_data_dict = {i: bag for i, bag in enumerate(bag_data)}
                    test_selector.select_learners_for_bags(bags, bag_data_dict)
                    
                    bag_learner_summary = test_selector.get_bag_learner_summary()
                    pairing_stats = test_selector.get_pairing_statistics()
                    ensemble_coherence = test_selector.get_ensemble_coherence()
                    
                    # Calculate weightage sensitivity
                    weightage_sensitivity = np.std([
                        scenario_config.get('feature_ratio_weight', self.feature_ratio_weight),
                        scenario_config.get('variance_weight', self.variance_weight),
                        scenario_config.get('dimensionality_weight', self.dimensionality_weight)
                    ])
                    
                    results[scenario_name] = {
                        'scenario_config': scenario_config,
                        'bag_learner_summary': bag_learner_summary,
                        'pairing_statistics': pairing_stats,
                        'ensemble_coherence': ensemble_coherence,
                        'weightage_sensitivity': weightage_sensitivity,
                        'robustness_score': np.mean([
                            pairing_stats.get('modality_learner_match_rate', 0),
                            pairing_stats.get('pairing_consistency', 0),
                            ensemble_coherence.get('overall_coherence', 0),
                            1.0 - min(weightage_sensitivity, 1.0)  # Lower sensitivity = higher robustness
                        ]),
                        'success': True
                    }
                else:
                    results[scenario_name] = {
                        'scenario_config': scenario_config,
                        'error': 'No bag data available for testing',
                        'success': False
                    }
                
            except Exception as e:
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'error': str(e),
                    'success': False
                }
        
        return results

    def test_bag_learner_pairing_robustness(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test robustness of bag-learner pairing strategies."""
        results = {}
        
        for scenario_name, scenario_config in test_scenarios.items():
            try:
                # Create test selector with different pairing configuration
                test_selector = BaseLearnerSelector(
                    task_type=self.task_type,
                    optimization_mode=self.optimization_mode,
                    n_classes=self.n_classes,
                    random_state=self.random_state,
                    modality_aware=self.modality_aware,
                    bag_learner_pairing=scenario_config.get('bag_learner_pairing', self.bag_learner_pairing),
                    metadata_level=scenario_config.get('metadata_level', self.metadata_level),
                    pairing_focus=scenario_config.get('pairing_focus', self.pairing_focus),
                    feature_ratio_weight=self.feature_ratio_weight,
                    variance_weight=self.variance_weight,
                    dimensionality_weight=self.dimensionality_weight,
                    base_performance=self.base_performance,
                    diversity_bonus=self.diversity_bonus,
                    weightage_bonus=self.weightage_bonus,
                    dropout_penalty=self.dropout_penalty
                )
                
                # Test with same bag data - get from bag_data parameter if available
                bag_data = scenario_config.get('bag_data', None)
                if bag_data is None and hasattr(self, 'bag_data') and self.bag_data:
                    bag_data = self.bag_data
                
                if bag_data:
                    # Get bags from scenario config or create default bags
                    bags = scenario_config.get('bags', None)
                    if bags is None:
                        # Create default bag objects if not provided
                        from dataclasses import dataclass
                        @dataclass
                        class DefaultBag:
                            bag_id: int
                        
                        bags = [DefaultBag(i) for i in range(len(bag_data))]
                    
                    # Convert bag_data to the expected format
                    bag_data_dict = {i: bag for i, bag in enumerate(bag_data)}
                    test_selector.select_learners_for_bags(bags, bag_data_dict)
                    
                    bag_learner_summary = test_selector.get_bag_learner_summary()
                    pairing_stats = test_selector.get_pairing_statistics()
                    ensemble_coherence = test_selector.get_ensemble_coherence()
                    metadata_completeness = test_selector.get_metadata_completeness()
                    
                    results[scenario_name] = {
                        'scenario_config': scenario_config,
                        'bag_learner_summary': bag_learner_summary,
                        'pairing_statistics': pairing_stats,
                        'ensemble_coherence': ensemble_coherence,
                        'metadata_completeness': metadata_completeness,
                        'robustness_score': np.mean([
                            pairing_stats.get('modality_learner_match_rate', 0),
                            pairing_stats.get('pairing_consistency', 0),
                            ensemble_coherence.get('overall_coherence', 0),
                            metadata_completeness.get('average_completeness', 0)
                        ]),
                        'success': True
                    }
                else:
                    results[scenario_name] = {
                        'scenario_config': scenario_config,
                        'error': 'No bag data available for testing',
                        'success': False
                    }
                
            except Exception as e:
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'error': str(e),
                    'success': False
                }
        
        return results

    def test_performance_prediction_robustness(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test robustness of performance prediction system."""
        results = {}
        
        for scenario_name, scenario_config in test_scenarios.items():
            try:
                # Create test selector with different performance prediction parameters
                test_selector = BaseLearnerSelector(
                    task_type=self.task_type,
                    optimization_mode=self.optimization_mode,
                    n_classes=self.n_classes,
                    random_state=self.random_state,
                    modality_aware=self.modality_aware,
                    bag_learner_pairing=self.bag_learner_pairing,
                    metadata_level=self.metadata_level,
                    pairing_focus=self.pairing_focus,
                    feature_ratio_weight=self.feature_ratio_weight,
                    variance_weight=self.variance_weight,
                    dimensionality_weight=self.dimensionality_weight,
                    base_performance=scenario_config.get('base_performance', self.base_performance),
                    diversity_bonus=scenario_config.get('diversity_bonus', self.diversity_bonus),
                    weightage_bonus=scenario_config.get('weightage_bonus', self.weightage_bonus),
                    dropout_penalty=scenario_config.get('dropout_penalty', self.dropout_penalty)
                )
                
                # Test with same bag data - get from bag_data parameter if available
                bag_data = scenario_config.get('bag_data', None)
                if bag_data is None and hasattr(self, 'bag_data') and self.bag_data:
                    bag_data = self.bag_data
                
                if bag_data:
                    # Get bags from scenario config or create default bags
                    bags = scenario_config.get('bags', None)
                    if bags is None:
                        # Create default bag objects if not provided
                        from dataclasses import dataclass
                        @dataclass
                        class DefaultBag:
                            bag_id: int
                        
                        bags = [DefaultBag(i) for i in range(len(bag_data))]
                    
                    # Convert bag_data to the expected format
                    bag_data_dict = {i: bag for i, bag in enumerate(bag_data)}
                    test_selector.select_learners_for_bags(bags, bag_data_dict)
                    
                    bag_learner_summary = test_selector.get_bag_learner_summary()
                    pairing_stats = test_selector.get_pairing_statistics()
                    ensemble_coherence = test_selector.get_ensemble_coherence()
                    
                    # Calculate performance prediction stability
                    expected_performances = [config.expected_performance for config in test_selector.bag_learner_configs]
                    performance_stability = 1.0 - (np.std(expected_performances) / np.mean(expected_performances)) if np.mean(expected_performances) > 0 else 0
                    
                    results[scenario_name] = {
                        'scenario_config': scenario_config,
                        'bag_learner_summary': bag_learner_summary,
                        'pairing_statistics': pairing_stats,
                        'ensemble_coherence': ensemble_coherence,
                        'performance_stability': performance_stability,
                        'robustness_score': np.mean([
                            pairing_stats.get('modality_learner_match_rate', 0),
                            pairing_stats.get('pairing_consistency', 0),
                            ensemble_coherence.get('overall_coherence', 0),
                            performance_stability
                        ]),
                        'success': True
                    }
                else:
                    results[scenario_name] = {
                        'scenario_config': scenario_config,
                        'error': 'No bag data available for testing',
                        'success': False
                    }
                
            except Exception as e:
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'error': str(e),
                    'success': False
                }
        
        return results

    def test_ensemble_size_robustness(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test robustness across different ensemble sizes."""
        results = {}
        
        for scenario_name, scenario_config in test_scenarios.items():
            try:
                # Create test selector with same parameters
                test_selector = BaseLearnerSelector(
                    task_type=self.task_type,
                    optimization_mode=self.optimization_mode,
                    n_classes=self.n_classes,
                    random_state=self.random_state,
                    modality_aware=self.modality_aware,
                    bag_learner_pairing=self.bag_learner_pairing,
                    metadata_level=self.metadata_level,
                    pairing_focus=self.pairing_focus,
                    feature_ratio_weight=self.feature_ratio_weight,
                    variance_weight=self.variance_weight,
                    dimensionality_weight=self.dimensionality_weight,
                    base_performance=self.base_performance,
                    diversity_bonus=self.diversity_bonus,
                    weightage_bonus=self.weightage_bonus,
                    dropout_penalty=self.dropout_penalty
                )
                
                # Test with subset of bag data based on ensemble size
                bag_data = scenario_config.get('bag_data', None)
                if bag_data is None and hasattr(self, 'bag_data') and self.bag_data:
                    bag_data = self.bag_data
                
                if bag_data:
                    ensemble_size = scenario_config.get('ensemble_size', len(bag_data))
                    subset_bag_data = bag_data[:min(ensemble_size, len(bag_data))]
                    
                    # Get bags from scenario config or create default bags
                    bags = scenario_config.get('bags', None)
                    if bags is None:
                        # Create default bag objects if not provided
                        from dataclasses import dataclass
                        @dataclass
                        class DefaultBag:
                            bag_id: int
                        
                        bags = [DefaultBag(i) for i in range(len(subset_bag_data))]
                    else:
                        # Subset bags to match ensemble size
                        bags = bags[:min(ensemble_size, len(bags))]
                    
                    # Convert bag_data to the expected format
                    bag_data_dict = {i: bag for i, bag in enumerate(subset_bag_data)}
                    test_selector.select_learners_for_bags(bags, bag_data_dict)
                    
                    bag_learner_summary = test_selector.get_bag_learner_summary()
                    pairing_stats = test_selector.get_pairing_statistics()
                    ensemble_coherence = test_selector.get_ensemble_coherence()
                    
                    # Calculate size scalability metrics
                    size_efficiency = bag_learner_summary.get('total_bags', 0) / ensemble_size if ensemble_size > 0 else 0
                    diversity_maintained = ensemble_coherence.get('learner_type_diversity', 0) / max(ensemble_size, 1)
                    
                    results[scenario_name] = {
                        'scenario_config': scenario_config,
                        'bag_learner_summary': bag_learner_summary,
                        'pairing_statistics': pairing_stats,
                        'ensemble_coherence': ensemble_coherence,
                        'size_efficiency': size_efficiency,
                        'diversity_maintained': diversity_maintained,
                        'robustness_score': np.mean([
                            pairing_stats.get('modality_learner_match_rate', 0),
                            pairing_stats.get('pairing_consistency', 0),
                            ensemble_coherence.get('overall_coherence', 0),
                            size_efficiency,
                            diversity_maintained
                        ]),
                        'success': True
                    }
                else:
                    results[scenario_name] = {
                        'scenario_config': scenario_config,
                        'error': 'No bag data available for testing',
                        'success': False
                    }
                
            except Exception as e:
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'error': str(e),
                    'success': False
                }
        
        return results

    # Helper methods for calculating interpretability metrics

    def _calculate_modality_importance_interpretability(self, bag_learner_summary, stage3_interpretability):
        """Calculate modality importance interpretability metrics."""
        return {
            'modality_importance_variance': np.var([config.modality_weights for config in self.bag_learner_configs]),
            'learner_type_distribution': bag_learner_summary.get('learner_type_distribution', {}),
            'average_expected_performance': bag_learner_summary.get('average_expected_performance', 0),
            'total_bags': bag_learner_summary.get('total_bags', 0)
        }

    def _calculate_modality_importance_interpretability_score(self, modality_importance):
        """Calculate overall modality importance interpretability score."""
        return np.mean([
            min(modality_importance['modality_importance_variance'], 1.0),  # Normalize variance
            modality_importance['average_expected_performance']
        ])

    def _calculate_learner_selection_interpretability(self, bag_learner_summary, pairing_stats, ensemble_coherence):
        """Calculate learner selection interpretability metrics."""
        return {
            'modality_learner_match_rate': pairing_stats.get('modality_learner_match_rate', 0),
            'pairing_consistency': pairing_stats.get('pairing_consistency', 0),
            'learner_type_diversity': ensemble_coherence.get('learner_type_diversity', 0),
            'total_bags': bag_learner_summary.get('total_bags', 0)
        }

    def _calculate_learner_selection_interpretability_score(self, selection_decisions):
        """Calculate overall learner selection interpretability score."""
        return np.mean([
            selection_decisions['modality_learner_match_rate'],
            selection_decisions['pairing_consistency'],
            selection_decisions['learner_type_diversity'] / 5.0  # Normalize to 0-1
        ])

    def _calculate_performance_prediction_interpretability(self, bag_learner_summary, stage3_interpretability):
        """Calculate performance prediction interpretability metrics."""
        return {
            'average_expected_performance': bag_learner_summary.get('average_expected_performance', 0),
            'performance_range': bag_learner_summary.get('performance_range', (0, 0)),
            'learner_type_distribution': bag_learner_summary.get('learner_type_distribution', {}),
            'total_bags': bag_learner_summary.get('total_bags', 0)
        }

    def _calculate_performance_prediction_interpretability_score(self, prediction_interpretability):
        """Calculate overall performance prediction interpretability score."""
        performance_range = prediction_interpretability['performance_range']
        range_size = performance_range[1] - performance_range[0] if len(performance_range) == 2 else 0
        return np.mean([
            prediction_interpretability['average_expected_performance'],
            1.0 - min(range_size, 1.0)  # Lower range = higher interpretability
        ])

    def _calculate_bag_learner_pairing_interpretability(self, pairing_stats, metadata_completeness, ensemble_coherence):
        """Calculate bag-learner pairing interpretability metrics."""
        return {
            'modality_learner_match_rate': pairing_stats.get('modality_learner_match_rate', 0),
            'pairing_consistency': pairing_stats.get('pairing_consistency', 0),
            'metadata_completeness': metadata_completeness.get('average_completeness', 0),
            'ensemble_coherence': ensemble_coherence.get('overall_coherence', 0),
            'total_pairs': pairing_stats.get('total_pairs', 0)
        }

    def _calculate_bag_learner_pairing_interpretability_score(self, pairing_interpretability):
        """Calculate overall bag-learner pairing interpretability score."""
        return np.mean([
            pairing_interpretability['modality_learner_match_rate'],
            pairing_interpretability['pairing_consistency'],
            pairing_interpretability['metadata_completeness'],
            pairing_interpretability['ensemble_coherence']
        ])

    def _calculate_ensemble_coherence_interpretability(self, bag_learner_summary, pairing_stats, ensemble_coherence):
        """Calculate ensemble coherence interpretability metrics."""
        return {
            'total_bags': bag_learner_summary.get('total_bags', 0),
            'learner_type_diversity': ensemble_coherence.get('learner_type_diversity', 0),
            'ensemble_coherence': ensemble_coherence.get('overall_coherence', 0),
            'pairing_consistency': pairing_stats.get('pairing_consistency', 0),
            'average_expected_performance': bag_learner_summary.get('average_expected_performance', 0)
        }

    def _calculate_ensemble_coherence_interpretability_score(self, coherence_interpretability):
        """Calculate overall ensemble coherence interpretability score."""
        return np.mean([
            coherence_interpretability['learner_type_diversity'] / 5.0,  # Normalize to 0-1
            coherence_interpretability['ensemble_coherence'],
            coherence_interpretability['pairing_consistency'],
            coherence_interpretability['average_expected_performance']
        ])

    def _calculate_optimization_mode_interpretability(self, bag_learner_summary, ensemble_coherence, stage3_interpretability):
        """Calculate optimization mode interpretability metrics."""
        return {
            'learner_type_diversity': ensemble_coherence.get('learner_type_diversity', 0),
            'optimization_consistency': ensemble_coherence.get('optimization_mode_consistency', False),
            'performance_coherence': ensemble_coherence.get('performance_coherence', 0),
            'average_expected_performance': bag_learner_summary.get('average_expected_performance', 0),
            'total_bags': bag_learner_summary.get('total_bags', 0)
        }

    def _calculate_optimization_mode_interpretability_score(self, optimization_interpretability):
        """Calculate overall optimization mode interpretability score."""
        return np.mean([
            optimization_interpretability['learner_type_diversity'] / 5.0,  # Normalize to 0-1
            float(optimization_interpretability['optimization_consistency']),
            optimization_interpretability['performance_coherence'],
            optimization_interpretability['average_expected_performance']
        ])
    
    def get_pairing_statistics(self) -> Dict[str, Any]:
        """Get bag-learner pairing quality statistics."""
        if not self.bag_learner_configs:
            return {}
        
        # Calculate pairing quality metrics
        total_pairs = len(self.bag_learner_configs)
        modality_learner_matches = 0
        performance_predictions = []
        pairing_consistency = 0
        
        for config in self.bag_learner_configs:
            # Check modality-learner matching
            active_modalities = [k for k, v in config.modality_mask.items() if v]
            if len(active_modalities) == 1:
                modality = active_modalities[0]
                if (modality in ['text', 'clinical_text'] and config.learner_type == 'text_learner') or \
                   (modality in ['image', 'visual'] and config.learner_type == 'image_learner') or \
                   (modality not in ['text', 'clinical_text', 'image', 'visual'] and config.learner_type == 'tabular_learner'):
                    modality_learner_matches += 1
            elif len(active_modalities) == 2 and config.learner_type == 'simple_fusion_learner':
                modality_learner_matches += 1
            elif len(active_modalities) > 2 and config.learner_type == 'advanced_fusion_learner':
                modality_learner_matches += 1
            
            performance_predictions.append(config.expected_performance)
        
        # Calculate consistency (how similar are similar bags)
        learner_type_groups = {}
        for config in self.bag_learner_configs:
            if config.learner_type not in learner_type_groups:
                learner_type_groups[config.learner_type] = []
            learner_type_groups[config.learner_type].append(config.expected_performance)
        
        consistency_scores = []
        for learner_type, performances in learner_type_groups.items():
            if len(performances) > 1:
                consistency = 1.0 - np.std(performances)  # Lower std = higher consistency
                consistency_scores.append(max(0, consistency))
        
        self.pairing_statistics = {
            'total_pairs': total_pairs,
            'modality_learner_match_rate': modality_learner_matches / total_pairs if total_pairs > 0 else 0,
            'average_expected_performance': np.mean(performance_predictions),
            'performance_std': np.std(performance_predictions),
            'pairing_consistency': np.mean(consistency_scores) if consistency_scores else 0,
            'learner_type_distribution': {k: len(v) for k, v in learner_type_groups.items()},
            'pairing_focus': self.pairing_focus,
            'metadata_level': self.metadata_level
        }
        
        return self.pairing_statistics
    
    def get_metadata_completeness(self) -> Dict[str, Any]:
        """Get metadata completeness statistics."""
        if not self.bag_learner_configs:
            return {}
        
        # Check metadata completeness based on level
        total_configs = len(self.bag_learner_configs)
        completeness_scores = []
        
        for config in self.bag_learner_configs:
            score = 0
            total_fields = 0
            
            # Basic fields (always present)
            basic_fields = ['bag_id', 'learner_type', 'optimization_mode', 'task_type']
            for field in basic_fields:
                total_fields += 1
                if hasattr(config, field) and getattr(config, field) is not None:
                    score += 1
            
            # Complete fields (metadata_level >= 'complete')
            if self.metadata_level in ['complete', 'enhanced']:
                complete_fields = ['modality_weights', 'learner_config', 'expected_performance']
                for field in complete_fields:
                    total_fields += 1
                    if hasattr(config, field) and getattr(config, field) is not None:
                        score += 1
            
            # Enhanced fields (metadata_level == 'enhanced')
            if self.metadata_level == 'enhanced':
                enhanced_fields = ['modality_mask', 'feature_mask', 'bag_data']
                for field in enhanced_fields:
                    total_fields += 1
                    if hasattr(config, field) and getattr(config, field) is not None:
                        score += 1
            
            completeness_scores.append(score / total_fields if total_fields > 0 else 0)
        
        self.metadata_completeness = {
            'total_configs': total_configs,
            'average_completeness': np.mean(completeness_scores),
            'completeness_std': np.std(completeness_scores),
            'metadata_level': self.metadata_level,
            'min_completeness': min(completeness_scores),
            'max_completeness': max(completeness_scores),
            'complete_configs': sum(1 for score in completeness_scores if score >= 0.9)
        }
        
        return self.metadata_completeness
    
    def get_ensemble_coherence(self) -> Dict[str, Any]:
        """Get ensemble coherence statistics."""
        if not self.bag_learner_configs:
            return {}
        
        # Calculate coherence metrics
        learner_types = [config.learner_type for config in self.bag_learner_configs]
        optimization_modes = [config.optimization_mode for config in self.bag_learner_configs]
        expected_performances = [config.expected_performance for config in self.bag_learner_configs]
        
        # Learner type diversity
        unique_learner_types = len(set(learner_types))
        learner_type_entropy = -sum((learner_types.count(lt) / len(learner_types)) * 
                                   np.log2(learner_types.count(lt) / len(learner_types)) 
                                   for lt in set(learner_types))
        
        # Optimization mode consistency
        mode_consistency = len(set(optimization_modes)) == 1
        
        # Performance coherence (how well do similar bags perform similarly)
        performance_coherence = 1.0 - (np.std(expected_performances) / np.mean(expected_performances)) if np.mean(expected_performances) > 0 else 0
        
        # Bag-learner relationship coherence
        relationship_coherence = 0
        if self.bag_learner_pairing:
            # Check if similar modality patterns get similar learners
            modality_patterns = {}
            for config in self.bag_learner_configs:
                pattern = tuple(sorted([k for k, v in config.modality_mask.items() if v]))
                if pattern not in modality_patterns:
                    modality_patterns[pattern] = []
                modality_patterns[pattern].append(config.learner_type)
            
            pattern_consistency = []
            for pattern, learners in modality_patterns.items():
                if len(learners) > 1:
                    consistency = len(set(learners)) == 1  # All same learner type
                    pattern_consistency.append(consistency)
            
            relationship_coherence = np.mean(pattern_consistency) if pattern_consistency else 1.0
        
        self.ensemble_coherence = {
            'total_bags': len(self.bag_learner_configs),
            'learner_type_diversity': unique_learner_types,
            'learner_type_entropy': learner_type_entropy,
            'optimization_mode_consistency': mode_consistency,
            'performance_coherence': max(0, performance_coherence),
            'relationship_coherence': relationship_coherence,
            'overall_coherence': np.mean([
                learner_type_entropy / np.log2(unique_learner_types) if unique_learner_types > 1 else 1,
                float(mode_consistency),
                max(0, performance_coherence),
                relationship_coherence
            ]),
            'bag_learner_pairing_enabled': self.bag_learner_pairing
        }
        
        return self.ensemble_coherence
    
    def _create_learner_instance(self, bag_data: Dict[str, np.ndarray], 
                                learner_type: str, learner_config: Dict[str, Any], 
                                bag_labels: np.ndarray):
        """Create the actual learner instance based on the configuration."""
        import torch.nn as nn
        
        # Calculate input dimensions from bag data (after feature sampling)
        total_features = 0
        for key, data in bag_data.items():
            if key == 'labels':  # Skip labels
                continue
            if isinstance(data, np.ndarray) and len(data.shape) > 1:
                total_features += data.shape[1]
            elif isinstance(data, np.ndarray) and len(data.shape) == 1:
                total_features += 1
        
        
        total_features = max(total_features, 10)  # Minimum 10 features
        output_dim = max(len(np.unique(bag_labels)), 2) if self.task_type == "classification" else 1
        
        # Create real learners based on learner_type
        if learner_type == 'text_learner':
            return self._create_text_learner(total_features, output_dim, learner_config)
        elif learner_type == 'image_learner':
            return self._create_image_learner(total_features, output_dim, learner_config)
        elif learner_type == 'tabular_learner':
            return self._create_tabular_learner(total_features, output_dim, learner_config)
        elif learner_type == 'simple_fusion_learner':
            return self._create_simple_fusion_learner(total_features, output_dim, learner_config)
        elif learner_type == 'advanced_fusion_learner':
            return self._create_advanced_fusion_learner(total_features, output_dim, learner_config)
        elif learner_type == 'sklearn_logistic_regression':
            return self._create_sklearn_logistic_regression(total_features, output_dim, learner_config)
        elif learner_type == 'sklearn_random_forest':
            return self._create_sklearn_random_forest(total_features, output_dim, learner_config)
        elif learner_type == 'sklearn_gradient_boosting':
            return self._create_sklearn_gradient_boosting(total_features, output_dim, learner_config)
        elif learner_type == 'sklearn_svm':
            return self._create_sklearn_svm(total_features, output_dim, learner_config)
        
        # Deep Learning Learners
        elif learner_type == 'deep_text_lstm':
            return self._create_deep_text_lstm(total_features, output_dim, learner_config)
        elif learner_type == 'deep_metadata_mlp':
            return self._create_deep_metadata_mlp(total_features, output_dim, learner_config)
        elif learner_type == 'deep_image_cnn':
            return self._create_deep_image_cnn(total_features, output_dim, learner_config)
        elif learner_type == 'deep_text_dominant':
            return self._create_deep_text_dominant(total_features, output_dim, learner_config)
        elif learner_type == 'deep_metadata_dominant':
            return self._create_deep_metadata_dominant(total_features, output_dim, learner_config)
        elif learner_type == 'deep_simple_fusion':
            return self._create_deep_simple_fusion(total_features, output_dim, learner_config)
        elif learner_type == 'deep_multimodal_fusion':
            return self._create_deep_multimodal_fusion(total_features, output_dim, learner_config)
        
        # Transformer Learners
        elif learner_type == 'transformer_bert':
            return self._create_transformer_bert(total_features, output_dim, learner_config)
        elif learner_type == 'transformer_tabular':
            return self._create_transformer_tabular(total_features, output_dim, learner_config)
        elif learner_type == 'transformer_vision':
            return self._create_transformer_vision(total_features, output_dim, learner_config)
        elif learner_type == 'transformer_cross_modal':
            return self._create_transformer_cross_modal(total_features, output_dim, learner_config)
        elif learner_type == 'transformer_multimodal':
            return self._create_transformer_multimodal(total_features, output_dim, learner_config)
        
        else:
            # Default to tabular learner
            return self._create_tabular_learner(total_features, output_dim, learner_config)
    
    def _create_text_learner(self, input_dim, output_dim, config):
        """Create a text-specific neural network learner."""
        import torch.nn as nn
        
        class TextLearner(nn.Module):
            def __init__(self, input_dim, output_dim, config):
                super().__init__()
                self.hidden_dim = config.get('hidden_layers', (128, 64))[0]
                self.dropout = config.get('dropout', 0.2)
                self.output_dim = output_dim
                self.layers = None
                self.input_dim = input_dim
            
            def _build_layers(self, actual_input_dim):
                """Build layers with correct input dimension."""
                self.layers = nn.Sequential(
                    nn.Linear(actual_input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim // 2, self.output_dim)
                )
                self.input_dim = actual_input_dim
            
            def forward(self, x):
                if len(x.shape) > 2:
                    x = x.view(x.size(0), -1)
                
                # Build layers with correct input dimension if needed
                if self.layers is None or x.shape[1] != self.input_dim:
                    self._build_layers(x.shape[1])
                
                return self.layers(x)
        
        return TextLearner(input_dim, output_dim, config)
    
    def _create_image_learner(self, input_dim, output_dim, config):
        """Create an image-specific neural network learner."""
        import torch.nn as nn
        
        class ImageLearner(nn.Module):
            def __init__(self, input_dim, output_dim, config):
                super().__init__()
                self.hidden_dim = 256
                self.dropout = 0.3
                self.output_dim = output_dim
                self.layers = None
                self.input_dim = input_dim
            
            def _build_layers(self, actual_input_dim):
                """Build layers with correct input dimension."""
                self.layers = nn.Sequential(
                    nn.Linear(actual_input_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                    nn.BatchNorm1d(self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim // 2, self.output_dim)
                )
                self.input_dim = actual_input_dim
            
            def forward(self, x):
                if len(x.shape) > 2:
                    x = x.view(x.size(0), -1)
                
                # Build layers with correct input dimension if needed
                if self.layers is None or x.shape[1] != self.input_dim:
                    self._build_layers(x.shape[1])
                
                return self.layers(x)
        
        return ImageLearner(input_dim, output_dim, config)
    
    def _create_tabular_learner(self, input_dim, output_dim, config):
        """Create a tabular-specific neural network learner."""
        import torch.nn as nn
        
        class TabularLearner(nn.Module):
            def __init__(self, input_dim, output_dim, config):
                super().__init__()
                hidden_dim = 128
                dropout = 0.2
                
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, output_dim)
                )
            
            def forward(self, x):
                if len(x.shape) > 2:
                    x = x.view(x.size(0), -1)
                return self.layers(x)
        
        return TabularLearner(input_dim, output_dim, config)
    
    def _create_simple_fusion_learner(self, input_dim, output_dim, config):
        """Create a simple fusion learner for 2 modalities."""
        import torch.nn as nn
        
        class SimpleFusionLearner(nn.Module):
            def __init__(self, input_dim, output_dim, config):
                super().__init__()
                self.hidden_dim = 256
                self.dropout = 0.2
                self.output_dim = output_dim
                self.layers = None
                self.input_dim = input_dim
            
            def _build_layers(self, actual_input_dim):
                """Build layers with correct input dimension."""
                self.layers = nn.Sequential(
                    nn.Linear(actual_input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim // 2, self.output_dim)
                )
                self.input_dim = actual_input_dim
            
            def forward(self, x):
                if len(x.shape) > 2:
                    x = x.view(x.size(0), -1)
                
                # Build layers with correct input dimension if needed
                if self.layers is None or x.shape[1] != self.input_dim:
                    self._build_layers(x.shape[1])
                
                return self.layers(x)
        
        return SimpleFusionLearner(input_dim, output_dim, config)
    
    def _create_advanced_fusion_learner(self, input_dim, output_dim, config):
        """Create an advanced fusion learner for multiple modalities."""
        import torch.nn as nn
        
        class AdvancedFusionLearner(nn.Module):
            def __init__(self, input_dim, output_dim, config):
                super().__init__()
                self.hidden_dim = 512
                self.dropout = 0.3
                self.output_dim = output_dim
                self.layers = None
                self.input_dim = input_dim
            
            def _build_layers(self, actual_input_dim):
                """Build layers with correct input dimension."""
                self.layers = nn.Sequential(
                    nn.Linear(actual_input_dim, self.hidden_dim),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                    nn.BatchNorm1d(self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim // 4, self.output_dim)
                )
                self.input_dim = actual_input_dim
            
            def forward(self, x):
                if len(x.shape) > 2:
                    x = x.view(x.size(0), -1)
                
                # Build layers with correct input dimension if needed
                if self.layers is None or x.shape[1] != self.input_dim:
                    self._build_layers(x.shape[1])
                
                return self.layers(x)
        
        return AdvancedFusionLearner(input_dim, output_dim, config)
    
    def _create_sklearn_logistic_regression(self, input_dim, output_dim, config):
        """Create a sklearn Logistic Regression learner wrapper."""
        from sklearn.linear_model import LogisticRegression
        
        class SklearnLogisticRegressionWrapper:
            def __init__(self, input_dim, output_dim, config):
                self.model = LogisticRegression(
                    random_state=config.get('random_state', 42),
                    max_iter=1000,
                    multi_class='ovr' if output_dim > 2 else 'auto'
                )
                self.is_fitted = False
            
            def fit(self, X, y):
                self.model.fit(X, y)
                self.is_fitted = True
            
            def predict(self, X):
                return self.model.predict(X)
            
            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                return self.model.predict_proba(X)
        
        return SklearnLogisticRegressionWrapper(input_dim, output_dim, config)
    
    def _create_sklearn_random_forest(self, input_dim, output_dim, config):
        """Create a sklearn Random Forest learner wrapper."""
        from sklearn.ensemble import RandomForestClassifier
        
        class SklearnRandomForestWrapper:
            def __init__(self, input_dim, output_dim, config):
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=config.get('random_state', 42),
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2
                )
                self.is_fitted = False
            
            def fit(self, X, y):
                self.model.fit(X, y)
                self.is_fitted = True
            
            def predict(self, X):
                return self.model.predict(X)
            
            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                return self.model.predict_proba(X)
        
        return SklearnRandomForestWrapper(input_dim, output_dim, config)
    
    def _create_sklearn_gradient_boosting(self, input_dim, output_dim, config):
        """Create a sklearn Gradient Boosting learner wrapper."""
        from sklearn.ensemble import GradientBoostingClassifier
        
        class SklearnGradientBoostingWrapper:
            def __init__(self, input_dim, output_dim, config):
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=config.get('random_state', 42)
                )
                self.is_fitted = False
            
            def fit(self, X, y):
                self.model.fit(X, y)
                self.is_fitted = True
            
            def predict(self, X):
                return self.model.predict(X)
            
            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                return self.model.predict_proba(X)
        
        return SklearnGradientBoostingWrapper(input_dim, output_dim, config)
    
    def _create_sklearn_svm(self, input_dim, output_dim, config):
        """Create a sklearn SVM learner wrapper."""
        from sklearn.svm import SVC
        
        class SklearnSVMWrapper:
            def __init__(self, input_dim, output_dim, config):
                self.model = SVC(
                    random_state=config.get('random_state', 42),
                    probability=True,
                    kernel='rbf',
                    C=1.0,
                    gamma='scale'
                )
                self.is_fitted = False
            
            def fit(self, X, y):
                self.model.fit(X, y)
                self.is_fitted = True
            
            def predict(self, X):
                return self.model.predict(X)
            
            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                return self.model.predict_proba(X)
        
        return SklearnSVMWrapper(input_dim, output_dim, config)
    
    # Deep Learning Learners
    def _create_deep_text_lstm(self, input_dim, output_dim, config):
        """Create a deep learning LSTM for text features."""
        import torch
        import torch.nn as nn
        
        class DeepTextLSTM(nn.Module):
            def __init__(self, input_dim, output_dim, config):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.hidden_size = config.get('hidden_size', 128)
                self.num_layers = config.get('num_layers', 2)
                self.dropout = config.get('dropout', 0.2)
                
                # Simplified architecture for better performance
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_dim)
                )
                self.is_fitted = False
                
            def forward(self, x):
                return self.network(x)
            
            def fit(self, X, y):
                """Train the deep learning model."""
                train_pytorch_model(self, X, y, epochs=10, batch_size=32, lr=0.001)
                self.is_fitted = True
                
            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                with torch.no_grad():
                    # Convert numpy array to tensor if needed
                    if isinstance(X, np.ndarray):
                        X = torch.FloatTensor(X)
                    logits = self.forward(X)
                    return torch.softmax(logits, dim=1).numpy()
        
        return DeepTextLSTM(input_dim, output_dim, config)
    
    def _create_deep_metadata_mlp(self, input_dim, output_dim, config):
        """Create a deep learning MLP for metadata features."""
        import torch
        import torch.nn as nn
        
        class DeepMetadataMLP(nn.Module):
            def __init__(self, input_dim, output_dim, config):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                hidden_sizes = config.get('hidden_sizes', [256, 128, 64])
                dropout = config.get('dropout', 0.3)
                
                layers = []
                prev_size = input_dim
                for hidden_size in hidden_sizes:
                    layers.extend([
                        nn.Linear(prev_size, hidden_size),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                    prev_size = hidden_size
                
                layers.append(nn.Linear(prev_size, output_dim))
                self.network = nn.Sequential(*layers)
                self.is_fitted = False
                
            def forward(self, x):
                return self.network(x)
            
            def fit(self, X, y):
                """Train the deep learning model."""
                train_pytorch_model(self, X, y, epochs=10, batch_size=32, lr=0.001)
                self.is_fitted = True
                
            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                with torch.no_grad():
                    # Convert numpy array to tensor if needed
                    if isinstance(X, np.ndarray):
                        X = torch.FloatTensor(X)
                    logits = self.forward(X)
                    return torch.softmax(logits, dim=1).numpy()
        
        return DeepMetadataMLP(input_dim, output_dim, config)
    
    def _create_deep_image_cnn(self, input_dim, output_dim, config):
        """Create a deep learning CNN for image features."""
        import torch
        import torch.nn as nn
        
        class DeepImageCNN(nn.Module):
            def __init__(self, input_dim, output_dim, config):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                
                # Simple CNN for image features
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(0.3)
                
                # Calculate flattened size
                self.flattened_size = 64 * 8 * 8  # Adjust based on input size
                self.fc1 = nn.Linear(self.flattened_size, 128)
                self.fc2 = nn.Linear(128, output_dim)
                self.is_fitted = False
                
            def forward(self, x):
                # Reshape if needed
                if x.dim() == 2:
                    x = x.view(x.size(0), 3, 32, 32)  # Reshape to image format
                
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.dropout(torch.relu(self.fc1(x)))
                return self.fc2(x)
            
            def fit(self, X, y):
                """Train the deep learning model."""
                train_pytorch_model(self, X, y, epochs=10, batch_size=32, lr=0.001)
                self.is_fitted = True
                
            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                with torch.no_grad():
                    # Convert numpy array to tensor if needed
                    if isinstance(X, np.ndarray):
                        X = torch.FloatTensor(X)
                    logits = self.forward(X)
                    return torch.softmax(logits, dim=1).numpy()
        
        return DeepImageCNN(input_dim, output_dim, config)
    
    def _create_deep_text_dominant(self, input_dim, output_dim, config):
        """Create a deep learning model with text-dominant architecture."""
        import torch
        import torch.nn as nn
        
        class DeepTextDominant(nn.Module):
            def __init__(self, input_dim, output_dim, config):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                
                # Use all features as a single input for simplicity
                self.fusion_network = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_dim)
                )
                self.is_fitted = False
                
            def forward(self, x):
                return self.fusion_network(x)
            
            def fit(self, X, y):
                """Train the deep learning model."""
                train_pytorch_model(self, X, y, epochs=10, batch_size=32, lr=0.001)
                self.is_fitted = True
                
            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                with torch.no_grad():
                    # Convert numpy array to tensor if needed
                    if isinstance(X, np.ndarray):
                        X = torch.FloatTensor(X)
                    logits = self.forward(X)
                    return torch.softmax(logits, dim=1).numpy()
        
        return DeepTextDominant(input_dim, output_dim, config)
    
    def _create_deep_metadata_dominant(self, input_dim, output_dim, config):
        """Create a deep learning model with metadata-dominant architecture."""
        import torch
        import torch.nn as nn
        
        class DeepMetadataDominant(nn.Module):
            def __init__(self, input_dim, output_dim, config):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                
                # Use all features as a single input for simplicity
                self.fusion_network = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_dim)
                )
                self.is_fitted = False
                
            def forward(self, x):
                return self.fusion_network(x)
            
            def fit(self, X, y):
                """Train the deep learning model."""
                train_pytorch_model(self, X, y, epochs=10, batch_size=32, lr=0.001)
                self.is_fitted = True
                
            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                with torch.no_grad():
                    # Convert numpy array to tensor if needed
                    if isinstance(X, np.ndarray):
                        X = torch.FloatTensor(X)
                    logits = self.forward(X)
                    return torch.softmax(logits, dim=1).numpy()
        
        return DeepMetadataDominant(input_dim, output_dim, config)
    
    def _create_deep_simple_fusion(self, input_dim, output_dim, config):
        """Create a simple deep learning fusion model."""
        import torch
        import torch.nn as nn
        
        class DeepSimpleFusion(nn.Module):
            def __init__(self, input_dim, output_dim, config):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                
                self.fusion_network = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_dim)
                )
                self.is_fitted = False
                
            def forward(self, x):
                return self.fusion_network(x)
            
            def fit(self, X, y):
                """Train the deep learning model."""
                train_pytorch_model(self, X, y, epochs=10, batch_size=32, lr=0.001)
                self.is_fitted = True
                
            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                with torch.no_grad():
                    # Convert numpy array to tensor if needed
                    if isinstance(X, np.ndarray):
                        X = torch.FloatTensor(X)
                    logits = self.forward(X)
                    return torch.softmax(logits, dim=1).numpy()
        
        return DeepSimpleFusion(input_dim, output_dim, config)
    
    def _create_deep_multimodal_fusion(self, input_dim, output_dim, config):
        """Create an advanced deep learning multimodal fusion model."""
        import torch
        import torch.nn as nn
        
        class DeepMultimodalFusion(nn.Module):
            def __init__(self, input_dim, output_dim, config):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                
                # Use all features as a single input for simplicity
                self.fusion_network = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_dim)
                )
                self.is_fitted = False
                
            def forward(self, x):
                return self.fusion_network(x)
            
            def fit(self, X, y):
                """Train the deep learning model."""
                train_pytorch_model(self, X, y, epochs=10, batch_size=32, lr=0.001)
                self.is_fitted = True
                
            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                with torch.no_grad():
                    # Convert numpy array to tensor if needed
                    if isinstance(X, np.ndarray):
                        X = torch.FloatTensor(X)
                    logits = self.forward(X)
                    return torch.softmax(logits, dim=1).numpy()
        
        return DeepMultimodalFusion(input_dim, output_dim, config)
    
    # Transformer Learners
    def _create_transformer_bert(self, input_dim, output_dim, config):
        """Create a BERT-based transformer learner."""
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer
        
        class TransformerBERT(nn.Module):
            def __init__(self, input_dim, output_dim, config):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                
                # Simplified transformer architecture to avoid memory issues
                self.d_model = config.get('d_model', 64)  # Smaller model
                self.nhead = config.get('nhead', 4)  # Fewer heads
                self.num_layers = config.get('num_layers', 1)  # Fewer layers
                
                # Input projection
                self.input_projection = nn.Linear(input_dim, self.d_model)
                
                # Lightweight transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=128,  # Smaller feedforward
                    dropout=0.1,
                    batch_first=True  # Better performance
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
                
                # Classification head
                self.classifier = nn.Linear(self.d_model, output_dim)
                self.is_fitted = False
                
            def forward(self, x):
                # Project input to d_model
                x = self.input_projection(x)
                
                # Add sequence dimension for transformer
                x = x.unsqueeze(1)  # (batch_size, 1, d_model)
                
                # Transformer encoding
                encoded = self.transformer(x)
                
                # Global average pooling
                pooled = encoded.mean(dim=1)
                
                return self.classifier(pooled)
            
            def fit(self, X, y):
                """Train the deep learning model."""
                train_pytorch_model(self, X, y, epochs=10, batch_size=32, lr=0.001)
                self.is_fitted = True
                
            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                with torch.no_grad():
                    # Convert numpy array to tensor if needed
                    if isinstance(X, np.ndarray):
                        X = torch.FloatTensor(X)
                    logits = self.forward(X)
                    return torch.softmax(logits, dim=1).numpy()
        
        return TransformerBERT(input_dim, output_dim, config)
    
    def _create_transformer_tabular(self, input_dim, output_dim, config):
        """Create a transformer for tabular data."""
        import torch
        import torch.nn as nn
        
        class TransformerTabular(nn.Module):
            def __init__(self, input_dim, output_dim, config):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.d_model = config.get('d_model', 64)  # Smaller model
                self.nhead = config.get('nhead', 4)  # Fewer heads
                self.num_layers = config.get('num_layers', 1)  # Fewer layers
                
                # Input projection
                self.input_projection = nn.Linear(input_dim, self.d_model)
                
                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=128,  # Smaller feedforward
                    dropout=0.1,
                    batch_first=True  # Better performance
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
                
                # Classification head
                self.classifier = nn.Linear(self.d_model, output_dim)
                self.is_fitted = False
                
            def forward(self, x):
                # Project input to d_model
                x = self.input_projection(x)
                
                # Add sequence dimension for transformer
                x = x.unsqueeze(1)  # (batch_size, 1, d_model)
                
                # Transformer encoding
                encoded = self.transformer(x)
                
                # Global average pooling
                pooled = encoded.mean(dim=1)
                
                return self.classifier(pooled)
            
            def fit(self, X, y):
                """Train the deep learning model."""
                train_pytorch_model(self, X, y, epochs=10, batch_size=32, lr=0.001)
                self.is_fitted = True
                
            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                with torch.no_grad():
                    # Convert numpy array to tensor if needed
                    if isinstance(X, np.ndarray):
                        X = torch.FloatTensor(X)
                    logits = self.forward(X)
                    return torch.softmax(logits, dim=1).numpy()
        
        return TransformerTabular(input_dim, output_dim, config)
    
    def _create_transformer_vision(self, input_dim, output_dim, config):
        """Create a vision transformer for image data."""
        import torch
        import torch.nn as nn
        
        class TransformerVision(nn.Module):
            def __init__(self, input_dim, output_dim, config):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.d_model = config.get('d_model', 64)  # Smaller model
                self.nhead = config.get('nhead', 4)  # Fewer heads
                self.num_layers = config.get('num_layers', 1)  # Fewer layers
                
                # Patch embedding (simplified)
                self.patch_embedding = nn.Linear(input_dim, self.d_model)
                self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.d_model))
                
                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=128,  # Smaller feedforward
                    dropout=0.1,
                    batch_first=True  # Better performance
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
                
                # Classification head
                self.classifier = nn.Linear(self.d_model, output_dim)
                self.is_fitted = False
                
            def forward(self, x):
                # Patch embedding
                x = self.patch_embedding(x)
                
                # Add positional encoding
                x = x + self.pos_embedding
                
                # Add sequence dimension
                x = x.unsqueeze(1)  # (batch_size, 1, d_model)
                
                # Transformer encoding
                encoded = self.transformer(x)
                
                # Global average pooling
                pooled = encoded.mean(dim=1)
                
                return self.classifier(pooled)
            
            def fit(self, X, y):
                """Train the deep learning model."""
                train_pytorch_model(self, X, y, epochs=10, batch_size=32, lr=0.001)
                self.is_fitted = True
                
            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                with torch.no_grad():
                    # Convert numpy array to tensor if needed
                    if isinstance(X, np.ndarray):
                        X = torch.FloatTensor(X)
                    logits = self.forward(X)
                    return torch.softmax(logits, dim=1).numpy()
        
        return TransformerVision(input_dim, output_dim, config)
    
    def _create_transformer_cross_modal(self, input_dim, output_dim, config):
        """Create a cross-modal transformer."""
        import torch
        import torch.nn as nn
        
        class TransformerCrossModal(nn.Module):
            def __init__(self, input_dim, output_dim, config):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.d_model = config.get('d_model', 64)  # Smaller model
                self.nhead = config.get('nhead', 4)  # Fewer heads
                self.num_layers = config.get('num_layers', 1)  # Fewer layers
                
                # Input projection
                self.input_projection = nn.Linear(input_dim, self.d_model)
                
                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=128,  # Smaller feedforward
                    dropout=0.1,
                    batch_first=True  # Better performance
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
                
                # Classification head
                self.classifier = nn.Linear(self.d_model, output_dim)
                self.is_fitted = False
                
            def forward(self, x):
                # Project input to d_model
                x = self.input_projection(x)
                
                # Add sequence dimension for transformer
                x = x.unsqueeze(1)  # (batch_size, 1, d_model)
                
                # Transformer encoding
                encoded = self.transformer(x)
                
                # Global average pooling
                pooled = encoded.mean(dim=1)
                
                return self.classifier(pooled)
            
            def fit(self, X, y):
                """Train the deep learning model."""
                train_pytorch_model(self, X, y, epochs=10, batch_size=32, lr=0.001)
                self.is_fitted = True
                
            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                with torch.no_grad():
                    # Convert numpy array to tensor if needed
                    if isinstance(X, np.ndarray):
                        X = torch.FloatTensor(X)
                    logits = self.forward(X)
                    return torch.softmax(logits, dim=1).numpy()
        
        return TransformerCrossModal(input_dim, output_dim, config)
    
    def _create_transformer_multimodal(self, input_dim, output_dim, config):
        """Create a multimodal transformer."""
        import torch
        import torch.nn as nn
        
        class TransformerMultimodal(nn.Module):
            def __init__(self, input_dim, output_dim, config):
                super().__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.d_model = config.get('d_model', 128)
                self.nhead = config.get('nhead', 8)
                self.num_layers = config.get('num_layers', 1)  # Fewer layers
                
                # Input projection
                self.input_projection = nn.Linear(input_dim, self.d_model)
                
                # Multimodal transformer
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=128,  # Smaller feedforward
                    dropout=0.1,
                    batch_first=True  # Better performance
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
                
                # Classification head
                self.classifier = nn.Linear(self.d_model, output_dim)
                self.is_fitted = False
                
            def forward(self, x):
                # Project input to d_model
                x = self.input_projection(x)
                
                # Add sequence dimension for transformer
                x = x.unsqueeze(1)  # (batch_size, 1, d_model)
                
                # Transformer encoding
                encoded = self.transformer(x)
                
                # Global average pooling
                pooled = encoded.mean(dim=1)
                
                return self.classifier(pooled)
            
            def fit(self, X, y):
                """Train the deep learning model."""
                train_pytorch_model(self, X, y, epochs=10, batch_size=32, lr=0.001)
                self.is_fitted = True
                
            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before prediction")
                with torch.no_grad():
                    # Convert numpy array to tensor if needed
                    if isinstance(X, np.ndarray):
                        X = torch.FloatTensor(X)
                    logits = self.forward(X)
                    return torch.softmax(logits, dim=1).numpy()
        
        return TransformerMultimodal(input_dim, output_dim, config)
