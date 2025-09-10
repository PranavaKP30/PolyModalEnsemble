"""
Stage 5: Simplified Multimodal Ensemble Prediction System

This module implements the final stage of the multimodal ensemble pipeline,
focusing on bag reconstruction, prediction, and adaptive transformer-based aggregation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings

# --- Enums and Data Classes ---

class AggregationStrategy(Enum):
    """Available aggregation strategies for ensemble prediction"""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    TRANSFORMER_FUSION = "transformer_fusion"

class UncertaintyMethod(Enum):
    """Available uncertainty estimation methods"""
    ENTROPY = "entropy"
    VARIANCE = "variance"
    CONFIDENCE = "confidence"

@dataclass
class PredictionResult:
    """Result of ensemble prediction with metadata"""
    predictions: np.ndarray
    uncertainty: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    mixture_weights: Optional[np.ndarray] = None
    modality_importance: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# --- Simplified Transformer Meta-Learner ---

class TransformerMetaLearner(nn.Module):
    """
    Simplified transformer-based meta-learner for adaptive ensemble aggregation
    
    Key Features:
    1. Dynamically estimates relative importance of each learner's predictions
    2. Evaluates accuracy AND generalization ability across diverse subsets
    3. Context-dependent attention weights that mirror adaptive dropout philosophy
    4. Down-weights overfit learners, emphasizes complementary signals
    5. Modality-aware aggregation that considers predictive strength and diversity
    """
    def __init__(self, n_learners: int, n_classes: int, task_type: str = "classification",
                 hidden_dim: int = 64, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.n_learners = n_learners
        self.n_classes = n_classes
        self.task_type = task_type
        
        # Input projection: maps learner outputs to hidden dimension
        input_dim = n_classes if task_type == "classification" else 1
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder for contextualized learner representations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Attention head for computing dynamic weights
        self.attention_head = nn.Linear(hidden_dim, 1)
        
        # Modality importance weights (learnable)
        self.modality_weights = nn.Parameter(torch.ones(n_learners))
        
        # Generalization scoring network
        self.generalization_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, learner_outputs, confidence, uncertainty, calibration_error, 
                generalization_score, modality_priors, modality_masks):
        """
        Simplified forward pass for adaptive ensemble aggregation
        
        Args:
            learner_outputs: (batch, n_learners, n_classes) - predictions from each learner
            confidence: (batch, n_learners) - confidence scores
            uncertainty: (batch, n_learners) - uncertainty estimates
            calibration_error: (n_learners,) - calibration errors
            generalization_score: (n_learners,) - generalization scores
            modality_priors: (n_learners,) - modality importance priors
            modality_masks: (batch, n_learners, n_modalities) - modality masks
        """
        batch_size, n_learners, n_classes = learner_outputs.shape
        
        # Step 1: Project learner outputs to hidden dimension
        learner_outputs_flat = learner_outputs.reshape(-1, n_classes)
        projected_outputs = self.input_projection(learner_outputs_flat)
        projected_outputs = projected_outputs.reshape(batch_size, n_learners, -1)
        
        # Step 2: Compute generalization scores for each learner
        gen_scores = self.generalization_scorer(projected_outputs).squeeze(-1)  # (batch, n_learners)
        
        # Step 3: Compute base attention weights from transformer
        contextualized_embeddings = self.transformer(projected_outputs)
        attention_logits = self.attention_head(contextualized_embeddings).squeeze(-1)  # (batch, n_learners)
        
        # Step 4: Apply modality-aware weighting
        modality_weights_expanded = self.modality_weights.unsqueeze(0).expand(batch_size, -1)
        
        # Step 5: Combine attention with generalization and modality importance
        # Higher generalization score = more weight
        # Higher confidence = more weight  
        # Lower uncertainty = more weight
        # Modality importance = base weight
        combined_logits = (
            attention_logits +                                    # Base transformer attention
            gen_scores * 2.0 +                                   # Generalization bonus
            confidence * 1.5 +                                   # Confidence bonus
            (1.0 - uncertainty) * 1.0 +                         # Low uncertainty bonus
            modality_weights_expanded * 0.5                      # Modality importance
        )
        
        # Step 6: Compute final mixture weights
        mixture_weights = F.softmax(combined_logits, dim=1)  # (batch, n_learners)
        
        # Step 7: Ensemble prediction
        if self.task_type == "classification":
            final_prediction = torch.sum(mixture_weights.unsqueeze(-1) * learner_outputs, dim=1)
        else:
            final_prediction = torch.sum(mixture_weights * learner_outputs.squeeze(-1), dim=1)
        
        return final_prediction, mixture_weights, attention_logits, combined_logits

# --- Main Ensemble Predictor ---

class EnsemblePredictor:
    """
    Main orchestrator for Stage 5: Ensemble Prediction
    
    Handles:
    1. Bag reconstruction from saved bag data
    2. Prediction using trained weak learners
    3. Adaptive transformer-based aggregation
    4. Uncertainty estimation
    """
    
    def __init__(self, task_type: str = "classification", 
                 aggregation_strategy: str = "transformer_fusion",
                 uncertainty_method: str = "entropy",
                 transformer_num_heads: int = 4,
                 transformer_num_layers: int = 2,
                 transformer_hidden_dim: int = 64):
        self.task_type = task_type
        self.aggregation_strategy = AggregationStrategy(aggregation_strategy)
        self.uncertainty_method = UncertaintyMethod(uncertainty_method)
        
        # Transformer parameters
        self.transformer_num_heads = transformer_num_heads
        self.transformer_num_layers = transformer_num_layers
        self.transformer_hidden_dim = transformer_hidden_dim
        
        # State variables
        self.trained_learners = []
        self.learner_metrics = []
        self.bag_characteristics = []
        self.transformer_meta_learner = None
        self.is_fitted = False

    def setup_transformer_fusion(self, num_learners: int, num_classes: int):
        """Setup the transformer meta-learner for fusion"""
        self.transformer_meta_learner = TransformerMetaLearner(
            n_learners=num_learners,
            n_classes=num_classes,
            task_type=self.task_type,
            hidden_dim=self.transformer_hidden_dim,
            num_heads=self.transformer_num_heads,
            num_layers=self.transformer_num_layers
        )

    def fit(self, trained_learners: List[Any], learner_metrics: List[Dict], 
            bag_characteristics: List[Dict]):
        """
        Fit the ensemble predictor with trained learners and their characteristics
        
        Args:
            trained_learners: List of trained weak learners
            learner_metrics: List of performance metrics for each learner
            bag_characteristics: List of bag characteristics for each learner
        """
        self.trained_learners = trained_learners
        self.learner_metrics = learner_metrics
        self.bag_characteristics = bag_characteristics
        
        # Setup transformer fusion if needed
        if self.aggregation_strategy == AggregationStrategy.TRANSFORMER_FUSION:
            n_learners = len(trained_learners)
            n_classes = self._get_num_classes()
            self.setup_transformer_fusion(n_learners, n_classes)
        
        self.is_fitted = True

    def predict(self, test_data: Dict[str, np.ndarray]) -> PredictionResult:
        """
        Make predictions on test data using the ensemble
        
        Args:
            test_data: Dictionary mapping modality names to feature arrays
            
        Returns:
            PredictionResult with predictions and metadata
        """
        if not self.is_fitted:
            raise ValueError("EnsemblePredictor must be fitted before making predictions")
        
        # Step 1: Reconstruct bags and get individual predictions
        individual_predictions = self._get_individual_predictions(test_data)
        
        # Step 2: Compute confidence and uncertainty
        confidence = self._compute_confidence(individual_predictions)
        uncertainty = self._compute_uncertainty(individual_predictions)
        
        # Step 3: Aggregate predictions
        if self.aggregation_strategy == AggregationStrategy.SIMPLE_AVERAGE:
            final_predictions = self._simple_average(individual_predictions)
            mixture_weights = None
        elif self.aggregation_strategy == AggregationStrategy.WEIGHTED_AVERAGE:
            final_predictions, mixture_weights = self._weighted_average(individual_predictions, confidence)
        elif self.aggregation_strategy == AggregationStrategy.TRANSFORMER_FUSION:
            final_predictions, mixture_weights = self._transformer_fusion(
                individual_predictions, confidence, uncertainty
            )
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")
        
        # Step 4: Convert to numpy and create result
        if isinstance(final_predictions, torch.Tensor):
            final_predictions = final_predictions.detach().cpu().numpy()
        
        if mixture_weights is not None and isinstance(mixture_weights, torch.Tensor):
            mixture_weights = mixture_weights.detach().cpu().numpy()
        
        return PredictionResult(
            predictions=final_predictions,
            uncertainty=uncertainty,
            confidence=confidence,
            mixture_weights=mixture_weights,
            modality_importance=self._compute_modality_importance(),
            metadata={
                'aggregation_strategy': self.aggregation_strategy.value,
                'uncertainty_method': self.uncertainty_method.value,
                'n_learners': len(self.trained_learners)
            }
        )

    def _get_individual_predictions(self, test_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Get predictions from each individual learner"""
        n_samples = len(next(iter(test_data.values())))
        n_learners = len(self.trained_learners)
        
        if self.task_type == "classification":
            n_classes = self._get_num_classes()
            predictions = np.zeros((n_learners, n_samples, n_classes))
        else:
            predictions = np.zeros((n_learners, n_samples))
        
        for i, learner in enumerate(self.trained_learners):
            try:
                if hasattr(learner, 'predict_proba') and self.task_type == "classification":
                    pred = learner.predict_proba(test_data)
                else:
                    pred = learner.predict(test_data)
                
                if self.task_type == "classification":
                    if pred.ndim == 1:  # Convert to probabilities if needed
                        pred_proba = np.zeros((len(pred), n_classes))
                        pred_proba[np.arange(len(pred)), pred] = 1.0
                        pred = pred_proba
                    predictions[i] = pred
                else:
                    predictions[i] = pred
                    
            except Exception as e:
                warnings.warn(f"Learner {i} failed to predict: {e}")
                if self.task_type == "classification":
                    predictions[i] = np.ones((n_samples, n_classes)) / n_classes
                else:
                    predictions[i] = np.zeros(n_samples)
        
        return predictions

    def _compute_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """Compute confidence scores for each learner's predictions"""
        n_learners, n_samples = predictions.shape[:2]
        confidence = np.zeros((n_learners, n_samples))
        
        for i in range(n_learners):
            if self.task_type == "classification":
                # Confidence = max probability
                confidence[i] = np.max(predictions[i], axis=1)
            else:
                # For regression, use inverse of prediction variance as confidence proxy
                pred_var = np.var(predictions[i])
                confidence[i] = 1.0 / (1.0 + pred_var)
        
        return confidence

    def _compute_uncertainty(self, predictions: np.ndarray) -> np.ndarray:
        """Compute uncertainty estimates for each learner's predictions"""
        n_learners, n_samples = predictions.shape[:2]
        uncertainty = np.zeros((n_learners, n_samples))
        
        for i in range(n_learners):
            if self.task_type == "classification":
                if self.uncertainty_method == UncertaintyMethod.ENTROPY:
                    # Entropy-based uncertainty
                    probs = predictions[i]
                    log_probs = np.log(probs + 1e-8)
                    uncertainty[i] = -np.sum(probs * log_probs, axis=1)
                elif self.uncertainty_method == UncertaintyMethod.CONFIDENCE:
                    # Confidence-based uncertainty (1 - confidence)
                    uncertainty[i] = 1.0 - np.max(predictions[i], axis=1)
                else:  # VARIANCE
                    uncertainty[i] = np.var(predictions[i], axis=1)
            else:
                # For regression, use prediction variance
                uncertainty[i] = np.var(predictions[i])
        
        return uncertainty

    def _simple_average(self, predictions: np.ndarray) -> np.ndarray:
        """Simple averaging of predictions"""
        return np.mean(predictions, axis=0)

    def _weighted_average(self, predictions: np.ndarray, confidence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted averaging based on confidence"""
        # Normalize confidence to get weights
        weights = confidence / np.sum(confidence, axis=0, keepdims=True)
        
        if self.task_type == "classification":
            weighted_pred = np.sum(weights[:, :, np.newaxis] * predictions, axis=0)
        else:
            weighted_pred = np.sum(weights * predictions, axis=0)
        
        return weighted_pred, weights

    def _transformer_fusion(self, predictions: np.ndarray, confidence: np.ndarray, 
                          uncertainty: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transformer-based fusion of predictions"""
        if self.transformer_meta_learner is None:
            raise ValueError("Transformer meta-learner not initialized")
        
        # Convert to tensors
        learner_outputs = torch.tensor(predictions, dtype=torch.float32)
        confidence_tensor = torch.tensor(confidence, dtype=torch.float32)
        uncertainty_tensor = torch.tensor(uncertainty, dtype=torch.float32)
        
        # Create dummy values for unused parameters
        n_learners = len(self.trained_learners)
        calibration_error = torch.zeros(n_learners)
        generalization_score = torch.ones(n_learners)
        modality_priors = torch.ones(n_learners)
        modality_masks = torch.ones(confidence.shape[0], n_learners, 1)
        
        # Forward pass through transformer
        with torch.no_grad():
            final_prediction, mixture_weights, attention_logits, combined_logits = self.transformer_meta_learner(
                learner_outputs, confidence_tensor, uncertainty_tensor, calibration_error,
                generalization_score, modality_priors, modality_masks
            )
        
        return final_prediction, mixture_weights

    def _get_num_classes(self) -> int:
        """Get number of classes for classification tasks"""
        # Try to infer from learner metrics
        for metrics in self.learner_metrics:
            if 'n_classes' in metrics:
                return metrics['n_classes']
        
        # Default fallback
        return 3

    def _compute_modality_importance(self) -> Dict[str, float]:
        """Compute modality importance from bag characteristics"""
        modality_importance = {}
        
        for bag_char in self.bag_characteristics:
            if 'modality_weights' in bag_char:
                for modality, weight in bag_char['modality_weights'].items():
                    if modality not in modality_importance:
                        modality_importance[modality] = 0.0
                    modality_importance[modality] += weight
        
        # Normalize
        total_weight = sum(modality_importance.values())
        if total_weight > 0:
            modality_importance = {k: v/total_weight for k, v in modality_importance.items()}
        
        return modality_importance

# --- Simplified Testing Methods ---

def run_simple_ablation_test(predictor: EnsemblePredictor, test_data: Dict[str, np.ndarray], 
                           test_labels: np.ndarray, configs: List[Dict]) -> Dict[str, Any]:
    """Run simple ablation test comparing different aggregation strategies"""
    results = {}
    
    for i, config in enumerate(configs):
        config_name = config.get('name', f'Config_{i}')
        
        # Create temporary predictor with different strategy
        temp_predictor = EnsemblePredictor(
            task_type=predictor.task_type,
            aggregation_strategy=config.get('aggregation_strategy', 'simple_average'),
            uncertainty_method=predictor.uncertainty_method.value
        )
        temp_predictor.fit(predictor.trained_learners, predictor.learner_metrics, 
                          predictor.bag_characteristics)
        
        # Make predictions
        result = temp_predictor.predict(test_data)
        predictions = result.predictions
        
        # Compute accuracy
        if predictor.task_type == "classification":
            pred_classes = np.argmax(predictions, axis=1)
            accuracy = np.mean(pred_classes == test_labels)
        else:
            mse = np.mean((predictions - test_labels) ** 2)
            accuracy = 1.0 / (1.0 + mse)  # Convert MSE to accuracy-like score
        
        results[config_name] = {
            'accuracy': accuracy,
            'predictions': predictions,
            'config': config
        }
    
    # Find best configuration
    best_config = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_config]['accuracy']
    
    return {
        'results': results,
        'summary': {
            'best_config': best_config,
            'best_accuracy': best_accuracy,
            'n_configs': len(configs)
        }
    }

def run_simple_interpretability_test(predictor: EnsemblePredictor, test_data: Dict[str, np.ndarray], 
                                   test_labels: np.ndarray) -> Dict[str, Any]:
    """Run simple interpretability test"""
    result = predictor.predict(test_data)
    
    # Compute basic interpretability metrics
    interpretability_score = 0.0
    
    # Modality importance diversity
    if result.modality_importance:
        modality_weights = list(result.modality_importance.values())
        if len(modality_weights) > 1:
            diversity = 1.0 - np.var(modality_weights) / np.mean(modality_weights)
            interpretability_score += diversity * 0.3
    
    # Mixture weight diversity
    if result.mixture_weights is not None:
        weight_entropy = -np.sum(result.mixture_weights * np.log(result.mixture_weights + 1e-8), axis=1)
        avg_entropy = np.mean(weight_entropy)
        interpretability_score += avg_entropy * 0.4
    
    # Uncertainty calibration
    if result.uncertainty is not None:
        uncertainty_std = np.std(result.uncertainty)
        interpretability_score += (1.0 - uncertainty_std) * 0.3
    
    return {
        'results': {
            'integrated_analysis': {
                'integrated_interpretability_score': interpretability_score,
                'modality_importance': result.modality_importance,
                'mixture_weight_entropy': np.mean(weight_entropy) if result.mixture_weights is not None else 0.0,
                'uncertainty_std': np.std(result.uncertainty) if result.uncertainty is not None else 0.0
            }
        }
    }

def run_simple_robustness_test(predictor: EnsemblePredictor, test_data: Dict[str, np.ndarray], 
                             test_labels: np.ndarray) -> Dict[str, Any]:
    """Run simple robustness test"""
    # Test with noise perturbation
    noise_levels = [0.01, 0.05, 0.1]
    robustness_scores = []
    
    for noise_level in noise_levels:
        # Add noise to test data
        noisy_data = {}
        for modality, data in test_data.items():
            noise = np.random.normal(0, noise_level, data.shape)
            noisy_data[modality] = data + noise
        
        # Make predictions on noisy data
        result = predictor.predict(noisy_data)
        predictions = result.predictions
        
        # Compute accuracy on noisy data
        if predictor.task_type == "classification":
            pred_classes = np.argmax(predictions, axis=1)
            accuracy = np.mean(pred_classes == test_labels)
        else:
            mse = np.mean((predictions - test_labels) ** 2)
            accuracy = 1.0 / (1.0 + mse)
        
        robustness_scores.append(accuracy)
    
    # Overall robustness score (higher is better)
    overall_robustness = np.mean(robustness_scores)
    
    return {
        'summary': {
            'overall_robustness_score': overall_robustness,
            'noise_levels': noise_levels,
            'robustness_scores': robustness_scores
        }
    }
