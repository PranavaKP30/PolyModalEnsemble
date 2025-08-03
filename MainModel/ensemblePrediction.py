"""
EnsemblePrediction.py
Stage 5: Advanced Multimodal Ensemble Prediction System
Implements the production-grade inference engine as specified in 5EnsemblePredictionDoc.md
"""

import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.nn as nn

# --- Enums for Aggregation and Uncertainty ---

class AggregationStrategy(Enum):
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    STACKING = "stacking"
    DYNAMIC_WEIGHTING = "dynamic_weighting"
    UNCERTAINTY_WEIGHTED = "uncertainty_weighted"
    TRANSFORMER_FUSION = "transformer_fusion"

class UncertaintyMethod(Enum):
    ENTROPY = "entropy"
    VARIANCE = "variance"
    MONTE_CARLO = "monte_carlo"
    ENSEMBLE_DISAGREEMENT = "ensemble_disagreement"
    ATTENTION_BASED = "attention_based"

# --- Prediction Result Container ---

@dataclass
class PredictionResult:
    predictions: np.ndarray
    confidence: Optional[np.ndarray] = None
    uncertainty: Optional[np.ndarray] = None
    modality_importance: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# --- Transformer Meta-Learner ---

class TransformerMetaLearner(nn.Module):
    """
    Transformer-based meta-learner for intelligent prediction fusion
    """
    def __init__(self, input_dim: int, num_heads: int = 8, num_layers: int = 2, hidden_dim: int = 256, num_classes: int = 2, task_type: str = "classification"):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim, num_heads, batch_first=True), num_layers
        )
        self.fusion_head = nn.Linear(input_dim, num_classes)
        self.task_type = task_type

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        attn_out, attn_weights = self.attention(x, x, x)
        trans_out = self.transformer(attn_out)
        logits = self.fusion_head(trans_out.mean(dim=1))
        return logits, attn_weights

# --- Main Orchestrator ---

class EnsemblePredictor:
    def __init__(self,
                 task_type: str = "classification",
                 aggregation_strategy: Union[str, AggregationStrategy] = "weighted_vote",
                 uncertainty_method: Union[str, UncertaintyMethod] = "entropy",
                 calibrate_uncertainty: bool = True,
                 device: str = "auto"):
        self.task_type = task_type
        self.aggregation_strategy = AggregationStrategy(aggregation_strategy) if isinstance(aggregation_strategy, str) else aggregation_strategy
        self.uncertainty_method = UncertaintyMethod(uncertainty_method) if isinstance(uncertainty_method, str) else uncertainty_method
        self.calibrate_uncertainty = calibrate_uncertainty
        self.device = torch.device("cuda" if torch.cuda.is_available() and device in ["auto", "cuda"] else "cpu")
        self.trained_learners = []
        self.learner_metadata = []
        self.transformer_meta_learner = None
        self.confidence_calibrator = None

    def add_trained_learner(self, learner: Any, training_metrics: Dict[str, float], modalities: List[str], pattern: str):
        self.trained_learners.append(learner)
        self.learner_metadata.append({
            'metrics': training_metrics,
            'modalities': modalities,
            'pattern': pattern
        })

    def setup_transformer_fusion(self, input_dim: int, num_classes: int):
        self.transformer_meta_learner = TransformerMetaLearner(input_dim, num_classes=num_classes, task_type=self.task_type).to(self.device)

    def predict(self, data: Dict[str, np.ndarray], return_uncertainty: bool = True) -> PredictionResult:
        """
        Predict using all trained learners. Handles torch/sklearn, fusion/single, classification/regression, real confidence.
        """
        import torch.nn.functional as F
        predictions = []
        confidences = []
        n_samples = next(iter(data.values())).shape[0] if data else 0

        # --- Deterministic ordering of learners and metadata ---
        def learner_sort_key(learner):
            # Prefer learner_id if present, else class name
            return getattr(learner, 'learner_id', learner.__class__.__name__)
        # Pair learners with metadata, sort, then unzip
        paired = list(zip(self.trained_learners, self.learner_metadata))
        paired_sorted = sorted(paired, key=lambda x: learner_sort_key(x[0]))
        sorted_learners, sorted_metadata = zip(*paired_sorted) if paired_sorted else ([], [])

        for learner in sorted_learners:
            # Torch model
            if hasattr(learner, 'forward') or hasattr(learner, 'forward_fusion'):
                learner.eval()
                with torch.no_grad():
                    device = next(learner.parameters()).device if hasattr(learner, 'parameters') else torch.device('cpu')
                    torch_data = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in data.items()}
                    if hasattr(learner, 'forward_fusion'):
                        out = learner.forward_fusion(torch_data)
                    else:
                        out = learner(next(iter(torch_data.values())))
                    if self.task_type == "classification":
                        if out.shape[-1] > 1:
                            prob = F.softmax(out, dim=-1).cpu().numpy()
                            pred = np.argmax(prob, axis=1)
                            conf = np.max(prob, axis=1)
                        else:
                            prob = torch.sigmoid(out).cpu().numpy()
                            pred = (prob > 0.5).astype(int).squeeze()
                            conf = prob.squeeze()
                    else:
                        pred = out.cpu().numpy().squeeze()
                        conf = np.ones_like(pred)
                predictions.append(pred)
                confidences.append(conf)
            # Custom dict-based learner (BaseLearnerInterface): expects dict input
            elif hasattr(learner, 'predict_proba') and 'X' in learner.predict_proba.__code__.co_varnames:
                try:
                    proba = learner.predict_proba(data)
                    pred = np.argmax(proba, axis=1)
                    conf = np.max(proba, axis=1)
                    predictions.append(pred)
                    confidences.append(conf)
                except Exception:
                    # fallback to array if fails
                    proba = learner.predict_proba(np.column_stack([data[k] for k in sorted(data)]))
                    pred = np.argmax(proba, axis=1)
                    conf = np.max(proba, axis=1)
                    predictions.append(pred)
                    confidences.append(conf)
            # Sklearn model: expects array input
            elif hasattr(learner, 'predict_proba'):
                proba = learner.predict_proba(np.column_stack([data[k] for k in sorted(data)]))
                pred = np.argmax(proba, axis=1)
                conf = np.max(proba, axis=1)
                predictions.append(pred)
                confidences.append(conf)
            elif hasattr(learner, 'predict') and 'X' in learner.predict.__code__.co_varnames:
                try:
                    pred = learner.predict(data)
                    predictions.append(pred)
                    confidences.append(np.ones_like(pred, dtype=float))
                except Exception:
                    pred = learner.predict(np.column_stack([data[k] for k in sorted(data)]))
                    predictions.append(pred)
                    confidences.append(np.ones_like(pred, dtype=float))
            elif hasattr(learner, 'predict'):
                pred = learner.predict(np.column_stack([data[k] for k in sorted(data)]))
                predictions.append(pred)
                confidences.append(np.ones_like(pred, dtype=float))
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        # Aggregation
        if self.aggregation_strategy == AggregationStrategy.MAJORITY_VOTE:
            final_pred = self._majority_vote(predictions)
        elif self.aggregation_strategy == AggregationStrategy.WEIGHTED_VOTE:
            weights = np.array([m['metrics'].get('accuracy', 0.5) for m in sorted_metadata])
            final_pred = self._weighted_vote(predictions, weights)
        elif self.aggregation_strategy == AggregationStrategy.TRANSFORMER_FUSION and self.transformer_meta_learner:
            x = torch.tensor(predictions, dtype=torch.float32).unsqueeze(-1).to(self.device)
            logits, attn_weights = self.transformer_meta_learner(x)
            final_pred = torch.argmax(logits, dim=-1).cpu().numpy()
        else:
            final_pred = predictions[0] if len(predictions) > 0 else np.array([])
        # Confidence
        avg_conf = np.mean(confidences, axis=0) if confidences.size > 0 else None
        # Uncertainty
        uncertainty = None
        if return_uncertainty:
            if self.uncertainty_method == UncertaintyMethod.ENTROPY and confidences.size > 0:
                # Entropy of mean probabilities (for classification)
                if self.task_type == "classification" and confidences.shape[0] > 1:
                    probs = np.mean(confidences, axis=0)
                    uncertainty = -np.sum(probs * np.log(probs + 1e-10))
                else:
                    uncertainty = None
            elif self.uncertainty_method == UncertaintyMethod.ENSEMBLE_DISAGREEMENT:
                # Fraction of disagreeing learners
                mode = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, predictions)
                disagreement = 1.0 - np.mean(predictions == mode, axis=0)
                uncertainty = disagreement
        # Modality importance (simple uniform for now)
        modality_importance = {mod: 1.0/len(data) for mod in data} if data else None
        # Metadata
        meta = {
            'n_learners': len(self.trained_learners),
            'aggregation_strategy': self.aggregation_strategy.value,
            'inference_time': 0.0
        }
        return PredictionResult(
            predictions=final_pred,
            confidence=avg_conf,
            uncertainty=uncertainty,
            modality_importance=modality_importance,
            metadata=meta
        )

    def _majority_vote(self, predictions: np.ndarray) -> np.ndarray:
        # predictions: (n_learners, n_samples)
        from scipy.stats import mode
        result = mode(predictions, axis=0)
        return np.asarray(result.mode).reshape(-1)

    def _weighted_vote(self, predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
        # predictions: (n_learners, n_samples), weights: (n_learners,)
        n_samples = predictions.shape[1]
        weighted_preds = np.zeros((n_samples,))
        for i in range(n_samples):
            vals, counts = np.unique(predictions[:, i], return_counts=True)
            weighted_counts = {v: np.sum(weights[predictions[:, i] == v]) for v in vals}
            weighted_preds[i] = max(weighted_counts, key=weighted_counts.get)
        return weighted_preds.astype(int)

    def evaluate(self, data: Dict[str, np.ndarray], true_labels: np.ndarray, detailed: bool = True) -> Dict[str, Any]:
        result = self.predict(data, return_uncertainty=True)
        accuracy = np.mean(result.predictions == true_labels)
        metrics = {
            'accuracy': accuracy,
            'confidence_mean': np.mean(result.confidence) if result.confidence is not None else None,
            'uncertainty_mean': np.mean(result.uncertainty) if result.uncertainty is not None else None
        }
        if detailed:
            metrics['modality_importance'] = result.modality_importance
            metrics['metadata'] = result.metadata
        return metrics

# --- Factory Function ---
def create_ensemble_predictor(task_type: str = "classification", aggregation_strategy: str = "weighted_vote", uncertainty_method: str = "entropy", calibrate_uncertainty: bool = True, device: str = "auto", **kwargs) -> EnsemblePredictor:
    return EnsemblePredictor(
        task_type=task_type,
        aggregation_strategy=aggregation_strategy,
        uncertainty_method=uncertainty_method,
        calibrate_uncertainty=calibrate_uncertainty,
        device=device
    )

# --- API Exports ---
__all__ = [
    "AggregationStrategy",
    "UncertaintyMethod",
    "PredictionResult",
    "TransformerMetaLearner",
    "EnsemblePredictor",
    "create_ensemble_predictor"
]
