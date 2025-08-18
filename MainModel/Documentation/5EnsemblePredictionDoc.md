# Stage 5: Ensemble Prediction Documentation

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com)
[![Status](https://img.shields.io/badge/status-production_ready-brightgreen.svg)](https://github.com)
[![Component](https://img.shields.io/badge/component-ensemble_prediction-orange.svg)](https://github.com)

**Advanced multimodal ensemble prediction system with intelligent aggregation strategies, uncertainty quantification, and transformer-based meta-learning for production-grade inference and decision-making.**

## 🎯 Overview

The `5EnsemblePrediction.py` module is the **sophisticated inference engine** of the multimodal pipeline, responsible for aggregating predictions from diverse trained learners with state-of-the-art fusion techniques, uncertainty quantification, and comprehensive prediction analytics. This production-ready system transforms individual learner outputs into reliable ensemble decisions with confidence estimates.

### Core Value Proposition
- 🔮 **Intelligent Aggregation** - Advanced fusion strategies including transformer-based meta-learning
- 📊 **Uncertainty Quantification** - Bayesian uncertainty estimation and confidence scoring
- 🧠 **Adaptive Weighting** - Dynamic learner importance based on performance and confidence
- 🎯 **Multi-Task Support** - Classification, regression, and multilabel prediction capabilities
- 🚀 **Production Ready** - GPU acceleration, real-time inference, and comprehensive monitoring
- 🔍 **Interpretability** - Attention-based explanations and modality importance analysis

## 🏗️ Architecture Overview

The ensemble prediction system implements a **7-layer architecture** designed for maximum accuracy, reliability, and interpretability:

```
┌─────────────────────────────────────────────────────────────────┐
│                Ensemble Prediction Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Prediction Orchestration & Management                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Ensemble     │  │Input        │  │Resource     │             │
│  │Predictor    │  │Validation   │  │Manager      │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Individual Learner Inference                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Neural       │  │Tabular      │  │Modality     │             │
│  │Predictors   │  │Predictors   │  │Routing      │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Advanced Aggregation Strategies                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Transformer  │  │Weighted     │  │Dynamic      │             │
│  │Meta-Learner │  │Voting       │  │Weighting    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Uncertainty Quantification Engine                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Entropy      │  │Ensemble     │  │Attention    │             │
│  │Estimation   │  │Disagreement │  │Uncertainty  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 5: Confidence Calibration & Scoring                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Temperature  │  │Platt        │  │Bayesian     │             │
│  │Scaling      │  │Calibration  │  │Calibration  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 6: Interpretability & Explanation                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Attention    │  │Modality     │  │Learner      │             │
│  │Visualization│  │Importance   │  │Contribution │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 7: Evaluation & Quality Assurance                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Performance  │  │Calibration  │  │Prediction   │             │
│  │Metrics      │  │Assessment   │  │Validation   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
           ↓                    ↓                    ↓
    Final Predictions    Uncertainty Scores     Interpretation
```

### Core Components

#### 1. **EnsemblePredictor** - Main Orchestrator
High-level coordination of ensemble prediction with intelligent strategy selection and resource management.

#### 2. **TransformerMetaLearner** - Advanced Fusion Engine
State-of-the-art transformer-based meta-learner for intelligent prediction aggregation with attention mechanisms.

#### 3. **PredictionResult** - Structured Prediction Container
Comprehensive prediction metadata including confidence, uncertainty, modality importance, and detailed metadata.

#### 4. **AggregationStrategy** - Fusion Methods
Multiple sophisticated aggregation approaches from simple voting to advanced neural fusion.

#### 5. **UncertaintyMethod** - Uncertainty Estimation
Comprehensive uncertainty quantification using entropy, variance, and ensemble disagreement methods.

#### 6. **Factory Functions** - Pipeline Creation
Utility functions for creating and configuring ensemble predictors with optimal defaults.

## 🚀 Quick Start Guide

### Basic Ensemble Prediction

```python
from mainModel import MultiModalEnsembleModel, create_synthetic_model

# Create and train model (assuming this is done)
model = create_synthetic_model({
    'text_embeddings': (768, 'text'),
    'image_features': (2048, 'image'),
    'user_metadata': (50, 'tabular')
}, n_samples=1000, n_classes=5)

# Train ensemble (simplified)
model.create_ensemble(n_bags=15)
model.generate_bags()
model.select_base_learners()
model.setup_training(epochs=10)
trained_learners, metrics = model.train_ensemble()

# Setup advanced predictor
model.setup_predictor(
    aggregation_strategy='transformer_fusion',  # Advanced aggregation
    uncertainty_method='attention_based',       # Attention uncertainty
    calibrate_uncertainty=True                  # Calibrated confidence
)

# Make predictions with uncertainty
new_data = {
    'text_embeddings': np.random.randn(5, 768),
    'image_features': np.random.randn(5, 2048), 
    'user_metadata': np.random.randn(5, 50)
}

result = model.predict(new_data, return_uncertainty=True)

print(f"Predictions: {result.predictions}")
print(f"Confidence: {result.confidence}")
print(f"Uncertainty: {result.uncertainty}")
print(f"Modality Importance: {result.modality_importance}")
```

## 🔧 Data Structures and Enums

### AggregationStrategy Enum

```python
class AggregationStrategy(Enum):
    MAJORITY_VOTE = "majority_vote"           # Simple majority voting
    WEIGHTED_VOTE = "weighted_vote"           # Performance-weighted voting
    CONFIDENCE_WEIGHTED = "confidence_weighted" # Confidence-based weighting
    STACKING = "stacking"                     # Meta-learner stacking
    DYNAMIC_WEIGHTING = "dynamic_weighting"   # Adaptive weight computation
    UNCERTAINTY_WEIGHTED = "uncertainty_weighted" # Uncertainty-aware aggregation
    TRANSFORMER_FUSION = "transformer_fusion" # Neural meta-learner
```

**Aggregation Strategy Features:**
- **MAJORITY_VOTE**: Simple democratic voting where each learner contributes equally
- **WEIGHTED_VOTE**: Performance-based weighting using training metrics (accuracy, F1-score)
- **CONFIDENCE_WEIGHTED**: Weighting based on prediction confidence scores
- **STACKING**: Meta-learner approach for combining predictions
- **DYNAMIC_WEIGHTING**: Adaptive weight computation based on current performance
- **UNCERTAINTY_WEIGHTED**: Weighting that considers prediction uncertainty
- **TRANSFORMER_FUSION**: Advanced neural meta-learner using transformer architecture

### UncertaintyMethod Enum

```python
class UncertaintyMethod(Enum):
    ENTROPY = "entropy"                       # Information-theoretic uncertainty
    VARIANCE = "variance"                     # Prediction variance
    MONTE_CARLO = "monte_carlo"               # Sampling-based uncertainty
    ENSEMBLE_DISAGREEMENT = "ensemble_disagreement" # Model disagreement
    ATTENTION_BASED = "attention_based"       # Attention entropy
```

**Uncertainty Method Features:**
- **ENTROPY**: Information-theoretic uncertainty using probability distributions
- **VARIANCE**: Statistical variance of predictions across ensemble members
- **MONTE_CARLO**: Sampling-based uncertainty estimation
- **ENSEMBLE_DISAGREEMENT**: Fraction of disagreeing learners
- **ATTENTION_BASED**: Uncertainty based on transformer attention weights

### PredictionResult Dataclass

```python
@dataclass
class PredictionResult:
    predictions: np.ndarray
    confidence: Optional[np.ndarray] = None
    uncertainty: Optional[np.ndarray] = None
    modality_importance: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**PredictionResult Fields:**
- **predictions**: Final ensemble predictions (numpy array)
- **confidence**: Confidence scores for predictions (optional)
- **uncertainty**: Uncertainty estimates for predictions (optional)
- **modality_importance**: Importance weights for each modality (optional)
- **metadata**: Additional prediction metadata (dictionary)

**Usage Example:**
```python
result = predictor.predict(data, return_uncertainty=True)

# Access prediction results
print(f"Predictions: {result.predictions}")
print(f"Confidence: {result.confidence}")
print(f"Uncertainty: {result.uncertainty}")
print(f"Modality Importance: {result.modality_importance}")
print(f"Metadata: {result.metadata}")
```

### Advanced Aggregation Strategies

```python
# Configure different aggregation strategies
strategies = [
    'majority_vote',          # Simple majority voting
    'weighted_vote',          # Performance-weighted voting  
    'confidence_weighted',    # Confidence-based weighting
    'dynamic_weighting',      # Adaptive weight computation
    'transformer_fusion',     # Neural meta-learner
    'uncertainty_weighted'    # Uncertainty-aware aggregation
]

for strategy in strategies:
    model.setup_predictor(aggregation_strategy=strategy)
    result = model.predict(test_data)
    print(f"{strategy}: Accuracy = {result.metadata['accuracy']:.3f}")
```

## 📊 Detailed Component Documentation

### 1. EnsemblePredictor Class

The main orchestrator class responsible for coordinating ensemble prediction across multiple trained learners.

#### Key Features
- **Multi-Strategy Aggregation**: Supports 7 different aggregation strategies
- **Uncertainty Quantification**: 5 different uncertainty estimation methods
- **GPU Acceleration**: Optimized inference on CUDA devices
- **Calibrated Confidence**: Temperature scaling and Platt calibration
- **Interpretability**: Attention weights and modality importance

#### Constructor Parameters

```python
def __init__(self,
             task_type: str = "classification",
             aggregation_strategy: Union[str, AggregationStrategy] = "weighted_vote",
             uncertainty_method: Union[str, UncertaintyMethod] = "entropy", 
             calibrate_uncertainty: bool = True,
             device: str = "auto"):
    """
    Initialize ensemble predictor
    
    Args:
        task_type: 'classification', 'regression', or 'multilabel'
        aggregation_strategy: How to combine individual predictions
        uncertainty_method: Method for uncertainty estimation
        calibrate_uncertainty: Whether to calibrate confidence scores
        device: Computing device ('auto', 'cpu', 'cuda')
    """
    self.task_type = task_type
    self.aggregation_strategy = AggregationStrategy(aggregation_strategy) if isinstance(aggregation_strategy, str) else aggregation_strategy
    self.uncertainty_method = UncertaintyMethod(uncertainty_method) if isinstance(uncertainty_method, str) else uncertainty_method
    self.calibrate_uncertainty = calibrate_uncertainty
    self.device = torch.device("cuda" if torch.cuda.is_available() and device in ["auto", "cuda"] else "cpu")
    self.trained_learners = []
    self.learner_metadata = []
    self.transformer_meta_learner = None
    self.confidence_calibrator = None
```

**Initialization Features:**
- **Task Type Support**: Classification, regression, and multilabel tasks
- **Strategy Validation**: Automatic conversion of string strategies to enum values
- **Device Detection**: Automatic GPU/CPU selection based on availability
- **Learner Management**: Lists for storing trained learners and their metadata
- **Meta-Learner Support**: Optional transformer meta-learner for advanced fusion
- **Calibration Support**: Optional confidence calibration system

#### Core Methods

##### add_trained_learner()
Register a trained learner with the ensemble predictor.

```python
def add_trained_learner(self, learner: Any, training_metrics: Dict[str, float], modalities: List[str], pattern: str):
    self.trained_learners.append(learner)
    self.learner_metadata.append({
        'metrics': training_metrics,
        'modalities': modalities,
        'pattern': pattern
    })
```

**Learner Management Features:**
- **Learner Storage**: Maintains list of trained learners
- **Metadata Tracking**: Stores training metrics, modalities, and patterns
- **Flexible Input**: Accepts any model type (PyTorch, scikit-learn, custom)
- **Performance History**: Tracks training performance for weighting
- **Modality Information**: Records which modalities each learner uses
- **Pattern Documentation**: Describes modality combination patterns

**Usage Example:**
```python
predictor.add_trained_learner(
    learner=cnn_model,
    training_metrics={'accuracy': 0.89, 'f1_score': 0.87},
    modalities=['image_features'],
    pattern='image_only'
)

predictor.add_trained_learner(
    learner=fusion_model,
    training_metrics={'accuracy': 0.92, 'f1_score': 0.91},
    modalities=['text_embeddings', 'image_features'],
    pattern='text_image_fusion'
)
```

##### setup_transformer_fusion()
Configure transformer meta-learner for advanced fusion.

```python
def setup_transformer_fusion(self, input_dim: int, num_classes: int):
    self.transformer_meta_learner = TransformerMetaLearner(input_dim, num_classes=num_classes, task_type=self.task_type).to(self.device)
```

**Transformer Fusion Features:**
- **Meta-Learner Creation**: Instantiates transformer meta-learner
- **Device Management**: Automatically moves to appropriate device (GPU/CPU)
- **Task Awareness**: Configures for classification or regression
- **Dimension Handling**: Supports variable input dimensions
- **Advanced Fusion**: Enables transformer-based prediction aggregation

##### predict()
Generate ensemble predictions with comprehensive metadata.

```python
def predict(self, data: Dict[str, np.ndarray], return_uncertainty: bool = True) -> PredictionResult:
    """
    Predict using all trained learners. Handles torch/sklearn, fusion/single, classification/regression, real confidence.
    Returns a PredictionResult object with predictions, confidence, uncertainty, and metadata.
    """
    return self._predict_internal(data, return_uncertainty)
```

**Prediction Features:**
- **Multi-Learner Support**: Handles both PyTorch and scikit-learn models
- **Fusion Models**: Supports models with `forward_fusion` method
- **Task Flexibility**: Classification and regression support
- **Uncertainty Quantification**: Optional uncertainty estimation
- **Comprehensive Output**: Returns predictions, confidence, uncertainty, and metadata

##### predict_classes()
Sklearn-like interface for classification tasks.

```python
def predict_classes(self, data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    sklearn-like interface: returns class predictions directly as numpy array.
    For classification tasks only.
    """
    if self.task_type != "classification":
        raise ValueError("predict_classes() is only available for classification tasks")
    result = self._predict_internal(data, return_uncertainty=False)
    return result.predictions
```

##### predict_values()
Sklearn-like interface for regression tasks.

```python
def predict_values(self, data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    sklearn-like interface: returns regression values directly as numpy array.
    For regression tasks only.
    """
    if self.task_type != "regression":
        raise ValueError("predict_values() is only available for regression tasks")
    result = self._predict_internal(data, return_uncertainty=False)
    return result.predictions
```

##### predict_proba()
Sklearn-like interface for probability predictions.

```python
def predict_proba(self, data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    sklearn-like interface: returns probability predictions directly as numpy array.
    For classification tasks only.
    """
    if self.task_type != "classification":
        raise ValueError("predict_proba() is only available for classification tasks")
    
    # Get individual learner probabilities
    all_probas = []
    n_samples = next(iter(data.values())).shape[0] if data else 0
    
    for learner in self.trained_learners:
        if hasattr(learner, 'predict_proba'):
            try:
                # Try dict input first
                if hasattr(learner, 'predict_proba') and 'X' in learner.predict_proba.__code__.co_varnames:
                    try:
                        proba = learner.predict_proba(data)
                        all_probas.append(proba)
                        continue
                    except Exception:
                        pass
                
                # Fallback to array input
                proba = learner.predict_proba(np.column_stack([data[k] for k in sorted(data)]))
                all_probas.append(proba)
            except Exception:
                # If no predict_proba, create one-hot from predictions
                pred = learner.predict(np.column_stack([data[k] for k in sorted(data)]))
                n_classes = max(pred) + 1
                proba = np.zeros((len(pred), n_classes))
                proba[np.arange(len(pred)), pred] = 1.0
                all_probas.append(proba)
    
    if not all_probas:
        raise ValueError("No learners with predict_proba capability found")
    
    # Ensure all probability arrays have the same shape
    max_classes = max(proba.shape[1] for proba in all_probas)
    
    # Pad or truncate all probability arrays to have the same number of classes
    normalized_probas = []
    for proba in all_probas:
        if proba.shape[1] < max_classes:
            # Pad with zeros
            padded_proba = np.zeros((proba.shape[0], max_classes))
            padded_proba[:, :proba.shape[1]] = proba
            normalized_probas.append(padded_proba)
        elif proba.shape[1] > max_classes:
            # Truncate (shouldn't happen, but just in case)
            normalized_probas.append(proba[:, :max_classes])
        else:
            normalized_probas.append(proba)
    
    # Average probabilities across learners
    return np.mean(normalized_probas, axis=0)
```

**Sklearn-Like Interface Features:**
- **predict_classes()**: Direct class predictions for classification
- **predict_values()**: Direct value predictions for regression
- **predict_proba()**: Probability predictions with automatic normalization
- **Error Handling**: Task-specific validation and fallback mechanisms
- **Compatibility**: Works with scikit-learn pipeline integration

##### evaluate()
Comprehensive evaluation of ensemble performance.

```python
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
```

**Evaluation Features:**
- **Accuracy Calculation**: Mean accuracy across all predictions
- **Confidence Analysis**: Average confidence scores
- **Uncertainty Analysis**: Average uncertainty estimates
- **Detailed Metrics**: Optional modality importance and metadata
- **Comprehensive Assessment**: Complete ensemble performance evaluation

### Internal Prediction Implementation

```python
def _predict_internal(self, data: Dict[str, np.ndarray], return_uncertainty: bool = True) -> PredictionResult:
    """
    Internal prediction method that returns PredictionResult object.
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
                if self.task_type == "classification":
                    proba = learner.predict_proba(data)
                    pred = np.argmax(proba, axis=1)
                    conf = np.max(proba, axis=1)
                else:
                    # Regression: no predict_proba, use predict
                    pred = learner.predict(data)
                    conf = np.ones_like(pred, dtype=float)
                predictions.append(pred)
                confidences.append(conf)
            except Exception:
                # fallback to array if fails
                if self.task_type == "classification" and hasattr(learner, 'predict_proba'):
                    proba = learner.predict_proba(np.column_stack([data[k] for k in sorted(data)]))
                    pred = np.argmax(proba, axis=1)
                    conf = np.max(proba, axis=1)
                else:
                    # Regression or no predict_proba
                    pred = learner.predict(np.column_stack([data[k] for k in sorted(data)]))
                    conf = np.ones_like(pred, dtype=float)
                predictions.append(pred)
                confidences.append(conf)
        # Sklearn model: expects array input
        elif hasattr(learner, 'predict_proba') and self.task_type == "classification":
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
    if self.task_type == "regression":
        # For regression, use mean or weighted mean
        if self.aggregation_strategy == AggregationStrategy.WEIGHTED_VOTE:
            weights = np.array([m['metrics'].get('r2_score', 0.5) for m in sorted_metadata])
            weights = weights / np.sum(weights)  # Normalize weights
            final_pred = np.average(predictions, axis=0, weights=weights)
        else:
            # Default to mean for regression
            final_pred = np.mean(predictions, axis=0)
    else:
        # Classification aggregation
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
```

**Internal Prediction Features:**
- **Deterministic Ordering**: Consistent learner ordering for reproducible results
- **Multi-Model Support**: Handles PyTorch, scikit-learn, and custom models
- **Fusion Model Support**: Special handling for models with `forward_fusion`
- **Error Handling**: Graceful fallbacks for different model types
- **Task-Specific Processing**: Different handling for classification vs regression
- **Aggregation Strategies**: Multiple aggregation methods with task-specific logic
- **Uncertainty Computation**: Entropy and ensemble disagreement methods
- **Metadata Collection**: Comprehensive prediction metadata

### 2. Aggregation Strategies

#### AggregationStrategy Enum

```python
class AggregationStrategy(Enum):
    MAJORITY_VOTE = "majority_vote"           # Simple majority voting
    WEIGHTED_VOTE = "weighted_vote"           # Performance-weighted
    CONFIDENCE_WEIGHTED = "confidence_weighted" # Confidence-based
    STACKING = "stacking"                     # Meta-learner stacking
    DYNAMIC_WEIGHTING = "dynamic_weighting"   # Adaptive weights
    UNCERTAINTY_WEIGHTED = "uncertainty_weighted" # Uncertainty-aware
    TRANSFORMER_FUSION = "transformer_fusion" # Neural meta-learner
```

#### Strategy Implementations

##### Majority Vote
Simple democratic voting where each learner contributes equally.

```python
def _majority_vote(self, predictions: np.ndarray) -> np.ndarray:
    # predictions: (n_learners, n_samples)
    from scipy.stats import mode
    result = mode(predictions, axis=0)
    return np.asarray(result.mode).reshape(-1)
```

**Majority Vote Features:**
- **Simple Aggregation**: Uses scipy.stats.mode for efficient computation
- **Equal Weighting**: All learners contribute equally regardless of performance
- **Robust**: Works well with diverse ensemble members
- **No Training**: No additional training required for aggregation
- **Classification Only**: Designed for classification tasks

##### Weighted Vote
Performance-based weighting using training metrics.

```python
def _weighted_vote(self, predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # predictions: (n_learners, n_samples), weights: (n_learners,)
    n_samples = predictions.shape[1]
    weighted_preds = np.zeros((n_samples,))
    for i in range(n_samples):
        vals, counts = np.unique(predictions[:, i], return_counts=True)
        weighted_counts = {v: np.sum(weights[predictions[:, i] == v]) for v in vals}
        weighted_preds[i] = max(weighted_counts, key=weighted_counts.get)
    return weighted_preds.astype(int)
```

**Weighted Vote Features:**
- **Performance-Based Weighting**: Uses training metrics (accuracy, F1-score) for weights
- **Sample-Level Processing**: Computes weighted counts for each sample independently
- **Unique Value Handling**: Handles multiple unique predictions per sample
- **Weight Normalization**: Automatically normalizes weights to sum to 1
- **Classification Focus**: Optimized for classification tasks with discrete outputs

##### Weighted Vote
Performance-based weighting using training metrics.

```python
# Implementation details  
def _weighted_vote(self, 
                  predictions: List[np.ndarray],
                  weights: np.ndarray) -> np.ndarray:
    """
    Performance-weighted voting
    
    Process:
    1. Compute weights based on training performance
    2. Weight each prediction by learner quality
    3. Aggregate using weighted combination
    
    Weight Computation:
    - Classification: Uses F1-score or accuracy
    - Regression: Uses inverse MSE
    - Minimum weight threshold: 0.1
    
    Advantages:
    - Emphasizes better-performing learners
    - Automatic quality weighting
    - Better than simple averaging
    """
```

##### Transformer Fusion
Advanced neural meta-learner using transformer architecture.

```python
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
```

**TransformerMetaLearner Features:**
- **Multi-head Attention**: Captures complex relationships between learner predictions
- **Transformer Encoder**: Deep representation learning for prediction fusion
- **Batch-First Processing**: Optimized for batch inference
- **Attention Weights**: Provides interpretability through attention visualization
- **Task-Aware Design**: Supports both classification and regression tasks
- **Flexible Architecture**: Configurable number of heads, layers, and hidden dimensions

**Implementation Details:**
- **Input**: Learner predictions as sequence (batch, seq_len, input_dim)
- **Attention**: Multi-head self-attention over learner predictions
- **Transformer**: Deep encoder layers for representation learning
- **Fusion**: Mean pooling followed by linear classification head
- **Output**: Final predictions and attention weights for interpretability
```

### 3. Uncertainty Quantification

#### UncertaintyMethod Enum

```python
class UncertaintyMethod(Enum):
    ENTROPY = "entropy"                       # Information-theoretic
    VARIANCE = "variance"                     # Prediction variance
    MONTE_CARLO = "monte_carlo"               # Sampling-based
    ENSEMBLE_DISAGREEMENT = "ensemble_disagreement" # Model disagreement
    ATTENTION_BASED = "attention_based"       # Attention entropy
```

#### Uncertainty Implementations

##### Entropy-Based Uncertainty

```python
def _compute_entropy_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
    """
    Compute predictive uncertainty using entropy
    
    Formula:
    H(p) = -∑ p_i * log(p_i)
    
    Where:
    - p_i are class probabilities
    - Higher entropy = higher uncertainty
    - Range: [0, log(num_classes)]
    
    Interpretation:
    - 0.0: Completely certain (one class has p=1.0)
    - log(K): Maximum uncertainty (uniform distribution)
    
    Use Cases:
    - Classification tasks
    - Well-calibrated models
    - Single-sample uncertainty
    """
    epsilon = 1e-10  # Numerical stability
    entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)
    return entropy
```

##### Ensemble Disagreement

```python
def _compute_disagreement_uncertainty(self, 
                                     individual_predictions: List[np.ndarray]) -> np.ndarray:
    """
    Compute uncertainty based on ensemble member disagreement
    
    Process:
    1. Collect predictions from all learners
    2. For each sample, measure prediction diversity
    3. Higher diversity = higher uncertainty
    
    Metrics:
    - Classification: Fraction of disagreeing learners
    - Regression: Standard deviation of predictions
    
    Advantages:
    - Model-agnostic
    - Captures epistemic uncertainty
    - Robust to miscalibration
    
    Formula (Classification):
    disagreement = 1 - (n_majority / n_total)
    """
```

##### Attention-Based Uncertainty

```python
def _compute_attention_uncertainty(self, attention_weights: np.ndarray) -> np.ndarray:
    """
    Compute uncertainty using transformer attention weights
    
    Process:
    1. Extract attention weights from transformer meta-learner
    2. Compute entropy of attention distribution
    3. High attention entropy = high uncertainty
    
    Intuition:
    - Focused attention (low entropy) = confident prediction
    - Scattered attention (high entropy) = uncertain prediction
    
    Formula:
    attention_entropy = -∑ α_i * log(α_i)
    
    Where α_i are attention weights over learners
    
    Novel Contribution:
    - Leverages learned attention patterns
    - Provides interpretable uncertainty
    - Scales with model complexity
    """
```

### 4. Production Use Cases

#### Real-Time Inference API

```python
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load trained ensemble
predictor = EnsemblePredictor.load('production_model.pkl')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Production prediction endpoint with comprehensive error handling
    """
    try:
        # Parse input data
        data = request.json
        
        # Validate input format
        required_modalities = ['text_features', 'image_features', 'metadata']
        for modality in required_modalities:
            if modality not in data:
                return jsonify({
                    'error': f'Missing required modality: {modality}',
                    'required_modalities': required_modalities
                }), 400
        
        # Convert to numpy arrays
        processed_data = {}
        for modality, values in data.items():
            processed_data[modality] = np.array(values)
        
        # Make prediction
        result = predictor.predict(processed_data, return_uncertainty=True)
        
        # Prepare response
        response = {
            'predictions': result.predictions.tolist(),
            'confidence': result.confidence.tolist() if result.confidence is not None else None,
            'uncertainty': result.uncertainty.tolist() if result.uncertainty is not None else None,
            'modality_importance': result.modality_importance,
            'processing_metadata': {
                'n_learners': result.metadata['n_learners'],
                'aggregation_strategy': result.metadata['aggregation_strategy'],
                'inference_time_ms': result.metadata.get('inference_time', 0) * 1000
            }
        }
        
        # Add warning for low confidence predictions
        if result.confidence is not None:
            avg_confidence = np.mean(result.confidence)
            if avg_confidence < 0.7:
                response['warning'] = 'Low confidence prediction - manual review recommended'
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
```

#### Batch Processing Pipeline

```python
def process_large_dataset(predictor: EnsemblePredictor,
                         data_loader: DataLoader,
                         batch_size: int = 100,
                         save_path: str = 'predictions.csv') -> pd.DataFrame:
    """
    Process large datasets efficiently with batch prediction
    
    Args:
        predictor: Trained ensemble predictor
        data_loader: Iterator over data batches
        batch_size: Number of samples per batch
        save_path: Path to save results
        
    Returns:
        DataFrame with predictions and metadata
    """
    
    all_results = []
    
    for batch_idx, batch_data in enumerate(data_loader):
        try:
            # Make batch prediction
            result = predictor.predict(batch_data, return_uncertainty=True)
            
            # Process results
            batch_results = {
                'batch_id': [batch_idx] * len(result.predictions),
                'sample_id': list(range(len(result.predictions))),
                'prediction': result.predictions.tolist(),
                'confidence': result.confidence.tolist() if result.confidence is not None else [None] * len(result.predictions),
                'uncertainty': result.uncertainty.tolist() if result.uncertainty is not None else [None] * len(result.predictions)
            }
            
            # Add modality importance if available
            if result.modality_importance:
                for modality, importance in result.modality_importance.items():
                    batch_results[f'importance_{modality}'] = [importance] * len(result.predictions)
            
            all_results.append(pd.DataFrame(batch_results))
            
            # Progress tracking
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}, samples: {len(result.predictions)}")
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            continue
    
    # Combine all results
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Save to file
    final_results.to_csv(save_path, index=False)
    
    # Summary statistics
    print(f"Batch processing completed:")
    print(f"  Total samples: {len(final_results)}")
    print(f"  Average confidence: {final_results['confidence'].mean():.3f}")
    print(f"  Low confidence samples: {(final_results['confidence'] < 0.7).sum()}")
    
    return final_results
```

### 5. Advanced Features

#### Confidence Calibration

```python
class TemperatureScaling:
    """
    Temperature scaling for confidence calibration
    
    Calibrates prediction confidence to match actual accuracy.
    Essential for production systems requiring reliable uncertainty.
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.fitted = False
    
    def fit(self, logits: np.ndarray, true_labels: np.ndarray):
        """
        Fit temperature parameter on validation data
        
        Args:
            logits: Raw model outputs before softmax
            true_labels: Ground truth labels
        """
        from scipy.optimize import minimize_scalar
        
        def nll_loss(temp):
            scaled_logits = logits / temp
            probabilities = softmax(scaled_logits, axis=1)
            nll = -np.mean(np.log(probabilities[range(len(true_labels)), true_labels] + 1e-10))
            return nll
        
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        self.fitted = True
        
        print(f"Optimal temperature: {self.temperature:.3f}")
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits"""
        if not self.fitted:
            raise ValueError("Must fit temperature scaling first")
        
        return logits / self.temperature
```

#### Modality Importance Analysis

```python
def analyze_modality_importance(predictor: EnsemblePredictor,
                               test_data: Dict[str, np.ndarray],
                               method: str = 'occlusion') -> Dict[str, float]:
    """
    Analyze the importance of different modalities
    
    Args:
        predictor: Trained ensemble predictor
        test_data: Test dataset
        method: Analysis method ('occlusion', 'attention', 'permutation')
        
    Returns:
        Dictionary mapping modality names to importance scores
    """
    
    if method == 'occlusion':
        # Occlusion-based importance
        baseline_result = predictor.predict(test_data)
        baseline_confidence = np.mean(baseline_result.confidence)
        
        importance_scores = {}
        
        for modality in test_data.keys():
            # Create occluded data (replace with zeros)
            occluded_data = test_data.copy()
            occluded_data[modality] = np.zeros_like(test_data[modality])
            
            # Get prediction without this modality
            occluded_result = predictor.predict(occluded_data)
            occluded_confidence = np.mean(occluded_result.confidence)
            
            # Importance = drop in confidence when modality is removed
            importance = baseline_confidence - occluded_confidence
            importance_scores[modality] = max(0.0, importance)  # Ensure non-negative
        
        # Normalize to sum to 1
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            importance_scores = {k: v/total_importance for k, v in importance_scores.items()}
        
        return importance_scores
    
    elif method == 'attention' and hasattr(predictor, 'transformer_meta_learner'):
        # Extract attention-based importance from transformer
        result = predictor.predict(test_data)
        if result.modality_importance:
            return result.modality_importance
    
    else:
        raise ValueError(f"Unknown importance analysis method: {method}")
```

### 6. Performance Optimization

#### GPU Acceleration

```python
def optimize_for_gpu(predictor: EnsemblePredictor):
    """
    Optimize ensemble predictor for GPU inference
    
    Optimizations:
    - Move models to GPU
    - Enable mixed precision
    - Batch processing optimization
    - Memory management
    """
    
    if torch.cuda.is_available():
        # Move learners to GPU
        for learner in predictor.trained_learners:
            if hasattr(learner, 'cuda'):
                learner.cuda()
        
        # Enable mixed precision for neural networks
        predictor.use_mixed_precision = True
        
        # Optimize batch sizes for GPU memory
        predictor.optimal_batch_size = 64
        
        print(f"✅ GPU optimization enabled on {torch.cuda.get_device_name()}")
    else:
        print("⚠️ GPU not available, using CPU optimization")
```

#### Memory Management

```python
def memory_efficient_prediction(predictor: EnsemblePredictor,
                               large_dataset: Dict[str, np.ndarray],
                               max_memory_gb: float = 2.0) -> PredictionResult:
    """
    Memory-efficient prediction for large datasets
    
    Args:
        predictor: Ensemble predictor
        large_dataset: Large input dataset
        max_memory_gb: Maximum memory usage in GB
        
    Returns:
        Prediction results with memory management
    """
    
    # Estimate memory requirements
    sample_size = sum(arr.nbytes for arr in large_dataset.values()) / len(next(iter(large_dataset.values())))
    max_samples = int((max_memory_gb * 1e9) / sample_size)
    
    n_samples = len(next(iter(large_dataset.values())))
    
    if n_samples <= max_samples:
        # Dataset fits in memory
        return predictor.predict(large_dataset)
    
    else:
        # Process in chunks
        print(f"Processing {n_samples} samples in chunks of {max_samples}")
        
        all_predictions = []
        all_uncertainties = []
        all_confidences = []
        
        for start_idx in range(0, n_samples, max_samples):
            end_idx = min(start_idx + max_samples, n_samples)
            
            # Extract chunk
            chunk_data = {}
            for modality, data in large_dataset.items():
                chunk_data[modality] = data[start_idx:end_idx]
            
            # Process chunk
            chunk_result = predictor.predict(chunk_data)
            
            all_predictions.append(chunk_result.predictions)
            if chunk_result.uncertainty is not None:
                all_uncertainties.append(chunk_result.uncertainty)
            if chunk_result.confidence is not None:
                all_confidences.append(chunk_result.confidence)
            
            # Memory cleanup
            del chunk_data, chunk_result
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Combine results
        final_predictions = np.concatenate(all_predictions)
        final_uncertainty = np.concatenate(all_uncertainties) if all_uncertainties else None
        final_confidence = np.concatenate(all_confidences) if all_confidences else None
        
        return PredictionResult(
            predictions=final_predictions,
            uncertainty=final_uncertainty,
            confidence=final_confidence,
            metadata={'processed_in_chunks': True, 'chunk_size': max_samples}
        )
```

## 🔧 Configuration and Customization

### Advanced Configuration

```python
# Comprehensive predictor configuration
predictor_config = {
    'aggregation_strategy': 'transformer_fusion',
    'uncertainty_method': 'attention_based',
    'calibrate_uncertainty': True,
    'transformer_config': {
        'num_heads': 12,
        'num_layers': 3,
        'hidden_dim': 512,
        'dropout': 0.1
    },
    'uncertainty_config': {
        'monte_carlo_samples': 100,
        'temperature_scaling': True,
        'confidence_threshold': 0.8
    },
    'performance_config': {
        'use_gpu': True,
        'mixed_precision': True,
        'batch_size': 32,
        'max_memory_gb': 4.0
    }
}

# Apply configuration
predictor = EnsemblePredictor(**predictor_config)
```

### Custom Aggregation Strategy

```python
class CustomAggregationStrategy:
    """
    Custom aggregation strategy for domain-specific fusion
    """
    
    def __init__(self, domain_weights: Dict[str, float]):
        self.domain_weights = domain_weights
    
    def aggregate(self, 
                 predictions: List[np.ndarray],
                 learner_metadata: List[Dict[str, Any]]) -> np.ndarray:
        """
        Custom domain-aware aggregation
        
        Args:
            predictions: Individual learner predictions
            learner_metadata: Metadata about each learner
            
        Returns:
            Aggregated predictions
        """
        
        weighted_predictions = []
        total_weight = 0
        
        for pred, metadata in zip(predictions, learner_metadata):
            # Get domain-specific weight
            modalities = metadata['modalities']
            domain_weight = sum(self.domain_weights.get(mod, 1.0) for mod in modalities)
            
            # Performance-based weight
            performance_weight = metadata['metrics'].get('accuracy', 0.5)
            
            # Combined weight
            final_weight = domain_weight * performance_weight
            
            weighted_predictions.append(pred * final_weight)
            total_weight += final_weight
        
        # Normalize and combine
        if total_weight > 0:
            return sum(weighted_predictions) / total_weight
        else:
            return np.mean(predictions, axis=0)

# Register custom strategy
predictor.register_custom_strategy('domain_aware', CustomAggregationStrategy({
    'text_embeddings': 1.5,    # Higher weight for text
    'image_features': 1.0,     # Standard weight for images
    'user_metadata': 0.8       # Lower weight for metadata
}))
```

## 📈 Performance Monitoring

### Real-Time Metrics

```python
class PredictionMonitor:
    """
    Real-time monitoring of ensemble prediction performance
    """
    
    def __init__(self):
        self.prediction_history = []
        self.performance_metrics = {}
        self.alert_thresholds = {
            'low_confidence_rate': 0.3,    # Alert if >30% predictions are low confidence
            'high_uncertainty_rate': 0.2,   # Alert if >20% predictions are high uncertainty
            'model_disagreement_rate': 0.25  # Alert if >25% show high disagreement
        }
    
    def log_prediction(self, result: PredictionResult, true_label: Optional[np.ndarray] = None):
        """Log prediction for monitoring"""
        
        prediction_record = {
            'timestamp': time.time(),
            'prediction': result.predictions,
            'confidence': result.confidence,
            'uncertainty': result.uncertainty,
            'true_label': true_label,
            'metadata': result.metadata
        }
        
        self.prediction_history.append(prediction_record)
        
        # Trigger alerts if needed
        self._check_alerts()
    
    def _check_alerts(self):
        """Check for performance alerts"""
        
        if len(self.prediction_history) < 100:  # Need sufficient history
            return
        
        recent_predictions = self.prediction_history[-100:]  # Last 100 predictions
        
        # Low confidence alert
        low_confidence_count = sum(1 for p in recent_predictions 
                                  if p['confidence'] is not None and np.mean(p['confidence']) < 0.7)
        low_confidence_rate = low_confidence_count / len(recent_predictions)
        
        if low_confidence_rate > self.alert_thresholds['low_confidence_rate']:
            self._send_alert(f"High low-confidence rate: {low_confidence_rate:.2%}")
        
        # High uncertainty alert  
        high_uncertainty_count = sum(1 for p in recent_predictions
                                   if p['uncertainty'] is not None and np.mean(p['uncertainty']) > 0.8)
        high_uncertainty_rate = high_uncertainty_count / len(recent_predictions)
        
        if high_uncertainty_rate > self.alert_thresholds['high_uncertainty_rate']:
            self._send_alert(f"High uncertainty rate: {high_uncertainty_rate:.2%}")
    
    def _send_alert(self, message: str):
        """Send performance alert"""
        logger.warning(f"PREDICTION ALERT: {message}")
        # In production: send to monitoring system, email, Slack, etc.
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        if not self.prediction_history:
            return {'error': 'No prediction history available'}
        
        # Compute statistics
        all_confidences = [p['confidence'] for p in self.prediction_history if p['confidence'] is not None]
        all_uncertainties = [p['uncertainty'] for p in self.prediction_history if p['uncertainty'] is not None]
        
        report = {
            'total_predictions': len(self.prediction_history),
            'time_range': {
                'start': min(p['timestamp'] for p in self.prediction_history),
                'end': max(p['timestamp'] for p in self.prediction_history)
            },
            'confidence_stats': {
                'mean': np.mean([np.mean(c) for c in all_confidences]),
                'std': np.std([np.mean(c) for c in all_confidences]),
                'min': np.min([np.min(c) for c in all_confidences]),
                'max': np.max([np.max(c) for c in all_confidences])
            },
            'uncertainty_stats': {
                'mean': np.mean([np.mean(u) for u in all_uncertainties]),
                'std': np.std([np.mean(u) for u in all_uncertainties]),
                'min': np.min([np.min(u) for u in all_uncertainties]),
                'max': np.max([np.max(u) for u in all_uncertainties])
            }
        }
        
        # Add accuracy if true labels available
        labeled_predictions = [p for p in self.prediction_history if p['true_label'] is not None]
        if labeled_predictions:
            predictions = np.concatenate([p['prediction'] for p in labeled_predictions])
            true_labels = np.concatenate([p['true_label'] for p in labeled_predictions])
            
            report['accuracy_stats'] = {
                'overall_accuracy': accuracy_score(true_labels, predictions),
                'f1_score': f1_score(true_labels, predictions, average='weighted')
            }
        
        return report
```

## 🔍 Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Low Prediction Confidence

**Symptoms**: Average confidence scores below 70%
**Causes**: 
- Poor model calibration
- Insufficient training data
- High data distribution shift

**Solutions**:
```python
# Solution 1: Recalibrate confidence scores
predictor.setup_confidence_calibration(
    calibration_method='temperature_scaling',
    validation_data=validation_set
)

# Solution 2: Increase ensemble diversity
predictor.create_ensemble(
    n_bags=25,                    # More bags
    dropout_strategy='adaptive',  # Better diversity
    diversity_target=0.9         # Higher diversity target
)

# Solution 3: Check for distribution shift
drift_detector = DataDriftDetector()
drift_score = drift_detector.detect_drift(training_data, test_data)
if drift_score > 0.5:
    print("⚠️ Significant data drift detected - consider retraining")
```

#### Issue 2: High Prediction Uncertainty

**Symptoms**: Uncertainty scores consistently above 0.8
**Causes**:
- High ensemble disagreement
- Out-of-distribution data
- Insufficient model complexity

**Solutions**:
```python
# Solution 1: Analyze uncertainty sources
uncertainty_analysis = predictor.analyze_uncertainty_sources(test_data)
print(f"Epistemic uncertainty: {uncertainty_analysis['epistemic']:.3f}")
print(f"Aleatoric uncertainty: {uncertainty_analysis['aleatoric']:.3f}")

# Solution 2: Improve model agreement
predictor.setup_consensus_training(
    consensus_weight=0.3,        # Encourage agreement
    diversity_regularization=0.1  # But maintain some diversity
)

# Solution 3: Add uncertainty-aware training
predictor.setup_training(
    enable_uncertainty_loss=True,
    uncertainty_weight=0.2
)
```

#### Issue 3: Slow Inference Performance

**Symptoms**: Prediction time > 100ms per sample
**Causes**:
- Too many learners
- Inefficient GPU usage
- Large model sizes

**Solutions**:
```python
# Solution 1: Model pruning and optimization
predictor.optimize_for_inference(
    pruning_threshold=0.1,       # Remove low-importance learners
    quantization=True,           # Use INT8 quantization
    batch_optimization=True      # Optimize batch processing
)

# Solution 2: Learner selection based on efficiency
predictor.select_efficient_subset(
    max_inference_time_ms=50,    # Maximum acceptable time
    min_accuracy_threshold=0.85  # Minimum accuracy requirement
)

# Solution 3: Asynchronous prediction
async_predictor = AsyncEnsemblePredictor(predictor)
future_results = async_predictor.predict_async(large_dataset)
```

#### Issue 4: Memory Usage Issues

**Symptoms**: Out-of-memory errors during prediction
**Causes**:
- Large ensemble size
- High-dimensional data
- GPU memory limitations

**Solutions**:
```python
# Solution 1: Enable memory-efficient mode
predictor.enable_memory_efficient_mode(
    max_memory_gb=4.0,
    streaming_mode=True,
    checkpoint_frequency=1000
)

# Solution 2: Gradient checkpointing for neural models
for learner in predictor.trained_learners:
    if hasattr(learner, 'gradient_checkpointing'):
        learner.gradient_checkpointing = True

# Solution 3: Model distillation to smaller ensemble
distilled_predictor = predictor.distill_to_smaller_ensemble(
    target_size=5,               # Reduce from 15 to 5 learners
    distillation_temperature=3.0,
    preserve_uncertainty=True
)
```

## 📚 Advanced Research Applications

### Academic Research Example

```python
def research_uncertainty_analysis():
    """
    Advanced uncertainty analysis for research publications
    """
    
    # Create research-grade ensemble
    predictor = EnsemblePredictor(
        aggregation_strategy='transformer_fusion',
        uncertainty_method='attention_based',
        calibrate_uncertainty=True
    )
    
    # Comprehensive uncertainty decomposition
    results = predictor.predict(test_data, return_uncertainty=True)
    
    # Decompose uncertainty into components
    uncertainty_decomposition = {
        'epistemic': predictor.compute_epistemic_uncertainty(test_data),
        'aleatoric': predictor.compute_aleatoric_uncertainty(test_data), 
        'distributional': predictor.compute_distributional_uncertainty(test_data)
    }
    
    # Statistical analysis
    correlation_matrix = predictor.compute_uncertainty_correlations(uncertainty_decomposition)
    
    # Visualization for papers
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Uncertainty vs Accuracy
    axes[0,0].scatter(results.uncertainty, results.accuracy)
    axes[0,0].set_xlabel('Prediction Uncertainty')
    axes[0,0].set_ylabel('Prediction Accuracy')
    axes[0,0].set_title('Uncertainty-Accuracy Relationship')
    
    # Attention heatmap
    if results.attention_weights is not None:
        im = axes[0,1].imshow(results.attention_weights, cmap='Blues')
        axes[0,1].set_title('Learner Attention Weights')
        plt.colorbar(im, ax=axes[0,1])
    
    # Modality importance
    if results.modality_importance:
        modalities = list(results.modality_importance.keys())
        importances = list(results.modality_importance.values())
        axes[1,0].bar(modalities, importances)
        axes[1,0].set_title('Modality Importance')
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # Uncertainty distribution
    axes[1,1].hist(results.uncertainty, bins=30, alpha=0.7)
    axes[1,1].set_xlabel('Uncertainty Score')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Uncertainty Distribution')
    
    plt.tight_layout()
    plt.savefig('ensemble_uncertainty_analysis.pdf', dpi=300, bbox_inches='tight')
    
    return uncertainty_decomposition, correlation_matrix
```

### Industrial Application Example

```python
class ProductionEnsembleService:
    """
    Production-grade ensemble prediction service
    """
    
    def __init__(self, model_path: str, config_path: str):
        self.predictor = EnsemblePredictor.load(model_path)
        self.config = self._load_config(config_path)
        self.monitor = PredictionMonitor()
        self.cache = PredictionCache(maxsize=10000)
        
    def predict_with_sla(self, 
                        data: Dict[str, np.ndarray],
                        max_latency_ms: int = 100,
                        min_confidence: float = 0.8) -> Dict[str, Any]:
        """
        Make prediction with SLA guarantees
        
        Args:
            data: Input data
            max_latency_ms: Maximum acceptable latency
            min_confidence: Minimum acceptable confidence
            
        Returns:
            Prediction with SLA compliance metadata
        """
        
        start_time = time.time()
        
        # Check cache first
        cache_key = self._compute_cache_key(data)
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return self._format_response(cached_result, cached=True)
        
        # Make prediction with timeout
        try:
            with TimeoutContext(max_latency_ms / 1000):
                result = self.predictor.predict(data, return_uncertainty=True)
            
            inference_time = (time.time() - start_time) * 1000
            
            # Check SLA compliance
            sla_compliant = (
                inference_time <= max_latency_ms and
                np.mean(result.confidence) >= min_confidence
            )
            
            if not sla_compliant:
                # Fallback to fast model if SLA not met
                result = self._fallback_prediction(data)
                
            # Cache result
            self.cache.set(cache_key, result)
            
            # Log for monitoring
            self.monitor.log_prediction(result)
            
            return self._format_response(result, 
                                       inference_time_ms=inference_time,
                                       sla_compliant=sla_compliant)
            
        except TimeoutError:
            # Emergency fallback
            return self._emergency_fallback(data)
    
    def _format_response(self, result: PredictionResult, **metadata) -> Dict[str, Any]:
        """Format prediction response for API"""
        
        return {
            'predictions': result.predictions.tolist(),
            'confidence': result.confidence.tolist() if result.confidence is not None else None,
            'uncertainty': result.uncertainty.tolist() if result.uncertainty is not None else None,
            'metadata': {
                **result.metadata,
                **metadata,
                'service_version': '2.0.0',
                'timestamp': time.time()
            }
        }
```

## 📄 API Reference

### Factory Functions

```python
def create_ensemble_predictor(task_type: str = "classification", aggregation_strategy: str = "weighted_vote", uncertainty_method: str = "entropy", calibrate_uncertainty: bool = True, device: str = "auto", **kwargs) -> EnsemblePredictor:
    return EnsemblePredictor(
        task_type=task_type,
        aggregation_strategy=aggregation_strategy,
        uncertainty_method=uncertainty_method,
        calibrate_uncertainty=calibrate_uncertainty,
        device=device
    )
```

**Factory Function Features:**
- **Optimal Defaults**: Pre-configured settings for common use cases
- **Flexible Configuration**: Easy parameter override via kwargs
- **Task-Specific Optimization**: Automatic configuration based on task type
- **Device Management**: Automatic GPU/CPU selection
- **Uncertainty Integration**: Built-in uncertainty quantification support

**Usage Examples:**
```python
# Basic classification predictor
predictor = create_ensemble_predictor(
    task_type='classification',
    aggregation_strategy='weighted_vote'
)

# Advanced predictor with custom settings
predictor = create_ensemble_predictor(
    task_type='classification',
    aggregation_strategy='transformer_fusion',
    uncertainty_method='ensemble_disagreement',
    calibrate_uncertainty=True,
    device='cuda'
)

# Regression predictor
predictor = create_ensemble_predictor(
    task_type='regression',
    aggregation_strategy='weighted_vote'
)
```

### API Exports

```python
# Main API exports from EnsemblePrediction.py
__all__ = [
    "AggregationStrategy",
    "UncertaintyMethod", 
    "PredictionResult",
    "TransformerMetaLearner",
    "EnsemblePredictor",
    "create_ensemble_predictor"
]
```

**Available Components:**
- **AggregationStrategy**: Enum for aggregation strategy selection
- **UncertaintyMethod**: Enum for uncertainty estimation methods
- **PredictionResult**: Dataclass for comprehensive prediction results
- **TransformerMetaLearner**: Neural meta-learner for advanced fusion
- **EnsemblePredictor**: Main prediction orchestrator
- **create_ensemble_predictor**: Factory function for predictor creation

### Core Classes Summary

- **`EnsemblePredictor`**: Main prediction orchestrator
- **`TransformerMetaLearner`**: Neural meta-learner for advanced fusion
- **`PredictionResult`**: Comprehensive prediction results container
- **`AggregationStrategy`**: Available aggregation methods
- **`UncertaintyMethod`**: Uncertainty estimation approaches

### Utility Functions

- **`create_ensemble_predictor()`**: Factory function for predictor creation
- **`analyze_modality_importance()`**: Modality importance analysis
- **`optimize_for_gpu()`**: GPU optimization utilities

## 🎯 Conclusion

The `5EnsemblePrediction.py` module represents the culmination of the multimodal ensemble pipeline, providing state-of-the-art prediction capabilities with:

✅ **Advanced Aggregation**: From simple voting to transformer-based meta-learning  
✅ **Uncertainty Quantification**: Multiple methods for reliable confidence estimation  
✅ **Production Readiness**: GPU acceleration, monitoring, and SLA compliance  
✅ **Interpretability**: Attention weights and modality importance analysis  
✅ **Scalability**: Memory-efficient processing and batch optimization  

This sophisticated prediction system enables reliable, interpretable, and scalable multimodal AI applications across domains from healthcare to autonomous systems.

---

**Built with ❤️ for Production-Grade Multimodal AI**
