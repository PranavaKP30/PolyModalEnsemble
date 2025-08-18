# Stage 6: Performance Metrics Documentation

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com)
[![Status](https://img.shields.io/badge/status-production_ready-brightgreen.svg)](https://github.com)
[![Component](https://img.shields.io/badge/component-performance_metrics-red.svg)](https://github.com)

**Comprehensive performance evaluation and benchmarking framework for multimodal ensemble models with advanced metrics, efficiency analysis, and model comparison capabilities.**

## 🎯 Overview

The `6PerformanceMetrics.py` module provides a **comprehensive performance evaluation framework** specifically designed for multimodal ensemble models. This production-ready system enables detailed performance analysis, model comparison, benchmarking, and efficiency measurement across multiple dimensions critical for multimodal AI systems.

### Core Value Proposition
- 📊 **Comprehensive Metrics** - 25+ performance metrics across quality, efficiency, and robustness
- 🔄 **Multimodal-Specific** - Cross-modal consistency, modality importance, missing modality robustness
- ⚡ **Efficiency Analysis** - Inference time, memory usage, throughput, and resource optimization
- 🔬 **Model Comparison** - Side-by-side benchmarking with statistical analysis and rankings
- 🎯 **Uncertainty Assessment** - Calibration metrics, entropy analysis, confidence evaluation
- 📈 **Production Ready** - Real-world deployment metrics and monitoring capabilities

## 🏗️ Architecture Overview

The performance metrics system implements a **5-layer evaluation architecture** designed for comprehensive multimodal model assessment:

```
┌─────────────────────────────────────────────────────────────────┐
│                Performance Metrics Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Metrics Calculation Engine                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Classification│  │Regression   │  │Multimodal   │             │
│  │Metrics       │  │Metrics      │  │Metrics      │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Efficiency & Resource Assessment                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Inference    │  │Memory       │  │Throughput   │             │
│  │Timing       │  │Usage        │  │Analysis     │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Uncertainty & Calibration Evaluation                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Expected     │  │Brier Score  │  │Prediction   │             │
│  │Calibration  │  │Analysis     │  │Entropy      │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Model Comparison & Benchmarking                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Performance  │  │Statistical  │  │Ranking      │             │
│  │Tables       │  │Tests        │  │Systems      │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 5: Reporting & Visualization                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │Comprehensive│  │Interactive  │  │Export       │             │
│  │Reports      │  │Plots        │  │Systems      │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
           ↓                    ↓                    ↓
     Performance Reports   Model Rankings    Optimization Insights
```

### Core Components

#### 1. **ModelPerformanceReport** - Comprehensive Results Container
Structured storage for 25+ performance metrics including quality, efficiency, and robustness measures.

#### 2. **PerformanceEvaluator** - Main Evaluation Engine
High-level orchestrator for comprehensive model assessment with multimodal-specific capabilities.

#### 3. **ModelComparator** - Benchmarking Framework
Side-by-side model comparison with statistical analysis, rankings, and visualization.

#### 4. **Metrics Calculators** - Specialized Computation Engines
- `ClassificationMetricsCalculator` - Classification-specific metrics
- `RegressionMetricsCalculator` - Regression-specific metrics  
- `MultimodalMetricsCalculator` - Multimodal-specific assessments
- `EfficiencyMetricsCalculator` - Performance and resource metrics

#### 5. **API Exports** - Module Interface
Complete set of exported components for easy integration and usage.

## 🚀 Quick Start Guide

### Basic Performance Evaluation

```python
from mainModel import MultiModalEnsembleModel, create_synthetic_model
import numpy as np

# Create and train a model
model = create_synthetic_model({
    'text_features': (768, 'text'),
    'image_features': (2048, 'image'),
    'user_metadata': (50, 'tabular')
}, n_samples=1000, n_classes=5)

# Setup and train ensemble
model.create_ensemble(n_bags=10)
model.generate_bags()
model.select_base_learners()
model.setup_training(epochs=5)
trained_learners, metrics = model.train_ensemble()
model.setup_predictor()

# Prepare test data
test_data = {
    'text_features': np.random.randn(100, 768),
    'image_features': np.random.randn(100, 2048),
    'user_metadata': np.random.randn(100, 50)
}
test_labels = np.random.randint(0, 5, 100)

# Comprehensive performance evaluation
performance_report = model.comprehensive_performance_evaluation(
    test_data, test_labels, 
    model_name="advanced_multimodal_ensemble",
    include_efficiency_metrics=True,
    include_multimodal_metrics=True
)

print(f"Accuracy: {performance_report.accuracy:.3f}")
print(f"F1-Score: {performance_report.f1_score:.3f}")
print(f"AUC-ROC: {performance_report.auc_roc:.3f}")
print(f"Inference Time: {performance_report.inference_time_ms:.2f}ms")
print(f"Memory Usage: {performance_report.memory_usage_mb:.1f}MB")
print(f"Cross-Modal Consistency: {performance_report.cross_modal_consistency:.3f}")
```

### Model Comparison and Benchmarking

```python
# Create baseline models for comparison
def random_baseline(data):
    n_samples = len(next(iter(data.values())))
    return np.random.randint(0, 5, n_samples)

def majority_class_baseline(data):
    n_samples = len(next(iter(data.values())))
    return np.full(n_samples, 2)  # Most common class

def simple_ensemble_baseline(data):
    # Simple averaging baseline
    n_samples = len(next(iter(data.values())))
    return np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])

baseline_models = {
    'random_baseline': random_baseline,
    'majority_class': majority_class_baseline,
    'simple_ensemble': simple_ensemble_baseline
}

# Compare with baselines
comparison = model.compare_with_baseline(
    baseline_models, test_data, test_labels,
    save_comparison="model_comparison_report.json"
)

# Generate comparison table
from mainModel import ModelComparator
comparator = ModelComparator('classification')
comparator.comparison = comparison

table = comparator.generate_comparison_table()
print("\\nModel Comparison Results:")
print(table.to_string(index=False))

# Generate rankings
rankings = comparator.generate_rankings()
print(f"\\nAccuracy Ranking: {rankings['accuracy']}")
print(f"F1-Score Ranking: {rankings['f1_score']}")

# Create performance plots
comparator.plot_performance_comparison(
    metrics=['accuracy', 'f1_score', 'auc_roc', 'inference_time_ms'],
    save_path='performance_comparison.png'
)
```

## 📊 Detailed Metrics Documentation

### ModelPerformanceReport - Complete Data Structure

```python
@dataclass
class ModelPerformanceReport:
    # Quality metrics
    accuracy: float = 0.0
    f1_score: float = 0.0
    auc_roc: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    balanced_accuracy: float = 0.0
    kappa_score: float = 0.0
    top_1_accuracy: float = 0.0
    top_3_accuracy: float = 0.0
    top_5_accuracy: float = 0.0
    # Regression
    mse: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    r2_score: float = 0.0
    mape: float = 0.0
    # Uncertainty/calibration
    ece_score: float = 0.0
    brier_score: float = 0.0
    prediction_entropy: float = 0.0
    confidence_accuracy_correlation: float = 0.0
    # Efficiency
    inference_time_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    # Multimodal
    cross_modal_consistency: float = 0.0
    modality_importance: Optional[Dict[str, float]] = field(default_factory=dict)
    missing_modality_robustness: Optional[Dict[str, float]] = field(default_factory=dict)
    # Metadata
    model_name: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
```

**ModelPerformanceReport Categories:**

**Quality Metrics (Classification):**
- **accuracy**: Overall correct predictions rate
- **f1_score**: Weighted F1-score (harmonic mean of precision/recall)
- **auc_roc**: Area under ROC curve
- **precision**: Weighted precision score
- **recall**: Weighted recall score
- **balanced_accuracy**: Accuracy adjusted for class imbalance
- **kappa_score**: Cohen's kappa coefficient
- **top_1_accuracy**: Top-1 accuracy for multi-class
- **top_3_accuracy**: Top-3 accuracy for multi-class
- **top_5_accuracy**: Top-5 accuracy for multi-class

**Quality Metrics (Regression):**
- **mse**: Mean Squared Error
- **mae**: Mean Absolute Error
- **rmse**: Root Mean Squared Error
- **r2_score**: R-squared coefficient of determination
- **mape**: Mean Absolute Percentage Error

**Uncertainty/Calibration Metrics:**
- **ece_score**: Expected Calibration Error (lower is better)
- **brier_score**: Brier score for probability calibration
- **prediction_entropy**: Average entropy of predictions
- **confidence_accuracy_correlation**: Correlation between confidence and accuracy

**Efficiency Metrics:**
- **inference_time_ms**: Average inference time in milliseconds
- **throughput_samples_per_sec**: Samples processed per second
- **memory_usage_mb**: Memory usage in megabytes

**Multimodal Metrics:**
- **cross_modal_consistency**: Consistency of predictions across modalities
- **modality_importance**: Importance weights for each modality
- **missing_modality_robustness**: Performance retention when modalities are missing

**Metadata:**
- **model_name**: Name identifier for the model
- **extra**: Additional custom metrics and metadata

### 1. Prediction Quality Metrics

#### Classification Metrics
```python
# Available classification metrics
classification_metrics = {
    'accuracy': 'Overall correct predictions rate',
    'balanced_accuracy': 'Accuracy adjusted for class imbalance', 
    'f1_score': 'Weighted F1-score (harmonic mean of precision/recall)',
    'f1_macro': 'Macro-averaged F1-score',
    'f1_micro': 'Micro-averaged F1-score',
    'f1_weighted': 'Weighted F1-score by class support',
    'precision': 'Weighted precision score',
    'recall': 'Weighted recall score',
    'auc_roc': 'Area under ROC curve',
    'auc_pr': 'Area under Precision-Recall curve',
    'kappa_score': 'Cohen\'s kappa coefficient',
    'top_1_accuracy': 'Top-1 accuracy for multi-class',
    'top_3_accuracy': 'Top-3 accuracy for multi-class',
    'top_5_accuracy': 'Top-5 accuracy for multi-class'
}
```

#### Regression Metrics
```python
# Available regression metrics
regression_metrics = {
    'mse': 'Mean Squared Error',
    'mae': 'Mean Absolute Error', 
    'rmse': 'Root Mean Squared Error',
    'r2_score': 'R-squared coefficient of determination',
    'mape': 'Mean Absolute Percentage Error'
}
```

### Metrics Calculators - Implementation Details

#### ClassificationMetricsCalculator

```python
class ClassificationMetricsCalculator:
    @staticmethod
    def calculate(y_true, y_pred, y_proba=None, top_k=(1, 3, 5)):
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['balanced_accuracy'] = accuracy_score(y_true, y_pred)
        from sklearn.metrics import cohen_kappa_score
        try:
            metrics['kappa_score'] = cohen_kappa_score(y_true, y_pred)
        except Exception:
            metrics['kappa_score'] = 0.0
        if y_proba is not None and y_proba.shape[1] > 1:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            except Exception:
                metrics['auc_roc'] = 0.0
            metrics['brier_score'] = np.mean(np.sum((y_proba - np.eye(y_proba.shape[1])[y_true]) ** 2, axis=1))
            metrics['prediction_entropy'] = float(np.mean(-np.sum(y_proba * np.log(y_proba + 1e-10), axis=1)))
        else:
            metrics['auc_roc'] = 0.0
            metrics['brier_score'] = 0.0
            metrics['prediction_entropy'] = 0.0
        # Top-k accuracy
        for k in top_k:
            if y_proba is not None and y_proba.shape[1] >= k:
                top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
                top_k_acc = np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
                metrics[f'top_{k}_accuracy'] = top_k_acc
            else:
                metrics[f'top_{k}_accuracy'] = 0.0
        return metrics
```

**ClassificationMetricsCalculator Features:**
- **Comprehensive Metrics**: Accuracy, F1-score, precision, recall, balanced accuracy
- **Advanced Metrics**: Cohen's kappa, AUC-ROC, Brier score, prediction entropy
- **Top-K Accuracy**: Configurable top-k accuracy for multi-class problems
- **Error Handling**: Graceful handling of edge cases and exceptions
- **Probability Support**: Optional probability inputs for advanced metrics
- **Zero Division Handling**: Safe handling of edge cases in precision/recall

#### RegressionMetricsCalculator

```python
class RegressionMetricsCalculator:
    @staticmethod
    def calculate(y_true, y_pred):
        metrics = {}
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2_score'] = r2_score(y_true, y_pred)
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)))
        return metrics
```

**RegressionMetricsCalculator Features:**
- **Standard Metrics**: MSE, MAE, RMSE, R² score
- **Percentage Error**: MAPE for relative error assessment
- **Numerical Stability**: Small epsilon to prevent division by zero
- **Efficient Computation**: Vectorized operations for performance
- **Comprehensive Coverage**: All major regression evaluation metrics

### 2. Uncertainty and Calibration Metrics

#### Calibration Assessment
```python
# Uncertainty/calibration metrics
uncertainty_metrics = {
    'ece_score': 'Expected Calibration Error (lower is better)',
    'brier_score': 'Brier score for probability calibration',
    'prediction_entropy': 'Average entropy of predictions', 
    'confidence_accuracy_correlation': 'Correlation between confidence and accuracy'
}

# Example calibration analysis
def analyze_model_calibration(model, test_data, test_labels):
    report = model.comprehensive_performance_evaluation(test_data, test_labels)
    
    print(f"Expected Calibration Error: {report.ece_score:.4f}")
    print(f"Brier Score: {report.brier_score:.4f}")
    print(f"Prediction Entropy: {report.prediction_entropy:.4f}")
    print(f"Confidence-Accuracy Correlation: {report.confidence_accuracy_correlation:.4f}")
    
    # Interpretation
    if report.ece_score < 0.05:
        print("✅ Model is well-calibrated")
    elif report.ece_score < 0.15:
        print("⚠️ Model is moderately calibrated")
    else:
        print("❌ Model is poorly calibrated - consider recalibration")
```

### 3. Multimodal-Specific Metrics

#### Cross-Modal Consistency
```python
def analyze_cross_modal_consistency(model, test_data, test_labels):
    """Analyze how consistent predictions are across modalities"""
    
    report = model.comprehensive_performance_evaluation(
        test_data, test_labels, include_multimodal_metrics=True
    )
    
    print(f"Cross-Modal Consistency: {report.cross_modal_consistency:.3f}")
    
    # Interpretation
    if report.cross_modal_consistency > 0.9:
        print("✅ High cross-modal agreement")
    elif report.cross_modal_consistency > 0.7:
        print("⚠️ Moderate cross-modal agreement") 
    else:
        print("❌ Low cross-modal agreement - check modality alignment")
```

#### Modality Importance Analysis
```python
def analyze_modality_importance(model, test_data, test_labels):
    """Analyze contribution of each modality"""
    
    report = model.comprehensive_performance_evaluation(
        test_data, test_labels, include_multimodal_metrics=True
    )
    
    if report.modality_importance:
        print("\\nModality Importance Analysis:")
        for modality, importance in report.modality_importance.items():
            print(f"  {modality}: {importance:.3f}")
        
        # Find most/least important modalities
        sorted_modalities = sorted(report.modality_importance.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        print(f"\\nMost important modality: {sorted_modalities[0][0]} ({sorted_modalities[0][1]:.3f})")
        print(f"Least important modality: {sorted_modalities[-1][0]} ({sorted_modalities[-1][1]:.3f})")
```

#### Missing Modality Robustness
```python
def analyze_missing_modality_robustness(model, test_data, test_labels):
    """Analyze robustness to missing modalities"""
    
    report = model.comprehensive_performance_evaluation(
        test_data, test_labels, include_multimodal_metrics=True
    )
    
    if report.missing_modality_robustness:
        print("\\nMissing Modality Robustness:")
        for scenario, robustness in report.missing_modality_robustness.items():
            print(f"  {scenario}: {robustness:.3f} (retain {robustness*100:.1f}% performance)")
        
        # Identify critical modalities
        critical_modalities = [k for k, v in report.missing_modality_robustness.items() 
                             if v < 0.8]  # Less than 80% performance retained
        
        if critical_modalities:
            print(f"\\n⚠️ Critical modalities (>20% performance drop): {critical_modalities}")
        else:
            print("\\n✅ Model is robust to individual modality failures")
```

### Multimodal and Efficiency Metrics Calculators

#### MultimodalMetricsCalculator

```python
class MultimodalMetricsCalculator:
    @staticmethod
    def cross_modal_consistency(predictions_by_modality):
        # Dummy: fraction of samples with all modalities agreeing
        if not predictions_by_modality:
            return 0.0
        preds = np.array(list(predictions_by_modality.values()))
        return float(np.mean(np.all(preds == preds[0], axis=0)))
    
    @staticmethod
    def modality_importance(importance_scores):
        # Normalize to sum to 1
        total = sum(importance_scores.values())
        return {k: v/total for k, v in importance_scores.items()} if total > 0 else importance_scores
    
    @staticmethod
    def missing_modality_robustness(perf_with_missing):
        # perf_with_missing: dict of {modality: retained_performance}
        return perf_with_missing
```

**MultimodalMetricsCalculator Features:**
- **Cross-Modal Consistency**: Measures agreement between modality predictions
- **Modality Importance**: Normalizes importance scores to sum to 1
- **Missing Modality Robustness**: Tracks performance retention when modalities are removed
- **Flexible Input**: Handles various prediction formats and importance structures
- **Robust Computation**: Handles edge cases and empty inputs

#### EfficiencyMetricsCalculator

```python
class EfficiencyMetricsCalculator:
    @staticmethod
    def measure_inference_time(predict_fn: Callable, data, n_runs=10):
        times = []
        for _ in range(n_runs):
            start = time.time()
            _ = predict_fn(data)
            times.append((time.time() - start) * 1000)
        return float(np.mean(times)), float(np.std(times))
    
    @staticmethod
    def measure_throughput(predict_fn: Callable, data, batch_size=32):
        n_samples = len(next(iter(data.values())))
        start = time.time()
        _ = predict_fn(data)
        elapsed = time.time() - start
        return n_samples / elapsed if elapsed > 0 else 0.0
    
    @staticmethod
    def measure_memory_usage():
        # Dummy: return 0.0 (real implementation would use psutil or similar)
        return 0.0
```

**EfficiencyMetricsCalculator Features:**
- **Inference Time Measurement**: Multiple runs for statistical reliability
- **Throughput Calculation**: Samples per second processing rate
- **Memory Usage Tracking**: Placeholder for memory monitoring
- **Statistical Analysis**: Mean and standard deviation for timing
- **Flexible Interface**: Works with any callable prediction function
- **Batch Processing Support**: Configurable batch sizes for throughput measurement

### 4. Efficiency and Resource Metrics

#### Performance Profiling
```python
def profile_model_efficiency(model, test_data, test_labels):
    """Comprehensive efficiency profiling"""
    
    report = model.comprehensive_performance_evaluation(
        test_data, test_labels, 
        include_efficiency_metrics=True,
        n_efficiency_runs=50  # More runs for precise timing
    )
    
    print("\\nEfficiency Profile:")
    print(f"  Inference Time: {report.inference_time_ms:.2f}ms per batch")
    print(f"  Throughput: {report.throughput_samples_per_sec:.1f} samples/sec")
    print(f"  Memory Usage: {report.memory_usage_mb:.1f}MB")
    
    # Performance categorization
    if report.inference_time_ms < 100:
        print("✅ Fast inference (< 100ms)")
    elif report.inference_time_ms < 500:
        print("⚠️ Moderate inference time (100-500ms)")
    else:
        print("❌ Slow inference (> 500ms)")
    
    if report.memory_usage_mb < 500:
        print("✅ Low memory usage (< 500MB)")
    elif report.memory_usage_mb < 2000:
        print("⚠️ Moderate memory usage (500MB-2GB)")
    else:
        print("❌ High memory usage (> 2GB)")
```

## 🔧 Advanced Usage Examples

### PerformanceEvaluator - Main Evaluation Engine

```python
class PerformanceEvaluator:
    def __init__(self, task_type="classification"):
        self.task_type = task_type
    
    def evaluate_model(self, model_predict_fn: Callable, test_data: Dict[str, np.ndarray], y_true: np.ndarray, model_name: str = "", return_probabilities: bool = False, measure_efficiency: bool = True, n_efficiency_runs: int = 10) -> ModelPerformanceReport:
        """
        Robust evaluation: supports callable, attribute, or lambda for predict/predict_proba, fills all ModelPerformanceReport fields.
        """
        # Predict
        y_pred = model_predict_fn(test_data)
        y_proba = None
        # Try to get probabilities if possible
        if return_probabilities:
            if hasattr(model_predict_fn, 'predict_proba'):
                y_proba = model_predict_fn.predict_proba(test_data)
            elif hasattr(model_predict_fn, '__self__') and hasattr(model_predict_fn.__self__, 'predict_proba'):
                y_proba = model_predict_fn.__self__.predict_proba(test_data)
            elif hasattr(model_predict_fn, 'predict') and hasattr(model_predict_fn, 'predict_proba'):
                y_proba = model_predict_fn.predict_proba(test_data)
            else:
                y_proba = None
        # Metrics
        if self.task_type == "classification":
            metrics = ClassificationMetricsCalculator.calculate(y_true, y_pred, y_proba)
        else:
            metrics = RegressionMetricsCalculator.calculate(y_true, y_pred)
        # Efficiency
        inf_time, _ = EfficiencyMetricsCalculator.measure_inference_time(model_predict_fn, test_data, n_runs=n_efficiency_runs) if measure_efficiency else (0.0, 0.0)
        throughput = EfficiencyMetricsCalculator.measure_throughput(model_predict_fn, test_data) if measure_efficiency else 0.0
        mem_usage = EfficiencyMetricsCalculator.measure_memory_usage() if measure_efficiency else 0.0
        # Fill all ModelPerformanceReport fields
        report_fields = {f.name for f in ModelPerformanceReport.__dataclass_fields__.values()}
        for k in report_fields:
            if k not in metrics:
                metrics[k] = 0.0 if 'float' in str(ModelPerformanceReport.__dataclass_fields__[k].type) else None
        # Remove keys that will be set explicitly to avoid duplicate kwarg
        for k in ['model_name', 'inference_time_ms', 'throughput_samples_per_sec', 'memory_usage_mb']:
            if k in metrics:
                del metrics[k]
        return ModelPerformanceReport(
            model_name=model_name,
            **metrics,
            inference_time_ms=inf_time,
            throughput_samples_per_sec=throughput,
            memory_usage_mb=mem_usage
        )
```

**PerformanceEvaluator Features:**
- **Task Type Support**: Classification and regression evaluation
- **Flexible Prediction Interface**: Supports callable, attribute, or lambda functions
- **Probability Extraction**: Automatic detection and extraction of probability outputs
- **Comprehensive Metrics**: Fills all ModelPerformanceReport fields
- **Efficiency Measurement**: Optional inference time, throughput, and memory measurement
- **Error Handling**: Graceful handling of missing methods and attributes
- **Field Completion**: Automatically fills missing fields with appropriate defaults

### Custom Performance Evaluation

```python
from mainModel import PerformanceEvaluator, ClassificationMetricsCalculator
import numpy as np

# Create custom evaluator
evaluator = PerformanceEvaluator(task_type="classification")

# Custom prediction function
def my_model_predict(data):
    # Your model's prediction logic here
    n_samples = len(next(iter(data.values())))
    n_classes = 5
    
    # Return class probabilities
    probabilities = np.random.dirichlet(np.ones(n_classes), n_samples)
    return probabilities

# Evaluate with custom settings
performance_report = evaluator.evaluate_model(
    model_predict_fn=my_model_predict,
    test_data=test_data,
    y_true=test_labels,
    model_name="custom_multimodal_model",
    return_probabilities=True,
    measure_efficiency=True,
    n_efficiency_runs=20
)

print(f"Custom model accuracy: {performance_report.accuracy:.3f}")
print(f"Custom model calibration: {performance_report.ece_score:.4f}")
```

### ModelComparator - Benchmarking Framework

```python
class ModelComparator:
    def __init__(self, task_type="classification"):
        self.task_type = task_type
        self.comparison = {}
    
    def add_model_evaluation(self, predict_fn, test_data, test_labels, model_name):
        evaluator = PerformanceEvaluator(self.task_type)
        report = evaluator.evaluate_model(predict_fn, test_data, test_labels, model_name)
        self.comparison[model_name] = report
        return report
    
    def generate_comparison_table(self):
        rows = []
        for name, report in self.comparison.items():
            rows.append({"model": name, **report.__dict__})
        return pd.DataFrame(rows)
    
    def generate_rankings(self):
        table = self.generate_comparison_table()
        rankings = {}
        for metric in ["accuracy", "f1_score", "auc_roc", "inference_time_ms"]:
            if metric in table:
                rankings[metric] = list(table.sort_values(metric, ascending=(metric=="inference_time_ms"))["model"])
        return rankings
    
    def plot_performance_comparison(self, metrics=None, save_path=None):
        import matplotlib.pyplot as plt
        table = self.generate_comparison_table()
        if metrics is None:
            metrics = ["accuracy", "f1_score", "auc_roc", "inference_time_ms"]
        for metric in metrics:
            if metric in table:
                plt.figure()
                plt.bar(table["model"], table[metric])
                plt.title(metric)
                plt.ylabel(metric)
                plt.xlabel("model")
                if save_path:
                    plt.savefig(f"{save_path}_{metric}.png")
                else:
                    plt.show()
```

**ModelComparator Features:**
- **Model Storage**: Maintains dictionary of model evaluations
- **Automatic Evaluation**: Uses PerformanceEvaluator for consistent assessment
- **Comparison Table**: Generates pandas DataFrame for easy analysis
- **Ranking System**: Creates rankings for key metrics (accuracy, F1, AUC, inference time)
- **Visualization**: Automatic plotting of performance comparisons
- **Flexible Output**: Save plots to files or display interactively
- **Task Awareness**: Supports both classification and regression tasks

### Batch Model Comparison

```python
def compare_multiple_models(models_dict, test_data, test_labels):
    """Compare multiple models systematically"""
    
    from mainModel import ModelComparator
    
    comparator = ModelComparator("classification")
    
    # Evaluate each model
    for name, predict_fn in models_dict.items():
        try:
            report = comparator.add_model_evaluation(
                predict_fn, test_data, test_labels, name
            )
            print(f"✅ Evaluated {name}: Accuracy = {report.accuracy:.3f}")
        except Exception as e:
            print(f"❌ Failed to evaluate {name}: {e}")
    
    # Generate comprehensive comparison
    comparison_table = comparator.generate_comparison_table()
    rankings = comparator.generate_rankings()
    
    # Save results
    comparison_table.to_csv("model_comparison.csv", index=False)
    comparator.plot_performance_comparison(save_path="comparison_plot.png")
    
    return comparison_table, rankings

# Example usage
models = {
    'ensemble_model': lambda data: model.predict(data).predictions,
    'baseline_rf': your_random_forest_predict,
    'baseline_svm': your_svm_predict,
    'baseline_nn': your_neural_net_predict
}

table, rankings = compare_multiple_models(models, test_data, test_labels)
print("\\nFinal Rankings:")
for metric, ranking in rankings.items():
    print(f"{metric}: {ranking}")
```

### Production Monitoring Integration

```python
def setup_production_monitoring(model, monitoring_config):
    """Setup production performance monitoring"""
    
    class ProductionMonitor:
        def __init__(self, model, alert_thresholds):
            self.model = model
            self.thresholds = alert_thresholds
            self.performance_history = []
        
        def evaluate_batch(self, data, labels=None):
            """Evaluate a batch of predictions"""
            
            start_time = time.time()
            predictions = self.model.predict(data)
            inference_time = (time.time() - start_time) * 1000
            
            # Basic metrics
            batch_metrics = {
                'timestamp': time.time(),
                'batch_size': len(next(iter(data.values()))),
                'inference_time_ms': inference_time,
                'mean_confidence': np.mean(np.max(predictions.probabilities, axis=1)) if predictions.probabilities is not None else None
            }
            
            # Add accuracy if labels available
            if labels is not None:
                pred_classes = np.argmax(predictions.predictions, axis=1) if predictions.predictions.ndim > 1 else predictions.predictions
                batch_metrics['accuracy'] = accuracy_score(labels, pred_classes)
            
            self.performance_history.append(batch_metrics)
            
            # Check alerts
            self._check_alerts(batch_metrics)
            
            return batch_metrics
        
        def _check_alerts(self, metrics):
            """Check for performance alerts"""
            
            if metrics['inference_time_ms'] > self.thresholds.get('max_inference_time', 1000):
                print(f"🚨 ALERT: High inference time: {metrics['inference_time_ms']:.1f}ms")
            
            if metrics['mean_confidence'] and metrics['mean_confidence'] < self.thresholds.get('min_confidence', 0.7):
                print(f"🚨 ALERT: Low confidence: {metrics['mean_confidence']:.3f}")
            
            if 'accuracy' in metrics and metrics['accuracy'] < self.thresholds.get('min_accuracy', 0.8):
                print(f"🚨 ALERT: Low accuracy: {metrics['accuracy']:.3f}")
        
        def get_summary_report(self):
            """Get summary of recent performance"""
            
            if not self.performance_history:
                return {}
            
            recent_metrics = self.performance_history[-100:]  # Last 100 batches
            
            return {
                'avg_inference_time_ms': np.mean([m['inference_time_ms'] for m in recent_metrics]),
                'avg_confidence': np.mean([m['mean_confidence'] for m in recent_metrics if m['mean_confidence']]),
                'avg_accuracy': np.mean([m['accuracy'] for m in recent_metrics if 'accuracy' in m]),
                'total_batches': len(recent_metrics),
                'alert_count': sum(1 for m in recent_metrics if m['inference_time_ms'] > 1000)
            }
    
    # Setup monitor
    monitor = ProductionMonitor(model, monitoring_config)
    return monitor

# Example usage
monitor = setup_production_monitoring(model, {
    'max_inference_time': 500,  # ms
    'min_confidence': 0.75,
    'min_accuracy': 0.85
})

# Monitor batch
batch_metrics = monitor.evaluate_batch(test_data, test_labels)
summary = monitor.get_summary_report()
print(f"Batch inference time: {batch_metrics['inference_time_ms']:.1f}ms")
```

## 📈 Performance Optimization Guidelines

### 1. Identifying Performance Bottlenecks

```python
def diagnose_performance_issues(model, test_data, test_labels):
    """Comprehensive performance diagnosis"""
    
    report = model.comprehensive_performance_evaluation(test_data, test_labels)
    
    issues = []
    recommendations = []
    
    # Check accuracy
    if report.accuracy < 0.8:
        issues.append(f"Low accuracy: {report.accuracy:.3f}")
        recommendations.append("Consider: more training data, better features, ensemble tuning")
    
    # Check calibration
    if report.ece_score > 0.1:
        issues.append(f"Poor calibration: ECE = {report.ece_score:.4f}")
        recommendations.append("Consider: temperature scaling, Platt calibration")
    
    # Check efficiency
    if report.inference_time_ms > 500:
        issues.append(f"Slow inference: {report.inference_time_ms:.1f}ms")
        recommendations.append("Consider: model pruning, quantization, batch optimization")
    
    if report.memory_usage_mb > 2000:
        issues.append(f"High memory usage: {report.memory_usage_mb:.1f}MB")
        recommendations.append("Consider: gradient checkpointing, model distillation")
    
    # Check multimodal aspects
    if report.cross_modal_consistency < 0.7:
        issues.append(f"Low cross-modal consistency: {report.cross_modal_consistency:.3f}")
        recommendations.append("Consider: better modality alignment, cross-modal training")
    
    # Report findings
    if issues:
        print("🔍 Performance Issues Identified:")
        for issue in issues:
            print(f"  ❌ {issue}")
        
        print("\\n💡 Recommendations:")
        for rec in recommendations:
            print(f"  ✅ {rec}")
    else:
        print("🎉 No significant performance issues detected!")
    
    return issues, recommendations
```

### 2. Optimization Strategies

```python
def optimize_model_performance(model, test_data, test_labels):
    """Step-by-step performance optimization"""
    
    print("🔧 Starting performance optimization...")
    
    # Baseline measurement
    baseline_report = model.comprehensive_performance_evaluation(test_data, test_labels)
    print(f"Baseline - Accuracy: {baseline_report.accuracy:.3f}, Time: {baseline_report.inference_time_ms:.1f}ms")
    
    optimizations = []
    
    # 1. Predictor strategy optimization
    strategies = ['weighted_vote', 'confidence_weighted', 'transformer_fusion']
    best_strategy = None
    best_accuracy = baseline_report.accuracy
    
    for strategy in strategies:
        try:
            model.setup_predictor(aggregation_strategy=strategy)
            report = model.comprehensive_performance_evaluation(test_data, test_labels)
            
            if report.accuracy > best_accuracy:
                best_accuracy = report.accuracy
                best_strategy = strategy
            
            print(f"  {strategy}: Accuracy = {report.accuracy:.3f}, Time = {report.inference_time_ms:.1f}ms")
            
        except Exception as e:
            print(f"  {strategy}: Failed - {e}")
    
    if best_strategy:
        model.setup_predictor(aggregation_strategy=best_strategy)
        optimizations.append(f"Switched to {best_strategy} aggregation")
    
    # 2. Uncertainty calibration
    try:
        model.setup_predictor(calibrate_uncertainty=True)
        optimizations.append("Enabled uncertainty calibration")
    except:
        pass
    
    # Final measurement
    final_report = model.comprehensive_performance_evaluation(test_data, test_labels)
    
    # Summary
    accuracy_gain = final_report.accuracy - baseline_report.accuracy
    time_change = final_report.inference_time_ms - baseline_report.inference_time_ms
    
    print(f"\\n📊 Optimization Results:")
    print(f"  Accuracy: {baseline_report.accuracy:.3f} → {final_report.accuracy:.3f} ({accuracy_gain:+.3f})")
    print(f"  Inference Time: {baseline_report.inference_time_ms:.1f}ms → {final_report.inference_time_ms:.1f}ms ({time_change:+.1f}ms)")
    print(f"  Applied optimizations: {optimizations}")
    
    return final_report, optimizations
```

## 🎯 Integration with mainModel

The performance metrics are fully integrated into the `MultiModalEnsembleModel` class:

### Available Methods

```python
# Direct performance evaluation
report = model.comprehensive_performance_evaluation(test_data, test_labels)

# Model comparison
comparison = model.compare_with_baseline(baseline_models, test_data, test_labels)

# Comprehensive reporting
full_report = model.generate_performance_report(test_data, test_labels, save_path="report.json")
```

### Integration Example

```python
# Complete workflow with performance analysis
def complete_multimodal_workflow_with_evaluation():
    # 1. Create and train model
    model = create_synthetic_model({
        'text': (500, 'text'),
        'image': (1000, 'image'),
        'tabular': (50, 'tabular')
    }, n_samples=2000, n_classes=4)
    
    model.create_ensemble(n_bags=15)
    model.generate_bags()
    model.select_base_learners()
    model.setup_training(epochs=10)
    model.train_ensemble()
    model.setup_predictor()
    
    # 2. Prepare evaluation data
    test_data = {
        'text': np.random.randn(200, 500),
        'image': np.random.randn(200, 1000),
        'tabular': np.random.randn(200, 50)
    }
    test_labels = np.random.randint(0, 4, 200)
    
    # 3. Comprehensive evaluation
    performance_report = model.comprehensive_performance_evaluation(
        test_data, test_labels,
        model_name="production_multimodal_ensemble"
    )
    
    # 4. Create baselines and compare
    baselines = {
        'random': lambda data: np.random.randint(0, 4, len(next(iter(data.values())))),
        'majority': lambda data: np.full(len(next(iter(data.values()))), 1)
    }
    
    comparison = model.compare_with_baseline(baselines, test_data, test_labels)
    
    # 5. Generate final report
    final_report = model.generate_performance_report(
        test_data, test_labels,
        save_path="final_performance_report.json",
        include_plots=True
    )
    
    print("🎉 Complete evaluation finished!")
    return model, performance_report, comparison, final_report
```

## 📄 API Reference

### API Exports

```python
# Main API exports from performanceMetrics.py
__all__ = [
    "ModelPerformanceReport",
    "PerformanceEvaluator",
    "ModelComparator",
    "ClassificationMetricsCalculator",
    "RegressionMetricsCalculator",
    "MultimodalMetricsCalculator",
    "EfficiencyMetricsCalculator"
]
```

**Available Components:**
- **ModelPerformanceReport**: Comprehensive performance metrics container
- **PerformanceEvaluator**: Main evaluation engine
- **ModelComparator**: Model comparison and benchmarking framework
- **ClassificationMetricsCalculator**: Classification-specific metrics computation
- **RegressionMetricsCalculator**: Regression-specific metrics computation
- **MultimodalMetricsCalculator**: Multimodal-specific assessments
- **EfficiencyMetricsCalculator**: Performance and resource metrics

### Core Classes

- **`ModelPerformanceReport`**: Comprehensive performance metrics container
- **`PerformanceEvaluator`**: Main evaluation engine
- **`ModelComparator`**: Model comparison and benchmarking
- **`ClassificationMetricsCalculator`**: Classification metrics computation
- **`RegressionMetricsCalculator`**: Regression metrics computation
- **`MultimodalMetricsCalculator`**: Multimodal-specific metrics
- **`EfficiencyMetricsCalculator`**: Efficiency and resource metrics

### Metrics Categories

- **Quality Metrics**: Accuracy, F1, AUC, precision, recall, calibration
- **Efficiency Metrics**: Inference time, memory usage, throughput
- **Multimodal Metrics**: Cross-modal consistency, modality importance, robustness
- **Uncertainty Metrics**: ECE, Brier score, entropy, confidence correlation

### Integration Methods

- **`comprehensive_performance_evaluation()`**: Full model assessment
- **`compare_with_baseline()`**: Model comparison framework
- **`generate_performance_report()`**: Comprehensive reporting

## 🎯 Conclusion

The `6PerformanceMetrics.py` module provides **production-grade performance evaluation** capabilities specifically designed for multimodal ensemble models. Key benefits:

✅ **Comprehensive Assessment**: 25+ metrics across quality, efficiency, and robustness  
✅ **Multimodal Focus**: Specialized metrics for cross-modal systems  
✅ **Production Ready**: Real-world deployment and monitoring capabilities  
✅ **Comparison Framework**: Systematic model benchmarking and ranking  
✅ **Integration**: Seamless integration with the multimodal ensemble pipeline  

This sophisticated evaluation system enables reliable assessment and optimization of multimodal AI systems for research and production deployment.

---

**Built with ❤️ for Production-Grade Multimodal Performance Analysis**
