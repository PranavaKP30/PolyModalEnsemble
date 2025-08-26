"""
performanceMetrics.py
Stage 6: Comprehensive Performance Metrics for Multimodal Ensemble Models
Implements the evaluation and benchmarking system as specified in 6PerformanceMetricsDoc.md
"""

import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score, log_loss
)

# --- Core Results Container ---

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

# --- Metrics Calculators ---

class ClassificationMetricsCalculator:
    @staticmethod
    def calculate(y_true, y_pred, y_proba=None, top_k=(1, 3, 5)):
        """
        Calculate 5 key classification metrics:
        1. Accuracy
        2. F1 Score (weighted)
        3. Precision (weighted)
        4. Recall (weighted)
        5. Balanced Accuracy
        """
        metrics = {}
        
        # 1. Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # 2. F1 Score (weighted for multi-class)
        try:
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        except Exception:
            metrics['f1_score'] = 0.0
        
        # 3. Precision (weighted for multi-class)
        try:
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        except Exception:
            metrics['precision'] = 0.0
        
        # 4. Recall (weighted for multi-class)
        try:
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        except Exception:
            metrics['recall'] = 0.0
        
        # 5. Balanced Accuracy
        try:
            from sklearn.metrics import balanced_accuracy_score
            metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        except Exception:
            # Fallback to regular accuracy if balanced_accuracy_score not available
            metrics['balanced_accuracy'] = accuracy_score(y_true, y_pred)
        
        # Additional metrics for comprehensive evaluation
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

class RegressionMetricsCalculator:
    @staticmethod
    def calculate(y_true, y_pred):
        """
        Calculate 5 key regression metrics:
        1. Mean Squared Error (MSE)
        2. Mean Absolute Error (MAE)
        3. Root Mean Squared Error (RMSE)
        4. R-squared Score (R²)
        5. Mean Absolute Percentage Error (MAPE)
        """
        metrics = {}
        
        # 1. Mean Squared Error (MSE)
        try:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
        except Exception:
            metrics['mse'] = 0.0
        
        # 2. Mean Absolute Error (MAE)
        try:
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
        except Exception:
            metrics['mae'] = 0.0
        
        # 3. Root Mean Squared Error (RMSE)
        try:
            metrics['rmse'] = np.sqrt(metrics['mse'])
        except Exception:
            metrics['rmse'] = 0.0
        
        # 4. R-squared Score (R²)
        try:
            metrics['r2_score'] = r2_score(y_true, y_pred)
        except Exception:
            metrics['r2_score'] = 0.0
        
        # 5. Mean Absolute Percentage Error (MAPE)
        try:
            # Handle division by zero and avoid extreme values
            mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10)))
            # Cap MAPE at 1000% to avoid extreme values
            metrics['mape'] = min(mape, 1000.0)
        except Exception:
            metrics['mape'] = 0.0
        
        return metrics

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

# --- Main Evaluation Engine ---

class PerformanceEvaluator:
    def __init__(self, task_type="classification"):
        self.task_type = task_type
    def evaluate_model(self, model_predict_fn: Callable, test_data: Dict[str, np.ndarray], y_true: np.ndarray, model_name: str = "", return_probabilities: bool = False, measure_efficiency: bool = True, n_efficiency_runs: int = 10) -> ModelPerformanceReport:
        """
        Robust evaluation: supports callable, attribute, or lambda for predict/predict_proba, fills all ModelPerformanceReport fields.
        """
        # Predict
        try:
            y_pred = model_predict_fn(test_data)
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            y_pred = np.zeros_like(y_true)
        
        y_proba = None
        # Try to get probabilities if possible
        if return_probabilities and self.task_type == "classification":
            try:
                if hasattr(model_predict_fn, 'predict_proba'):
                    y_proba = model_predict_fn.predict_proba(test_data)
                elif hasattr(model_predict_fn, '__self__') and hasattr(model_predict_fn.__self__, 'predict_proba'):
                    y_proba = model_predict_fn.__self__.predict_proba(test_data)
                elif hasattr(model_predict_fn, 'predict') and hasattr(model_predict_fn, 'predict_proba'):
                    y_proba = model_predict_fn.predict_proba(test_data)
            except Exception as e:
                print(f"Warning: Probability prediction failed: {e}")
                y_proba = None
        
        # Calculate metrics based on task type
        if self.task_type == "classification":
            metrics = ClassificationMetricsCalculator.calculate(y_true, y_pred, y_proba)
        else:
            metrics = RegressionMetricsCalculator.calculate(y_true, y_pred)
        
        # Efficiency metrics
        if measure_efficiency:
            try:
                inf_time, _ = EfficiencyMetricsCalculator.measure_inference_time(model_predict_fn, test_data, n_runs=n_efficiency_runs)
                throughput = EfficiencyMetricsCalculator.measure_throughput(model_predict_fn, test_data)
                mem_usage = EfficiencyMetricsCalculator.measure_memory_usage()
            except Exception as e:
                print(f"Warning: Efficiency measurement failed: {e}")
                inf_time, throughput, mem_usage = 0.0, 0.0, 0.0
        else:
            inf_time, throughput, mem_usage = 0.0, 0.0, 0.0
        
        # Initialize default values for all ModelPerformanceReport fields
        default_metrics = {
            'accuracy': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'precision': 0.0, 'recall': 0.0,
            'balanced_accuracy': 0.0, 'kappa_score': 0.0, 'top_1_accuracy': 0.0, 'top_3_accuracy': 0.0, 'top_5_accuracy': 0.0,
            'mse': 0.0, 'mae': 0.0, 'rmse': 0.0, 'r2_score': 0.0, 'mape': 0.0,
            'ece_score': 0.0, 'brier_score': 0.0, 'prediction_entropy': 0.0, 'confidence_accuracy_correlation': 0.0,
            'inference_time_ms': 0.0, 'throughput_samples_per_sec': 0.0, 'memory_usage_mb': 0.0,
            'cross_modal_consistency': 0.0, 'modality_importance': {}, 'missing_modality_robustness': {}
        }
        
        # Update with calculated metrics
        default_metrics.update(metrics)
        
        # Set efficiency metrics
        default_metrics['inference_time_ms'] = inf_time
        default_metrics['throughput_samples_per_sec'] = throughput
        default_metrics['memory_usage_mb'] = mem_usage
        
        return ModelPerformanceReport(
            model_name=model_name,
            **default_metrics
        )

# --- Model Comparison ---

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

# --- API Exports ---
__all__ = [
    "ModelPerformanceReport",
    "PerformanceEvaluator",
    "ModelComparator",
    "ClassificationMetricsCalculator",
    "RegressionMetricsCalculator",
    "MultimodalMetricsCalculator",
    "EfficiencyMetricsCalculator"
]
