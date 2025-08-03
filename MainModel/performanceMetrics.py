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
