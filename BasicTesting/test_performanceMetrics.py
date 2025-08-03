"""
test_performanceMetrics.py
Unit tests for performanceMetrics.py (Stage 6)
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MainModel import performanceMetrics

# Dummy classifier for testing
class DummyClassifier:
    def __init__(self, y_pred, y_proba=None):
        self._y_pred = y_pred
        self._y_proba = y_proba
    def __call__(self, X):
        return self._y_pred
    def predict_proba(self, X):
        return self._y_proba

def test_classification_metrics_calculator():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
        [0.7, 0.3],
        [0.6, 0.4],
        [0.2, 0.8],
    ])
    metrics = performanceMetrics.ClassificationMetricsCalculator.calculate(y_true, y_pred, y_proba)
    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert 0.0 <= metrics['f1_score'] <= 1.0
    assert 0.0 <= metrics['precision'] <= 1.0
    assert 0.0 <= metrics['recall'] <= 1.0
    assert 0.0 <= metrics['auc_roc'] <= 1.0
    assert 'top_1_accuracy' in metrics
    assert 'top_3_accuracy' in metrics
    assert 'top_5_accuracy' in metrics

def test_regression_metrics_calculator():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    metrics = performanceMetrics.RegressionMetricsCalculator.calculate(y_true, y_pred)
    assert metrics['mse'] >= 0.0
    assert metrics['mae'] >= 0.0
    assert metrics['rmse'] >= 0.0
    assert -1.0 <= metrics['r2_score'] <= 1.0
    assert metrics['mape'] >= 0.0

def test_multimodal_metrics_calculator():
    preds = {'text': np.array([0, 1, 1]), 'image': np.array([0, 1, 1])}
    consistency = performanceMetrics.MultimodalMetricsCalculator.cross_modal_consistency(preds)
    assert 0.0 <= consistency <= 1.0
    importance = performanceMetrics.MultimodalMetricsCalculator.modality_importance({'text': 0.7, 'image': 0.3})
    assert abs(sum(importance.values()) - 1.0) < 1e-6
    robustness = performanceMetrics.MultimodalMetricsCalculator.missing_modality_robustness({'text': 0.9, 'image': 0.8})
    assert 'text' in robustness and 'image' in robustness

def test_efficiency_metrics_calculator():
    def dummy_predict(X):
        return np.zeros(len(next(iter(X.values()))))
    data = {'text': np.zeros((10, 5))}
    inf_time, inf_std = performanceMetrics.EfficiencyMetricsCalculator.measure_inference_time(dummy_predict, data, n_runs=2)
    assert inf_time >= 0.0
    throughput = performanceMetrics.EfficiencyMetricsCalculator.measure_throughput(dummy_predict, data)
    assert throughput >= 0.0
    mem = performanceMetrics.EfficiencyMetricsCalculator.measure_memory_usage()
    assert mem == 0.0

def test_performance_evaluator_classification():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
        [0.7, 0.3],
        [0.6, 0.4],
        [0.2, 0.8],
    ])
    clf = DummyClassifier(y_pred, y_proba)
    evaluator = performanceMetrics.PerformanceEvaluator(task_type="classification")
    report = evaluator.evaluate_model(clf, {'text': np.zeros((5, 2))}, y_true, model_name="dummy", return_probabilities=True, measure_efficiency=False)
    assert isinstance(report, performanceMetrics.ModelPerformanceReport)
    assert 0.0 <= report.accuracy <= 1.0
    assert hasattr(report, 'f1_score')

def test_model_comparator():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    clf = DummyClassifier(y_pred)
    comparator = performanceMetrics.ModelComparator(task_type="classification")
    comparator.add_model_evaluation(clf, {'text': np.zeros((5, 2))}, y_true, "dummy")
    table = comparator.generate_comparison_table()
    assert "dummy" in table["model"].values
    rankings = comparator.generate_rankings()
    assert "accuracy" in rankings
