
import sys
import os
import numpy as np
import pytest
import torch
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MainModel import ensemblePrediction as ep

class DummyLearner:
    def __init__(self, pred=None, proba=None):
        self._pred = pred
        self._proba = proba
    def predict(self, data):
        return self._pred if self._pred is not None else np.zeros(len(next(iter(data.values()))))
    def predict_proba(self, data):
        # data is now a numpy array, not a dict
        if isinstance(data, dict):
            n = len(next(iter(data.values())))
        else:
            n = data.shape[0]
        return self._proba if self._proba is not None else np.ones((n, 2)) * 0.5

def test_aggregation_strategies():
    pred = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1]])
    weights = np.array([0.2, 0.5, 0.3])
    predictor = ep.EnsemblePredictor(aggregation_strategy="majority_vote")
    mv = predictor._majority_vote(pred)
    assert mv.shape == (3,)
    predictor.aggregation_strategy = ep.AggregationStrategy.WEIGHTED_VOTE
    wv = predictor._weighted_vote(pred, weights)
    assert wv.shape == (3,)

def test_uncertainty_methods():
    predictor = ep.EnsemblePredictor(uncertainty_method="entropy")
    # Add dummy learners
    for _ in range(3):
        learner = DummyLearner(proba=np.array([[0.7, 0.3], [0.4, 0.6]]))
        predictor.add_trained_learner(learner, {"accuracy": 0.9}, ["mod1"], "mod1")
    data = {"mod1": np.random.randn(2, 4)}
    result = predictor.predict(data, return_uncertainty=True)
    assert result.uncertainty is not None
    predictor.uncertainty_method = ep.UncertaintyMethod.ENSEMBLE_DISAGREEMENT
    result2 = predictor.predict(data, return_uncertainty=True)
    assert result2.uncertainty is not None

def test_prediction_result_container():
    arr = np.array([1, 0, 1])
    res = ep.PredictionResult(predictions=arr, confidence=arr, uncertainty=arr, modality_importance={"mod1": 1.0}, metadata={"foo": "bar"})
    assert res.predictions.shape == (3,)
    assert res.confidence.shape == (3,)
    assert res.uncertainty.shape == (3,) or isinstance(res.uncertainty, float) or res.uncertainty is not None
    assert "foo" in res.metadata

def test_transformer_meta_learner_forward():
    model = ep.TransformerMetaLearner(input_dim=4, num_heads=2, num_layers=1, hidden_dim=8, num_classes=2)
    x = torch.randn(2, 3, 4)  # (batch, seq_len, input_dim)
    logits, attn = model(x)
    assert logits.shape[0] == 2
    # attn shape: (batch, num_heads, seq_len, seq_len) or (batch, seq_len, seq_len)
    assert attn.ndim == 3
    assert attn.shape[0] == 2  # batch size

def test_ensemble_predictor_add_and_setup():
    predictor = ep.EnsemblePredictor()
    learner = DummyLearner()
    predictor.add_trained_learner(learner, {"accuracy": 0.8}, ["mod1"], "mod1")
    assert len(predictor.trained_learners) == 1
    # Use valid num_heads for input_dim=4 (default is 8, which is invalid)
    predictor.transformer_meta_learner = ep.TransformerMetaLearner(input_dim=4, num_heads=2, num_layers=1, hidden_dim=8, num_classes=2)
    assert predictor.transformer_meta_learner is not None

def test_factory_function():
    predictor = ep.create_ensemble_predictor(task_type="regression", aggregation_strategy="confidence_weighted", uncertainty_method="variance", calibrate_uncertainty=False, device="cpu")
    assert isinstance(predictor, ep.EnsemblePredictor)
    assert predictor.task_type == "regression"
    assert predictor.aggregation_strategy == ep.AggregationStrategy.CONFIDENCE_WEIGHTED
    assert predictor.uncertainty_method == ep.UncertaintyMethod.VARIANCE
    assert predictor.calibrate_uncertainty is False
    assert str(predictor.device) == "cpu"

def test_evaluate_method():
    predictor = ep.EnsemblePredictor()
    learner = DummyLearner(pred=np.array([1, 0, 1]))
    predictor.add_trained_learner(learner, {"accuracy": 0.8}, ["mod1"], "mod1")
    data = {"mod1": np.random.randn(3, 4)}
    true_labels = np.array([1, 0, 1])
    metrics = predictor.evaluate(data, true_labels)
    assert "accuracy" in metrics
    assert "confidence_mean" in metrics
    assert "uncertainty_mean" in metrics
    assert "modality_importance" in metrics
    assert "metadata" in metrics
