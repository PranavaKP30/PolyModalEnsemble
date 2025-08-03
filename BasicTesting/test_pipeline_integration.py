"""
test_pipeline_integration.py
Integration test for the full pipeline: dataIntegration -> modalityDropoutBagger -> modalityAwareBaseLearnerSelector -> trainingPipeline -> ensemblePrediction -> performanceMetrics
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MainModel import dataIntegration
from MainModel import modalityDropoutBagger
from MainModel import modalityAwareBaseLearnerSelector
from MainModel import trainingPipeline
from MainModel import ensemblePrediction
from MainModel import performanceMetrics

# Dummy data for integration test
N = 20
n_features = 8
n_classes = 3
np.random.seed(42)

def test_full_pipeline():
    # 1. Data Integration
    data = {
        'text': np.random.randn(N, n_features),
        'image': np.random.randn(N, n_features),
        'metadata': np.random.randn(N, n_features)
    }
    labels = np.random.randint(0, n_classes, N)
    # Simulate dataIntegration output
    integrated_data = data
    from types import SimpleNamespace
    modality_configs = [
        SimpleNamespace(name='text', feature_dim=n_features),
        SimpleNamespace(name='image', feature_dim=n_features),
        SimpleNamespace(name='metadata', feature_dim=n_features)
    ]
    integration_metadata = {'dataset_size': N}

    # 2. Modality Dropout Bagger
    bagger = modalityDropoutBagger.ModalityDropoutBagger.from_data_integration(
        integrated_data=integrated_data,
        modality_configs=modality_configs,
        integration_metadata=integration_metadata,
        n_bags=2,
        dropout_strategy='random'
    )
    bags = bagger.generate_bags()

    # 3. Base Learner Selector
    # Build modality_feature_dims dict
    modality_feature_dims = {mc.name: mc.feature_dim for mc in modality_configs}
    selector = modalityAwareBaseLearnerSelector.ModalityAwareBaseLearnerSelector(
        bags, modality_feature_dims, integration_metadata
    )
    learner_configs = selector.generate_learners()
    # Wrap configs in dummy learners for pipeline compatibility
    import torch
    class DummyLearner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        def fit(self, X, y):
            pass
        def predict(self, X):
            n = len(next(iter(X.values())))
            return np.random.randint(0, n_classes, n)
        def predict_proba(self, X):
            n = len(next(iter(X.values())))
            proba = np.random.rand(n, n_classes)
            proba /= proba.sum(axis=1, keepdims=True)
            return proba
        def forward(self, x):
            # Return a tensor of the same shape as input, with requires_grad
            return x * self.dummy_param
    selected_learners = {k: DummyLearner() for k in learner_configs}
    learner_metadata = [{'metrics': {}, 'modalities': ['text', 'image', 'metadata'], 'pattern': 'text+image+metadata'} for _ in selected_learners]
    # Convert bags to dict keyed by learner_id, each value a dict of modality arrays
    bag_data = {}
    bag_labels = {}
    for i, (learner_id, learner) in enumerate(selected_learners.items()):
        bag = bags[i]
        bag_data[learner_id] = {mod: data[mod][bag.data_indices] for mod in data}
        bag_labels[learner_id] = labels[bag.data_indices]

    # 4. Training Pipeline
    pipeline = trainingPipeline.create_training_pipeline(epochs=1, batch_size=4)
    trained_learners, training_metrics = pipeline.train_ensemble(selected_learners, [None]*len(selected_learners), bag_data, bag_labels=bag_labels)

    # 5. Ensemble Prediction
    ensemble = ensemblePrediction.create_ensemble_predictor()
    for learner, meta in zip(trained_learners.values(), learner_metadata):
        ensemble.add_trained_learner(learner, meta.get('metrics', {}), meta.get('modalities', []), meta.get('pattern', ''))
    test_data = {k: v for k, v in data.items()}
    pred_result = ensemble.predict(test_data)

    # 6. Performance Metrics
    evaluator = performanceMetrics.PerformanceEvaluator(task_type="classification")
    report = evaluator.evaluate_model(lambda X: pred_result.predictions, test_data, labels, model_name="Ensemble")
    assert 0.0 <= report.accuracy <= 1.0
    assert hasattr(report, 'f1_score')
    assert hasattr(report, 'inference_time_ms')
    print("Pipeline integration test passed.")
