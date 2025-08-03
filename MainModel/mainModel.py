"""
mainModel.py
Production-ready entry point for the full multimodal ensemble pipeline.
Provides a unified API for data integration, training, prediction, and evaluation.
"""

import numpy as np
from types import SimpleNamespace
from MainModel import dataIntegration
from MainModel import modalityDropoutBagger
from MainModel import modalityAwareBaseLearnerSelector
from MainModel import trainingPipeline
from MainModel import ensemblePrediction
from MainModel import performanceMetrics
import torch


class MultiModalEnsembleModel:
    def save(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    def __init__(self, data_loader, n_bags=10, dropout_strategy='random', epochs=10, batch_size=32, random_state=42):
        self.n_bags = n_bags
        self.dropout_strategy = dropout_strategy
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.trained_learners = None
        self.learner_metadata = None
        self.ensemble = None
        self.pipeline = None
        self.modality_configs = None
        self.integration_metadata = None
        self.data = None
        self.labels = None
        self.data_loader = data_loader
        self.n_classes = None
        self.n_features = None

    def load_and_integrate_data(self):
        # Use the provided dataIntegration loader to get train/test splits
        self.train_data, self.train_labels = self.data_loader.get_split('train')
        self.test_data, self.test_labels = self.data_loader.get_split('test')
        self.modality_configs = [
            SimpleNamespace(name=k, feature_dim=v.shape[1]) for k, v in self.train_data.items()
        ]
        self.integration_metadata = {'dataset_size': len(self.train_labels)}
        # Infer n_classes and n_features for downstream use
        self.n_features = list(self.train_data.values())[0].shape[1]
        self.n_classes = len(np.unique(self.train_labels))

    def fit(self):
        # 1. Modality Dropout Bagger
        bagger = modalityDropoutBagger.ModalityDropoutBagger.from_data_integration(
            integrated_data=self.train_data,
            modality_configs=self.modality_configs,
            integration_metadata=self.integration_metadata,
            n_bags=self.n_bags,
            dropout_strategy=self.dropout_strategy
        )
        bags = bagger.generate_bags()
        # 2. Base Learner Selector (auto-instantiates learners)
        modality_feature_dims = {mc.name: mc.feature_dim for mc in self.modality_configs}
        selector = modalityAwareBaseLearnerSelector.ModalityAwareBaseLearnerSelector(
            bags, modality_feature_dims, self.integration_metadata
        )
        selected_learners = selector.generate_learners(instantiate=True)
        learner_metadata = []
        bag_data = {}
        bag_labels = {}
        for i, (learner_id, learner) in enumerate(selected_learners.items()):
            bag = bags[i]
            bag_data[learner_id] = {mod: self.train_data[mod][bag.data_indices] for mod in self.train_data}
            bag_labels[learner_id] = self.train_labels[bag.data_indices]
            learner_metadata.append({
                'learner_id': learner_id,
                'metrics': {},
                'modalities': [mc.name for mc in self.modality_configs],
                'pattern': '+'.join([mc.name for mc in self.modality_configs])
            })
        # Pass task_type to pipeline
        task_type = 'regression' if len(np.unique(self.train_labels)) > 20 else 'classification'
        self.pipeline = trainingPipeline.create_training_pipeline(task_type=task_type, epochs=self.epochs, batch_size=self.batch_size)
        self.trained_learners, training_metrics = self.pipeline.train_ensemble(selected_learners, [None]*len(selected_learners), bag_data, bag_labels=bag_labels)
        # Optionally propagate metrics to learner_metadata
        for i, (lid, metrics) in enumerate(training_metrics.items()):
            learner_metadata[i]['metrics'] = metrics[-1].__dict__ if metrics else {}
        # Sort both by learner_id for determinism
        learner_id_order = sorted(self.trained_learners.keys())
        self.trained_learners = [self.trained_learners[lid] for lid in learner_id_order]
        self.learner_metadata = [next(m for m in learner_metadata if m['learner_id'] == lid) for lid in learner_id_order]
        return training_metrics

    def predict(self, data_dict=None):
        # 4. Ensemble Prediction
        if data_dict is None:
            data_dict = self.test_data
        # Use task_type from fit
        task_type = 'regression' if len(np.unique(self.train_labels)) > 20 else 'classification'
        self.ensemble = ensemblePrediction.create_ensemble_predictor(task_type=task_type)
        # Ensure deterministic order by learner_id
        for learner, meta in zip(self.trained_learners, self.learner_metadata):
            self.ensemble.add_trained_learner(learner, meta.get('metrics', {}), meta.get('modalities', []), meta.get('pattern', ''))
        pred_result = self.ensemble.predict(data_dict)
        return pred_result

    def evaluate(self, data_dict=None, labels=None):
        if data_dict is None:
            data_dict = self.test_data
        if labels is None:
            labels = self.test_labels
        # Use task_type from fit
        task_type = 'regression' if len(np.unique(self.train_labels)) > 20 else 'classification'
        pred_result = self.predict(data_dict)
        evaluator = performanceMetrics.PerformanceEvaluator(task_type=task_type)
        report = evaluator.evaluate_model(lambda X: pred_result.predictions, data_dict, labels, model_name="Ensemble")
        return report

# Example usage (for script/demo, not run on import)
if __name__ == "__main__":
    # Example: create a generic loader and add features/labels for any application
    loader = dataIntegration.GenericMultiModalDataLoader()
    # For demonstration, create dummy train/test split data:
    N_train = 80
    N_test = 20
    n_features = 8
    n_classes = 3
    np.random.seed(42)
    loader.add_modality_split('text', np.random.randn(N_train, n_features), np.random.randn(N_test, n_features))
    loader.add_modality_split('image', np.random.randn(N_train, n_features), np.random.randn(N_test, n_features))
    loader.add_modality_split('metadata', np.random.randn(N_train, n_features), np.random.randn(N_test, n_features))
    loader.add_labels_split(np.random.randint(0, n_classes, N_train), np.random.randint(0, n_classes, N_test))

    model = MultiModalEnsembleModel(data_loader=loader, n_bags=2, epochs=2, batch_size=8)
    print("Loading and integrating data...")
    model.load_and_integrate_data()
    print("Fitting model...")
    model.fit()
    print("Evaluating model...")
    report = model.evaluate()
    print("Performance report:", report)
