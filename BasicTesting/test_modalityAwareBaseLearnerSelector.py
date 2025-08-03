
import sys
import os
import numpy as np
import pytest
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from MainModel import modalityAwareBaseLearnerSelector as mls

class DummyBag:
    def __init__(self, bag_id, modality_mask, data_indices, diversity_score=0.0, dropout_rate=0.0):
        self.bag_id = bag_id
        self.modality_mask = modality_mask
        self.data_indices = data_indices
        self.diversity_score = diversity_score
        self.dropout_rate = dropout_rate

def make_bags():
    return [
        DummyBag(0, {'text': True, 'image': False, 'tabular': False}, np.arange(10)),
        DummyBag(1, {'text': True, 'image': True, 'tabular': False}, np.arange(10)),
        DummyBag(2, {'text': False, 'image': True, 'tabular': True}, np.arange(10)),
        DummyBag(3, {'text': True, 'image': True, 'tabular': True}, np.arange(10)),
    ]

def make_feature_dims():
    return {'text': 100, 'image': 64, 'tabular': 10}

def make_metadata():
    return {'dataset_size': 10, 'num_modalities': 3, 'modality_names': ['text','image','tabular'], 'feature_dimensions': {'text':100,'image':64,'tabular':10}}

def test_learner_config_structure():
    config = mls.LearnerConfig('id1', 'cnn', 'image', ['image'], {'channels':[64]}, 'classification')
    assert config.learner_id == 'id1'
    assert config.learner_type == 'cnn'
    assert config.modality_pattern == 'image'
    assert config.modalities_used == ['image']
    assert isinstance(config.architecture_params, dict)
    assert config.task_type == 'classification'

def test_generate_learners_and_types():
    bags = make_bags()
    feature_dims = make_feature_dims()
    meta = make_metadata()
    selector = mls.ModalityAwareBaseLearnerSelector(bags, feature_dims, meta)
    learners = selector.generate_learners(instantiate=False)
    assert isinstance(learners, dict)
    assert all(isinstance(cfg, mls.LearnerConfig) for cfg in learners.values())
    types = set(cfg.learner_type for cfg in learners.values())
    assert 'transformer' in types or 'cnn' in types or 'tree' in types or 'fusion' in types

def test_performance_prediction():
    bags = make_bags()
    feature_dims = make_feature_dims()
    meta = make_metadata()
    selector = mls.ModalityAwareBaseLearnerSelector(bags, feature_dims, meta)
    learners = selector.generate_learners(instantiate=False)
    for cfg in learners.values():
        assert 0.5 <= cfg.expected_performance <= 1.0

def test_hyperparameter_tuning_flag():
    bags = make_bags()
    feature_dims = make_feature_dims()
    meta = make_metadata()
    selector = mls.ModalityAwareBaseLearnerSelector(bags, feature_dims, meta, hyperparameter_tuning=True)
    learners = selector.generate_learners(instantiate=False)
    for cfg in learners.values():
        assert hasattr(cfg, 'hyperparameters')

def test_learner_summary():
    bags = make_bags()
    feature_dims = make_feature_dims()
    meta = make_metadata()
    selector = mls.ModalityAwareBaseLearnerSelector(bags, feature_dims, meta)
    selector.generate_learners(instantiate=False)
    summary = selector.get_learner_summary()
    assert 'total_learners' in summary
    assert 'learner_distribution' in summary
    assert 'by_type' in summary['learner_distribution']
    assert 'by_pattern' in summary['learner_distribution']

def test_performance_tracker():
    tracker = mls.PerformanceTracker()
    tdata = tracker.start_tracking('L1')
    import time; time.sleep(0.01)
    tracker.end_tracking('L1', tdata, {'expected_accuracy': 0.8})
    report = tracker.get_performance_report()
    assert 'average_training_time' in report
    assert 'learner_performances' in report
    assert 'top_performers' in report

def test_from_ensemble_bags_classmethod():
    bags = make_bags()
    feature_dims = make_feature_dims()
    meta = make_metadata()
    selector = mls.ModalityAwareBaseLearnerSelector.from_ensemble_bags(bags, feature_dims, meta)
    assert isinstance(selector, mls.ModalityAwareBaseLearnerSelector)
    learners = selector.generate_learners(instantiate=False)
    assert isinstance(learners, dict)


def test_empty_bags():
    feature_dims = make_feature_dims()
    meta = make_metadata()
    selector = mls.ModalityAwareBaseLearnerSelector([], feature_dims, meta)
    learners = selector.generate_learners(instantiate=False)
    assert learners == {}

def test_unknown_modality():
    bags = [DummyBag(0, {'unknown': True}, np.arange(5))]
    feature_dims = {'unknown': 7}
    meta = {'dataset_size': 5, 'num_modalities': 1, 'modality_names': ['unknown'], 'feature_dimensions': {'unknown':7}}
    selector = mls.ModalityAwareBaseLearnerSelector(bags, feature_dims, meta)
    learners = selector.generate_learners(instantiate=False)
    for cfg in learners.values():
        assert cfg.learner_type == 'fusion'

def test_extreme_feature_dims():
    bags = [DummyBag(0, {'text': True}, np.arange(10))]
    feature_dims = {'text': 10000}
    meta = {'dataset_size': 10, 'num_modalities': 1, 'modality_names': ['text'], 'feature_dimensions': {'text':10000}}
    selector = mls.ModalityAwareBaseLearnerSelector(bags, feature_dims, meta)
    learners = selector.generate_learners(instantiate=False)
    for cfg in learners.values():
        assert cfg.architecture_params['embedding_dim'] == 256

def test_missing_bag_attributes():
    class IncompleteBag:
        def __init__(self, bag_id):
            self.bag_id = bag_id
            self.modality_mask = {'text': True}
            self.data_indices = np.arange(5)
    bags = [IncompleteBag(0)]
    feature_dims = {'text': 5}
    meta = {'dataset_size': 5, 'num_modalities': 1, 'modality_names': ['text'], 'feature_dimensions': {'text':5}}
    selector = mls.ModalityAwareBaseLearnerSelector(bags, feature_dims, meta)
    learners = selector.generate_learners(instantiate=False)
    assert isinstance(list(learners.values())[0], mls.LearnerConfig)

def test_resource_limit_and_performance_threshold():
    bags = make_bags()
    feature_dims = make_feature_dims()
    meta = make_metadata()
    selector = mls.ModalityAwareBaseLearnerSelector(bags, feature_dims, meta, resource_limit={'max_memory': 1024}, performance_threshold=0.8)
    learners = selector.generate_learners(instantiate=False)
    for cfg in learners.values():
        assert hasattr(cfg, 'expected_performance')

def test_performance_prediction_boundaries():
    bags = [DummyBag(0, {'text': True}, np.arange(10), diversity_score=0.0, dropout_rate=10.0)]
    feature_dims = {'text': 100}
    meta = {'dataset_size': 10, 'num_modalities': 1, 'modality_names': ['text'], 'feature_dimensions': {'text':100}}
    selector = mls.ModalityAwareBaseLearnerSelector(bags, feature_dims, meta)
    learners = selector.generate_learners(instantiate=False)
    for cfg in learners.values():
        assert 0.5 <= cfg.expected_performance <= 1.0
