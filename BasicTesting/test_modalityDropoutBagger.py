
import sys
import os
import numpy as np
import pytest
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from MainModel import modalityDropoutBagger as mdb

class DummyConfig:
    def __init__(self, name, feature_dim, min_feature_ratio=0.3, max_feature_ratio=1.0):
        self.name = name
        self.feature_dim = feature_dim
        self.min_feature_ratio = min_feature_ratio
        self.max_feature_ratio = max_feature_ratio

def make_configs():
    return [DummyConfig('mod1', 4), DummyConfig('mod2', 3), DummyConfig('mod3', 2)]

def make_metadata():
    return {'dataset_size': 100, 'num_modalities': 3, 'modality_names': ['mod1','mod2','mod3'], 'feature_dimensions': {'mod1':4,'mod2':3,'mod3':2}}

def make_data():
    return {
        'mod1': np.random.randn(100, 4),
        'mod2': np.random.randn(100, 3),
        'mod3': np.random.randn(100, 2),
        'labels': np.random.randint(0, 2, 100)
    }

def test_bag_config_structure():
    cfg = mdb.BagConfig(0, np.arange(10), {'mod1':True}, {'mod1':np.ones(4,dtype=bool)}, 0.2)
    assert cfg.bag_id == 0
    assert isinstance(cfg.data_indices, np.ndarray)
    assert isinstance(cfg.modality_mask, dict)
    assert isinstance(cfg.feature_mask, dict)
    assert isinstance(cfg.dropout_rate, float)

def test_param_validation():
    configs = make_configs()
    meta = make_metadata()
    mdb.ModalityDropoutBagger(configs, meta, n_bags=5, dropout_strategy='linear', max_dropout_rate=0.5, min_modalities=1, sample_ratio=0.8, diversity_target=0.5)
    with pytest.raises(AssertionError):
        mdb.ModalityDropoutBagger(configs, meta, n_bags=0)
    with pytest.raises(AssertionError):
        mdb.ModalityDropoutBagger(configs, meta, dropout_strategy='bad')
    with pytest.raises(AssertionError):
        mdb.ModalityDropoutBagger(configs, meta, max_dropout_rate=1.1)
    with pytest.raises(AssertionError):
        mdb.ModalityDropoutBagger(configs, meta, sample_ratio=0.05)
    with pytest.raises(AssertionError):
        mdb.ModalityDropoutBagger(configs, meta, min_modalities=0)
    with pytest.raises(AssertionError):
        mdb.ModalityDropoutBagger(configs, meta, diversity_target=1.1)

def test_generate_bags_and_masks():
    configs = make_configs()
    meta = make_metadata()
    bagger = mdb.ModalityDropoutBagger(configs, meta, n_bags=10, dropout_strategy='random', max_dropout_rate=0.5, min_modalities=1)
    bags = bagger.generate_bags()
    assert len(bags) == 10
    for bag in bags:
        assert isinstance(bag, mdb.BagConfig)
        assert set(bag.modality_mask.keys()) == set(meta['modality_names'])
        for k, mask in bag.feature_mask.items():
            assert mask.shape[0] == meta['feature_dimensions'][k]

def test_get_bag_data():
    configs = make_configs()
    meta = make_metadata()
    data = make_data()
    bagger = mdb.ModalityDropoutBagger(configs, meta, n_bags=3)
    bagger.generate_bags()
    bag_data, mask = bagger.get_bag_data(0, data)
    assert isinstance(bag_data, dict)
    assert 'labels' in bag_data
    for k in ['mod1','mod2','mod3']:
        if mask[k]:
            assert bag_data[k].shape[0] == bag_data['labels'].shape[0]

def test_ensemble_stats():
    configs = make_configs()
    meta = make_metadata()
    bagger = mdb.ModalityDropoutBagger(configs, meta, n_bags=5)
    bagger.generate_bags()
    stats = bagger.get_ensemble_stats()
    assert 'modality_coverage' in stats
    assert 'diversity_metrics' in stats
    assert 'dropout_statistics' in stats
    detailed = bagger.get_ensemble_stats(return_detailed=True)
    assert 'bags' in detailed

def test_save_and_load_ensemble(tmp_path):
    configs = make_configs()
    meta = make_metadata()
    bagger = mdb.ModalityDropoutBagger(configs, meta, n_bags=2)
    bagger.generate_bags()
    file = tmp_path / 'ensemble.pkl'
    bagger.save_ensemble(str(file))
    loaded = mdb.ModalityDropoutBagger.load_ensemble(str(file))
    assert isinstance(loaded, mdb.ModalityDropoutBagger)
    assert len(loaded.bags) == 2


def test_min_max_modalities():
    configs = [DummyConfig('m1', 2), DummyConfig('m2', 2)]
    meta = {'dataset_size': 10, 'num_modalities': 2, 'modality_names': ['m1','m2'], 'feature_dimensions': {'m1':2,'m2':2}}
    # min_modalities = total modalities
    bagger = mdb.ModalityDropoutBagger(configs, meta, n_bags=3, min_modalities=2)
    bags = bagger.generate_bags()
    for bag in bags:
        assert all(bag.modality_mask.values())  # No dropout possible
    # min_modalities = 1, max_dropout_rate = 1.0
    bagger = mdb.ModalityDropoutBagger(configs, meta, n_bags=3, min_modalities=1, max_dropout_rate=0.9)
    bags = bagger.generate_bags()
    for bag in bags:
        assert 1 <= sum(bag.modality_mask.values()) <= 2

def test_feature_sampling_boundaries():
    configs = [DummyConfig('m1', 5, min_feature_ratio=0.2, max_feature_ratio=0.2)]
    meta = {'dataset_size': 10, 'num_modalities': 1, 'modality_names': ['m1'], 'feature_dimensions': {'m1':5}}
    bagger = mdb.ModalityDropoutBagger(configs, meta, n_bags=5, feature_sampling=True)
    bags = bagger.generate_bags()
    for bag in bags:
        mask = bag.feature_mask['m1']
        assert mask.sum() == 1  # 0.2*5=1

def test_dropout_zero_and_max():
    configs = make_configs()
    meta = make_metadata()
    # Dropout rate 0
    bagger = mdb.ModalityDropoutBagger(configs, meta, n_bags=2, dropout_strategy='linear', max_dropout_rate=0.0)
    bags = bagger.generate_bags()
    for bag in bags:
        assert all(bag.modality_mask.values())
    # Dropout rate max
    bagger = mdb.ModalityDropoutBagger(configs, meta, n_bags=2, dropout_strategy='linear', max_dropout_rate=0.9, min_modalities=1)
    bags = bagger.generate_bags()
    for bag in bags:
        assert 1 <= sum(bag.modality_mask.values()) <= 3

def test_empty_configs():
    meta = {'dataset_size': 10, 'num_modalities': 0, 'modality_names': [], 'feature_dimensions': {}}
    with pytest.raises(AssertionError):
        mdb.ModalityDropoutBagger([], meta)

def test_serialization_with_nondefault_params(tmp_path):
    configs = make_configs()
    meta = make_metadata()
    bagger = mdb.ModalityDropoutBagger(configs, meta, n_bags=2, dropout_strategy='exponential', max_dropout_rate=0.7, min_modalities=2, sample_ratio=0.5, diversity_target=0.9, feature_sampling=False, enable_validation=False, random_state=123)
    bagger.generate_bags()
    file = tmp_path / 'ensemble2.pkl'
    bagger.save_ensemble(str(file))
    loaded = mdb.ModalityDropoutBagger.load_ensemble(str(file))
    assert loaded.dropout_strategy == 'exponential'
    assert loaded.max_dropout_rate == 0.7
    assert loaded.min_modalities == 2
    assert loaded.sample_ratio == 0.5
    assert loaded.diversity_target == 0.9
    assert loaded.feature_sampling is False
    assert loaded.enable_validation is False
    assert loaded.random_state == 123

def test_integration_with_stage1_like_output():
    # Simulate Stage 1 output
    class FakeConfig:
        def __init__(self, name, feature_dim):
            self.name = name
            self.feature_dim = feature_dim
            self.min_feature_ratio = 0.3
            self.max_feature_ratio = 1.0
    configs = [FakeConfig('tab', 3), FakeConfig('img', 2)]
    meta = {'dataset_size': 20, 'num_modalities': 2, 'modality_names': ['tab','img'], 'feature_dimensions': {'tab':3,'img':2}}
    data = {'tab': np.random.randn(20,3), 'img': np.random.randn(20,2), 'labels': np.random.randint(0,2,20)}
    bagger = mdb.ModalityDropoutBagger(configs, meta, n_bags=4)
    bagger.generate_bags()
    for i in range(4):
        bag_data, mask = bagger.get_bag_data(i, data)
        assert 'labels' in bag_data
        for k in ['tab','img']:
            if mask[k]:
                assert bag_data[k].shape[0] == bag_data['labels'].shape[0]
