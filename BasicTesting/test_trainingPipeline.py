
import sys
import os
import numpy as np
import pytest
import torch
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from MainModel import trainingPipeline as tp

class DummyLearner(torch.nn.Module):
    def __init__(self, in_dim=4, out_dim=4):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.linear(x)

def test_config_defaults():
    cfg = tp.AdvancedTrainingConfig()
    assert cfg.epochs == 100
    assert cfg.enable_denoising is True
    assert isinstance(cfg.denoising_objectives, list)
    assert cfg.optimizer_type in ['adamw', 'sgd']

def test_metrics_structure():
    m = tp.ComprehensiveTrainingMetrics(epoch=1, train_loss=0.5, val_loss=0.4, accuracy=0.9)
    assert m.epoch == 1
    assert m.train_loss == 0.5
    assert m.val_loss == 0.4
    assert m.accuracy == 0.9

def test_denoising_loss_forward():
    cfg = tp.AdvancedTrainingConfig(denoising_objectives=["reconstruction"])
    loss_fn = tp.AdvancedCrossModalDenoisingLoss(cfg)
    learner = DummyLearner(4, 4)
    data = {"mod1": torch.randn(2, 4)}
    total_loss, losses = loss_fn(learner, data)
    assert isinstance(total_loss, torch.Tensor)
    assert "reconstruction_mod1" in losses

def test_training_pipeline_instantiation():
    pipeline = tp.create_training_pipeline(epochs=2, enable_denoising=True)
    assert isinstance(pipeline, tp.EnsembleTrainingPipeline)
    assert pipeline.config.epochs == 2
    assert pipeline.config.enable_denoising is True

def test_training_pipeline_run():
    pipeline = tp.create_training_pipeline(epochs=2, enable_denoising=True, batch_size=1)
    learners = {"L1": DummyLearner(4, 4)}
    learner_configs = [None]
    bag_data = {"L1": {"mod1": np.random.randn(2, 4)}}
    bag_labels = {"L1": np.random.randint(0, 2, 2)}
    trained, metrics = pipeline.train_ensemble(learners, learner_configs, bag_data, bag_labels=bag_labels)
    assert "L1" in trained
    assert "L1" in metrics
    assert len(metrics["L1"]) == 2
    summary = pipeline.get_training_summary(metrics)
    assert "average_performance" in summary
    assert "total_training_time" in summary

def test_optimizer_and_scheduler_selection():
    cfg = tp.AdvancedTrainingConfig(optimizer_type="sgd", scheduler_type="plateau", epochs=2)
    learner = DummyLearner(4, 4)
    pipeline = tp.EnsembleTrainingPipeline(cfg)
    opt = pipeline._get_optimizer(learner)
    sch = pipeline._get_scheduler(opt)
    assert isinstance(opt, torch.optim.Optimizer)
    assert sch is not None


def test_zero_epochs_and_empty_learners():
    pipeline = tp.create_training_pipeline(epochs=0)
    learners = {}
    learner_configs = []
    bag_data = {}
    trained, metrics = pipeline.train_ensemble(learners, learner_configs, bag_data)
    assert trained == {}
    assert metrics == {}

def test_invalid_optimizer_scheduler_types():
    cfg = tp.AdvancedTrainingConfig(optimizer_type="invalid", scheduler_type="invalid")
    learner = DummyLearner(4, 4)
    pipeline = tp.EnsembleTrainingPipeline(cfg)
    opt = pipeline._get_optimizer(learner)
    sch = pipeline._get_scheduler(opt)
    assert isinstance(opt, torch.optim.Optimizer)
    assert sch is None

def test_denoising_loss_multiple_objectives():
    cfg = tp.AdvancedTrainingConfig(denoising_objectives=["reconstruction", "consistency"])
    loss_fn = tp.AdvancedCrossModalDenoisingLoss(cfg)
    learner = DummyLearner(4, 4)
    data = {"mod1": torch.randn(2, 4)}
    orig = {"mod1": torch.randn(2, 4)}
    total_loss, losses = loss_fn(learner, data, original_representations=orig)
    assert "reconstruction_mod1" in losses
    assert "consistency_mod1" in losses

def test_training_multiple_learners_modalities():
    pipeline = tp.create_training_pipeline(epochs=1, enable_denoising=False, batch_size=1)
    learners = {"L1": DummyLearner(4, 4), "L2": DummyLearner(4, 4)}
    learner_configs = [None, None]
    bag_data = {"L1": {"mod1": np.random.randn(2, 4)}, "L2": {"mod2": np.random.randn(2, 4)}}
    bag_labels = {"L1": np.random.randint(0, 2, 2), "L2": np.random.randint(0, 2, 2)}
    trained, metrics = pipeline.train_ensemble(learners, learner_configs, bag_data, bag_labels=bag_labels)
    assert set(trained.keys()) == {"L1", "L2"}
    assert set(metrics.keys()) == {"L1", "L2"}

def test_factory_function_custom_args():
    pipeline = tp.create_training_pipeline(task_type="regression", num_classes=5, enable_denoising=False, epochs=3, optimizer_type="sgd")
    assert isinstance(pipeline, tp.EnsembleTrainingPipeline)
    assert pipeline.config.task_type == "regression"
    assert pipeline.config.epochs == 3
    assert pipeline.config.optimizer_type == "sgd"
