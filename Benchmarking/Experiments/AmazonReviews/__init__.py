#!/usr/bin/env python3
"""
Amazon Reviews 5-Class Rating Prediction Experiments

A comprehensive modular framework for evaluating machine learning models
on Amazon Reviews rating prediction task.

Modalities:
- Text features (TF-IDF, word embeddings)
- Metadata features (review length, helpful votes, etc.)

Models evaluated:
- 30+ baseline models across 5 categories
- MainModel multimodal ensemble with dropout strategies
- Advanced ensemble techniques

Analysis includes:
- Cross-validation evaluation
- Robustness testing
- Interpretability analysis
- Modality ablation studies
- Statistical significance testing
"""

__version__ = "1.0.0"
__author__ = "PolyModalEnsemble Project"

# Import main components for easy access
from .config import ExperimentConfig
from .data_loader import AmazonReviewsDataLoader
from .baseline_evaluator import BaselineEvaluator
from .mainmodel_evaluator import MainModelEvaluator
from .advanced_evaluator import AdvancedEvaluator
from .results_manager import ResultsManager
from .experiment_orchestrator import ExperimentOrchestrator

__all__ = [
    'ExperimentConfig',
    'AmazonReviewsDataLoader', 
    'BaselineEvaluator',
    'MainModelEvaluator',
    'AdvancedEvaluator',
    'ResultsManager',
    'ExperimentOrchestrator'
]

# Quick access functions
def run_quick_experiment(config_overrides=None):
    """Run a quick experiment for fast feedback"""
    orchestrator = ExperimentOrchestrator(config_overrides)
    return orchestrator.run_quick_experiment()

def run_full_experiment(config_overrides=None):
    """Run the complete experimental pipeline"""
    orchestrator = ExperimentOrchestrator(config_overrides)
    return orchestrator.run_full_experiment()

def run_baseline_comparison(config_overrides=None):
    """Run baseline model comparison only"""
    orchestrator = ExperimentOrchestrator(config_overrides)
    return orchestrator.run_baseline_only()

def run_mainmodel_evaluation(config_overrides=None, include_hp_search=True):
    """Run MainModel evaluation only"""
    orchestrator = ExperimentOrchestrator(config_overrides)
    return orchestrator.run_mainmodel_only(include_hyperparameter_search=include_hp_search)
