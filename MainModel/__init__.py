"""
MainModel: A Multimodal Ensemble Learning Framework

This package provides a comprehensive multimodal ensemble learning system with:
- Data integration and preprocessing
- Modality-aware ensemble generation
- Advanced training pipeline with cross-modal denoising
- Transformer-based ensemble prediction
- Uncertainty quantification and interpretability features
"""

from .mainModelAPI import MultiModalEnsembleModel
from .dataIntegration import SimpleDataLoader
from .modalityDropoutBagger import ModalityDropoutBagger, BagConfig
from .baseLearnerSelector import BaseLearnerSelector, BagLearnerConfig
from .trainingPipeline import EnsembleTrainingPipeline, TrainedLearnerInfo, AdvancedTrainingConfig, ComprehensiveTrainingMetrics
from .ensemblePrediction import EnsemblePredictor, PredictionResult, AggregationStrategy, UncertaintyMethod

__version__ = "1.0.0"
__author__ = "PolyModalEnsemble Team"

__all__ = [
    'MultiModalEnsembleModel',
    'SimpleDataLoader', 
    'ModalityDropoutBagger',
    'BagConfig',
    'BaseLearnerSelector',
    'BagLearnerConfig',
    'EnsembleTrainingPipeline',
    'TrainedLearnerInfo',
    'AdvancedTrainingConfig',
    'ComprehensiveTrainingMetrics',
    'EnsemblePredictor',
    'PredictionResult',
    'AggregationStrategy',
    'UncertaintyMethod'
]
