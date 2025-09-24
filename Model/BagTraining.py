"""
Stage 4: BagTraining - Training Pipeline for Multimodal Ensemble Learning

This module implements the training pipeline that takes selected weak learners from Stage 3
and their respective bags from Stage 2, trains each learner on its specific bag, and outputs
trained models aligned with their respective bags for Stage 5 ensemble prediction.

Author: PolyModal Ensemble Team
Date: 2024
"""

import os
import json
import pickle
import logging
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training a single bag learner."""
    bag_id: int
    learner_type: str
    hyperparameters: Dict[str, Any]
    training_epochs: int
    batch_size: int
    learning_rate: float
    device: str = 'cpu'
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    save_checkpoints: bool = True
    checkpoint_interval: int = 5

@dataclass
class TrainingMetrics:
    """Training metrics for a single bag learner."""
    bag_id: int
    learner_type: str
    final_train_loss: float
    final_val_loss: float
    final_train_accuracy: float
    final_val_accuracy: float
    best_val_accuracy: float
    training_epochs: int
    training_time: float
    convergence_epoch: int
    overfitting_detected: bool
    performance_metrics: Dict[str, float]

@dataclass
class TrainedModel:
    """Container for a trained model with its metadata."""
    bag_id: int
    learner_type: str
    model: Any  # Can be PyTorch model or sklearn model
    training_config: TrainingConfig
    training_metrics: TrainingMetrics
    model_path: str
    created_at: str

class BagTraining:
    """
    Stage 4: Training Pipeline for Multimodal Ensemble Learning.
    
    Takes selected weak learners from Stage 3 and their respective bags from Stage 2,
    trains each learner on its specific bag, and outputs trained models aligned with
    their respective bags for Stage 5 ensemble prediction.
    """
    
    def __init__(self, 
                 output_dir: str = "Model/trained_models",
                 device: str = 'cpu',
                 random_state: int = 42):
        """
        Initialize the Bag Training system.
        
        Parameters
        ----------
        output_dir : str
            Directory to save trained models and metadata
        device : str
            Device for training ('cpu' or 'cuda')
        random_state : int
            Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.device = device
        self.random_state = random_state
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.trained_models: Dict[int, TrainedModel] = {}
        self.training_configs: Dict[int, TrainingConfig] = {}
        self.training_metrics: Dict[int, TrainingMetrics] = {}
        
        # Set random seeds
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
        
        logger.info(f"Initialized BagTraining with device: {device}, output_dir: {output_dir}")
    
    def train_bags(self, 
                   bag_data: Dict[int, Dict[str, Any]], 
                   learner_configs: Dict[int, Dict[str, Any]],
                   train_data: Dict[str, np.ndarray],
                   train_labels: np.ndarray,
                   test_data: Dict[str, np.ndarray],
                   test_labels: np.ndarray) -> Dict[int, TrainedModel]:
        """
        Train all bag learners on their respective bags.
        
        Parameters
        ----------
        bag_data : Dict[int, Dict[str, Any]]
            Bag data from Stage 2 with bag_id as key
        learner_configs : Dict[int, Dict[str, Any]]
            Learner configurations from Stage 3 with bag_id as key
        train_data : Dict[str, np.ndarray]
            Training data from Stage 1
        train_labels : np.ndarray
            Training labels from Stage 1
        test_data : Dict[str, np.ndarray]
            Test data from Stage 1
        test_labels : np.ndarray
            Test labels from Stage 1
            
        Returns
        -------
        Dict[int, TrainedModel]
            Dictionary mapping bag_id to trained model
        """
        logger.info(f"Starting training for {len(bag_data)} bags")
        
        # Step 1: Initialize Training Pipeline
        self._initialize_training_pipeline(bag_data, learner_configs, train_data, train_labels, test_data, test_labels)
        
        # Step 2: Prepare Training Data
        prepared_data = self._prepare_training_data(bag_data, train_data, train_labels, test_data, test_labels)
        
        # Step 3: Initialize Base Learners
        learners = self._initialize_base_learners(learner_configs)
        
        # Step 4: Configure Training Components
        training_components = self._configure_training_components(learner_configs)
        
        # Step 5: Execute Training Loop
        for bag_id in bag_data.keys():
            logger.info(f"Training bag {bag_id} with {learner_configs[bag_id]['learner_type']}")
            
            # Step 6: Data Injection into Learner
            bag_specific_data = self._inject_data_into_learner(bag_id, prepared_data, bag_data)
            
            # Step 7: Training Loop Execution
            trained_model = self._execute_training_loop(
                bag_id, learners[bag_id], training_components[bag_id], 
                bag_specific_data, learner_configs[bag_id]
            )
            
            # Step 8: Model Storage and Serialization
            self._store_trained_model(bag_id, trained_model, learner_configs[bag_id])
        
        # Step 9: Convenience Functions
        self._run_final_validation()
        self._generate_performance_report()
        
        logger.info(f"Training completed for {len(self.trained_models)} bags")
        return self.trained_models
    
    def _initialize_training_pipeline(self, 
                                    bag_data: Dict[int, Dict[str, Any]], 
                                    learner_configs: Dict[int, Dict[str, Any]],
                                    train_data: Dict[str, np.ndarray],
                                    train_labels: np.ndarray,
                                    test_data: Dict[str, np.ndarray],
                                    test_labels: np.ndarray):
        """Step 1: Initialize Training Pipeline"""
        logger.info("Step 1: Initializing training pipeline")
        
        # Validate inputs
        if len(bag_data) != len(learner_configs):
            raise ValueError(f"Mismatch between bag_data ({len(bag_data)}) and learner_configs ({len(learner_configs)})")
        
        # Validate bag IDs match
        bag_ids = set(bag_data.keys())
        learner_ids = set(learner_configs.keys())
        if bag_ids != learner_ids:
            raise ValueError(f"Bag IDs mismatch: {bag_ids} vs {learner_ids}")
        
        # Store training data
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.bag_data = bag_data
        self.learner_configs = learner_configs
        
        logger.info(f"Training pipeline initialized for {len(bag_data)} bags")
    
    def _prepare_training_data(self, 
                              bag_data: Dict[int, Dict[str, Any]],
                              train_data: Dict[str, np.ndarray],
                              train_labels: np.ndarray,
                              test_data: Dict[str, np.ndarray],
                              test_labels: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Step 2: Prepare Training Data"""
        logger.info("Step 2: Preparing training data")
        
        prepared_data = {}
        
        for bag_id, bag_info in bag_data.items():
            # Get bag-specific data indices
            data_indices = bag_info['data_indices']
            modality_mask = bag_info['modality_mask']
            
            # Extract bag-specific training data
            bag_train_data = {}
            for modality, data in train_data.items():
                if modality_mask.get(modality, False):  # Only include active modalities
                    bag_train_data[modality] = data[data_indices]
            
            bag_train_labels = train_labels[data_indices]
            
            # Prepare test data (use all test data for evaluation)
            bag_test_data = {}
            for modality, data in test_data.items():
                if modality_mask.get(modality, False):  # Only include active modalities
                    bag_test_data[modality] = data
            
            prepared_data[bag_id] = {
                'train_data': bag_train_data,
                'train_labels': bag_train_labels,
                'test_data': bag_test_data,
                'test_labels': test_labels,
                'modality_mask': modality_mask,
                'data_indices': data_indices
            }
        
        logger.info(f"Training data prepared for {len(prepared_data)} bags")
        return prepared_data
    
    def _initialize_base_learners(self, learner_configs: Dict[int, Dict[str, Any]]) -> Dict[int, Any]:
        """Step 3: Initialize Base Learners"""
        logger.info("Step 3: Initializing base learners")
        
        learners = {}
        
        for bag_id, config in learner_configs.items():
            learner_type = config['learner_type']
            hyperparams = config['hyperparameters']
            
            if learner_type == 'Random Forest':
                learners[bag_id] = self._create_random_forest(hyperparams)
            elif learner_type == 'ConvNeXt-Base':
                learners[bag_id] = self._create_convnext(hyperparams)
            elif learner_type == 'EfficientNet B4':
                learners[bag_id] = self._create_efficientnet(hyperparams)
            elif learner_type == '1D-CNN ResNet Style':
                learners[bag_id] = self._create_1d_cnn(hyperparams)
            elif learner_type in ['Multi-Input ConvNeXt', 'Attention-based Fusion Network', 
                                'Cross-modal Attention Network', 'Temporal-Spatial Fusion Network',
                                'Multi-Head Attention Fusion Network']:
                learners[bag_id] = self._create_fusion_network(learner_type, hyperparams)
            else:
                raise ValueError(f"Unknown learner type: {learner_type}")
        
        logger.info(f"Initialized {len(learners)} base learners")
        return learners
    
    def _configure_training_components(self, learner_configs: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Step 4: Configure Training Components"""
        logger.info("Step 4: Configuring training components")
        
        training_components = {}
        
        for bag_id, config in learner_configs.items():
            learner_type = config['learner_type']
            hyperparams = config['hyperparameters']
            
            components = {
                'optimizer': None,
                'loss_function': None,
                'scheduler': None,
                'metrics': ['accuracy', 'precision', 'recall', 'f1']
            }
            
            if learner_type in ['ConvNeXt-Base', 'EfficientNet B4', '1D-CNN ResNet Style'] or \
               'Fusion' in learner_type or 'Multi-Input' in learner_type:
                # Deep learning components
                components['optimizer'] = self._create_optimizer(hyperparams)
                components['loss_function'] = self._create_loss_function(hyperparams)
                components['scheduler'] = self._create_scheduler(hyperparams)
            else:
                # Sklearn components (Random Forest)
                components['loss_function'] = 'cross_entropy'  # For sklearn compatibility
                components['metrics'] = ['accuracy', 'precision', 'recall', 'f1']
            
            training_components[bag_id] = components
        
        logger.info(f"Configured training components for {len(training_components)} bags")
        return training_components
    
    def _inject_data_into_learner(self, 
                                 bag_id: int,
                                 prepared_data: Dict[int, Dict[str, Any]],
                                 bag_data: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Step 6: Data Injection into Learner"""
        logger.info(f"Step 6: Injecting data into learner for bag {bag_id}")
        
        bag_specific_data = prepared_data[bag_id]
        bag_info = bag_data[bag_id]
        
        # Create data loaders for deep learning models
        learner_type = self.learner_configs[bag_id]['learner_type']
        
        if learner_type in ['ConvNeXt-Base', 'EfficientNet B4', '1D-CNN ResNet Style'] or \
           'Fusion' in learner_type or 'Multi-Input' in learner_type:
            # Create PyTorch DataLoaders
            data_loaders = self._create_pytorch_dataloaders(bag_specific_data, bag_id)
            bag_specific_data['data_loaders'] = data_loaders
        
        return bag_specific_data
    
    def _execute_training_loop(self, 
                              bag_id: int,
                              learner: Any,
                              training_components: Dict[str, Any],
                              bag_specific_data: Dict[str, Any],
                              learner_config: Dict[str, Any]) -> TrainedModel:
        """Step 7: Training Loop Execution"""
        logger.info(f"Step 7: Executing training loop for bag {bag_id}")
        
        learner_type = learner_config['learner_type']
        hyperparams = learner_config['hyperparameters']
        
        # Create training configuration
        training_config = TrainingConfig(
            bag_id=bag_id,
            learner_type=learner_type,
            hyperparameters=hyperparams,
            training_epochs=hyperparams.get('epochs', 50),
            batch_size=hyperparams.get('batch_size', 32),
            learning_rate=hyperparams.get('learning_rate', 0.001),
            device=self.device,
            validation_split=0.2,
            early_stopping_patience=hyperparams.get('early_stopping_patience', 10),
            save_checkpoints=True,
            checkpoint_interval=5
        )
        
        # Validate architecture inputs
        self._validate_architecture_inputs(learner_type, hyperparams, bag_specific_data['train_data'])
        
        # Adjust batch size based on memory availability
        data_size = len(bag_specific_data['train_labels'])
        original_batch_size = training_config.batch_size
        adjusted_batch_size = self._adjust_batch_size_for_memory(original_batch_size, data_size, learner_type)
        training_config.batch_size = adjusted_batch_size
        
        start_time = time.time()
        
        if learner_type == 'Random Forest':
            # Train sklearn model
            trained_model, metrics = self._train_sklearn_model(
                learner, bag_specific_data, training_config
            )
        else:
            # Train PyTorch model
            trained_model, metrics = self._train_pytorch_model(
                learner, training_components, bag_specific_data, training_config
            )
        
        training_time = time.time() - start_time
        metrics.training_time = training_time
        
        # Create trained model container
        trained_model_container = TrainedModel(
            bag_id=bag_id,
            learner_type=learner_type,
            model=trained_model,
            training_config=training_config,
            training_metrics=metrics,
            model_path="",  # Will be set in _store_trained_model
            created_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        logger.info(f"Training completed for bag {bag_id} in {training_time:.2f}s")
        return trained_model_container
    
    def _store_trained_model(self, 
                           bag_id: int, 
                           trained_model: TrainedModel, 
                           learner_config: Dict[str, Any]):
        """Step 8: Model Storage and Serialization"""
        logger.info(f"Step 8: Storing trained model for bag {bag_id}")
        
        # Create bag-specific directory
        bag_dir = self.output_dir / f"bag_{bag_id}"
        bag_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = bag_dir / "model.pkl"
        if trained_model.learner_type == 'Random Forest':
            # Save sklearn model
            with open(model_path, 'wb') as f:
                pickle.dump(trained_model.model, f)
        else:
            # Save PyTorch model
            torch.save(trained_model.model.state_dict(), model_path)
        
        # Save training configuration
        config_path = bag_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(trained_model.training_config), f, indent=2)
        
        # Save training metrics
        metrics_path = bag_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(asdict(trained_model.training_metrics), f, indent=2)
        
        # Save learner configuration
        learner_config_path = bag_dir / "learner_config.json"
        with open(learner_config_path, 'w') as f:
            json.dump(learner_config, f, indent=2)
        
        # Update model path
        trained_model.model_path = str(model_path)
        
        # Store in memory
        self.trained_models[bag_id] = trained_model
        self.training_configs[bag_id] = trained_model.training_config
        self.training_metrics[bag_id] = trained_model.training_metrics
        
        logger.info(f"Model stored for bag {bag_id} at {model_path}")
    
    def _run_final_validation(self):
        """Step 9: Final Validation"""
        logger.info("Step 9: Running final validation")
        
        validation_results = {}
        
        for bag_id, trained_model in self.trained_models.items():
            # Load test data for this bag
            bag_specific_data = self._get_bag_test_data(bag_id)
            
            # Evaluate model
            if trained_model.learner_type == 'Random Forest':
                # Flatten test data for sklearn
                test_features = []
                for data in bag_specific_data['test_data'].values():
                    if data.ndim > 2:
                        data_flat = data.reshape(data.shape[0], -1)
                    else:
                        data_flat = data
                    test_features.append(data_flat)
                X_test = np.concatenate(test_features, axis=1)
                
                predictions = trained_model.model.predict(X_test)
                probabilities = trained_model.model.predict_proba(X_test)
            else:
                # PyTorch model evaluation
                trained_model.model.eval()
                with torch.no_grad():
                    test_loader = self._create_test_dataloader(bag_specific_data, bag_id)
                    predictions = []
                    probabilities = []
                    for batch in test_loader:
                        outputs = trained_model.model(batch)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]  # Take first output for multi-output models
                        pred = torch.argmax(outputs, dim=1)
                        prob = torch.softmax(outputs, dim=1)
                        predictions.extend(pred.cpu().numpy())
                        probabilities.extend(prob.cpu().numpy())
            
            # Calculate metrics
            test_labels = bag_specific_data['test_labels']
            accuracy = accuracy_score(test_labels, predictions)
            precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
            recall = recall_score(test_labels, predictions, average='weighted', zero_division=0)
            f1 = f1_score(test_labels, predictions, average='weighted', zero_division=0)
            
            validation_results[bag_id] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                'probabilities': probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities
            }
        
        # Save validation results
        validation_path = self.output_dir / "validation_results.json"
        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Final validation completed for {len(validation_results)} bags")
    
    def _generate_performance_report(self):
        """Step 9: Generate Performance Report"""
        logger.info("Step 9: Generating performance report")
        
        report = {
            'summary': {
                'total_bags': len(self.trained_models),
                'training_time': sum(metrics.training_time for metrics in self.training_metrics.values()),
                'average_training_time': np.mean([metrics.training_time for metrics in self.training_metrics.values()]),
                'device_used': self.device
            },
            'bag_performance': {},
            'learner_type_performance': defaultdict(list),
            'convergence_analysis': {},
            'overfitting_analysis': {}
        }
        
        for bag_id, metrics in self.training_metrics.items():
            learner_type = metrics.learner_type
            
            bag_perf = {
                'learner_type': learner_type,
                'final_train_accuracy': metrics.final_train_accuracy,
                'final_val_accuracy': metrics.final_val_accuracy,
                'best_val_accuracy': metrics.best_val_accuracy,
                'training_epochs': metrics.training_epochs,
                'training_time': metrics.training_time,
                'convergence_epoch': metrics.convergence_epoch,
                'overfitting_detected': metrics.overfitting_detected
            }
            
            report['bag_performance'][bag_id] = bag_perf
            report['learner_type_performance'][learner_type].append(bag_perf)
            
            # Convergence analysis
            if metrics.convergence_epoch > 0:
                report['convergence_analysis'][bag_id] = {
                    'converged': True,
                    'epoch': metrics.convergence_epoch,
                    'efficiency': metrics.convergence_epoch / metrics.training_epochs
                }
            else:
                report['convergence_analysis'][bag_id] = {
                    'converged': False,
                    'epoch': metrics.training_epochs
                }
            
            # Overfitting analysis
            train_val_gap = metrics.final_train_accuracy - metrics.final_val_accuracy
            report['overfitting_analysis'][bag_id] = {
                'train_val_gap': train_val_gap,
                'overfitting_detected': metrics.overfitting_detected,
                'severity': 'high' if train_val_gap > 0.1 else 'medium' if train_val_gap > 0.05 else 'low'
            }
        
        # Save performance report
        report_path = self.output_dir / "performance_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report generated and saved to {report_path}")
    
    # Helper methods for model creation
    def _create_random_forest(self, hyperparams: Dict[str, Any]):
        """Create Random Forest model."""
        return RandomForestClassifier(
            n_estimators=hyperparams.get('n_estimators', 100),
            max_depth=hyperparams.get('max_depth', None),
            min_samples_split=hyperparams.get('min_samples_split', 2),
            min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
            random_state=self.random_state
        )
    
    def _create_convnext(self, hyperparams: Dict[str, Any]):
        """Create ConvNeXt model for visual data."""
        class ConvNeXtBlock(nn.Module):
            def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
                super().__init__()
                self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
                self.norm = nn.LayerNorm(dim, eps=1e-6)
                self.pwconv1 = nn.Linear(dim, 4 * dim)
                self.act = nn.GELU()
                self.pwconv2 = nn.Linear(4 * dim, dim)
                self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
                self.drop_path = nn.Identity() if drop_path <= 0.0 else nn.Dropout(drop_path)
            
            def forward(self, x):
                input = x
                x = self.dwconv(x)
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm(x)
                x = self.pwconv1(x)
                x = self.act(x)
                x = self.pwconv2(x)
                if self.gamma is not None:
                    x = self.gamma * x
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
                x = input + self.drop_path(x)
                return x
        
        class ConvNeXt(nn.Module):
            def __init__(self, num_classes=10, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.0):
                super().__init__()
                self.downsample_layers = nn.ModuleList()
                stem = nn.Sequential(
                    nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                    nn.LayerNorm(dims[0], eps=1e-6, elementwise_affine=True)
                )
                self.downsample_layers.append(stem)
                
                for i in range(3):
                    downsample_layer = nn.Sequential(
                        nn.LayerNorm(dims[i], eps=1e-6, elementwise_affine=True),
                        nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                    )
                    self.downsample_layers.append(downsample_layer)
                
                self.stages = nn.ModuleList()
                dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
                cur = 0
                for i in range(4):
                    stage = nn.Sequential(
                        *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=1e-6) for j in range(depths[i])]
                    )
                    self.stages.append(stage)
                    cur += depths[i]
                
                self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
                self.head = nn.Linear(dims[-1], num_classes)
                
            def forward(self, x):
                for i in range(4):
                    x = self.downsample_layers[i](x)
                    x = self.stages[i](x)
                x = self.norm(x.mean([-2, -1]))  # global average pooling
                x = self.head(x)
                return x
        
        return ConvNeXt(
            num_classes=hyperparams.get('num_classes', 10),
            depths=hyperparams.get('depths', [3, 3, 9, 3]),
            dims=hyperparams.get('dims', [96, 192, 384, 768]),
            drop_path_rate=hyperparams.get('drop_path_rate', 0.0)
        )
    
    def _create_efficientnet(self, hyperparams: Dict[str, Any]):
        """Create EfficientNet model for spectral data."""
        class MBConvBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=6, se_ratio=0.25, drop_connect_rate=0.0):
                super().__init__()
                self.stride = stride
                self.expand_ratio = expand_ratio
                self.use_res_connect = self.stride == 1 and in_channels == out_channels
                
                # Expansion phase
                expand_channels = in_channels * expand_ratio
                if expand_ratio != 1:
                    self.expand_conv = nn.Conv2d(in_channels, expand_channels, kernel_size=1, bias=False)
                    self.expand_bn = nn.BatchNorm2d(expand_channels)
                    self.expand_swish = nn.SiLU()
                else:
                    self.expand_conv = None
                
                # Depthwise convolution phase
                self.depthwise_conv = nn.Conv2d(expand_channels, expand_channels, kernel_size=kernel_size, stride=stride, 
                                              padding=kernel_size//2, groups=expand_channels, bias=False)
                self.depthwise_bn = nn.BatchNorm2d(expand_channels)
                self.depthwise_swish = nn.SiLU()
                
                # Squeeze and Excitation phase
                se_channels = max(1, int(in_channels * se_ratio))
                self.se_reduce = nn.Conv2d(expand_channels, se_channels, kernel_size=1)
                self.se_expand = nn.Conv2d(se_channels, expand_channels, kernel_size=1)
                
                # Output phase
                self.project_conv = nn.Conv2d(expand_channels, out_channels, kernel_size=1, bias=False)
                self.project_bn = nn.BatchNorm2d(out_channels)
                
                self.drop_connect = nn.Dropout(drop_connect_rate) if drop_connect_rate > 0 else nn.Identity()
                
            def forward(self, x):
                identity = x
                
                # Expansion
                if self.expand_conv is not None:
                    x = self.expand_conv(x)
                    x = self.expand_bn(x)
                    x = self.expand_swish(x)
                
                # Depthwise
                x = self.depthwise_conv(x)
                x = self.depthwise_bn(x)
                x = self.depthwise_swish(x)
                
                # Squeeze and Excitation
                x_se = torch.mean(x, dim=[2, 3], keepdim=True)
                x_se = self.se_reduce(x_se)
                x_se = torch.sigmoid(x_se)
                x_se = self.se_expand(x_se)
                x = x * x_se
                
                # Project
                x = self.project_conv(x)
                x = self.project_bn(x)
                
                # Skip connection and drop connect
                if self.use_res_connect:
                    x = self.drop_connect(x)
                    x = x + identity
                
                return x
        
        class EfficientNet(nn.Module):
            def __init__(self, num_classes=10, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2, drop_connect_rate=0.2):
                super().__init__()
                
                # Calculate compound scaling
                def round_filters(filters, width_coefficient):
                    return int(filters * width_coefficient)
                
                def round_repeats(repeats, depth_coefficient):
                    return int(math.ceil(depth_coefficient * repeats))
                
                # Stem
                out_channels = round_filters(32, width_coefficient)
                self.stem = nn.Sequential(
                    nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU()
                )
                
                # Blocks
                self.blocks = nn.ModuleList()
                
                # Block 0
                self.blocks.append(MBConvBlock(32, 16, kernel_size=3, stride=1, expand_ratio=1, drop_connect_rate=drop_connect_rate))
                
                # Block 1
                self.blocks.append(MBConvBlock(16, 24, kernel_size=3, stride=2, expand_ratio=6, drop_connect_rate=drop_connect_rate))
                for _ in range(round_repeats(2, depth_coefficient) - 1):
                    self.blocks.append(MBConvBlock(24, 24, kernel_size=3, stride=1, expand_ratio=6, drop_connect_rate=drop_connect_rate))
                
                # Block 2
                self.blocks.append(MBConvBlock(24, 40, kernel_size=5, stride=2, expand_ratio=6, drop_connect_rate=drop_connect_rate))
                for _ in range(round_repeats(2, depth_coefficient) - 1):
                    self.blocks.append(MBConvBlock(40, 40, kernel_size=5, stride=1, expand_ratio=6, drop_connect_rate=drop_connect_rate))
                
                # Block 3
                self.blocks.append(MBConvBlock(40, 80, kernel_size=3, stride=2, expand_ratio=6, drop_connect_rate=drop_connect_rate))
                for _ in range(round_repeats(3, depth_coefficient) - 1):
                    self.blocks.append(MBConvBlock(80, 80, kernel_size=3, stride=1, expand_ratio=6, drop_connect_rate=drop_connect_rate))
                
                # Head
                in_channels = 80
                out_channels = round_filters(1280, width_coefficient)
                self.head = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU()
                )
                
                self.avgpool = nn.AdaptiveAvgPool2d(1)
                self.dropout = nn.Dropout(dropout_rate)
                self.classifier = nn.Linear(out_channels, num_classes)
                
            def forward(self, x):
                x = self.stem(x)
                
                for block in self.blocks:
                    x = block(x)
                
                x = self.head(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.dropout(x)
                x = self.classifier(x)
                return x
        
        return EfficientNet(
            num_classes=hyperparams.get('num_classes', 10),
            width_coefficient=hyperparams.get('width_coefficient', 1.0),
            depth_coefficient=hyperparams.get('depth_coefficient', 1.0),
            dropout_rate=hyperparams.get('dropout_rate', 0.2),
            drop_connect_rate=hyperparams.get('drop_connect_rate', 0.2)
        )
    
    def _create_1d_cnn(self, hyperparams: Dict[str, Any]):
        """Create 1D CNN ResNet model for time-series data."""
        class BasicBlock1D(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
                super().__init__()
                self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
                self.bn1 = nn.BatchNorm1d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
                self.bn2 = nn.BatchNorm1d(out_channels)
                self.downsample = downsample
                self.stride = stride
                
            def forward(self, x):
                identity = x
                
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                
                out = self.conv2(out)
                out = self.bn2(out)
                
                if self.downsample is not None:
                    identity = self.downsample(x)
                
                out += identity
                out = self.relu(out)
                
                return out
        
        class ResNet1D(nn.Module):
            def __init__(self, num_classes=10, input_channels=1, layers=[2, 2, 2, 2], base_channels=64):
                super().__init__()
                self.in_channels = base_channels
                
                # Initial convolution
                self.conv1 = nn.Conv1d(input_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
                self.bn1 = nn.BatchNorm1d(base_channels)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
                
                # ResNet layers
                self.layer1 = self._make_layer(BasicBlock1D, base_channels, layers[0])
                self.layer2 = self._make_layer(BasicBlock1D, base_channels * 2, layers[1], stride=2)
                self.layer3 = self._make_layer(BasicBlock1D, base_channels * 4, layers[2], stride=2)
                self.layer4 = self._make_layer(BasicBlock1D, base_channels * 8, layers[3], stride=2)
                
                # Global average pooling and classifier
                self.avgpool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(base_channels * 8, num_classes)
                
            def _make_layer(self, block, out_channels, blocks, stride=1):
                downsample = None
                if stride != 1 or self.in_channels != out_channels:
                    downsample = nn.Sequential(
                        nn.Conv1d(self.in_channels, out_channels, 1, stride, bias=False),
                        nn.BatchNorm1d(out_channels),
                    )
                
                layers = []
                layers.append(block(self.in_channels, out_channels, stride=stride, downsample=downsample))
                self.in_channels = out_channels
                for _ in range(1, blocks):
                    layers.append(block(out_channels, out_channels))
                
                return nn.Sequential(*layers)
            
            def forward(self, x):
                # Ensure input is 3D: (batch, channels, sequence_length)
                if x.dim() == 2:
                    x = x.unsqueeze(1)  # Add channel dimension
                
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                
                return x
        
        return ResNet1D(
            num_classes=hyperparams.get('num_classes', 10),
            input_channels=hyperparams.get('input_channels', 1),
            layers=hyperparams.get('layers', [2, 2, 2, 2]),
            base_channels=hyperparams.get('base_channels', 64)
        )
    
    def _create_fusion_network(self, learner_type: str, hyperparams: Dict[str, Any]):
        """Create fusion network model for multi-modal data."""
        
        if learner_type == 'Multi-Input ConvNeXt':
            return self._create_multi_input_convnext(hyperparams)
        elif learner_type == 'Attention-based Fusion Network':
            return self._create_attention_fusion(hyperparams)
        elif learner_type == 'Cross-modal Attention Network':
            return self._create_cross_modal_attention(hyperparams)
        elif learner_type == 'Temporal-Spatial Fusion Network':
            return self._create_temporal_spatial_fusion(hyperparams)
        elif learner_type == 'Multi-Head Attention Fusion Network':
            return self._create_multi_head_attention_fusion(hyperparams)
        else:
            # Fallback to simple fusion
            return self._create_simple_fusion(hyperparams)
    
    def _create_multi_input_convnext(self, hyperparams: Dict[str, Any]):
        """Multi-Input ConvNeXt for visual+spectral fusion."""
        class MultiInputConvNeXt(nn.Module):
            def __init__(self, num_classes=10, visual_dim=768, spectral_dim=256, hidden_dim=512):
                super().__init__()
                
                # Visual branch (ConvNeXt-like)
                self.visual_conv = nn.Sequential(
                    nn.Conv2d(3, 64, 7, 2, 3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1)
                )
                self.visual_fc = nn.Linear(64, visual_dim)
                
                # Spectral branch
                self.spectral_fc = nn.Sequential(
                    nn.Linear(spectral_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, visual_dim)
                )
                
                # Fusion
                self.fusion = nn.Sequential(
                    nn.Linear(visual_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, num_classes)
                )
                
            def forward(self, x):
                if isinstance(x, dict):
                    visual = x.get('visual', x.get('visual_rgb', None))
                    spectral = x.get('spectral', x.get('near_infrared', None))
                    
                    if visual is not None and spectral is not None:
                        # Multi-modal
                        visual_feat = self.visual_conv(visual).view(visual.size(0), -1)
                        visual_feat = self.visual_fc(visual_feat)
                        
                        spectral_feat = self.spectral_fc(spectral)
                        
                        fused = torch.cat([visual_feat, spectral_feat], dim=1)
                        return self.fusion(fused)
                    else:
                        # Single modality fallback
                        if visual is not None:
                            visual_feat = self.visual_conv(visual).view(visual.size(0), -1)
                            return self.visual_fc(visual_feat)
                        else:
                            return self.spectral_fc(spectral)
                else:
                    # Assume concatenated input
                    return self.fusion(x)
        
        return MultiInputConvNeXt(
            num_classes=hyperparams.get('num_classes', 10),
            visual_dim=hyperparams.get('visual_dim', 768),
            spectral_dim=hyperparams.get('spectral_dim', 256),
            hidden_dim=hyperparams.get('hidden_dim', 512)
        )
    
    def _create_optimizer(self, hyperparams: Dict[str, Any]):
        """Create optimizer with validation."""
        optimizer_name = hyperparams.get('optimizer', 'adam')
        learning_rate = hyperparams.get('learning_rate', 0.001)
        weight_decay = hyperparams.get('weight_decay', 0.0001)
        
        # Validate hyperparameters
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError(f"Invalid learning_rate: {learning_rate}. Must be a positive number.")
        
        if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}. Must be a non-negative number.")
        
        if learning_rate > 1.0:
            logger.warning(f"Learning rate {learning_rate} is very high. Consider using a smaller value.")
        
        if optimizer_name.lower() == 'adam':
            return lambda model: optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            return lambda model: optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        else:
            logger.warning(f"Unknown optimizer '{optimizer_name}', defaulting to Adam")
            return lambda model: optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    def _create_loss_function(self, hyperparams: Dict[str, Any]):
        """Create loss function."""
        loss_name = hyperparams.get('loss_function', 'cross_entropy')
        
        if loss_name.lower() == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif loss_name.lower() == 'mse':
            return nn.MSELoss()
        else:
            return nn.CrossEntropyLoss()
    
    def _create_scheduler(self, hyperparams: Dict[str, Any]):
        """Create learning rate scheduler."""
        scheduler_name = hyperparams.get('scheduler', 'step')
        
        if scheduler_name.lower() == 'step':
            return lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif scheduler_name.lower() == 'cosine':
            return lambda optimizer: optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        else:
            return lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    def _create_attention_fusion(self, hyperparams: Dict[str, Any]):
        """Attention-based fusion for tabular+time-series."""
        class AttentionFusion(nn.Module):
            def __init__(self, num_classes=10, tabular_dim=64, temporal_dim=128, hidden_dim=256, num_heads=8):
                super().__init__()
                
                # Tabular encoder
                self.tabular_encoder = nn.Sequential(
                    nn.Linear(tabular_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                
                # Temporal encoder (1D CNN)
                self.temporal_encoder = nn.Sequential(
                    nn.Conv1d(1, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(32, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                
                # Multi-head attention
                self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
                self.norm = nn.LayerNorm(hidden_dim)
                
                # Classifier
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, num_classes)
                )
                
            def forward(self, x):
                if isinstance(x, dict):
                    tabular = x.get('tabular', x.get('tabular_features', None))
                    temporal = x.get('temporal', x.get('time_series', None))
                    
                    if tabular is not None and temporal is not None:
                        # Encode modalities
                        tabular_feat = self.tabular_encoder(tabular)
                        temporal_feat = self.temporal_encoder(temporal.unsqueeze(1))
                        
                        # Attention fusion
                        combined = torch.stack([tabular_feat, temporal_feat], dim=1)
                        attended, _ = self.attention(combined, combined, combined)
                        attended = self.norm(attended + combined)
                        
                        # Flatten and classify
                        fused = attended.view(attended.size(0), -1)
                        return self.classifier(fused)
                    else:
                        # Single modality fallback
                        if tabular is not None:
                            return self.tabular_encoder(tabular)
                        else:
                            return self.temporal_encoder(temporal.unsqueeze(1))
                else:
                    # Assume concatenated input
                    return self.classifier(x)
        
        return AttentionFusion(
            num_classes=hyperparams.get('num_classes', 10),
            tabular_dim=hyperparams.get('tabular_dim', 64),
            temporal_dim=hyperparams.get('temporal_dim', 128),
            hidden_dim=hyperparams.get('hidden_dim', 256),
            num_heads=hyperparams.get('num_heads', 8)
        )
    
    def _create_cross_modal_attention(self, hyperparams: Dict[str, Any]):
        """Cross-modal attention for tabular+visual."""
        class CrossModalAttention(nn.Module):
            def __init__(self, num_classes=10, tabular_dim=64, visual_dim=512, hidden_dim=256, num_heads=8):
                super().__init__()
                
                # Tabular encoder
                self.tabular_encoder = nn.Sequential(
                    nn.Linear(tabular_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                
                # Visual encoder
                self.visual_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 7, 2, 3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(64, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                
                # Cross-modal attention
                self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
                self.norm1 = nn.LayerNorm(hidden_dim)
                self.norm2 = nn.LayerNorm(hidden_dim)
                
                # Classifier
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, num_classes)
                )
                
            def forward(self, x):
                if isinstance(x, dict):
                    tabular = x.get('tabular', x.get('tabular_features', None))
                    visual = x.get('visual', x.get('visual_rgb', None))
                    
                    if tabular is not None and visual is not None:
                        # Encode modalities
                        tabular_feat = self.tabular_encoder(tabular)
                        visual_feat = self.visual_encoder(visual)
                        
                        # Cross-modal attention
                        tabular_attended, _ = self.cross_attention(
                            tabular_feat.unsqueeze(1), visual_feat.unsqueeze(1), visual_feat.unsqueeze(1)
                        )
                        visual_attended, _ = self.cross_attention(
                            visual_feat.unsqueeze(1), tabular_feat.unsqueeze(1), tabular_feat.unsqueeze(1)
                        )
                        
                        tabular_attended = self.norm1(tabular_attended.squeeze(1) + tabular_feat)
                        visual_attended = self.norm2(visual_attended.squeeze(1) + visual_feat)
                        
                        # Fuse and classify
                        fused = torch.cat([tabular_attended, visual_attended], dim=1)
                        return self.classifier(fused)
                    else:
                        # Single modality fallback
                        if tabular is not None:
                            return self.tabular_encoder(tabular)
                        else:
                            return self.visual_encoder(visual)
                else:
                    # Assume concatenated input
                    return self.classifier(x)
        
        return CrossModalAttention(
            num_classes=hyperparams.get('num_classes', 10),
            tabular_dim=hyperparams.get('tabular_dim', 64),
            visual_dim=hyperparams.get('visual_dim', 512),
            hidden_dim=hyperparams.get('hidden_dim', 256),
            num_heads=hyperparams.get('num_heads', 8)
        )
    
    def _create_temporal_spatial_fusion(self, hyperparams: Dict[str, Any]):
        """Temporal-spatial fusion for time-series+visual."""
        class TemporalSpatialFusion(nn.Module):
            def __init__(self, num_classes=10, temporal_dim=128, visual_dim=512, hidden_dim=256):
                super().__init__()
                
                # Temporal encoder (1D CNN)
                self.temporal_encoder = nn.Sequential(
                    nn.Conv1d(1, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(64, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                
                # Spatial encoder (2D CNN)
                self.spatial_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 7, 2, 3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(64, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                
                # Temporal-spatial fusion
                self.fusion = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, num_classes)
                )
                
            def forward(self, x):
                if isinstance(x, dict):
                    temporal = x.get('temporal', x.get('time_series', None))
                    spatial = x.get('spatial', x.get('visual', x.get('visual_rgb', None)))
                    
                    if temporal is not None and spatial is not None:
                        # Encode both modalities
                        temporal_feat = self.temporal_encoder(temporal.unsqueeze(1))
                        spatial_feat = self.spatial_encoder(spatial)
                        
                        # Fuse and classify
                        fused = torch.cat([temporal_feat, spatial_feat], dim=1)
                        return self.fusion(fused)
                    else:
                        # Single modality fallback
                        if temporal is not None:
                            return self.temporal_encoder(temporal.unsqueeze(1))
                        else:
                            return self.spatial_encoder(spatial)
                else:
                    # Assume concatenated input
                    return self.fusion(x)
        
        return TemporalSpatialFusion(
            num_classes=hyperparams.get('num_classes', 10),
            temporal_dim=hyperparams.get('temporal_dim', 128),
            visual_dim=hyperparams.get('visual_dim', 512),
            hidden_dim=hyperparams.get('hidden_dim', 256)
        )
    
    def _create_multi_head_attention_fusion(self, hyperparams: Dict[str, Any]):
        """Multi-head attention fusion for 3+ modalities."""
        class MultiHeadAttentionFusion(nn.Module):
            def __init__(self, num_classes=10, modality_dims=[64, 128, 256], hidden_dim=256, num_heads=8):
                super().__init__()
                
                # Modality encoders
                self.encoders = nn.ModuleList()
                for dim in modality_dims:
                    encoder = nn.Sequential(
                        nn.Linear(dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    )
                    self.encoders.append(encoder)
                
                # Multi-head attention
                self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
                self.norm = nn.LayerNorm(hidden_dim)
                
                # Classifier
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, num_classes)
                )
                
            def forward(self, x):
                if isinstance(x, dict):
                    # Encode all modalities
                    encoded_modalities = []
                    for i, (key, value) in enumerate(x.items()):
                        if i < len(self.encoders):
                            encoded = self.encoders[i](value)
                            encoded_modalities.append(encoded)
                    
                    if len(encoded_modalities) > 1:
                        # Stack for attention
                        stacked = torch.stack(encoded_modalities, dim=1)
                        
                        # Multi-head attention
                        attended, _ = self.attention(stacked, stacked, stacked)
                        attended = self.norm(attended + stacked)
                        
                        # Global average pooling across modalities
                        fused = attended.mean(dim=1)
                        return self.classifier(fused)
                    else:
                        # Single modality
                        return self.classifier(encoded_modalities[0])
                else:
                    # Assume concatenated input
                    return self.classifier(x)
        
        return MultiHeadAttentionFusion(
            num_classes=hyperparams.get('num_classes', 10),
            modality_dims=hyperparams.get('modality_dims', [64, 128, 256]),
            hidden_dim=hyperparams.get('hidden_dim', 256),
            num_heads=hyperparams.get('num_heads', 8)
        )
    
    def _create_simple_fusion(self, hyperparams: Dict[str, Any]):
        """Simple fusion fallback."""
        class SimpleFusion(nn.Module):
            def __init__(self, num_classes=10, input_dim=512, hidden_dim=256):
                super().__init__()
                self.fusion = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, num_classes)
                )
            
            def forward(self, x):
                if isinstance(x, dict):
                    # Concatenate all modalities
                    x = torch.cat([v for v in x.values()], dim=1)
                return self.fusion(x)
        
        return SimpleFusion(
            num_classes=hyperparams.get('num_classes', 10),
            input_dim=hyperparams.get('input_dim', 512),
            hidden_dim=hyperparams.get('hidden_dim', 256)
        )
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage and available resources."""
        memory_info = {
            'device': self.device,
            'cpu_memory_available': True,
            'gpu_memory_available': False,
            'gpu_memory_allocated': 0,
            'gpu_memory_cached': 0,
            'gpu_memory_total': 0
        }
        
        try:
            if torch.cuda.is_available() and self.device != 'cpu':
                memory_info['gpu_memory_available'] = True
                memory_info['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_info['gpu_memory_cached'] = torch.cuda.memory_reserved() / 1024**3  # GB
                memory_info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                
                # Log memory status
                logger.info(f"GPU Memory - Allocated: {memory_info['gpu_memory_allocated']:.2f}GB, "
                          f"Cached: {memory_info['gpu_memory_cached']:.2f}GB, "
                          f"Total: {memory_info['gpu_memory_total']:.2f}GB")
            else:
                logger.info("Using CPU for training")
                
        except Exception as e:
            logger.warning(f"Could not check GPU memory: {e}")
            
        return memory_info
    
    def _adjust_batch_size_for_memory(self, original_batch_size: int, data_size: int, learner_type: str) -> int:
        """Adjust batch size based on available memory and model complexity."""
        try:
            memory_info = self._check_memory_usage()
            adjusted_batch_size = original_batch_size
            
            if memory_info['gpu_memory_available']:
                # GPU memory management
                available_memory = memory_info['gpu_memory_total'] - memory_info['gpu_memory_allocated']
                
                # Model complexity factors
                complexity_factors = {
                    'ConvNeXt-Base': 2.0,
                    'EfficientNet B4': 1.5,
                    '1D-CNN ResNet Style': 1.0,
                    'Multi-Input ConvNeXt': 2.5,
                    'Attention-based Fusion Network': 2.0,
                    'Cross-modal Attention Network': 2.0,
                    'Temporal-Spatial Fusion Network': 2.0,
                    'Multi-Head Attention Fusion Network': 2.5,
                    'Random Forest': 0.1  # sklearn models don't use GPU memory
                }
                
                complexity = complexity_factors.get(learner_type, 1.0)
                
                # Adjust based on available memory
                if available_memory < 2.0:  # Less than 2GB available
                    adjusted_batch_size = max(1, original_batch_size // 4)
                elif available_memory < 4.0:  # Less than 4GB available
                    adjusted_batch_size = max(1, original_batch_size // 2)
                elif complexity > 2.0:  # High complexity models
                    adjusted_batch_size = max(1, original_batch_size // 2)
                
                # Ensure batch size doesn't exceed data size
                adjusted_batch_size = min(adjusted_batch_size, data_size)
                
                if adjusted_batch_size != original_batch_size:
                    logger.info(f"Adjusted batch size from {original_batch_size} to {adjusted_batch_size} "
                              f"for {learner_type} (available GPU memory: {available_memory:.2f}GB)")
            
            return adjusted_batch_size
            
        except Exception as e:
            logger.warning(f"Could not adjust batch size: {e}")
            return original_batch_size
    
    def _validate_architecture_inputs(self, learner_type: str, hyperparams: Dict[str, Any], train_data: Dict[str, np.ndarray]) -> None:
        """Validate that data matches architecture requirements."""
        try:
            if learner_type == 'ConvNeXt-Base':
                # Check for visual data with proper shape
                visual_keys = [k for k in train_data.keys() if 'visual' in k.lower() or 'rgb' in k.lower()]
                if not visual_keys:
                    logger.warning(f"ConvNeXt expects visual data but none found in {list(train_data.keys())}")
                    return
                
                for key in visual_keys:
                    data = train_data[key]
                    if data.ndim != 4:
                        raise ValueError(f"ConvNeXt expects 4D visual input (N,C,H,W), got {data.ndim}D for {key}")
                    if data.shape[1] != 3:
                        logger.warning(f"ConvNeXt expects 3-channel RGB input, got {data.shape[1]} channels for {key}")
            
            elif learner_type == 'EfficientNet B4':
                # Check for spectral/visual data
                spectral_keys = [k for k in train_data.keys() if 'spectral' in k.lower() or 'infrared' in k.lower() or 'visual' in k.lower()]
                if not spectral_keys:
                    logger.warning(f"EfficientNet expects spectral/visual data but none found in {list(train_data.keys())}")
                    return
                
                # Validate compound scaling parameters
                width_coeff = hyperparams.get('width_coefficient', 1.0)
                depth_coeff = hyperparams.get('depth_coefficient', 1.0)
                if not (0.1 <= width_coeff <= 3.0):
                    logger.warning(f"EfficientNet width_coefficient should be in [0.1, 3.0], got {width_coeff}")
                if not (0.1 <= depth_coeff <= 3.0):
                    logger.warning(f"EfficientNet depth_coefficient should be in [0.1, 3.0], got {depth_coeff}")
            
            elif learner_type == '1D-CNN ResNet Style':
                # Check for time-series data
                temporal_keys = [k for k in train_data.keys() if 'time' in k.lower() or 'series' in k.lower() or 'eeg' in k.lower()]
                if not temporal_keys:
                    logger.warning(f"1D-CNN expects time-series data but none found in {list(train_data.keys())}")
                    return
                
                for key in temporal_keys:
                    data = train_data[key]
                    if data.ndim < 2:
                        raise ValueError(f"1D-CNN expects at least 2D time-series input (N, features), got {data.ndim}D for {key}")
            
            elif 'Fusion' in learner_type or 'Multi-Input' in learner_type:
                # Check for multiple modalities
                if len(train_data) < 2:
                    logger.warning(f"Fusion network expects multiple modalities, got {len(train_data)} modalities")
                
                # Validate modality dimensions for fusion networks
                modality_dims = hyperparams.get('modality_dims', [])
                if modality_dims and len(modality_dims) != len(train_data):
                    logger.warning(f"Fusion network modality_dims length ({len(modality_dims)}) doesn't match data modalities ({len(train_data)})")
            
            # General validation
            for key, data in train_data.items():
                if data.size == 0:
                    raise ValueError(f"Empty data found for modality {key}")
                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    logger.warning(f"NaN or Inf values found in modality {key}")
                
        except Exception as e:
            logger.error(f"Architecture validation failed for {learner_type}: {e}")
            # Don't raise - just log the warning and continue
    
    def _create_multimodal_tensor(self, train_data: Dict[str, np.ndarray], learner_type: str, bag_id: int) -> torch.Tensor:
        """Create properly structured multi-modal tensor for fusion networks."""
        try:
            if learner_type == 'Multi-Input ConvNeXt':
                # Handle visual + spectral fusion
                visual_data = None
                spectral_data = None
                
                for modality_name, data in train_data.items():
                    if 'visual' in modality_name.lower() or 'rgb' in modality_name.lower():
                        visual_data = torch.FloatTensor(data)
                    elif 'spectral' in modality_name.lower() or 'infrared' in modality_name.lower():
                        spectral_data = torch.FloatTensor(data)
                
                if visual_data is not None and spectral_data is not None:
                    # Return as tuple for multi-input model
                    return (visual_data, spectral_data)
                else:
                    # Fallback to concatenation
                    X_list = [torch.FloatTensor(data.reshape(data.shape[0], -1)) for data in train_data.values()]
                    return torch.cat(X_list, dim=1)
            
            elif 'Attention' in learner_type:
                # For attention-based fusion, concatenate flattened data
                X_list = []
                for modality_name, data in train_data.items():
                    if data.ndim > 2:
                        data_flat = data.reshape(data.shape[0], -1)
                    else:
                        data_flat = data
                    X_list.append(torch.FloatTensor(data_flat))
                
                return torch.cat(X_list, dim=1)
            
            else:
                # Default: concatenate all modalities
                X_list = []
                for modality_name, data in train_data.items():
                    if data.ndim > 2:
                        data_flat = data.reshape(data.shape[0], -1)
                    else:
                        data_flat = data
                    X_list.append(torch.FloatTensor(data_flat))
                
                return torch.cat(X_list, dim=1)
                
        except Exception as e:
            logger.error(f"Error creating multi-modal tensor for {learner_type}: {e}")
            # Fallback to simple concatenation
            X_list = []
            for modality_name, data in train_data.items():
                data_flat = data.reshape(data.shape[0], -1)
                X_list.append(torch.FloatTensor(data_flat))
            return torch.cat(X_list, dim=1)

    def _create_pytorch_dataloaders(self, bag_specific_data: Dict[str, Any], bag_id: int) -> Dict[str, DataLoader]:
        """Create PyTorch DataLoaders for training and validation."""
        try:
            train_data = bag_specific_data['train_data']
            train_labels = bag_specific_data['train_labels']
            
            # Validate input data
            if not train_data or len(train_data) == 0:
                raise ValueError(f"No training data found for bag {bag_id}")
            
            if len(train_labels) == 0:
                raise ValueError(f"No training labels found for bag {bag_id}")
            
            # Validate data consistency
            sample_counts = [len(data) for data in train_data.values()]
            if not all(count == sample_counts[0] for count in sample_counts):
                raise ValueError(f"Inconsistent sample counts across modalities for bag {bag_id}: {sample_counts}")
            
            if len(train_labels) != sample_counts[0]:
                raise ValueError(f"Label count ({len(train_labels)}) doesn't match data count ({sample_counts[0]}) for bag {bag_id}")
            
            # Handle multi-modal data with proper structure preservation
            learner_type = self.learner_configs[bag_id]['learner_type']
            
            if len(train_data) == 1:
                # Single modality - preserve structure for appropriate models
                modality_name = list(train_data.keys())[0]
                data = train_data[modality_name]
                
                # Shape validation
                if data.ndim == 0:
                    raise ValueError(f"Scalar data not supported for modality {modality_name} in bag {bag_id}")
                
                # Preserve structure for visual models, flatten for others
                if learner_type in ['ConvNeXt-Base', 'EfficientNet B4'] and data.ndim == 4:
                    # Keep 4D structure for visual models
                    X = torch.FloatTensor(data)
                    logger.info(f"Preserved 4D structure for {modality_name}: {X.shape} for {learner_type}")
                elif learner_type == '1D-CNN ResNet Style' and data.ndim == 3:
                    # Keep 3D structure for 1D CNN
                    X = torch.FloatTensor(data)
                    logger.info(f"Preserved 3D structure for {modality_name}: {X.shape} for {learner_type}")
                else:
                    # Flatten for other models
                    data_flat = data.reshape(data.shape[0], -1)
                    X = torch.FloatTensor(data_flat)
                    logger.info(f"Flattened {modality_name} from {data.shape} to {X.shape} for {learner_type}")
            else:
                # Multi-modal: handle based on learner type
                if 'Fusion' in learner_type or 'Multi-Input' in learner_type:
                    # For fusion networks, preserve structure and handle separately
                    X = self._create_multimodal_tensor(train_data, learner_type, bag_id)
                else:
                    # For non-fusion models, concatenate flattened data
                    X_list = []
                    for modality_name, data in train_data.items():
                        if data.ndim == 0:
                            raise ValueError(f"Scalar data not supported for modality {modality_name} in bag {bag_id}")
                        
                        # Flatten all modalities for concatenation
                        data_flat = data.reshape(data.shape[0], -1)
                        X_list.append(torch.FloatTensor(data_flat))
                    
                    X = torch.cat(X_list, dim=1)
                    logger.info(f"Concatenated {len(train_data)} modalities into shape {X.shape} for {learner_type}")
            
            y = torch.LongTensor(train_labels)
            
            # Validate final tensor shapes
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"Feature tensor shape {X.shape} doesn't match label tensor shape {y.shape} for bag {bag_id}")
            
            # Create train/validation split
            dataset_size = len(X)
            validation_split = self.learner_configs[bag_id]['hyperparameters'].get('validation_split', 0.2)
            val_size = int(validation_split * dataset_size)
            train_size = dataset_size - val_size
            
            # Ensure minimum training samples
            if train_size < 1:
                raise ValueError(f"Not enough samples for training in bag {bag_id}: {dataset_size} total, {train_size} train")
            
            # Create datasets
            train_dataset = TensorDataset(X[:train_size], y[:train_size])
            val_dataset = TensorDataset(X[train_size:], y[train_size:]) if val_size > 0 else None
            
            # Create data loaders
            batch_size = self.learner_configs[bag_id]['hyperparameters'].get('batch_size', 32)
            train_loader = DataLoader(train_dataset, batch_size=min(batch_size, train_size), shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=min(batch_size, val_size), shuffle=False) if val_dataset else None
            
            result = {'train': train_loader}
            if val_loader:
                result['val'] = val_loader
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating PyTorch data loaders for bag {bag_id}: {str(e)}")
            raise
    
    def _train_sklearn_model(self, model, bag_specific_data: Dict[str, Any], training_config: TrainingConfig) -> Tuple[Any, TrainingMetrics]:
        """Train sklearn model (Random Forest)."""
        train_data = bag_specific_data['train_data']
        train_labels = bag_specific_data['train_labels']
        test_data = bag_specific_data['test_data']
        test_labels = bag_specific_data['test_labels']
        
        # Flatten data for sklearn - handle different shapes
        train_features = []
        for data in train_data.values():
            if data.ndim > 2:
                # Flatten high-dimensional data (e.g., images)
                data_flat = data.reshape(data.shape[0], -1)
            else:
                data_flat = data
            train_features.append(data_flat)
        
        test_features = []
        for data in test_data.values():
            if data.ndim > 2:
                # Flatten high-dimensional data (e.g., images)
                data_flat = data.reshape(data.shape[0], -1)
            else:
                data_flat = data
            test_features.append(data_flat)
        
        X_train = np.concatenate(train_features, axis=1)
        X_test = np.concatenate(test_features, axis=1)
        
        # Train model
        model.fit(X_train, train_labels)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_accuracy = accuracy_score(train_labels, train_pred)
        test_accuracy = accuracy_score(test_labels, test_pred)
        
        # Create metrics
        metrics = TrainingMetrics(
            bag_id=training_config.bag_id,
            learner_type=training_config.learner_type,
            final_train_loss=0.0,  # Not applicable for sklearn
            final_val_loss=0.0,    # Not applicable for sklearn
            final_train_accuracy=train_accuracy,
            final_val_accuracy=test_accuracy,
            best_val_accuracy=test_accuracy,
            training_epochs=1,  # sklearn trains in one go
            training_time=0.0,  # Will be set by caller
            convergence_epoch=1,
            overfitting_detected=train_accuracy - test_accuracy > 0.1,
            performance_metrics={
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy
            }
        )
        
        return model, metrics
    
    def _train_pytorch_model(self, model, training_components: Dict[str, Any], bag_specific_data: Dict[str, Any], training_config: TrainingConfig) -> Tuple[Any, TrainingMetrics]:
        """Train PyTorch model."""
        model = model.to(self.device)
        
        # Get training components
        optimizer = training_components['optimizer'](model)
        criterion = training_components['loss_function']
        scheduler = training_components['scheduler'](optimizer)
        
        # Get data loaders
        train_loader = bag_specific_data['data_loaders']['train']
        val_loader = bag_specific_data['data_loaders'].get('val', None)
        
        # Training loop
        best_val_accuracy = 0.0
        convergence_epoch = 0
        overfitting_detected = False
        early_stopping_counter = 0
        
        for epoch in range(training_config.training_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_data, batch_y in train_loader:
                batch_y = batch_y.to(self.device)
                
                # Handle multi-input models
                if isinstance(batch_data, tuple):
                    # Multi-input model (e.g., Multi-Input ConvNeXt)
                    batch_x = [x.to(self.device) for x in batch_data]
                else:
                    # Single input model
                    batch_x = batch_data.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_accuracy = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase (using proper validation data if available)
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_data, batch_y in val_loader:
                        batch_y = batch_y.to(self.device)
                        
                        # Handle multi-input models
                        if isinstance(batch_data, tuple):
                            batch_x = [x.to(self.device) for x in batch_data]
                        else:
                            batch_x = batch_data.to(self.device)
                        
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                
                val_accuracy = val_correct / val_total
                avg_val_loss = val_loss / len(val_loader)
            else:
                # No validation data available, use training data for validation
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_data, batch_y in train_loader:
                        batch_y = batch_y.to(self.device)
                        
                        # Handle multi-input models
                        if isinstance(batch_data, tuple):
                            batch_x = [x.to(self.device) for x in batch_data]
                        else:
                            batch_x = batch_data.to(self.device)
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                
                val_accuracy = val_correct / val_total
                avg_val_loss = val_loss / len(train_loader)
            
            # Early stopping check
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                convergence_epoch = epoch
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter >= training_config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch} for bag {training_config.bag_id}")
                break
            
            # Check for overfitting
            if train_accuracy - val_accuracy > 0.1:
                overfitting_detected = True
            
            # Update scheduler
            scheduler.step()
        
        # Create metrics
        metrics = TrainingMetrics(
            bag_id=training_config.bag_id,
            learner_type=training_config.learner_type,
            final_train_loss=avg_train_loss,
            final_val_loss=avg_val_loss,
            final_train_accuracy=train_accuracy,
            final_val_accuracy=val_accuracy,
            best_val_accuracy=best_val_accuracy,
            training_epochs=training_config.training_epochs,
            training_time=0.0,  # Will be set by caller
            convergence_epoch=convergence_epoch,
            overfitting_detected=overfitting_detected,
            performance_metrics={
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }
        )
        
        return model, metrics
    
    def _get_bag_test_data(self, bag_id: int) -> Dict[str, Any]:
        """Get test data for a specific bag."""
        bag_info = self.bag_data[bag_id]
        modality_mask = bag_info['modality_mask']
        
        bag_test_data = {}
        for modality, data in self.test_data.items():
            if modality_mask.get(modality, False):
                bag_test_data[modality] = data
        
        return {
            'test_data': bag_test_data,
            'test_labels': self.test_labels
        }
    
    def _create_test_dataloader(self, bag_specific_data: Dict[str, Any], bag_id: int) -> DataLoader:
        """Create test DataLoader for a single bag with multi-modal support."""
        test_data = bag_specific_data['test_data']
        test_labels = bag_specific_data['test_labels']
        learner_type = self.learner_configs[bag_id]['learner_type']
        
        # Use same logic as training data preparation for consistency
        if len(test_data) == 1:
            # Single modality - preserve structure for appropriate models
            modality_name = list(test_data.keys())[0]
            data = test_data[modality_name]
            
            if learner_type in ['ConvNeXt-Base', 'EfficientNet B4'] and data.ndim == 4:
                # Keep 4D structure for visual models
                X = torch.FloatTensor(data)
            elif learner_type == '1D-CNN ResNet Style' and data.ndim == 3:
                # Keep 3D structure for 1D CNN
                X = torch.FloatTensor(data)
            else:
                # Flatten for other models
                if data.ndim > 2:
                    data_flat = data.reshape(data.shape[0], -1)
                else:
                    data_flat = data
                X = torch.FloatTensor(data_flat)
        else:
            # Multi-modal: handle based on learner type
            if 'Fusion' in learner_type or 'Multi-Input' in learner_type:
                # For fusion networks, preserve structure and handle separately
                X = self._create_multimodal_tensor(test_data, learner_type, bag_id)
            else:
                # For non-fusion models, concatenate flattened data
                X_list = []
                for modality_name, data in test_data.items():
                    if data.ndim > 2:
                        data_flat = data.reshape(data.shape[0], -1)
                    else:
                        data_flat = data
                    X_list.append(torch.FloatTensor(data_flat))
                X = torch.cat(X_list, dim=1)
        
        y = torch.LongTensor(test_labels)
        dataset = TensorDataset(X, y)
        batch_size = self.learner_configs[bag_id]['hyperparameters'].get('batch_size', 32)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    def get_trained_models(self) -> Dict[int, TrainedModel]:
        """Get all trained models."""
        return self.trained_models
    
    def get_training_metrics(self) -> Dict[int, TrainingMetrics]:
        """Get all training metrics."""
        return self.training_metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.trained_models:
            return {}
        
        total_bags = len(self.trained_models)
        total_time = sum(metrics.training_time for metrics in self.training_metrics.values())
        avg_accuracy = np.mean([metrics.final_val_accuracy for metrics in self.training_metrics.values()])
        
        learner_types = {}
        for metrics in self.training_metrics.values():
            learner_type = metrics.learner_type
            if learner_type not in learner_types:
                learner_types[learner_type] = []
            learner_types[learner_type].append(metrics.final_val_accuracy)
        
        return {
            'total_bags': total_bags,
            'total_training_time': total_time,
            'average_training_time': total_time / total_bags,
            'average_accuracy': avg_accuracy,
            'learner_type_performance': {k: np.mean(v) for k, v in learner_types.items()},
            'device_used': self.device
        }
