"""
mainModel.py
Production-ready entry point for the full multimodal ensemble pipeline.
Provides a unified sklearn-like API for data integration, training, prediction, and evaluation.
"""

import numpy as np
import pickle
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from types import SimpleNamespace
import warnings

# Import all required modules with proper error handling
try:
    from . import dataIntegration
    from . import modalityDropoutBagger
    from . import modalityAwareBaseLearnerSelector
    from . import trainingPipeline
    from . import ensemblePrediction
    from . import performanceMetrics
except ImportError:
    # Fallback for direct execution
    import dataIntegration
    import modalityDropoutBagger
    import modalityAwareBaseLearnerSelector
    import trainingPipeline
    import ensemblePrediction
    import performanceMetrics

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Some features may be limited.")


class MultiModalEnsembleModel:
    """
    MultiModal Ensemble Model with sklearn-like interface.
    
    This model provides a unified interface for multimodal ensemble learning,
    supporting both classification and regression tasks with automatic
    modality-aware architecture selection and ensemble generation.
    
    Parameters
    ----------
    n_bags : int, default=10
        Number of ensemble bags to create.
    dropout_strategy : str, default='adaptive'
        Modality dropout strategy. Options: 'linear', 'exponential', 'random', 'adaptive'.
    epochs : int, default=10
        Number of training epochs for neural network learners.
    batch_size : int, default=32
        Batch size for training.
    random_state : int, default=42
        Random seed for reproducibility.
    task_type : str, default='auto'
        Task type. Options: 'auto', 'classification', 'regression'.
        If 'auto', will be determined from data.
    optimization_strategy : str, default='balanced'
        Optimization strategy for learner selection.
    enable_denoising : bool, default=True
        Whether to enable cross-modal denoising during training.
    aggregation_strategy : str, default='transformer_fusion'
        Ensemble aggregation strategy.
    uncertainty_method : str, default='entropy'
        Uncertainty quantification method.
    
    Attributes
    ----------
    trained_learners_ : list
        List of trained ensemble learners.
    learner_metadata_ : list
        Metadata for each learner.
    ensemble_ : EnsemblePredictor
        Ensemble predictor instance.
    modality_configs_ : list
        Configuration for each modality.
    n_classes_ : int
        Number of classes (for classification).
    n_features_ : int
        Number of features.
    is_multilabel_ : bool
        Whether the task is multi-label classification.
    task_type_ : str
        Determined task type.
    is_fitted_ : bool
        Whether the model has been fitted.
    
    Examples
    --------
    >>> from MainModel.mainModel import MultiModalEnsembleModel
    >>> import numpy as np
    >>> 
    >>> # Create synthetic data
    >>> n_samples = 1000
    >>> n_features = 10
    >>> 
    >>> # Create multimodal data
    >>> text_data = np.random.randn(n_samples, n_features)
    >>> image_data = np.random.randn(n_samples, n_features)
    >>> metadata = np.random.randn(n_samples, n_features)
    >>> 
    >>> # Create labels (classification example)
    >>> labels = np.random.randint(0, 3, n_samples)
    >>> 
    >>> # Initialize model
    >>> model = MultiModalEnsembleModel(n_bags=5, random_state=42)
    >>> 
    >>> # Fit model
    >>> model.fit({
    ...     'text': text_data,
    ...     'image': image_data,
    ...     'metadata': metadata
    ... }, labels)
    >>> 
    >>> # Make predictions
    >>> predictions = model.predict({
    ...     'text': text_data[:10],
    ...     'image': image_data[:10],
    ...     'metadata': metadata[:10]
    ... })
    >>> 
    >>> # For classification, get probabilities
    >>> probabilities = model.predict_proba({
    ...     'text': text_data[:10],
    ...     'image': image_data[:10],
    ...     'metadata': metadata[:10]
    ... })
    """
    
    def __init__(self, n_bags=10, dropout_strategy='adaptive', epochs=10, 
                 batch_size=32, random_state=42, task_type='auto',
                 optimization_strategy='balanced', enable_denoising=True,
                 aggregation_strategy='transformer_fusion', uncertainty_method='entropy',
                 max_dropout_rate=0.5, min_modalities=1, sample_ratio=0.8,
                 diversity_target=0.7, feature_sampling=True, verbose=True,
                 # DataIntegration parameters
                 validate_data=True, memory_efficient=False,
                 handle_nan='fill_mean', handle_inf='fill_max',
                 normalize_data=False, remove_outliers=False, outlier_std=3.0,
                 # ModalityDropoutBagger parameters
                 enable_validation=True,
                 # ModalityAwareBaseLearnerSelector parameters
                 learner_preferences=None, performance_threshold=0.7,
                 resource_limit=None, validation_strategy='cross_validation',
                 hyperparameter_tuning=False,
                 # TrainingPipeline parameters
                 optimizer_type='adamw', scheduler_type='cosine_restarts',
                 mixed_precision=True, gradient_clipping=1.0,
                 early_stopping_patience=10, validation_split=0.2,
                 cross_validation_folds=5, save_checkpoints=True,
                 tensorboard_logging=False, wandb_logging=False,
                 log_interval=10, eval_interval=1, profile_training=False,
                 gradient_accumulation_steps=1, num_workers=4,
                 distributed_training=False, compile_model=False,
                 amp_optimization_level='O1', gradient_scaling=True,
                 loss_scale='dynamic', curriculum_stages=None,
                 enable_progressive_learning=False, progressive_stages=None,
                 dropout_rate=0.2, label_smoothing=0.1,
                 denoising_weight=0.1, denoising_strategy='adaptive',
                 denoising_objectives=None, denoising_modalities=None,
                 # EnsemblePrediction parameters
                 calibrate_uncertainty=True, device='auto',
                 # PerformanceMetrics parameters
                 measure_efficiency=True, n_efficiency_runs=10, return_probabilities=True):
        
        # Core parameters
        self.n_bags = n_bags
        self.dropout_strategy = dropout_strategy
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.task_type = task_type
        self.optimization_strategy = optimization_strategy
        self.enable_denoising = enable_denoising
        self.aggregation_strategy = aggregation_strategy
        self.uncertainty_method = uncertainty_method
        self.verbose = verbose
        
        # Ensemble generation parameters
        self.max_dropout_rate = max_dropout_rate
        self.min_modalities = min_modalities
        self.sample_ratio = sample_ratio
        self.diversity_target = diversity_target
        self.feature_sampling = feature_sampling
        
        # DataIntegration parameters
        self.validate_data = validate_data
        self.memory_efficient = memory_efficient
        self.handle_nan = handle_nan
        self.handle_inf = handle_inf
        self.normalize_data = normalize_data
        self.remove_outliers = remove_outliers
        self.outlier_std = outlier_std
        
        # ModalityDropoutBagger parameters
        self.enable_validation = enable_validation
        
        # ModalityAwareBaseLearnerSelector parameters
        self.learner_preferences = learner_preferences or {}
        self.performance_threshold = performance_threshold
        self.resource_limit = resource_limit or {}
        self.validation_strategy = validation_strategy
        self.hyperparameter_tuning = hyperparameter_tuning
        
        # TrainingPipeline parameters
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.mixed_precision = mixed_precision
        self.gradient_clipping = gradient_clipping
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.cross_validation_folds = cross_validation_folds
        self.save_checkpoints = save_checkpoints
        self.tensorboard_logging = tensorboard_logging
        self.wandb_logging = wandb_logging
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.profile_training = profile_training
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_workers = num_workers
        self.distributed_training = distributed_training
        self.compile_model = compile_model
        self.amp_optimization_level = amp_optimization_level
        self.gradient_scaling = gradient_scaling
        self.loss_scale = loss_scale
        self.curriculum_stages = curriculum_stages
        self.enable_progressive_learning = enable_progressive_learning
        self.progressive_stages = progressive_stages
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing
        self.denoising_weight = denoising_weight
        self.denoising_strategy = denoising_strategy
        self.denoising_objectives = denoising_objectives or ["reconstruction", "alignment"]
        self.denoising_modalities = denoising_modalities or []
        
        # EnsemblePrediction parameters
        self.calibrate_uncertainty = calibrate_uncertainty
        self.device = device
        
        # PerformanceMetrics parameters
        self.measure_efficiency = measure_efficiency
        self.n_efficiency_runs = n_efficiency_runs
        self.return_probabilities = return_probabilities
        
        # Internal state
        self.trained_learners_ = None
        self.learner_metadata_ = None
        self.ensemble_ = None
        self.modality_configs_ = None
        self.integration_metadata_ = None
        self.n_classes_ = None
        self.n_features_ = None
        self.is_multilabel_ = False
        self.task_type_ = None
        self.is_fitted_ = False
        
        # Set random seed
        np.random.seed(random_state)
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate initialization parameters."""
        if self.n_bags < 1:
            raise ValueError("n_bags must be >= 1")
        if self.dropout_strategy not in ['linear', 'exponential', 'random', 'adaptive']:
            raise ValueError("dropout_strategy must be one of: linear, exponential, random, adaptive")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.task_type not in ['auto', 'classification', 'regression']:
            raise ValueError("task_type must be one of: auto, classification, regression")
        if self.optimization_strategy not in ['balanced', 'accuracy', 'speed', 'memory']:
            raise ValueError("optimization_strategy must be one of: balanced, accuracy, speed, memory")
        if not 0 <= self.max_dropout_rate <= 0.9:
            raise ValueError("max_dropout_rate must be in [0, 0.9]")
        if self.min_modalities < 1:
            raise ValueError("min_modalities must be >= 1")
        if not 0.1 <= self.sample_ratio <= 1.0:
            raise ValueError("sample_ratio must be in [0.1, 1.0]")
        if not 0 <= self.diversity_target <= 1:
            raise ValueError("diversity_target must be in [0, 1]")
        
        # Validate DataIntegration parameters
        if self.handle_nan not in ['drop', 'fill_mean', 'fill_zero']:
            raise ValueError("handle_nan must be one of: drop, fill_mean, fill_zero")
        if self.handle_inf not in ['drop', 'fill_max', 'fill_zero']:
            raise ValueError("handle_inf must be one of: drop, fill_max, fill_zero")
        if self.outlier_std <= 0:
            raise ValueError("outlier_std must be positive")
        
        # Validate ModalityDropoutBagger parameters
        if not isinstance(self.enable_validation, bool):
            raise ValueError("enable_validation must be a boolean")
        
        # Validate ModalityAwareBaseLearnerSelector parameters
        if not isinstance(self.performance_threshold, (int, float)) or not 0 <= self.performance_threshold <= 1:
            raise ValueError("performance_threshold must be a float between 0 and 1")
        if self.validation_strategy not in ['cross_validation', 'holdout', 'bootstrap']:
            raise ValueError("validation_strategy must be one of: cross_validation, holdout, bootstrap")
        if not isinstance(self.hyperparameter_tuning, bool):
            raise ValueError("hyperparameter_tuning must be a boolean")
        
        # Validate TrainingPipeline parameters
        if self.optimizer_type not in ['adamw', 'adam', 'sgd', 'rmsprop']:
            raise ValueError("optimizer_type must be one of: adamw, adam, sgd, rmsprop")
        if self.scheduler_type not in ['cosine_restarts', 'onecycle', 'plateau', 'step', 'none']:
            raise ValueError("scheduler_type must be one of: cosine_restarts, onecycle, plateau, step, none")
        if not isinstance(self.mixed_precision, bool):
            raise ValueError("mixed_precision must be a boolean")
        if not isinstance(self.gradient_clipping, (int, float)) or self.gradient_clipping <= 0:
            raise ValueError("gradient_clipping must be a positive number")
        if not isinstance(self.early_stopping_patience, int) or self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be a non-negative integer")
        if not isinstance(self.validation_split, (int, float)) or not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be a float between 0 and 1")
        if not isinstance(self.cross_validation_folds, int) or self.cross_validation_folds < 2:
            raise ValueError("cross_validation_folds must be an integer >= 2")
        if not isinstance(self.save_checkpoints, bool):
            raise ValueError("save_checkpoints must be a boolean")
        if not isinstance(self.tensorboard_logging, bool):
            raise ValueError("tensorboard_logging must be a boolean")
        if not isinstance(self.wandb_logging, bool):
            raise ValueError("wandb_logging must be a boolean")
        if not isinstance(self.log_interval, int) or self.log_interval < 1:
            raise ValueError("log_interval must be a positive integer")
        if not isinstance(self.eval_interval, int) or self.eval_interval < 1:
            raise ValueError("eval_interval must be a positive integer")
        if not isinstance(self.profile_training, bool):
            raise ValueError("profile_training must be a boolean")
        if not isinstance(self.gradient_accumulation_steps, int) or self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if not isinstance(self.distributed_training, bool):
            raise ValueError("distributed_training must be a boolean")
        if not isinstance(self.compile_model, bool):
            raise ValueError("compile_model must be a boolean")
        if self.amp_optimization_level not in ['O0', 'O1', 'O2', 'O3']:
            raise ValueError("amp_optimization_level must be one of: O0, O1, O2, O3")
        if not isinstance(self.gradient_scaling, bool):
            raise ValueError("gradient_scaling must be a boolean")
        if self.loss_scale not in ['dynamic', 'static']:
            raise ValueError("loss_scale must be one of: dynamic, static")
        if not isinstance(self.enable_progressive_learning, bool):
            raise ValueError("enable_progressive_learning must be a boolean")
        if not isinstance(self.dropout_rate, (int, float)) or not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be a float between 0 and 1")
        if not isinstance(self.label_smoothing, (int, float)) or not 0 <= self.label_smoothing <= 1:
            raise ValueError("label_smoothing must be a float between 0 and 1")
        if not isinstance(self.denoising_weight, (int, float)) or self.denoising_weight < 0:
            raise ValueError("denoising_weight must be a non-negative number")
        if self.denoising_strategy not in ['adaptive', 'fixed', 'progressive']:
            raise ValueError("denoising_strategy must be one of: adaptive, fixed, progressive")
        
        # Validate EnsemblePrediction parameters
        if not isinstance(self.calibrate_uncertainty, bool):
            raise ValueError("calibrate_uncertainty must be a boolean")
        if self.device not in ['auto', 'cpu', 'cuda']:
            raise ValueError("device must be one of: auto, cpu, cuda")
        
        # Validate PerformanceMetrics parameters
        if not isinstance(self.measure_efficiency, bool):
            raise ValueError("measure_efficiency must be a boolean")
        if not isinstance(self.n_efficiency_runs, int) or self.n_efficiency_runs < 1:
            raise ValueError("n_efficiency_runs must be a positive integer")
        if not isinstance(self.return_probabilities, bool):
            raise ValueError("return_probabilities must be a boolean")
    
    def _determine_task_type(self, y):
        """Determine task type from labels."""
        if self.task_type != 'auto':
            return self.task_type
        
        unique_values = np.unique(y)
        n_unique = len(unique_values)
        
        # Check if it's multi-label
        if y.ndim == 2 and y.shape[1] > 1:
            return 'classification'
        
        # Check if it's regression (more than 20 unique values or float type)
        if n_unique > 20 or np.issubdtype(y.dtype, np.floating):
            return 'regression'
        
        # Otherwise, it's classification
        return 'classification'
    
    def _prepare_data_loader(self, X, y):
        """Prepare data loader from input data with full dataIntegration capabilities."""
        # Create data loader with validation
        loader = dataIntegration.GenericMultiModalDataLoader(validate_data=True, memory_efficient=False)
        
        # Determine split sizes (80% train, 20% test)
        n_samples = len(next(iter(X.values())))
        n_train = int(0.8 * n_samples)
        n_test = n_samples - n_train
        
        # Add modalities with proper data type detection
        for modality_name, modality_data in X.items():
            if modality_data.ndim == 1:
                modality_data = modality_data.reshape(-1, 1)
            
            train_data = modality_data[:n_train]
            test_data = modality_data[n_train:]
            
            # Determine data type based on modality name and features
            if 'text' in modality_name.lower() or 'text' in modality_name:
                data_type = 'text'
            elif 'image' in modality_name.lower() or 'img' in modality_name.lower():
                data_type = 'image'
            elif 'audio' in modality_name.lower():
                data_type = 'audio'
            else:
                data_type = 'tabular'
            
            # Add modality with proper configuration
            loader.add_modality_split(
                name=modality_name,
                train_data=train_data,
                test_data=test_data,
                data_type=data_type,
                is_required=True,  # All modalities are required by default
                feature_dim=train_data.shape[1]
            )
        
        # Add labels
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Note: Labels are kept as-is for compatibility with learners
        # The 1-indexed to 0-indexed conversion should be handled by individual learners if needed
        
        train_labels = y[:n_train]
        test_labels = y[n_train:]
        
        loader.add_labels_split(train_labels, test_labels)
        
        return loader
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the multimodal ensemble model.
        
        Parameters
        ----------
        X : dict
            Dictionary of modality data. Keys are modality names, values are numpy arrays.
        y : array-like
            Target values.
        sample_weight : array-like, optional
            Sample weights. Not currently used.
        
        Returns
        -------
        self : object
            Returns self.
        """
        if self.verbose:
            print("Starting multimodal ensemble training...")
        
        start_time = time.time()
        
        # Convert inputs to numpy arrays
        y = np.asarray(y)
        
        # Determine task type
        self.task_type_ = self._determine_task_type(y)
        if self.verbose:
            print(f"Detected task type: {self.task_type_}")
        
        # Handle multi-label classification
        if y.ndim == 2 and y.shape[1] > 1:
            self.is_multilabel_ = True
            self.n_classes_ = y.shape[1]
            if self.verbose:
                print(f"Multi-label classification with {self.n_classes_} labels")
        elif self.task_type_ == 'classification':
            self.is_multilabel_ = False
            self.n_classes_ = len(np.unique(y))
            if self.verbose:
                print(f"Single-label classification with {self.n_classes_} classes")
        else:
            self.is_multilabel_ = False
            self.n_classes_ = 1
            if self.verbose:
                print("Regression task")
        
        # Prepare data loader
        if self.verbose:
            print("Preparing data loader...")
        data_loader = self._prepare_data_loader(X, y)
        
        # Apply data preprocessing if enabled
        if self.normalize_data or self.remove_outliers:
            if self.verbose:
                print("Applying data preprocessing...")
            data_loader = self._apply_data_preprocessing(data_loader)
        
        # Clean data if needed
        if self.handle_nan != 'drop' or self.handle_inf != 'drop':
            if self.verbose:
                print("Cleaning data...")
            data_loader.clean_data(handle_nan=self.handle_nan, handle_inf=self.handle_inf)
        
        # Load and integrate data
        if self.verbose:
            print("Loading and integrating data...")
        self._load_and_integrate_data(data_loader)
        
        # Generate data quality report
        if self.verbose:
            print("Generating data quality report...")
        self._generate_data_quality_report(data_loader)
        
        # Generate ensemble bags
        if self.verbose:
            print(f"Generating {self.n_bags} ensemble bags...")
        self._generate_ensemble_bags()
        
        # Select and train base learners
        if self.verbose:
            print("Selecting and training base learners...")
        self._train_base_learners()
        
        # Create ensemble predictor
        if self.verbose:
            print("Creating ensemble predictor...")
        self._create_ensemble_predictor()
        
        self.is_fitted_ = True
        
        training_time = time.time() - start_time
        if self.verbose:
            print(f"Training completed in {training_time:.2f} seconds")
        
        return self
    
    def _load_and_integrate_data(self, data_loader):
        """Load and integrate data using the data loader."""
        # Check if this is a preprocessed loader (has add_modality) or split loader (has add_modality_split)
        if hasattr(data_loader, 'data') and any('_train' in k for k in data_loader.data.keys()):
            # Split loader - get train/test splits
            self.train_data, self.train_labels = data_loader.get_split('train')
            self.test_data, self.test_labels = data_loader.get_split('test')
        else:
            # Preprocessed loader - use all data as train, create a small test split
            all_data, all_labels = data_loader.load_and_preprocess()
            n_samples = len(all_labels)
            n_test = max(1, int(0.2 * n_samples))
            n_train = n_samples - n_test
            
            # Split data
            self.train_data = {k: v[:n_train] for k, v in all_data.items()}
            self.test_data = {k: v[n_train:] for k, v in all_data.items()}
            self.train_labels = all_labels[:n_train]
            self.test_labels = all_labels[n_train:]
        
        # Validate data consistency
        self._validate_data_consistency()
        
        # Create modality configs
        self.modality_configs_ = [
            SimpleNamespace(name=k, feature_dim=v.shape[1]) 
            for k, v in self.train_data.items()
        ]
        
        # Create integration metadata
        self.integration_metadata_ = {
            'dataset_size': len(self.train_labels),
            'n_modalities': len(self.modality_configs_),
            'modality_names': [mc.name for mc in self.modality_configs_],
            'feature_dimensions': {mc.name: mc.feature_dim for mc in self.modality_configs_}
        }
        
        # Set n_features
        self.n_features_ = list(self.train_data.values())[0].shape[1]
    
    def _apply_data_preprocessing(self, data_loader):
        """Apply data preprocessing using dataIntegration capabilities."""
        # Create a new loader for preprocessing
        new_loader = dataIntegration.GenericMultiModalDataLoader(
            validate_data=data_loader.validate_data,
            memory_efficient=data_loader.memory_efficient
        )
        
        # Get train data for preprocessing
        train_data, train_labels = data_loader.get_split('train')
        
        # First pass: apply non-destructive preprocessing (cleaning, normalization)
        cleaned_data = {}
        for modality_name, modality_data in train_data.items():
            data = modality_data.copy()
            
            # Handle missing values
            if self.handle_nan == 'fill_mean':
                if np.issubdtype(data.dtype, np.floating):
                    means = np.nanmean(data, axis=0)
                    inds = np.where(np.isnan(data))
                    data[inds] = np.take(means, inds[1])
            elif self.handle_nan == 'fill_zero':
                data = np.nan_to_num(data, nan=0.0)
            
            # Handle Inf values
            if self.handle_inf == 'fill_max':
                maxs = np.nanmax(data, axis=0)
                maxs = np.where(np.isinf(maxs), 1e6, maxs)
                inds = np.where(np.isinf(data))
                data[inds] = np.take(maxs, inds[1])
            elif self.handle_inf == 'fill_zero':
                data = np.where(np.isinf(data), 0.0, data)
            
            # Normalize
            if self.normalize_data and np.issubdtype(data.dtype, np.floating):
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                std[std == 0] = 1.0
                data = (data - mean) / std
            
            cleaned_data[modality_name] = data
        
        # Second pass: apply destructive preprocessing (outlier removal, drop) with consistent masking
        if self.remove_outliers or self.handle_nan == 'drop' or self.handle_inf == 'drop':
            # Find common valid indices across all modalities
            valid_mask = np.ones(len(train_labels), dtype=bool)
            
            for modality_name, data in cleaned_data.items():
                modality_mask = np.ones(len(data), dtype=bool)
                
                # Apply drop operations
                if self.handle_nan == 'drop' and np.issubdtype(data.dtype, np.floating):
                    modality_mask &= ~np.isnan(data).any(axis=1)
                
                if self.handle_inf == 'drop':
                    modality_mask &= ~np.isinf(data).any(axis=1)
                
                if self.remove_outliers and np.issubdtype(data.dtype, np.floating):
                    mean = np.mean(data, axis=0)
                    std = np.std(data, axis=0)
                    outlier_mask = np.all(np.abs(data - mean) < self.outlier_std * std, axis=1)
                    modality_mask &= outlier_mask
                
                # Update common mask
                valid_mask &= modality_mask
            
            # Apply common mask to all data
            for modality_name in cleaned_data:
                cleaned_data[modality_name] = cleaned_data[modality_name][valid_mask]
            
            # Apply mask to labels
            train_labels = train_labels[valid_mask]
        
        # Add all modalities to new loader
        for modality_name, data in cleaned_data.items():
            original_config = next(c for c in data_loader.modality_configs if c.name == modality_name)
            
            new_loader.add_modality(
                name=modality_name,
                data=data,
                data_type=original_config.data_type,
                is_required=original_config.is_required,
                feature_dim=data.shape[1]
            )
        
        # Add labels
        new_loader.add_labels(train_labels)
        
        return new_loader
    
    def _generate_data_quality_report(self, data_loader):
        """Generate and store data quality report."""
        try:
            self.data_quality_report_ = data_loader.get_data_quality_report()
            if self.verbose:
                print("Data Quality Report:")
                print(f"  - Total samples: {self.data_quality_report_['overall']['total_samples']}")
                print(f"  - Total modalities: {self.data_quality_report_['overall']['total_modalities']}")
                print(f"  - Required modalities: {self.data_quality_report_['overall']['required_modalities']}")
                for modality, stats in self.data_quality_report_['modalities'].items():
                    print(f"  - {modality}: shape={stats['shape']}, NaN={stats['nan_count']}, Inf={stats['inf_count']}")
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not generate data quality report: {e}")
            # Create a simple data quality report
            self.data_quality_report_ = {
                'overall': {
                    'total_samples': len(self.train_labels) if hasattr(self, 'train_labels') else 0,
                    'total_modalities': len(self.modality_configs_) if hasattr(self, 'modality_configs_') else 0,
                    'required_modalities': len(self.modality_configs_) if hasattr(self, 'modality_configs_') else 0
                },
                'modalities': {}
            }
    
    def get_data_quality_report(self):
        """Get the data quality report."""
        if not hasattr(self, 'data_quality_report_'):
            raise ValueError("Data quality report not available. Call fit() first.")
        return self.data_quality_report_
    
    def create_synthetic_dataset(self, modality_specs, n_samples=1000, n_classes=10, 
                                noise_level=0.1, missing_data_rate=0.0):
        """Create synthetic dataset using dataIntegration capabilities."""
        return dataIntegration.create_synthetic_dataset(
            modality_specs=modality_specs,
            n_samples=n_samples,
            n_classes=n_classes,
            noise_level=noise_level,
            missing_data_rate=missing_data_rate,
            random_state=self.random_state
        )
    
    def load_from_files(self, file_paths, modality_types=None, required_modalities=None):
        """Load data from files using dataIntegration capabilities."""
        return dataIntegration.QuickDatasetBuilder.from_files(
            file_paths=file_paths,
            modality_types=modality_types,
            required_modalities=required_modalities
        )
    
    def load_from_directory(self, data_dir, modality_patterns, modality_types=None, 
                           required_modalities=None):
        """Load data from directory using dataIntegration capabilities."""
        return dataIntegration.QuickDatasetBuilder.from_directory(
            data_dir=data_dir,
            modality_patterns=modality_patterns,
            modality_types=modality_types,
            required_modalities=required_modalities
        )
    
    def load_from_arrays(self, modality_data, modality_types=None, labels=None, 
                        required_modalities=None):
        """Load data from arrays using dataIntegration capabilities."""
        return dataIntegration.QuickDatasetBuilder.from_arrays(
            modality_data=modality_data,
            modality_types=modality_types,
            labels=labels,
            required_modalities=required_modalities
        )
    
    def _validate_data_consistency(self):
        """Validate that train and test data are consistent."""
        # Check modalities
        train_modalities = set(self.train_data.keys())
        test_modalities = set(self.test_data.keys())
        if train_modalities != test_modalities:
            raise ValueError(f"Train and test data must have same modalities. Train: {train_modalities}, Test: {test_modalities}")
        
        # Check feature dimensions
        for modality in train_modalities:
            train_features = self.train_data[modality].shape[1]
            test_features = self.test_data[modality].shape[1]
            if train_features != test_features:
                raise ValueError(f"Modality '{modality}' must have same number of features in train and test. Got {train_features} vs {test_features}")
        
        # Check label dimensions
        if self.train_labels.ndim != self.test_labels.ndim:
            raise ValueError(f"Train and test labels must have same number of dimensions. Got {self.train_labels.ndim} vs {self.test_labels.ndim}")
        
        # Check multi-label consistency
        if self.train_labels.ndim == 2:
            if self.train_labels.shape[1] != self.test_labels.shape[1]:
                raise ValueError(f"Train and test labels must have same number of label columns. Got {self.train_labels.shape[1]} vs {self.test_labels.shape[1]}")
        
        # Check sample counts
        test_data_samples = list(self.test_data.values())[0].shape[0]
        test_label_samples = self.test_labels.shape[0]
        if test_data_samples != test_label_samples:
            raise ValueError(f"Test data and test labels must have same number of samples. Got {test_data_samples} vs {test_label_samples}")
    
    def _generate_ensemble_bags(self):
        """Generate ensemble bags using modality dropout."""
        self.bagger = modalityDropoutBagger.ModalityDropoutBagger.from_data_integration(
            integrated_data=self.train_data,
            modality_configs=self.modality_configs_,
            integration_metadata=self.integration_metadata_,
            n_bags=self.n_bags,
            dropout_strategy=self.dropout_strategy,
            max_dropout_rate=self.max_dropout_rate,
            min_modalities=self.min_modalities,
            sample_ratio=self.sample_ratio,
            diversity_target=self.diversity_target,
            feature_sampling=self.feature_sampling,
            enable_validation=self.enable_validation,
            random_state=self.random_state
        )
        self.bags = self.bagger.generate_bags()
    
    def _train_base_learners(self):
        """Select and train base learners."""
        # Create base learner selector
        modality_feature_dims = {mc.name: mc.feature_dim for mc in self.modality_configs_}
        
        self.selector = modalityAwareBaseLearnerSelector.ModalityAwareBaseLearnerSelector(
            bags=self.bags,
            modality_feature_dims=modality_feature_dims,
            integration_metadata=self.integration_metadata_,
            task_type=self.task_type_,
            optimization_strategy=self.optimization_strategy,
            learner_preferences=self.learner_preferences,
            performance_threshold=self.performance_threshold,
            resource_limit=self.resource_limit,
            validation_strategy=self.validation_strategy,
            hyperparameter_tuning=self.hyperparameter_tuning,
            n_classes=self.n_classes_,
            random_state=self.random_state
        )
        
        # Generate learners
        selected_learners = self.selector.generate_learners(instantiate=True)
        
        # Prepare training data
        learner_metadata = []
        bag_data = {}
        bag_labels = {}
        
        # Add labels to train_data for get_bag_data
        train_data_with_labels = self.train_data.copy()
        train_data_with_labels['labels'] = self.train_labels
        
        for i, (learner_id, learner) in enumerate(selected_learners.items()):
            # Use proper get_bag_data method
            bag_data_single, modality_mask, bag_metadata = self.bagger.get_bag_data(
                bag_id=i, 
                multimodal_data=train_data_with_labels, 
                return_metadata=True
            )
            
            # Extract labels from bag_data
            bag_labels[learner_id] = bag_data_single.pop('labels', self.train_labels[self.bags[i].data_indices])
            bag_data[learner_id] = bag_data_single
            
            # Get active modalities
            active_modalities = [k for k, v in modality_mask.items() if v]
            pattern = '+'.join(sorted(active_modalities))
            
            learner_metadata.append({
                'learner_id': learner_id,
                'metrics': {},
                'modalities': active_modalities,
                'pattern': pattern,
                'bag_metadata': bag_metadata
            })
        
        # Create training pipeline with custom parameters
        # Use a simpler approach to avoid parameter conflicts
        self.pipeline = trainingPipeline.create_training_pipeline(
            task_type=self.task_type_,
            epochs=self.epochs,
            batch_size=self.batch_size,
            enable_denoising=self.enable_denoising,
            optimizer_type=self.optimizer_type,
            scheduler_type=self.scheduler_type,
            mixed_precision=self.mixed_precision,
            gradient_clipping=self.gradient_clipping,
            validation_split=self.validation_split,
            cross_validation_folds=self.cross_validation_folds,
            save_checkpoints=self.save_checkpoints,
            tensorboard_logging=self.tensorboard_logging,
            wandb_logging=self.wandb_logging,
            log_interval=self.log_interval,
            eval_interval=self.eval_interval,
            profile_training=self.profile_training,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_workers=self.num_workers,
            distributed_training=self.distributed_training,
            compile_model=self.compile_model,
            amp_optimization_level=self.amp_optimization_level,
            gradient_scaling=self.gradient_scaling,
            loss_scale=self.loss_scale,
            curriculum_stages=self.curriculum_stages,
            enable_progressive_learning=self.enable_progressive_learning,
            progressive_stages=self.progressive_stages,
            dropout_rate=self.dropout_rate,
            denoising_weight=self.denoising_weight,
            denoising_strategy=self.denoising_strategy,
            denoising_objectives=self.denoising_objectives,
            denoising_modalities=self.denoising_modalities
        )
        
        # Train ensemble
        self.trained_learners_, self.training_metrics_ = self.pipeline.train_ensemble(
            selected_learners, 
            [None] * len(selected_learners), 
            bag_data, 
            bag_labels=bag_labels
        )
        
        # Update learner metadata with training metrics
        for i, (lid, metrics) in enumerate(self.training_metrics_.items()):
            if metrics:
                learner_metadata[i]['metrics'] = metrics[-1].__dict__
        
        # Sort for determinism
        learner_id_order = sorted(self.trained_learners_.keys())
        self.trained_learners_ = [self.trained_learners_[lid] for lid in learner_id_order]
        self.learner_metadata_ = [
            next(m for m in learner_metadata if m['learner_id'] == lid) 
            for lid in learner_id_order
        ]
    
    def _create_ensemble_predictor(self):
        """Create ensemble predictor."""
        self.ensemble_ = ensemblePrediction.create_ensemble_predictor(
            task_type=self.task_type_,
            aggregation_strategy=self.aggregation_strategy,
            uncertainty_method=self.uncertainty_method,
            calibrate_uncertainty=self.calibrate_uncertainty,
            device=self.device
        )
        
        # Add trained learners to ensemble with bag configurations
        for i, (learner, meta) in enumerate(zip(self.trained_learners_, self.learner_metadata_)):
            # Get bag configuration for this learner
            bag_config = self.bags[i] if i < len(self.bags) else None
            
            self.ensemble_.add_trained_learner(
                learner, 
                meta.get('metrics', {}), 
                meta.get('modalities', []), 
                meta.get('pattern', ''),
                bag_config=bag_config
            )
    
    def predict(self, X):
        """
        Predict using the fitted model.
        
        Parameters
        ----------
        X : dict
            Dictionary of modality data for prediction.
        
        Returns
        -------
        predictions : array-like
            Predicted values.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to numpy arrays and clean data
        X = {k: np.asarray(v) for k, v in X.items()}
        
        # Apply same data cleaning as during training
        X = self._clean_prediction_data(X)
        
        # Handle multi-label classification
        if self.is_multilabel_:
            return self._predict_multilabel(X)
        else:
            # Use ensemble predictor
            pred_result = self.ensemble_.predict(X)
            return pred_result.predictions
    
    def _predict_multilabel(self, X):
        """Predict for multi-label classification."""
        # Get predictions from each trained learner
        all_predictions = []
        
        for learner in self.trained_learners_:
            if hasattr(learner, 'predict'):
                try:
                    pred = learner.predict(X)
                    all_predictions.append(pred)
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Learner prediction failed: {e}")
                    continue
        
        if not all_predictions:
            # Fallback: return zeros
            n_samples = list(X.values())[0].shape[0]
            return np.zeros((n_samples, self.n_classes_), dtype=int)
        
        # Average predictions
        avg_predictions = np.mean(all_predictions, axis=0)
        
        # Add small random variation for diversity
        np.random.seed(self.random_state)
        variation = np.random.normal(0, 0.01, avg_predictions.shape)
        avg_predictions += variation
        
        # Threshold to get binary predictions
        threshold = 0.3
        final_predictions = (avg_predictions > threshold).astype(int)
        
        return final_predictions
    
    def _clean_prediction_data(self, X):
        """Clean prediction data using the same methods as training data."""
        cleaned_X = {}
        
        for modality_name, modality_data in X.items():
            data = modality_data.copy()
            
            # Handle missing values
            if self.handle_nan == 'fill_mean':
                if np.issubdtype(data.dtype, np.floating):
                    means = np.nanmean(data, axis=0)
                    inds = np.where(np.isnan(data))
                    data[inds] = np.take(means, inds[1])
            elif self.handle_nan == 'fill_zero':
                data = np.nan_to_num(data, nan=0.0)
            elif self.handle_nan == 'drop':
                if np.issubdtype(data.dtype, np.floating):
                    mask = ~np.isnan(data).any(axis=1)
                    data = data[mask]
            
            # Handle Inf values
            if self.handle_inf == 'fill_max':
                maxs = np.nanmax(data, axis=0)
                maxs = np.where(np.isinf(maxs), 1e6, maxs)
                inds = np.where(np.isinf(data))
                data[inds] = np.take(maxs, inds[1])
            elif self.handle_inf == 'fill_zero':
                data = np.where(np.isinf(data), 0.0, data)
            elif self.handle_inf == 'drop':
                mask = ~np.isinf(data).any(axis=1)
                data = data[mask]
            
            # Normalize if enabled
            if self.normalize_data and np.issubdtype(data.dtype, np.floating):
                # Use the same normalization as training (would need to store means/stds)
                # For now, just ensure no NaN/Inf values
                pass
            
            cleaned_X[modality_name] = data
        
        return cleaned_X
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : dict
            Dictionary of modality data for prediction.
        
        Returns
        -------
        probabilities : array-like
            Class probabilities.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.task_type_ != 'classification':
            raise ValueError("predict_proba() is only available for classification tasks")
        
        # Convert to numpy arrays and clean data
        X = {k: np.asarray(v) for k, v in X.items()}
        
        # Apply same data cleaning as during training
        X = self._clean_prediction_data(X)
        
        # Handle multi-label classification
        if self.is_multilabel_:
            # For multi-label, return the raw predictions as probabilities
            predictions = self._predict_multilabel(X)
            return predictions.astype(float)
        else:
            # Use ensemble predictor
            return self.ensemble_.predict_proba(X)
    
    def predict_classes(self, X):
        """
        Predict class labels (classification only).
        
        Parameters
        ----------
        X : dict
            Dictionary of modality data for prediction.
        
        Returns
        -------
        predictions : array-like
            Predicted class labels.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.task_type_ != 'classification':
            raise ValueError("predict_classes() is only available for classification tasks")
        
        return self.predict(X)
    
    def predict_values(self, X):
        """
        Predict values (regression only).
        
        Parameters
        ----------
        X : dict
            Dictionary of modality data for prediction.
        
        Returns
        -------
        predictions : array-like
            Predicted values.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.task_type_ != 'regression':
            raise ValueError("predict_values() is only available for regression tasks")
        
        return self.predict(X)
    
    def score(self, X, y, sample_weight=None):
        """
        Return the score on the given test data and labels.
        
        Parameters
        ----------
        X : dict
            Dictionary of modality data.
        y : array-like
            True labels.
        sample_weight : array-like, optional
            Sample weights. Not currently used.
        
        Returns
        -------
        score : float
            Mean accuracy for classification, R² for regression.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before scoring")
        
        predictions = self.predict(X)
        y = np.asarray(y)
        
        if self.task_type_ == 'classification':
            return np.mean(predictions == y)
        else:
            # R² score for regression
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    def predict_with_uncertainty(self, X, return_uncertainty=True):
        """
        Predict with uncertainty estimation.
        
        Parameters
        ----------
        X : dict
            Dictionary of modality data for prediction.
        return_uncertainty : bool, optional
            Whether to return uncertainty estimates.
        
        Returns
        -------
        result : PredictionResult
            Prediction result with predictions, confidence, and uncertainty.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to numpy arrays and clean data
        X = {k: np.asarray(v) for k, v in X.items()}
        X = self._clean_prediction_data(X)
        
        return self.ensemble_.predict(X, return_uncertainty=return_uncertainty)
    
    def get_ensemble_info(self):
        """
        Get information about the ensemble predictor.
        
        Returns
        -------
        info : dict
            Ensemble information including number of learners and strategies.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting ensemble info")
        
        return {
            'n_learners': len(self.trained_learners_),
            'aggregation_strategy': self.aggregation_strategy,
            'uncertainty_method': self.uncertainty_method,
            'task_type': self.task_type_,
            'calibrate_uncertainty': self.calibrate_uncertainty,
            'device': self.device
        }
    
    def evaluate_ensemble(self, X, y, detailed=True):
        """
        Evaluate the ensemble performance.
        
        Parameters
        ----------
        X : dict
            Dictionary of modality data for evaluation.
        y : array-like
            True labels.
        detailed : bool, optional
            Whether to return detailed metrics.
        
        Returns
        -------
        metrics : dict
            Evaluation metrics.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before evaluation")
        
        # Convert to numpy arrays and clean data
        X = {k: np.asarray(v) for k, v in X.items()}
        X = self._clean_prediction_data(X)
        y = np.asarray(y)
        
        return self.ensemble_.evaluate(X, y, detailed=detailed)
    
    def evaluate_comprehensive(self, X, y, model_name="MultiModalEnsemble"):
        """
        Comprehensive model evaluation with all performance metrics.
        
        Parameters
        ----------
        X : dict
            Dictionary of modality data for evaluation.
        y : array-like
            True labels.
        model_name : str, optional
            Name of the model for reporting.
        
        Returns
        -------
        report : ModelPerformanceReport
            Comprehensive performance report with all metrics.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before evaluation")
        
        # Convert to numpy arrays and clean data
        X = {k: np.asarray(v) for k, v in X.items()}
        X = self._clean_prediction_data(X)
        y = np.asarray(y)
        
        # Create evaluator
        evaluator = performanceMetrics.PerformanceEvaluator(task_type=self.task_type_)
        
        # Evaluate model with comprehensive metrics
        report = evaluator.evaluate_model(
            lambda X_dict: self.predict(X_dict), 
            X, 
            y, 
            model_name=model_name,
            return_probabilities=self.return_probabilities,
            measure_efficiency=self.measure_efficiency,
            n_efficiency_runs=self.n_efficiency_runs
        )
        
        return report
    
    def get_performance_metrics(self, X, y):
        """
        Get detailed performance metrics.
        
        Parameters
        ----------
        X : dict
            Dictionary of modality data for evaluation.
        y : array-like
            True labels.
        
        Returns
        -------
        metrics : dict
            Dictionary of performance metrics.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting performance metrics")
        
        # Convert to numpy arrays and clean data
        X = {k: np.asarray(v) for k, v in X.items()}
        X = self._clean_prediction_data(X)
        y = np.asarray(y)
        
        # Get predictions
        predictions = self.predict(X)
        
        # Calculate metrics based on task type
        if self.task_type_ == "classification":
            # Try to get probabilities
            try:
                probabilities = self.predict_proba(X)
                metrics = performanceMetrics.ClassificationMetricsCalculator.calculate(y, predictions, probabilities)
            except:
                metrics = performanceMetrics.ClassificationMetricsCalculator.calculate(y, predictions)
        else:
            metrics = performanceMetrics.RegressionMetricsCalculator.calculate(y, predictions)
        
        return metrics
    
    def measure_efficiency_metrics(self, X, n_runs=None):
        """
        Measure model efficiency metrics.
        
        Parameters
        ----------
        X : dict
            Dictionary of modality data for testing.
        n_runs : int, optional
            Number of runs for efficiency measurement.
        
        Returns
        -------
        efficiency : dict
            Dictionary of efficiency metrics.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before measuring efficiency")
        
        # Convert to numpy arrays and clean data
        X = {k: np.asarray(v) for k, v in X.items()}
        X = self._clean_prediction_data(X)
        
        n_runs = n_runs or self.n_efficiency_runs
        
        # Measure efficiency
        inference_time, inference_std = performanceMetrics.EfficiencyMetricsCalculator.measure_inference_time(
            lambda X_dict: self.predict(X_dict), X, n_runs
        )
        throughput = performanceMetrics.EfficiencyMetricsCalculator.measure_throughput(
            lambda X_dict: self.predict(X_dict), X
        )
        memory_usage = performanceMetrics.EfficiencyMetricsCalculator.measure_memory_usage()
        
        return {
            'inference_time_ms': inference_time,
            'inference_time_std_ms': inference_std,
            'throughput_samples_per_sec': throughput,
            'memory_usage_mb': memory_usage
        }
    
    def get_multimodal_metrics(self, X, y):
        """
        Get multimodal-specific metrics.
        
        Parameters
        ----------
        X : dict
            Dictionary of modality data for evaluation.
        y : array-like
            True labels.
        
        Returns
        -------
        metrics : dict
            Dictionary of multimodal metrics.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting multimodal metrics")
        
        # Convert to numpy arrays and clean data
        X = {k: np.asarray(v) for k, v in X.items()}
        X = self._clean_prediction_data(X)
        y = np.asarray(y)
        
        # Get predictions for each modality
        predictions_by_modality = {}
        for modality in X.keys():
            try:
                # Create single modality data
                single_modality_data = {modality: X[modality]}
                predictions_by_modality[modality] = self.predict(single_modality_data)
            except:
                continue
        
        # Calculate multimodal metrics
        cross_modal_consistency = performanceMetrics.MultimodalMetricsCalculator.cross_modal_consistency(
            predictions_by_modality
        )
        
        # Get modality importance from ensemble
        try:
            result = self.predict_with_uncertainty(X)
            modality_importance = result.modality_importance or {}
        except:
            modality_importance = {modality: 1.0/len(X) for modality in X.keys()}
        
        # Calculate missing modality robustness
        missing_modality_robustness = {}
        for modality in X.keys():
            try:
                # Remove one modality
                reduced_data = {k: v for k, v in X.items() if k != modality}
                if reduced_data:
                    reduced_predictions = self.predict(reduced_data)
                    # Calculate performance with missing modality
                    if self.task_type_ == "classification":
                        accuracy = np.mean(reduced_predictions == y)
                    else:
                        accuracy = 1.0 - np.mean(np.abs(reduced_predictions - y))
                    missing_modality_robustness[modality] = accuracy
            except:
                missing_modality_robustness[modality] = 0.0
        
        return {
            'cross_modal_consistency': cross_modal_consistency,
            'modality_importance': modality_importance,
            'missing_modality_robustness': missing_modality_robustness
        }
    
    def create_model_comparator(self):
        """
        Create a model comparator for comparing multiple models.
        
        Returns
        -------
        comparator : ModelComparator
            Model comparator instance.
        """
        return performanceMetrics.ModelComparator(task_type=self.task_type_)
    
    def get_stage2_interpretability_data(self):
        """
        Get comprehensive interpretability data for Stage 2 (Ensemble Generation).
        
        Returns
        -------
        interpretability_data : dict
            Comprehensive interpretability data including:
            - ensemble_stats: Basic ensemble statistics
            - feature_sampling_analysis: Feature sampling patterns
            - adaptive_behavior: Adaptive strategy behavior
            - modality_patterns: Modality combination patterns
            - interpretability_metadata: Configuration metadata
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting interpretability data")
        
        # Get interpretability data from the bagger
        if hasattr(self, 'bagger') and self.bagger is not None:
            analyzer = performanceMetrics.Stage2InterpretabilityAnalyzer()
            return analyzer.get_stage2_interpretability_data(self.bagger)
        else:
            return {
                'ensemble_stats': self.get_ensemble_stats(return_detailed=True),
                'feature_sampling_analysis': {},
                'adaptive_behavior': {},
                'modality_patterns': {},
                'interpretability_metadata': {}
            }
    
    def analyze_stage2_interpretability(self):
        """
        Perform comprehensive Stage 2 interpretability analysis.
        
        Returns
        -------
        Stage2InterpretabilityReport
            Comprehensive interpretability analysis report including:
            - ensemble_diversity: Diversity analysis of ensemble bags
            - modality_coverage: Coverage analysis across modalities
            - dropout_distribution: Analysis of dropout rate distribution
            - bag_configurations: Analysis of bag configuration patterns
            - feature_sampling: Analysis of feature sampling patterns
            - adaptive_behavior: Analysis of adaptive behavior patterns
            - bootstrap_sampling: Analysis of bootstrap sampling patterns
            - interpretability_metadata: Configuration metadata
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before interpretability analysis")
        
        # Get interpretability data from the bagger
        if hasattr(self, 'bagger') and self.bagger is not None:
            analyzer = performanceMetrics.Stage2InterpretabilityAnalyzer()
            return analyzer.analyze_stage2_interpretability(self.bagger)
        else:
            return performanceMetrics.Stage2InterpretabilityReport()
    
    def run_stage2_robustness_tests(self, 
                                   n_runs_per_test: int = 3,
                                   performance_threshold: float = 0.7,
                                   verbose: bool = True) -> performanceMetrics.Stage2RobustnessReport:
        """
        Run comprehensive Stage 2 robustness tests.
        
        Parameters
        ----------
        n_runs_per_test : int, default=3
            Number of runs per test for statistical significance
        performance_threshold : float, default=0.7
            Minimum performance threshold for passing tests
        verbose : bool, default=True
            Whether to print progress information
        
        Returns
        -------
        Stage2RobustnessReport
            Comprehensive robustness test report including:
            - modality_dropout_robustness: Tests for missing modalities, extreme dropout rates, different strategies
            - ensemble_size_robustness: Tests for different ensemble sizes (3, 5, 8, 15, 20, 30)
            - feature_sampling_robustness: Tests for different feature sampling ratios and scenarios
            - diversity_target_robustness: Tests for different diversity targets (0.3, 0.5, 0.7, 0.9)
            - bootstrap_sampling_robustness: Tests for different sample ratios and dataset sizes
            - modality_configuration_robustness: Tests for different modality counts and balance
            - random_seed_robustness: Tests for stability across different random seeds
            - data_quality_robustness: Tests for noise, outliers, and missing data scenarios
            - overall_robustness_score: Weighted overall robustness score
            - robustness_summary: Detailed statistics and pass rates
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running robustness tests")
        
        # Create model factory function
        def model_factory(**kwargs):
            # Get current model parameters
            params = self.get_params()
            # Update with test-specific parameters
            params.update(kwargs)
            # Create new model instance
            return MultiModalEnsembleModel(**params)
        
        # Create test data factory function
        def data_factory(**kwargs):
            # Use the same data as the fitted model
            return self.train_data, self.train_labels
        
        # Create robustness tester
        tester = performanceMetrics.Stage2RobustnessTester(
            base_model_factory=model_factory,
            test_data_factory=data_factory
        )
        
        # Run comprehensive robustness tests
        return tester.run_comprehensive_robustness_tests(
            n_runs_per_test=n_runs_per_test,
            performance_threshold=performance_threshold,
            verbose=verbose
        )
    
    def run_single_robustness_test(self, 
                                  test_category: str,
                                  test_config: Dict[str, Any],
                                  n_runs: int = 3,
                                  performance_threshold: float = 0.7) -> performanceMetrics.RobustnessTestResult:
        """
        Run a single robustness test for Stage 2.
        
        Parameters
        ----------
        test_category : str
            Category of test to run ('modality_dropout', 'ensemble_size', 'feature_sampling', etc.)
        test_config : dict
            Configuration parameters for the test
        n_runs : int, default=3
            Number of runs for statistical significance
        performance_threshold : float, default=0.7
            Minimum performance threshold for passing
        
        Returns
        -------
        RobustnessTestResult
            Result of the single robustness test
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running robustness tests")
        
        # Create model factory function
        def model_factory(**kwargs):
            params = self.get_params()
            params.update(kwargs)
            return MultiModalEnsembleModel(**params)
        
        # Create test data factory function
        def data_factory(**kwargs):
            return self.train_data, self.train_labels
        
        # Create robustness tester
        tester = performanceMetrics.Stage2RobustnessTester(
            base_model_factory=model_factory,
            test_data_factory=data_factory
        )
        
        # Run single test
        return tester._run_single_test(
            test_name=f"custom_{test_category}",
            test_category=test_category,
            test_config=test_config,
            n_runs=n_runs,
            threshold=performance_threshold,
            verbose=False
        )
    
    def get_stage3_interpretability_data(self):
        """
        Get comprehensive interpretability data for Stage 3 (Base Learner Selection).
        
        Returns
        -------
        interpretability_data : dict
            Comprehensive interpretability data including:
            - learner_selections: Learner selection patterns
            - modality_patterns: Modality combination patterns
            - optimization_impact: Optimization strategy effects
            - compatibility_matrix: Modality-learner compatibility
            - threshold_analysis: Performance threshold effects
            - diversity_metrics: Learner diversity metrics
            - adaptive_behavior: Adaptive selection behavior
            - validation_impact: Validation process effects
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting interpretability data")
        
        # Get interpretability data from the selector
        if hasattr(self, 'selector') and self.selector is not None:
            analyzer = performanceMetrics.Stage3InterpretabilityAnalyzer()
            return analyzer.get_stage3_interpretability_data(self.selector)
        else:
            return {
                'learner_selections': {},
                'modality_patterns': {},
                'learner_frequency': {},
                'optimization_impact': {},
                'compatibility_matrix': {},
                'compatibility_scores': {},
                'threshold_analysis': {},
                'diversity_metrics': {},
                'adaptive_behavior': {},
                'validation_impact': {},
                'interpretability_metadata': {}
            }
    
    def analyze_stage3_interpretability(self):
        """
        Perform comprehensive Stage 3 interpretability analysis.
        
        Returns
        -------
        Stage3InterpretabilityReport
            Comprehensive interpretability analysis report including:
            - learner_selection_patterns: Analysis of which learners are selected for different modality patterns
            - optimization_strategy_impact: Analysis of how optimization strategy affects learner selection
            - cross_modal_compatibility: Analysis of modality-learner compatibility patterns
            - performance_threshold_impact: Analysis of how performance thresholds affect learner selection
            - learner_diversity: Analysis of diversity patterns in selected learners
            - adaptive_selection_behavior: Analysis of how adaptive selection strategy behaves
            - validation_impact: Analysis of how validation affects learner selection
            - interpretability_metadata: Configuration metadata
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before interpretability analysis")
        
        # Get interpretability data from the selector
        if hasattr(self, 'selector') and self.selector is not None:
            analyzer = performanceMetrics.Stage3InterpretabilityAnalyzer()
            return analyzer.analyze_stage3_interpretability(self.selector)
        else:
            return performanceMetrics.Stage3InterpretabilityReport()
    
    def run_stage3_robustness_tests(self, 
                                   n_runs_per_test: int = 3,
                                   performance_threshold: float = 0.7,
                                   verbose: bool = True) -> performanceMetrics.Stage3RobustnessReport:
        """
        Run comprehensive Stage 3 robustness tests.
        
        Parameters
        ----------
        n_runs_per_test : int, default=3
            Number of runs per test for statistical significance
        performance_threshold : float, default=0.7
            Minimum performance threshold for passing tests
        verbose : bool, default=True
            Whether to print progress information
            
        Returns
        -------
        Stage3RobustnessReport
            Comprehensive robustness test report including:
            - optimization_strategy_robustness: Tests for different optimization strategies
            - performance_threshold_robustness: Tests for different threshold values
            - learner_type_robustness: Tests for different learner type configurations
            - modality_pattern_robustness: Tests for different modality patterns
            - validation_strategy_robustness: Tests for different validation strategies
            - task_type_robustness: Tests for different task types
            - random_seed_robustness: Tests for different random seeds
            - data_quality_robustness: Tests for different data quality scenarios
            - ensemble_size_robustness: Tests for different ensemble sizes
            - hyperparameter_robustness: Tests for different hyperparameter configurations
            - overall_robustness_score: Overall robustness score across all tests
            - robustness_summary: Detailed summary of test results
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running robustness tests")
        
        # Create model factory function
        def model_factory(**kwargs):
            # Get current model parameters
            params = self.get_params()
            # Update with test-specific parameters
            params.update(kwargs)
            # Create new model instance
            return MultiModalEnsembleModel(**params)
        
        # Create test data factory function
        def data_factory(**kwargs):
            # Use the same data as the fitted model
            return self.train_data, self.train_labels
        
        # Create robustness tester
        tester = performanceMetrics.Stage3RobustnessTester(
            base_model_factory=model_factory,
            test_data_factory=data_factory
        )
        
        # Run comprehensive robustness tests
        return tester.run_comprehensive_robustness_tests(
            n_runs_per_test=n_runs_per_test,
            performance_threshold=performance_threshold,
            verbose=verbose
        )
    
    def run_single_stage3_robustness_test(self, 
                                         test_category: str,
                                         test_config: Dict[str, Any],
                                         n_runs: int = 3,
                                         performance_threshold: float = 0.7) -> performanceMetrics.RobustnessTestResult:
        """
        Run a single robustness test for Stage 3.
        
        Parameters
        ----------
        test_category : str
            Category of test to run (e.g., 'optimization_strategy', 'performance_threshold')
        test_config : dict
            Configuration for the test
        n_runs : int, default=3
            Number of runs for the test
        performance_threshold : float, default=0.7
            Minimum performance threshold for passing the test
            
        Returns
        -------
        RobustnessTestResult
            Result of the single robustness test
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running robustness tests")
        
        # Create model factory function
        def model_factory(**kwargs):
            params = self.get_params()
            params.update(kwargs)
            return MultiModalEnsembleModel(**params)
        
        # Create test data factory function
        def data_factory(**kwargs):
            return self.train_data, self.train_labels
        
        # Create robustness tester
        tester = performanceMetrics.Stage3RobustnessTester(
            base_model_factory=model_factory,
            test_data_factory=data_factory
        )
        
        # Run single test
        return tester._run_single_test(
            test_name=f"custom_{test_category}",
            test_category=test_category,
            test_config=test_config,
            n_runs=n_runs,
            threshold=performance_threshold,
            verbose=False
        )
    
    def get_stage4_interpretability_data(self):
        """
        Get comprehensive interpretability data for Stage 4 (Training Pipeline).
        
        Returns
        -------
        dict
            Comprehensive interpretability data including:
            - denoising_metrics: Cross-modal denoising effectiveness
            - progressive_learning_metrics: Progressive learning behavior
            - multi_objective_metrics: Multi-objective optimization analysis
            - convergence_metrics: Training convergence patterns
            - cross_modal_metrics: Cross-modal learning dynamics
            - resource_metrics: Resource utilization analysis
            - robustness_metrics: Training robustness analysis
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting interpretability data")
        
        if hasattr(self, 'get_training_metrics'):
            analyzer = performanceMetrics.Stage4InterpretabilityAnalyzer()
            return analyzer.get_stage4_interpretability_data(self)
        else:
            return {
                'denoising_metrics': {},
                'progressive_learning_metrics': {},
                'multi_objective_metrics': {},
                'convergence_metrics': {},
                'cross_modal_metrics': {},
                'resource_metrics': {},
                'robustness_metrics': {},
                'interpretability_metadata': {}
            }
    
    def analyze_stage4_interpretability(self):
        """
        Perform comprehensive Stage 4 interpretability analysis.
        
        Returns
        -------
        Stage4InterpretabilityReport
            Comprehensive interpretability analysis report
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before interpretability analysis")
        
        if hasattr(self, 'get_training_metrics'):
            analyzer = performanceMetrics.Stage4InterpretabilityAnalyzer()
            return analyzer.analyze_stage4_interpretability(self)
        else:
            return performanceMetrics.Stage4InterpretabilityReport()
    
    def run_stage4_robustness_tests(self, 
                                   n_runs_per_test: int = 3,
                                   performance_threshold: float = 0.7,
                                   verbose: bool = True) -> performanceMetrics.Stage4RobustnessReport:
        """
        Run comprehensive Stage 4 robustness tests.
        
        Parameters
        ----------
        n_runs_per_test : int, default=3
            Number of runs for each test
        performance_threshold : float, default=0.7
            Minimum performance threshold for passing tests
        verbose : bool, default=True
            Whether to print progress information
            
        Returns
        -------
        Stage4RobustnessReport
            Comprehensive robustness test report
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running robustness tests")
        
        # Create model factory function
        def model_factory(**kwargs):
            params = self.get_params()
            params.update(kwargs)
            return MultiModalEnsembleModel(**params)
        
        # Create test data factory function
        def data_factory(**kwargs):
            return self.train_data, self.train_labels
        
        # Create robustness tester
        tester = performanceMetrics.Stage4RobustnessTester(
            base_model_factory=model_factory,
            test_data_factory=data_factory
        )
        
        # Run comprehensive robustness tests
        return tester.run_comprehensive_robustness_tests(
            n_runs_per_test=n_runs_per_test,
            performance_threshold=performance_threshold,
            verbose=verbose
        )
    
    def run_single_stage4_robustness_test(self, 
                                         test_category: str,
                                         test_config: Dict[str, Any],
                                         n_runs: int = 3,
                                         performance_threshold: float = 0.7) -> performanceMetrics.RobustnessTestResult:
        """
        Run a single robustness test for Stage 4.
        
        Parameters
        ----------
        test_category : str
            Category of test to run (e.g., 'cross_modal_denoising', 'progressive_learning')
        test_config : dict
            Configuration for the test
        n_runs : int, default=3
            Number of runs for the test
        performance_threshold : float, default=0.7
            Minimum performance threshold for passing the test
            
        Returns
        -------
        RobustnessTestResult
            Result of the single robustness test
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running robustness tests")
        
        # Create model factory function
        def model_factory(**kwargs):
            params = self.get_params()
            params.update(kwargs)
            return MultiModalEnsembleModel(**params)
        
        # Create test data factory function
        def data_factory(**kwargs):
            return self.train_data, self.train_labels
        
        # Create robustness tester
        tester = performanceMetrics.Stage4RobustnessTester(
            base_model_factory=model_factory,
            test_data_factory=data_factory
        )
        
        # Run single test
        return tester._run_single_test(
            test_name=f"custom_{test_category}",
            test_category=test_category,
            test_config=test_config,
            n_runs=n_runs,
            threshold=performance_threshold,
            verbose=False
        )
    
    def get_stage5_interpretability_data(self):
        """
        Get comprehensive interpretability data for Stage 5 (Ensemble Prediction).
        
        Returns
        -------
        dict
            Comprehensive interpretability data including:
            - transformer_fusion_attention: Attention patterns in transformer meta-learner
            - dynamic_weighting_behavior: Dynamic weighting adaptation patterns
            - uncertainty_weighted_aggregation: Uncertainty-based weighting analysis
            - attention_based_uncertainty: Attention-based uncertainty estimation
            - ensemble_prediction_consistency: Prediction consistency across strategies
            - modality_importance: Modality contribution analysis
            - bag_reconstruction_effectiveness: Bag reconstruction quality analysis
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting interpretability data")
        
        if hasattr(self, 'ensemble_') and self.ensemble_ is not None:
            analyzer = performanceMetrics.Stage5InterpretabilityAnalyzer()
            return analyzer.get_stage5_interpretability_data(self)
        else:
            return {
                'transformer_fusion_attention': {},
                'dynamic_weighting_behavior': {},
                'uncertainty_weighted_aggregation': {},
                'attention_based_uncertainty': {},
                'ensemble_prediction_consistency': {},
                'modality_importance': {},
                'bag_reconstruction_effectiveness': {},
                'interpretability_metadata': {}
            }
    
    def analyze_stage5_interpretability(self):
        """
        Perform comprehensive Stage 5 interpretability analysis.
        
        Returns
        -------
        Stage5InterpretabilityReport
            Comprehensive interpretability analysis report
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before interpretability analysis")
        
        if hasattr(self, 'ensemble_') and self.ensemble_ is not None:
            analyzer = performanceMetrics.Stage5InterpretabilityAnalyzer()
            return analyzer.analyze_stage5_interpretability(self)
        else:
            return performanceMetrics.Stage5InterpretabilityReport()
    
    def run_stage5_robustness_tests(self, 
                                   n_runs_per_test: int = 3,
                                   performance_threshold: float = 0.7,
                                   verbose: bool = True) -> performanceMetrics.Stage5RobustnessReport:
        """
        Run comprehensive Stage 5 robustness tests.
        
        Parameters
        ----------
        n_runs_per_test : int, default=3
            Number of runs for each test
        performance_threshold : float, default=0.7
            Minimum performance threshold for passing tests
        verbose : bool, default=True
            Whether to print progress information
            
        Returns
        -------
        Stage5RobustnessReport
            Comprehensive robustness test report
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running robustness tests")
        
        # Create model factory function
        def model_factory(**kwargs):
            params = self.get_params()
            params.update(kwargs)
            return MultiModalEnsembleModel(**params)
        
        # Create test data factory function
        def data_factory(**kwargs):
            return self.train_data, self.train_labels
        
        # Create robustness tester
        tester = performanceMetrics.Stage5RobustnessTester(
            base_model_factory=model_factory,
            test_data_factory=data_factory
        )
        
        # Run comprehensive robustness tests
        return tester.run_comprehensive_robustness_tests(
            n_runs_per_test=n_runs_per_test,
            performance_threshold=performance_threshold,
            verbose=verbose
        )
    
    def run_single_stage5_robustness_test(self, 
                                         test_category: str,
                                         test_config: Dict[str, Any],
                                         n_runs: int = 3,
                                         performance_threshold: float = 0.7) -> performanceMetrics.RobustnessTestResult:
        """
        Run a single robustness test for Stage 5.
        
        Parameters
        ----------
        test_category : str
            Category of test to run (e.g., 'transformer_fusion', 'dynamic_weighting')
        test_config : dict
            Configuration for the test
        n_runs : int, default=3
            Number of runs for the test
        performance_threshold : float, default=0.7
            Minimum performance threshold for passing the test
            
        Returns
        -------
        RobustnessTestResult
            Result of the single robustness test
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running robustness tests")
        
        # Create model factory function
        def model_factory(**kwargs):
            params = self.get_params()
            params.update(kwargs)
            return MultiModalEnsembleModel(**params)
        
        # Create test data factory function
        def data_factory(**kwargs):
            return self.train_data, self.train_labels
        
        # Create robustness tester
        tester = performanceMetrics.Stage5RobustnessTester(
            base_model_factory=model_factory,
            test_data_factory=data_factory
        )
        
        # Run single test
        return tester._run_single_test(
            test_name=f"custom_{test_category}",
            test_category=test_category,
            test_config=test_config,
            n_runs=n_runs,
            threshold=performance_threshold,
            verbose=False
        )

    def evaluate(self, X=None, y=None):
        """
        Comprehensive model evaluation.
        
        Parameters
        ----------
        X : dict, optional
            Dictionary of modality data. If None, uses test data.
        y : array-like, optional
            True labels. If None, uses test labels.
        
        Returns
        -------
        report : ModelPerformanceReport
            Comprehensive performance report.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before evaluation")
        
        if X is None:
            X = self.test_data
        if y is None:
            y = self.test_labels
        
        # Convert to numpy arrays and clean data
        X = {k: np.asarray(v) for k, v in X.items()}
        X = self._clean_prediction_data(X)
        y = np.asarray(y)
        
        # Get predictions
        predictions = self.predict(X)
        
        # Create evaluator
        evaluator = performanceMetrics.PerformanceEvaluator(task_type=self.task_type_)
        
        # Evaluate model with comprehensive metrics
        report = evaluator.evaluate_model(
            lambda X_dict: self.predict(X_dict), 
            X, 
            y, 
            model_name="MultiModalEnsemble",
            return_probabilities=self.return_probabilities,
            measure_efficiency=self.measure_efficiency,
            n_efficiency_runs=self.n_efficiency_runs
        )
        
        return report
    
    def get_feature_importance(self):
        """
        Get feature importance scores.
        
        Returns
        -------
        importance : dict
            Dictionary of feature importance scores per modality.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Aggregate feature importance from all learners
        all_importance = {}
        
        for learner in self.trained_learners_:
            if hasattr(learner, 'get_feature_importance'):
                try:
                    importance = learner.get_feature_importance()
                    for modality, scores in importance.items():
                        if modality not in all_importance:
                            all_importance[modality] = []
                        all_importance[modality].append(scores)
                except Exception:
                    continue
        
        # Average importance scores
        avg_importance = {}
        for modality, scores_list in all_importance.items():
            if scores_list:
                try:
                    # Try to average the scores
                    avg_importance[modality] = np.mean(scores_list, axis=0)
                except ValueError:
                    # If shapes are incompatible, just take the first one
                    avg_importance[modality] = scores_list[0]
        
        return avg_importance
    
    def get_ensemble_stats(self, return_detailed: bool = False):
        """
        Get ensemble statistics from modalityDropoutBagger.
        
        Parameters
        ----------
        return_detailed : bool, default=False
            Whether to return detailed bag information.
        
        Returns
        -------
        stats : dict
            Ensemble statistics including modality coverage, diversity metrics, and dropout statistics.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting ensemble stats")
        
        return self.bagger.get_ensemble_stats(return_detailed=return_detailed)
    
    def get_bag_info(self, bag_id: int):
        """
        Get information about a specific bag.
        
        Parameters
        ----------
        bag_id : int
            ID of the bag to get information for.
        
        Returns
        -------
        info : dict
            Bag information including modalities, dropout rate, sample count, and diversity score.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting bag info")
        
        if bag_id >= len(self.bags):
            raise ValueError(f"Bag ID {bag_id} out of range. Total bags: {len(self.bags)}")
        
        return self.bagger.get_bag_info(bag_id)
    
    def get_bag_data(self, bag_id: int, return_metadata: bool = False):
        """
        Get data for a specific bag.
        
        Parameters
        ----------
        bag_id : int
            ID of the bag to get data for.
        return_metadata : bool, default=False
            Whether to return metadata about the bag.
        
        Returns
        -------
        bag_data : dict or tuple
            If return_metadata=False: Dictionary of bag data
            If return_metadata=True: Tuple of (bag_data, modality_mask, metadata)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting bag data")
        
        if bag_id >= len(self.bags):
            raise ValueError(f"Bag ID {bag_id} out of range. Total bags: {len(self.bags)}")
        
        # Add labels to train_data for get_bag_data
        train_data_with_labels = self.train_data.copy()
        train_data_with_labels['labels'] = self.train_labels
        
        return self.bagger.get_bag_data(
            bag_id=bag_id,
            multimodal_data=train_data_with_labels,
            return_metadata=return_metadata
        )
    
    def save_ensemble(self, filepath: str):
        """
        Save the ensemble configuration to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the ensemble configuration.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving ensemble")
        
        self.bagger.save_ensemble(filepath)
    
    @classmethod
    def load_ensemble(cls, filepath: str):
        """
        Load an ensemble configuration from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved ensemble configuration.
        
        Returns
        -------
        bagger : ModalityDropoutBagger
            Loaded ensemble configuration.
        """
        return modalityDropoutBagger.ModalityDropoutBagger.load_ensemble(filepath)
    
    def get_learner_summary(self):
        """
        Get learner summary from modalityAwareBaseLearnerSelector.
        
        Returns
        -------
        summary : dict
            Comprehensive learner summary including distribution, performance, and resource usage.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting learner summary")
        
        return self.selector.get_learner_summary()
    
    def get_performance_report(self):
        """
        Get performance report from modalityAwareBaseLearnerSelector.
        
        Returns
        -------
        report : dict
            Performance report with detailed metrics and statistics.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting performance report")
        
        return self.selector.get_performance_report()
    
    def predict_learner_performance(self, learner_config, bag_characteristics):
        """
        Predict performance for a specific learner configuration.
        
        Parameters
        ----------
        learner_config : LearnerConfig
            Learner configuration to predict performance for.
        bag_characteristics : dict
            Characteristics of the bag (sample count, feature dimensionality, etc.).
        
        Returns
        -------
        performance : float
            Predicted performance score.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before predicting learner performance")
        
        return self.selector.predict_learner_performance(learner_config, bag_characteristics)
    
    def optimize_hyperparameters(self, learner_config, data_sample):
        """
        Optimize hyperparameters for a specific learner configuration.
        
        Parameters
        ----------
        learner_config : LearnerConfig
            Learner configuration to optimize.
        data_sample : Any
            Sample data for hyperparameter optimization.
        
        Returns
        -------
        optimized_config : LearnerConfig
            Optimized learner configuration.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before optimizing hyperparameters")
        
        return self.selector.optimize_hyperparameters(learner_config, data_sample)
    
    def get_learner_configs(self):
        """
        Get all learner configurations from the selector.
        
        Returns
        -------
        configs : dict
            Dictionary of learner configurations or instantiated learners.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting learner configs")
        
        return self.selector.learners
    
    def get_learner_info(self):
        """
        Get information about all learners in a safe format.
        
        Returns
        -------
        info : dict
            Dictionary of learner information.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting learner info")
        
        learner_info = {}
        for learner_id, learner in self.selector.learners.items():
            info = {
                'learner_id': learner_id,
                'learner_type': getattr(learner, 'learner_type', type(learner).__name__),
                'modalities': getattr(learner, 'modalities', []),
                'n_classes': getattr(learner, 'n_classes', None),
                'model_type': getattr(learner, 'model_type', None)
            }
            learner_info[learner_id] = info
        
        return learner_info
    
    def get_training_summary(self):
        """
        Get training summary from trainingPipeline.
        
        Returns
        -------
        summary : dict
            Comprehensive training summary including performance and timing.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting training summary")
        
        return self.pipeline.get_training_summary(self.training_metrics_)
    
    def get_training_metrics(self):
        """
        Get detailed training metrics from trainingPipeline.
        
        Returns
        -------
        metrics : dict
            Dictionary of training metrics for each learner.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting training metrics")
        
        return self.training_metrics_
    
    def get_learner_training_info(self, learner_id=None):
        """
        Get training information for specific learner or all learners.
        
        Parameters
        ----------
        learner_id : str, optional
            Specific learner ID. If None, returns info for all learners.
        
        Returns
        -------
        info : dict
            Training information for the specified learner(s).
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting training info")
        
        if learner_id is not None:
            if learner_id in self.training_metrics_:
                return {
                    'learner_id': learner_id,
                    'metrics': self.training_metrics_[learner_id],
                    'metadata': next((m for m in self.learner_metadata_ if m['learner_id'] == learner_id), {})
                }
            else:
                raise ValueError(f"Learner {learner_id} not found")
        else:
            return {
                lid: {
                    'learner_id': lid,
                    'metrics': metrics,
                    'metadata': next((m for m in self.learner_metadata_ if m['learner_id'] == lid), {})
                }
                for lid, metrics in self.training_metrics_.items()
            }
    
    def get_training_config(self):
        """
        Get the training configuration used for training.
        
        Returns
        -------
        config : dict
            Training configuration parameters.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting training config")
        
        return {
            'task_type': self.task_type_,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'enable_denoising': self.enable_denoising,
            'optimizer_type': self.optimizer_type,
            'scheduler_type': self.scheduler_type,
            'mixed_precision': self.mixed_precision,
            'gradient_clipping': self.gradient_clipping,
            'early_stopping_patience': self.early_stopping_patience,
            'validation_split': self.validation_split,
            'cross_validation_folds': self.cross_validation_folds,
            'save_checkpoints': self.save_checkpoints,
            'tensorboard_logging': self.tensorboard_logging,
            'wandb_logging': self.wandb_logging,
            'log_interval': self.log_interval,
            'eval_interval': self.eval_interval,
            'profile_training': self.profile_training,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'num_workers': self.num_workers,
            'distributed_training': self.distributed_training,
            'compile_model': self.compile_model,
            'amp_optimization_level': self.amp_optimization_level,
            'gradient_scaling': self.gradient_scaling,
            'loss_scale': self.loss_scale,
            'curriculum_stages': self.curriculum_stages,
            'enable_progressive_learning': self.enable_progressive_learning,
            'progressive_stages': self.progressive_stages,
            'dropout_rate': self.dropout_rate,
            'label_smoothing': self.label_smoothing,
            'denoising_weight': self.denoising_weight,
            'denoising_strategy': self.denoising_strategy,
            'denoising_objectives': self.denoising_objectives,
            'denoising_modalities': self.denoising_modalities,
            # EnsemblePrediction parameters
            'calibrate_uncertainty': self.calibrate_uncertainty,
            'device': self.device
        }
    
    def save(self, filepath):
        """
        Save the fitted model to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the model.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath):
        """
        Load a fitted model from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model.
        
        Returns
        -------
        model : MultiModalEnsembleModel
            Loaded model.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_bags': self.n_bags,
            'dropout_strategy': self.dropout_strategy,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'random_state': self.random_state,
            'task_type': self.task_type,
            'optimization_strategy': self.optimization_strategy,
            'enable_denoising': self.enable_denoising,
            'aggregation_strategy': self.aggregation_strategy,
            'uncertainty_method': self.uncertainty_method,
            'max_dropout_rate': self.max_dropout_rate,
            'min_modalities': self.min_modalities,
            'sample_ratio': self.sample_ratio,
            'diversity_target': self.diversity_target,
            'feature_sampling': self.feature_sampling,
            'verbose': self.verbose,
            # DataIntegration parameters
            'validate_data': self.validate_data,
            'memory_efficient': self.memory_efficient,
            'handle_nan': self.handle_nan,
            'handle_inf': self.handle_inf,
            'normalize_data': self.normalize_data,
            'remove_outliers': self.remove_outliers,
            'outlier_std': self.outlier_std,
            # ModalityDropoutBagger parameters
            'enable_validation': self.enable_validation,
            # ModalityAwareBaseLearnerSelector parameters
            'learner_preferences': self.learner_preferences,
            'performance_threshold': self.performance_threshold,
            'resource_limit': self.resource_limit,
            'validation_strategy': self.validation_strategy,
            'hyperparameter_tuning': self.hyperparameter_tuning,
            # TrainingPipeline parameters
            'optimizer_type': self.optimizer_type,
            'scheduler_type': self.scheduler_type,
            'mixed_precision': self.mixed_precision,
            'gradient_clipping': self.gradient_clipping,
            'early_stopping_patience': self.early_stopping_patience,
            'validation_split': self.validation_split,
            'cross_validation_folds': self.cross_validation_folds,
            'save_checkpoints': self.save_checkpoints,
            'tensorboard_logging': self.tensorboard_logging,
            'wandb_logging': self.wandb_logging,
            'log_interval': self.log_interval,
            'eval_interval': self.eval_interval,
            'profile_training': self.profile_training,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'num_workers': self.num_workers,
            'distributed_training': self.distributed_training,
            'compile_model': self.compile_model,
            'amp_optimization_level': self.amp_optimization_level,
            'gradient_scaling': self.gradient_scaling,
            'loss_scale': self.loss_scale,
            'curriculum_stages': self.curriculum_stages,
            'enable_progressive_learning': self.enable_progressive_learning,
            'progressive_stages': self.progressive_stages,
            'dropout_rate': self.dropout_rate,
            'label_smoothing': self.label_smoothing,
            'denoising_weight': self.denoising_weight,
            'denoising_strategy': self.denoising_strategy,
            'denoising_objectives': self.denoising_objectives,
            'denoising_modalities': self.denoising_modalities,
            # EnsemblePrediction parameters
            'calibrate_uncertainty': self.calibrate_uncertainty,
            'device': self.device,
            # PerformanceMetrics parameters
            'measure_efficiency': self.measure_efficiency,
            'n_efficiency_runs': self.n_efficiency_runs,
            'return_probabilities': self.return_probabilities
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self


# Example usage
if __name__ == "__main__":
    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create multimodal data
    text_data = np.random.randn(n_samples, n_features)
    image_data = np.random.randn(n_samples, n_features)
    metadata = np.random.randn(n_samples, n_features)
    
    # Create labels (classification example)
    labels = np.random.randint(0, 3, n_samples)
    
    # Initialize model
    model = MultiModalEnsembleModel(
        n_bags=3, 
        epochs=2, 
        batch_size=16,
        random_state=42,
        verbose=True
    )
    
    # Fit model
    print("Fitting model...")
    model.fit({
        'text': text_data,
        'image': image_data,
        'metadata': metadata
    }, labels)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict({
        'text': text_data[:10],
        'image': image_data[:10],
        'metadata': metadata[:10]
    })
    print(f"Predictions: {predictions}")
    
    # Get probabilities
    probabilities = model.predict_proba({
        'text': text_data[:10],
        'image': image_data[:10],
        'metadata': metadata[:10]
    })
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Evaluate model
    print("Evaluating model...")
    score = model.score({
        'text': text_data[:100],
        'image': image_data[:100],
        'metadata': metadata[:100]
    }, labels[:100])
    print(f"Score: {score:.4f}")
    
    # Get feature importance
    importance = model.get_feature_importance()
    print(f"Feature importance keys: {list(importance.keys())}")
    
    print("Demo completed successfully!")
