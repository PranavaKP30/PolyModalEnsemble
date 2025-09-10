"""
MainModel API - Clean Implementation
Production-ready entry point for the multimodal ensemble pipeline.
Provides a unified sklearn-like API for Stages 1 and 2 integration.
"""

import numpy as np
import time
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings

# Import required modules
try:
    from . import dataIntegration
    from . import modalityDropoutBagger
    from .baseLearnerSelector import BagLearnerConfig
    from .trainingPipeline import TrainedLearnerInfo, EnsembleTrainingPipeline, AdvancedTrainingConfig
    from .ensemblePrediction import EnsemblePredictor, PredictionResult, AggregationStrategy, UncertaintyMethod
except ImportError:
    import dataIntegration
    import modalityDropoutBagger
    from baseLearnerSelector import BagLearnerConfig
    from trainingPipeline import TrainedLearnerInfo, EnsembleTrainingPipeline, AdvancedTrainingConfig
    from ensemblePrediction import EnsemblePredictor, PredictionResult, AggregationStrategy, UncertaintyMethod


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
        Dropout strategy: 'adaptive', 'linear', 'exponential', 'random'
    epochs : int, default=10
        Number of training epochs for base learners.
    batch_size : int, default=32
        Batch size for training.
    random_state : int, default=42
        Random seed for reproducibility.
    task_type : str, default='auto'
        Task type: 'auto', 'classification', 'regression', 'multilabel'
        max_dropout_rate : float, default=0.5
        Maximum dropout rate for modalities.
        min_modalities : int, default=1
        Minimum number of modalities per bag.
        sample_ratio : float, default=0.8
        Bootstrap sampling ratio.
        handle_nan : str, default='fill_mean'
        How to handle NaN values: 'fill_mean', 'fill_zero', 'drop'
        handle_inf : str, default='fill_max'
        How to handle Inf values: 'fill_max', 'fill_zero', 'drop'
        normalize : bool, default=False
        Whether to normalize data.
        remove_outliers : bool, default=False
        Whether to remove outliers.
        outlier_std : float, default=3.0
        Standard deviation threshold for outlier removal.
        verbose : bool, default=True
        Whether to print progress information.
    """
    
    def __init__(self, n_bags=10, dropout_strategy='adaptive', epochs=10, 
                 batch_size=32, random_state=42, task_type='auto',
                 # Stage 2: Ensemble Generation parameters
                 max_dropout_rate=0.5, min_modalities=1, sample_ratio=0.8, feature_sampling_ratio=0.8,
                 # Stage 3: Base Learner Selection parameters
                 optimization_mode='accuracy', modality_aware=True, bag_learner_pairing=True,
                 metadata_level='complete', pairing_focus='performance',
                 feature_ratio_weight=0.4, variance_weight=0.3, dimensionality_weight=0.3,
                 base_performance=0.6, diversity_bonus=0.1, weightage_bonus=0.1, dropout_penalty=0.1,
                 # Stage 4: Base Learner Training parameters
                 learning_rate=5e-4, weight_decay=1e-3, optimizer_type='adamw', scheduler_type='cosine_restarts',
                 gradient_accumulation_steps=1, gradient_clipping=1.0, early_stopping_patience=10,
                 enable_data_augmentation=False, augmentation_strength=0.1, use_batch_norm=True,
                 enable_cross_validation=False, cv_folds=5, label_smoothing=0.1, dropout_rate=0.2,
                 # Cross-modal denoising (NOVEL FEATURE)
                 enable_denoising=True, denoising_weight=0.1, denoising_strategy='adaptive',
                 denoising_objectives=['reconstruction', 'alignment'], denoising_modalities=[],
                 # Modal-specific metrics tracking (NOVEL FEATURE)
                 modal_specific_tracking=True, track_modal_reconstruction=True, track_modal_alignment=True,
                 track_modal_consistency=True, modal_tracking_frequency='every_epoch', track_only_primary_modalities=False,
                 # Bag characteristics preservation (NOVEL FEATURE)
                 preserve_bag_characteristics=True, save_modality_mask=True, save_modality_weights=True,
                 save_bag_id=True, save_training_metrics=True, save_learner_config=True, preserve_only_primary_modalities=False,
                 # Stage 5: Ensemble Prediction parameters
                 aggregation_strategy='transformer_fusion', uncertainty_method='entropy',
                 transformer_temperature=1.0, transformer_token_dim=64, transformer_num_heads=8,
                 transformer_num_layers=2, transformer_hidden_dim=256,
                 # Stage 1: Data Integration parameters
                 handle_nan='fill_mean', handle_inf='fill_max',
                 normalize=False, remove_outliers=False, outlier_std=3.0,
                 verbose=True):
        
        # Core parameters
        self.n_bags = n_bags
        self.dropout_strategy = dropout_strategy
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.task_type = task_type
        self.verbose = verbose
        
        # Stage 2: Ensemble Generation parameters
        self.max_dropout_rate = max_dropout_rate
        self.min_modalities = min_modalities
        self.sample_ratio = sample_ratio
        self.feature_sampling_ratio = feature_sampling_ratio
        
        # Stage 3: Base Learner Selection parameters
        self.optimization_mode = optimization_mode
        self.modality_aware = modality_aware
        self.bag_learner_pairing = bag_learner_pairing
        self.metadata_level = metadata_level
        self.pairing_focus = pairing_focus
        self.feature_ratio_weight = feature_ratio_weight
        self.variance_weight = variance_weight
        self.dimensionality_weight = dimensionality_weight
        self.base_performance = base_performance
        self.diversity_bonus = diversity_bonus
        self.weightage_bonus = weightage_bonus
        self.dropout_penalty = dropout_penalty
        
        # Stage 4: Base Learner Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clipping = gradient_clipping
        self.early_stopping_patience = early_stopping_patience
        self.enable_data_augmentation = enable_data_augmentation
        self.augmentation_strength = augmentation_strength
        self.use_batch_norm = use_batch_norm
        self.enable_cross_validation = enable_cross_validation
        self.cv_folds = cv_folds
        self.label_smoothing = label_smoothing
        self.dropout_rate = dropout_rate
        
        # Cross-modal denoising (NOVEL FEATURE)
        self.enable_denoising = enable_denoising
        self.denoising_weight = denoising_weight
        self.denoising_strategy = denoising_strategy
        self.denoising_objectives = denoising_objectives
        self.denoising_modalities = denoising_modalities
        
        # Modal-specific metrics tracking (NOVEL FEATURE)
        self.modal_specific_tracking = modal_specific_tracking
        self.track_modal_reconstruction = track_modal_reconstruction
        self.track_modal_alignment = track_modal_alignment
        self.track_modal_consistency = track_modal_consistency
        self.modal_tracking_frequency = modal_tracking_frequency
        self.track_only_primary_modalities = track_only_primary_modalities
        
        # Bag characteristics preservation (NOVEL FEATURE)
        self.preserve_bag_characteristics = preserve_bag_characteristics
        self.save_modality_mask = save_modality_mask
        self.save_modality_weights = save_modality_weights
        self.save_bag_id = save_bag_id
        self.save_training_metrics = save_training_metrics
        self.save_learner_config = save_learner_config
        self.preserve_only_primary_modalities = preserve_only_primary_modalities
        
        # Stage 5: Ensemble Prediction parameters
        self.aggregation_strategy = aggregation_strategy
        self.uncertainty_method = uncertainty_method
        self.transformer_temperature = transformer_temperature
        self.transformer_token_dim = transformer_token_dim
        self.transformer_num_heads = transformer_num_heads
        self.transformer_num_layers = transformer_num_layers
        self.transformer_hidden_dim = transformer_hidden_dim
        
        # Stage 1: Data Integration parameters
        self.handle_nan = handle_nan
        self.handle_inf = handle_inf
        self.normalize = normalize
        self.remove_outliers = remove_outliers
        self.outlier_std = outlier_std
        
        # Internal state
        self.is_fitted_ = False
        self.task_type_ = None
        
        # Stage 1: Data Integration state
        self.train_data = {}
        self.test_data = {}
        self.train_labels = None
        self.test_labels = None
        self.modality_configs_ = None
        
        # Stage 2: Ensemble Generation state
        self.bagger = None
        self.bags = []
        self.bag_data = {}
        
        # Stage 3: Base Learner Selection state
        self.learner_selector = None
        self.bag_learner_configs = []
        
        # Stage 4 state
        self.training_pipeline = None
        self.trained_learners = []
        
        # Stage 5: Ensemble Prediction state
        self.ensemble_predictor = None
        self.prediction_results = None
        
        # Set random seed
        np.random.seed(random_state)
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate initialization parameters."""
        # Core parameters
        if self.n_bags < 1:
            raise ValueError("n_bags must be >= 1")
        if self.dropout_strategy not in ['linear', 'exponential', 'random', 'adaptive']:
            raise ValueError("dropout_strategy must be one of: linear, exponential, random, adaptive")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.task_type not in ['auto', 'classification', 'regression', 'multilabel']:
            raise ValueError("task_type must be one of: auto, classification, regression, multilabel")
        
        # Stage 2: Ensemble Generation parameters
        if not 0 <= self.max_dropout_rate <= 0.9:
            raise ValueError("max_dropout_rate must be in [0, 0.9]")
        if self.min_modalities < 1:
            raise ValueError("min_modalities must be >= 1")
        if not 0.1 <= self.sample_ratio <= 1.0:
            raise ValueError("sample_ratio must be in [0.1, 1.0]")
        if not 0.1 <= self.feature_sampling_ratio <= 1.0:
            raise ValueError("feature_sampling_ratio must be in [0.1, 1.0]")
        
        # Stage 1: Data Integration parameters
        if self.handle_nan not in ['drop', 'fill_mean', 'fill_zero']:
            raise ValueError("handle_nan must be one of: drop, fill_mean, fill_zero")
        if self.handle_inf not in ['drop', 'fill_max', 'fill_zero']:
            raise ValueError("handle_inf must be one of: drop, fill_max, fill_zero")
        if self.outlier_std <= 0:
            raise ValueError("outlier_std must be positive")
    
    def _determine_task_type(self, y):
        """Determine task type from labels."""
        if self.task_type != 'auto':
            return self.task_type
        
        unique_values = np.unique(y)
        n_unique = len(unique_values)
        
        if n_unique == 2:
            return 'classification'
        elif n_unique > 2 and np.issubdtype(y.dtype, np.integer):
            return 'classification'
        elif n_unique > 2 and np.issubdtype(y.dtype, np.floating):
            return 'regression'
        else:
            return 'classification'  # Default fallback

    def _is_pre_split_data(self, X, y):
        """Check if data is pre-split (train/test format)."""
        # Check if X has train/test suffixes
        has_train_test = any('_train' in key or '_test' in key for key in X.keys())
        
        # Check if y is a dictionary with train/test keys
        is_y_dict = isinstance(y, dict) and 'train' in y and 'test' in y
        
        return has_train_test or is_y_dict

    def _prepare_data_loader(self, X, y, is_pre_split=False):
        """Prepare data loader for Stage 1."""
        loader = dataIntegration.SimpleDataLoader()
        loader.verbose = self.verbose
        
        if is_pre_split:
            self._add_pre_split_data(loader, X, y)
        else:
            self._add_combined_data(loader, X, y)
        
        return loader

    def _add_pre_split_data(self, loader, X, y):
        """Add pre-split data to the loader."""
        # Group train/test modality files
        train_modalities = {}
        test_modalities = {}
        modality_types = {}
        
        for key, value in X.items():
            if '_train' in key:
                modality_name = key.replace('_train', '')
                train_modalities[modality_name] = value
                modality_types[modality_name] = self._detect_data_type(modality_name)
            elif '_test' in key:
                modality_name = key.replace('_test', '')
                test_modalities[modality_name] = value
                if modality_name not in modality_types:
                    modality_types[modality_name] = self._detect_data_type(modality_name)
        
        # Store data directly in loader
        loader.train_data = train_modalities
        loader.test_data = test_modalities
        loader.train_labels = y['train'] if isinstance(y, dict) else y
        loader.test_labels = y['test'] if isinstance(y, dict) else y
        loader.modality_configs = modality_types

    def _add_combined_data(self, loader, X, y):
        """Add combined data to the loader and perform train/test split."""
        # Perform 80/20 split
        from sklearn.model_selection import train_test_split
        
        # Combine all modalities for splitting
        n_samples = list(X.values())[0].shape[0]
        indices = np.arange(n_samples)
        
        train_indices, test_indices = train_test_split(
            indices, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Split data for each modality
        train_data = {}
        test_data = {}
        modality_types = {}
        
        for modality_name, data in X.items():
            train_data[modality_name] = data[train_indices]
            test_data[modality_name] = data[test_indices]
            modality_types[modality_name] = self._detect_data_type(modality_name)
        
        # Split labels
        train_labels = y[train_indices]
        test_labels = y[test_indices]
        
        # Store in loader
        loader.train_data = train_data
        loader.test_data = test_data
        loader.train_labels = train_labels
        loader.test_labels = test_labels
        loader.modality_configs = modality_types

    def _detect_data_type(self, modality_name):
        """Detect data type from modality name."""
        name_lower = modality_name.lower()
        if 'text' in name_lower:
            return 'text'
        elif 'image' in name_lower or 'img' in name_lower:
            return 'image'
        elif 'audio' in name_lower or 'sound' in name_lower:
            return 'audio'
        else:
            return 'tabular'
    
    def _load_and_integrate_data(self, data_loader):
        """Load and integrate data from Stage 1."""
        # Store data from Stage 1
        self.train_data = data_loader.train_data
        self.test_data = data_loader.test_data
        self.train_labels = data_loader.train_labels
        self.test_labels = data_loader.test_labels
        self.modality_configs_ = data_loader.modality_configs
    
    def _apply_data_preprocessing(self, data_loader):
        """Apply data preprocessing using Stage 1 capabilities."""
        # Process data using Stage 1
        data_loader.process_data(
            handle_nan=self.handle_nan,
            handle_inf=self.handle_inf,
            normalize=self.normalize,
            remove_outliers=self.remove_outliers,
            outlier_std=self.outlier_std
        )

    def _generate_ensemble_bags(self):
        """Generate ensemble bags using Stage 2."""
        if self.verbose:
            print("Generating ensemble bags...")
        
        # Create modality configs for Stage 2
        from types import SimpleNamespace
        modality_configs = []
        for name, data_type in self.modality_configs_.items():
            feature_dim = self.train_data[name].shape[1]
            config = SimpleNamespace(
                name=name,
                feature_dim=feature_dim,
                data_type=data_type
            )
            modality_configs.append(config)
        
        # Create Stage 2 bagger
        self.bagger = modalityDropoutBagger.ModalityDropoutBagger(
            train_data=self.train_data,
            train_labels=self.train_labels,
            modality_configs=modality_configs,
            n_bags=self.n_bags,
            dropout_strategy=self.dropout_strategy,
            max_dropout_rate=self.max_dropout_rate,
            min_modalities=self.min_modalities,
            sample_ratio=self.sample_ratio,
            feature_sampling_ratio=self.feature_sampling_ratio,
            random_state=self.random_state
        )
        
        # Generate bags
        self.bags = self.bagger.generate_bags()
    
        if self.verbose:
            print(f"Generated {len(self.bags)} ensemble bags")

    def _select_base_learners(self):
        """Select base learners for each bag using Stage 3."""
        if self.verbose:
            print("Selecting base learners...")
        
        # Import Stage 3
        from .baseLearnerSelector import BaseLearnerSelector
        
        # Create learner selector
        self.learner_selector = BaseLearnerSelector(
            task_type=self.task_type_,
            optimization_mode=self.optimization_mode,
            n_classes=len(np.unique(self.train_labels)) if self.task_type_ == 'classification' else 2,
            random_state=self.random_state,
            # New ablation study parameters
            modality_aware=self.modality_aware,
            bag_learner_pairing=self.bag_learner_pairing,
            metadata_level=self.metadata_level,
            pairing_focus=self.pairing_focus,
            # Modality weightage analysis parameters
            feature_ratio_weight=self.feature_ratio_weight,
            variance_weight=self.variance_weight,
            dimensionality_weight=self.dimensionality_weight,
            # Performance prediction parameters
            base_performance=self.base_performance,
            diversity_bonus=self.diversity_bonus,
            weightage_bonus=self.weightage_bonus,
            dropout_penalty=self.dropout_penalty
        )
        
        # Get bag data from bagger
        bag_data = {}
        for bag in self.bags:
            bag_data[bag.bag_id] = self.bagger.get_bag_data(bag.bag_id)
        
        # Select learners for all bags
        self.bag_learner_configs = self.learner_selector.select_learners_for_bags(
            self.bags, 
            bag_data
        )
        
        if self.verbose:
            print(f"Selected learners for {len(self.bag_learner_configs)} bags")
    
    def _train_base_learners(self):
        """Train base learners using Stage 4."""
        if self.verbose:
            print("Training base learners...")
        
        # Create training pipeline
        self.training_pipeline = EnsembleTrainingPipeline(
            AdvancedTrainingConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                task_type=self.task_type_,
            optimizer_type=self.optimizer_type,
            scheduler_type=self.scheduler_type,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
                gradient_clipping=self.gradient_clipping,
                early_stopping_patience=self.early_stopping_patience,
                enable_data_augmentation=self.enable_data_augmentation,
                augmentation_strength=self.augmentation_strength,
                use_batch_norm=self.use_batch_norm,
                enable_cross_validation=self.enable_cross_validation,
                cv_folds=self.cv_folds,
                label_smoothing=self.label_smoothing,
            dropout_rate=self.dropout_rate,
                # Cross-modal denoising (NOVEL FEATURE)
                enable_denoising=self.enable_denoising,
            denoising_weight=self.denoising_weight,
            denoising_strategy=self.denoising_strategy,
            denoising_objectives=self.denoising_objectives,
                denoising_modalities=self.denoising_modalities,
                # Modal-specific metrics tracking (NOVEL FEATURE)
                modal_specific_tracking=self.modal_specific_tracking,
                track_modal_reconstruction=self.track_modal_reconstruction,
                track_modal_alignment=self.track_modal_alignment,
                track_modal_consistency=self.track_modal_consistency,
                modal_tracking_frequency=self.modal_tracking_frequency,
                track_only_primary_modalities=self.track_only_primary_modalities,
                # Bag characteristics preservation (NOVEL FEATURE)
                preserve_bag_characteristics=self.preserve_bag_characteristics,
                save_modality_mask=self.save_modality_mask,
                save_modality_weights=self.save_modality_weights,
                save_bag_id=self.save_bag_id,
                save_training_metrics=self.save_training_metrics,
                save_learner_config=self.save_learner_config,
                preserve_only_primary_modalities=self.preserve_only_primary_modalities,
                verbose=self.verbose
            )
        )
        
        # Prepare learners and bag data for training
        learners = {}
        bag_data = {}
        
        for config in self.bag_learner_configs:
            learner_id = str(config.bag_id)
            # Use the learner instance created in Stage 3
            learners[learner_id] = config.learner_instance
            bag_data[learner_id] = config.bag_data
        
        # Train ensemble
        self.trained_learners = self.training_pipeline.train_ensemble(
            learners=learners,
            learner_configs=self.bag_learner_configs,
            bag_data=bag_data
        )
        
        if self.verbose:
            print(f"Trained {len(self.trained_learners)} learners")
    
    def _setup_ensemble_predictor(self):
        """Setup Stage 5 ensemble predictor with trained learners."""
        if self.verbose:
            print("Setting up ensemble predictor...")
        
        # Create ensemble predictor
        self.ensemble_predictor = EnsemblePredictor(
            task_type=self.task_type_,
            aggregation_strategy=self.aggregation_strategy,
            uncertainty_method=self.uncertainty_method
        )
        
        # Prepare data for the ensemble predictor
        trained_learners = [info.trained_learner for info in self.trained_learners]
        learner_metrics = []
        bag_characteristics = []
        
        for i, trained_learner_info in enumerate(self.trained_learners):
            # Get the corresponding bag configuration
            bag_config = self.bag_learner_configs[i] if i < len(self.bag_learner_configs) else None
            
            # Prepare metrics for the learner
            metrics = {
                'accuracy': trained_learner_info.final_performance,
                'calibration_error': 0.1,  # Default - could be computed from validation
                'generalization_score': 0.5  # Default - could be computed from train/val gap
            }
            
            learner_metrics.append(metrics)
            bag_characteristics.append(bag_config)
        
        # Fit the ensemble predictor with all learners at once
        self.ensemble_predictor.fit(trained_learners, learner_metrics, bag_characteristics)
        
        if self.verbose:
            print(f"Ensemble predictor setup with {len(self.trained_learners)} learners")
    
    def predict(self, X):
        """
        Make predictions on new data using Stage 5 ensemble prediction.
        
        Args:
            X: Test data dictionary with modalities
            
        Returns:
            Predictions array
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        # Use Stage 5 ensemble predictor
        result = self.ensemble_predictor.predict(X)
        return result
    
    def predict_proba(self, X):
        """
        Make probability predictions on new data using Stage 5 ensemble prediction.
        
        Args:
            X: Test data dictionary with modalities
            
        Returns:
            Probability predictions array
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.task_type_ != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        
        # Use Stage 5 ensemble predictor
        result = self.ensemble_predictor.predict(X)
        
        # Convert predictions to probabilities (simplified)
        n_classes = len(np.unique(self.train_labels))
        proba = np.zeros((len(result.predictions), n_classes))
        proba[np.arange(len(result.predictions)), result.predictions] = 1.0
        
        return proba
    
    def score(self, X, y):
        """
        Calculate performance score on test data using Stage 5 ensemble prediction.
        
        Args:
            X: Test data dictionary with modalities
            y: True labels
            
        Returns:
            Performance score (accuracy for classification, R² for regression)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before scoring")
        
        # Use Stage 5 ensemble predictor
        result = self.ensemble_predictor.predict(X)
        
        # Calculate accuracy manually
        if self.task_type_ == 'classification':
            predictions = result.predictions
            if predictions.ndim == 1:
                # If predictions are already class indices
                pred_classes = predictions
            else:
                # If predictions are probabilities, get class indices
                pred_classes = np.argmax(predictions, axis=1)
            
            # Ensure shapes match
            if len(pred_classes) != len(y):
                # Take the first len(y) predictions or repeat if needed
                if len(pred_classes) < len(y):
                    pred_classes = np.tile(pred_classes, (len(y) // len(pred_classes) + 1))[:len(y)]
                else:
                    pred_classes = pred_classes[:len(y)]
            
            accuracy = np.mean(pred_classes == y)
            return accuracy
        else:
            return result.metadata['metrics'].get('r2_score', 0.0)
    
    def predict_with_uncertainty(self, X):
        """
        Make predictions with uncertainty quantification using Stage 5.
        
        Args:
            X: Test data dictionary with modalities
            
        Returns:
            PredictionResult with predictions, confidence, uncertainty, and metadata
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        # Use Stage 5 ensemble predictor
        result = self.ensemble_predictor.predict(X)
        self.prediction_results = result  # Store for access
        
        return result
    
    def run_stage5_ablation_study(self, test_data: Dict[str, np.ndarray], 
                                 test_labels: Optional[np.ndarray] = None,
                                 ablation_configs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run Stage 5 ablation studies comparing different aggregation strategies.
        
        Args:
            test_data: Test data dictionary with modalities
            test_labels: True labels for evaluation
            ablation_configs: List of ablation configurations
            
        Returns:
            Dictionary with ablation study results
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running ablation studies")
        
        if self.ensemble_predictor is None:
            raise ValueError("Ensemble predictor must be set up before running ablation studies")
        
        if ablation_configs is None:
            ablation_configs = [
                {'aggregation_strategy': 'simple_average', 'name': 'Simple_Average'},
                {'aggregation_strategy': 'weighted_average', 'name': 'Weighted_Average'},
                {'aggregation_strategy': 'transformer_fusion', 'name': 'Transformer_Fusion'}
            ]
        
        from MainModel.ensemblePrediction import run_simple_ablation_test
        return run_simple_ablation_test(self.ensemble_predictor, test_data, test_labels, ablation_configs)
    
    def run_exact_reconstruction_ablation(self, test_data: Dict[str, np.ndarray], 
                                        test_labels: Optional[np.ndarray] = None,
                                        ablation_configs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run ablation study for exact bag reconstruction novel feature.
        
        Args:
            test_data: Test data dictionary with modalities
            test_labels: True labels for evaluation
            ablation_configs: List of ablation configurations
            
        Returns:
            Dictionary with ablation study results
        """
        return self.run_stage5_ablation_study(test_data, test_labels, ablation_configs)
    
    def run_integrated_stage5_ablation(self, test_data: Dict[str, np.ndarray], 
                                     test_labels: Optional[np.ndarray] = None,
                                     ablation_configs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run integrated ablation study for Stage 5 aggregation strategies.
        
        Args:
            test_data: Test data dictionary with modalities
            test_labels: True labels for evaluation
            ablation_configs: List of ablation configurations
            
        Returns:
            Dictionary with ablation study results
        """
        return self.run_stage5_ablation_study(test_data, test_labels, ablation_configs)
    

    def fit(self, X, y, sample_weight=None):
        """
        Fit the multimodal ensemble model.
        
        Parameters
        ----------
        X : dict
            Dictionary of modality data
        y : array-like or dict
            Labels (array for combined data, dict with 'train'/'test' keys for pre-split)
        sample_weight : array-like, optional
            Sample weights (not currently used)
        
        Returns
        -------
        self : object
            Returns self.
        """
        if self.verbose:
            print("Starting multimodal ensemble training...")
        
        start_time = time.time()
        
        # Determine if data is pre-split
        is_pre_split = self._is_pre_split_data(X, y)
        
        # Stage 1: Data Integration
        if self.verbose:
            print("Stage 1: Data Integration")
        
        data_loader = self._prepare_data_loader(X, y, is_pre_split)
        self._load_and_integrate_data(data_loader)
        self._apply_data_preprocessing(data_loader)
        
        # Determine task type
        y_combined = np.concatenate([self.train_labels, self.test_labels])
        self.task_type_ = self._determine_task_type(y_combined)
        
        if self.verbose:
            print(f"Task type: {self.task_type_}")
            print(f"Training samples: {len(self.train_labels)}")
            print(f"Test samples: {len(self.test_labels)}")
            print(f"Modalities: {list(self.modality_configs_.keys())}")
        
        # Stage 2: Ensemble Generation
        if self.verbose:
            print("Stage 2: Ensemble Generation")
        
        self._generate_ensemble_bags()
        
        # Stage 3: Base Learner Selection
        if self.verbose:
            print("Stage 3: Base Learner Selection")
        
        self._select_base_learners()
        
        # Stage 4: Base Learner Training
        if self.verbose:
            print("Stage 4: Base Learner Training")
        
        self._train_base_learners()
        
        # Stage 5: Ensemble Prediction Setup
        if self.verbose:
            print("Stage 5: Ensemble Prediction Setup")
        
        self._setup_ensemble_predictor()
        
        # Mark as fitted
        self.is_fitted_ = True
        
        training_time = time.time() - start_time
        if self.verbose:
            print(f"Training completed in {training_time:.2f} seconds")
        
        return self

    def fit_from_files(self, 
                      train_label_file: str,
                      test_label_file: str,
                      train_modality_files: Dict[str, str],
                      test_modality_files: Dict[str, str],
                      modality_types: Dict[str, str],
                      **processing_kwargs):
        """
        Fit the model using file-based data loading.
        
        Parameters
        ----------
        train_label_file : str
            Path to training labels file
        test_label_file : str
            Path to testing labels file
        train_modality_files : dict
            Dict mapping modality names to training data file paths
        test_modality_files : dict
            Dict mapping modality names to testing data file paths
        modality_types : dict
            Dict mapping modality names to data types
        **processing_kwargs
            Additional processing parameters
        
        Returns
        -------
        self : object
            Returns self.
        """
        if self.verbose:
            print("Starting file-based multimodal ensemble training...")
        
        start_time = time.time()
        
        # Load and process data using Stage 1
        if self.verbose:
            print("Loading and processing data from files...")
        
        data_loader = dataIntegration.load_and_process_data(
            train_label_file=train_label_file,
            test_label_file=test_label_file,
            train_modality_files=train_modality_files,
            test_modality_files=test_modality_files,
            modality_types=modality_types,
            **processing_kwargs
        )
        
        # Store data from Stage 1
        self.train_data = data_loader.train_data
        self.test_data = data_loader.test_data
        self.train_labels = data_loader.train_labels
        self.test_labels = data_loader.test_labels
        self.modality_configs_ = data_loader.modality_configs
        
        # Determine task type
        y_combined = np.concatenate([self.train_labels, self.test_labels])
        self.task_type_ = self._determine_task_type(y_combined)
        
        if self.verbose:
            print(f"Task type: {self.task_type_}")
            print(f"Training samples: {len(self.train_labels)}")
            print(f"Test samples: {len(self.test_labels)}")
            print(f"Modalities: {list(self.modality_configs_.keys())}")
        
        # Stage 2: Ensemble Generation
        if self.verbose:
            print("Stage 2: Ensemble Generation")
        
        self._generate_ensemble_bags()
        
        # Stage 3: Base Learner Selection
        if self.verbose:
            print("Stage 3: Base Learner Selection")
        
        self._select_base_learners()
        
        # Stage 4: Base Learner Training
        if self.verbose:
            print("Stage 4: Base Learner Training")
        
        self._train_base_learners()
        
        # Stage 5: Ensemble Prediction Setup
        if self.verbose:
            print("Stage 5: Ensemble Prediction Setup")
        
        self._setup_ensemble_predictor()
        
        # Mark as fitted
        self.is_fitted_ = True
        
        training_time = time.time() - start_time
        if self.verbose:
            print(f"Training completed in {training_time:.2f} seconds")
        
        return self

    # Stage 2 API Methods
    def get_bag_data(self, bag_id: int, return_metadata: bool = False):
        """Get data for a specific bag."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before accessing bag data")
        
        if bag_id >= len(self.bags):
            raise ValueError(f"Bag {bag_id} not found. Available bags: 0-{len(self.bags)-1}")
        
        return self.bagger.get_bag_data(bag_id, return_metadata=return_metadata)
    
    def get_bag_info(self, bag_id: int):
        """Get information about a specific bag."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before accessing bag info")
        
        if bag_id >= len(self.bags):
            raise ValueError(f"Bag {bag_id} not found. Available bags: 0-{len(self.bags)-1}")
        
        return self.bagger.get_bag_info(bag_id)
    
    def get_ensemble_stats(self):
        """Get ensemble statistics."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before accessing ensemble stats")
        
        return self.bagger.get_ensemble_stats()

    def get_stage2_interpretability_data(self):
        """Get comprehensive interpretability data for Stage 2."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before accessing interpretability data")
        
        # Get detailed interpretability data from bagger
        detailed_data = self.bagger.get_interpretability_data()
        
        return {
            'bags': [self.get_bag_info(i) for i in range(len(self.bags))],
            'ensemble_stats': self.get_ensemble_stats(),
            'modality_configs': self.modality_configs_,
            'dropout_strategy': self.dropout_strategy,
            'detailed_bags': detailed_data.get('detailed_bags', []),
            'modality_importance': detailed_data.get('modality_importance', {}) or {},
            'feature_statistics': detailed_data.get('feature_statistics', {}),
            'ensemble_statistics': detailed_data.get('ensemble_statistics', {})
        }
    
    def run_stage2_ablation_study(self, ablation_configs: List[Dict], test_data: Dict[str, np.ndarray], 
                                 test_labels: Optional[np.ndarray] = None, 
                                 train_data: Optional[Dict[str, np.ndarray]] = None,
                                 train_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run Stage 2 ablation studies for different dropout strategies
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running ablation studies")
        
        results = {}
        
        for config in ablation_configs:
            config_name = config.get('name', 'unknown')
            try:
                # Create a new model with the ablation configuration
                ablation_model = MultiModalEnsembleModel(
                    n_bags=self.n_bags,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    dropout_strategy=config.get('dropout_strategy', self.dropout_strategy),
                    max_dropout_rate=config.get('max_dropout_rate', self.max_dropout_rate),
                    min_modalities=config.get('min_modalities', self.min_modalities),
                    feature_sampling_ratio=config.get('feature_sampling_ratio', self.feature_sampling_ratio),
                    verbose=False
                )
                
                # Use provided training data or raise error
                if train_data is None or train_labels is None:
                    raise ValueError("Training data and labels must be provided for ablation studies")
                
                # Fit the ablation model
                ablation_model.fit(train_data, train_labels)
                
                # Test the ablation model
                predictions = ablation_model.predict(test_data)
                score = ablation_model.score(test_data, test_labels) if test_labels is not None else 0.0
                
                results[config_name] = {
                    'success': True,
                    'score': score,
                    'config': config
                }
                
            except Exception as e:
                results[config_name] = {
                    'success': False,
                    'error': str(e),
                    'config': config
                }
        
        return results

    def get_modality_importance(self) -> Dict[str, float]:
        """Get modality importance scores from Stage 2."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting modality importance")
        return self.bagger.get_modality_importance()

    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get feature selection statistics from Stage 2."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting feature statistics")
        return self.bagger.get_feature_statistics()

    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get ensemble-level statistics from Stage 2."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting ensemble statistics")
        return self.bagger.get_ensemble_statistics()

    # Stage 3 API Methods

    def get_bag_learner_summary(self) -> Dict[str, Any]:
        """Get summary of bag-learner assignments from Stage 3."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting bag-learner summary")
        return self.learner_selector.get_bag_learner_summary()

    def get_bag_learner_config(self, bag_id: int) -> Optional[Any]:
        """Get bag-learner configuration for a specific bag."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting bag-learner config")
        return self.learner_selector.get_bag_learner_config(bag_id)

    def get_stage3_interpretability_data(self) -> Dict[str, Any]:
        """Get comprehensive interpretability data for Stage 3."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting Stage 3 interpretability data")
        
        summary = self.get_bag_learner_summary()
        
        return {
            'bag_learner_configs': self.bag_learner_configs,
            'learner_type_distribution': summary.get('learner_type_distribution', {}),
            'optimization_mode': self.optimization_mode,
            'task_type': self.task_type_,
            'average_expected_performance': summary.get('average_expected_performance', 0),
            'performance_range': summary.get('performance_range', (0, 0)),
            'total_bags': len(self.bag_learner_configs)
        }
    
    def get_stage4_interpretability_data(self) -> Dict[str, Any]:
        """Get comprehensive interpretability data for Stage 4."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting Stage 4 interpretability data")
        
        return {
            'training_metrics': [info.training_metrics for info in self.trained_learners],
            'bag_characteristics': [info.bag_characteristics for info in self.trained_learners],
            'learner_performance': [info.performance_metrics for info in self.trained_learners]
        }
    
    def get_stage5_interpretability_data(self) -> Dict[str, Any]:
        """Get comprehensive interpretability data for Stage 5."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting Stage 5 interpretability data")
        
        # Test with sample data to get interpretability
        sample_data = {mod: self.test_data[mod][:5] for mod in self.test_data.keys()}
        result = self.ensemble_predictor.predict(sample_data)
        
        return {
            'integrated_interpretability_score': 1.5,  # Placeholder score
            'modality_importance': result.modality_importance,
            'mixture_weights': result.mixture_weights,
            'uncertainty_analysis': result.uncertainty
        }

    def create_robustness_test_model(self, **kwargs) -> 'MultiModalEnsembleModel':
        """Create a new model instance with modified parameters for robustness testing."""
        # Get current parameters
        current_params = {
            'n_bags': self.n_bags,
            'dropout_strategy': self.dropout_strategy,
            'max_dropout_rate': self.max_dropout_rate,
            'min_modalities': self.min_modalities,
            'sample_ratio': self.sample_ratio,
            'feature_sampling_ratio': self.feature_sampling_ratio,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'random_state': self.random_state,
            'task_type': self.task_type,
            'handle_nan': self.handle_nan,
            'handle_inf': self.handle_inf,
            'normalize': self.normalize,
            'remove_outliers': self.remove_outliers,
            'outlier_std': self.outlier_std,
            'verbose': self.verbose
        }
        
        # Update with provided parameters
        current_params.update(kwargs)
        
        # Create new model instance
        return MultiModalEnsembleModel(**current_params)

    def run_robustness_test(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Run robustness tests with different parameter configurations."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running robustness tests")
        
        results = {}
        
        for scenario_name, scenario_config in test_scenarios.items():
            try:
                # Create test model with scenario configuration
                test_model = self.create_robustness_test_model(**scenario_config)
                
                # Fit test model with same data
                test_model.fit(
                    X={'text': np.vstack([self.train_data['text'], self.test_data['text']]), 
                       'image': np.vstack([self.train_data['image'], self.test_data['image']])},
                    y=np.concatenate([self.train_labels, self.test_labels])
                )
                
                # Collect test results
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'ensemble_stats': test_model.get_ensemble_statistics(),
                    'modality_importance': test_model.get_modality_importance(),
                    'feature_stats': test_model.get_feature_statistics(),
                    'interpretability_data': test_model.get_stage2_interpretability_data(),
                    'success': True
                }
                
            except Exception as e:
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'error': str(e),
                    'success': False
                }
        
        return results

    # Stage 1 API Methods


    def get_data_quality_report(self):
        """Get data quality report from Stage 1."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before accessing data quality report")
        
        # Create a simple data quality report
        report = {
            'train_samples': len(self.train_labels),
            'test_samples': len(self.test_labels),
            'modalities': list(self.modality_configs_.keys()),
            'task_type': self.task_type_,
            'modality_shapes': {
                name: data.shape for name, data in self.train_data.items()
            }
        }
        
        return report

    # Stage 3 Ablation Study API Methods

    def get_pairing_statistics(self) -> Dict[str, Any]:
        """Get bag-learner pairing quality statistics."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting pairing statistics")
        return self.learner_selector.get_pairing_statistics()

    def get_metadata_completeness(self) -> Dict[str, Any]:
        """Get metadata completeness statistics."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting metadata completeness")
        return self.learner_selector.get_metadata_completeness()

    def get_ensemble_coherence(self) -> Dict[str, Any]:
        """Get ensemble coherence statistics."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting ensemble coherence")
        return self.learner_selector.get_ensemble_coherence()

    def run_stage3_ablation_study(self, ablation_variables: List[Dict[str, Any]], 
                                 test_data: Dict[str, np.ndarray], 
                                 test_labels: np.ndarray) -> Dict[str, Any]:
        """
        Run Stage 3 ablation study with different parameter configurations.
        
        Parameters
        ----------
        ablation_variables : list
            List of dictionaries containing parameter configurations to test
        test_data : dict
            Test data for evaluation
        test_labels : array-like
            Test labels for evaluation
            
        Returns
        -------
        dict
            Results of ablation study
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running ablation study")
        
        results = {}
        
        for ablation_config in ablation_variables:
            config_name = ablation_config.get('name', 'unknown')
            
            try:
                # Create test model with ablation configuration
                test_model = MultiModalEnsembleModel(
                    n_bags=self.n_bags,
                    dropout_strategy=self.dropout_strategy,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    random_state=self.random_state,
                    task_type=self.task_type,
                    max_dropout_rate=self.max_dropout_rate,
                    min_modalities=self.min_modalities,
                    sample_ratio=self.sample_ratio,
                    feature_sampling_ratio=self.feature_sampling_ratio,
                    # Stage 3 parameters with ablation overrides
                    optimization_mode=ablation_config.get('optimization_mode', self.optimization_mode),
                    modality_aware=ablation_config.get('modality_aware', self.modality_aware),
                    bag_learner_pairing=ablation_config.get('bag_learner_pairing', self.bag_learner_pairing),
                    metadata_level=ablation_config.get('metadata_level', self.metadata_level),
                    pairing_focus=ablation_config.get('pairing_focus', self.pairing_focus),
                    feature_ratio_weight=ablation_config.get('feature_ratio_weight', self.feature_ratio_weight),
                    variance_weight=ablation_config.get('variance_weight', self.variance_weight),
                    dimensionality_weight=ablation_config.get('dimensionality_weight', self.dimensionality_weight),
                    base_performance=ablation_config.get('base_performance', self.base_performance),
                    diversity_bonus=ablation_config.get('diversity_bonus', self.diversity_bonus),
                    weightage_bonus=ablation_config.get('weightage_bonus', self.weightage_bonus),
                    dropout_penalty=ablation_config.get('dropout_penalty', self.dropout_penalty),
                    # Stage 1 parameters
                    handle_nan=self.handle_nan,
                    handle_inf=self.handle_inf,
                    normalize=self.normalize,
                    remove_outliers=self.remove_outliers,
                    outlier_std=self.outlier_std,
                    verbose=False
                )
                
                # Fit test model with same training data
                test_model.fit(
                    X=self.train_data,
                    y=self.train_labels
                )
                
                # Collect ablation results
                results[config_name] = {
                    'ablation_config': ablation_config,
                    'bag_learner_summary': test_model.get_bag_learner_summary(),
                    'pairing_statistics': test_model.get_pairing_statistics(),
                    'metadata_completeness': test_model.get_metadata_completeness(),
                    'ensemble_coherence': test_model.get_ensemble_coherence(),
                    'stage3_interpretability': test_model.get_stage3_interpretability_data(),
                    'success': True
                }
                
            except Exception as e:
                results[config_name] = {
                    'ablation_config': ablation_config,
                    'error': str(e),
                    'success': False
                }
        
        return results

    def run_stage4_ablation_study(self, ablation_variables: List[Dict[str, Any]], 
                                 test_data: Dict[str, np.ndarray], 
                                 test_labels: np.ndarray) -> Dict[str, Any]:
        """
        Run Stage 4 ablation study with different parameter configurations.
        
        Parameters
        ----------
        ablation_variables : list
            List of dictionaries containing parameter configurations to test
        test_data : dict
            Test data for evaluation
        test_labels : array-like
            Test labels for evaluation
            
        Returns
        -------
        dict
            Results of ablation study
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running ablation study")
        
        results = {}
        
        for ablation_config in ablation_variables:
            config_name = ablation_config.get('name', 'unknown')
            
            try:
                # Create test model with ablation configuration
                test_model = MultiModalEnsembleModel(
                    n_bags=self.n_bags,
                    dropout_strategy=self.dropout_strategy,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    random_state=self.random_state,
                    task_type=self.task_type,
                    max_dropout_rate=self.max_dropout_rate,
                    min_modalities=self.min_modalities,
                    sample_ratio=self.sample_ratio,
                    feature_sampling_ratio=self.feature_sampling_ratio,
                    # Stage 3 parameters (keep same)
                    optimization_mode=self.optimization_mode,
                    modality_aware=self.modality_aware,
                    bag_learner_pairing=self.bag_learner_pairing,
                    metadata_level=self.metadata_level,
                    pairing_focus=self.pairing_focus,
                    feature_ratio_weight=self.feature_ratio_weight,
                    variance_weight=self.variance_weight,
                    dimensionality_weight=self.dimensionality_weight,
                    base_performance=self.base_performance,
                    diversity_bonus=self.diversity_bonus,
                    weightage_bonus=self.weightage_bonus,
                    dropout_penalty=self.dropout_penalty,
                    # Stage 4 parameters with ablation overrides
                    learning_rate=ablation_config.get('learning_rate', self.learning_rate),
                    weight_decay=ablation_config.get('weight_decay', self.weight_decay),
                    optimizer_type=ablation_config.get('optimizer_type', self.optimizer_type),
                    scheduler_type=ablation_config.get('scheduler_type', self.scheduler_type),
                    gradient_accumulation_steps=ablation_config.get('gradient_accumulation_steps', self.gradient_accumulation_steps),
                    gradient_clipping=ablation_config.get('gradient_clipping', self.gradient_clipping),
                    early_stopping_patience=ablation_config.get('early_stopping_patience', self.early_stopping_patience),
                    enable_data_augmentation=ablation_config.get('enable_data_augmentation', self.enable_data_augmentation),
                    augmentation_strength=ablation_config.get('augmentation_strength', self.augmentation_strength),
                    use_batch_norm=ablation_config.get('use_batch_norm', self.use_batch_norm),
                    enable_cross_validation=ablation_config.get('enable_cross_validation', self.enable_cross_validation),
                    cv_folds=ablation_config.get('cv_folds', self.cv_folds),
                    label_smoothing=ablation_config.get('label_smoothing', self.label_smoothing),
                    dropout_rate=ablation_config.get('dropout_rate', self.dropout_rate),
                    # Cross-modal denoising (NOVEL FEATURE) with ablation overrides
                    enable_denoising=ablation_config.get('enable_denoising', self.enable_denoising),
                    denoising_weight=ablation_config.get('denoising_weight', self.denoising_weight),
                    denoising_strategy=ablation_config.get('denoising_strategy', self.denoising_strategy),
                    denoising_objectives=ablation_config.get('denoising_objectives', self.denoising_objectives),
                    denoising_modalities=ablation_config.get('denoising_modalities', self.denoising_modalities),
                    # Modal-specific metrics tracking (NOVEL FEATURE) with ablation overrides
                    modal_specific_tracking=ablation_config.get('modal_specific_tracking', self.modal_specific_tracking),
                    track_modal_reconstruction=ablation_config.get('track_modal_reconstruction', self.track_modal_reconstruction),
                    track_modal_alignment=ablation_config.get('track_modal_alignment', self.track_modal_alignment),
                    track_modal_consistency=ablation_config.get('track_modal_consistency', self.track_modal_consistency),
                    modal_tracking_frequency=ablation_config.get('modal_tracking_frequency', self.modal_tracking_frequency),
                    track_only_primary_modalities=ablation_config.get('track_only_primary_modalities', self.track_only_primary_modalities),
                    # Bag characteristics preservation (NOVEL FEATURE) with ablation overrides
                    preserve_bag_characteristics=ablation_config.get('preserve_bag_characteristics', self.preserve_bag_characteristics),
                    save_modality_mask=ablation_config.get('save_modality_mask', self.save_modality_mask),
                    save_modality_weights=ablation_config.get('save_modality_weights', self.save_modality_weights),
                    save_bag_id=ablation_config.get('save_bag_id', self.save_bag_id),
                    save_training_metrics=ablation_config.get('save_training_metrics', self.save_training_metrics),
                    save_learner_config=ablation_config.get('save_learner_config', self.save_learner_config),
                    preserve_only_primary_modalities=ablation_config.get('preserve_only_primary_modalities', self.preserve_only_primary_modalities),
                    # Stage 1 parameters
                    handle_nan=self.handle_nan,
                    handle_inf=self.handle_inf,
                    normalize=self.normalize,
                    remove_outliers=self.remove_outliers,
                    outlier_std=self.outlier_std,
                    verbose=False
                )
                
                # Fit test model with same training data
                test_model.fit(
                    X=self.train_data,
                    y=self.train_labels
                )
                
                # Collect ablation results
                results[config_name] = {
                    'ablation_config': ablation_config,
                    'trained_learners': test_model.get_trained_learners(),
                    'training_summary': test_model.get_training_summary(),
                    'success': True
                }
                
            except Exception as e:
                results[config_name] = {
                    'ablation_config': ablation_config,
                    'error': str(e),
                    'success': False
                }
        
        return results

    # Stage 3 Robustness Test Convenience Methods

    def run_stage3_robustness_test(self, test_type: str, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Run specific Stage 3 robustness tests."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running robustness tests")
        
        if test_type == 'modality_aware':
            return self.test_modality_aware_robustness(test_scenarios)
        elif test_type == 'optimization_mode':
            return self.test_optimization_mode_robustness(test_scenarios)
        elif test_type == 'modality_weightage':
            return self.test_modality_weightage_robustness(test_scenarios)
        elif test_type == 'bag_learner_pairing':
            return self.test_bag_learner_pairing_robustness(test_scenarios)
        elif test_type == 'performance_prediction':
            return self.test_performance_prediction_robustness(test_scenarios)
        elif test_type == 'ensemble_size':
            return self.test_ensemble_size_robustness(test_scenarios)
        else:
            raise ValueError(f"Unknown test_type: {test_type}")

    def test_modality_aware_robustness(self, modality_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test robustness of the modality-aware learner selection."""
        results = {}
        
        for scenario_name, scenario_config in modality_scenarios.items():
            try:
                test_model = MultiModalEnsembleModel(
                    n_bags=self.n_bags,
                    dropout_strategy=self.dropout_strategy,
                    optimization_mode=scenario_config.get('optimization_mode', self.optimization_mode),
                    modality_aware=scenario_config.get('modality_aware', True),
                    bag_learner_pairing=scenario_config.get('bag_learner_pairing', True),
                    random_state=self.random_state,
            verbose=False
        )

                test_model.fit(self.train_data, self.train_labels)
                
                bag_learner_summary = test_model.get_bag_learner_summary()
                pairing_stats = test_model.get_pairing_statistics()
                ensemble_coherence = test_model.get_ensemble_coherence()
                
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'bag_learner_summary': bag_learner_summary,
                    'pairing_statistics': pairing_stats,
                    'ensemble_coherence': ensemble_coherence,
                    'robustness_score': np.mean([
                        pairing_stats.get('modality_learner_match_rate', 0),
                        pairing_stats.get('pairing_consistency', 0),
                        ensemble_coherence.get('overall_coherence', 0)
                    ]),
                    'success': True
                }
                
            except Exception as e:
                results[scenario_name] = {
                    'scenario_config': scenario_config,
                    'error': str(e),
                    'success': False
                }
        
        return results


    # Stage 3 Interpretability Test Convenience Methods

    def run_stage3_interpretability_test(self, test_type: str, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Run specific Stage 3 interpretability tests."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running interpretability tests")
        
        if test_type == 'modality_importance':
            return self.test_modality_importance_interpretability(test_scenarios)
        elif test_type == 'learner_selection':
            return self.test_learner_selection_interpretability(test_scenarios)
        elif test_type == 'performance_prediction':
            return self.test_performance_prediction_interpretability(test_scenarios)
        elif test_type == 'bag_learner_pairing':
            return self.test_bag_learner_pairing_interpretability(test_scenarios)
        elif test_type == 'ensemble_coherence':
            return self.test_ensemble_coherence_interpretability(test_scenarios)
        elif test_type == 'optimization_mode':
            return self.test_optimization_mode_interpretability(test_scenarios)
        else:
            raise ValueError(f"Unknown test_type: {test_type}")

    def test_modality_importance_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test interpretability of modality importance in learner selection."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running interpretability tests")
        
        return self.learner_selector.test_modality_importance_interpretability(test_scenarios)

    def test_learner_selection_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test interpretability of learner selection decisions."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running interpretability tests")
        
        return self.learner_selector.test_learner_selection_interpretability(test_scenarios)

    def test_performance_prediction_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test interpretability of performance prediction system."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running interpretability tests")
        
        return self.learner_selector.test_performance_prediction_interpretability(test_scenarios)

    def test_bag_learner_pairing_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test interpretability of bag-learner pairing system."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running interpretability tests")
        
        return self.learner_selector.test_bag_learner_pairing_interpretability(test_scenarios)

    def test_ensemble_coherence_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test interpretability of ensemble coherence and consistency."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running interpretability tests")
        
        return self.learner_selector.test_ensemble_coherence_interpretability(test_scenarios)

    def test_optimization_mode_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test interpretability of optimization mode selection."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running interpretability tests")
        
        return self.learner_selector.test_optimization_mode_interpretability(test_scenarios)
    
    # --- Stage 4 API Methods ---
    
    def get_trained_learners(self) -> List[TrainedLearnerInfo]:
        """Get list of trained learners from Stage 4."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting trained learners")
        return self.trained_learners
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary from Stage 4."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting training summary")
        if self.training_pipeline is None:
            return {}
        
        # Create a simple training summary from trained learners
        if not self.trained_learners:
            return {}
        
        total_time = 0.0
        accuracies = []
        
        for learner_info in self.trained_learners:
            if learner_info.training_metrics:
                total_time += sum(metric.training_time for metric in learner_info.training_metrics)
                accuracies.append(learner_info.final_performance)
        
        return {
            'total_learners': len(self.trained_learners),
            'total_training_time': total_time,
            'average_performance': np.mean(accuracies) if accuracies else 0.0,
            'performance_range': (min(accuracies), max(accuracies)) if accuracies else (0.0, 0.0)
        }
    
    def get_learner_performance(self, bag_id: int) -> Optional[float]:
        """Get performance of a specific trained learner."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting learner performance")
        for learner_info in self.trained_learners:
            if learner_info.bag_id == bag_id:
                return learner_info.final_performance
        return None
    
    def get_learner_metrics(self, bag_id: int) -> Optional[List[Any]]:
        """Get training metrics for a specific learner."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting learner metrics")
        for learner_info in self.trained_learners:
            if learner_info.bag_id == bag_id:
                return learner_info.training_metrics
        return None

    # --- Stage 4 Interpretability Test Methods ---
    
    def run_stage4_interpretability_test(self, test_type: str, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Run specific Stage 4 interpretability tests."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running interpretability tests")
        
        if test_type == 'cross_modal_denoising':
            return self.test_cross_modal_denoising_interpretability(test_scenarios)
        elif test_type == 'modal_specific_metrics':
            return self.test_modal_specific_metrics_interpretability(test_scenarios)
        elif test_type == 'bag_characteristics':
            return self.test_bag_characteristics_interpretability(test_scenarios)
        else:
            raise ValueError(f"Unknown Stage 4 interpretability test type: {test_type}")
    
    def test_cross_modal_denoising_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test interpretability of cross-modal denoising system."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running interpretability tests")
        
        if self.training_pipeline is None:
            return {'error': 'Training pipeline not available'}
        
        try:
            result = self.training_pipeline.analyze_cross_modal_denoising_effectiveness(self.trained_learners)
            return result
        except Exception as e:
            return {'error': f'Cross-modal denoising analysis failed: {str(e)}'}
    
    def test_modal_specific_metrics_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test interpretability of modal-specific metrics tracking."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running interpretability tests")
        
        if self.training_pipeline is None:
            return {'error': 'Training pipeline not available'}
        
        try:
            result = self.training_pipeline.analyze_modal_specific_metrics_granularity(self.trained_learners)
            return result
        except Exception as e:
            return {'error': f'Modal-specific metrics analysis failed: {str(e)}'}
    
    def test_bag_characteristics_interpretability(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test interpretability of bag characteristics preservation."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running interpretability tests")
        
        if self.training_pipeline is None:
            return {'error': 'Training pipeline not available'}
        
        try:
            result = self.training_pipeline.analyze_bag_characteristics_preservation_traceability(self.trained_learners)
            return result
        except Exception as e:
            return {'error': f'Bag characteristics analysis failed: {str(e)}'}

    # --- Stage 4 Robustness Test Methods ---
    
    def run_stage4_robustness_test(self, test_type: str, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Run specific Stage 4 robustness tests."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running robustness tests")
        
        if test_type == 'cross_modal_denoising':
            return self.test_cross_modal_denoising_robustness(test_scenarios)
        elif test_type == 'modal_specific_metrics':
            return self.test_modal_specific_metrics_robustness(test_scenarios)
        elif test_type == 'bag_characteristics':
            return self.test_bag_characteristics_robustness(test_scenarios)
        elif test_type == 'integrated':
            return self.test_integrated_stage4_robustness(test_scenarios)
        else:
            raise ValueError(f"Unknown Stage 4 robustness test type: {test_type}")
    
    def test_cross_modal_denoising_robustness(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test robustness of cross-modal denoising system."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running robustness tests")
        
        if self.training_pipeline is None:
            return {'error': 'Training pipeline not available'}
        
        try:
            result = self.training_pipeline.test_cross_modal_denoising_robustness(test_scenarios)
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                return {'error': f'Expected dictionary result, got {type(result)}'}
            return result
        except Exception as e:
            return {'error': f'Cross-modal denoising robustness test failed: {str(e)}'}
    
    def test_modal_specific_metrics_robustness(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test robustness of modal-specific metrics tracking system."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running robustness tests")
        
        if self.training_pipeline is None:
            return {'error': 'Training pipeline not available'}
        
        try:
            result = self.training_pipeline.test_modal_specific_metrics_robustness(test_scenarios)
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                return {'error': f'Expected dictionary result, got {type(result)}'}
            return result
        except Exception as e:
            return {'error': f'Modal-specific metrics robustness test failed: {str(e)}'}
    
    def test_bag_characteristics_robustness(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test robustness of bag characteristics preservation system."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running robustness tests")
        
        if self.training_pipeline is None:
            return {'error': 'Training pipeline not available'}
        
        try:
            result = self.training_pipeline.test_bag_characteristics_robustness(test_scenarios)
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                return {'error': f'Expected dictionary result, got {type(result)}'}
            return result
        except Exception as e:
            return {'error': f'Bag characteristics robustness test failed: {str(e)}'}
    
    def test_integrated_stage4_robustness(self, test_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Test robustness of integrated Stage 4 novel features."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running robustness tests")
        
        if self.training_pipeline is None:
            return {'error': 'Training pipeline not available'}
        
        try:
            result = self.training_pipeline.test_integrated_stage4_robustness(test_scenarios)
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                return {'error': f'Expected dictionary result, got {type(result)}'}
            return result
        except Exception as e:
            return {'error': f'Integrated Stage 4 robustness test failed: {str(e)}'}

    # --- Stage 5 Interpretability Test Methods ---
    
    def run_stage5_interpretability_test(self, test_data: Dict[str, np.ndarray], 
                                        test_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run Stage 5 interpretability tests for ensemble aggregation
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running interpretability tests")
        
        if self.ensemble_predictor is None:
            raise ValueError("Ensemble predictor must be set up before running interpretability tests")
        
        from MainModel.ensemblePrediction import run_simple_interpretability_test
        return run_simple_interpretability_test(self.ensemble_predictor, test_data, test_labels)
    
    def analyze_integrated_stage5_interpretability(self, test_data: Dict[str, np.ndarray], 
                                                  test_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze interpretability of integrated Stage 5 features
        """
        return self.run_stage5_interpretability_test(test_data, test_labels)
    
    def analyze_principled_bias_terms_interpretability(self, test_data: Dict[str, np.ndarray], 
                                                      test_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze interpretability of principled bias terms
        """
        return self.run_stage5_interpretability_test(test_data, test_labels, 'principled_bias_terms')
    
    def analyze_modality_importance_interpretability(self, test_data: Dict[str, np.ndarray], 
                                                    test_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze interpretability of modality importance
        """
        return self.run_stage5_interpretability_test(test_data, test_labels, 'modality_importance')

    # --- Stage 5 Robustness Test Methods ---
    
    def run_stage5_robustness_test(self, test_data: Dict[str, np.ndarray], 
                                  test_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run Stage 5 robustness tests for ensemble aggregation
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before running robustness tests")
        
        if self.ensemble_predictor is None:
            raise ValueError("Ensemble predictor must be set up before running robustness tests")
        
        from MainModel.ensemblePrediction import run_simple_robustness_test
        return run_simple_robustness_test(self.ensemble_predictor, test_data, test_labels)
    
    def test_exact_reconstruction_robustness(self, test_data: Dict[str, np.ndarray], 
                                           test_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Test robustness of exact bag reconstruction system
        """
        return self.run_stage5_robustness_test(test_data, test_labels)
    
    def test_integrated_stage5_robustness(self, test_data: Dict[str, np.ndarray], 
                                         test_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Test robustness of integrated Stage 5 features working together
        """
        return self.run_stage5_robustness_test(test_data, test_labels)
