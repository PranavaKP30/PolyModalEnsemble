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
    aggregation_strategy : str, default='weighted_vote'
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
                 aggregation_strategy='weighted_vote', uncertainty_method='entropy',
                 max_dropout_rate=0.5, min_modalities=1, sample_ratio=0.8,
                 diversity_target=0.7, feature_sampling=True, verbose=True):
        
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
        """Prepare data loader from input data."""
        # Create data loader
        loader = dataIntegration.GenericMultiModalDataLoader(validate_data=True)
        
        # Determine split sizes (80% train, 20% test)
        n_samples = len(next(iter(X.values())))
        n_train = int(0.8 * n_samples)
        n_test = n_samples - n_train
        
        # Add modalities
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
            else:
                data_type = 'tabular'
            
            loader.add_modality_split(
                name=modality_name,
                train_data=train_data,
                test_data=test_data,
                data_type=data_type
            )
        
        # Add labels
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
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
        
        # Load and integrate data
        if self.verbose:
            print("Loading and integrating data...")
        self._load_and_integrate_data(data_loader)
        
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
        # Get train/test splits
        self.train_data, self.train_labels = data_loader.get_split('train')
        self.test_data, self.test_labels = data_loader.get_split('test')
        
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
            'modality_names': [mc.name for mc in self.modality_configs_]
        }
        
        # Set n_features
        self.n_features_ = list(self.train_data.values())[0].shape[1]
    
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
            n_classes=self.n_classes_,
            random_state=self.random_state
        )
        
        # Generate learners
        selected_learners = self.selector.generate_learners(instantiate=True)
        
        # Prepare training data
        learner_metadata = []
        bag_data = {}
        bag_labels = {}
        
        for i, (learner_id, learner) in enumerate(selected_learners.items()):
            bag = self.bags[i]
            bag_data[learner_id] = {
                mod: self.train_data[mod][bag.data_indices] 
                for mod in self.train_data
            }
            bag_labels[learner_id] = self.train_labels[bag.data_indices]
            
            learner_metadata.append({
                'learner_id': learner_id,
                'metrics': {},
                'modalities': [mc.name for mc in self.modality_configs_],
                'pattern': '+'.join([mc.name for mc in self.modality_configs_])
            })
        
        # Create training pipeline
        self.pipeline = trainingPipeline.create_training_pipeline(
            task_type=self.task_type_,
            epochs=self.epochs,
            batch_size=self.batch_size,
            enable_denoising=self.enable_denoising
        )
        
        # Train ensemble
        self.trained_learners_, training_metrics = self.pipeline.train_ensemble(
            selected_learners, 
            [None] * len(selected_learners), 
            bag_data, 
            bag_labels=bag_labels
        )
        
        # Update learner metadata with training metrics
        for i, (lid, metrics) in enumerate(training_metrics.items()):
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
            uncertainty_method=self.uncertainty_method
        )
        
        # Add trained learners to ensemble
        for learner, meta in zip(self.trained_learners_, self.learner_metadata_):
            self.ensemble_.add_trained_learner(
                learner, 
                meta.get('metrics', {}), 
                meta.get('modalities', []), 
                meta.get('pattern', '')
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
        
        # Convert to numpy arrays
        X = {k: np.asarray(v) for k, v in X.items()}
        
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
        
        # Convert to numpy arrays
        X = {k: np.asarray(v) for k, v in X.items()}
        
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
        
        # Convert to numpy arrays
        X = {k: np.asarray(v) for k, v in X.items()}
        y = np.asarray(y)
        
        # Get predictions
        predictions = self.predict(X)
        
        # Create evaluator
        evaluator = performanceMetrics.PerformanceEvaluator(task_type=self.task_type_)
        
        # Evaluate model
        report = evaluator.evaluate_model(
            lambda X_dict: predictions, 
            X, 
            y, 
            model_name="MultiModalEnsemble"
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
                avg_importance[modality] = np.mean(scores_list, axis=0)
        
        return avg_importance
    
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
            'verbose': self.verbose
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
