"""
PolyModalEnsemble - Main Model API

Unified API for the multimodal ensemble architecture that can process any combination 
of modalities across different datasets (EuroSAT, OASIS, MUTLA).

This API provides convenient access to all stages of the multimodal ensemble pipeline:
- Stage 1: DataIntegration - Load and preprocess multimodal data
- Stage 2: modalityDropoutBagger - Modality-aware ensemble bagging
- Stage 3: baseLearnerSelector - Intelligent learner selection
- Stage 4: trainingPipeline - Unified training pipeline
- Stage 5: ensemblePrediction - Ensemble prediction and inference
"""

import logging
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import numpy as np
import torch

# Import Stage 1: DataIntegration
try:
    from .DataIntegration import SimpleDataLoader, load_eurosat_data, load_oasis_data, load_mutla_data, load_custom_data
except ImportError:
    from DataIntegration import SimpleDataLoader, load_eurosat_data, load_oasis_data, load_mutla_data, load_custom_data

# Import Stage 2: BagGeneration
try:
    from .BagGeneration import BagGeneration
except ImportError:
    from BagGeneration import BagGeneration

# Import Stage 3: BaseLearnerSelector
try:
    from .BagLearnerParing import BaseLearnerSelector, LearnerConfig
except ImportError:
    from BagLearnerParing import BaseLearnerSelector, LearnerConfig

# Stage 3 is now imported above

# Import Stage 4: trainingPipeline (TODO: Uncomment when implemented)
# try:
#     from .trainingPipeline import TrainingPipeline
# except ImportError:
#     from trainingPipeline import TrainingPipeline

# Import Stage 5: ensemblePrediction (TODO: Uncomment when implemented)
# try:
#     from .ensemblePrediction import EnsemblePrediction
# except ImportError:
#     from ensemblePrediction import EnsemblePrediction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelAPI:
    """
    Main API for the PolyModal Ensemble architecture.
    
    Provides convenient access to all stages of the multimodal ensemble pipeline
    with standardized data flow from Stage 1 through Stage 5.
    """
    
    def __init__(self, device: str = 'cpu', cache_dir: Optional[str] = None, 
                 lazy_loading: bool = False, chunk_size: int = 1000,
                 handle_missing_modalities: bool = True, missing_modality_strategy: str = "zero_fill",
                 handle_class_imbalance: bool = True, class_imbalance_strategy: str = "report",
                 fast_mode: bool = True, max_samples: int = 1000):
        """
        Initialize the PolyModal Ensemble API.
        
        Parameters
        ----------
        device : str
            Device for tensor operations ('cpu' or 'cuda')
        cache_dir : str, optional
            Directory for caching processed data
        lazy_loading : bool
            Enable lazy loading for memory efficiency with large datasets
        chunk_size : int
            Size of chunks for lazy loading (number of samples per chunk)
        handle_missing_modalities : bool
            Whether to handle samples with missing modalities
        missing_modality_strategy : str
            Strategy for handling missing modalities ('zero_fill', 'skip', 'interpolate')
        handle_class_imbalance : bool
            Whether to handle class imbalance
        class_imbalance_strategy : str
            Strategy for handling class imbalance ('report', 'resample', 'weight')
        fast_mode : bool
            Enable fast loading for large datasets
        max_samples : int
            Maximum samples to load in fast mode
        """
        self.device = device
        self.cache_dir = cache_dir
        self.lazy_loading = lazy_loading
        self.chunk_size = chunk_size
        
        # Initialize Stage 1: DataIntegration
        self.data_loader = SimpleDataLoader(
            cache_dir=cache_dir,
            device=device,
            lazy_loading=lazy_loading,
            chunk_size=chunk_size,
            handle_missing_modalities=handle_missing_modalities,
            missing_modality_strategy=missing_modality_strategy,
            handle_class_imbalance=handle_class_imbalance,
            class_imbalance_strategy=class_imbalance_strategy,
            fast_mode=fast_mode,
            max_samples=max_samples
        )
        
        # Initialize Stage 2: BagGeneration
        self.bag_generator = None
        
        # Initialize Stage 3: BaseLearnerSelector
        self.learner_selector = None
        
        # Stage 3 is now initialized above
        
        # Initialize Stage 4: trainingPipeline (TODO: Uncomment when implemented)
        # self.training_pipeline = None
        
        # Initialize Stage 5: ensemblePrediction (TODO: Uncomment when implemented)
        # self.ensemble_predictor = None
        
        # Data storage for pipeline flow
        self.current_data = None
        self.current_labels = None
        self.current_splits = None
        
        logger.info(f"Initialized ModelAPI with device: {device}")
    
    # ============================================================================
    # STAGE 1: DATA INTEGRATION API
    # ============================================================================
    
    def load_eurosat_data(self, data_dir: str = "ProcessedData/EuroSAT", 
                         normalize: bool = True, test_size: float = 0.2,
                         target_size: Tuple[int, int] = (224, 224),
                         channels_first: bool = True, random_state: int = 42, **kwargs) -> 'ModelAPI':
        """
        Load EuroSAT dataset (visual + spectral data) for unified model.
        
        Parameters
        ----------
        data_dir : str
            Path to EuroSAT processed data directory
        normalize : bool
            Whether to normalize the data
        test_size : float
            Fraction of data to use for testing
        target_size : tuple
            Target size for image resizing (height, width)
        channels_first : bool
            If True, return images in PyTorch NCHW format
        **kwargs
            Additional parameters passed to load_multimodal_data including:
            - fast_mode: Enable fast loading for large datasets (default: True)
            - max_samples: Maximum samples to load in fast mode (default: 1000)
            - lazy_loading: Enable lazy loading (default: True)
            - handle_missing_modalities: Handle missing modality data (default: True)
            - handle_class_imbalance: Handle class imbalance (default: True)
            
        Returns
        -------
        ModelAPI
            Self for method chaining
        """
        # Parameter validation
        if not Path(data_dir).exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if 'max_samples' in kwargs and kwargs['max_samples'] is not None and kwargs['max_samples'] <= 0:
            raise ValueError("max_samples must be positive")
        
        logger.info("Loading EuroSAT dataset...")
        
        # Use the existing data_loader instead of creating new one
        modality_types = {
            "visual_rgb": "image",
            "near_infrared": "spectral", 
            "red_edge": "spectral",
            "short_wave_infrared": "spectral",
            "atmospheric": "spectral"
        }
        
        modality_files = {
            "visual_rgb": f"{data_dir}/visual_rgb",
            "near_infrared": f"{data_dir}/near_infrared", 
            "red_edge": f"{data_dir}/red_edge",
            "short_wave_infrared": f"{data_dir}/short_wave_infrared",
            "atmospheric": f"{data_dir}/atmospheric"
        }
        
        # Extract SimpleDataLoader parameters from kwargs
        loader_kwargs = {}
        data_kwargs = {}
        
        for key, value in kwargs.items():
            if key in ['fast_mode', 'max_samples', 'lazy_loading', 'chunk_size', 
                      'handle_missing_modalities', 'missing_modality_strategy',
                      'handle_class_imbalance', 'class_imbalance_strategy', 'cache_dir']:
                loader_kwargs[key] = value
            else:
                data_kwargs[key] = value
        
        # Update existing data_loader attributes instead of creating new instance
        if loader_kwargs:
            for key, value in loader_kwargs.items():
                if hasattr(self.data_loader, key):
                    setattr(self.data_loader, key, value)
                else:
                    logger.warning(f"Parameter {key} not found in SimpleDataLoader, ignoring")
        
        self.data_loader.load_multimodal_data(
            label_file=f"{data_dir}/labels.csv",
            modality_files=modality_files,
            modality_types=modality_types,
            normalize=normalize,
            test_size=test_size,
            target_size=target_size,
            channels_first=channels_first,
            random_state=random_state,
            **data_kwargs
        )
        
        # Store data for pipeline flow
        self._store_current_data()
        
        logger.info("EuroSAT data loaded successfully")
        return self
    
    def load_oasis_data(self, data_dir: str = "ProcessedData/OASIS",
                       normalize: bool = True, test_size: float = 0.2, 
                       random_state: int = 42, **kwargs) -> 'ModelAPI':
        """
        Load OASIS dataset (tabular data) for unified model.
        
        Parameters
        ----------
        data_dir : str
            Path to OASIS processed data directory
        normalize : bool
            Whether to normalize the data
        test_size : float
            Fraction of data to use for testing
        **kwargs
            Additional parameters passed to load_multimodal_data including:
            - fast_mode: Enable fast loading for large datasets (default: True)
            - max_samples: Maximum samples to load in fast mode (default: 1000)
            - lazy_loading: Enable lazy loading (default: True)
            - handle_missing_modalities: Handle missing modality data (default: True)
            - handle_class_imbalance: Handle class imbalance (default: True)
            
        Returns
        -------
        ModelAPI
            Self for method chaining
        """
        # Parameter validation
        if not Path(data_dir).exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        
        logger.info("Loading OASIS dataset...")
        
        # Use the existing data_loader instead of creating new one
        modality_types = {
            "tabular_features": "tabular"
        }
        
        modality_files = {
            "tabular_features": f"{data_dir}/tabular_features.csv"
        }
        
        # Extract SimpleDataLoader parameters from kwargs
        loader_kwargs = {}
        data_kwargs = {}
        
        for key, value in kwargs.items():
            if key in ['fast_mode', 'max_samples', 'lazy_loading', 'chunk_size', 
                      'handle_missing_modalities', 'missing_modality_strategy',
                      'handle_class_imbalance', 'class_imbalance_strategy', 'cache_dir']:
                loader_kwargs[key] = value
            else:
                data_kwargs[key] = value
        
        # Update existing data_loader attributes instead of creating new instance
        if loader_kwargs:
            for key, value in loader_kwargs.items():
                if hasattr(self.data_loader, key):
                    setattr(self.data_loader, key, value)
                else:
                    logger.warning(f"Parameter {key} not found in SimpleDataLoader, ignoring")
        
        self.data_loader.load_multimodal_data(
            label_file=f"{data_dir}/labels.csv",
            modality_files=modality_files,
            modality_types=modality_types,
            normalize=normalize,
            test_size=test_size,
            random_state=random_state,
            **data_kwargs
        )
        
        # Store data for pipeline flow
        self._store_current_data()
        
        logger.info("OASIS data loaded successfully")
        return self
    
    def load_mutla_data(self, data_dir: str = "Data/MUTLA",
                       normalize: bool = True, test_size: float = 0.2, 
                       random_state: int = 42, **kwargs) -> 'ModelAPI':
        """
        Load MUTLA dataset (tabular + time-series + visual data) for unified model.
        
        Parameters
        ----------
        data_dir : str
            Path to MUTLA raw data directory
        normalize : bool
            Whether to normalize the data
        test_size : float
            Fraction of data to use for testing
        **kwargs
            Additional parameters passed to load_multimodal_data including:
            - fast_mode: Enable fast loading for large datasets (default: True)
            - max_samples: Maximum samples to load in fast mode (default: 1000)
            - lazy_loading: Enable lazy loading (default: True)
            - handle_missing_modalities: Handle missing modality data (default: True)
            - handle_class_imbalance: Handle class imbalance (default: True)
            
        Returns
        -------
        ModelAPI
            Self for method chaining
        """
        # Parameter validation
        if not Path(data_dir).exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        
        logger.info("Loading MUTLA dataset...")
        
        # Use the existing data_loader instead of creating new one
        modality_types = {
            "tabular": "tabular",            # CSV files with user interaction data
            "timeseries": "timeseries",      # LOG files with EEG/attention time-series
            "visual": "visual"               # NPY files with webcam tracking data
        }
        
        modality_files = {
            "tabular": f"{data_dir}/User records/math_record_cleaned.csv",  # Use one of the tabular files as labels
            "timeseries": f"{data_dir}/Brainwave",
            "visual": f"{data_dir}/Webcam"
        }
        
        # Extract SimpleDataLoader parameters from kwargs
        loader_kwargs = {}
        data_kwargs = {}
        
        for key, value in kwargs.items():
            if key in ['fast_mode', 'max_samples', 'lazy_loading', 'chunk_size', 
                      'handle_missing_modalities', 'missing_modality_strategy',
                      'handle_class_imbalance', 'class_imbalance_strategy', 'cache_dir']:
                loader_kwargs[key] = value
            else:
                data_kwargs[key] = value
        
        # Update existing data_loader attributes instead of creating new instance
        if loader_kwargs:
            for key, value in loader_kwargs.items():
                if hasattr(self.data_loader, key):
                    setattr(self.data_loader, key, value)
                else:
                    logger.warning(f"Parameter {key} not found in SimpleDataLoader, ignoring")
        
        self.data_loader.load_multimodal_data(
            label_file=f"{data_dir}/User records/math_record_cleaned.csv",  # Use one of the tabular files as labels
            modality_files=modality_files,
            modality_types=modality_types,
            normalize=normalize,
            test_size=test_size,
            random_state=random_state,
            **data_kwargs
        )
        
        # Store data for pipeline flow
        self._store_current_data()
        
        logger.info("MUTLA data loaded successfully")
        return self
    
    def load_custom_data(self, label_file: str, modality_files: Dict[str, str],
                        modality_types: Dict[str, str], **kwargs) -> 'ModelAPI':
        """
        Load custom multimodal data for unified model.
        
        Parameters
        ----------
        label_file : str
            Path to labels file
        modality_files : dict
            Dictionary mapping modality names to file paths
        modality_types : dict
            Dictionary mapping modality names to types
        **kwargs
            Additional parameters passed to load_multimodal_data including:
            - fast_mode: Enable fast loading for large datasets (default: True)
            - max_samples: Maximum samples to load in fast mode (default: 1000)
            - lazy_loading: Enable lazy loading (default: True)
            - handle_missing_modalities: Handle missing modality data (default: True)
            - handle_class_imbalance: Handle class imbalance (default: True)
            
        Returns
        -------
        ModelAPI
            Self for method chaining
        """
        # Parameter validation
        if not Path(label_file).exists():
            raise ValueError(f"Label file does not exist: {label_file}")
        if not modality_files:
            raise ValueError("modality_files cannot be empty")
        if not modality_types:
            raise ValueError("modality_types cannot be empty")
        if set(modality_files.keys()) != set(modality_types.keys()):
            raise ValueError("modality_files and modality_types must have the same keys")
        
        # Validate that all modality files exist
        for modality_name, file_path in modality_files.items():
            if not Path(file_path).exists():
                raise ValueError(f"Modality file does not exist: {file_path}")
        
        logger.info("Loading custom multimodal data...")
        
        # Extract SimpleDataLoader parameters from kwargs
        loader_kwargs = {}
        data_kwargs = {}
        
        for key, value in kwargs.items():
            if key in ['fast_mode', 'max_samples', 'lazy_loading', 'chunk_size', 
                      'handle_missing_modalities', 'missing_modality_strategy',
                      'handle_class_imbalance', 'class_imbalance_strategy', 'cache_dir']:
                loader_kwargs[key] = value
            else:
                data_kwargs[key] = value
        
        # Update existing data_loader attributes instead of creating new instance
        if loader_kwargs:
            for key, value in loader_kwargs.items():
                if hasattr(self.data_loader, key):
                    setattr(self.data_loader, key, value)
                else:
                    logger.warning(f"Parameter {key} not found in SimpleDataLoader, ignoring")
        
        # Use the existing data_loader instead of creating new one
        self.data_loader.load_multimodal_data(
            label_file=label_file,
            modality_files=modality_files,
            modality_types=modality_types,
            **data_kwargs
        )
        
        # Store data for pipeline flow
        self._store_current_data()
        
        logger.info("Custom data loaded successfully")
        return self
    
    def _store_current_data(self) -> None:
        """Store current data for pipeline flow to subsequent stages."""
        try:
            if hasattr(self.data_loader, 'data') and self.data_loader.data:
                self.current_data = self.data_loader.data
                logger.debug("Data stored for pipeline flow")
            else:
                logger.warning("No data available in data_loader to store")
        except Exception as e:
            logger.warning(f"Could not store data for pipeline: {e}")
    
    # DATA ACCESS METHODS
    def get_train_data(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Get training data and labels.
        
        Returns
        -------
        tuple
            (train_data_dict, train_labels) ready for Stage 2
        """
        if self.data_loader is None:
            raise ValueError("No data loaded. Call one of the load_* methods first.")
        
        if not hasattr(self.data_loader, 'data') or not self.data_loader.data:
            raise ValueError("No data available in data_loader. Data may not have been loaded successfully.")
        
        try:
            # Use the DataIntegration get_train_data method
            train_data, train_labels = self.data_loader.get_train_data()
            logger.debug(f"Successfully retrieved training data: {len(train_labels)} samples, {len(train_data)} modalities")
            return train_data, train_labels
        except Exception as e:
            logger.error(f"Failed to retrieve training data: {e}")
            raise ValueError(f"Failed to retrieve training data: {e}") from e
    
    def get_test_data(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Get test data and labels.
        
        Returns
        -------
        tuple
            (test_data_dict, test_labels) ready for evaluation
        """
        if self.data_loader is None:
            raise ValueError("No data loaded. Call one of the load_* methods first.")
        
        if not hasattr(self.data_loader, 'data') or not self.data_loader.data:
            raise ValueError("No data available in data_loader. Data may not have been loaded successfully.")
        
        try:
            # Use the DataIntegration get_test_data method
            test_data, test_labels = self.data_loader.get_test_data()
            logger.debug(f"Successfully retrieved test data: {len(test_labels)} samples, {len(test_data)} modalities")
            return test_data, test_labels
        except Exception as e:
            logger.error(f"Failed to retrieve test data: {e}")
            raise ValueError(f"Failed to retrieve test data: {e}") from e
    
    def get_tensors(self, device: Optional[str] = None) -> Dict[str, Any]:
        """
        Get data as PyTorch tensors ready for model training.
        
        Parameters
        ----------
        device : str, optional
            Device to place tensors on. If None, uses API device.
            
        Returns
        -------
        dict
            Dictionary containing train/test data as PyTorch tensors
        """
        if self.data_loader is None:
            raise ValueError("No data loaded. Call one of the load_* methods first.")
        
        if not hasattr(self.data_loader, 'data') or not self.data_loader.data:
            raise ValueError("No data available in data_loader. Data may not have been loaded successfully.")
        
        try:
            target_device = device or self.device
            tensors = self.data_loader.get_tensors(target_device)
            
            # Validate tensor conversion
            if not isinstance(tensors, dict):
                raise ValueError("Tensor conversion failed: expected dictionary, got {type(tensors)}")
            
            if 'train_data' not in tensors or 'test_data' not in tensors:
                raise ValueError("Tensor conversion failed: missing train_data or test_data keys")
            
            logger.debug(f"Successfully converted data to tensors on device: {target_device}")
            return tensors
            
        except Exception as e:
            logger.error(f"Tensor conversion failed: {e}")
            raise ValueError(f"Failed to convert data to tensors: {e}") from e
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded data.
        
        Returns
        -------
        dict
            Dictionary containing data information and statistics
        """
        if self.data_loader is None:
            raise ValueError("No data loaded. Call one of the load_* methods first.")
        
        # Use existing data from data_loader instead of recreating
        if 'train_data' in self.data_loader.data and 'train_labels' in self.data_loader.data:
            train_data = self.data_loader.data['train_data']
            train_labels = self.data_loader.data['train_labels']
            test_data = self.data_loader.data.get('test_data', {})
            test_labels = self.data_loader.data.get('test_labels', np.array([]))
        else:
            raise ValueError("No data available in data_loader")
        
        info = {
            'n_train': len(train_labels),
            'n_test': len(test_labels),
            'modalities': list(train_data.keys()),
            'modality_shapes': {name: data.shape for name, data in train_data.items()},
            'device': self.device,
            'lazy_loading': getattr(self.data_loader, 'lazy_loading', False)
        }
        
        return info
    
    # ============================================================================
    # STAGE 2: BagGeneration API
    # ============================================================================
    
    def generate_bags(self, n_bags: int = 10, dropout_strategy: str = 'adaptive',
                     max_dropout_rate: float = 0.7, min_modalities: int = 1,
                     sample_ratio: float = 0.8, random_state: int = 42, **kwargs) -> 'ModelAPI':
        """
        Generate ensemble bags using Stage 2: BagGeneration.
        
        Parameters
        ----------
        n_bags : int
            Number of bags to generate
        dropout_strategy : str
            Dropout strategy ('adaptive', 'uniform', 'fixed')
        max_dropout_rate : float
            Maximum dropout rate for adaptive strategy
        min_modalities : int
            Minimum number of modalities per bag
        sample_ratio : float
            Ratio of samples to use in each bag
        random_state : int
            Random seed for reproducibility
        **kwargs
            Additional parameters for BagGeneration
            
        Returns
        -------
        ModelAPI
            Self for method chaining
        """
        # Parameter validation
        if n_bags <= 0:
            raise ValueError("n_bags must be positive")
        if dropout_strategy not in ['adaptive', 'random', 'fixed']:
            raise ValueError("dropout_strategy must be 'adaptive', 'random', or 'fixed'")
        if not 0 < max_dropout_rate <= 1:
            raise ValueError("max_dropout_rate must be between 0 and 1")
        if min_modalities < 1:
            raise ValueError("min_modalities must be at least 1")
        if not 0 < sample_ratio <= 1:
            raise ValueError("sample_ratio must be between 0 and 1")
        
        logger.info(f"Generating {n_bags} bags with {dropout_strategy} strategy...")
        
        # Get training data
        train_data, train_labels = self.get_train_data()
        
        # Initialize BagGeneration
        self.bag_generator = BagGeneration(
            train_data=train_data,
            train_labels=train_labels,
            n_bags=n_bags,
            dropout_strategy=dropout_strategy,
            max_dropout_rate=max_dropout_rate,
            min_modalities=min_modalities,
            sample_ratio=sample_ratio,
            random_state=random_state,
            **kwargs
        )
        
        # Generate bags
        self.bags = self.bag_generator.generate_bags()
        
        logger.info(f"Successfully generated {len(self.bags)} bags")
        return self
    
    def get_bags(self) -> list:
        """
        Get generated bags.
        
        Returns
        -------
        list
            List of BagConfig objects
        """
        if self.bag_generator is None:
            raise ValueError("No bags generated. Call generate_bags() first.")
        return self.bags
    
    def get_bag_info(self) -> Dict[str, Any]:
        """
        Get information about generated bags.
        
        Returns
        -------
        dict
            Dictionary containing bag information and statistics
        """
        if self.bag_generator is None:
            raise ValueError("No bags generated. Call generate_bags() first.")
        
        info = {
            'n_bags': len(self.bags),
            'dropout_strategy': self.bag_generator.dropout_strategy,
            'max_dropout_rate': self.bag_generator.max_dropout_rate,
            'min_modalities': self.bag_generator.min_modalities,
            'sample_ratio': self.bag_generator.sample_ratio,
            'modality_names': self.bag_generator.modality_names,
            'n_modalities': self.bag_generator.n_modalities
        }
        
        # Add bag-specific statistics
        if self.bags:
            bag_stats = {
                'avg_modalities_per_bag': np.mean([len([m for m in bag.modality_mask.values() if m]) for bag in self.bags]),
                'min_modalities_per_bag': min([len([m for m in bag.modality_mask.values() if m]) for bag in self.bags]),
                'max_modalities_per_bag': max([len([m for m in bag.modality_mask.values() if m]) for bag in self.bags]),
                'avg_samples_per_bag': np.mean([len(bag.data_indices) for bag in self.bags])
            }
            info.update(bag_stats)
        
        return info
    
    # ============================================================================
    # STAGE 3 BagLearnerParing API
    # ============================================================================
    
    def select_learners(self, configuration_method: str = "optimal", add_configuration_diversity: bool = False,
                       validation_folds: int = 5, early_stopping_patience: int = 10,
                       transfer_learning: bool = True, random_state: Optional[int] = 42) -> 'ModelAPI':
        """
        Stage 3: Select optimal base learners for each ensemble bag.
        
        Parameters
        ----------
        configuration_method : str, optional
            Hyperparameter configuration strategy, by default "optimal"
        add_configuration_diversity : bool, optional
            Add small random variations for ensemble diversity, by default False
        validation_folds : int, optional
            Cross-validation folds for evaluation, by default 5
        early_stopping_patience : int, optional
            Early stopping patience for deep learning models, by default 10
        transfer_learning : bool, optional
            Enable transfer learning for visual learners, by default True
        random_state : Optional[int], optional
            Random seed for reproducibility, by default 42
            
        Returns
        -------
        ModelAPI
            Self with selected learners ready for Stage 4
        """
        if self.bag_generator is None or not hasattr(self, 'bags') or not self.bags:
            raise ValueError("No bags available. Call generate_bags() first.")
        
        # Parameter validation
        if configuration_method not in ['optimal', 'random', 'grid']:
            raise ValueError("configuration_method must be 'optimal', 'random', or 'grid'")
        if validation_folds < 2:
            raise ValueError("validation_folds must be at least 2")
        if early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1")
        
        logger.info("Stage 3: Starting base learner selection...")
        
        # Initialize BaseLearnerSelector
        self.learner_selector = BaseLearnerSelector(
            configuration_method=configuration_method,
            add_configuration_diversity=add_configuration_diversity,
            validation_folds=validation_folds,
            early_stopping_patience=early_stopping_patience,
            transfer_learning=transfer_learning,
            random_state=random_state
        )
        
        # Retrieve bags from Stage 2
        train_data = self.current_data['train_data']
        train_labels = self.current_data['train_labels']
        bags_data = self.learner_selector.retrieve_bags(
            self.bags, 
            train_data, 
            train_labels
        )
        
        # Analyze bags for learner selection
        bag_analyses = self.learner_selector.analyze_bags()
        
        # Select optimal learners
        learner_configs = self.learner_selector.select_learners()
        
        # Configure hyperparameters
        configured_configs = self.learner_selector.configure_hyperparameters()
        
        # Store learner configurations
        output_dir = self.learner_selector.store_learners("Model/learner_configs")
        
        logger.info(f"Stage 3: Selected and configured {len(configured_configs)} learners")
        logger.info(f"Learner configurations stored in: {output_dir}")
        
        return self
    
    def get_learner_configs(self) -> List[LearnerConfig]:
        """
        Get the selected learner configurations from Stage 3.
        
        Returns
        -------
        List[LearnerConfig]
            List of optimized learner configurations
        """
        if self.learner_selector is None:
            raise ValueError("No learners selected. Call select_learners() first.")
        
        return self.learner_selector.learner_configs
    
    def get_bag_analyses(self) -> List[Dict[str, Any]]:
        """
        Get the bag analysis results from Stage 3.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of bag analysis results
        """
        if self.learner_selector is None:
            raise ValueError("No learners selected. Call select_learners() first.")
        
        return self.learner_selector.bag_analyses    

    def get_learner_info(self) -> Dict[str, Any]:
        """
        Get information about selected learners.
        
        Returns
        -------
        dict
            Dictionary containing learner information and statistics
        """
        if self.learner_selector is None:
            raise ValueError("No learners selected. Call select_learners() first.")
        
        info = {
            'n_learners': len(self.learner_selector.learner_configs),
            'configuration_method': self.learner_selector.configuration_method,
            'add_configuration_diversity': self.learner_selector.add_configuration_diversity,
            'validation_folds': self.learner_selector.validation_folds,
            'early_stopping_patience': self.learner_selector.early_stopping_patience,
            'transfer_learning': self.learner_selector.transfer_learning
        }
        
        # Add learner-specific statistics
        if self.learner_selector.learner_configs:
            learner_types = [config.learner_type for config in self.learner_selector.learner_configs]
            learner_stats = {
                'unique_learner_types': list(set(learner_types)),
                'learner_type_counts': {learner_type: learner_types.count(learner_type) for learner_type in set(learner_types)},
                'avg_optimization_time': np.mean([getattr(config, 'optimization_time', 0) for config in self.learner_selector.learner_configs]),
                'has_performance_metrics': any(hasattr(config, 'performance_metrics') and config.performance_metrics for config in self.learner_selector.learner_configs)
            }
            info.update(learner_stats)
        
        return info
    
    def get_learner_summary(self) -> Dict[str, Any]:
        """
        Get a summary of learner configurations and their performance.
        
        Returns
        -------
        dict
            Summary of learner configurations with performance metrics
        """
        if self.learner_selector is None:
            raise ValueError("No learners selected. Call select_learners() first.")
        
        summary = {
            'total_learners': len(self.learner_selector.learner_configs),
            'learner_breakdown': {},
            'performance_summary': {},
            'modality_coverage': {}
        }
        
        # Analyze learner types and performance
        for config in self.learner_selector.learner_configs:
            learner_type = config.learner_type
            
            # Count learner types
            if learner_type not in summary['learner_breakdown']:
                summary['learner_breakdown'][learner_type] = 0
            summary['learner_breakdown'][learner_type] += 1
            
            # Collect performance metrics
            if hasattr(config, 'performance_metrics') and config.performance_metrics:
                for metric, value in config.performance_metrics.items():
                    if metric not in summary['performance_summary']:
                        summary['performance_summary'][metric] = []
                    summary['performance_summary'][metric].append(value)
            
            # Analyze modality coverage
            if hasattr(config, 'modality_info') and config.modality_info:
                modality_types = config.modality_info.get('modality_types', [])
                for modality in modality_types:
                    if modality not in summary['modality_coverage']:
                        summary['modality_coverage'][modality] = 0
                    summary['modality_coverage'][modality] += 1
        
        # Calculate average performance metrics
        for metric, values in summary['performance_summary'].items():
            if values:
                summary['performance_summary'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary
    
    def export_learner_configs(self, output_path: str = "learner_configs.json") -> str:
        """
        Export learner configurations to a JSON file.
        
        Parameters
        ----------
        output_path : str
            Path to save the JSON file
            
        Returns
        -------
        str
            Path to the saved file
        """
        if self.learner_selector is None:
            raise ValueError("No learners selected. Call select_learners() first.")
        
        import json
        
        # Convert learner configs to serializable format
        configs_data = []
        for config in self.learner_selector.learner_configs:
            config_dict = {
                'bag_id': config.bag_id,
                'learner_type': config.learner_type,
                'architecture': config.architecture,
                'hyperparameters': config.hyperparameters,
                'performance_metrics': config.performance_metrics,
                'modality_info': config.modality_info,
                'creation_timestamp': config.creation_timestamp
            }
            configs_data.append(config_dict)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(configs_data, f, indent=2, default=str)
        
        logger.info(f"Learner configurations exported to: {output_path}")
        return output_path
 
    # ============================================================================
    # STAGE 4: TRAINING PIPELINE API
    # ============================================================================
    
    # to update with actual methods

 
    # ============================================================================
    # STAGE 5: ENSEMBLE PREDICTION API
    # ============================================================================
    
    #to update with actual methods
    
    # ============================================================================
    # PIPELINE FLOW METHODS
    # ============================================================================
    
    def prepare_for_stage2(self) -> Dict[str, Any]:
        """
        Prepare data for Stage 2: BagGeneration.
        
        Returns
        -------
        dict
            Data and metadata ready for Stage 2 processing
        """
        if self.data_loader is None:
            raise ValueError("No data loaded. Call one of the load_* methods first.")
        
        # Use existing data directly instead of recreating
        if 'train_data' in self.data_loader.data and 'train_labels' in self.data_loader.data:
            train_data = self.data_loader.data['train_data']
            train_labels = self.data_loader.data['train_labels']
            test_data = self.data_loader.data.get('test_data', {})
            test_labels = self.data_loader.data.get('test_labels', np.array([]))
        else:
            raise ValueError("No data available in data_loader")
        
        stage2_input = {
            'train_data': train_data,
            'train_labels': train_labels,
            'test_data': test_data,
            'test_labels': test_labels,
            'device': self.device,
            'modalities': list(train_data.keys()),
            'n_train': len(train_labels),
            'n_test': len(test_labels)
        }
        
        logger.info(f"Data prepared for Stage 2: {stage2_input['n_train']} train, {stage2_input['n_test']} test samples")
        return stage2_input
    
    def prepare_for_stage3(self) -> Dict[str, Any]:
        """
        Prepare data for Stage 3: BaseLearnerSelector.
        
        Returns
        -------
        dict
            Data and metadata ready for Stage 3 processing
        """
        if self.bag_generator is None:
            raise ValueError("No bags generated. Call generate_bags() first.")
        
        stage3_input = {
            'bags': self.bags,
            'bag_generator': self.bag_generator,
            'device': self.device,
            'modalities': self.bag_generator.modality_names,
            'n_bags': len(self.bags),
            'n_modalities': self.bag_generator.n_modalities
        }
        
        logger.info(f"Data prepared for Stage 3: {stage3_input['n_bags']} bags, {stage3_input['n_modalities']} modalities")
        return stage3_input
    
    def prepare_for_stage4(self) -> Dict[str, Any]:
        """
        Prepare data for Stage 4: TrainingPipeline.
        
        Returns
        -------
        Dict[str, Any]
            Data and metadata ready for Stage 4 processing
        """
        if self.learner_selector is None:
            raise ValueError("No learners selected. Call select_learners() first.")
        
        # Get training and test data
        train_data, train_labels = self.data_loader.get_train_data()
        test_data, test_labels = self.data_loader.get_test_data()
        
        stage4_input = {
            'bags': self.bags,
            'learner_configs': self.learner_selector.learner_configs,
            'bag_analyses': self.learner_selector.bag_analyses,
            'train_data': train_data,
            'train_labels': train_labels,
            'test_data': test_data,
            'test_labels': test_labels,
            'device': self.device,
            'modalities': self.bag_generator.modality_names if self.bag_generator else [],
            'n_bags': len(self.bags),
            'n_learners': len(self.learner_selector.learner_configs)
        }
        
        logger.info(f"Data prepared for Stage 4: {stage4_input['n_bags']} bags, {stage4_input['n_learners']} learners")
        return stage4_input

    # to update every time
    def _validate_cross_stage_data_flow(self) -> None:
        """
        CRITICAL FIX: Validate data flow between stages to catch inconsistencies early.
        """
        logger.info("Validating cross-stage data flow...")
        
        # Validate Stage 1 output
        if self.data_loader is None:
            raise ValueError("Stage 1: No data loader found")
        
        if not hasattr(self.data_loader, 'data') or not self.data_loader.data:
            raise ValueError("Stage 1: No data available in data_loader")
        
        # Check data consistency using proper data access methods
        try:
            train_data, train_labels = self.data_loader.get_train_data()
            test_data, test_labels = self.data_loader.get_test_data()
        except Exception as e:
            raise ValueError(f"Stage 1: Failed to retrieve data: {e}")
        
        if len(train_labels) == 0:
            raise ValueError("Stage 1: Empty training labels")
        
        # Validate modality consistency
        modality_sample_counts = {name: len(data) for name, data in train_data.items()}
        if len(set(modality_sample_counts.values())) > 1:
            logger.warning(f"Stage 1: Inconsistent sample counts: {modality_sample_counts}")
        
        # Validate data types and shapes
        for modality_name, data in train_data.items():
            if not isinstance(data, np.ndarray):
                raise ValueError(f"Stage 1: Modality {modality_name} is not a numpy array")
            if data.size == 0:
                raise ValueError(f"Stage 1: Modality {modality_name} is empty")
        
        logger.info("✅ Cross-stage data flow validation passed")
    
    # ============================================================================
    # FULL MODEL RUN - to update as stages are implemented
    # ============================================================================

    def run_complete_pipeline(self, **kwargs) -> 'ModelAPI':
        """
        Run the complete multimodal ensemble pipeline from Stage 1 to Stage 5.
        
        This method follows the proper flow:
        1. All Stage 1 API methods (data loading, validation, etc.)
        2. prepare_for_stage2()
        3. All Stage 2 API methods (bag generation, tests, etc.)
        4. prepare_for_stage3()
        5. All Stage 3 API methods (learner selection, tests, etc.)
        6. prepare_for_stage4()
        
        Parameters
        ----------
        **kwargs
            Parameters for each stage of the pipeline:
            
            Stage 1 Parameters:
            - dataset: Dataset to load ('eurosat', 'oasis', 'mutla', 'custom')
            - data_dir: Path to dataset directory
            - test_size: Test split size (default: 0.2)
            - fast_mode: Enable fast loading (default: True)
            - max_samples: Maximum samples in fast mode (default: 1000)
            - random_state: Random seed (default: 42)
            
            Stage 2 Parameters:
            - n_bags: Number of bags to generate (default: 10)
            - dropout_strategy: Dropout strategy (default: 'adaptive')
            - max_dropout_rate: Maximum dropout rate (default: 0.7)
            - min_modalities: Minimum modalities per bag (default: 1)
            - sample_ratio: Bootstrap sampling ratio (default: 0.8)
            
            Stage 2 Testing Parameters:
            - run_stage2_tests: Whether to run Stage 2 tests (default: False)
            - test_noise_level: Noise level for robustness testing (default: 0.1)
            - test_verbose: Verbose test output (default: True)
            - continue_on_test_failure: Continue pipeline if tests fail (default: True)
            
            Stage 3 Parameters:
            - configuration_method: Hyperparameter configuration strategy (default: 'optimal')
            - add_configuration_diversity: Add configuration diversity (default: False)
            - validation_folds: Cross-validation folds for evaluation (default: 5)
            - early_stopping_patience: Early stopping patience (default: 10)
            - transfer_learning: Enable transfer learning (default: True)
            - run_stage3_tests: Whether to run Stage 3 tests (default: False)
            
            Cross-Stage Validation Parameters:
            - validate_cross_stage: Whether to validate data flow between stages (default: True)
            
        Returns
        -------
        ModelAPI
            Self with trained ensemble ready for prediction
            
        Example
        -------
        ```python
        api = ModelAPI(device='cuda')
        api = api.run_complete_pipeline(
            # Stage 1 parameters
            dataset='eurosat',
            data_dir='ProcessedData/EuroSAT',
            test_size=0.2,
            fast_mode=True,
            max_samples=1000,
                   # Stage 2 parameters
                   n_bags=10,
            dropout_strategy='adaptive',
            max_dropout_rate=0.6,
            # Stage 2 testing parameters
            run_stage2_tests=True,
            test_noise_level=0.1,
                   # Stage 3 parameters
            configuration_method='optimal',
            run_stage3_tests=True,
            # Cross-stage validation
            validate_cross_stage=True
        )
        ```
        """
        logger.info("Starting complete multimodal ensemble pipeline...")
        
        # ============================================================================
        # STAGE 1: DATA INTEGRATION - All API Methods
        # ============================================================================
        logger.info("=== STAGE 1: DATA INTEGRATION ===")
        
        # Extract Stage 1 parameters
        dataset = kwargs.get('dataset', 'eurosat')
        data_dir = kwargs.get('data_dir', 'ProcessedData/EuroSAT')
        test_size = kwargs.get('test_size', 0.2)
        fast_mode = kwargs.get('fast_mode', True)
        max_samples = kwargs.get('max_samples', 1000)
        random_state = kwargs.get('random_state', 42)
        
        # Load dataset based on parameter
        if dataset.lower() == 'eurosat':
            self.load_eurosat_data(
                data_dir=data_dir,
                test_size=test_size,
                fast_mode=fast_mode,
                max_samples=max_samples,
                random_state=random_state
            )
        elif dataset.lower() == 'oasis':
            self.load_oasis_data(
                data_dir=data_dir,
                test_size=test_size,
                fast_mode=fast_mode,
                max_samples=max_samples,
                random_state=random_state
            )
        elif dataset.lower() == 'mutla':
            self.load_mutla_data(
                data_dir=data_dir,
                test_size=test_size,
                fast_mode=fast_mode,
                max_samples=max_samples,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        # Stage 1: Data validation and consistency checks
        data_info = self.get_data_info()
        logger.info(f"Stage 1: Loaded {data_info['n_train']} train, {data_info['n_test']} test samples")
        logger.info(f"Stage 1: Modalities: {data_info['modalities']}")
        
        # Validate data consistency
        if not self.validate_data_consistency():
            raise ValueError("Stage 1: Data consistency validation failed")
        
        logger.info("✅ Stage 1: DataIntegration completed")
        
        # ============================================================================
        # PREPARE FOR STAGE 2
        # ============================================================================
        logger.info("=== PREPARING FOR STAGE 2 ===")
        stage2_input = self.prepare_for_stage2()
        logger.info(f"Stage 2 preparation: {stage2_input['n_train']} train samples, {len(stage2_input['modalities'])} modalities")
        
        # Cross-stage validation
        validate_cross_stage = kwargs.get('validate_cross_stage', True)
        if validate_cross_stage:
            self._validate_cross_stage_data_flow()
        
        # ============================================================================
        # STAGE 2: BAG GENERATION - All API Methods
        # ============================================================================
        logger.info("=== STAGE 2: BAG GENERATION ===")
        
        # Extract Stage 2 parameters
        stage2_params = {
            'n_bags': kwargs.get('n_bags', 10),
            'dropout_strategy': kwargs.get('dropout_strategy', 'adaptive'),
            'max_dropout_rate': kwargs.get('max_dropout_rate', 0.7),
            'min_modalities': kwargs.get('min_modalities', 1),
            'sample_ratio': kwargs.get('sample_ratio', 0.8),
            'random_state': random_state
        }
        
        # Generate bags
        self.generate_bags(**stage2_params)
        
        # Get bag information
        bag_info = self.get_bag_info()
        logger.info(f"Stage 2: Generated {bag_info['n_bags']} bags")
        logger.info(f"Stage 2: Average {bag_info['avg_modalities_per_bag']:.1f} modalities per bag")
        
        # Stage 2: Testing (optional)
        run_stage2_tests = kwargs.get('run_stage2_tests', False)
        if run_stage2_tests:
            test_noise_level = kwargs.get('test_noise_level', 0.1)
            test_verbose = kwargs.get('test_verbose', True)
            
            if test_verbose:
                logger.info("Running Stage 2 tests...")
            
            try:
                test_results = self.run_stage2_tests(noise_level=test_noise_level)
                
                if test_verbose:
                    logger.info("Stage 2 tests completed successfully")
                    logger.info(f"  - Interpretability: {len(test_results.get('interpretability', {}))} analysis categories")
                    logger.info(f"  - Robustness: {len(test_results.get('robustness', {}))} analysis categories")
                    logger.info(f"  - Consistency: {len(test_results.get('consistency', {}))} validation checks")
            except Exception as test_error:
                logger.warning(f"Stage 2 tests failed: {test_error}")
                if not kwargs.get('continue_on_test_failure', True):
                    raise
        
        logger.info("✅ Stage 2: BagGeneration completed")
        
        # ============================================================================
        # PREPARE FOR STAGE 3
        # ============================================================================
        logger.info("=== PREPARING FOR STAGE 3 ===")
        stage3_input = self.prepare_for_stage3()
        logger.info(f"Stage 3 preparation: {stage3_input['n_bags']} bags, {stage3_input['n_modalities']} modalities")
        
        # ============================================================================
        # STAGE 3: BASE LEARNER SELECTION - All API Methods
        # ============================================================================
        logger.info("=== STAGE 3: BASE LEARNER SELECTION ===")
        
        # Extract Stage 3 parameters
        stage3_params = {
            'configuration_method': kwargs.get('configuration_method', 'optimal'),
            'add_configuration_diversity': kwargs.get('add_configuration_diversity', False),
            'validation_folds': kwargs.get('validation_folds', 5),
            'early_stopping_patience': kwargs.get('early_stopping_patience', 10),
            'transfer_learning': kwargs.get('transfer_learning', True),
            'random_state': random_state
        }
        
        # Select learners
        self.select_learners(**stage3_params)
        
        # Get learner information
        learner_info = self.get_learner_info()
        logger.info(f"Stage 3: Selected {learner_info['n_learners']} learners")
        logger.info(f"Stage 3: Learner types: {learner_info['unique_learner_types']}")
        
        # Get bag analyses
        bag_analyses = self.get_bag_analyses()
        logger.info(f"Stage 3: Analyzed {len(bag_analyses)} bags")
        
        # Get learner summary
        learner_summary = self.get_learner_summary()
        logger.info(f"Stage 3: Learner breakdown: {learner_summary['learner_breakdown']}")
        
        # Stage 3: Testing (optional)
        run_stage3_tests = kwargs.get('run_stage3_tests', False)
        if run_stage3_tests:
            logger.info("Running Stage 3 tests...")
            try:
                stage3_test_results = self.run_learner_selection_tests()
                logger.info(f"Stage 3 tests completed: {len(stage3_test_results)} test categories")
            except Exception as test_error:
                logger.warning(f"Stage 3 tests failed: {test_error}")
                if not kwargs.get('continue_on_test_failure', True):
                    raise
        
        logger.info("✅ Stage 3: BaseLearnerSelector completed")
        
        # ============================================================================
        # PREPARE FOR STAGE 4
        # ============================================================================
        logger.info("=== PREPARING FOR STAGE 4 ===")
        stage4_input = self.prepare_for_stage4()
        logger.info(f"Stage 4 preparation: {stage4_input['n_bags']} bags, {stage4_input['n_learners']} learners")
        
        # ============================================================================
        # STAGE 4: TRAINING PIPELINE (TODO)
        # ============================================================================
        logger.info("=== STAGE 4: TRAINING PIPELINE ===")
        try:
            #TODO: Implement when trainingPipeline is ready
            logger.info("Stage 4: trainingPipeline not yet implemented")
        except NotImplementedError:
            logger.warning("Stage 4: trainingPipeline not yet implemented")
        
        # ============================================================================
        # STAGE 5: ENSEMBLE PREDICTION (TODO)
        # ============================================================================
        logger.info("=== STAGE 5: ENSEMBLE PREDICTION ===")
        try:
            #TODO: Implement when ensemblePrediction is ready
            logger.info("Stage 5: ensemblePrediction not yet implemented")
        except NotImplementedError:
            logger.warning("Stage 5: ensemblePrediction not yet implemented")
        
        logger.info("🎯 Complete pipeline execution finished!")
        return self
    
    
    # ============================================================================
    # CONVENIENCE METHODS FOR STAGE 1 - DATA INTEGRATION
    # ============================================================================
    
    def get_available_datasets(self) -> list:
        """Get list of available datasets."""
        return ['eurosat', 'oasis', 'mutla']
    
    def get_supported_modalities(self) -> list:
        """Get list of supported modality types."""
        return ['visual', 'tabular', 'spectral', 'time-series']
    
    def validate_data_consistency(self) -> bool:
        """
        Validate data consistency for loaded data.
        
        Returns
        -------
        bool
            True if data is consistent, False otherwise
        """
        if self.data_loader is None:
            raise ValueError("No data loaded. Call one of the load_* methods first.")
        
        try:
            # This will raise an exception if validation fails
            train_data, train_labels = self.get_train_data()
            return True
        except Exception as e:
            logger.error(f"Data consistency validation failed: {e}")
            return False

    # ============================================================================
    # CONVENIENCE METHODS FOR STAGE 2 - INTERPRETABILITY AND ROBUSTNESS TESTS
    # ============================================================================
    
    def run_interpretability_test(self) -> Dict[str, Any]:
        """
        Run interpretability test for Stage 2: BagGeneration.
        
        Returns
        -------
        dict
            Interpretability analysis results
        """
        if self.bag_generator is None:
            raise ValueError("No bags generated. Call generate_bags() first.")
        
        logger.info("Running Stage 2 interpretability test...")
        
        try:
            interpretability_results = self.bag_generator.interpretability_test()
            logger.info("Stage 2 interpretability test completed successfully")
            return interpretability_results
        except Exception as e:
            logger.error(f"Stage 2 interpretability test failed: {e}")
            raise
    
    def run_robustness_test(self, noise_level: float = 0.1, **kwargs) -> Dict[str, Any]:
        """
        Run robustness test for Stage 2: BagGeneration.
        
        Parameters
        ----------
        noise_level : float
            Level of noise to add for robustness testing
        **kwargs
            Additional parameters for robustness test
            
        Returns
        -------
        dict
            Robustness analysis results
        """
        if self.bag_generator is None:
            raise ValueError("No bags generated. Call generate_bags() first.")
        
        logger.info(f"Running Stage 2 robustness test with noise level: {noise_level}")
        
        try:
            robustness_results = self.bag_generator.robustness_test(noise_level=noise_level, **kwargs)
            logger.info("Stage 2 robustness test completed successfully")
            return robustness_results
        except Exception as e:
            logger.error(f"Stage 2 robustness test failed: {e}")
            raise
    
    def run_bag_consistency_test(self) -> Dict[str, Any]:
        """
        Run bag consistency test for Stage 2: BagGeneration.
        
        Returns
        -------
        dict
            Bag consistency analysis results
        """
        if self.bag_generator is None:
            raise ValueError("No bags generated. Call generate_bags() first.")
        
        logger.info("Running Stage 2 bag consistency test...")
        
        try:
            consistency_results = self.bag_generator.test_bag_consistency()
            logger.info("Stage 2 bag consistency test completed successfully")
            return consistency_results
        except Exception as e:
            logger.error(f"Stage 2 bag consistency test failed: {e}")
            raise
    
    def run_stage2_tests(self, noise_level: float = 0.1, **kwargs) -> Dict[str, Any]:
        """
        Run all Stage 2 tests (interpretability, robustness, consistency).
        
        Parameters
        ----------
        noise_level : float
            Level of noise to add for robustness testing
        **kwargs
            Additional parameters for tests
            
        Returns
        -------
        dict
            Combined results from all Stage 2 tests
        """
        if self.bag_generator is None:
            raise ValueError("No bags generated. Call generate_bags() first.")
        
        logger.info("Running all Stage 2 tests...")
        
        results = {}
        
        try:
            # Run interpretability test
            results['interpretability'] = self.run_interpretability_test()
            
            # Run robustness test
            results['robustness'] = self.run_robustness_test(noise_level=noise_level, **kwargs)
            
            # Run consistency test
            results['consistency'] = self.run_bag_consistency_test()
            
            logger.info("All Stage 2 tests completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Stage 2 tests failed: {e}")
            raise

    #Convenience Methods for Stage 3 - Intepretation Tests, and Robustness Tests

    def run_learner_selection_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive tests for learner selection quality.
        
        Returns
        -------
        Dict[str, Any]
            Test results including robustness, interpretability, and selection quality
        """
        if self.learner_selector is None:
            raise ValueError("No learners selected. Call select_learners() first.")
        
        logger.info("Running Stage 3 learner selection tests...")
        test_results = self.learner_selector.run_selection_tests()
        logger.info("Stage 3 tests completed")
        
        return test_results

    #Convenience Methods for Stage 4 - Abalation Studies, Intepretation Tests, and Robustness Tests

    #Convenience Methods for Stage 5 - Abalation Studies, Intepretation Tests, and Robustness Tests


    # ============================================================================
    # End of API
    # ============================================================================

"""
IMPLEMENTATION STATUS:
=====================

Stage 1: DataIntegration - COMPLETE
   - SimpleDataLoader class implemented
   - All dataset loaders (EuroSAT, OASIS, MUTLA, Custom) working
   - API wrapper integrated with method chaining
   - Data validation and consistency checks
   - Memory efficiency features (lazy loading, chunking)

Stage 2: BagGeneration - COMPLETE
   - BagGeneration class implemented and integrated
   - Adaptive modality dropout strategy
   - Ensemble bag generation with bootstrap sampling
   - Interpretability and robustness tests
   - API wrapper integrated with method chaining

Stage 3: BaseLearnerSelector - COMPLETE ✅
   - BaseLearnerSelector class fully implemented
   - Performance-based learner selection with actual optimization
   - Modality-specific learner optimization with grid search
   - Ensemble diversity metrics and comprehensive testing

Stage 4: trainingPipeline - TODO
   - TrainingPipeline class needs implementation
   - Unified training across modalities
   - Loss function design for multimodal ensembles
   - Hyperparameter optimization

Stage 5: ensemblePrediction - TODO
   - EnsemblePrediction class needs implementation
   - Prediction aggregation strategies
   - Uncertainty quantification
   - Model interpretation and explainability

NEXT STEPS:
===========
1. ✅ Implement Stage 2: BagGeneration - COMPLETED
2. ✅ Update API imports and method implementations - COMPLETED
3. ✅ Test Stage 1 → Stage 2 integration - COMPLETED
4. ✅ Implement Stage 3: BaseLearnerSelector - COMPLETED
5. Continue with Stage 4, 5 in sequence
6. Add comprehensive testing for full pipeline
"""
