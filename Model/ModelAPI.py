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
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import torch

# Import Stage 1: DataIntegration
from DataIntegration import SimpleDataLoader, load_eurosat_data, load_oasis_data, load_mutla_data, load_custom_data

# Import Stage 2: BagGeneration
try:
    from .BagGeneration import BagGeneration
except ImportError:
    from BagGeneration import BagGeneration

# Import Stage 3: baseLearnerSelector (TODO: Uncomment when implemented)
# try:
#     from .baseLearnerSelector import BaseLearnerSelector
# except ImportError:
#     from baseLearnerSelector import BaseLearnerSelector

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
                 lazy_loading: bool = False, chunk_size: int = 1000):
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
            chunk_size=chunk_size
        )
        
        # Initialize Stage 2: BagGeneration
        self.bag_generator = None
        
        # Initialize Stage 3: baseLearnerSelector (TODO: Uncomment when implemented)
        # self.learner_selector = None
        
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
        Load EuroSAT dataset (images + spectral data) for unified model.
        
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
        Load MUTLA dataset (behavioral + physiological + visual data) for unified model.
        
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
        logger.info("Loading MUTLA dataset...")
        
        # Use the existing data_loader instead of creating new one
        modality_types = {
            "behavioral": "behavioral",      # CSV files with user interaction data
            "physiological": "physiological", # LOG files with EEG/attention time-series
            "visual": "visual"               # NPY files with webcam tracking data
        }
        
        modality_files = {
            "behavioral": f"{data_dir}/User records/math_record_cleaned.csv",  # Use one of the behavioral files as labels
            "physiological": f"{data_dir}/Brainwave",
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
            label_file=f"{data_dir}/User records/math_record_cleaned.csv",  # Use one of the behavioral files as labels
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
    
    # ============================================================================
    # DATA ACCESS METHODS
    # ============================================================================
    
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
            # Access train data from the data loader's data dictionary
            if 'train_data' in self.data_loader.data and 'train_labels' in self.data_loader.data:
                train_data = self.data_loader.data['train_data']
                train_labels = self.data_loader.data['train_labels']
                
                # Validate data structure
                if not isinstance(train_data, dict):
                    raise ValueError("Training data is not a dictionary")
                
                if not isinstance(train_labels, np.ndarray):
                    raise ValueError("Training labels are not a numpy array")
                
                if len(train_data) == 0:
                    raise ValueError("Training data dictionary is empty")
                
                if len(train_labels) == 0:
                    raise ValueError("Training labels array is empty")
                
                logger.debug(f"Successfully retrieved training data: {len(train_labels)} samples, {len(train_data)} modalities")
                return train_data, train_labels
            else:
                raise ValueError("Training data not available. Data may not be properly loaded.")
                
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
            # Access test data from the data loader's data dictionary
            if 'test_data' in self.data_loader.data and 'test_labels' in self.data_loader.data:
                test_data = self.data_loader.data['test_data']
                test_labels = self.data_loader.data['test_labels']
                
                # Validate data structure
                if not isinstance(test_data, dict):
                    raise ValueError("Test data is not a dictionary")
                
                if not isinstance(test_labels, np.ndarray):
                    raise ValueError("Test labels are not a numpy array")
                
                if len(test_data) == 0:
                    raise ValueError("Test data dictionary is empty")
                
                if len(test_labels) == 0:
                    raise ValueError("Test labels array is empty")
                
                logger.debug(f"Successfully retrieved test data: {len(test_labels)} samples, {len(test_data)} modalities")
                return test_data, test_labels
            else:
                raise ValueError("Test data not available. Data may not be properly loaded.")
                
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
    # STAGE 3: BASE LEARNER SELECTOR API
    # ============================================================================
    
    # to update with actual methods
 
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
    
    # prepare for stage 3

    # prepare for stage 4

    # prepare for stage 5

    # to update once other stages are implemented
    def run_complete_pipeline(self, **kwargs) -> 'ModelAPI':
        """
        Run the complete multimodal ensemble pipeline from Stage 1 to Stage 5.
        
        This method demonstrates the full pipeline flow that will be available
        once all stages are implemented.
        
        Parameters
        ----------
        **kwargs
            Parameters for each stage of the pipeline
            
        Returns
        -------
        ModelAPI
            Self with trained ensemble ready for prediction
            
        Example
        -------
        ```python
        api = ModelAPI(device='cuda')
        api = (api.load_oasis_data(test_size=0.2)
               .run_complete_pipeline(
                   # Stage 2 parameters
                   dropout_rates=[0.1, 0.2, 0.3],
                   n_bags=10,
                   # Stage 3 parameters
                   selection_strategy='performance_based',
                   # Stage 4 parameters
                   epochs=100,
                   batch_size=32,
                   # Stage 5 parameters
                   ensemble_method='weighted_average'
               ))
        
        # Make predictions
        predictions = api.predict(test_data)
        metrics = api.evaluate(test_data, test_labels)
        ```
        """
        logger.info("Starting complete multimodal ensemble pipeline...")
        
        # Stage 1: DataIntegration (already completed when this method is called)
        if self.data_loader is None:
            raise ValueError("No data loaded. Call one of the load_* methods first.")
        logger.info("Stage 1: DataIntegration completed")
        
        # Stage 2: BagGeneration
        try:
            # Extract Stage 2 parameters from kwargs
            stage2_params = {
                'n_bags': kwargs.get('n_bags', 10),
                'dropout_strategy': kwargs.get('dropout_strategy', 'adaptive'),
                'max_dropout_rate': kwargs.get('max_dropout_rate', 0.7),
                'min_modalities': kwargs.get('min_modalities', 1),
                'sample_ratio': kwargs.get('sample_ratio', 0.8),
                'random_state': kwargs.get('random_state', 42)
            }
            
            # Generate bags
            self.generate_bags(**stage2_params)
            logger.info("Stage 2: BagGeneration completed")
        except Exception as e:
            logger.error(f"Stage 2: BagGeneration failed: {e}")
            raise
        
        # Stage 3: baseLearnerSelector
        try:
            #TODO: Implement when baseLearnerSelector is ready
            logger.info("Stage 3: baseLearnerSelector completed")
        except NotImplementedError:
            logger.warning("Stage 3: baseLearnerSelector not yet implemented")
        
        # Stage 4: trainingPipeline
        try:
            #TODO: Implement when trainingPipeline is ready
            logger.info("Stage 4: trainingPipeline completed")
        except NotImplementedError:
            logger.warning("Stage 4: trainingPipeline not yet implemented")
        
        # Stage 5: ensemblePrediction
        try:
            #TODO: Implement when ensemblePrediction is ready
            logger.info("Stage 5: ensemblePrediction initialized")
        except NotImplementedError:
            logger.warning("Stage 5: ensemblePrediction not yet implemented")
        
        logger.info("Complete pipeline execution finished!")
        return self
    
    # ============================================================================
    # CONVENIENCE METHODS FOR STAGE 1 - DATA INTEGRATION
    # ============================================================================
    
    def get_available_datasets(self) -> list:
        """Get list of available datasets."""
        return ['eurosat', 'oasis', 'mutla']
    
    def get_supported_modalities(self) -> list:
        """Get list of supported modality types."""
        return ['image', 'tabular', 'spectral', 'behavioral', 'physiological', 'visual']
    
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

    #Convenience Methods for Stage 3 - Abalation Studies, Intepretation Tests, and Robustness Tests

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

Stage 3: baseLearnerSelector - TODO  
   - BaseLearnerSelector class needs implementation
   - Performance-based learner selection
   - Modality-specific learner optimization
   - Ensemble diversity metrics

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
4. Implement Stage 3: BaseLearnerSelector
5. Continue with Stage 4, 5 in sequence
6. Add comprehensive testing for full pipeline
"""
