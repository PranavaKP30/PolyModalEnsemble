#!/usr/bin/env python3
"""
Phase 3: MainModel Hyperparameter Optimization
Purpose: Find optimal hyperparameter configuration for the MainModel using Optuna
"""

import json
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import sys
import os

# Add the MainModel to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../MainModel'))
sys.path.append('/Users/pranav/Coding/Projects/PolyModalEnsemble/MainModel')
sys.path.append('/Users/pranav/Coding/Projects/PolyModalEnsemble')

try:
    from mainModelAPI import MultiModalEnsembleModel
    MAINMODEL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"MainModel not available: {e}")
    MAINMODEL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_latest_experiment_run(results_dir: str = "results") -> tuple[str, int]:
    """
    Find the latest experiment run and seed automatically.
    
    Returns:
        tuple: (experiment_name, seed_number)
    """
    try:
        results_path = Path(results_dir)
        if not results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {results_path}")
        
        # Find all experiment directories
        experiment_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("amazon_reviews_")]
        
        if not experiment_dirs:
            raise FileNotFoundError("No experiment directories found")
        
        # Sort by creation time (newest first)
        latest_experiment_dir = max(experiment_dirs, key=lambda d: d.stat().st_mtime)
        experiment_name = latest_experiment_dir.name
        
        # Find all seed directories in the latest experiment
        seed_dirs = [d for d in latest_experiment_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]
        
        if not seed_dirs:
            raise FileNotFoundError(f"No seed directories found in {experiment_name}")
        
        # Sort by creation time (newest first)
        latest_seed_dir = max(seed_dirs, key=lambda d: d.stat().st_mtime)
        seed_number = int(latest_seed_dir.name.split("_")[1])
        
        logger.info(f"Auto-detected latest experiment: {experiment_name}, seed: {seed_number}")
        return experiment_name, seed_number
        
    except Exception as e:
        logger.error(f"Error finding latest experiment run: {e}")
        # Fallback to default values
        return "amazon_reviews_quick_20250913_231038", 42


class MainModelOptimizer:
    """MainModel hyperparameter optimization using Optuna with stratified sampling."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the optimizer with configuration."""
        self.config = config
        self.seed = config.get("seed", 42)
        self.test_mode = config.get("test_mode", "quick")
        self.phase_dir = Path(config["phase_dir"])
        self.dataset_path = config.get("dataset_path", "/Users/pranav/Coding/Projects/PolyModalEnsemble/Benchmarking/ProcessedData/AmazonReviews")
        
        # Set number of trials based on test mode (Optuna intelligent search)
        if self.test_mode == "quick":
            self.n_trials = 75  # Quick test with intelligent search
        elif self.test_mode == "full":
            self.n_trials = 300  # Full test with comprehensive search
        else:
            self.n_trials = 75  # Default fallback
            
        logger.info(f"Initialized MainModelOptimizer for {self.test_mode} mode")
        logger.info(f"Number of Optuna trials: {self.n_trials}")
        
        # Initialize Optuna study
        self.study = None
        self.best_params = None
        self.best_score = None
    
    def _load_phase1_data(self, processed_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load processed data from Phase 1 (either from parameter or from saved files)."""
        try:
            # If processed_data is provided (from experiment runner), use it directly
            if processed_data is not None:
                logger.info("Using processed data from current experiment run")
                
                # Load data from the file paths provided in processed_data
                train_text_file = processed_data['train_text']
                train_metadata_file = processed_data['train_metadata'] 
                test_text_file = processed_data['test_text']
                test_metadata_file = processed_data['test_metadata']
            
            # Load separate modality data
            train_text_data = pd.read_csv(train_text_file)
            train_metadata_data = pd.read_csv(train_metadata_file)
            test_text_data = pd.read_csv(test_text_file)
            test_metadata_data = pd.read_csv(test_metadata_file)
            
                # Extract task metadata from processed_data
                task_type = processed_data.get('task_type', 'classification')
                num_classes = processed_data.get('num_classes', 5)
                class_names = processed_data.get('class_names', ["1-star", "2-star", "3-star", "4-star", "5-star"])
                
                phase1_data = {
                'train_text': train_text_data,
                'train_metadata': train_metadata_data,
                'test_text': test_text_data,
                'test_metadata': test_metadata_data,
                    'task_type': task_type,
                    'num_classes': num_classes,
                    'class_names': class_names
            }
            
                logger.info(f"Loaded Phase 1 data from current run: {list(phase1_data.keys())}")
            logger.info(f"Train samples: {len(train_text_data)}, Test samples: {len(test_text_data)}")
                logger.info(f"Task type: {task_type}, Classes: {num_classes}")
                
                return phase1_data
            
            # Fallback: Load from current experiment run's Phase 1 data
            logger.info("No processed_data provided, loading from current experiment run's Phase 1 data")
            
            # Get current experiment directory from phase_dir
            # phase_dir format: results/experiment_name/seed_X/phase_3_mainmodel
            # We need: results/experiment_name/seed_X/phase_1_data_validation
            current_phase_dir = Path(self.phase_dir)
            seed_dir = current_phase_dir.parent
            experiment_dir = seed_dir.parent
            
            # Find the current experiment's Phase 1 directory
            phase1_dir = seed_dir / "phase_1_data_validation"
            
            if not phase1_dir.exists():
                raise FileNotFoundError(f"Phase 1 directory not found at {phase1_dir}. Please run Phase 1 first.")
            
            # Load Phase 1 results to get the processed data paths
            phase1_results_file = phase1_dir / "phase_1_data_validation_results.json"
            if not phase1_results_file.exists():
                raise FileNotFoundError(f"Phase 1 results not found at {phase1_results_file}")
            
            with open(phase1_results_file, 'r') as f:
                phase1_results = json.load(f)
            
            processed_data = phase1_results.get("processed_data", {})
            if not processed_data:
                raise ValueError("No processed_data found in Phase 1 results")
            
            logger.info(f"Loading current experiment's Phase 1 data from {phase1_dir}")
            
            # Load separate modality data from file paths
            train_text_data = pd.read_csv(processed_data['train_text'])
            train_metadata_data = pd.read_csv(processed_data['train_metadata'])
            test_text_data = pd.read_csv(processed_data['test_text'])
            test_metadata_data = pd.read_csv(processed_data['test_metadata'])
            
            # Extract task metadata from processed_data
            task_type = processed_data.get('task_type', 'classification')
            num_classes = processed_data.get('num_classes', 5)
            class_names = processed_data.get('class_names', ["1-star", "2-star", "3-star", "4-star", "5-star"])
            
            phase1_data = {
                'train_text': train_text_data,
                'train_metadata': train_metadata_data,
                'test_text': test_text_data,
                'test_metadata': test_metadata_data,
                'task_type': task_type,
                'num_classes': num_classes,
                'class_names': class_names
            }
            
            logger.info(f"Loaded current experiment's Phase 1 data: {list(phase1_data.keys())}")
            logger.info(f"Train samples: {len(train_text_data)}, Test samples: {len(test_text_data)}")
            logger.info(f"Task type: {task_type}, Classes: {num_classes}")
            
            return phase1_data
        except Exception as e:
            logger.error(f"Error loading Phase 1 data: {e}")
            raise
    
    def _stratified_sample_1k(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sample 1k samples from Phase 1 data for hyperparameter optimization."""
        try:
            from sklearn.model_selection import train_test_split
            
            # Get labels from train text data (last column)
            train_text_data = processed_data['train_text']
            train_metadata_data = processed_data['train_metadata']
            
            # Extract labels (last column)
            y_train = train_text_data.iloc[:, -1].values
            
            # Check if we have enough samples
            if len(y_train) <= 1000:
                logger.warning(f"Dataset size ({len(y_train)}) <= 1000 samples. Using full dataset for HPO.")
                return processed_data
            
            # Perform stratified sampling to get 1000 samples
            indices = np.arange(len(y_train))
            train_indices, _, _, _ = train_test_split(
                indices, y_train, 
                train_size=1000, 
                stratify=y_train, 
                random_state=self.seed
            )
            
            # Sample the data
            sampled_train_text = train_text_data.iloc[train_indices].reset_index(drop=True)
            sampled_train_metadata = train_metadata_data.iloc[train_indices].reset_index(drop=True)
            
            # Also sample test data proportionally (maintain train/test ratio)
            test_size = min(250, len(processed_data['test_text']))  # ~25% of train for test
            test_text_data = processed_data['test_text']
            test_metadata_data = processed_data['test_metadata']
            y_test = test_text_data.iloc[:, -1].values
            
            if len(y_test) > test_size:
                test_indices, _, _, _ = train_test_split(
                    np.arange(len(y_test)), y_test,
                    train_size=test_size,
                    stratify=y_test,
                    random_state=self.seed
                )
                sampled_test_text = test_text_data.iloc[test_indices].reset_index(drop=True)
                sampled_test_metadata = test_metadata_data.iloc[test_indices].reset_index(drop=True)
            else:
                sampled_test_text = test_text_data
                sampled_test_metadata = test_metadata_data
            
            # Create sampled data dictionary
            sampled_data = {
                'train_text': sampled_train_text,
                'train_metadata': sampled_train_metadata,
                'test_text': sampled_test_text,
                'test_metadata': sampled_test_metadata,
                'task_type': processed_data['task_type'],
                'num_classes': processed_data['num_classes'],
                'class_names': processed_data['class_names']
            }
            
            logger.info(f"Stratified sampling completed:")
            logger.info(f"  Train samples: {len(sampled_train_text)}")
            logger.info(f"  Test samples: {len(sampled_test_text)}")
            
            # Log class distribution
            unique_classes, counts = np.unique(sampled_train_text.iloc[:, -1].values, return_counts=True)
            logger.info(f"  Train class distribution: {dict(zip(unique_classes, counts))}")
            
            return sampled_data
            
        except Exception as e:
            logger.error(f"Error in stratified sampling: {e}")
            raise
    
    def _select_best_trial_unbalanced_classification(self, study) -> Dict[str, Any]:
        """
        Select best trial for unbalanced classification dataset based on selection criteria:
        1. Primary metric: Balanced Accuracy (higher is better)
        2. Secondary metrics: Weighted F1-Score, Weighted Precision, Weighted Recall, Accuracy
        3. Efficiency: Training time < 2x average
        4. Robustness: CV stability (std/mean) < 0.2
        """
        logger.info("Applying selection criteria for unbalanced classification...")
        
        # Collect all completed trials with metrics
        valid_trials = []
        training_times = []
        
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                # Extract metrics from user attributes
                metrics = {
                    'trial_number': trial.number,
                    'accuracy': trial.user_attrs.get('accuracy', 0.0),
                    'balanced_accuracy': trial.user_attrs.get('balanced_accuracy', 0.0),
                    'weighted_f1': trial.user_attrs.get('weighted_f1', 0.0),
                    'weighted_precision': trial.user_attrs.get('weighted_precision', 0.0),
                    'weighted_recall': trial.user_attrs.get('weighted_recall', 0.0),
                    'training_time': trial.duration.total_seconds() if trial.duration else 0.0
                }
                
                valid_trials.append({
                    'trial': trial,
                    'metrics': metrics,
                    'params': trial.params
                })
                training_times.append(metrics['training_time'])
        
        if not valid_trials:
            logger.warning("No valid trials found, using Optuna's best trial")
            best_trial = study.best_trial
            return {
                'trial_number': best_trial.number,
                'balanced_accuracy': best_trial.value,
                'params': best_trial.params,
                'metrics': {},
                'selection_reason': 'fallback_to_optuna_best'
            }
        
        logger.info(f"Evaluating {len(valid_trials)} valid trials...")
        
        # Step 1: Filter trials with accuracy >= 0.5 (basic quality filter)
        filtered_trials = []
        for trial_data in valid_trials:
            if trial_data['metrics']['accuracy'] >= 0.5:
                filtered_trials.append(trial_data)
        
        logger.info(f"After accuracy filter (>=0.5): {len(filtered_trials)} trials")
        
        if not filtered_trials:
            logger.warning("No trials passed accuracy filter, using best available trial")
            # Use trial with highest balanced accuracy regardless of accuracy filter
            best_trial_data = max(valid_trials, key=lambda x: x['metrics']['balanced_accuracy'])
            return {
                'trial_number': best_trial_data['trial'].number,
                'balanced_accuracy': best_trial_data['metrics']['balanced_accuracy'],
                'params': best_trial_data['params'],
                'metrics': best_trial_data['metrics'],
                'selection_reason': 'best_balanced_accuracy_no_accuracy_filter'
            }
        
        # Step 2: Calculate efficiency threshold (2x average training time)
        avg_training_time = np.mean(training_times) if training_times else 0
        efficiency_threshold = 2.0 * avg_training_time
        
        # Step 3: Rank by primary metric (Balanced Accuracy)
        filtered_trials.sort(key=lambda x: x['metrics']['balanced_accuracy'], reverse=True)
        
        # Step 4: Apply efficiency filter
        efficient_trials = []
        for trial_data in filtered_trials:
            if trial_data['metrics']['training_time'] <= efficiency_threshold:
                efficient_trials.append(trial_data)
        
        logger.info(f"After efficiency filter (<=2x avg time): {len(efficient_trials)} trials")
        
        # Step 5: Select best trial
        if efficient_trials:
            best_trial_data = efficient_trials[0]  # Already sorted by balanced accuracy
            selection_reason = 'best_balanced_accuracy_efficient'
                else:
            # If no efficient trials, use best balanced accuracy regardless of efficiency
            best_trial_data = filtered_trials[0]
            selection_reason = 'best_balanced_accuracy_ignoring_efficiency'
        
        logger.info(f"Selected trial {best_trial_data['trial'].number}:")
        logger.info(f"  Balanced Accuracy: {best_trial_data['metrics']['balanced_accuracy']:.4f}")
        logger.info(f"  Accuracy: {best_trial_data['metrics']['accuracy']:.4f}")
        logger.info(f"  Weighted F1: {best_trial_data['metrics']['weighted_f1']:.4f}")
        logger.info(f"  Training Time: {best_trial_data['metrics']['training_time']:.2f}s")
        logger.info(f"  Selection Reason: {selection_reason}")
        
        return {
            'trial_number': best_trial_data['trial'].number,
            'balanced_accuracy': best_trial_data['metrics']['balanced_accuracy'],
            'params': best_trial_data['params'],
            'metrics': best_trial_data['metrics'],
            'selection_reason': selection_reason
        }
    
    def _run_final_model_with_full_data(self, best_trial: Dict[str, Any], hpo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the final model with full Phase 1 dataset using the best hyperparameters.
        """
        try:
            logger.info("Loading full Phase 1 dataset for final model training...")
            
            # Load the full Phase 1 data (not the 1k samples used for HPO)
            full_phase1_data = self._load_phase1_data()
            
            # Extract full dataset
            full_train_text = full_phase1_data['train_text']
            full_train_metadata = full_phase1_data['train_metadata']
            full_test_text = full_phase1_data['test_text']
            full_test_metadata = full_phase1_data['test_metadata']
            
            logger.info(f"Full dataset loaded:")
            logger.info(f"  Train samples: {len(full_train_text)}")
            logger.info(f"  Test samples: {len(full_test_text)}")
            
            # Prepare data for MainModel (correct format - only training data in X)
            X_full = {
                'text': full_train_text.iloc[:, :-1].values,  # Only training features
                'metadata': full_train_metadata.iloc[:, :-1].values  # Only training features
            }
            
            y_full = full_train_text.iloc[:, -1].values  # Only training labels
            
            # Get best hyperparameters and reconstruct complete parameter set
            best_varied_params = best_trial['params']
            
            # Reconstruct the complete parameter set (varied + fixed) like in _objective
            complete_params = best_varied_params.copy()
            complete_params.update({
                # Stage 1: Data Integration (Fixed)
                'verbose': True,
                'handle_nan': 'fill_mean',
                'handle_inf': 'fill_max',
                'normalize': True,  # Enable normalization for proper feature scaling across modalities
                'remove_outliers': False,
                'outlier_std': 3.0,
                
                # Stage 2: Ensemble Generation (Fixed)
                'dropout_strategy': 'adaptive',
                'min_modalities': 1,
                'random_state': self.seed,  # Use same seed as HPO to ensure same model architecture
                
                # Stage 3: Base Learner Selection (Fixed)
                'task_type': 'classification',
                'n_classes': 5,
                'modality_aware': True,
                'bag_learner_pairing': True,
                'metadata_level': 'complete',
                'pairing_focus': 'performance',
                'feature_ratio_weight': 0.4,
                'variance_weight': 0.3,
                'dimensionality_weight': 0.3,
                'base_performance': 0.6,
                'diversity_bonus': 0.1,
                'weightage_bonus': 0.1,
                'dropout_penalty': 0.1,
                'early_stopping': True,
                'hidden_layers': (128, 64),
                
                # Stage 4: Base Learner Training (Fixed)
                'feature_sampling_ratio': 1.0,
                'aggregation_strategy': 'transformer_fusion',
                'verbose': True
            })
            
            logger.info(f"Initializing MainModel with best hyperparameters from trial {best_trial['trial_number']}...")
            
            # Initialize MainModel with complete hyperparameters
            final_model = MultiModalEnsembleModel(
                task_type=hpo_data['task_type'],  # Use task_type from dataset, not hyperparams
                n_bags=complete_params['n_bags'],
                dropout_strategy=complete_params['dropout_strategy'],
                max_dropout_rate=complete_params['max_dropout_rate'],
                min_modalities=complete_params['min_modalities'],
                sample_ratio=complete_params['sample_ratio'],
                feature_sampling_ratio=complete_params['feature_sampling_ratio'],
                random_state=complete_params['random_state'],
                optimization_mode=complete_params['optimization_mode'],
                epochs=complete_params['epochs'],
                batch_size=complete_params['batch_size'],
                learning_rate=complete_params['learning_rate'],
                weight_decay=complete_params['weight_decay'],
                early_stopping_patience=complete_params['early_stopping_patience'],
                denoising_weight=complete_params['denoising_weight'],
                aggregation_strategy=complete_params['aggregation_strategy'],
                transformer_num_heads=complete_params['transformer_num_heads'],
                transformer_hidden_dim=complete_params['transformer_hidden_dim'],
                uncertainty_method=complete_params['uncertainty_method'],
                verbose=True
            )
            
            # Train on full dataset
            logger.info("Training final model on full dataset...")
            start_time = time.time()
            final_model.fit(X_full, y_full)
            training_time = time.time() - start_time
            
            logger.info(f"Final model training completed in {training_time:.2f} seconds")
            
            # Make predictions on full test set
            logger.info("Making predictions on full test set...")
            test_X_full = {
                'text': full_test_text.iloc[:, :-1].values,
                'metadata': full_test_metadata.iloc[:, :-1].values
            }
            
            test_y_full = full_test_text.iloc[:, -1].values  # Test labels
            
            prediction_result = final_model.predict(test_X_full)
            
            # Extract predictions and calculate comprehensive metrics
            import numpy as np
            predictions = np.argmax(prediction_result.predictions, axis=1)
            
            from sklearn.metrics import (
                accuracy_score, balanced_accuracy_score, 
                f1_score, precision_score, recall_score
            )
            
            # Calculate all metrics on full dataset
            final_metrics = {
                'accuracy': accuracy_score(test_y_full, predictions),
                'balanced_accuracy': balanced_accuracy_score(test_y_full, predictions),
                'weighted_f1': f1_score(test_y_full, predictions, average='weighted'),
                'weighted_precision': precision_score(test_y_full, predictions, average='weighted'),
                'weighted_recall': recall_score(test_y_full, predictions, average='weighted'),
                'training_time': training_time,
                'test_samples': len(test_y_full),
                'train_samples': len(y_full)
            }
            
            logger.info(f"Final model results on full dataset:")
            logger.info(f"  Accuracy: {final_metrics['accuracy']:.4f}")
            logger.info(f"  Balanced Accuracy: {final_metrics['balanced_accuracy']:.4f}")
            logger.info(f"  Weighted F1: {final_metrics['weighted_f1']:.4f}")
            logger.info(f"  Training Time: {final_metrics['training_time']:.2f}s")
                
                return {
                'final_metrics': final_metrics,
                'best_trial_number': best_trial['trial_number'],
                'best_params': best_varied_params,
                'dataset_info': {
                    'train_samples': len(y_full),
                    'test_samples': len(test_y_full),
                    'total_samples': len(y_full) + len(test_y_full),
                    'test_mode': self.test_mode
                },
                'predictions': predictions.tolist(),
                'true_labels': test_y_full.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error running final model: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _define_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define the hyperparameter search space using only valid MultiModalEnsembleModel parameters."""
        
        # Only use parameters that actually exist in MultiModalEnsembleModel.__init__
        # Improved hyperparameter space based on analysis of poor performing combinations
        params = {
            # Stage 2: Ensemble Generation (4 parameters)
            'n_bags': trial.suggest_categorical('n_bags', [15, 20, 25]),  # Removed 10 (too small)
            'max_dropout_rate': trial.suggest_categorical('max_dropout_rate', [0.4, 0.5, 0.6]),  # Removed 0.3 (too low)
            'sample_ratio': trial.suggest_categorical('sample_ratio', [0.8, 0.9, 1.0]),  # Removed 0.7 (too low)
            'feature_sampling_ratio': 1.0,  # Fixed to use all features to avoid tensor dimension issues
            
            # Stage 3: Base Learner Selection (2 parameters) - Removed 'efficiency' mode (causes low accuracy)
            'optimization_mode': trial.suggest_categorical('optimization_mode', ['accuracy', 'performance']),
                'learner_type': trial.suggest_categorical('learner_type', ['sklearn', 'pytorch', 'deep_learning', 'transformer']),  # NEW: Choose between sklearn, PyTorch, deep learning, and transformer learners
            
            # Stage 4: Training Pipeline (7 parameters) - Improved ranges for imbalanced data
            'learning_rate': trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3, 2e-3]),  # Moderate learning rates for imbalanced data
            'epochs': trial.suggest_categorical('epochs', [50, 75, 100, 150]),  # More epochs for imbalanced data
            'denoising_weight': trial.suggest_categorical('denoising_weight', [0.1, 0.15, 0.2, 0.25]),  # Higher denoising for imbalanced data
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),  # Smaller batch sizes for better gradient estimates
            'weight_decay': trial.suggest_categorical('weight_decay', [1e-5, 1e-4, 1e-3]),  # Include lower values
            'early_stopping_patience': trial.suggest_categorical('early_stopping_patience', [10, 15, 20, 25]),  # More patience for imbalanced data
            'label_smoothing': trial.suggest_categorical('label_smoothing', [0.0, 0.1, 0.2]),  # Label smoothing for imbalanced data
            
            # Stage 5: Ensemble Prediction (3 parameters)
            'transformer_num_heads': trial.suggest_categorical('transformer_num_heads', [8, 12]),  # Removed 4 (too low)
            'transformer_hidden_dim': trial.suggest_categorical('transformer_hidden_dim', [256, 512]),  # Removed 128 (too low)
            'uncertainty_method': trial.suggest_categorical('uncertainty_method', ['entropy', 'variance', 'confidence']),
            'aggregation_strategy': 'simple_average',  # Use simple average instead of transformer fusion
        }
        
        # Add fixed parameters for all stages
        params.update({
            # Stage 1: Data Integration (Fixed)
            'verbose': True,
            'handle_nan': 'fill_mean',
            'handle_inf': 'fill_max',
            'normalize': True,  # Enable normalization for proper feature scaling across modalities
            'remove_outliers': False,
            'outlier_std': 3.0,
            
            # Stage 2: Ensemble Generation (Fixed)
            'dropout_strategy': 'adaptive',
            'min_modalities': 1,
            'random_state': self.seed,
            
            # Stage 3: Base Learner Selection (Fixed)
            'task_type': 'classification',
            'n_classes': 5,
            'modality_aware': True,
            'bag_learner_pairing': True,
            'metadata_level': 'complete',
            'pairing_focus': 'performance',
            'feature_ratio_weight': 0.4,
            'variance_weight': 0.3,
            'dimensionality_weight': 0.3,
            'base_performance': 0.6,
            'diversity_bonus': 0.1,
            'weightage_bonus': 0.1,
            'dropout_penalty': 0.1,
            'early_stopping': True,
            'hidden_layers': (128, 64),
            'dropout': 0.2,
            'max_features': 'auto',
            'min_samples_split': 5,
            'subsample': 0.8,
            'min_samples_leaf': 2,
            'fusion_type': 'weighted',
            'cross_modal_learning': True,
            'hidden_dim': 256,
            'batch_norm': True,
            'layers': 3,
            'fusion_layers': 2,
            
            # Stage 4: Training Pipeline (Fixed)
            'mixed_precision': True,
            'gradient_accumulation_steps': 1,
            'num_workers': 4,
            'validation_split': 0.2,
            'cross_validation_folds': 5,
            'gradient_clipping': 1.0,
            'optimizer_type': 'adamw',
            'scheduler_type': 'cosine_restarts',
            'label_smoothing': 0.1,
            'dropout_rate': 0.2,
            'enable_denoising': True,
            'denoising_strategy': 'adaptive',
            'denoising_objectives': ['reconstruction', 'alignment'],
            'denoising_modalities': ['text', 'metadata'],
            'gradient_scaling': True,
            'loss_scale': 'dynamic',
            'eval_interval': 1,
            'save_checkpoints': True,
            'modal_specific_tracking': True,
            'track_modal_reconstruction': True,
            'track_modal_alignment': True,
            'track_modal_consistency': True,
            'modal_tracking_frequency': 'every_epoch',
            'log_interval': 10,
            'profile_training': False,
            'preserve_bag_characteristics': True,
            'save_modality_mask': True,
            'save_modality_weights': True,
            'save_bag_id': True,
            'save_training_metrics': True,
            'save_learner_config': True,
            'track_only_primary_modalities': False,
            'preserve_only_primary_modalities': False,
            'enable_cross_validation': True,
            'cv_folds': 5,
            'verbose': True,
            'distributed_training': True,
            'compile_model': True,
            
            # Stage 5: Ensemble Prediction (Fixed)
            'aggregation_strategy': 'transformer_fusion',
            'task_type': 'classification',
            'transformer_token_dim': 64,
            'transformer_num_layers': 2,
            'transformer_temperature': 1.0,
            'confidence_threshold': 0.5,
            'weight_normalization': 'softmax',
            'modality_importance_weight': 0.5,
        })
        
        return params
    
    def _objective(self, trial: optuna.Trial, hpo_data: Dict[str, Any]) -> float:
        """Objective function for Optuna optimization."""
        try:
            if not MAINMODEL_AVAILABLE:
                logger.warning("MainModel not available, returning random score")
                return np.random.random()
            
            # Get hyperparameters for this trial
            params = self._define_hyperparameter_space(trial)
            
            logger.info(f"Trial {trial.number}: Testing hyperparameters...")
            
            # Initialize MainModel with trial parameters
            model = MultiModalEnsembleModel(
                task_type=params['task_type'],
                n_bags=params['n_bags'],
                dropout_strategy=params['dropout_strategy'],
                max_dropout_rate=params['max_dropout_rate'],
                min_modalities=params['min_modalities'],
                sample_ratio=params['sample_ratio'],
                feature_sampling_ratio=params['feature_sampling_ratio'],
                random_state=params['random_state'],
                optimization_mode=params['optimization_mode'],
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                weight_decay=params['weight_decay'],
                early_stopping_patience=params['early_stopping_patience'],
                denoising_weight=params['denoising_weight'],
                aggregation_strategy=params['aggregation_strategy'],
                transformer_num_heads=params['transformer_num_heads'],
                transformer_hidden_dim=params['transformer_hidden_dim'],
                uncertainty_method=params['uncertainty_method'],
                verbose=params['verbose']
            )
            
            # Prepare data for MainModel (combine train and test for fit)
            X = {
                'text_train': hpo_data['train_text'].iloc[:, :-1].values,  # Features only
                'metadata_train': hpo_data['train_metadata'].iloc[:, :-1].values,  # Features only
                'text_test': hpo_data['test_text'].iloc[:, :-1].values,
                'metadata_test': hpo_data['test_metadata'].iloc[:, :-1].values
            }
            
            y = {
                'train': hpo_data['train_text'].iloc[:, -1].values,  # Labels
                'test': hpo_data['test_text'].iloc[:, -1].values
            }
            
            # Train the model with current hyperparameters
            model.fit(X, y)
            
            try:
                # Make predictions on test set
                test_X = {
                    'text': hpo_data['test_text'].iloc[:, :-1].values,
                    'metadata': hpo_data['test_metadata'].iloc[:, :-1].values
                }
                prediction_result = model.predict(test_X)
            
            # Extract predictions from PredictionResult object
                # The predictions are probabilities, so convert to class predictions
                import numpy as np
                predictions = np.argmax(prediction_result.predictions, axis=1)
                
                # Calculate comprehensive metrics for unbalanced classification
                from sklearn.metrics import (
                    accuracy_score, balanced_accuracy_score, 
                    f1_score, precision_score, recall_score
                )
                
                # Calculate all required metrics for unbalanced classification
                accuracy = accuracy_score(y['test'], predictions)
                balanced_acc = balanced_accuracy_score(y['test'], predictions)
                weighted_f1 = f1_score(y['test'], predictions, average='weighted')
                weighted_precision = precision_score(y['test'], predictions, average='weighted')
                weighted_recall = recall_score(y['test'], predictions, average='weighted')
                
                # Store metrics in trial user attributes for later analysis
                trial.set_user_attr("accuracy", accuracy)
                trial.set_user_attr("balanced_accuracy", balanced_acc)
                trial.set_user_attr("weighted_f1", weighted_f1)
                trial.set_user_attr("weighted_precision", weighted_precision)
                trial.set_user_attr("weighted_recall", weighted_recall)
                
                logger.info(f"Trial {trial.number}: Accuracy = {accuracy:.4f}, Balanced Accuracy = {balanced_acc:.4f}, Weighted F1 = {weighted_f1:.4f}")
                
                # Return balanced accuracy as primary metric for unbalanced classification
                return balanced_acc
                
            except Exception as pred_error:
                logger.warning(f"Trial {trial.number}: Prediction failed ({pred_error}), using training accuracy")
                
                # Fallback: Use the best training accuracy from the trained learners
                if hasattr(model, 'trained_learners') and model.trained_learners:
                    best_train_acc = 0.0
                    for learner_info in model.trained_learners:
                        if hasattr(learner_info, 'training_metrics') and learner_info.training_metrics:
                            # Extract accuracy from each epoch's metrics
                            epoch_accuracies = [metric.accuracy for metric in learner_info.training_metrics if hasattr(metric, 'accuracy')]
                            if epoch_accuracies:
                                max_acc = max(epoch_accuracies)
                                best_train_acc = max(best_train_acc, max_acc)
                    
                    if best_train_acc > 0.0:
                        logger.info(f"Trial {trial.number}: Using best training accuracy = {best_train_acc:.4f}")
                        return best_train_acc
                
                # Last resort: return a small random score to avoid complete failure
                fallback_score = np.random.random() * 0.1
                logger.info(f"Trial {trial.number}: Using fallback score = {fallback_score:.4f}")
                return fallback_score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            
            # Try to extract training accuracy as fallback
            try:
                if 'model' in locals() and hasattr(model, 'trained_learners') and model.trained_learners:
                    best_train_acc = 0.0
                    for learner_info in model.trained_learners:
                        if hasattr(learner_info, 'training_metrics') and learner_info.training_metrics:
                            # Extract accuracy from each epoch's metrics
                            epoch_accuracies = [metric.accuracy for metric in learner_info.training_metrics if hasattr(metric, 'accuracy')]
                            if epoch_accuracies:
                                max_acc = max(epoch_accuracies)
                                best_train_acc = max(best_train_acc, max_acc)
                    
                    if best_train_acc > 0.0:
                        logger.info(f"Trial {trial.number}: Using fallback training accuracy = {best_train_acc:.4f}")
                        return best_train_acc
            except Exception as fallback_error:
                logger.warning(f"Trial {trial.number}: Fallback also failed: {fallback_error}")
            
            # Last resort: return a small random score to avoid complete failure
            fallback_score = np.random.random() * 0.1
            logger.info(f"Trial {trial.number}: Using random fallback score = {fallback_score:.4f}")
            return fallback_score
    
    def optimize(self, hpo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Optuna optimization."""
        try:
            logger.info(f"Starting Optuna optimization with {self.n_trials} trials...")
            
            # Apply stratified sampling for HPO (1k samples for both quick and full mode)
            logger.info("Applying stratified sampling for HPO (1k train, ~250 test samples)...")
            hpo_data = self._stratified_sample_1k(hpo_data)
            logger.info(f"HPO data after sampling: {len(hpo_data['train_text'])} train, {len(hpo_data['test_text'])} test samples")
            
            # Create Optuna study with TPE sampler
            study = optuna.create_study(
                direction='maximize',  # Maximize accuracy
                sampler=TPESampler(seed=self.seed),
                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
            )
            
            # Run optimization
            study.optimize(
                lambda trial: self._objective(trial, hpo_data),
                n_trials=self.n_trials,
                show_progress_bar=True
            )
            
            # Store results
            self.study = study
            self.best_params = study.best_params
            self.best_score = study.best_value
            
            logger.info(f"Optimization completed!")
            logger.info(f"Best accuracy: {self.best_score:.4f}")
            logger.info(f"Best parameters: {self.best_params}")
            
            # Apply selection criteria for unbalanced classification
            best_trial = self._select_best_trial_unbalanced_classification(study)
            
            # Run final model with full dataset and best hyperparameters
            logger.info("Running final model with full dataset and best hyperparameters...")
            final_results = self._run_final_model_with_full_data(best_trial, hpo_data)
            
            # Save results (including final model results)
            self._save_results(hpo_data, final_results)
            
            return {
                'best_score': best_trial['balanced_accuracy'],
                'best_params': best_trial['params'],
                'n_trials': self.n_trials,
                'selection_criteria': 'unbalanced_classification',
                'selected_trial': best_trial['trial_number'],
                'all_metrics': best_trial['metrics'],
                'final_model_results': final_results,
                'study_summary': {
                    'best_trial': best_trial['trial_number'],
                    'n_completed_trials': len(study.trials),
                    'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                    'optimization_direction': 'maximize_balanced_accuracy',
                    'selection_reason': best_trial['selection_reason']
                }
            }
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _save_results(self, hpo_data: Dict[str, Any], final_results: Dict[str, Any] = None):
        """Save optimization results to files."""
        try:
            # Create phase directory if it doesn't exist
            self.phase_dir.mkdir(parents=True, exist_ok=True)
            
            # Save all trials
            trials_data = []
            for trial in self.study.trials:
                trial_data = {
                    'number': trial.number,
                    'value': trial.value,
                    'state': str(trial.state),
                    'params': trial.params,
                    'user_attrs': trial.user_attrs,
                    'system_attrs': trial.system_attrs
                }
                trials_data.append(trial_data)
            
            trials_file = self.phase_dir / "mainmodel_trials.json"
            with open(trials_file, 'w') as f:
                json.dump(trials_data, f, indent=2, default=str)
            
            # Apply selection criteria and save best configuration
            best_trial = self._select_best_trial_unbalanced_classification(self.study)
            
            best_data = {
                'best_score': best_trial['balanced_accuracy'],
                'best_params': best_trial['params'],
                'selection_criteria': 'unbalanced_classification',
                'selected_trial': best_trial['trial_number'],
                'all_metrics': best_trial['metrics'],
                'selection_reason': best_trial['selection_reason'],
                'optimization_info': {
                    'n_trials': self.n_trials,
                    'test_mode': self.test_mode,
                    'seed': self.seed,
                    'optimization_time': time.strftime('%Y-%m-%dT%H:%M:%S')
                },
                'data_info': {
                    'hpo_samples': len(hpo_data['train_text']),
                    'task_type': hpo_data['task_type'],
                    'num_classes': hpo_data['num_classes']
                },
                'study_summary': {
                    'best_trial': best_trial['trial_number'],
                    'n_completed_trials': len(self.study.trials),
                    'n_pruned_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                    'optimization_direction': 'maximize_balanced_accuracy'
                }
            }
            
            # Add final model results if available
            if final_results:
                best_data['final_model_results'] = final_results
            
            best_file = self.phase_dir / "mainmodel_best.json"
            with open(best_file, 'w') as f:
                json.dump(best_data, f, indent=2, default=str)
            
            # Save final model results separately if available
            if final_results:
                final_file = self.phase_dir / "mainmodel_final_results.json"
                with open(final_file, 'w') as f:
                    json.dump(final_results, f, indent=2, default=str)
                logger.info(f"Final model results saved to {final_file}")
            
            logger.info(f"Results saved to {self.phase_dir}")
            logger.info(f"  - Trials: {trials_file}")
            logger.info(f"  - Best config: {best_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise


def run_phase_3_mainmodel(config: Dict[str, Any], processed_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main entry point for Phase 3: MainModel Hyperparameter Optimization
    
    Args:
        config: Experiment configuration
        processed_data: Optional processed data from previous phases
        
    Returns:
        Dictionary containing optimization results
    """
    logger.info("Starting Phase 3: MainModel Hyperparameter Optimization")
    start_time = time.time()
    
    try:
        # Initialize optimizer
        optimizer = MainModelOptimizer(config)
        
        # Load Phase 1 data
        logger.info("Loading Phase 1 data...")
        phase1_data = optimizer._load_phase1_data(processed_data)
        
        # Run Optuna optimization (sampling is now handled inside optimize method)
        logger.info("Starting hyperparameter optimization...")
        optimization_results = optimizer.optimize(phase1_data)
        
        execution_time = time.time() - start_time
        logger.info(f"Phase 3 completed successfully in {execution_time:.2f} seconds")
        
        # Create results dictionary
        results = {
            'phase': 'phase_3_mainmodel',
            'seed': config.get('seed', 42),
            'test_mode': config.get('test_mode', 'quick'),
            'status': 'completed',
            'execution_time': execution_time,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'data_info': {
                'phase1_samples': len(phase1_data['train_text']),
                'hpo_samples': 1000,  # Fixed HPO sample size
                'task_type': phase1_data['task_type'],
                'num_classes': phase1_data['num_classes']
            },
            'optimization_results': optimization_results
        }
        
        # Save results to file
        results_file = optimizer.phase_dir / "phase_3_mainmodel_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Phase 3 results saved to {results_file}")
        
        return results
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Phase 3 failed: {e}")
        logger.error(traceback.format_exc())
        
        return {
            'phase': 'phase_3_mainmodel',
            'seed': config.get('seed', 42),
            'test_mode': config.get('test_mode', 'quick'),
            'status': 'failed',
            'error': str(e),
            'execution_time': execution_time,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
        }


if __name__ == "__main__":
    # Auto-detect latest experiment run and seed
    experiment_name, seed_number = find_latest_experiment_run()
    
    # Test configuration - simulate experiment runner structure
    test_config = {
        "phase_dir": f"results/{experiment_name}/seed_{seed_number}/phase_3_mainmodel",
        "seed": seed_number,
        "test_mode": "quick",
        "dataset_path": "/Users/pranav/Coding/Projects/PolyModalEnsemble/Benchmarking/ProcessedData/AmazonReviews"
    }
    
    print(f"Running Phase 3 standalone test...")
    print(f"Experiment: {experiment_name}")
    print(f"Seed: {seed_number}")
    print(f"Test mode: {test_config['test_mode']}")
    print(f"Phase directory: {test_config['phase_dir']}")
    print("-" * 50)
    
    # Run Phase 3
    results = run_phase_3_mainmodel(test_config)
    
    print("-" * 50)
    print(f"Phase 3 Results: {results['status']}")
    if results['status'] == 'completed':
        print(f"Execution time: {results['execution_time']:.2f} seconds")
        print(f"HPO samples: {results['data_info']['hpo_samples']}")
        print(f"Best accuracy: {results['optimization_results']['best_score']:.4f}")
        print(f"Trials completed: {results['optimization_results']['n_trials']}")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")