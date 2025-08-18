#!/usr/bin/env python3
"""
MainModel evaluation module for Amazon Reviews experiments
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error, r2_score
import sys
from pathlib import Path

# Add the main model to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from MainModel.dataIntegration import GenericMultiModalDataLoader
from MainModel.mainModel import MultiModalEnsembleModel

from config import ExperimentConfig
from data_loader import AmazonReviewsDataLoader


class MainModelEvaluator:
    """Evaluator for the MainModel on Amazon Reviews"""
    
    def __init__(self, exp_config: ExperimentConfig, data_loader: AmazonReviewsDataLoader):
        self.exp_config = exp_config
        self.data_loader = data_loader
    
    def run_hyperparameter_search(self) -> Dict[str, Any]:
        """Run comprehensive hyperparameter optimization for MainModel"""
        print("\n🔍 HYPERPARAMETER OPTIMIZATION FOR MAINMODEL")
        print("=" * 60)
        
        # Define hyperparameter search space based on mode
        if self.exp_config.use_small_sample:
            # Quick Run Mode: Reduced hyperparameter space
            param_grid = {
                'n_bags': [3, 5],  # Reduced from [3, 5, 10]
                'dropout_strategy': ['linear', 'exponential'],  # Reduced from ['linear', 'exponential', 'adaptive']
                'epochs': [5, 10, 15]  # Keep same for diversity
            }
            max_trials = self.exp_config.get_hp_trials()  # 13 trials max
        else:
            # Full Run Mode: Complete hyperparameter space
            param_grid = {
                'n_bags': [3, 5, 10],
                'dropout_strategy': ['linear', 'exponential', 'adaptive'],
                'epochs': [5, 10, 15]
            }
            max_trials = self.exp_config.full_hyperparameter_search_trials  # 54 trials
        
        # Generate all combinations
        from itertools import product
        param_combinations = list(product(
            param_grid['n_bags'],
            param_grid['dropout_strategy'],
            param_grid['epochs']
        ))
        
        # Limit trials for quick mode
        if self.exp_config.use_small_sample and len(param_combinations) > max_trials:
            # Take first max_trials combinations for consistent results
            param_combinations = param_combinations[:max_trials]
        
        print(f"🎯 Testing {len(param_combinations)} hyperparameter combinations")
        print(f"📊 Parameter space: {param_grid}")
        if self.exp_config.use_small_sample:
            print(f"⚡ Quick mode: Limited to {max_trials} trials")
        
        best_params = None
        best_score = float('inf')  # Lower MAE is better
        all_trials = []
        
        for i, (n_bags, dropout_strategy, epochs) in enumerate(param_combinations):
            print(f"\n🔧 Trial {i+1}/{len(param_combinations)}: n_bags={n_bags}, dropout={dropout_strategy}, epochs={epochs}")
            
            try:
                # Create configuration for this trial with UNIQUE random seed
                trial_seed = self.exp_config.random_seed + i  # Different seed for each trial
                config = {
                    'n_bags': n_bags,
                    'dropout_strategy': dropout_strategy,
                    'epochs': epochs,
                    'batch_size': self.exp_config.default_batch_size,
                    'random_state': trial_seed  # Use unique seed for this trial
                }
                
                # Run quick evaluation
                result = self._evaluate_config(config, quick=True)
                score = result['star_mae']  # Use MAE as optimization metric
                
                # Track trial
                trial_result = {
                    'trial': i + 1,
                    'n_bags': n_bags,
                    'dropout_strategy': dropout_strategy,
                    'epochs': epochs,
                    'star_mae': score,
                    'accuracy': result['accuracy'],
                    'close_accuracy': result['close_accuracy'],
                    'training_time': result['training_time']
                }
                all_trials.append(trial_result)
                
                # Update best
                if score < best_score:
                    best_score = score
                    best_params = config.copy()
                    print(f"   🏆 New best! MAE: {score:.3f}, Acc: {result['accuracy']:.3f}, ±1 Star: {result['close_accuracy']:.3f}")
                else:
                    print(f"   📊 MAE: {score:.3f}, Acc: {result['accuracy']:.3f}, ±1 Star: {result['close_accuracy']:.3f}")
                
            except Exception as e:
                print(f"   ❌ Trial failed: {str(e)}")
                # Record failed trial
                trial_result = {
                    'trial': i + 1,
                    'n_bags': n_bags,
                    'dropout_strategy': dropout_strategy,
                    'epochs': epochs,
                    'star_mae': 10.0,  # Penalty for failure
                    'accuracy': 0.0,
                    'close_accuracy': 0.0,
                    'training_time': 0.0,
                    'error': str(e)
                }
                all_trials.append(trial_result)
        
        # Hyperparameter search results
        hp_results = {
            'best_params': best_params,
            'best_score': best_score,
            'total_trials': len(param_combinations),
            'successful_trials': len([t for t in all_trials if 'error' not in t]),
            'all_trials': all_trials,
            'optimization_time': sum(t.get('training_time', 0) for t in all_trials),
            'parameter_importance': self._analyze_parameter_importance(all_trials)
        }
        
        print(f"\n🏆 HYPERPARAMETER OPTIMIZATION COMPLETE")
        print(f"✅ Best parameters: {best_params}")
        print(f"📊 Best MAE score: {best_score:.3f}")
        print(f"⏱️ Total optimization time: {hp_results['optimization_time']:.1f}s")
        
        return hp_results
    
    def run_simple_test(self) -> Optional[Dict[str, Any]]:
        """Run MainModel with default parameters"""
        print("\n🚀 MAINMODEL SIMPLE TEST (DEFAULT PARAMETERS)")
        print("=" * 60)
        
        default_config = {
            'n_bags': self.exp_config.default_n_bags,
            'dropout_strategy': 'linear',
            'epochs': self.exp_config.default_epochs,
            'batch_size': self.exp_config.default_batch_size
        }
        
        print(f"📋 Using default configuration: {default_config}")
        
        return self._evaluate_config(default_config, quick=False)
    
    def run_optimized_test(self, best_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run MainModel with optimized parameters"""
        print("\n🚀 MAINMODEL OPTIMIZED TEST")
        print("=" * 60)
        
        if not best_params:
            print("⚠️ No optimal parameters provided, using defaults")
            return self.run_simple_test()
        
        print(f"📋 Using optimized configuration: {best_params}")
        
        result = self._evaluate_config(best_params, quick=False)
        if result:
            result['optimized_params'] = best_params
        
        return result
    
    def _evaluate_config(self, config: Dict[str, Any], quick: bool = False) -> Optional[Dict[str, Any]]:
        """Evaluate MainModel with given configuration"""
        
        try:
            # Create data loader for MainModel
            loader = self._create_mainmodel_loader()
            
            # Create and train model
            start_time = time.time()
            
            model = MultiModalEnsembleModel(
                data_loader=loader,
                n_bags=config['n_bags'],
                dropout_strategy=config['dropout_strategy'],
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                random_state=config.get('random_state', self.exp_config.random_seed)
            )
            
            # Load and integrate data
            model.load_and_integrate_data()
            
            # Train model
            model.fit()
            train_time = time.time() - start_time
            
            # Make predictions
            pred_start = time.time()
            pred_result = model.predict()
            inference_time = time.time() - pred_start
            
            y_pred = pred_result.predictions
            
            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(y_pred)
            
            # Get model insights
            insights = self._extract_model_insights(model, pred_result)
            
            result = {
                'config_used': config,
                'training_time': train_time,
                'inference_time': inference_time,
                'predictions': y_pred,
                'true_labels': self.data_loader.test_labels,
                **metrics,
                **insights
            }
            
            if not quick:
                print(f"✅ MainModel evaluation completed:")
                print(f"   📊 Accuracy: {metrics['accuracy']:.3f}")
                print(f"   ⭐ ±1 Star Accuracy: {metrics['close_accuracy']:.3f}")
                print(f"   📈 Star MAE: {metrics['star_mae']:.3f}")
                print(f"   📉 Star RMSE: {metrics['star_rmse']:.3f}")
                print(f"   🎯 F1-Score: {metrics['f1_score']:.3f}")
                print(f"   📈 R² Score: {metrics['r2_score']:.3f}")
                print(f"   ⏱️ Training time: {train_time:.1f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ MainModel evaluation failed: {str(e)}")
            if not quick:
                import traceback
                traceback.print_exc()
            return None
    
    def _create_mainmodel_loader(self) -> GenericMultiModalDataLoader:
        """Create data loader compatible with MainModel"""
        
        loader = GenericMultiModalDataLoader()
        
        # Add modalities
        loader.add_modality_split('text', 
                                 self.data_loader.train_text, 
                                 self.data_loader.test_text)
        loader.add_modality_split('metadata', 
                                 self.data_loader.train_metadata, 
                                 self.data_loader.test_metadata)
        
        # Add labels
        loader.add_labels_split(self.data_loader.train_labels, 
                               self.data_loader.test_labels)
        
        return loader
    
    def _calculate_comprehensive_metrics(self, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics for Amazon Reviews rating prediction"""
        
        y_true = self.data_loader.test_labels
        
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        # Convert to 1-5 rating scale for interpretable metrics
        pred_ratings = y_pred + 1
        true_ratings = y_true + 1
        
        # Rating-specific metrics
        close_accuracy = np.mean(np.abs(pred_ratings - true_ratings) <= 1)  # Within ±1 star
        star_mae = mean_absolute_error(true_ratings, pred_ratings)
        star_rmse = np.sqrt(np.mean((pred_ratings - true_ratings) ** 2))
        r2 = r2_score(true_ratings, pred_ratings)
        
        # Per-class accuracy
        per_class_acc = {}
        for class_idx in range(self.exp_config.n_classes):
            class_mask = y_true == class_idx
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
                per_class_acc[self.exp_config.class_names[class_idx]] = class_acc
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'close_accuracy': close_accuracy,
            'star_mae': star_mae,
            'star_rmse': star_rmse,
            'r2_score': r2,
            'per_class_accuracy': per_class_acc
        }
    
    def _extract_model_insights(self, model, pred_result) -> Dict[str, Any]:
        """Extract interpretability insights from trained model"""
        
        insights = {}
        
        try:
            # Modality importance (if available)
            if hasattr(pred_result, 'modality_importance'):
                insights['modality_importance'] = pred_result.modality_importance
            
            # Prediction confidence (if available)
            if hasattr(pred_result, 'confidence_scores'):
                insights['confidence_scores'] = pred_result.confidence_scores
                insights['mean_confidence'] = float(np.mean(pred_result.confidence_scores))
            
            # Ensemble information
            insights['ensemble_info'] = {
                'n_bags': model.n_bags,
                'dropout_strategy': model.dropout_strategy,
                'epochs_trained': model.epochs
            }
            
            # Training convergence (if available)
            if hasattr(model, 'training_history'):
                insights['training_convergence'] = model.training_history
            
        except Exception as e:
            print(f"⚠️ Warning: Could not extract some model insights: {e}")
        
        return insights
    
    def _analyze_parameter_importance(self, all_trials: List[Dict]) -> Dict[str, float]:
        """Analyze which hyperparameters matter most"""
        
        # Simple correlation analysis between parameters and performance
        successful_trials = [t for t in all_trials if 'error' not in t and t['star_mae'] < 10.0]
        
        if len(successful_trials) < 3:
            return {}
        
        try:
            import pandas as pd
            
            df = pd.DataFrame(successful_trials)
            
            # Convert categorical variables to numeric
            df['dropout_numeric'] = df['dropout_strategy'].map({
                'linear': 0, 'exponential': 1, 'adaptive': 2
            })
            
            # Calculate correlations with performance (lower MAE is better, so negative correlation is good)
            correlations = {}
            for param in ['n_bags', 'epochs', 'dropout_numeric']:
                if param in df.columns:
                    corr = df[param].corr(-df['star_mae'])  # Negative because lower MAE is better
                    correlations[param.replace('_numeric', '')] = float(corr) if not pd.isna(corr) else 0.0
            
            return correlations
            
        except ImportError:
            # Fallback without pandas
            return {'n_bags': 0.0, 'epochs': 0.0, 'dropout': 0.0}
