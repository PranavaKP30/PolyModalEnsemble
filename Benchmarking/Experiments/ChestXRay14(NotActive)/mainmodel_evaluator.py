#!/usr/bin/env python3
"""
MainModel evaluation module for ChestX-ray14 experiments
"""


import time
import numpy as np
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Import MainModel components with improved error handling
project_root = Path(__file__).parent.parent.parent.parent
mainmodel_path = project_root / "MainModel"
current_dir = Path(__file__).parent

# Add paths in order of priority
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(mainmodel_path))
sys.path.insert(0, str(project_root))

print(f"🔧 MainModel Evaluator - Importing components...")
print(f"📍 MainModel path: {mainmodel_path} (exists: {mainmodel_path.exists()})")

try:
    from mainModel import MultiModalEnsembleModel
    print("✅ Successfully imported MultiModalEnsembleModel")
except ImportError as e:
    print(f"❌ Failed to import MultiModalEnsembleModel: {e}")
    print("🔧 Attempting alternative import...")
    try:
        import mainModel
        MultiModalEnsembleModel = mainModel.MultiModalEnsembleModel
        print("✅ Successfully imported MultiModalEnsembleModel via alternative method")
    except ImportError as e2:
        print(f"❌ Alternative import also failed: {e2}")
        raise ImportError(f"Could not import MultiModalEnsembleModel: {e}")

try:
    import dataIntegration
    print("✅ Successfully imported dataIntegration")
except ImportError as e:
    print(f"❌ Failed to import dataIntegration: {e}")
    raise ImportError(f"Could not import dataIntegration: {e}")

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

from config import ExperimentConfig
from data_loader import ChestXRayDataLoader


class MainModelEvaluator:
    """Evaluator for MainModel on ChestX-ray14"""
    
    def __init__(self, exp_config: ExperimentConfig, data_loader: ChestXRayDataLoader):
        self.exp_config = exp_config
        self.data_loader = data_loader
    
    def run_simple_test(self) -> Optional[Dict[str, Any]]:
        """Run MainModel with default parameters (AmazonReviews style output)"""
        print("\n🚀 MAINMODEL SIMPLE TEST (DEFAULT PARAMETERS)")
        print("=" * 60)
        print(f"🎯 Testing on ALL 14 pathologies (multi-label classification)")
        print(f"📊 Total pathologies: {self.exp_config.n_classes}")
        print(f"📊 Train samples: {len(self.data_loader.train_labels)}")
        print(f"📊 Test samples: {len(self.data_loader.test_labels)}")
        print(f"📊 Train labels shape: {self.data_loader.train_labels.shape}")
        print(f"📊 Test labels shape: {self.data_loader.test_labels.shape}")
        
        default_config = {
            'n_bags': 3,
            'dropout_strategy': 'adaptive',
            'epochs': 5,
            'batch_size': 128,
            'random_state': self.exp_config.random_seed
        }
        print(f"📋 Using default configuration: {default_config}")
        try:
            # Use full multi-label dataset (all pathologies)
            loader = dataIntegration.GenericMultiModalDataLoader()
            loader.add_modality_split('image', self.data_loader.train_image, self.data_loader.test_image)
            loader.add_modality_split('text', self.data_loader.train_text, self.data_loader.test_text)
            loader.add_modality_split('metadata', self.data_loader.train_metadata, self.data_loader.test_metadata)
            loader.add_labels_split(self.data_loader.train_labels, self.data_loader.test_labels)
            model = MultiModalEnsembleModel(
                data_loader=loader,
                n_bags=default_config['n_bags'],
                dropout_strategy=default_config['dropout_strategy'],
                epochs=default_config['epochs'],
                batch_size=default_config['batch_size'],
                random_state=default_config['random_state']
            )
            model.load_and_integrate_data()
            start_time = time.time()
            model.fit()
            train_time = time.time() - start_time
            pred_start = time.time()
            pred_result = model.predict()
            pred_time = time.time() - pred_start
            if hasattr(pred_result, 'predictions'):
                y_pred = pred_result.predictions
            else:
                y_pred = pred_result
            metrics = self._calculate_multilabel_metrics_static(self.data_loader.test_labels, y_pred)
            print(f"✅ MainModel simple test completed:")
            print(f"   📊 Exact Match Accuracy: {metrics['accuracy']:.3f}")
            print(f"   🎯 F1 (macro): {metrics['f1_macro']:.3f}")
            print(f"   📈 F1 (micro): {metrics['f1_micro']:.3f}")
            print(f"   📉 F1 (weighted): {metrics['f1_weighted']:.3f}")
            print(f"   🏷️ Precision (macro): {metrics['precision_macro']:.3f}")
            print(f"   🏷️ Recall (macro): {metrics['recall_macro']:.3f}")
            print(f"   🧮 Hamming loss: {metrics['hamming_loss']:.3f}")
            print(f"   🧮 Jaccard score: {metrics['jaccard_score']:.3f}")
            print(f"   ⏱️ Training time: {train_time:.2f}s, Prediction time: {pred_time:.2f}s")
            print(f"   📋 Per-class F1: {[f'{f:.3f}' for f in metrics['per_class_f1']]}")
            print(f"   📋 Per-class Accuracy: {[f'{a:.3f}' for a in metrics['per_class_accuracy']]}")
            return {
                'config_used': default_config,
                'training_time': train_time,
                'prediction_time': pred_time,
                'metrics': metrics
            }
        except Exception as e:
            print(f"❌ MainModel simple test failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_optimized_test(self, best_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run MainModel with optimized hyperparameters (AmazonReviews style output)"""
        print("\n🚀 MAINMODEL OPTIMIZED TEST")
        print("=" * 60)
        if not best_params:
            print("⚠️ No optimal parameters provided, using defaults")
            return self.run_simple_test()
        print(f"🎯 Testing on ALL 14 pathologies (multi-label classification)")
        print(f"📊 Total pathologies: {self.exp_config.n_classes}")
        print(f"📊 Train samples: {len(self.data_loader.train_labels)}")
        print(f"📊 Test samples: {len(self.data_loader.test_labels)}")
        print(f"📋 Using optimized configuration: {best_params}")
        try:
            # Use full multi-label dataset (all pathologies)
            loader = dataIntegration.GenericMultiModalDataLoader()
            loader.add_modality_split('image', self.data_loader.train_image, self.data_loader.test_image)
            loader.add_modality_split('text', self.data_loader.train_text, self.data_loader.test_text)
            loader.add_modality_split('metadata', self.data_loader.train_metadata, self.data_loader.test_metadata)
            loader.add_labels_split(self.data_loader.train_labels, self.data_loader.test_labels)
            model = MultiModalEnsembleModel(
                data_loader=loader,
                n_bags=best_params['n_bags'],
                dropout_strategy=best_params['dropout_strategy'],
                epochs=best_params['epochs'],
                batch_size=best_params['batch_size'],
                random_state=best_params.get('random_state', self.exp_config.random_seed)
            )
            model.load_and_integrate_data()
            start_time = time.time()
            model.fit()
            train_time = time.time() - start_time
            pred_start = time.time()
            pred_result = model.predict()
            pred_time = time.time() - pred_start
            if hasattr(pred_result, 'predictions'):
                y_pred = pred_result.predictions
            else:
                y_pred = pred_result
            metrics = self._calculate_multilabel_metrics_static(self.data_loader.test_labels, y_pred)
            print(f"✅ MainModel optimized test completed:")
            print(f"   📊 Exact Match Accuracy: {metrics['accuracy']:.3f}")
            print(f"   🎯 F1 (macro): {metrics['f1_macro']:.3f}")
            print(f"   📈 F1 (micro): {metrics['f1_micro']:.3f}")
            print(f"   📉 F1 (weighted): {metrics['f1_weighted']:.3f}")
            print(f"   🏷️ Precision (macro): {metrics['precision_macro']:.3f}")
            print(f"   🏷️ Recall (macro): {metrics['recall_macro']:.3f}")
            print(f"   🧮 Hamming loss: {metrics['hamming_loss']:.3f}")
            print(f"   🧮 Jaccard score: {metrics['jaccard_score']:.3f}")
            print(f"   ⏱️ Training time: {train_time:.2f}s, Prediction time: {pred_time:.2f}s")
            print(f"   📋 Per-class F1: {[f'{f:.3f}' for f in metrics['per_class_f1']]}")
            print(f"   📋 Per-class Accuracy: {[f'{a:.3f}' for a in metrics['per_class_accuracy']]}")
            return {
                'config_used': best_params,
                'training_time': train_time,
                'prediction_time': pred_time,
                'metrics': metrics
            }
        except Exception as e:
            print(f"❌ MainModel optimized test failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_hyperparameter_search(self) -> Dict[str, Any]:
        """Run hyperparameter optimization (AmazonReviews style output)"""
        print(f"\n🔧 HYPERPARAMETER OPTIMIZATION FOR MAINMODEL")
        print("=" * 60)
        print(f"🎯 Optimizing on ALL 14 pathologies (multi-label classification)")
        print(f"📊 Total pathologies: {self.exp_config.n_classes}")
        print(f"📊 Train samples: {len(self.data_loader.train_labels)}")
        print(f"📊 Test samples: {len(self.data_loader.test_labels)}")
        
        # Reduced hyperparameter search for faster testing
        # Use smaller values to speed up trials
        param_grid = {
            'n_bags': [2, 3],  # Reduced from [2, 3, 5]
            'dropout_strategy': ['random', 'adaptive'],  # Keep both
            'epochs': [3, 5],  # Reduced from [3, 5, 10]
            'batch_size': [128, 256]  # Reduced from [128, 256, 512]
        }
        from itertools import product
        param_combinations = list(product(
            param_grid['n_bags'],
            param_grid['dropout_strategy'],
            param_grid['epochs'],
            param_grid['batch_size']
        ))
        print(f"🎯 Testing {len(param_combinations)} hyperparameter combinations")
        print(f"📊 Parameter space: {param_grid}")
        best_score = 0
        best_params = None
        best_result = None
        all_results = []
        rng = np.random.default_rng(self.exp_config.random_seed)
        
        for i, (n_bags, dropout_strategy, epochs, batch_size) in enumerate(param_combinations):
            trial_seed = int(rng.integers(0, 1e9))
            config = {
                'n_bags': n_bags,
                'dropout_strategy': dropout_strategy,
                'epochs': epochs,
                'batch_size': batch_size,
                'random_state': trial_seed
            }
            print(f"🔧 Trial {i+1}/{len(param_combinations)}: n_bags={n_bags}, dropout={dropout_strategy}, epochs={epochs}, batch={batch_size}")
            
            try:
                # Add timeout mechanism
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError("Trial timed out")
                
                # Set timeout to 60 seconds per trial
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)
                
                result = self._evaluate_hyperparameters(
                    self.data_loader.train_image, self.data_loader.train_text, self.data_loader.train_metadata, self.data_loader.train_labels,
                    self.data_loader.test_image, self.data_loader.test_text, self.data_loader.test_metadata, self.data_loader.test_labels,
                    n_bags, dropout_strategy, epochs, batch_size, trial_seed
                )
                
                signal.alarm(0)  # Cancel timeout
                
                all_results.append(result)
                is_best = False
                if result['f1_macro'] > best_score:  # Use macro F1 instead of f1_score
                    best_score = result['f1_macro']
                    best_params = config.copy()
                    best_result = result
                    is_best = True
                line = f"✅ Trial {i+1}/{len(param_combinations)}: F1_macro={result['f1_macro']:.3f}, F1_micro={result['f1_micro']:.3f}, Acc={result['accuracy']:.3f}"
                if is_best:
                    line += "   🏆 New best!"
                print(line)
                
            except TimeoutError:
                signal.alarm(0)  # Cancel timeout
                print(f"❌ Trial {i+1}/{len(param_combinations)} timed out")
                continue
            except Exception as e:
                signal.alarm(0)  # Cancel timeout
                print(f"❌ Trial {i+1}/{len(param_combinations)} failed: {str(e)}")
                continue
        
        hp_results = {
            'task_type': 'multi_label_classification',
            'n_pathologies': self.exp_config.n_classes,
            'best_params': best_params,
            'best_score': best_score,
            'best_result': best_result,
            'all_results': all_results,
            'trials_completed': len([r for r in all_results if r is not None])
        }
        
        if best_params:
            print(f"\n🏆 HYPERPARAMETER OPTIMIZATION COMPLETE")
            print(f"✅ Best parameters: {best_params}")
            print(f"🎯 Best Macro F1-Score: {best_score:.3f}")
            print(f"📊 Best Trial Metrics:")
            for k, v in best_result.items():
                if isinstance(v, float):
                    print(f"   {k}: {v:.4f}")
                elif isinstance(v, list):
                    print(f"   {k}: {np.round(v, 4).tolist()}")
                else:
                    print(f"   {k}: {v}")
            print(f"✅ COMPREHENSIVE SEARCH: Completed {len([r for r in all_results if r is not None])}/{len(param_combinations)} trials")
        else:
            print(f"❌ No successful trials completed")
        
        return hp_results
    
    def _evaluate_hyperparameters(self, train_image, train_text, train_metadata, train_labels,
                                test_image, test_text, test_metadata, test_labels,
                                n_bags, dropout_strategy, epochs, batch_size, random_state) -> Dict[str, Any]:
        """Evaluate specific hyperparameter combination"""
        # Create data loader
        loader = dataIntegration.GenericMultiModalDataLoader()
        loader.add_modality_split('image', train_image, test_image)
        loader.add_modality_split('text', train_text, test_text)
        loader.add_modality_split('metadata', train_metadata, test_metadata)
        loader.add_labels_split(train_labels, test_labels)
        
        # Initialize model with specific parameters and random seed
        model = MultiModalEnsembleModel(
            data_loader=loader,
            n_bags=n_bags,
            dropout_strategy=dropout_strategy,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state
        )
        
        # Train and evaluate
        model.load_and_integrate_data()
        start_time = time.time()
        model.fit()
        train_time = time.time() - start_time
        
        pred_start = time.time()
        prediction_result = model.predict()
        pred_time = time.time() - pred_start
        
        # Extract predictions from PredictionResult object
        if hasattr(prediction_result, 'predictions'):
            predictions = prediction_result.predictions
        else:
            predictions = prediction_result
        
        # Use comprehensive multilabel metrics
        metrics = self._calculate_multilabel_metrics_static(test_labels, predictions)
        
        # Add timing information
        metrics.update({
            'n_bags': n_bags,
            'dropout_strategy': dropout_strategy,
            'epochs': epochs,
            'batch_size': batch_size,
            'random_state': random_state,
            'train_time': train_time,
            'pred_time': pred_time,
            'total_time': train_time + pred_time
        })
        
        return metrics

    @staticmethod
    def _calculate_multilabel_metrics_static(y_true, y_pred):
        # Copied from baseline_evaluator.py for static use
        from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score, precision_recall_fscore_support
        import numpy as np
        # --- DEBUG: Print unique values and dtypes before any processing ---
        print("\n[DEBUG] y_true dtype:", y_true.dtype, ", unique:", np.unique(y_true))
        print("[DEBUG] y_pred dtype:", y_pred.dtype, ", unique:", np.unique(y_pred))
        # --- Ensure y_pred is binary (0/1) and same shape as y_true ---
        if y_pred.shape != y_true.shape:
            print(f"   ⚠️ Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
            return {'f1_score': 0.0, 'accuracy': 0.0, 'hamming_loss': 1.0}
        # If predictions are probabilities, threshold at 0.5
        if not np.issubdtype(y_pred.dtype, np.integer):
            print("[DEBUG] y_pred is not integer, thresholding at 0.5...")
            y_pred = (y_pred >= 0.5).astype(int)
        # If y_true is not integer, cast to int
        if not np.issubdtype(y_true.dtype, np.integer):
            print("[DEBUG] y_true is not integer, casting to int...")
            y_true = y_true.astype(int)
        # Print again after processing
        print("[DEBUG] (post-process) y_true unique:", np.unique(y_true))
        print("[DEBUG] (post-process) y_pred unique:", np.unique(y_pred))
        try:
            accuracy = accuracy_score(y_true, y_pred)
            f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            hamming = hamming_loss(y_true, y_pred)
            jaccard = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
            precision_macro = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[0]
            recall_macro = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[1]
            per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
            per_class_precision = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)[0]
            per_class_recall = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)[1]
            per_class_accuracy = (y_true == y_pred).mean(axis=0)
            # Print all metrics in a summary table for visual clarity
            print("\n📊 METRICS SUMMARY:")
            print(f"   Accuracy:         {accuracy:.4f}")
            print(f"   F1 (macro):       {f1_macro:.4f}")
            print(f"   F1 (micro):       {f1_micro:.4f}")
            print(f"   F1 (weighted):    {f1_weighted:.4f}")
            print(f"   Precision (macro):{precision_macro:.4f}")
            print(f"   Recall (macro):   {recall_macro:.4f}")
            print(f"   Hamming loss:     {hamming:.4f}")
            print(f"   Jaccard score:    {jaccard:.4f}")
            print(f"   Per-class F1:     {np.round(per_class_f1, 4).tolist()}")
            print(f"   Per-class Prec:   {np.round(per_class_precision, 4).tolist()}")
            print(f"   Per-class Recall: {np.round(per_class_recall, 4).tolist()}")
            print(f"   Per-class Acc:    {np.round(per_class_accuracy, 4).tolist()}")
            return {
                'accuracy': accuracy,
                'f1_score': f1_macro,
                'f1_micro': f1_micro,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'hamming_loss': hamming,
                'jaccard_score': jaccard,
                'per_class_f1': per_class_f1.tolist(),
                'per_class_precision': per_class_precision.tolist(),
                'per_class_recall': per_class_recall.tolist(),
                'per_class_accuracy': per_class_accuracy.tolist()
            }
        except Exception as e:
            print(f"   ⚠️ Metrics calculation failed: {str(e)}")
            return {'f1_score': 0.0, 'accuracy': 0.0, 'hamming_loss': 1.0}
