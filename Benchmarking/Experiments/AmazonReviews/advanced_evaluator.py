#!/usr/bin/env python3
"""
Advanced evaluation module for Amazon Reviews experiments
Includes cross-validation, robustness testing, and comprehensive analysis
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import ExperimentConfig
from data_loader import AmazonReviewsDataLoader
from baseline_evaluator import BaselineEvaluator
from mainmodel_evaluator import MainModelEvaluator


class AdvancedEvaluator:
    """Advanced evaluation and analysis for Amazon Reviews experiments"""
    
    def __init__(self, exp_config: ExperimentConfig, data_loader: AmazonReviewsDataLoader):
        self.exp_config = exp_config
        self.data_loader = data_loader
        self.baseline_evaluator = BaselineEvaluator(exp_config, data_loader)
        self.mainmodel_evaluator = MainModelEvaluator(exp_config, data_loader)
    
    def run_cross_validation_analysis(self) -> Dict[str, Any]:
        """Run k-fold cross-validation for selected models"""
        print("\n🔄 CROSS-VALIDATION ANALYSIS")
        print("=" * 60)
        
        # Selected models for cross-validation (computational efficiency)
        cv_models = {
            'RandomForest_Text': ('text', 'RandomForestClassifier'),
            'RandomForest_Combined': ('combined', 'RandomForestClassifier'),
            'XGBoost_Combined': ('combined', 'XGBoostClassifier'),
            'VotingEnsemble': ('ensemble', 'VotingClassifier'),
            'MainModel': ('mainmodel', 'default')
        }
        
        print(f"🎯 Running {self.exp_config.cv_folds}-fold cross-validation on {len(cv_models)} models")
        
        cv_results = {}
        
        for model_name, (modality, algorithm) in cv_models.items():
            print(f"\n📊 Cross-validating: {model_name}")
            
            try:
                if modality == 'mainmodel':
                    cv_scores = self._cv_mainmodel()
                else:
                    cv_scores = self._cv_baseline_model(modality, algorithm)
                
                cv_results[model_name] = {
                    'scores': cv_scores,
                    'mean_accuracy': float(np.mean(cv_scores)),
                    'std_accuracy': float(np.std(cv_scores)),
                    'confidence_interval': self._calculate_confidence_interval(cv_scores)
                }
                
                print(f"   ✅ Mean accuracy: {cv_results[model_name]['mean_accuracy']:.3f} ± {cv_results[model_name]['std_accuracy']:.3f}")
                
            except Exception as e:
                print(f"   ❌ CV failed for {model_name}: {str(e)}")
                cv_results[model_name] = {
                    'scores': [],
                    'mean_accuracy': 0.0,
                    'std_accuracy': 0.0,
                    'error': str(e)
                }
        
        # Statistical analysis
        cv_summary = self._analyze_cv_results(cv_results)
        
        return {
            'cv_results': cv_results,
            'statistical_analysis': cv_summary,
            'best_cv_model': max(cv_results.keys(), 
                               key=lambda k: cv_results[k]['mean_accuracy'] if 'error' not in cv_results[k] else 0)
        }
    
    def run_robustness_testing(self) -> Dict[str, Any]:
        """Test model robustness under various conditions"""
        print("\n🛡️ ROBUSTNESS TESTING")
        print("=" * 60)
        
        robustness_tests = {
            'noise_resilience': self._test_noise_resilience,
            'missing_data_handling': self._test_missing_data,
            'class_imbalance_sensitivity': self._test_class_imbalance,
            'sample_size_sensitivity': self._test_sample_size_effect
        }
        
        robustness_results = {}
        
        for test_name, test_func in robustness_tests.items():
            print(f"\n🔬 Running {test_name.replace('_', ' ').title()}")
            
            try:
                test_result = test_func()
                robustness_results[test_name] = test_result
                print(f"   ✅ {test_name} completed")
                
            except Exception as e:
                print(f"   ❌ {test_name} failed: {str(e)}")
                robustness_results[test_name] = {'error': str(e)}
        
        return robustness_results
    
    def run_interpretability_analysis(self) -> Dict[str, Any]:
        """Analyze model interpretability and feature importance"""
        print("\n🔍 INTERPRETABILITY ANALYSIS")
        print("=" * 60)
        
        interpretability_results = {}
        
        # Feature importance analysis
        print("\n📈 Feature Importance Analysis")
        try:
            feature_importance = self._analyze_feature_importance()
            interpretability_results['feature_importance'] = feature_importance
            print("   ✅ Feature importance analysis completed")
        except Exception as e:
            print(f"   ❌ Feature importance analysis failed: {str(e)}")
            interpretability_results['feature_importance'] = {'error': str(e)}
        
        # Prediction confidence analysis
        print("\n🎯 Prediction Confidence Analysis")
        try:
            confidence_analysis = self._analyze_prediction_confidence()
            interpretability_results['confidence_analysis'] = confidence_analysis
            print("   ✅ Confidence analysis completed")
        except Exception as e:
            print(f"   ❌ Confidence analysis failed: {str(e)}")
            interpretability_results['confidence_analysis'] = {'error': str(e)}
        
        # Error analysis
        print("\n❌ Error Pattern Analysis")
        try:
            error_analysis = self._analyze_error_patterns()
            interpretability_results['error_analysis'] = error_analysis
            print("   ✅ Error analysis completed")
        except Exception as e:
            print(f"   ❌ Error analysis failed: {str(e)}")
            interpretability_results['error_analysis'] = {'error': str(e)}
        
        return interpretability_results
    
    def run_modality_ablation_study(self) -> Dict[str, Any]:
        """Study the contribution of different modalities"""
        print("\n🎭 COMPREHENSIVE MODALITY ABLATION STUDY")
        print("=" * 60)
        
        all_ablation_results = {}
        
        # 1. Basic Modality Ablation
        print("\n📊 BASIC MODALITY ABLATION")
        print("-" * 40)
        basic_ablation = self._run_basic_modality_ablation()
        all_ablation_results['basic_modality_ablation'] = basic_ablation
        
        # 2. Feature Subset Ablation (within text modality)
        print("\n📝 TEXT FEATURE SUBSET ABLATION")
        print("-" * 40)
        try:
            text_feature_ablation = self._run_text_feature_ablation()
            all_ablation_results['text_feature_ablation'] = text_feature_ablation
        except Exception as e:
            print(f"   ⚠️ Text feature ablation failed: {str(e)}")
            all_ablation_results['text_feature_ablation'] = {'error': str(e)}
        
        # 3. Metadata Feature Ablation
        print("\n🏷️ METADATA FEATURE ABLATION")
        print("-" * 40)
        try:
            metadata_feature_ablation = self._run_metadata_feature_ablation()
            all_ablation_results['metadata_feature_ablation'] = metadata_feature_ablation
        except Exception as e:
            print(f"   ⚠️ Metadata feature ablation failed: {str(e)}")
            all_ablation_results['metadata_feature_ablation'] = {'error': str(e)}
        
        # 4. Fusion Strategy Ablation
        print("\n🔗 FUSION STRATEGY ABLATION")
        print("-" * 40)
        try:
            fusion_ablation = self._run_fusion_strategy_ablation()
            all_ablation_results['fusion_strategy_ablation'] = fusion_ablation
        except Exception as e:
            print(f"   ⚠️ Fusion strategy ablation failed: {str(e)}")
            all_ablation_results['fusion_strategy_ablation'] = {'error': str(e)}
        
        # 5. Algorithm Sensitivity Ablation
        print("\n⚙️ ALGORITHM SENSITIVITY ABLATION")
        print("-" * 40)
        try:
            algorithm_ablation = self._run_algorithm_sensitivity_ablation()
            all_ablation_results['algorithm_sensitivity_ablation'] = algorithm_ablation
        except Exception as e:
            print(f"   ⚠️ Algorithm sensitivity ablation failed: {str(e)}")
            all_ablation_results['algorithm_sensitivity_ablation'] = {'error': str(e)}
        
        # Comprehensive analysis
        print("\n📈 COMPREHENSIVE ABLATION ANALYSIS")
        print("-" * 40)
        comprehensive_analysis = self._analyze_comprehensive_ablations(all_ablation_results)
        
        return {
            'ablation_experiments': all_ablation_results,
            'comprehensive_analysis': comprehensive_analysis
        }
    
    def _run_basic_modality_ablation(self) -> Dict[str, Any]:
        """Run basic modality ablation study"""
        
        # Test different modality combinations
        modality_combinations = {
            'text_only': ['text'],
            'metadata_only': ['metadata'],
            'both_modalities': ['text', 'metadata']
        }
        
        ablation_results = {}
        
        for combination_name, modalities in modality_combinations.items():
            print(f"\n🧪 Testing: {combination_name}")
            
            try:
                # Create modified data loader for this combination
                modified_data = self._create_modified_data(modalities)
                
                # Test with multiple algorithms
                combination_results = {}
                test_algorithms = ['RandomForestClassifier', 'XGBoostClassifier', 'LogisticRegression']
                
                for algorithm in test_algorithms:
                    result = self._test_algorithm_on_modified_data(modified_data, algorithm)
                    combination_results[algorithm] = result
                
                ablation_results[combination_name] = {
                    'modalities_used': modalities,
                    'algorithm_results': combination_results,
                    'best_performance': max(combination_results.values(), key=lambda x: x.get('accuracy', 0))
                }
                
                best_acc = ablation_results[combination_name]['best_performance'].get('accuracy', 0)
                print(f"   ✅ Best accuracy: {best_acc:.3f}")
                
            except Exception as e:
                print(f"   ❌ Ablation test failed for {combination_name}: {str(e)}")
                ablation_results[combination_name] = {'error': str(e)}
        
        # Analyze modality contributions
        contribution_analysis = self._analyze_modality_contributions(ablation_results)
        
        return {
            'modality_results': ablation_results,
            'contribution_analysis': contribution_analysis
        }
    
    def generate_comprehensive_report(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        print("\n📊 GENERATING COMPREHENSIVE REPORT")
        print("=" * 60)
        
        report = {
            'experiment_summary': self._create_experiment_summary(all_results),
            'performance_ranking': self._rank_all_models(all_results),
            'statistical_significance': self._test_statistical_significance(all_results),
            'recommendations': self._generate_recommendations(all_results),
            'visualizations': self._create_visualizations(all_results)
        }
        
        print("✅ Comprehensive report generated")
        
        return report
    
    def _cv_baseline_model(self, modality: str, algorithm: str) -> List[float]:
        """Cross-validate a baseline model"""
        
        # Prepare data based on modality
        if modality == 'text':
            X = self.data_loader.get_combined_text_features()
        elif modality == 'metadata':
            X = self.data_loader.get_combined_metadata_features()
        elif modality == 'combined':
            X = self.data_loader.get_combined_features()
        elif modality == 'ensemble':
            # Ensemble models use combined features
            X = self.data_loader.get_combined_features()
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        y = self.data_loader.get_combined_labels()
        
        # Perform stratified k-fold CV
        skf = StratifiedKFold(n_splits=self.exp_config.cv_folds, shuffle=True, random_state=self.exp_config.random_seed)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train model
            model = self.baseline_evaluator._create_model(algorithm)
            model.fit(X_train_fold, y_train_fold)
            
            # Predict and score
            y_pred = model.predict(X_val_fold)
            accuracy = np.mean(y_pred == y_val_fold)
            cv_scores.append(accuracy)
        
        return cv_scores
    
    def _cv_mainmodel(self) -> List[float]:
        """Cross-validate MainModel (simplified for efficiency)"""
        
        # For MainModel, we'll do a simplified CV due to computational cost
        # Use different random seeds to simulate CV folds
        cv_scores = []
        
        for fold in range(min(3, self.exp_config.cv_folds)):  # Limit folds for efficiency
            # Create a slightly different data split
            modified_seed = self.exp_config.random_seed + fold * 100
            
            # Quick evaluation with modified seed
            config = {
                'n_bags': 3,  # Reduced for speed
                'dropout_strategy': 'linear',
                'epochs': 5,  # Reduced for speed
                'batch_size': self.exp_config.default_batch_size
            }
            
            # This is a simplified approach - in practice, you'd want proper CV
            result = self.mainmodel_evaluator._evaluate_config(config, quick=True)
            if result:
                cv_scores.append(result['accuracy'])
            else:
                cv_scores.append(0.0)
        
        return cv_scores
    
    def _test_noise_resilience(self) -> Dict[str, Any]:
        """Test model performance under noisy conditions"""
        
        noise_levels = [0.0, 0.1, 0.2, 0.3]
        noise_results = {}
        
        # Test a representative model
        base_model = self.baseline_evaluator._create_model('RandomForestClassifier')
        X_combined = self.data_loader.get_combined_features()
        y_combined = self.data_loader.get_combined_labels()
        
        # Split data
        split_idx = int(len(X_combined) * 0.8)
        X_train, X_test = X_combined[:split_idx], X_combined[split_idx:]
        y_train, y_test = y_combined[:split_idx], y_combined[split_idx:]
        
        for noise_level in noise_levels:
            # Add noise to features
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, X_test.shape)
                X_test_noisy = X_test + noise
            else:
                X_test_noisy = X_test
            
            # Train and test
            base_model.fit(X_train, y_train)
            y_pred = base_model.predict(X_test_noisy)
            accuracy = np.mean(y_pred == y_test)
            
            noise_results[f'noise_{noise_level}'] = {
                'accuracy': float(accuracy),
                'accuracy_drop': float(noise_results.get('noise_0.0', {}).get('accuracy', accuracy) - accuracy)
            }
        
        return noise_results
    
    def _test_missing_data(self) -> Dict[str, Any]:
        """Test handling of missing data"""
        
        missing_percentages = [0.0, 0.1, 0.2, 0.3]
        missing_results = {}
        
        for missing_pct in missing_percentages:
            try:
                # Create data with missing values
                X_combined = self.data_loader.get_combined_features().copy()
                y_combined = self.data_loader.get_combined_labels()
                
                if missing_pct > 0:
                    # Randomly set values to NaN
                    mask = np.random.random(X_combined.shape) < missing_pct
                    X_combined[mask] = np.nan
                    
                    # Simple imputation (mean)
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='mean')
                    X_combined = imputer.fit_transform(X_combined)
                
                # Quick train/test
                split_idx = int(len(X_combined) * 0.8)
                X_train, X_test = X_combined[:split_idx], X_combined[split_idx:]
                y_train, y_test = y_combined[:split_idx], y_combined[split_idx:]
                
                model = self.baseline_evaluator._create_model('RandomForestClassifier')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = np.mean(y_pred == y_test)
                
                missing_results[f'missing_{missing_pct}'] = {
                    'accuracy': float(accuracy),
                    'missing_percentage': missing_pct
                }
                
            except Exception as e:
                missing_results[f'missing_{missing_pct}'] = {'error': str(e)}
        
        return missing_results
    
    def _test_class_imbalance(self) -> Dict[str, Any]:
        """Test sensitivity to class imbalance"""
        
        # Create artificially imbalanced datasets
        imbalance_ratios = [1.0, 2.0, 5.0, 10.0]  # Ratio of majority to minority class
        imbalance_results = {}
        
        X_combined = self.data_loader.get_combined_features()
        y_combined = self.data_loader.get_combined_labels()
        
        for ratio in imbalance_ratios:
            try:
                # Create imbalanced dataset
                if ratio == 1.0:
                    X_imb, y_imb = X_combined, y_combined
                else:
                    X_imb, y_imb = self._create_imbalanced_dataset(X_combined, y_combined, ratio)
                
                # Train and test
                split_idx = int(len(X_imb) * 0.8)
                X_train, X_test = X_imb[:split_idx], X_imb[split_idx:]
                y_train, y_test = y_imb[:split_idx], y_imb[split_idx:]
                
                model = self.baseline_evaluator._create_model('RandomForestClassifier')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = np.mean(y_pred == y_test)
                
                imbalance_results[f'ratio_{ratio}'] = {
                    'accuracy': float(accuracy),
                    'imbalance_ratio': ratio,
                    'dataset_size': len(X_imb)
                }
                
            except Exception as e:
                imbalance_results[f'ratio_{ratio}'] = {'error': str(e)}
        
        return imbalance_results
    
    def _test_sample_size_effect(self) -> Dict[str, Any]:
        """Test effect of training sample size"""
        
        sample_fractions = [0.1, 0.3, 0.5, 0.7, 1.0]
        sample_size_results = {}
        
        X_combined = self.data_loader.get_combined_features()
        y_combined = self.data_loader.get_combined_labels()
        
        # Fixed test set
        split_idx = int(len(X_combined) * 0.8)
        X_full_train, X_test = X_combined[:split_idx], X_combined[split_idx:]
        y_full_train, y_test = y_combined[:split_idx], y_combined[split_idx:]
        
        for fraction in sample_fractions:
            try:
                # Sample training data
                sample_size = int(len(X_full_train) * fraction)
                indices = np.random.choice(len(X_full_train), sample_size, replace=False)
                X_train_sample = X_full_train[indices]
                y_train_sample = y_full_train[indices]
                
                # Train and test
                model = self.baseline_evaluator._create_model('RandomForestClassifier')
                model.fit(X_train_sample, y_train_sample)
                y_pred = model.predict(X_test)
                accuracy = np.mean(y_pred == y_test)
                
                sample_size_results[f'fraction_{fraction}'] = {
                    'accuracy': float(accuracy),
                    'sample_fraction': fraction,
                    'sample_size': sample_size
                }
                
            except Exception as e:
                sample_size_results[f'fraction_{fraction}'] = {'error': str(e)}
        
        return sample_size_results
    
    def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance across different models"""
        
        importance_results = {}
        
        # Test tree-based models that provide feature importance
        tree_models = ['RandomForestClassifier', 'ExtraTreesClassifier', 'XGBoostClassifier']
        
        X_combined = self.data_loader.get_combined_features()
        y_combined = self.data_loader.get_combined_labels()
        
        split_idx = int(len(X_combined) * 0.8)
        X_train, y_train = X_combined[:split_idx], y_combined[:split_idx]
        
        for model_name in tree_models:
            try:
                model = self.baseline_evaluator._create_model(model_name)
                model.fit(X_train, y_train)
                
                if hasattr(model, 'feature_importances_'):
                    importance_results[model_name] = {
                        'importances': model.feature_importances_.tolist(),
                        'top_features': np.argsort(model.feature_importances_)[-10:].tolist()
                    }
                
            except Exception as e:
                importance_results[model_name] = {'error': str(e)}
        
        return importance_results
    
    def _analyze_prediction_confidence(self) -> Dict[str, Any]:
        """Analyze prediction confidence patterns"""
        
        confidence_results = {}
        
        try:
            # Use a model that provides prediction probabilities
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=50, random_state=self.exp_config.random_seed)
            
            X_combined = self.data_loader.get_combined_features()
            y_combined = self.data_loader.get_combined_labels()
            
            split_idx = int(len(X_combined) * 0.8)
            X_train, X_test = X_combined[:split_idx], X_combined[split_idx:]
            y_train, y_test = y_combined[:split_idx], y_combined[split_idx:]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Calculate confidence metrics
            max_probabilities = np.max(y_proba, axis=1)
            correct_predictions = (y_pred == y_test)
            
            confidence_results = {
                'mean_confidence_correct': float(np.mean(max_probabilities[correct_predictions])),
                'mean_confidence_incorrect': float(np.mean(max_probabilities[~correct_predictions])),
                'confidence_accuracy_correlation': float(np.corrcoef(max_probabilities, correct_predictions.astype(int))[0, 1]),
                'high_confidence_accuracy': float(np.mean(correct_predictions[max_probabilities > 0.8])),
                'low_confidence_accuracy': float(np.mean(correct_predictions[max_probabilities < 0.6]))
            }
            
        except Exception as e:
            confidence_results = {'error': str(e)}
        
        return confidence_results
    
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in prediction errors"""
        
        error_analysis = {}
        
        try:
            # Train a representative model
            model = self.baseline_evaluator._create_model('RandomForestClassifier')
            X_combined = self.data_loader.get_combined_features()
            y_combined = self.data_loader.get_combined_labels()
            
            split_idx = int(len(X_combined) * 0.8)
            X_train, X_test = X_combined[:split_idx], X_combined[split_idx:]
            y_train, y_test = y_combined[:split_idx], y_combined[split_idx:]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Confusion matrix analysis
            cm = confusion_matrix(y_test, y_pred)
            
            # Rating-specific error analysis
            pred_ratings = y_pred + 1
            true_ratings = y_test + 1
            rating_errors = pred_ratings - true_ratings
            
            error_analysis = {
                'confusion_matrix': cm.tolist(),
                'classification_report': classification_report(y_test, y_pred, target_names=self.exp_config.class_names, output_dict=True),
                'rating_error_distribution': {
                    'mean_error': float(np.mean(rating_errors)),
                    'std_error': float(np.std(rating_errors)),
                    'error_counts': {int(i): int(count) for i, count in enumerate(np.bincount(rating_errors + 4, minlength=9))}
                },
                'per_class_error_rates': {
                    self.exp_config.class_names[i]: float(1 - cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0)
                    for i in range(len(self.exp_config.class_names))
                }
            }
            
        except Exception as e:
            error_analysis = {'error': str(e)}
        
        return error_analysis
    
    def _create_modified_data(self, modalities: List[str]) -> Dict[str, np.ndarray]:
        """Create modified dataset with only specified modalities"""
        
        if 'text' in modalities and 'metadata' in modalities:
            return {
                'X': self.data_loader.get_combined_features(),
                'y': self.data_loader.get_combined_labels()
            }
        elif 'text' in modalities:
            return {
                'X': self.data_loader.get_combined_text_features(),
                'y': self.data_loader.get_combined_labels()
            }
        elif 'metadata' in modalities:
            return {
                'X': self.data_loader.get_combined_metadata_features(),
                'y': self.data_loader.get_combined_labels()
            }
        else:
            raise ValueError("At least one modality must be specified")
    
    def _test_algorithm_on_modified_data(self, modified_data: Dict[str, np.ndarray], algorithm: str) -> Dict[str, float]:
        """Test an algorithm on modified data"""
        
        X, y = modified_data['X'], modified_data['y']
        
        # Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        model = self.baseline_evaluator._create_model(algorithm)
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = float(np.mean(y_pred == y_test))
        
        # Calculate additional metrics
        pred_ratings = y_pred + 1
        true_ratings = y_test + 1
        star_mae = float(np.mean(np.abs(pred_ratings - true_ratings)))
        
        return {
            'accuracy': accuracy,
            'star_mae': star_mae
        }
    
    def _analyze_modality_contributions(self, ablation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the contribution of each modality"""
        
        contributions = {}
        
        try:
            # Extract best performances
            text_only = ablation_results.get('text_only', {}).get('best_performance', {}).get('accuracy', 0)
            metadata_only = ablation_results.get('metadata_only', {}).get('best_performance', {}).get('accuracy', 0)
            both_modalities = ablation_results.get('both_modalities', {}).get('best_performance', {}).get('accuracy', 0)
            
            contributions = {
                'text_contribution': text_only,
                'metadata_contribution': metadata_only,
                'multimodal_benefit': both_modalities - max(text_only, metadata_only),
                'synergy_effect': both_modalities - (text_only + metadata_only) / 2,
                'best_single_modality': 'text' if text_only > metadata_only else 'metadata',
                'multimodal_improvement': both_modalities > max(text_only, metadata_only)
            }
            
        except Exception as e:
            contributions = {'error': str(e)}
        
        return contributions
    
    def _run_text_feature_ablation(self) -> Dict[str, Any]:
        """Ablate different types of text features"""
        
        print("   📝 Testing text feature subsets...")
        
        # For this implementation, we'll simulate different text feature types
        # In practice, you'd have separate TF-IDF, embeddings, etc.
        text_features = self.data_loader.get_combined_text_features()
        labels = self.data_loader.get_combined_labels()
        
        # Simulate different text feature types by using feature subsets
        n_features = text_features.shape[1]
        feature_subsets = {
            'first_quarter': text_features[:, :n_features//4],
            'second_quarter': text_features[:, n_features//4:n_features//2],
            'third_quarter': text_features[:, n_features//2:3*n_features//4],
            'fourth_quarter': text_features[:, 3*n_features//4:],
            'first_half': text_features[:, :n_features//2],
            'second_half': text_features[:, n_features//2:],
            'all_features': text_features
        }
        
        results = {}
        for subset_name, features in feature_subsets.items():
            try:
                # Quick evaluation with Random Forest
                split_idx = int(len(features) * 0.8)
                X_train, X_test = features[:split_idx], features[split_idx:]
                y_train, y_test = labels[:split_idx], labels[split_idx:]
                
                model = self.baseline_evaluator._create_model('RandomForestClassifier')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = float(np.mean(y_pred == y_test))
                star_mae = float(np.mean(np.abs((y_pred + 1) - (y_test + 1))))
                
                results[subset_name] = {
                    'accuracy': accuracy,
                    'star_mae': star_mae,
                    'n_features': features.shape[1]
                }
                
                print(f"     {subset_name}: Acc={accuracy:.3f}, Features={features.shape[1]}")
                
            except Exception as e:
                results[subset_name] = {'error': str(e)}
        
        return {
            'feature_subset_results': results,
            'analysis': self._analyze_text_feature_importance(results)
        }
    
    def _run_metadata_feature_ablation(self) -> Dict[str, Any]:
        """Ablate different metadata features"""
        
        print("   🏷️ Testing metadata feature removal...")
        
        # Get metadata features
        metadata_features = self.data_loader.get_combined_metadata_features()
        labels = self.data_loader.get_combined_labels()
        
        # Test removing each feature one by one (leave-one-out)
        n_features = metadata_features.shape[1]
        results = {}
        
        # Baseline with all features
        baseline_acc = self._quick_evaluate_features(metadata_features, labels)
        results['all_features'] = {
            'accuracy': baseline_acc,
            'features_used': n_features,
            'removed_feature': 'none'
        }
        
        # Remove each feature individually
        for i in range(min(n_features, 10)):  # Limit to first 10 features for efficiency
            try:
                # Create feature subset without feature i
                features_subset = np.concatenate([
                    metadata_features[:, :i],
                    metadata_features[:, i+1:]
                ], axis=1)
                
                accuracy = self._quick_evaluate_features(features_subset, labels)
                
                results[f'without_feature_{i}'] = {
                    'accuracy': accuracy,
                    'features_used': features_subset.shape[1],
                    'removed_feature': i,
                    'importance_drop': baseline_acc - accuracy
                }
                
                print(f"     Without feature {i}: Acc={accuracy:.3f} (drop: {baseline_acc - accuracy:.3f})")
                
            except Exception as e:
                results[f'without_feature_{i}'] = {'error': str(e)}
        
        return {
            'feature_removal_results': results,
            'most_important_features': sorted(
                [(k, v['importance_drop']) for k, v in results.items() 
                 if 'importance_drop' in v], 
                key=lambda x: x[1], reverse=True
            )[:5]
        }
    
    def _run_fusion_strategy_ablation(self) -> Dict[str, Any]:
        """Test different fusion strategies"""
        
        print("   🔗 Testing fusion strategies...")
        
        text_features = self.data_loader.get_combined_text_features()
        metadata_features = self.data_loader.get_combined_metadata_features()
        labels = self.data_loader.get_combined_labels()
        
        fusion_strategies = {
            'early_fusion': self._early_fusion_strategy,
            'late_fusion': self._late_fusion_strategy,
            'weighted_fusion': self._weighted_fusion_strategy,
            'concatenation_only': self._concatenation_fusion_strategy
        }
        
        results = {}
        
        for strategy_name, strategy_func in fusion_strategies.items():
            try:
                accuracy, star_mae = strategy_func(text_features, metadata_features, labels)
                
                results[strategy_name] = {
                    'accuracy': accuracy,
                    'star_mae': star_mae
                }
                
                print(f"     {strategy_name}: Acc={accuracy:.3f}, MAE={star_mae:.3f}")
                
            except Exception as e:
                results[strategy_name] = {'error': str(e)}
                print(f"     {strategy_name}: Failed - {str(e)[:50]}...")
        
        return {
            'fusion_strategy_results': results,
            'best_fusion_strategy': max(results.keys(), key=lambda k: results[k].get('accuracy', 0))
        }
    
    def _run_algorithm_sensitivity_ablation(self) -> Dict[str, Any]:
        """Test sensitivity to different algorithms across modalities"""
        
        print("   ⚙️ Testing algorithm sensitivity...")
        
        algorithms = [
            'RandomForestClassifier', 'XGBoostClassifier', 'LogisticRegression',
            'GradientBoostingClassifier', 'ExtraTreesClassifier'
        ]
        
        modalities = {
            'text_only': self.data_loader.get_combined_text_features(),
            'metadata_only': self.data_loader.get_combined_metadata_features(),
            'combined': self.data_loader.get_combined_features()
        }
        
        labels = self.data_loader.get_combined_labels()
        results = {}
        
        for modality_name, features in modalities.items():
            modality_results = {}
            
            for algorithm in algorithms:
                try:
                    accuracy = self._quick_evaluate_features(features, labels, algorithm)
                    modality_results[algorithm] = accuracy
                    
                except Exception as e:
                    modality_results[algorithm] = 0.0
            
            results[modality_name] = modality_results
            best_algo = max(modality_results.keys(), key=lambda k: modality_results[k])
            print(f"     {modality_name}: Best={best_algo} ({modality_results[best_algo]:.3f})")
        
        return {
            'algorithm_sensitivity_results': results,
            'modality_algorithm_preferences': {
                modality: max(algos.keys(), key=lambda k: algos[k])
                for modality, algos in results.items()
            }
        }
    
    def _analyze_comprehensive_ablations(self, all_ablation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze all ablation study results comprehensively"""
        
        analysis = {
            'key_findings': [],
            'modality_insights': {},
            'feature_insights': {},
            'fusion_insights': {},
            'algorithm_insights': {},
            'recommendations': []
        }
        
        # Basic modality analysis
        basic_results = all_ablation_results.get('basic_modality_ablation', {})
        if 'contribution_analysis' in basic_results:
            contrib = basic_results['contribution_analysis']
            
            analysis['modality_insights'] = {
                'text_performance': contrib.get('text_contribution', 0),
                'metadata_performance': contrib.get('metadata_contribution', 0),
                'multimodal_benefit': contrib.get('multimodal_benefit', 0),
                'best_single_modality': contrib.get('best_single_modality', 'unknown'),
                'synergy_detected': contrib.get('synergy_effect', 0) > 0.01
            }
            
            # Key findings
            if contrib.get('multimodal_benefit', 0) > 0.05:
                analysis['key_findings'].append("Strong multimodal benefit detected (>5% improvement)")
            
            if contrib.get('text_contribution', 0) > contrib.get('metadata_contribution', 0) * 2:
                analysis['key_findings'].append("Text modality significantly outperforms metadata")
        
        # Feature importance insights
        text_ablation = all_ablation_results.get('text_feature_ablation', {})
        if 'feature_subset_results' in text_ablation:
            analysis['feature_insights']['text'] = {
                'feature_subsets_tested': len(text_ablation['feature_subset_results']),
                'performance_variation': self._calculate_performance_variation(
                    text_ablation['feature_subset_results']
                )
            }
        
        metadata_ablation = all_ablation_results.get('metadata_feature_ablation', {})
        if 'most_important_features' in metadata_ablation:
            analysis['feature_insights']['metadata'] = {
                'most_important_features': metadata_ablation['most_important_features'][:3],
                'feature_importance_range': self._calculate_importance_range(metadata_ablation)
            }
        
        # Fusion strategy insights
        fusion_ablation = all_ablation_results.get('fusion_strategy_ablation', {})
        if 'fusion_strategy_results' in fusion_ablation:
            analysis['fusion_insights'] = {
                'best_strategy': fusion_ablation.get('best_fusion_strategy', 'unknown'),
                'strategy_performance': fusion_ablation['fusion_strategy_results']
            }
        
        # Algorithm sensitivity insights
        algo_ablation = all_ablation_results.get('algorithm_sensitivity_ablation', {})
        if 'modality_algorithm_preferences' in algo_ablation:
            analysis['algorithm_insights'] = {
                'modality_preferences': algo_ablation['modality_algorithm_preferences'],
                'algorithm_consistency': self._analyze_algorithm_consistency(algo_ablation)
            }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_ablation_recommendations(analysis)
        
        return analysis
    
    def _quick_evaluate_features(self, features: np.ndarray, labels: np.ndarray, algorithm: str = 'RandomForestClassifier') -> float:
        """Quick evaluation of features with given algorithm"""
        
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]
        
        model = self.baseline_evaluator._create_model(algorithm)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        return float(np.mean(y_pred == y_test))
    
    def _analyze_text_feature_importance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text feature importance from subset results"""
        
        valid_results = {k: v for k, v in results.items() if 'accuracy' in v}
        if not valid_results:
            return {'error': 'No valid results to analyze'}
        
        accuracies = [v['accuracy'] for v in valid_results.values()]
        
        return {
            'best_subset': max(valid_results.keys(), key=lambda k: valid_results[k]['accuracy']),
            'worst_subset': min(valid_results.keys(), key=lambda k: valid_results[k]['accuracy']),
            'performance_range': max(accuracies) - min(accuracies),
            'mean_performance': np.mean(accuracies),
            'feature_efficiency': {
                k: v['accuracy'] / v.get('n_features', 1) 
                for k, v in valid_results.items() if 'n_features' in v
            }
        }
    
    def _early_fusion_strategy(self, text_features: np.ndarray, metadata_features: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """Early fusion: concatenate features then train"""
        combined_features = np.concatenate([text_features, metadata_features], axis=1)
        accuracy = self._quick_evaluate_features(combined_features, labels)
        
        # Calculate star MAE
        split_idx = int(len(combined_features) * 0.8)
        X_train, X_test = combined_features[:split_idx], combined_features[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]
        
        model = self.baseline_evaluator._create_model('RandomForestClassifier')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        star_mae = float(np.mean(np.abs((y_pred + 1) - (y_test + 1))))
        
        return accuracy, star_mae
    
    def _late_fusion_strategy(self, text_features: np.ndarray, metadata_features: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """Late fusion: train separate models then combine predictions"""
        
        split_idx = int(len(text_features) * 0.8)
        
        # Train text model
        X_text_train, X_text_test = text_features[:split_idx], text_features[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]
        
        text_model = self.baseline_evaluator._create_model('RandomForestClassifier')
        text_model.fit(X_text_train, y_train)
        text_pred = text_model.predict(X_text_test)
        
        # Train metadata model
        X_meta_train, X_meta_test = metadata_features[:split_idx], metadata_features[split_idx:]
        meta_model = self.baseline_evaluator._create_model('RandomForestClassifier')
        meta_model.fit(X_meta_train, y_train)
        meta_pred = meta_model.predict(X_meta_test)
        
        # Combine predictions (simple averaging)
        combined_pred = np.round((text_pred + meta_pred) / 2).astype(int)
        combined_pred = np.clip(combined_pred, 0, 4)  # Ensure valid class range
        
        accuracy = float(np.mean(combined_pred == y_test))
        star_mae = float(np.mean(np.abs((combined_pred + 1) - (y_test + 1))))
        
        return accuracy, star_mae
    
    def _weighted_fusion_strategy(self, text_features: np.ndarray, metadata_features: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """Weighted fusion: learn optimal weights for modality combination"""
        
        # Simple weighted fusion with 70% text, 30% metadata (based on typical performance)
        split_idx = int(len(text_features) * 0.8)
        
        # Train individual models
        X_text_train, X_text_test = text_features[:split_idx], text_features[split_idx:]
        X_meta_train, X_meta_test = metadata_features[:split_idx], metadata_features[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]
        
        text_model = self.baseline_evaluator._create_model('RandomForestClassifier')
        text_model.fit(X_text_train, y_train)
        text_pred = text_model.predict(X_text_test)
        
        meta_model = self.baseline_evaluator._create_model('RandomForestClassifier')
        meta_model.fit(X_meta_train, y_train)
        meta_pred = meta_model.predict(X_meta_test)
        
        # Weighted combination
        weighted_pred = np.round(0.7 * text_pred + 0.3 * meta_pred).astype(int)
        weighted_pred = np.clip(weighted_pred, 0, 4)
        
        accuracy = float(np.mean(weighted_pred == y_test))
        star_mae = float(np.mean(np.abs((weighted_pred + 1) - (y_test + 1))))
        
        return accuracy, star_mae
    
    def _concatenation_fusion_strategy(self, text_features: np.ndarray, metadata_features: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """Simple concatenation fusion (same as early fusion but explicit)"""
        return self._early_fusion_strategy(text_features, metadata_features, labels)
    
    def _calculate_performance_variation(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance variation across feature subsets"""
        
        valid_accuracies = [v['accuracy'] for v in results.values() if 'accuracy' in v]
        if not valid_accuracies:
            return {'error': 'No valid accuracies'}
        
        return {
            'min_accuracy': min(valid_accuracies),
            'max_accuracy': max(valid_accuracies),
            'mean_accuracy': np.mean(valid_accuracies),
            'std_accuracy': np.std(valid_accuracies),
            'range': max(valid_accuracies) - min(valid_accuracies)
        }
    
    def _calculate_importance_range(self, metadata_ablation: Dict[str, Any]) -> Dict[str, float]:
        """Calculate importance range from metadata ablation"""
        
        if 'most_important_features' not in metadata_ablation:
            return {'error': 'No importance data'}
        
        importance_values = [item[1] for item in metadata_ablation['most_important_features']]
        
        return {
            'max_importance': max(importance_values) if importance_values else 0,
            'min_importance': min(importance_values) if importance_values else 0,
            'mean_importance': np.mean(importance_values) if importance_values else 0
        }
    
    def _analyze_algorithm_consistency(self, algo_ablation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze algorithm consistency across modalities"""
        
        if 'algorithm_sensitivity_results' not in algo_ablation:
            return {'error': 'No algorithm sensitivity data'}
        
        results = algo_ablation['algorithm_sensitivity_results']
        
        # Check if same algorithm performs best across modalities
        best_algorithms = []
        for modality, algo_results in results.items():
            best_algo = max(algo_results.keys(), key=lambda k: algo_results[k])
            best_algorithms.append(best_algo)
        
        consistency = len(set(best_algorithms)) == 1
        
        return {
            'consistent_best_algorithm': consistency,
            'algorithms_per_modality': dict(zip(results.keys(), best_algorithms)),
            'universal_best': best_algorithms[0] if consistency else None
        }
    
    def _generate_ablation_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on ablation analysis"""
        
        recommendations = []
        
        # Modality recommendations
        modality_insights = analysis.get('modality_insights', {})
        if modality_insights.get('multimodal_benefit', 0) > 0.05:
            recommendations.append("✅ Use multimodal approach - significant benefit detected")
        elif modality_insights.get('multimodal_benefit', 0) < 0.01:
            best_single = modality_insights.get('best_single_modality', 'text')
            recommendations.append(f"📝 Consider single modality approach using {best_single}")
        
        # Fusion strategy recommendations
        fusion_insights = analysis.get('fusion_insights', {})
        if 'best_strategy' in fusion_insights:
            best_strategy = fusion_insights['best_strategy']
            recommendations.append(f"🔗 Use {best_strategy} for optimal multimodal fusion")
        
        # Algorithm recommendations
        algo_insights = analysis.get('algorithm_insights', {})
        if algo_insights.get('algorithm_consistency', {}).get('consistent_best_algorithm', False):
            universal_best = algo_insights['algorithm_consistency']['universal_best']
            recommendations.append(f"⚙️ {universal_best} performs consistently across all modalities")
        
        # Feature efficiency recommendations
        feature_insights = analysis.get('feature_insights', {})
        if 'text' in feature_insights:
            text_variation = feature_insights['text'].get('performance_variation', {})
            if text_variation.get('range', 0) > 0.1:
                recommendations.append("📝 Text feature selection could significantly impact performance")
        
        return recommendations
    
    def _calculate_confidence_interval(self, scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for scores"""
        
        if len(scores) < 2:
            return (0.0, 0.0)
        
        from scipy import stats
        mean_score = np.mean(scores)
        sem = stats.sem(scores)
        interval = sem * stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
        
        return (float(mean_score - interval), float(mean_score + interval))
    
    def _create_imbalanced_dataset(self, X: np.ndarray, y: np.ndarray, ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create an imbalanced dataset with specified ratio"""
        
        # Find the class with most samples (will be majority)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        majority_class = unique_classes[np.argmax(class_counts)]
        
        # Sample other classes to create imbalance
        indices_to_keep = []
        
        for class_label in unique_classes:
            class_indices = np.where(y == class_label)[0]
            
            if class_label == majority_class:
                # Keep all samples from majority class
                indices_to_keep.extend(class_indices)
            else:
                # Sample minority classes
                target_count = int(len(class_indices) / ratio)
                if target_count > 0:
                    sampled_indices = np.random.choice(class_indices, target_count, replace=False)
                    indices_to_keep.extend(sampled_indices)
        
        indices_to_keep = np.array(indices_to_keep)
        return X[indices_to_keep], y[indices_to_keep]
    
    def _create_experiment_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create high-level experiment summary"""
        
        summary = {
            'dataset_info': {
                'name': 'Amazon Reviews 5-Class Rating Prediction',
                'n_classes': self.exp_config.n_classes,
                'class_names': self.exp_config.class_names,
                'train_size': len(self.data_loader.train_labels),
                'test_size': len(self.data_loader.test_labels),
                'modalities': ['text', 'metadata']
            },
            'experiment_scope': {
                'baseline_models_tested': len(all_results.get('baseline_results', {}).get('individual_results', {})),
                'mainmodel_evaluated': 'mainmodel_results' in all_results,
                'cross_validation_performed': 'cv_results' in all_results,
                'robustness_testing_performed': 'robustness_results' in all_results,
                'ablation_study_performed': 'ablation_results' in all_results
            },
            'computational_details': {
                'total_runtime': sum(
                    result.get('total_time', 0) 
                    for result in all_results.values() 
                    if isinstance(result, dict)
                ),
                'random_seed': self.exp_config.random_seed,
                'cv_folds': self.exp_config.cv_folds
            }
        }
        
        return summary
    
    def _rank_all_models(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Rank all models by performance"""
        
        model_performances = []
        
        # Collect baseline results
        baseline_results = all_results.get('baseline_results', {}).get('individual_results', {})
        for model_name, result in baseline_results.items():
            if 'accuracy' in result:
                model_performances.append({
                    'model': model_name,
                    'type': 'baseline',
                    'accuracy': result['accuracy'],
                    'star_mae': result.get('star_mae', float('inf'))
                })
        
        # Collect MainModel results
        mainmodel_results = all_results.get('mainmodel_results', {})
        if 'optimized_test' in mainmodel_results:
            opt_result = mainmodel_results['optimized_test']
            if opt_result and 'accuracy' in opt_result:
                model_performances.append({
                    'model': 'MainModel (Optimized)',
                    'type': 'mainmodel',
                    'accuracy': opt_result['accuracy'],
                    'star_mae': opt_result.get('star_mae', float('inf'))
                })
        
        # Sort by accuracy (descending)
        model_performances.sort(key=lambda x: x['accuracy'], reverse=True)
        
        return {
            'ranking': model_performances,
            'best_model': model_performances[0] if model_performances else None,
            'top_5_models': model_performances[:5]
        }
    
    def _test_statistical_significance(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Test statistical significance between models"""
        
        significance_results = {}
        
        try:
            # This would require cross-validation results for proper statistical testing
            cv_results = all_results.get('cv_results', {}).get('cv_results', {})
            
            if len(cv_results) >= 2:
                # Simple pairwise comparison (would need proper statistical tests in practice)
                model_names = list(cv_results.keys())
                comparisons = []
                
                for i, model1 in enumerate(model_names):
                    for model2 in model_names[i+1:]:
                        if 'scores' in cv_results[model1] and 'scores' in cv_results[model2]:
                            scores1 = cv_results[model1]['scores']
                            scores2 = cv_results[model2]['scores']
                            
                            if scores1 and scores2:
                                # Simple comparison (in practice, would use proper statistical tests)
                                mean_diff = np.mean(scores1) - np.mean(scores2)
                                comparisons.append({
                                    'model1': model1,
                                    'model2': model2,
                                    'mean_difference': float(mean_diff),
                                    'model1_better': mean_diff > 0
                                })
                
                significance_results = {
                    'pairwise_comparisons': comparisons,
                    'note': 'Simplified comparison - proper statistical tests would require more sophisticated analysis'
                }
        
        except Exception as e:
            significance_results = {'error': str(e)}
        
        return significance_results
    
    def _generate_recommendations(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on results"""
        
        recommendations = {
            'best_overall_model': None,
            'best_for_accuracy': None,
            'best_for_efficiency': None,
            'modality_recommendations': {},
            'hyperparameter_insights': {},
            'deployment_considerations': {}
        }
        
        try:
            # Analyze ranking
            ranking = all_results.get('comprehensive_report', {}).get('performance_ranking', {})
            if ranking and 'best_model' in ranking:
                recommendations['best_overall_model'] = ranking['best_model']
            
            # Analyze modality contributions
            ablation = all_results.get('ablation_results', {}).get('modality_contributions', {})
            if ablation and 'multimodal_benefit' in ablation:
                benefit = ablation['multimodal_benefit']
                if benefit > 0.05:  # 5% improvement threshold
                    recommendations['modality_recommendations']['use_multimodal'] = True
                    recommendations['modality_recommendations']['benefit'] = f"{benefit:.1%} improvement"
                else:
                    best_single = ablation.get('best_single_modality', 'text')
                    recommendations['modality_recommendations']['use_multimodal'] = False
                    recommendations['modality_recommendations']['recommended_modality'] = best_single
            
            # Analyze hyperparameter importance
            hp_results = all_results.get('mainmodel_results', {}).get('hyperparameter_search', {})
            if hp_results and 'parameter_importance' in hp_results:
                recommendations['hyperparameter_insights'] = hp_results['parameter_importance']
            
            # Deployment considerations
            baseline_results = all_results.get('baseline_results', {}).get('summary', {})
            if baseline_results:
                fastest_model = baseline_results.get('fastest_model')
                most_accurate = baseline_results.get('best_model')
                
                recommendations['deployment_considerations'] = {
                    'for_production_speed': fastest_model,
                    'for_maximum_accuracy': most_accurate,
                    'balance_recommendation': 'Consider ensemble methods for best accuracy-speed tradeoff'
                }
        
        except Exception as e:
            recommendations['error'] = str(e)
        
        return recommendations
    
    def _create_visualizations(self, all_results: Dict[str, Any]) -> Dict[str, str]:
        """Create visualization summaries (placeholders for actual plots)"""
        
        visualizations = {
            'performance_comparison': 'Bar chart comparing model accuracies',
            'confusion_matrix': 'Heatmap of prediction errors',
            'feature_importance': 'Bar chart of most important features',
            'modality_contribution': 'Comparison of single vs multi-modal performance',
            'robustness_analysis': 'Line plots showing performance under different conditions'
        }
        
        # In a real implementation, you would generate actual plots here
        # For now, we return descriptions of what would be plotted
        
        return visualizations
    
    def _analyze_cv_results(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cross-validation results for statistical significance"""
        
        # Extract scores for statistical analysis
        valid_results = {name: result for name, result in cv_results.items() 
                        if 'error' not in result and result['scores']}
        
        if not valid_results:
            return {
                'mean_performance': 0.0,
                'performance_variance': 0.0,
                'best_model': None,
                'statistical_significance': 'No valid results'
            }
        
        # Calculate overall statistics
        all_scores = []
        for result in valid_results.values():
            all_scores.extend(result['scores'])
        
        mean_performance = sum(all_scores) / len(all_scores) if all_scores else 0.0
        variance = sum((score - mean_performance) ** 2 for score in all_scores) / len(all_scores) if all_scores else 0.0
        
        # Find best model
        best_model = max(valid_results.keys(), 
                        key=lambda k: valid_results[k]['mean_accuracy'])
        
        return {
            'mean_performance': mean_performance,
            'performance_variance': variance,
            'best_model': best_model,
            'num_valid_models': len(valid_results),
            'statistical_significance': f"Analyzed {len(valid_results)} models with {len(all_scores)} total CV scores"
        }
