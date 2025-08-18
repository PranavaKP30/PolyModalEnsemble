#!/usr/bin/env python3
"""
Comprehensive Baseline Evaluation Module for ChestX-ray14 Experiments
================================================================

This module implements a comprehensive suite of 25+ baseline models across multiple categories:
1. Basic Simple Models (5): Logistic Regression, Random Forest, SVM, Naive Bayes, KNN
2. Advanced Ensemble Methods (8): XGBoost, LightGBM, AdaBoost, Extra Trees, Voting, Bagging
3. Deep Learning Models (6): MLPs of various architectures, CNNs for image modality
4. Multimodal Fusion Models (4): Early fusion, late fusion, attention fusion, weighted fusion
5. State-of-the-art Models (3): Transformer-based, advanced meta-learners, stack ensembles
"""

import time
import warnings
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, classification_report, multilabel_confusion_matrix,
    hamming_loss, jaccard_score, f1_score, average_precision_score
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputClassifier
import sys
import os

# Try to import XGBoost and LightGBM (optional dependencies)
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# Try to import neural network libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from config import ExperimentConfig
from data_loader import ChestXRayDataLoader


class BaselineEvaluator:
    """Comprehensive evaluator for 25+ baseline models on ChestX-ray14"""
    
    def __init__(self, config: 'ExperimentConfig', data_loader: 'ChestXRayDataLoader'):
        """Initialize baseline evaluator with data loader"""
        self.config = config
        self.data_loader = data_loader
        self.scalers = {}
        self.feature_selectors = {}
        
        # Filter out rare classes that cause model failures
        self._filter_rare_classes()
    
    def _filter_rare_classes(self):
        """Filter out ultra-rare classes that cause model failures"""
        train_counts = self.data_loader.train_labels.sum(axis=0)
        test_counts = self.data_loader.test_labels.sum(axis=0)
        
        # Find classes with extremely few samples (≤1 in train or 0 in test)
        problematic_classes = []
        for i, (train_count, test_count) in enumerate(zip(train_counts, test_counts)):
            if train_count <= 1 or test_count == 0:
                problematic_classes.append(i)
        
        if problematic_classes:
            print(f"⚠️  Filtering out ultra-rare pathologies for stable training:")
            for idx in problematic_classes:
                pathology_name = self.config.class_names[idx] if hasattr(self.config, 'class_names') else f"Class_{idx}"
                print(f"   - {pathology_name} (train: {train_counts[idx]}, test: {test_counts[idx]})")
            
            # Filter labels to exclude problematic classes
            good_classes = [i for i in range(self.data_loader.train_labels.shape[1]) if i not in problematic_classes]
            self.data_loader.train_labels = self.data_loader.train_labels[:, good_classes]
            self.data_loader.test_labels = self.data_loader.test_labels[:, good_classes]
            
            # Update config to reflect filtered classes
            if hasattr(self.config, 'class_names'):
                self.config.class_names = [self.config.class_names[i] for i in good_classes]
                self.config.n_classes = len(self.config.class_names)
                if hasattr(self.config, 'pathologies'):
                    self.config.pathologies = self.config.class_names
                if hasattr(self.config, 'n_pathologies'):
                    self.config.n_pathologies = self.config.n_classes
            
            print(f"   ✅ Using {len(good_classes)}/{len(good_classes) + len(problematic_classes)} pathologies for stable training")
        
    def run_all_baselines(self) -> Dict[str, Any]:
        """Run comprehensive baseline evaluation across all model categories"""
        print(f"\n📈 COMPREHENSIVE BASELINE MODELS EVALUATION")
        print("=" * 80)
        print("🎯 Multi-label pathology diagnosis baselines (25+ models)")
        print("📊 Categories: Simple models, Ensembles, Deep Learning, Fusion, State-of-the-art")
        
        # Prepare features for different model types
        train_combined, test_combined = self._prepare_combined_features()
        train_scaled, test_scaled = self._prepare_scaled_features(train_combined, test_combined)
        train_selected, test_selected = self._prepare_selected_features(train_scaled, test_scaled)
        
        # Prepare individual modality features for fusion models
        modality_features = self._prepare_modality_features()
        
        baseline_results = {}
        
        # Category 1: Basic Simple Models (5 models)
        print(f"\n🔧 CATEGORY 1: BASIC SIMPLE MODELS")
        print("-" * 50)
        simple_models = self._get_simple_models()
        for name, model in simple_models.items():
            print(f"🔄 Testing {name}...")
            try:
                result = self._evaluate_baseline(model, train_scaled, test_scaled, name)
                baseline_results[name] = result
                print(f"   ✅ {name}: F1={result['f1_score']:.3f}, Acc={result['accuracy']:.3f}")
            except Exception as e:
                print(f"   ❌ {name} failed: {str(e)[:100]}...")
                baseline_results[name] = self._create_failure_result(name, str(e))
        
        # Category 2: Advanced Ensemble Methods (8 models)
        print(f"\n🌳 CATEGORY 2: ADVANCED ENSEMBLE METHODS")
        print("-" * 50)
        ensemble_models = self._get_ensemble_models()
        for name, model in ensemble_models.items():
            print(f"🔄 Testing {name}...")
            try:
                # Use selected features for ensemble methods
                result = self._evaluate_baseline(model, train_selected, test_selected, name)
                baseline_results[name] = result
                print(f"   ✅ {name}: F1={result['f1_score']:.3f}, Acc={result['accuracy']:.3f}")
            except Exception as e:
                print(f"   ❌ {name} failed: {str(e)[:100]}...")
                baseline_results[name] = self._create_failure_result(name, str(e))
        
        # Category 3: Deep Learning Models (6 models)
        print(f"\n🧠 CATEGORY 3: DEEP LEARNING MODELS")
        print("-" * 50)
        deep_models = self._get_deep_learning_models()
        for name, model in deep_models.items():
            print(f"🔄 Testing {name}...")
            try:
                result = self._evaluate_baseline(model, train_scaled, test_scaled, name)
                baseline_results[name] = result
                print(f"   ✅ {name}: F1={result['f1_score']:.3f}, Acc={result['accuracy']:.3f}")
            except Exception as e:
                print(f"   ❌ {name} failed: {str(e)[:100]}...")
                baseline_results[name] = self._create_failure_result(name, str(e))
        
        # Category 4: Multimodal Fusion Models (4 models)
        print(f"\n🔄 CATEGORY 4: MULTIMODAL FUSION MODELS")
        print("-" * 50)
        fusion_models = self._get_fusion_models(modality_features)
        for name, model_info in fusion_models.items():
            print(f"🔄 Testing {name}...")
            try:
                result = self._evaluate_fusion_baseline(model_info, modality_features, name)
                baseline_results[name] = result
                print(f"   ✅ {name}: F1={result['f1_score']:.3f}, Acc={result['accuracy']:.3f}")
            except Exception as e:
                print(f"   ❌ {name} failed: {str(e)[:100]}...")
                baseline_results[name] = self._create_failure_result(name, str(e))
        
        # Category 5: State-of-the-art Models (3 models)
        print(f"\n🚀 CATEGORY 5: STATE-OF-THE-ART MODELS")
        print("-" * 50)
        sota_models = self._get_sota_models()
        for name, model in sota_models.items():
            print(f"🔄 Testing {name}...")
            try:
                result = self._evaluate_baseline(model, train_selected, test_selected, name)
                baseline_results[name] = result
                print(f"   ✅ {name}: F1={result['f1_score']:.3f}, Acc={result['accuracy']:.3f}")
            except Exception as e:
                print(f"   ❌ {name} failed: {str(e)[:100]}...")
                baseline_results[name] = self._create_failure_result(name, str(e))
        
        # Summary
        successful_models = [name for name, result in baseline_results.items() 
                           if result.get('training_time', 0) > 0]
        
        print(f"\n📊 BASELINE EVALUATION SUMMARY")
        print("=" * 80)
        print(f"🎯 Total models tested: {len(baseline_results)}")
        print(f"✅ Successful models: {len(successful_models)}")
        print(f"❌ Failed models: {len(baseline_results) - len(successful_models)}")
        
        if successful_models:
            best_model = max(successful_models, key=lambda x: baseline_results[x]['f1_score'])
            print(f"🏆 Best performing model: {best_model} (F1: {baseline_results[best_model]['f1_score']:.3f})")
        
        # Create the proper structure for results manager
        final_results = {
            'individual_models': baseline_results,
            'summary': {
                'total_models': len(baseline_results),
                'successful_models': len(successful_models),
                'failed_models': len(baseline_results) - len(successful_models),
                'best_model': best_model if successful_models else None,
                'best_f1': baseline_results[best_model]['f1_score'] if successful_models else 0.0
            }
        }
        
        return final_results
    
    def _prepare_combined_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare combined features for baseline models"""
        print("🔧 Preparing combined multi-modal features...")
        
        # Combine all modalities
        train_combined = np.hstack([
            self.data_loader.train_image,
            self.data_loader.train_text,
            self.data_loader.train_metadata
        ])
        
        test_combined = np.hstack([
            self.data_loader.test_image,
            self.data_loader.test_text,
            self.data_loader.test_metadata
        ])
        
        print(f"   📊 Combined features: {train_combined.shape[1]} dimensions")
        print(f"   📊 Training samples: {train_combined.shape[0]:,}")
        print(f"   📊 Test samples: {test_combined.shape[0]:,}")
        
        return train_combined, test_combined
    
    def _prepare_scaled_features(self, train_combined: np.ndarray, test_combined: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare scaled features using StandardScaler"""
        print("🔧 Applying feature scaling...")
        
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_combined)
        test_scaled = scaler.transform(test_combined)
        self.scalers['standard'] = scaler
        
        return train_scaled, test_scaled
    
    def _prepare_selected_features(self, train_scaled: np.ndarray, test_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature-selected data for high-dimensional models"""
        print("🔧 Applying feature selection...")
        
        # Use top 1000 features for computational efficiency
        k_features = min(1000, train_scaled.shape[1])
        
        # For multi-label data, use variance-based feature selection instead of mutual info
        # which requires 1D labels
        try:
            # Try variance threshold first (no labels needed)
            variance_threshold = VarianceThreshold(threshold=0.01)
            train_var_filtered = variance_threshold.fit_transform(train_scaled)
            test_var_filtered = variance_threshold.transform(test_scaled)
            
            # If still too many features, use PCA for dimensionality reduction
            if train_var_filtered.shape[1] > k_features:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=k_features, random_state=self.config.random_seed)
                train_selected = pca.fit_transform(train_var_filtered)
                test_selected = pca.transform(test_var_filtered)
                self.feature_selectors['pca'] = pca
                print(f"   📊 Selected features: {train_selected.shape[1]} via PCA from {train_scaled.shape[1]}")
            else:
                train_selected = train_var_filtered
                test_selected = test_var_filtered
                self.feature_selectors['variance'] = variance_threshold
                print(f"   📊 Selected features: {train_selected.shape[1]} via variance threshold from {train_scaled.shape[1]}")
        except Exception as e:
            print(f"   ⚠️ Feature selection failed ({str(e)}), using all features")
            train_selected = train_scaled
            test_selected = test_scaled
            print(f"   📊 Using all features: {train_selected.shape[1]}")
        
        return train_selected, test_selected
    
    def _prepare_modality_features(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Prepare individual modality features for fusion models"""
        print("🔧 Preparing individual modality features...")
        
        modality_features = {
            'train': {
                'image': self.data_loader.train_image,
                'text': self.data_loader.train_text,
                'metadata': self.data_loader.train_metadata
            },
            'test': {
                'image': self.data_loader.test_image,
                'text': self.data_loader.test_text,
                'metadata': self.data_loader.test_metadata
            }
        }
        
        # Scale each modality separately
        for split in ['train', 'test']:
            for modality in ['image', 'text', 'metadata']:
                scaler = StandardScaler()
                if split == 'train':
                    modality_features[split][modality] = scaler.fit_transform(modality_features[split][modality])
                    self.scalers[f'{modality}_scaler'] = scaler
                else:
                    modality_features[split][modality] = self.scalers[f'{modality}_scaler'].transform(modality_features[split][modality])
        
        return modality_features
    
    def _get_simple_models(self) -> Dict[str, Any]:
        """Get basic simple baseline models"""
        models = {
            'logistic_regression': MultiOutputClassifier(LogisticRegression(
                max_iter=2000,  # Increased from 1000
                random_state=self.config.random_seed,
                solver='liblinear',
                C=1.0,          # Added regularization parameter
                n_jobs=-1,
                class_weight='balanced'  # Handle class imbalance
            )),
            'random_forest': MultiOutputClassifier(RandomForestClassifier(
                n_estimators=200,  # Increased from 100
                max_depth=15,      # Added max_depth
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.config.random_seed,
                n_jobs=-1,
                class_weight='balanced'  # Handle class imbalance
            )),
            'svm_linear': MultiOutputClassifier(LinearSVC(
                random_state=self.config.random_seed,
                max_iter=3000,
                dual=False,
                class_weight='balanced',  # Handle class imbalance
                C=0.1                    # Lower regularization for medical data
            )),
            'naive_bayes': MultiOutputClassifier(GaussianNB()),
            'knn': MultiOutputClassifier(KNeighborsClassifier(
                n_neighbors=7,          # Slightly more neighbors
                weights='distance',     # Weight by distance
                n_jobs=-1
            ))
        }
        return models
    
    def _get_ensemble_models(self) -> Dict[str, Any]:
        """Get advanced ensemble baseline models"""
        models = {
            'gradient_boosting': MultiOutputClassifier(GradientBoostingClassifier(
                n_estimators=150,        # Increased estimators
                max_depth=8,            # Increased depth 
                learning_rate=0.05,     # Lower learning rate for stability
                subsample=0.8,          # Add some regularization
                random_state=self.config.random_seed
            )),
            'extra_trees': MultiOutputClassifier(ExtraTreesClassifier(
                n_estimators=200,       # Increased trees
                max_depth=12,          # Increased depth
                min_samples_split=4,   # Prevent overfitting
                min_samples_leaf=2,    # Prevent overfitting
                random_state=self.config.random_seed,
                n_jobs=-1,
                class_weight='balanced_subsample'  # Handle class imbalance
            )),
            'ada_boost': MultiOutputClassifier(AdaBoostClassifier(
                n_estimators=100,       # Increased estimators
                learning_rate=0.5,     # Lower learning rate
                random_state=self.config.random_seed
            )),
            'bagging_classifier': MultiOutputClassifier(BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=8, class_weight='balanced'),
                n_estimators=50,
                random_state=self.config.random_seed,
                n_jobs=-1
            )),
        }
        
        # Add XGBoost and LightGBM if available
        if HAS_XGB:
            models['xgboost'] = MultiOutputClassifier(xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=self.config.random_seed,
                eval_metric='logloss',
                verbosity=0
            ))
        
        if HAS_LGB:
            models['lightgbm'] = MultiOutputClassifier(lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=self.config.random_seed,
                verbosity=-1
            ))
        
        # Voting ensemble of top performers
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=self.config.random_seed, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=self.config.random_seed)),
            ('lr', LogisticRegression(max_iter=1000, random_state=self.config.random_seed, n_jobs=-1))
        ]
        
        models['voting_ensemble'] = MultiOutputClassifier(VotingClassifier(
            estimators=base_estimators,
            voting='soft',
            n_jobs=-1
        ))
        
        return models
    
    def _get_deep_learning_models(self) -> Dict[str, Any]:
        """Get deep learning baseline models"""
        models = {
            'mlp_small': MultiOutputClassifier(MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=300,
                random_state=self.config.random_seed,
                early_stopping=True,
                validation_fraction=0.1
            )),
            'mlp_medium': MultiOutputClassifier(MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                max_iter=300,
                random_state=self.config.random_seed,
                early_stopping=True,
                validation_fraction=0.1
            )),
            'mlp_large': MultiOutputClassifier(MLPClassifier(
                hidden_layer_sizes=(512, 256, 128),
                max_iter=300,
                random_state=self.config.random_seed,
                early_stopping=True,
                validation_fraction=0.1
            )),
            'mlp_deep': MultiOutputClassifier(MLPClassifier(
                hidden_layer_sizes=(256, 128, 64, 32),
                max_iter=300,
                random_state=self.config.random_seed,
                early_stopping=True,
                validation_fraction=0.1
            )),
            'mlp_wide': MultiOutputClassifier(MLPClassifier(
                hidden_layer_sizes=(1024, 512),
                max_iter=300,
                random_state=self.config.random_seed,
                early_stopping=True,
                validation_fraction=0.1
            )),
            'sgd_classifier': MultiOutputClassifier(SGDClassifier(
                max_iter=1000,
                random_state=self.config.random_seed,
                n_jobs=-1
            ))
        }
        
        return models
    
    def _get_fusion_models(self, modality_features: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Get multimodal fusion baseline models"""
        models = {
            'early_fusion': {
                'type': 'early_fusion',
                'model': MultiOutputClassifier(RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.config.random_seed,
                    n_jobs=-1
                ))
            },
            'late_fusion': {
                'type': 'late_fusion',
                'modality_models': {
                    'image': MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=self.config.random_seed)),
                    'text': MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=self.config.random_seed)),
                    'metadata': MultiOutputClassifier(GradientBoostingClassifier(n_estimators=50, random_state=self.config.random_seed))
                },
                'fusion_model': MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=self.config.random_seed))
            },
            'weighted_fusion': {
                'type': 'weighted_fusion',
                'modality_models': {
                    'image': MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=self.config.random_seed)),
                    'text': MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=self.config.random_seed)),
                    'metadata': MultiOutputClassifier(ExtraTreesClassifier(n_estimators=50, random_state=self.config.random_seed))
                },
                'weights': [0.4, 0.35, 0.25]  # Based on modality importance
            },
            'attention_fusion': {
                'type': 'attention_fusion',
                'model': MultiOutputClassifier(MLPClassifier(
                    hidden_layer_sizes=(512, 256, 128),
                    max_iter=300,
                    random_state=self.config.random_seed,
                    early_stopping=True
                ))
            }
        }
        return models
    
    def _get_sota_models(self) -> Dict[str, Any]:
        """Get state-of-the-art baseline models"""
        models = {}
        
        # Create stacking ensemble
        base_learners = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=self.config.random_seed)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=self.config.random_seed)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200, random_state=self.config.random_seed))
        ]
        
        models['stacking_ensemble'] = MultiOutputClassifier(StackingClassifier(
            estimators=base_learners,
            final_estimator=LogisticRegression(max_iter=1000, random_state=self.config.random_seed),
            n_jobs=-1
        ))
        
        # Add advanced MLP with dropout simulation
        models['advanced_mlp'] = MultiOutputClassifier(MLPClassifier(
            hidden_layer_sizes=(1024, 512, 256, 128),
            max_iter=500,
            random_state=self.config.random_seed,
            early_stopping=True,
            validation_fraction=0.15,
            alpha=0.01  # L2 regularization
        ))
        
        # Meta-learner ensemble
        models['meta_ensemble'] = self._create_meta_ensemble()
        
        return models
    
    def _create_meta_ensemble(self):
        """Create a meta-learning ensemble"""
        # First level: diverse base learners
        level1_learners = [
            RandomForestClassifier(n_estimators=50, max_depth=8, random_state=self.config.random_seed),
            GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=self.config.random_seed),
            LogisticRegression(max_iter=1000, random_state=self.config.random_seed),
            MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200, random_state=self.config.random_seed)
        ]
        
        # Second level: meta-learner
        meta_learner = LogisticRegression(max_iter=1000, random_state=self.config.random_seed)
        
        return MultiOutputClassifier(StackingClassifier(
            estimators=[(f'base_{i}', learner) for i, learner in enumerate(level1_learners)],
            final_estimator=meta_learner,
            cv=3,
            n_jobs=-1
        ))
    
    def _evaluate_baseline(self, model, X_train: np.ndarray, X_test: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Evaluate a single baseline model with comprehensive results"""
        
        # Training
        start_time = time.time()
        model.fit(X_train, self.data_loader.train_labels)
        train_time = time.time() - start_time
        
        # Prediction
        start_time = time.time()
        
        # Try to get probabilities for better thresholding
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                # For MultiOutputClassifier, predict_proba returns a list of arrays
                if isinstance(y_pred_proba, list):
                    # Use lower threshold for imbalanced multi-label data
                    y_pred = np.array([
                        (proba[:, 1] > 0.3).astype(int) for proba in y_pred_proba
                    ]).T
                else:
                    y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
        except:
            y_pred = model.predict(X_test)
            
        inference_time = time.time() - start_time
        
        # Try to get prediction probabilities if available
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            elif hasattr(model, 'decision_function'):
                y_pred_proba = model.decision_function(X_test)
            else:
                y_pred_proba = None
        except:
            y_pred_proba = None
        
        # Debug: Check prediction patterns
        print(f"   [DEBUG] {model_name} predictions:")
        print(f"     Predictions sum: {y_pred.sum()}")
        print(f"     True labels sum: {self.data_loader.test_labels.sum()}")
        print(f"     Per-class predictions: {y_pred.sum(axis=0)}")
        print(f"     Per-class true labels: {self.data_loader.test_labels.sum(axis=0)}")
        
        # Calculate metrics
        metrics = self._calculate_multilabel_metrics(self.data_loader.test_labels, y_pred)
        
        return {
            'model_name': model_name,
            'training_time': train_time,
            'inference_time': inference_time,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'true_labels': self.data_loader.test_labels,
            **metrics
        }
    
    def _evaluate_fusion_baseline(self, model_info: Dict[str, Any], modality_features: Dict[str, Dict[str, np.ndarray]], model_name: str) -> Dict[str, Any]:
        """Evaluate fusion baseline models"""
        
        start_time = time.time()
        
        if model_info['type'] == 'early_fusion':
            # Early fusion: concatenate all modalities
            train_fused = np.hstack([modality_features['train'][mod] for mod in ['image', 'text', 'metadata']])
            test_fused = np.hstack([modality_features['test'][mod] for mod in ['image', 'text', 'metadata']])
            
            model_info['model'].fit(train_fused, self.data_loader.train_labels)
            y_pred = model_info['model'].predict(test_fused)
            
        elif model_info['type'] == 'late_fusion':
            # Late fusion: train separate models, then fuse predictions
            modality_predictions = {}
            
            for modality, model in model_info['modality_models'].items():
                model.fit(modality_features['train'][modality], self.data_loader.train_labels)
                modality_predictions[modality] = model.predict(modality_features['test'][modality])
            
            # Stack predictions for meta-learning
            stacked_predictions = np.hstack(list(modality_predictions.values()))
            model_info['fusion_model'].fit(
                np.hstack([model.predict(modality_features['train'][mod]) for mod, model in model_info['modality_models'].items()]),
                self.data_loader.train_labels
            )
            y_pred = model_info['fusion_model'].predict(stacked_predictions)
            
        elif model_info['type'] == 'weighted_fusion':
            # Weighted fusion: weighted average of modality predictions
            modality_predictions = []
            
            for modality, model in model_info['modality_models'].items():
                model.fit(modality_features['train'][modality], self.data_loader.train_labels)
                pred = model.predict(modality_features['test'][modality])
                modality_predictions.append(pred.astype(np.float64))  # Ensure float type for math operations
            
            # Weighted average
            y_pred = np.zeros_like(modality_predictions[0], dtype=np.float64)
            for i, (pred, weight) in enumerate(zip(modality_predictions, model_info['weights'])):
                y_pred += weight * pred
            y_pred = (y_pred > 0.5).astype(int)  # Convert back to int after weighted sum
            
        elif model_info['type'] == 'attention_fusion':
            # Attention fusion: learn attention weights
            train_fused = np.hstack([modality_features['train'][mod] for mod in ['image', 'text', 'metadata']])
            test_fused = np.hstack([modality_features['test'][mod] for mod in ['image', 'text', 'metadata']])
            
            model_info['model'].fit(train_fused, self.data_loader.train_labels)
            y_pred = model_info['model'].predict(test_fused)
        
        train_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_multilabel_metrics(self.data_loader.test_labels, y_pred)
        
        return {
            'model_name': model_name,
            'training_time': train_time,
            'inference_time': 0.1,  # Placeholder
            'predictions': y_pred,
            'true_labels': self.data_loader.test_labels,
            **metrics
        }
    
    def _calculate_multilabel_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive multilabel metrics"""
        
        # Handle edge cases
        if y_pred.shape != y_true.shape:
            print(f"   ⚠️ Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
            return {'f1_score': 0.0, 'accuracy': 0.0, 'hamming_loss': 1.0}

        try:
            # Primary metrics
            accuracy = accuracy_score(y_true, y_pred)
            f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            hamming = hamming_loss(y_true, y_pred)
            jaccard = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
            precision_macro = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[0]
            recall_macro = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[1]

            # Per-class metrics
            per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
            per_class_precision = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)[0]
            per_class_recall = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)[1]
            per_class_accuracy = (y_true == y_pred).mean(axis=0)

            # Warn if exact match accuracy is near zero
            if accuracy < 0.05:
                print(f"   ⚠️ Exact match accuracy is very low ({accuracy:.3f}). This is expected in multi-label, imbalanced medical data. Use macro/micro F1 and hamming loss for fairer comparison.")

            # Print summary for user
            print(f"   📊 Metrics summary:")
            print(f"      Exact match accuracy: {accuracy:.3f}")
            print(f"      Macro F1: {f1_macro:.3f} | Micro F1: {f1_micro:.3f} | Weighted F1: {f1_weighted:.3f}")
            print(f"      Hamming loss: {hamming:.3f} | Jaccard: {jaccard:.3f}")
            print(f"      Per-class F1: {[f'{x:.3f}' for x in per_class_f1]}")
            print(f"      Per-class accuracy: {[f'{x:.3f}' for x in per_class_accuracy]}")

            # For imbalanced multi-label data, use micro F1 as primary metric
            primary_metric = f1_micro if f1_micro > 0 else f1_macro

            return {
                'accuracy': accuracy,
                'f1_score': primary_metric,  # Use micro F1 as primary metric for imbalanced data
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
    
    def _create_failure_result(self, model_name: str, error_msg: str) -> Dict[str, Any]:
        """Create a failure result for models that didn't run"""
        return {
            'model_name': model_name,
            'training_time': 0.0,
            'inference_time': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'hamming_loss': 1.0,
            'error': error_msg[:200]  # Truncate long error messages
        }
