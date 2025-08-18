#!/usr/bin/env python3
"""
Comprehensive Baseline Evaluation Module for Amazon Reviews Experiments
====================================================================

This module implements a comprehensive suite of 25+ baseline models across multiple categories:
1. Single Modality Models (6): Text-only and metadata-only models
2. Early Fusion Models (8): Simple concatenation with various algorithms
3. Advanced Ensemble Methods (6): XGBoost, LightGBM, advanced voting
4. Multimodal Fusion Models (6): Late fusion, attention fusion, tensor fusion
5. State-of-the-art Models (4): Transformer-based, neural architectures
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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import sys
import os

# Try to import advanced libraries
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

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from config import ExperimentConfig
from data_loader import AmazonReviewsDataLoader


class BaselineEvaluator:
    """Comprehensive evaluator for 30+ baseline models on Amazon Reviews"""
    
    def __init__(self, exp_config: ExperimentConfig, data_loader: AmazonReviewsDataLoader):
        self.exp_config = exp_config
        self.data_loader = data_loader
        
    def run_all_baselines(self) -> Dict[str, Any]:
        """Run comprehensive baseline evaluation across all model categories"""
        print(f"\n📈 COMPREHENSIVE BASELINE MODELS EVALUATION")
        print("=" * 80)
        print("🎯 5-class rating prediction baselines (30+ models)")
        print("📊 Categories: Single modality, Early fusion, Ensembles, Multimodal fusion, SOTA")
        
        # Prepare data for different model types
        modality_data = self.data_loader.prepare_modality_data(normalize=True)
        
        baseline_results = {}
        
        # Category 1: Single Modality Models (6 models)
        print(f"\n📝 CATEGORY 1: SINGLE MODALITY MODELS")
        print("-" * 50)
        single_models = self._get_single_modality_models()
        for name, (model, modality) in single_models.items():
            print(f"🔄 Testing {name} on {modality}...")
            try:
                X_train, X_test = modality_data[modality]
                result = self._evaluate_baseline(model, X_train, X_test, name)
                baseline_results[name] = result
                print(f"   ✅ {name}: Acc={result['accuracy']:.3f}, ±1 Star={result['close_accuracy']:.3f}, MAE={result['star_mae']:.3f}")
            except Exception as e:
                print(f"   ❌ {name} failed: {str(e)[:100]}...")
                baseline_results[name] = self._create_failure_result(name, str(e))
        
        # Category 2: Early Fusion Models (8 models)
        print(f"\n🔗 CATEGORY 2: EARLY FUSION MODELS")
        print("-" * 50)
        early_fusion_models = self._get_early_fusion_models()
        for name, model in early_fusion_models.items():
            print(f"🔄 Testing {name}...")
            try:
                X_train, X_test = modality_data['combined']
                result = self._evaluate_baseline(model, X_train, X_test, name)
                baseline_results[name] = result
                print(f"   ✅ {name}: Acc={result['accuracy']:.3f}, ±1 Star={result['close_accuracy']:.3f}, MAE={result['star_mae']:.3f}")
            except Exception as e:
                print(f"   ❌ {name} failed: {str(e)[:100]}...")
                baseline_results[name] = self._create_failure_result(name, str(e))
        
        # Category 3: Advanced Ensemble Methods (6 models)
        print(f"\n🌳 CATEGORY 3: ADVANCED ENSEMBLE METHODS")
        print("-" * 50)
        ensemble_models = self._get_ensemble_models()
        for name, model in ensemble_models.items():
            print(f"🔄 Testing {name}...")
            try:
                X_train, X_test = modality_data['combined']
                result = self._evaluate_baseline(model, X_train, X_test, name)
                baseline_results[name] = result
                print(f"   ✅ {name}: Acc={result['accuracy']:.3f}, ±1 Star={result['close_accuracy']:.3f}, MAE={result['star_mae']:.3f}")
            except Exception as e:
                print(f"   ❌ {name} failed: {str(e)[:100]}...")
                baseline_results[name] = self._create_failure_result(name, str(e))
        
        # Category 4: Multimodal Fusion Models (6 models)
        print(f"\n🔄 CATEGORY 4: MULTIMODAL FUSION MODELS")
        print("-" * 50)
        fusion_models = self._get_fusion_models()
        for name, model_info in fusion_models.items():
            print(f"🔄 Testing {name}...")
            try:
                result = self._evaluate_fusion_baseline(model_info, modality_data, name)
                baseline_results[name] = result
                print(f"   ✅ {name}: Acc={result['accuracy']:.3f}, ±1 Star={result['close_accuracy']:.3f}, MAE={result['star_mae']:.3f}")
            except Exception as e:
                print(f"   ❌ {name} failed: {str(e)[:100]}...")
                baseline_results[name] = self._create_failure_result(name, str(e))
        
        # Category 5: State-of-the-art Models (4 models)
        print(f"\n🚀 CATEGORY 5: STATE-OF-THE-ART MODELS")
        print("-" * 50)
        sota_models = self._get_sota_models()
        for name, model in sota_models.items():
            print(f"🔄 Testing {name}...")
            try:
                X_train, X_test = modality_data['combined']
                result = self._evaluate_baseline(model, X_train, X_test, name)
                baseline_results[name] = result
                print(f"   ✅ {name}: Acc={result['accuracy']:.3f}, ±1 Star={result['close_accuracy']:.3f}, MAE={result['star_mae']:.3f}")
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
        
        # Calculate summary statistics
        best_model = None
        best_accuracy = 0
        best_mae = float('inf')
        fastest_model = None
        fastest_time = float('inf')
        total_time = 0
        
        if successful_models:
            best_model = min(successful_models, key=lambda x: baseline_results[x]['star_mae'])
            best_accuracy = max(baseline_results[x]['accuracy'] for x in successful_models)
            best_mae = baseline_results[best_model]['star_mae']
            
            fastest_model = min(successful_models, key=lambda x: baseline_results[x]['training_time'])
            fastest_time = baseline_results[fastest_model]['training_time']
            
            total_time = sum(baseline_results[x]['training_time'] for x in successful_models)
            
            print(f"🏆 Best performing model: {best_model} (MAE: {best_mae:.3f})")
        
        # Return structured results for comprehensive reporting
        return {
            'individual_results': baseline_results,
            'summary': {
                'total_models': len(baseline_results),
                'successful_models': len(successful_models),
                'failed_models': len(baseline_results) - len(successful_models),
                'best_model': best_model,
                'best_accuracy': best_accuracy,
                'best_mae': best_mae,
                'fastest_model': fastest_model,
                'fastest_time': fastest_time,
                'total_time': total_time
            }
        }
    
    def _get_single_modality_models(self) -> Dict[str, Tuple[Any, str]]:
        """Get single modality baseline models"""
        models = {
            # Text-only models
            'text_rf': (RandomForestClassifier(n_estimators=100, random_state=self.exp_config.random_seed), 'text'),
            'text_lr': (LogisticRegression(random_state=self.exp_config.random_seed, max_iter=1000), 'text'),
            'text_mlp': (MLPClassifier(hidden_layer_sizes=(128, 64), random_state=self.exp_config.random_seed, max_iter=500), 'text'),
            
            # Metadata-only models
            'metadata_rf': (RandomForestClassifier(n_estimators=100, random_state=self.exp_config.random_seed), 'metadata'),
            'metadata_lr': (LogisticRegression(random_state=self.exp_config.random_seed, max_iter=1000), 'metadata'),
            'metadata_gb': (GradientBoostingClassifier(n_estimators=100, random_state=self.exp_config.random_seed), 'metadata'),
        }
        return models
    
    def _get_early_fusion_models(self) -> Dict[str, Any]:
        """Get early fusion baseline models"""
        models = {
            'combined_rf': RandomForestClassifier(n_estimators=100, random_state=self.exp_config.random_seed),
            'combined_lr': LogisticRegression(random_state=self.exp_config.random_seed, max_iter=1000),
            'combined_mlp': MLPClassifier(hidden_layer_sizes=(256, 128), random_state=self.exp_config.random_seed, max_iter=500),
            'combined_gb': GradientBoostingClassifier(n_estimators=100, random_state=self.exp_config.random_seed),
            'combined_svm': SVC(random_state=self.exp_config.random_seed, kernel='rbf'),
            'combined_knn': KNeighborsClassifier(n_neighbors=5),
            'combined_nb': GaussianNB(),
            'combined_dt': DecisionTreeClassifier(random_state=self.exp_config.random_seed, max_depth=10)
        }
        return models
    
    def _get_ensemble_models(self) -> Dict[str, Any]:
        """Get advanced ensemble baseline models"""
        models = {}
        
        # XGBoost
        if HAS_XGB:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.exp_config.random_seed,
                eval_metric='mlogloss'
            )
        
        # LightGBM
        if HAS_LGB:
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.exp_config.random_seed,
                verbose=-1
            )
        
        # CatBoost
        if HAS_CATBOOST:
            models['catboost'] = cb.CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=self.exp_config.random_seed,
                verbose=False
            )
        
        # Advanced scikit-learn ensembles
        models.update({
            'extra_trees': ExtraTreesClassifier(
                n_estimators=100,
                random_state=self.exp_config.random_seed
            ),
            'ada_boost': AdaBoostClassifier(
                n_estimators=100,
                random_state=self.exp_config.random_seed
            ),
            'voting_ensemble': VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=50, random_state=self.exp_config.random_seed)),
                    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=self.exp_config.random_seed)),
                    ('lr', LogisticRegression(random_state=self.exp_config.random_seed, max_iter=1000))
                ],
                voting='soft'
            )
        })
        
        return models
    
    def _get_fusion_models(self) -> Dict[str, Dict[str, Any]]:
        """Get multimodal fusion baseline models"""
        models = {
            'late_fusion_voting': {
                'type': 'late_fusion_voting',
                'text_model': RandomForestClassifier(n_estimators=50, random_state=self.exp_config.random_seed),
                'metadata_model': GradientBoostingClassifier(n_estimators=50, random_state=self.exp_config.random_seed),
                'weights': [0.7, 0.3]  # Text gets higher weight
            },
            'late_fusion_stacking': {
                'type': 'late_fusion_stacking',
                'base_models': [
                    ('text_rf', RandomForestClassifier(n_estimators=50, random_state=self.exp_config.random_seed)),
                    ('meta_gb', GradientBoostingClassifier(n_estimators=50, random_state=self.exp_config.random_seed))
                ],
                'meta_learner': LogisticRegression(random_state=self.exp_config.random_seed, max_iter=1000)
            },
            'attention_fusion': {
                'type': 'attention_fusion'
            },
            'tensor_fusion': {
                'type': 'tensor_fusion'
            },
            'multiplicative_fusion': {
                'type': 'multiplicative_fusion'
            },
            'wide_and_deep': {
                'type': 'wide_and_deep'
            }
        }
        return models
    
    def _get_sota_models(self) -> Dict[str, Any]:
        """Get state-of-the-art baseline models"""
        models = {}
        
        # Stacking ensemble
        base_learners = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=self.exp_config.random_seed)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=self.exp_config.random_seed)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=self.exp_config.random_seed))
        ]
        
        models['stacking_ensemble'] = StackingClassifier(
            estimators=base_learners,
            final_estimator=LogisticRegression(max_iter=1000, random_state=self.exp_config.random_seed),
            cv=3
        )
        
        # Advanced neural networks
        models.update({
            'deep_neural_net': MLPClassifier(
                hidden_layer_sizes=(512, 256, 128, 64),
                max_iter=500,
                random_state=self.exp_config.random_seed,
                early_stopping=True,
                validation_fraction=0.1
            ),
            'wide_neural_net': MLPClassifier(
                hidden_layer_sizes=(1024, 512),
                max_iter=500,
                random_state=self.exp_config.random_seed,
                early_stopping=True,
                validation_fraction=0.1
            ),
            'regularized_neural_net': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                max_iter=500,
                random_state=self.exp_config.random_seed,
                alpha=0.01,  # L2 regularization
                early_stopping=True
            )
        })
        
        return models
    
    def _evaluate_baseline(self, model, X_train: np.ndarray, X_test: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Evaluate a single baseline model with comprehensive Amazon Reviews metrics"""
        
        # Training
        start_time = time.time()
        model.fit(X_train, self.data_loader.train_labels)
        train_time = time.time() - start_time
        
        # Prediction
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Calculate comprehensive metrics
        metrics = self._calculate_rating_metrics(self.data_loader.test_labels, y_pred)
        
        return {
            'model_name': model_name,
            'training_time': train_time,
            'inference_time': inference_time,
            'predictions': y_pred,
            'true_labels': self.data_loader.test_labels,
            **metrics
        }
    
    def _evaluate_fusion_baseline(self, model_info: Dict[str, Any], modality_data: Dict[str, Tuple], model_name: str) -> Dict[str, Any]:
        """Evaluate multimodal fusion baseline models"""
        
        start_time = time.time()
        
        fusion_type = model_info['type']
        text_train, text_test = modality_data['text']
        meta_train, meta_test = modality_data['metadata']
        
        if fusion_type == 'late_fusion_voting':
            # Train separate models and combine predictions
            text_model = model_info['text_model']
            meta_model = model_info['metadata_model']
            weights = model_info['weights']
            
            text_model.fit(text_train, self.data_loader.train_labels)
            meta_model.fit(meta_train, self.data_loader.train_labels)
            
            # Weighted voting
            text_pred = text_model.predict(text_test)
            meta_pred = meta_model.predict(meta_test)
            
            y_pred = np.round(weights[0] * text_pred + weights[1] * meta_pred).astype(int)
            
        elif fusion_type == 'late_fusion_stacking':
            # Use stacking with separate modality features
            from sklearn.ensemble import StackingClassifier
            
            stacking_model = StackingClassifier(
                estimators=model_info['base_models'],
                final_estimator=model_info['meta_learner'],
                cv=3
            )
            
            # Use combined features for stacking
            combined_train, combined_test = modality_data['combined']
            stacking_model.fit(combined_train, self.data_loader.train_labels)
            y_pred = stacking_model.predict(combined_test)
            
        elif fusion_type == 'attention_fusion':
            # Attention-based multimodal fusion
            y_pred = self._attention_fusion(text_train, text_test, meta_train, meta_test)
            
        elif fusion_type == 'tensor_fusion':
            # Tensor fusion network
            y_pred = self._tensor_fusion(text_train, text_test, meta_train, meta_test)
            
        elif fusion_type == 'multiplicative_fusion':
            # Multiplicative multimodal fusion
            y_pred = self._multiplicative_fusion(text_train, text_test, meta_train, meta_test)
            
        elif fusion_type == 'wide_and_deep':
            # Wide & Deep architecture
            y_pred = self._wide_and_deep_fusion(text_train, text_test, meta_train, meta_test)
            
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        train_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_rating_metrics(self.data_loader.test_labels, y_pred)
        
        return {
            'model_name': model_name,
            'training_time': train_time,
            'inference_time': 0.1,  # Placeholder
            'predictions': y_pred,
            'true_labels': self.data_loader.test_labels,
            **metrics
        }
    
    def _attention_fusion(self, text_train, text_test, meta_train, meta_test):
        """Implement attention-based fusion"""
        # Simplified attention mechanism using MLPs
        from sklearn.neural_network import MLPClassifier
        
        # Project modalities to same dimension
        text_encoder = MLPClassifier(hidden_layer_sizes=(64,), random_state=self.exp_config.random_seed, max_iter=300)
        meta_encoder = MLPClassifier(hidden_layer_sizes=(64,), random_state=self.exp_config.random_seed, max_iter=300)
        
        # Train encoders
        text_encoder.fit(text_train, self.data_loader.train_labels)
        meta_encoder.fit(meta_train, self.data_loader.train_labels)
        
        # Get encoded representations
        text_encoded_train = text_encoder.predict_proba(text_train)
        meta_encoded_train = meta_encoder.predict_proba(meta_train)
        text_encoded_test = text_encoder.predict_proba(text_test)
        meta_encoded_test = meta_encoder.predict_proba(meta_test)
        
        # Simple attention: weighted average based on confidence
        text_confidence = np.max(text_encoded_train, axis=1)
        meta_confidence = np.max(meta_encoded_train, axis=1)
        
        # Normalize weights
        total_confidence = text_confidence + meta_confidence + 1e-8
        text_weights = text_confidence / total_confidence
        meta_weights = meta_confidence / total_confidence
        
        # Fuse representations
        fused_train = text_weights.reshape(-1, 1) * text_encoded_train + meta_weights.reshape(-1, 1) * meta_encoded_train
        
        # Train final classifier
        final_classifier = LogisticRegression(random_state=self.exp_config.random_seed, max_iter=1000)
        final_classifier.fit(fused_train, self.data_loader.train_labels)
        
        # Test phase
        text_confidence_test = np.max(text_encoded_test, axis=1)
        meta_confidence_test = np.max(meta_encoded_test, axis=1)
        total_confidence_test = text_confidence_test + meta_confidence_test + 1e-8
        text_weights_test = text_confidence_test / total_confidence_test
        meta_weights_test = meta_confidence_test / total_confidence_test
        
        fused_test = text_weights_test.reshape(-1, 1) * text_encoded_test + meta_weights_test.reshape(-1, 1) * meta_encoded_test
        
        return final_classifier.predict(fused_test)
    
    def _tensor_fusion(self, text_train, text_test, meta_train, meta_test):
        """Implement tensor fusion network"""
        from sklearn.neural_network import MLPClassifier
        from sklearn.decomposition import PCA
        
        # Reduce dimensionality for computational efficiency
        text_pca = PCA(n_components=min(50, text_train.shape[1]), random_state=self.exp_config.random_seed)
        meta_pca = PCA(n_components=min(20, meta_train.shape[1]), random_state=self.exp_config.random_seed)
        
        text_reduced_train = text_pca.fit_transform(text_train)
        meta_reduced_train = meta_pca.fit_transform(meta_train)
        text_reduced_test = text_pca.transform(text_test)
        meta_reduced_test = meta_pca.transform(meta_test)
        
        # Create tensor features (outer product)
        tensor_train = []
        for i in range(len(text_reduced_train)):
            outer = np.outer(text_reduced_train[i], meta_reduced_train[i])
            tensor_train.append(outer.flatten())
        tensor_train = np.array(tensor_train)
        
        tensor_test = []
        for i in range(len(text_reduced_test)):
            outer = np.outer(text_reduced_test[i], meta_reduced_test[i])
            tensor_test.append(outer.flatten())
        tensor_test = np.array(tensor_test)
        
        # Train classifier on tensor features
        classifier = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            random_state=self.exp_config.random_seed,
            max_iter=300
        )
        classifier.fit(tensor_train, self.data_loader.train_labels)
        
        return classifier.predict(tensor_test)
    
    def _multiplicative_fusion(self, text_train, text_test, meta_train, meta_test):
        """Implement multiplicative fusion"""
        from sklearn.neural_network import MLPClassifier
        
        # Project to same dimension
        text_proj = MLPClassifier(hidden_layer_sizes=(32,), random_state=self.exp_config.random_seed, max_iter=300)
        meta_proj = MLPClassifier(hidden_layer_sizes=(32,), random_state=self.exp_config.random_seed, max_iter=300)
        
        text_proj.fit(text_train, self.data_loader.train_labels)
        meta_proj.fit(meta_train, self.data_loader.train_labels)
        
        # Get projections
        text_proj_train = text_proj.predict_proba(text_train)
        meta_proj_train = meta_proj.predict_proba(meta_train)
        text_proj_test = text_proj.predict_proba(text_test)
        meta_proj_test = meta_proj.predict_proba(meta_test)
        
        # Element-wise multiplication
        fused_train = text_proj_train * meta_proj_train
        fused_test = text_proj_test * meta_proj_test
        
        # Final classifier
        classifier = LogisticRegression(random_state=self.exp_config.random_seed, max_iter=1000)
        classifier.fit(fused_train, self.data_loader.train_labels)
        
        return classifier.predict(fused_test)
    
    def _wide_and_deep_fusion(self, text_train, text_test, meta_train, meta_test):
        """Implement Wide & Deep architecture"""
        from sklearn.ensemble import VotingClassifier
        
        # Wide component: linear model on combined features
        combined_train = np.hstack([text_train, meta_train])
        combined_test = np.hstack([text_test, meta_test])
        
        wide_model = LogisticRegression(random_state=self.exp_config.random_seed, max_iter=1000)
        
        # Deep component: neural network
        deep_model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            random_state=self.exp_config.random_seed,
            max_iter=300
        )
        
        # Combine wide and deep
        ensemble = VotingClassifier(
            estimators=[('wide', wide_model), ('deep', deep_model)],
            voting='soft'
        )
        
        ensemble.fit(combined_train, self.data_loader.train_labels)
        return ensemble.predict(combined_test)
    
    def _calculate_rating_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive Amazon Reviews rating prediction metrics"""
        
        try:
            # Basic classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
            
            # Convert back to 1-5 rating scale for interpretable metrics
            pred_ratings = y_pred + 1
            true_ratings = y_true + 1
            
            # Rating-specific metrics
            close_accuracy = np.mean(np.abs(pred_ratings - true_ratings) <= 1)  # Within 1 star
            exact_accuracy = accuracy  # Exact match
            star_mae = np.mean(np.abs(pred_ratings - true_ratings))  # Mean absolute error in stars
            star_mse = mean_squared_error(true_ratings, pred_ratings)
            star_rmse = np.sqrt(star_mse)
            
            # Regression-style metrics
            r2 = r2_score(true_ratings, pred_ratings)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'close_accuracy': close_accuracy,  # Within ±1 star
                'star_mae': star_mae,
                'star_mse': star_mse,
                'star_rmse': star_rmse,
                'r2_score': r2
            }
            
        except Exception as e:
            print(f"   ⚠️ Metrics calculation failed: {str(e)}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'close_accuracy': 0.0,
                'star_mae': 5.0,
                'star_mse': 25.0,
                'star_rmse': 5.0,
                'r2_score': -1.0
            }
    
    def _create_model(self, model_name: str):
        """Create a model instance by name - for backward compatibility with advanced evaluator"""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier
        try:
            from xgboost import XGBClassifier
        except ImportError:
            XGBClassifier = None
        
        models = {
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=self.exp_config.random_seed),
            'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, random_state=self.exp_config.random_seed),
            'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=100, random_state=self.exp_config.random_seed),
            'AdaBoostClassifier': AdaBoostClassifier(n_estimators=100, random_state=self.exp_config.random_seed),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=self.exp_config.random_seed),
            'MLPClassifier': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=self.exp_config.random_seed),
            'SVC': SVC(random_state=self.exp_config.random_seed, probability=True),
            'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
            'GaussianNB': GaussianNB(),
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=self.exp_config.random_seed),
            'VotingClassifier': VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=50, random_state=self.exp_config.random_seed)),
                    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=self.exp_config.random_seed)),
                    ('lr', LogisticRegression(random_state=self.exp_config.random_seed, max_iter=1000))
                ],
                voting='soft'
            ),
        }
        
        # Add XGBoost if available
        if XGBClassifier is not None:
            models['XGBClassifier'] = XGBClassifier(n_estimators=100, random_state=self.exp_config.random_seed)
            models['XGBoostClassifier'] = XGBClassifier(n_estimators=100, random_state=self.exp_config.random_seed)
        
        if model_name in models:
            return models[model_name]
        else:
            # Default to RandomForest if model not found
            print(f"⚠️ Model {model_name} not found, using RandomForestClassifier")
            return RandomForestClassifier(n_estimators=100, random_state=self.exp_config.random_seed)
    
    def _create_failure_result(self, model_name: str, error_msg: str) -> Dict[str, Any]:
        """Create a failure result for models that didn't run"""
        return {
            'model_name': model_name,
            'training_time': 0.0,
            'inference_time': 0.0,
            'accuracy': 0.0,
            'close_accuracy': 0.0,
            'star_mae': 5.0,  # Worst possible MAE
            'star_rmse': 5.0,
            'r2_score': -1.0,
            'error': error_msg[:200]  # Truncate long error messages
        }
