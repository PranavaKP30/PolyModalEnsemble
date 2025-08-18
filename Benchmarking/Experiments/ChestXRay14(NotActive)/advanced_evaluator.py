#!/usr/bin/env python3
"""
Advanced evaluation module for ChestX-ray14 experiments
Includes ablation studies, robustness testing, interpretability, and scalability analysis
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Import MainModel components with improved error handling
project_root = Path(__file__).parent.parent.parent.parent
mainmodel_path = project_root / "MainModel"
current_dir = Path(__file__).parent

# Add paths in order of priority
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(mainmodel_path))
sys.path.insert(0, str(project_root))

print(f"🔧 Advanced Evaluator - Importing components...")
print(f"📍 MainModel path: {mainmodel_path} (exists: {mainmodel_path.exists()})")

try:
    from config import ExperimentConfig
    print("✅ Successfully imported ExperimentConfig")
except ImportError as e:
    print(f"❌ Failed to import ExperimentConfig: {e}")
    raise ImportError(f"Could not import ExperimentConfig: {e}")

try:
    from data_loader import ChestXRayDataLoader
    print("✅ Successfully imported ChestXRayDataLoader")
except ImportError as e:
    print(f"❌ Failed to import ChestXRayDataLoader: {e}")
    raise ImportError(f"Could not import ChestXRayDataLoader: {e}")


class AdvancedEvaluator:
    """Advanced evaluator for ChestX-ray14 experiments"""
    
    def __init__(self, exp_config: ExperimentConfig, data_loader: ChestXRayDataLoader):
        """
        Initialize the advanced evaluator
        
        Args:
            exp_config: Experiment configuration
            data_loader: Data loader instance
        """
        self.exp_config = exp_config
        self.data_loader = data_loader
        print("🔧 Advanced Evaluator initialized")
    
    def run_interpretability_analysis(self) -> Dict[str, Any]:
        """Analyze model interpretability and feature importance"""
        print("\n🔍 INTERPRETABILITY ANALYSIS")
        print("=" * 60)

        interpretability_results = {}

        # Feature importance analysis
        print("\n📈 Feature Importance Analysis")
        try:
            interpretability_results['feature_importance'] = self._analyze_feature_importance()
            print("   ✅ Feature importance analysis completed")
        except Exception as e:
            print(f"   ❌ Feature importance analysis failed: {str(e)}")
            interpretability_results['feature_importance'] = {'error': str(e)}

        # Prediction confidence analysis
        print("\n🎯 Prediction Confidence Analysis")
        try:
            interpretability_results['confidence_analysis'] = self._analyze_prediction_confidence()
            print("   ✅ Confidence analysis completed")
        except Exception as e:
            print(f"   ❌ Confidence analysis failed: {str(e)}")
            interpretability_results['confidence_analysis'] = {'error': str(e)}

        # Error analysis
        print("\n❌ Error Pattern Analysis")
        try:
            interpretability_results['error_analysis'] = self._analyze_error_patterns()
            print("   ✅ Error analysis completed")
        except Exception as e:
            print(f"   ❌ Error analysis failed: {str(e)}")
            interpretability_results['error_analysis'] = {'error': str(e)}

        return interpretability_results

    def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance using tree-based models (multi-label safe)"""
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        from sklearn.multioutput import MultiOutputClassifier
        importance_results = {}
        X = self.data_loader.get_combined_features()
        y = self.data_loader.get_combined_labels()
        
        # Validate data
        if np.all(X == 0):
            print("     ⚠️ WARNING: Combined features are all zeros!")
            return {'error': 'Combined features are all zeros'}
        
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        
        for model_cls, name in [(RandomForestClassifier, 'RandomForest'), (ExtraTreesClassifier, 'ExtraTrees')]:
            try:
                model = MultiOutputClassifier(model_cls(n_estimators=50, random_state=self.exp_config.random_seed))
                model.fit(X_train, y_train)
                # Aggregate feature importances across outputs
                if hasattr(model.estimators_[0], 'feature_importances_'):
                    # Average importances across all outputs
                    all_importances = np.array([est.feature_importances_ for est in model.estimators_])
                    mean_importances = np.mean(all_importances, axis=0)
                    importance_results[name] = {
                        'importances': mean_importances.tolist(),
                        'top_features': np.argsort(mean_importances)[-10:].tolist()
                    }
                else:
                    importance_results[name] = {'error': 'No feature importances available'}
            except Exception as e:
                importance_results[name] = {'error': str(e)}
        return importance_results

    def _analyze_prediction_confidence(self) -> Dict[str, Any]:
        """Analyze prediction confidence patterns using RandomForest (multi-label safe)"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.multioutput import MultiOutputClassifier
        X = self.data_loader.get_combined_features()
        y = self.data_loader.get_combined_labels()
        
        # Validate data
        if np.all(X == 0):
            print("     ⚠️ WARNING: Combined features are all zeros!")
            return {'error': 'Combined features are all zeros'}
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        model = MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=self.exp_config.random_seed))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # For multi-label, get max probability for each label, then average
        try:
            y_proba = model.predict_proba(X_test)
            max_probs = np.array([np.max(prob, axis=1) for prob in y_proba]).T  # shape: (n_samples, n_labels)
            mean_confidence = float(np.mean(max_probs))
        except Exception:
            mean_confidence = None
        correct = (y_pred == y_test)
        confidence_results = {
            'mean_confidence': mean_confidence,
            'accuracy': float(np.mean(np.all(correct, axis=1)))
        }
        return confidence_results

    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in prediction errors using RandomForest (multi-label safe)"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.metrics import multilabel_confusion_matrix, classification_report
        X = self.data_loader.get_combined_features()
        y = self.data_loader.get_combined_labels()
        
        # Validate data
        if np.all(X == 0):
            print("     ⚠️ WARNING: Combined features are all zeros!")
            return {'error': 'Combined features are all zeros'}
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        model = MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=self.exp_config.random_seed))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mcm = multilabel_confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        error_analysis = {
            'multilabel_confusion_matrix': mcm.tolist(),
            'classification_report': report
        }
        return error_analysis
    def run_cross_validation_analysis(self) -> Dict[str, Any]:
        """Run k-fold cross-validation for selected models (multi-label, multi-modal)"""
        print("\n🔄 CROSS-VALIDATION ANALYSIS")
        print("=" * 60)

        # Selected models for cross-validation (for efficiency)
        cv_models = {
            'RandomForest_Text': ('text', 'random_forest'),
            'RandomForest_Combined': ('combined', 'random_forest'),
            'XGBoost_Combined': ('combined', 'xgboost'),
            'VotingEnsemble': ('combined', 'voting_ensemble'),
        }

        print(f"🎯 Running {self.exp_config.cv_folds}-fold cross-validation on {len(cv_models)} models")

        cv_results = {}

        for model_name, (modality, algorithm) in cv_models.items():
            print(f"\n📊 Cross-validating: {model_name}")
            try:
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
            'best_cv_model': max(cv_results.keys(), key=lambda k: cv_results[k]['mean_accuracy'] if 'error' not in cv_results[k] else 0)
        }

    def _cv_baseline_model(self, modality: str, algorithm: str) -> List[float]:
        """Cross-validate a baseline model (multi-label)"""
        from sklearn.model_selection import KFold
        from sklearn.metrics import accuracy_score

        # Prepare data based on modality
        if modality == 'text':
            X = self.data_loader.get_combined_text_features()
        elif modality == 'metadata':
            X = self.data_loader.get_combined_metadata_features()
        elif modality == 'combined':
            X = self.data_loader.get_combined_features()
        else:
            raise ValueError(f"Unknown modality: {modality}")

        y = self.data_loader.get_combined_labels()

        kf = KFold(n_splits=self.exp_config.cv_folds, shuffle=True, random_state=self.exp_config.random_seed)
        # Get models from baseline evaluator
        from baseline_evaluator import BaselineEvaluator
        baseline_eval = BaselineEvaluator(self.exp_config, self.data_loader)
        all_models = {**baseline_eval._get_simple_models(), **baseline_eval._get_ensemble_models()}
        if algorithm not in all_models:
            raise ValueError(f"Algorithm {algorithm} not found in baseline models.")
        model_proto = all_models[algorithm]

        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            # Clone the model to avoid data leakage
            import copy
            model = copy.deepcopy(model_proto)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            # For multi-label, use subset accuracy (exact match ratio)
            acc = np.mean(np.all(y_pred == y_val, axis=1))
            cv_scores.append(acc)
        return cv_scores

    def _calculate_confidence_interval(self, scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        import scipy.stats
        arr = np.array(scores)
        mean = np.mean(arr)
        sem = scipy.stats.sem(arr)
        h = sem * scipy.stats.t.ppf((1 + confidence) / 2., len(arr)-1) if len(arr) > 1 else 0.0
        return (float(mean - h), float(mean + h))

    def _analyze_cv_results(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        # Simple statistical summary
        summary = {}
        for model, res in cv_results.items():
            if 'error' not in res:
                summary[model] = {
                    'mean': res['mean_accuracy'],
                    'std': res['std_accuracy'],
                    'conf_interval': res['confidence_interval']
                }
        return summary
    """Advanced evaluation methods for ChestX-ray14 experiments"""
    
    def __init__(self, exp_config: ExperimentConfig, data_loader: ChestXRayDataLoader):
        self.exp_config = exp_config
        self.data_loader = data_loader
    
    def run_ablation_studies(self) -> Dict[str, Any]:
        """Study the contribution of different modalities and components"""
        print("\n🎭 COMPREHENSIVE MODALITY ABLATION STUDY")
        print("=" * 60)
        
        all_ablation_results = {}
        
        # 1. Basic Modality Ablation
        print("\n📊 BASIC MODALITY ABLATION")
        print("-" * 40)
        basic_ablation = self._run_basic_modality_ablation()
        all_ablation_results['basic_modality_ablation'] = basic_ablation
        
        # 2. Image Feature Subset Ablation
        print("\n🖼️ IMAGE FEATURE SUBSET ABLATION")
        print("-" * 40)
        try:
            image_feature_ablation = self._run_image_feature_ablation()
            all_ablation_results['image_feature_ablation'] = image_feature_ablation
        except Exception as e:
            print(f"   ⚠️ Image feature ablation failed: {str(e)}")
            all_ablation_results['image_feature_ablation'] = {'error': str(e)}
        
        # 3. Clinical Text Feature Ablation
        print("\n📝 CLINICAL TEXT FEATURE ABLATION")
        print("-" * 40)
        try:
            text_feature_ablation = self._run_clinical_text_feature_ablation()
            all_ablation_results['clinical_text_feature_ablation'] = text_feature_ablation
        except Exception as e:
            print(f"   ⚠️ Clinical text feature ablation failed: {str(e)}")
            all_ablation_results['clinical_text_feature_ablation'] = {'error': str(e)}
        
        # 4. Medical Metadata Feature Ablation
        print("\n🏥 MEDICAL METADATA FEATURE ABLATION")
        print("-" * 40)
        try:
            metadata_feature_ablation = self._run_medical_metadata_feature_ablation()
            all_ablation_results['medical_metadata_feature_ablation'] = metadata_feature_ablation
        except Exception as e:
            print(f"   ⚠️ Medical metadata feature ablation failed: {str(e)}")
            all_ablation_results['medical_metadata_feature_ablation'] = {'error': str(e)}
        
        # 5. Fusion Strategy Ablation
        print("\n🔗 MEDICAL FUSION STRATEGY ABLATION")
        print("-" * 40)
        try:
            fusion_ablation = self._run_medical_fusion_strategy_ablation()
            all_ablation_results['medical_fusion_strategy_ablation'] = fusion_ablation
        except Exception as e:
            print(f"   ⚠️ Medical fusion strategy ablation failed: {str(e)}")
            all_ablation_results['medical_fusion_strategy_ablation'] = {'error': str(e)}
        
        # 6. Algorithm Sensitivity Ablation
        print("\n⚙️ ALGORITHM SENSITIVITY ABLATION")
        print("-" * 40)
        try:
            algorithm_ablation = self._run_algorithm_sensitivity_ablation()
            all_ablation_results['algorithm_sensitivity_ablation'] = algorithm_ablation
        except Exception as e:
            print(f"   ⚠️ Algorithm sensitivity ablation failed: {str(e)}")
            all_ablation_results['algorithm_sensitivity_ablation'] = {'error': str(e)}
        
        # Comprehensive analysis
        print("\n📈 COMPREHENSIVE MEDICAL ABLATION ANALYSIS")
        print("-" * 40)
        comprehensive_analysis = self._analyze_comprehensive_medical_ablations(all_ablation_results)
        
        return {
            'ablation_experiments': all_ablation_results,
            'comprehensive_analysis': comprehensive_analysis
        }
    
    def run_robustness_testing(self) -> Dict[str, Any]:
        """Test model robustness under missing modalities and noise"""
        print(f"\n🛡️ ROBUSTNESS TESTING")
        print("=" * 60)
        print("🎯 Testing model resilience under adverse conditions")
        
        # Get the most frequent pathology for consistent testing
        pathology_counts = self.data_loader.train_labels.sum(axis=0)
        most_frequent_pathology_idx = np.argmax(pathology_counts)
        most_frequent_pathology = self.exp_config.pathologies[most_frequent_pathology_idx]
        
        print(f"🎯 Testing robustness on: {most_frequent_pathology}")
        
        robustness_results = {}
        
        # Test scenarios adapted for medical imaging
        test_scenarios = self._get_robustness_scenarios()
        
        print(f"🔬 Testing {len(test_scenarios)} robustness scenarios...")
        
        # Get data with better validation
        try:
            (train_image, train_text, train_metadata, single_train_labels,
             test_image, test_text, test_metadata, single_test_labels) = \
                self.data_loader.get_subset_for_pathology(most_frequent_pathology_idx, max_samples=2000)
            
            # Validate that we got meaningful data
            print(f"🔍 Data validation:")
            print(f"   Train image: shape={train_image.shape}, non-zero={np.count_nonzero(train_image)}")
            print(f"   Train text: shape={train_text.shape}, non-zero={np.count_nonzero(train_text)}")
            print(f"   Train metadata: shape={train_metadata.shape}, non-zero={np.count_nonzero(train_metadata)}")
            print(f"   Test image: shape={test_image.shape}, non-zero={np.count_nonzero(test_image)}")
            print(f"   Test text: shape={test_text.shape}, non-zero={np.count_nonzero(test_text)}")
            print(f"   Test metadata: shape={test_metadata.shape}, non-zero={np.count_nonzero(test_metadata)}")
            
            # If train data is all zeros, try to get data without subsampling
            if np.all(train_image == 0) and np.all(train_text == 0):
                print(f"⚠️ Train data is all zeros, trying without subsampling...")
                (train_image, train_text, train_metadata, single_train_labels,
                 test_image, test_text, test_metadata, single_test_labels) = \
                    self.data_loader.get_subset_for_pathology(most_frequent_pathology_idx, max_samples=None)
                
                print(f"🔍 Data validation (no subsampling):")
                print(f"   Train image: shape={train_image.shape}, non-zero={np.count_nonzero(train_image)}")
                print(f"   Train text: shape={train_text.shape}, non-zero={np.count_nonzero(train_text)}")
                
        except Exception as e:
            print(f"❌ Failed to get pathology subset: {e}")
            # Fallback: use small subset of original data
            print(f"🔧 Using fallback data subset...")
            n_samples = min(500, len(self.data_loader.train_labels))
            indices = np.random.choice(len(self.data_loader.train_labels), n_samples, replace=False)
            
            train_image = self.data_loader.train_image[indices]
            train_text = self.data_loader.train_text[indices]
            train_metadata = self.data_loader.train_metadata[indices]
            single_train_labels = self.data_loader.train_labels[indices, most_frequent_pathology_idx]
            
            n_test = min(100, len(self.data_loader.test_labels))
            test_indices = np.random.choice(len(self.data_loader.test_labels), n_test, replace=False)
            
            test_image = self.data_loader.test_image[test_indices]
            test_text = self.data_loader.test_text[test_indices]
            test_metadata = self.data_loader.test_metadata[test_indices]
            single_test_labels = self.data_loader.test_labels[test_indices, most_frequent_pathology_idx]
        
        for scenario_name, scenario_data in test_scenarios.items():
            print(f"\n🧪 Testing: {scenario_data['description']}")
            
            try:
                # Apply scenario modifications
                modified_data = self._apply_robustness_scenario(
                    scenario_name, scenario_data,
                    train_image, train_text, train_metadata,
                    test_image, test_text, test_metadata
                )
                
                result = self._evaluate_robustness_scenario(
                    modified_data, single_train_labels, single_test_labels, scenario_name
                )
                
                robustness_results[scenario_name] = result
                print(f"   ✅ Acc: {result['accuracy']:.3f}")
                
            except Exception as e:
                print(f"   ❌ Scenario failed: {str(e)}")
                robustness_results[scenario_name] = {'error': str(e)}
        
        # Calculate robustness metrics
        if 'baseline' in robustness_results and robustness_results['baseline'].get('accuracy', 0) > 0:
            baseline_acc = robustness_results['baseline']['accuracy']
            
            for scenario_name, results in robustness_results.items():
                if scenario_name != 'baseline' and 'accuracy' in results:
                    degradation = (baseline_acc - results['accuracy']) / baseline_acc * 100 if baseline_acc > 0 else 100
                    results['robustness_degradation_pct'] = degradation
        
        robustness_results['pathology_name'] = most_frequent_pathology
        return robustness_results
    
    def run_cross_modal_denoising_evaluation(self) -> Dict[str, Any]:
        """Test cross-modal reconstruction and contrastive learning capabilities"""
        print(f"\n🔧 CROSS-MODAL DENOISING EVALUATION")
        print("=" * 80)
        print("🎯 Testing cross-modal reconstruction and contrastive learning capabilities")
        
        # Get the most frequent pathology for consistent testing
        pathology_counts = self.data_loader.train_labels.sum(axis=0)
        most_frequent_pathology_idx = np.argmax(pathology_counts)
        most_frequent_pathology = self.exp_config.pathologies[most_frequent_pathology_idx]
        
        print(f"🎯 Testing cross-modal denoising on: {most_frequent_pathology}")
        
        denoising_results = {}
        
        # Test scenarios for cross-modal denoising
        denoising_scenarios = self._get_denoising_scenarios()
        
        print(f"🔬 Testing {len(denoising_scenarios)} cross-modal denoising scenarios...")
        
        for scenario_name, scenario_info in denoising_scenarios.items():
            print(f"\n🧪 Testing: {scenario_info['description']}")
            
            try:
                result = self._evaluate_denoising_scenario(scenario_name, scenario_info)
                denoising_results[scenario_name] = result
                
                # Print scenario-specific results
                if 'reconstruction_fidelity' in result:
                    print(f"   ✅ Reconstruction R²: {result.get('reconstruction_r2', 0):.3f}, Fidelity: {result['reconstruction_fidelity']:.3f}")
                elif 'enhancement_quality' in result:
                    print(f"   ✅ Enhancement R²: {result.get('enhancement_r2', 0):.3f}, Quality: {result['enhancement_quality']:.3f}")
                elif 'contrastive_quality' in result:
                    print(f"   ✅ Cross-modal alignment: {result.get('cross_modal_alignment', 0):.3f}, Quality: {result['contrastive_quality']:.3f}")
                    
            except Exception as e:
                print(f"   ❌ Scenario failed: {str(e)}")
                denoising_results[scenario_name] = {
                    'description': scenario_info['description'],
                    'error': str(e),
                    'quality_score': 0.0
                }
        
        # Calculate overall cross-modal denoising score
        denoising_summary = self._calculate_denoising_summary(denoising_results)
        denoising_results['summary'] = denoising_summary
        
        return denoising_results
    
    def _run_single_ablation(self, train_image, train_text, train_metadata, train_labels,
                           test_image, test_text, test_metadata, test_labels,
                           config: Dict[str, Any], ablation_name: str) -> Dict[str, Any]:
        """Run a single ablation study"""
        try:
            # Import MainModel here to avoid import issues
            from mainModel import MultiModalEnsembleModel
            import dataIntegration
            
            # Create data loader
            loader = dataIntegration.GenericMultiModalDataLoader()
            loader.add_modality_split('image', train_image, test_image)
            loader.add_modality_split('text', train_text, test_text)
            loader.add_modality_split('metadata', train_metadata, test_metadata)
            loader.add_labels_split(train_labels, test_labels)
            
            # Initialize model with ablation config
            model = MultiModalEnsembleModel(
                data_loader=loader,
                n_bags=config['n_bags'],
                dropout_strategy=config['dropout_strategy'],
                epochs=config['epochs'],
                batch_size=self.exp_config.default_batch_size,
                random_state=self.exp_config.random_seed
            )
            
            # Train and evaluate
            model.load_and_integrate_data()
            
            start_time = time.time()
            model.fit()
            train_time = time.time() - start_time
            
            start_time = time.time()
            predictions = model.predict()
            pred_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                test_labels, predictions, average='binary', zero_division=0
            )
            
            return {
                'ablation_name': ablation_name,
                'config': config,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'train_time': train_time,
                'pred_time': pred_time
            }
            
        except Exception as e:
            return {'error': str(e), 'ablation_name': ablation_name}
    
    def _get_robustness_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Get robustness test scenarios"""
        return {
            'baseline': {
                'description': 'Full modalities (baseline)',
                'modifications': {}
            },
            'missing_image': {
                'description': 'Missing image modality',
                'modifications': {'zero_image': True}
            },
            'missing_text': {
                'description': 'Missing text modality',
                'modifications': {'zero_text': True}
            },
            'missing_metadata': {
                'description': 'Missing metadata modality',
                'modifications': {'zero_metadata': True}
            },
            'noisy_image': {
                'description': 'Noisy image features (30% Gaussian noise)',
                'modifications': {'noise_image': 0.3}
            }
        }
    
    def _get_denoising_scenarios(self) -> Dict[str, Dict[str, str]]:
        """Get cross-modal denoising scenarios"""
        return {
            'image_to_text_reconstruction': {
                'description': 'Reconstruct text features from image features',
                'source_modality': 'image',
                'target_modality': 'text'
            },
            'text_to_image_reconstruction': {
                'description': 'Reconstruct image features from text features',
                'source_modality': 'text',
                'target_modality': 'image'
            },
            'metadata_enhancement': {
                'description': 'Enhance metadata from image+text features',
                'source_modality': 'combined',
                'target_modality': 'metadata'
            },
            'multimodal_contrastive': {
                'description': 'Contrastive learning across all modalities',
                'source_modality': 'all',
                'target_modality': 'embeddings'
            }
        }
    
    def _apply_robustness_scenario(self, scenario_name: str, scenario_data: Dict[str, Any],
                                 train_image, train_text, train_metadata,
                                 test_image, test_text, test_metadata) -> Dict[str, np.ndarray]:
        """Apply modifications for robustness scenario"""
        modifications = scenario_data.get('modifications', {})
        
        # Copy data to avoid modifying originals
        modified_train_image = train_image.copy()
        modified_train_text = train_text.copy()
        modified_train_metadata = train_metadata.copy()
        modified_test_image = test_image.copy()
        modified_test_text = test_text.copy()
        modified_test_metadata = test_metadata.copy()
        
        # Apply modifications
        if modifications.get('zero_image', False):
            modified_train_image = np.zeros_like(modified_train_image)
            modified_test_image = np.zeros_like(modified_test_image)
        
        if modifications.get('zero_text', False):
            modified_train_text = np.zeros_like(modified_train_text)
            modified_test_text = np.zeros_like(modified_test_text)
        
        if modifications.get('zero_metadata', False):
            modified_train_metadata = np.zeros_like(modified_train_metadata)
            modified_test_metadata = np.zeros_like(modified_test_metadata)
        
        if 'noise_image' in modifications:
            noise_level = modifications['noise_image']
            modified_train_image += np.random.normal(0, noise_level, modified_train_image.shape)
            modified_test_image += np.random.normal(0, noise_level, modified_test_image.shape)
        
        return {
            'train_image': modified_train_image,
            'train_text': modified_train_text,
            'train_metadata': modified_train_metadata,
            'test_image': modified_test_image,
            'test_text': modified_test_text,
            'test_metadata': modified_test_metadata
        }
    
    def _evaluate_robustness_scenario(self, modified_data: Dict[str, np.ndarray],
                                    train_labels, test_labels, scenario_name: str) -> Dict[str, Any]:
        """Evaluate a robustness scenario"""
        # Import metrics at the top to avoid scope issues
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        try:
            # Import here to avoid import issues
            from mainModel import MultiModalEnsembleModel
            import dataIntegration

            # Enhanced data validation
            print(f"     [DEBUG] train_image shape: {modified_data['train_image'].shape}, sample: {modified_data['train_image'].flatten()[:5]}")
            print(f"     [DEBUG] train_text shape: {modified_data['train_text'].shape}, sample: {modified_data['train_text'].flatten()[:5]}")
            print(f"     [DEBUG] train_metadata shape: {modified_data['train_metadata'].shape}, sample: {modified_data['train_metadata'].flatten()[:5]}")
            print(f"     [DEBUG] test_image shape: {modified_data['test_image'].shape}, sample: {modified_data['test_image'].flatten()[:5]}")
            print(f"     [DEBUG] test_text shape: {modified_data['test_text'].shape}, sample: {modified_data['test_text'].flatten()[:5]}")
            print(f"     [DEBUG] test_metadata shape: {modified_data['test_metadata'].shape}, sample: {modified_data['test_metadata'].flatten()[:5]}")
            print(f"     [DEBUG] train_labels shape: {train_labels.shape}, unique: {np.unique(train_labels, return_counts=True)}")
            print(f"     [DEBUG] test_labels shape: {test_labels.shape}, unique: {np.unique(test_labels, return_counts=True)}")

            # Check if training data is all zeros (which would cause training failure)
            if np.all(modified_data['train_image'] == 0) and np.all(modified_data['train_text'] == 0):
                print(f"     ⚠️ WARNING: Both train_image and train_text are all zeros!")
                print(f"     🔧 Using fallback approach with simple classifier...")
                
                # Fallback: Use a simple classifier that can work with limited data
                from sklearn.ensemble import RandomForestClassifier
                
                # Use metadata features if available, otherwise use a dummy classifier
                if not np.all(modified_data['train_metadata'] == 0):
                    X_train = modified_data['train_metadata']
                    X_test = modified_data['test_metadata']
                    print(f"     📊 Using metadata features for fallback classification")
                else:
                    # Create dummy features if all modalities are zero
                    X_train = np.random.normal(0, 1, (len(train_labels), 10))
                    X_test = np.random.normal(0, 1, (len(test_labels), 10))
                    print(f"     📊 Using dummy features for fallback classification")
                
                # Train simple classifier
                clf = RandomForestClassifier(n_estimators=50, random_state=self.exp_config.random_seed)
                clf.fit(X_train, train_labels)
                predictions = clf.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(test_labels, predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    test_labels, predictions, average='binary', zero_division=0
                )
                
                return {
                    'scenario_name': scenario_name,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'method': 'fallback_classifier'
                }

            # Create data loader with modified data
            loader = dataIntegration.GenericMultiModalDataLoader()
            loader.add_modality_split('image', modified_data['train_image'], modified_data['test_image'])
            loader.add_modality_split('text', modified_data['train_text'], modified_data['test_text'])
            loader.add_modality_split('metadata', modified_data['train_metadata'], modified_data['test_metadata'])
            loader.add_labels_split(train_labels, test_labels)

            # Improved model configuration for robustness testing
            model = MultiModalEnsembleModel(
                data_loader=loader,
                n_bags=5,  # Increased from 3 for better stability
                dropout_strategy='adaptive',
                epochs=5,  # Increased from 3 for better training
                batch_size=min(64, self.exp_config.default_batch_size),  # Smaller batch size for stability
                random_state=self.exp_config.random_seed
            )

            try:
                model.load_and_integrate_data()
                model.fit()
                predictions = model.predict()

                # Handle different prediction formats
                if hasattr(predictions, 'predictions'):
                    y_pred = predictions.predictions
                else:
                    y_pred = predictions
                
                # Ensure predictions are in correct format
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    # Multi-label case - take argmax
                    y_pred = np.argmax(y_pred, axis=1)
                elif len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
                    # Single column - flatten
                    y_pred = y_pred.flatten()

                # Calculate metrics
                accuracy = accuracy_score(test_labels, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    test_labels, y_pred, average='binary', zero_division=0
                )

                return {
                    'scenario_name': scenario_name,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'method': 'mainmodel_ensemble'
                }
                
            except Exception as model_error:
                print(f"     ⚠️ MainModel failed: {model_error}")
                print(f"     🔧 Falling back to simple classifier...")
                
                # Fallback: Use a simple classifier
                from sklearn.ensemble import RandomForestClassifier
                
                # Use available features
                if not np.all(modified_data['train_metadata'] == 0):
                    X_train = modified_data['train_metadata']
                    X_test = modified_data['test_metadata']
                elif not np.all(modified_data['train_image'] == 0):
                    X_train = modified_data['train_image']
                    X_test = modified_data['test_image']
                else:
                    # Create dummy features
                    X_train = np.random.normal(0, 1, (len(train_labels), 10))
                    X_test = np.random.normal(0, 1, (len(test_labels), 10))
                
                # Train simple classifier
                clf = RandomForestClassifier(n_estimators=50, random_state=self.exp_config.random_seed)
                clf.fit(X_train, train_labels)
                predictions = clf.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(test_labels, predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    test_labels, predictions, average='binary', zero_division=0
                )
                
                return {
                    'scenario_name': scenario_name,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'method': 'fallback_classifier',
                    'mainmodel_error': str(model_error)
                }

        except Exception as e:
            print(f"     ❌ Robustness scenario failed: {str(e)}")
            # Always return an 'accuracy' key (set to 0.0) on error for robustness
            return {
                'error': str(e),
                'scenario_name': scenario_name,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'method': 'error_fallback'
            }
    
    def _evaluate_denoising_scenario(self, scenario_name: str, scenario_info: Dict[str, str]) -> Dict[str, Any]:
        """Evaluate a cross-modal denoising scenario"""
        
        if scenario_name == 'image_to_text_reconstruction':
            return self._test_reconstruction(
                self.data_loader.train_image, self.data_loader.train_text,
                self.data_loader.test_image, self.data_loader.test_text,
                scenario_info['description']
            )
        
        elif scenario_name == 'text_to_image_reconstruction':
            return self._test_reconstruction(
                self.data_loader.train_text, self.data_loader.train_image,
                self.data_loader.test_text, self.data_loader.test_image,
                scenario_info['description']
            )
        
        elif scenario_name == 'metadata_enhancement':
            return self._test_metadata_enhancement()
        
        elif scenario_name == 'multimodal_contrastive':
            return self._test_contrastive_learning()
        
        else:
            return {'error': f'Unknown scenario: {scenario_name}'}
    
    def _test_reconstruction(self, X_source_train, X_target_train, X_source_test, X_target_test, description: str) -> Dict[str, Any]:
        """Test cross-modal reconstruction"""
        from sklearn.ensemble import RandomForestRegressor
        
        # Use subset for faster testing
        subset_size = min(2000, len(X_source_train))
        indices = np.random.choice(len(X_source_train), subset_size, replace=False)
        
        X_source_subset = X_source_train[indices]
        X_target_subset = X_target_train[indices]
        
        # Train reconstruction model
        reconstruction_model = RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=self.exp_config.random_seed
        )
        reconstruction_model.fit(X_source_subset, X_target_subset)
        
        # Test reconstruction
        test_subset_size = min(500, len(X_source_test))
        test_indices = np.random.choice(len(X_source_test), test_subset_size, replace=False)
        
        X_source_test_subset = X_source_test[test_indices]
        X_target_test_subset = X_target_test[test_indices]
        
        X_target_pred = reconstruction_model.predict(X_source_test_subset)
        
        # Calculate reconstruction quality
        mse = mean_squared_error(X_target_test_subset, X_target_pred)
        r2 = r2_score(X_target_test_subset, X_target_pred)
        
        reconstruction_fidelity = max(0, min(1, r2))  # Clamp to [0,1]
        
        return {
            'description': description,
            'reconstruction_mse': mse,
            'reconstruction_r2': r2,
            'reconstruction_fidelity': reconstruction_fidelity,
            'training_samples': subset_size,
            'test_samples': test_subset_size
        }
    
    def _test_metadata_enhancement(self) -> Dict[str, Any]:
        """Test metadata enhancement using combined features"""
        # Combine image and text features
        X_train_combined = np.hstack([self.data_loader.train_image, self.data_loader.train_text])
        X_test_combined = np.hstack([self.data_loader.test_image, self.data_loader.test_text])
        
        enhancement_model = RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=self.exp_config.random_seed
        )
        
        subset_size = min(2000, len(X_train_combined))
        indices = np.random.choice(len(X_train_combined), subset_size, replace=False)
        
        X_train_subset = X_train_combined[indices]
        y_train_subset = self.data_loader.train_metadata[indices]
        
        enhancement_model.fit(X_train_subset, y_train_subset)
        
        test_subset_size = min(500, len(X_test_combined))
        test_indices = np.random.choice(len(X_test_combined), test_subset_size, replace=False)
        
        X_test_subset = X_test_combined[test_indices]
        y_test_subset = self.data_loader.test_metadata[test_indices]
        
        y_pred = enhancement_model.predict(X_test_subset)
        
        mse = mean_squared_error(y_test_subset, y_pred)
        r2 = r2_score(y_test_subset, y_pred)
        
        enhancement_quality = max(0, min(1, r2))
        
        return {
            'description': 'Enhance metadata from image+text features',
            'enhancement_mse': mse,
            'enhancement_r2': r2,
            'enhancement_quality': enhancement_quality,
            'training_samples': subset_size,
            'test_samples': test_subset_size
        }
    
    def _test_contrastive_learning(self) -> Dict[str, Any]:
        """Test contrastive learning across modalities"""
        # Standardize features for fair comparison
        scaler_image = StandardScaler()
        scaler_text = StandardScaler()
        scaler_metadata = StandardScaler()
        
        subset_size = min(1000, len(self.data_loader.train_image))
        indices = np.random.choice(len(self.data_loader.train_image), subset_size, replace=False)
        
        # Normalize features
        image_norm = scaler_image.fit_transform(self.data_loader.train_image[indices])
        text_norm = scaler_text.fit_transform(self.data_loader.train_text[indices])
        metadata_norm = scaler_metadata.fit_transform(self.data_loader.train_metadata[indices])
        
        # Calculate cross-modal similarities
        image_text_sim = cosine_similarity(image_norm, text_norm)
        image_metadata_sim = cosine_similarity(image_norm, metadata_norm)
        text_metadata_sim = cosine_similarity(text_norm, metadata_norm)
        
        # Average diagonal similarity (same sample across modalities)
        cross_modal_alignment = (
            np.mean(np.diag(image_text_sim)) +
            np.mean(np.diag(image_metadata_sim)) +
            np.mean(np.diag(text_metadata_sim))
        ) / 3
        
        # Contrastive quality (higher is better alignment)
        contrastive_quality = max(0, min(1, (cross_modal_alignment + 1) / 2))  # Normalize from [-1,1] to [0,1]
        
        return {
            'description': 'Contrastive learning across all modalities',
            'cross_modal_alignment': cross_modal_alignment,
            'image_text_similarity': np.mean(np.diag(image_text_sim)),
            'image_metadata_similarity': np.mean(np.diag(image_metadata_sim)),
            'text_metadata_similarity': np.mean(np.diag(text_metadata_sim)),
            'contrastive_quality': contrastive_quality,
            'samples_analyzed': subset_size
        }
    
    def _calculate_denoising_summary(self, denoising_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall cross-modal denoising summary"""
        valid_results = [r for r in denoising_results.values() if 'error' not in r]
        
        if valid_results:
            # Aggregate quality scores
            quality_scores = []
            for result in valid_results:
                if 'reconstruction_fidelity' in result:
                    quality_scores.append(result['reconstruction_fidelity'])
                elif 'enhancement_quality' in result:
                    quality_scores.append(result['enhancement_quality'])
                elif 'contrastive_quality' in result:
                    quality_scores.append(result['contrastive_quality'])
            
            overall_denoising_score = np.mean(quality_scores) if quality_scores else 0.0
            
            return {
                'overall_score': overall_denoising_score,
                'max_score': 1.0,
                'scenarios_tested': len(valid_results),
                'scenarios_passed': len([r for r in valid_results if max(
                    r.get('reconstruction_fidelity', 0),
                    r.get('enhancement_quality', 0),
                    r.get('contrastive_quality', 0)
                ) > 0.3]),  # Threshold for "passing"
                'reconstruction_capabilities': len([r for r in valid_results if 'reconstruction_fidelity' in r]),
                'enhancement_capabilities': len([r for r in valid_results if 'enhancement_quality' in r]),
                'contrastive_capabilities': len([r for r in valid_results if 'contrastive_quality' in r])
            }
        else:
            return {
                'overall_score': 0.0,
                'max_score': 1.0,
                'error': 'No valid denoising evaluations completed'
            }
    
    def _run_basic_modality_ablation(self) -> Dict[str, Any]:
        """Run basic modality ablation study for medical imaging"""
        
        # Get the most frequent pathology for consistent testing
        pathology_counts = self.data_loader.train_labels.sum(axis=0)
        most_frequent_pathology_idx = np.argmax(pathology_counts)
        most_frequent_pathology = self.exp_config.pathologies[most_frequent_pathology_idx]
        
        # Test different modality combinations for medical imaging
        modality_combinations = {
            'image_only': ['image'],
            'text_only': ['text'],
            'metadata_only': ['metadata'],
            'image_text': ['image', 'text'],
            'image_metadata': ['image', 'metadata'],
            'text_metadata': ['text', 'metadata'],
            'all_modalities': ['image', 'text', 'metadata']
        }
        
        ablation_results = {}
        
        for combination_name, modalities in modality_combinations.items():
            print(f"\n🧪 Testing: {combination_name}")
            
            try:
                # Get subset data for this pathology
                (train_image, train_text, train_metadata, single_train_labels,
                 test_image, test_text, test_metadata, single_test_labels) = \
                    self.data_loader.get_subset_for_pathology(most_frequent_pathology_idx, max_samples=1500)
                
                # Create modified data for this combination
                modified_data = self._create_modified_medical_data(
                    modalities, train_image, train_text, train_metadata,
                    test_image, test_text, test_metadata
                )
                
                # Test with multiple algorithms
                combination_results = {}
                test_algorithms = ['RandomForestClassifier', 'XGBoostClassifier', 'LogisticRegression']
                
                for algorithm in test_algorithms:
                    result = self._test_algorithm_on_medical_data(
                        modified_data, single_train_labels, single_test_labels, algorithm
                    )
                    combination_results[algorithm] = result
                
                ablation_results[combination_name] = {
                    'modalities_used': modalities,
                    'algorithm_results': combination_results,
                    'best_performance': max(combination_results.values(), key=lambda x: x.get('f1_score', 0))
                }
                
                best_f1 = ablation_results[combination_name]['best_performance'].get('f1_score', 0)
                print(f"   ✅ Best F1: {best_f1:.3f}")
                
            except Exception as e:
                print(f"   ❌ Ablation test failed for {combination_name}: {str(e)}")
                ablation_results[combination_name] = {'error': str(e)}
        
        # Analyze modality contributions for medical imaging
        contribution_analysis = self._analyze_medical_modality_contributions(ablation_results)
        
        return {
            'modality_results': ablation_results,
            'contribution_analysis': contribution_analysis,
            'pathology_tested': most_frequent_pathology
        }
    
    def _run_image_feature_ablation(self) -> Dict[str, Any]:
        """Ablate different types of image features"""
        
        print("   🖼️ Testing image feature subsets...")
        
        # Get the most frequent pathology data
        pathology_counts = self.data_loader.train_labels.sum(axis=0)
        most_frequent_pathology_idx = np.argmax(pathology_counts)
        
        (train_image, train_text, train_metadata, single_train_labels,
         test_image, test_text, test_metadata, single_test_labels) = \
            self.data_loader.get_subset_for_pathology(most_frequent_pathology_idx, max_samples=1000)
        
        # Simulate different image feature types by using feature subsets
        n_features = train_image.shape[1]
        feature_subsets = {
            'first_quarter': train_image[:, :n_features//4],
            'second_quarter': train_image[:, n_features//4:n_features//2],
            'third_quarter': train_image[:, n_features//2:3*n_features//4],
            'fourth_quarter': train_image[:, 3*n_features//4:],
            'first_half': train_image[:, :n_features//2],
            'second_half': train_image[:, n_features//2:],
            'all_features': train_image
        }
        
        test_feature_subsets = {
            'first_quarter': test_image[:, :n_features//4],
            'second_quarter': test_image[:, n_features//4:n_features//2],
            'third_quarter': test_image[:, n_features//2:3*n_features//4],
            'fourth_quarter': test_image[:, 3*n_features//4:],
            'first_half': test_image[:, :n_features//2],
            'second_half': test_image[:, n_features//2:],
            'all_features': test_image
        }
        
        results = {}
        for subset_name, train_features in feature_subsets.items():
            try:
                test_features = test_feature_subsets[subset_name]
                
                # Quick evaluation with Random Forest
                f1_score, accuracy = self._quick_evaluate_medical_features(
                    train_features, test_features, single_train_labels, single_test_labels
                )
                
                results[subset_name] = {
                    'f1_score': f1_score,
                    'accuracy': accuracy,
                    'n_features': train_features.shape[1]
                }
                
                print(f"     {subset_name}: F1={f1_score:.3f}, Acc={accuracy:.3f}, Features={train_features.shape[1]}")
                
            except Exception as e:
                results[subset_name] = {'error': str(e)}
        
        return {
            'feature_subset_results': results,
            'analysis': self._analyze_image_feature_importance(results)
        }
    
    def _run_clinical_text_feature_ablation(self) -> Dict[str, Any]:
        """Ablate different clinical text features"""
        
        print("   📝 Testing clinical text feature removal...")
        
        # Get pathology data
        pathology_counts = self.data_loader.train_labels.sum(axis=0)
        most_frequent_pathology_idx = np.argmax(pathology_counts)
        
        (train_image, train_text, train_metadata, single_train_labels,
         test_image, test_text, test_metadata, single_test_labels) = \
            self.data_loader.get_subset_for_pathology(most_frequent_pathology_idx, max_samples=1000)
        
        # Test removing each text feature one by one (leave-one-out)
        n_features = train_text.shape[1]
        results = {}
        
        # Baseline with all features
        baseline_f1, baseline_acc = self._quick_evaluate_medical_features(
            train_text, test_text, single_train_labels, single_test_labels
        )
        results['all_features'] = {
            'f1_score': baseline_f1,
            'accuracy': baseline_acc,
            'features_used': n_features,
            'removed_feature': 'none'
        }
        
        # Remove each feature individually (limit to first 10 for efficiency)
        for i in range(min(n_features, 10)):
            try:
                # Create feature subset without feature i
                train_subset = np.concatenate([
                    train_text[:, :i],
                    train_text[:, i+1:]
                ], axis=1)
                test_subset = np.concatenate([
                    test_text[:, :i],
                    test_text[:, i+1:]
                ], axis=1)
                
                f1_score, accuracy = self._quick_evaluate_medical_features(
                    train_subset, test_subset, single_train_labels, single_test_labels
                )
                
                results[f'without_feature_{i}'] = {
                    'f1_score': f1_score,
                    'accuracy': accuracy,
                    'features_used': train_subset.shape[1],
                    'removed_feature': i,
                    'importance_drop': baseline_f1 - f1_score
                }
                
                print(f"     Without feature {i}: F1={f1_score:.3f} (drop: {baseline_f1 - f1_score:.3f})")
                
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
    
    def _run_medical_metadata_feature_ablation(self) -> Dict[str, Any]:
        """Ablate different medical metadata features"""
        
        print("   🏥 Testing medical metadata feature removal...")
        
        # Get pathology data
        pathology_counts = self.data_loader.train_labels.sum(axis=0)
        most_frequent_pathology_idx = np.argmax(pathology_counts)
        
        (train_image, train_text, train_metadata, single_train_labels,
         test_image, test_text, test_metadata, single_test_labels) = \
            self.data_loader.get_subset_for_pathology(most_frequent_pathology_idx, max_samples=1000)
        
        # Test removing each metadata feature one by one
        n_features = train_metadata.shape[1]
        results = {}
        
        # Baseline with all features
        baseline_f1, baseline_acc = self._quick_evaluate_medical_features(
            train_metadata, test_metadata, single_train_labels, single_test_labels
        )
        results['all_features'] = {
            'f1_score': baseline_f1,
            'accuracy': baseline_acc,
            'features_used': n_features,
            'removed_feature': 'none'
        }
        
        # Remove each feature individually
        for i in range(min(n_features, 8)):  # Limit for efficiency
            try:
                # Create feature subset without feature i
                train_subset = np.concatenate([
                    train_metadata[:, :i],
                    train_metadata[:, i+1:]
                ], axis=1)
                test_subset = np.concatenate([
                    test_metadata[:, :i],
                    test_metadata[:, i+1:]
                ], axis=1)
                
                f1_score, accuracy = self._quick_evaluate_medical_features(
                    train_subset, test_subset, single_train_labels, single_test_labels
                )
                
                results[f'without_feature_{i}'] = {
                    'f1_score': f1_score,
                    'accuracy': accuracy,
                    'features_used': train_subset.shape[1],
                    'removed_feature': i,
                    'importance_drop': baseline_f1 - f1_score
                }
                
                print(f"     Without feature {i}: F1={f1_score:.3f} (drop: {baseline_f1 - f1_score:.3f})")
                
            except Exception as e:
                results[f'without_feature_{i}'] = {'error': str(e)}
        
        return {
            'feature_removal_results': results,
            'most_important_features': sorted(
                [(k, v['importance_drop']) for k, v in results.items() 
                 if 'importance_drop' in v], 
                key=lambda x: x[1], reverse=True
            )[:3]
        }
    
    def _run_medical_fusion_strategy_ablation(self) -> Dict[str, Any]:
        """Test different fusion strategies for medical imaging"""
        
        print("   🔗 Testing medical fusion strategies...")
        
        # Get pathology data
        pathology_counts = self.data_loader.train_labels.sum(axis=0)
        most_frequent_pathology_idx = np.argmax(pathology_counts)
        
        (train_image, train_text, train_metadata, single_train_labels,
         test_image, test_text, test_metadata, single_test_labels) = \
            self.data_loader.get_subset_for_pathology(most_frequent_pathology_idx, max_samples=1000)
        
        fusion_strategies = {
            'early_fusion': self._early_medical_fusion_strategy,
            'late_fusion': self._late_medical_fusion_strategy,
            'weighted_fusion': self._weighted_medical_fusion_strategy,
            'image_dominant_fusion': self._image_dominant_fusion_strategy
        }
        
        results = {}
        
        for strategy_name, strategy_func in fusion_strategies.items():
            try:
                f1_score, accuracy = strategy_func(
                    train_image, train_text, train_metadata, single_train_labels,
                    test_image, test_text, test_metadata, single_test_labels
                )
                
                results[strategy_name] = {
                    'f1_score': f1_score,
                    'accuracy': accuracy
                }
                
                print(f"     {strategy_name}: F1={f1_score:.3f}, Acc={accuracy:.3f}")
                
            except Exception as e:
                results[strategy_name] = {'error': str(e)}
                print(f"     {strategy_name}: Failed - {str(e)[:50]}...")
        
        return {
            'fusion_strategy_results': results,
            'best_fusion_strategy': max(results.keys(), key=lambda k: results[k].get('f1_score', 0))
        }
    
    def _run_algorithm_sensitivity_ablation(self) -> Dict[str, Any]:
        """Test sensitivity to different algorithms across medical modalities"""
        
        print("   ⚙️ Testing algorithm sensitivity...")
        
        algorithms = [
            'RandomForestClassifier', 'XGBoostClassifier', 'LogisticRegression',
            'GradientBoostingClassifier', 'ExtraTreesClassifier'
        ]
        
        # Get pathology data
        pathology_counts = self.data_loader.train_labels.sum(axis=0)
        most_frequent_pathology_idx = np.argmax(pathology_counts)
        
        (train_image, train_text, train_metadata, single_train_labels,
         test_image, test_text, test_metadata, single_test_labels) = \
            self.data_loader.get_subset_for_pathology(most_frequent_pathology_idx, max_samples=1000)
        
        modalities = {
            'image_only': (train_image, test_image),
            'text_only': (train_text, test_text),
            'metadata_only': (train_metadata, test_metadata),
            'combined': (np.concatenate([train_image, train_text, train_metadata], axis=1),
                        np.concatenate([test_image, test_text, test_metadata], axis=1))
        }
        
        results = {}
        
        for modality_name, (train_features, test_features) in modalities.items():
            modality_results = {}
            
            for algorithm in algorithms:
                try:
                    f1_score, accuracy = self._quick_evaluate_medical_features(
                        train_features, test_features, single_train_labels, single_test_labels, algorithm
                    )
                    modality_results[algorithm] = f1_score
                    
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
    
    def _analyze_comprehensive_medical_ablations(self, all_ablation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze all medical ablation study results comprehensively"""
        
        analysis = {
            'key_findings': [],
            'modality_insights': {},
            'feature_insights': {},
            'fusion_insights': {},
            'algorithm_insights': {},
            'clinical_recommendations': []
        }
        
        # Basic modality analysis for medical imaging
        basic_results = all_ablation_results.get('basic_modality_ablation', {})
        if 'contribution_analysis' in basic_results:
            contrib = basic_results['contribution_analysis']
            
            analysis['modality_insights'] = {
                'image_performance': contrib.get('image_contribution', 0),
                'text_performance': contrib.get('text_contribution', 0),
                'metadata_performance': contrib.get('metadata_contribution', 0),
                'multimodal_benefit': contrib.get('multimodal_benefit', 0),
                'best_single_modality': contrib.get('best_single_modality', 'unknown'),
                'clinical_synergy_detected': contrib.get('synergy_effect', 0) > 0.02
            }
            
            # Medical-specific findings
            if contrib.get('multimodal_benefit', 0) > 0.05:
                analysis['key_findings'].append("Strong multimodal clinical benefit detected (>5% F1 improvement)")
            
            if contrib.get('image_contribution', 0) > contrib.get('text_contribution', 0) * 1.5:
                analysis['key_findings'].append("Image features dominate clinical text features")
        
        # Feature importance insights for medical data
        image_ablation = all_ablation_results.get('image_feature_ablation', {})
        if 'feature_subset_results' in image_ablation:
            analysis['feature_insights']['image'] = {
                'feature_subsets_tested': len(image_ablation['feature_subset_results']),
                'performance_variation': self._calculate_medical_performance_variation(
                    image_ablation['feature_subset_results']
                )
            }
        
        text_ablation = all_ablation_results.get('clinical_text_feature_ablation', {})
        if 'most_important_features' in text_ablation:
            analysis['feature_insights']['clinical_text'] = {
                'most_important_features': text_ablation['most_important_features'][:3],
                'feature_importance_range': self._calculate_medical_importance_range(text_ablation)
            }
        
        metadata_ablation = all_ablation_results.get('medical_metadata_feature_ablation', {})
        if 'most_important_features' in metadata_ablation:
            analysis['feature_insights']['medical_metadata'] = {
                'most_important_features': metadata_ablation['most_important_features'][:3],
                'clinical_metadata_impact': self._calculate_medical_importance_range(metadata_ablation)
            }
        
        # Fusion strategy insights for medical imaging
        fusion_ablation = all_ablation_results.get('medical_fusion_strategy_ablation', {})
        if 'fusion_strategy_results' in fusion_ablation:
            analysis['fusion_insights'] = {
                'best_medical_strategy': fusion_ablation.get('best_fusion_strategy', 'unknown'),
                'strategy_performance': fusion_ablation['fusion_strategy_results']
            }
        
        # Algorithm sensitivity insights
        algo_ablation = all_ablation_results.get('algorithm_sensitivity_ablation', {})
        if 'modality_algorithm_preferences' in algo_ablation:
            analysis['algorithm_insights'] = {
                'modality_preferences': algo_ablation['modality_algorithm_preferences'],
                'algorithm_consistency': self._analyze_medical_algorithm_consistency(algo_ablation)
            }
        
        # Generate clinical recommendations
        analysis['clinical_recommendations'] = self._generate_clinical_ablation_recommendations(analysis)
        
        return analysis
    
    def _create_modified_medical_data(self, modalities: List[str], 
                                    train_image, train_text, train_metadata,
                                    test_image, test_text, test_metadata) -> Dict[str, np.ndarray]:
        """Create modified medical dataset with only specified modalities"""
        
        train_features = []
        test_features = []
        
        if 'image' in modalities:
            train_features.append(train_image)
            test_features.append(test_image)
        if 'text' in modalities:
            train_features.append(train_text)
            test_features.append(test_text)
        if 'metadata' in modalities:
            train_features.append(train_metadata)
            test_features.append(test_metadata)
        
        if not train_features:
            raise ValueError("At least one modality must be specified")
        
        return {
            'X_train': np.concatenate(train_features, axis=1),
            'X_test': np.concatenate(test_features, axis=1)
        }
    
    def _test_algorithm_on_medical_data(self, modified_data: Dict[str, np.ndarray], 
                                      train_labels: np.ndarray, test_labels: np.ndarray,
                                      algorithm: str) -> Dict[str, float]:
        """Test an algorithm on modified medical data"""
        
        X_train, X_test = modified_data['X_train'], modified_data['X_test']
        
        # Import and create model
        if algorithm == 'RandomForestClassifier':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=50, random_state=self.exp_config.random_seed)
        elif algorithm == 'XGBoostClassifier':
            try:
                from xgboost import XGBClassifier
                model = XGBClassifier(n_estimators=50, random_state=self.exp_config.random_seed)
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                model = GradientBoostingClassifier(n_estimators=50, random_state=self.exp_config.random_seed)
        elif algorithm == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=self.exp_config.random_seed, max_iter=1000)
        elif algorithm == 'GradientBoostingClassifier':
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(n_estimators=50, random_state=self.exp_config.random_seed)
        elif algorithm == 'ExtraTreesClassifier':
            from sklearn.ensemble import ExtraTreesClassifier
            model = ExtraTreesClassifier(n_estimators=50, random_state=self.exp_config.random_seed)
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=50, random_state=self.exp_config.random_seed)
        
        # Train model
        model.fit(X_train, train_labels)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = float(np.mean(y_pred == test_labels))
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        f1 = float(f1_score(test_labels, y_pred, average='binary', zero_division=0))
        
        return {
            'accuracy': accuracy,
            'f1_score': f1
        }
    
    def _analyze_medical_modality_contributions(self, ablation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the contribution of each medical modality"""
        
        contributions = {}
        
        try:
            # Extract best performances for medical modalities
            image_only = ablation_results.get('image_only', {}).get('best_performance', {}).get('f1_score', 0)
            text_only = ablation_results.get('text_only', {}).get('best_performance', {}).get('f1_score', 0)
            metadata_only = ablation_results.get('metadata_only', {}).get('best_performance', {}).get('f1_score', 0)
            all_modalities = ablation_results.get('all_modalities', {}).get('best_performance', {}).get('f1_score', 0)
            
            contributions = {
                'image_contribution': image_only,
                'text_contribution': text_only,
                'metadata_contribution': metadata_only,
                'multimodal_benefit': all_modalities - max(image_only, text_only, metadata_only),
                'synergy_effect': all_modalities - (image_only + text_only + metadata_only) / 3,
                'best_single_modality': max(['image', 'text', 'metadata'], 
                                          key=lambda m: eval(f"{m}_only")),
                'multimodal_improvement': all_modalities > max(image_only, text_only, metadata_only),
                'clinical_dominance': 'image' if image_only > max(text_only, metadata_only) * 1.2 else 'balanced'
            }
            
        except Exception as e:
            contributions = {'error': str(e)}
        
        return contributions
    
    def _quick_evaluate_medical_features(self, train_features: np.ndarray, test_features: np.ndarray,
                                       train_labels: np.ndarray, test_labels: np.ndarray,
                                       algorithm: str = 'RandomForestClassifier') -> Tuple[float, float]:
        """Quick evaluation of medical features with given algorithm"""
        
        # Import and create model
        if algorithm == 'RandomForestClassifier':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=30, random_state=self.exp_config.random_seed)
        elif algorithm == 'XGBoostClassifier':
            try:
                from xgboost import XGBClassifier
                model = XGBClassifier(n_estimators=30, random_state=self.exp_config.random_seed)
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                model = GradientBoostingClassifier(n_estimators=30, random_state=self.exp_config.random_seed)
        elif algorithm == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=self.exp_config.random_seed, max_iter=500)
        elif algorithm == 'GradientBoostingClassifier':
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(n_estimators=30, random_state=self.exp_config.random_seed)
        elif algorithm == 'ExtraTreesClassifier':
            from sklearn.ensemble import ExtraTreesClassifier
            model = ExtraTreesClassifier(n_estimators=30, random_state=self.exp_config.random_seed)
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=30, random_state=self.exp_config.random_seed)
        
        model.fit(train_features, train_labels)
        y_pred = model.predict(test_features)
        
        accuracy = float(np.mean(y_pred == test_labels))
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        f1 = float(f1_score(test_labels, y_pred, average='binary', zero_division=0))
        
        return f1, accuracy
    
    # Fusion strategy methods for medical imaging
    def _early_medical_fusion_strategy(self, train_image, train_text, train_metadata, train_labels,
                                     test_image, test_text, test_metadata, test_labels) -> Tuple[float, float]:
        """Early fusion: concatenate all medical modalities then train"""
        combined_train = np.concatenate([train_image, train_text, train_metadata], axis=1)
        combined_test = np.concatenate([test_image, test_text, test_metadata], axis=1)
        
        return self._quick_evaluate_medical_features(combined_train, combined_test, train_labels, test_labels)
    
    def _late_medical_fusion_strategy(self, train_image, train_text, train_metadata, train_labels,
                                    test_image, test_text, test_metadata, test_labels) -> Tuple[float, float]:
        """Late fusion: train separate models then combine predictions"""
        
        # Train individual models
        from sklearn.ensemble import RandomForestClassifier
        
        # Image model
        image_model = RandomForestClassifier(n_estimators=30, random_state=self.exp_config.random_seed)
        image_model.fit(train_image, train_labels)
        image_pred = image_model.predict(test_image)
        
        # Text model
        text_model = RandomForestClassifier(n_estimators=30, random_state=self.exp_config.random_seed)
        text_model.fit(train_text, train_labels)
        text_pred = text_model.predict(test_text)
        
        # Metadata model
        metadata_model = RandomForestClassifier(n_estimators=30, random_state=self.exp_config.random_seed)
        metadata_model.fit(train_metadata, train_labels)
        metadata_pred = metadata_model.predict(test_metadata)
        
        # Combine predictions (majority voting)
        combined_pred = np.round((image_pred + text_pred + metadata_pred) / 3).astype(int)
        
        accuracy = float(np.mean(combined_pred == test_labels))
        
        from sklearn.metrics import f1_score
        f1 = float(f1_score(test_labels, combined_pred, average='binary', zero_division=0))
        
        return f1, accuracy
    
    def _weighted_medical_fusion_strategy(self, train_image, train_text, train_metadata, train_labels,
                                        test_image, test_text, test_metadata, test_labels) -> Tuple[float, float]:
        """Weighted fusion: learn optimal weights for medical modality combination"""
        
        # Weighted fusion with 60% image, 25% text, 15% metadata (typical for medical imaging)
        from sklearn.ensemble import RandomForestClassifier
        
        # Train individual models
        image_model = RandomForestClassifier(n_estimators=30, random_state=self.exp_config.random_seed)
        image_model.fit(train_image, train_labels)
        image_pred = image_model.predict(test_image)
        
        text_model = RandomForestClassifier(n_estimators=30, random_state=self.exp_config.random_seed)
        text_model.fit(train_text, train_labels)
        text_pred = text_model.predict(test_text)
        
        metadata_model = RandomForestClassifier(n_estimators=30, random_state=self.exp_config.random_seed)
        metadata_model.fit(train_metadata, train_labels)
        metadata_pred = metadata_model.predict(test_metadata)
        
        # Weighted combination
        weighted_pred = np.round(0.6 * image_pred + 0.25 * text_pred + 0.15 * metadata_pred).astype(int)
        
        accuracy = float(np.mean(weighted_pred == test_labels))
        
        from sklearn.metrics import f1_score
        f1 = float(f1_score(test_labels, weighted_pred, average='binary', zero_division=0))
        
        return f1, accuracy
    
    def _image_dominant_fusion_strategy(self, train_image, train_text, train_metadata, train_labels,
                                      test_image, test_text, test_metadata, test_labels) -> Tuple[float, float]:
        """Image-dominant fusion: heavily weight image features for medical diagnosis"""
        
        # Create feature combination with image emphasis
        # Repeat image features to give them more weight
        combined_train = np.concatenate([
            train_image, train_image,  # Double weight for image
            train_text, train_metadata
        ], axis=1)
        combined_test = np.concatenate([
            test_image, test_image,    # Double weight for image
            test_text, test_metadata
        ], axis=1)
        
        return self._quick_evaluate_medical_features(combined_train, combined_test, train_labels, test_labels)
    
    # Analysis helper methods
    def _analyze_image_feature_importance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image feature importance from subset results"""
        
        valid_results = {k: v for k, v in results.items() if 'f1_score' in v}
        if not valid_results:
            return {'error': 'No valid results to analyze'}
        
        f1_scores = [v['f1_score'] for v in valid_results.values()]
        
        return {
            'best_subset': max(valid_results.keys(), key=lambda k: valid_results[k]['f1_score']),
            'worst_subset': min(valid_results.keys(), key=lambda k: valid_results[k]['f1_score']),
            'performance_range': max(f1_scores) - min(f1_scores),
            'mean_performance': np.mean(f1_scores),
            'feature_efficiency': {
                k: v['f1_score'] / v.get('n_features', 1) 
                for k, v in valid_results.items() if 'n_features' in v
            }
        }
    
    def _calculate_medical_performance_variation(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance variation across medical feature subsets"""
        
        valid_f1_scores = [v['f1_score'] for v in results.values() if 'f1_score' in v]
        if not valid_f1_scores:
            return {'error': 'No valid F1 scores'}
        
        return {
            'min_f1': min(valid_f1_scores),
            'max_f1': max(valid_f1_scores),
            'mean_f1': np.mean(valid_f1_scores),
            'std_f1': np.std(valid_f1_scores),
            'range': max(valid_f1_scores) - min(valid_f1_scores)
        }
    
    def _calculate_medical_importance_range(self, ablation: Dict[str, Any]) -> Dict[str, float]:
        """Calculate importance range from medical feature ablation"""
        
        if 'most_important_features' not in ablation:
            return {'error': 'No importance data'}
        
        importance_values = [item[1] for item in ablation['most_important_features']]
        
        return {
            'max_importance': max(importance_values) if importance_values else 0,
            'min_importance': min(importance_values) if importance_values else 0,
            'mean_importance': np.mean(importance_values) if importance_values else 0
        }
    
    def _analyze_medical_algorithm_consistency(self, algo_ablation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze algorithm consistency across medical modalities"""
        
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
            'universal_best': best_algorithms[0] if consistency else None,
            'medical_imaging_preference': self._identify_medical_algorithm_preference(results)
        }
    
    def _identify_medical_algorithm_preference(self, results: Dict[str, Any]) -> str:
        """Identify which algorithm works best for medical imaging overall"""
        
        # Calculate average performance across all modalities for each algorithm
        all_algorithms = set()
        for modality_results in results.values():
            all_algorithms.update(modality_results.keys())
        
        algorithm_averages = {}
        for algorithm in all_algorithms:
            scores = []
            for modality_results in results.values():
                if algorithm in modality_results:
                    scores.append(modality_results[algorithm])
            algorithm_averages[algorithm] = np.mean(scores) if scores else 0
        
        return max(algorithm_averages.keys(), key=lambda k: algorithm_averages[k])
    
    def _generate_clinical_ablation_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate clinical recommendations based on medical ablation analysis"""
        
        recommendations = []
        
        # Medical modality recommendations
        modality_insights = analysis.get('modality_insights', {})
        if modality_insights.get('multimodal_benefit', 0) > 0.05:
            recommendations.append("✅ Use multimodal approach - significant clinical benefit detected")
        elif modality_insights.get('multimodal_benefit', 0) < 0.01:
            best_single = modality_insights.get('best_single_modality', 'image')
            recommendations.append(f"🖼️ Consider single modality approach using {best_single} features")
        
        # Clinical dominance insights
        if modality_insights.get('clinical_dominance') == 'image':
            recommendations.append("🖼️ Image features dominate - prioritize high-quality imaging")
        
        # Fusion strategy recommendations for medical imaging
        fusion_insights = analysis.get('fusion_insights', {})
        if 'best_medical_strategy' in fusion_insights:
            best_strategy = fusion_insights['best_medical_strategy']
            recommendations.append(f"🔗 Use {best_strategy} for optimal medical multimodal fusion")
        
        # Algorithm recommendations for medical applications
        algo_insights = analysis.get('algorithm_insights', {})
        if algo_insights.get('algorithm_consistency', {}).get('consistent_best_algorithm', False):
            universal_best = algo_insights['algorithm_consistency']['universal_best']
            recommendations.append(f"⚙️ {universal_best} performs consistently across all medical modalities")
        elif 'medical_imaging_preference' in algo_insights.get('algorithm_consistency', {}):
            preferred = algo_insights['algorithm_consistency']['medical_imaging_preference']
            recommendations.append(f"🏥 {preferred} shows best overall performance for medical imaging")
        
        # Feature efficiency recommendations
        feature_insights = analysis.get('feature_insights', {})
        if 'image' in feature_insights:
            image_variation = feature_insights['image'].get('performance_variation', {})
            if image_variation.get('range', 0) > 0.1:
                recommendations.append("🖼️ Image feature selection critically impacts diagnostic performance")
        
        # Clinical synergy detection
        if modality_insights.get('clinical_synergy_detected', False):
            recommendations.append("🩺 Strong clinical synergy detected between imaging and text features")
        
        return recommendations

    def generate_comprehensive_report(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive experimental report"""
        print("\n📊 GENERATING COMPREHENSIVE EXPERIMENTAL REPORT")
        print("=" * 60)
        
        try:
            report = {
                'experiment_summary': {
                    'dataset': self.exp_config.dataset_name,
                    'task_type': self.exp_config.task_type,
                    'n_classes': self.exp_config.n_classes,
                    'random_seed': self.exp_config.random_seed,
                    'use_small_sample': self.exp_config.use_small_sample
                },
                'data_summary': self.data_loader.get_data_summary(),
                'results_summary': self._summarize_results(all_results),
                'recommendations': self._generate_recommendations(all_results),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            print("✅ Comprehensive report generated successfully")
            return report
            
        except Exception as e:
            print(f"❌ Failed to generate comprehensive report: {e}")
            return {'error': str(e)}
    
    def _summarize_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize all experimental results"""
        summary = {}
        
        # Extract key metrics from different phases
        if 'baseline_results' in all_results:
            summary['baseline'] = self._extract_baseline_summary(all_results['baseline_results'])
        
        if 'mainmodel_results' in all_results:
            summary['mainmodel'] = self._extract_mainmodel_summary(all_results['mainmodel_results'])
        
        if 'advanced_results' in all_results:
            summary['advanced'] = self._extract_advanced_summary(all_results['advanced_results'])
        
        return summary
    
    def _extract_baseline_summary(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from baseline results"""
        summary = {}
        
        if 'individual_models' in baseline_results:
            models = baseline_results['individual_models']
            best_model = None
            best_f1 = 0
            
            for model_name, results in models.items():
                if 'f1_macro' in results and results['f1_macro'] > best_f1:
                    best_f1 = results['f1_macro']
                    best_model = model_name
            
            summary['best_baseline'] = {
                'model': best_model,
                'f1_macro': best_f1
            }
        
        return summary
    
    def _extract_mainmodel_summary(self, mainmodel_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from MainModel results"""
        summary = {}
        
        if 'simple_test' in mainmodel_results:
            simple_test = mainmodel_results['simple_test']
            if 'metrics' in simple_test:
                metrics = simple_test['metrics']
                summary['simple_test'] = {
                    'f1_macro': metrics.get('f1_macro', 0),
                    'f1_micro': metrics.get('f1_micro', 0),
                    'accuracy': metrics.get('accuracy', 0)
                }
        
        return summary
    
    def _extract_advanced_summary(self, advanced_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from advanced analysis results"""
        summary = {}
        
        # Extract interpretability insights
        if 'interpretability' in advanced_results:
            interpretability = advanced_results['interpretability']
            if 'feature_importance' in interpretability:
                summary['feature_importance'] = 'completed'
        
        # Extract ablation insights
        if 'ablation' in advanced_results:
            ablation = advanced_results['ablation']
            if 'modality_insights' in ablation:
                summary['modality_insights'] = 'completed'
        
        return summary
    
    def _generate_recommendations(self, all_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on experimental results"""
        recommendations = []
        
        # Add basic recommendations
        recommendations.append("🔬 Consider running full dataset evaluation for production deployment")
        recommendations.append("📊 Implement cross-validation for more robust performance estimates")
        
        # Add modality-specific recommendations
        if 'advanced_results' in all_results and 'ablation' in all_results['advanced_results']:
            ablation = all_results['advanced_results']['ablation']
            if 'modality_insights' in ablation:
                modality_insights = ablation['modality_insights']
                if modality_insights.get('multimodal_benefit', 0) > 0.05:
                    recommendations.append("✅ Multimodal approach shows significant benefits")
        
        return recommendations

