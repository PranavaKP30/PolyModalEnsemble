#!/usr/bin/env python3
"""
Experiment orchestrator for Amazon Reviews experiments
Coordinates all evaluation components and manages the complete experimental pipeline
"""

import time
import traceback
from typing import Dict, Any, Optional, List
from pathlib import Path

from config import ExperimentConfig
from data_loader import AmazonReviewsDataLoader
from baseline_evaluator import BaselineEvaluator
from mainmodel_evaluator import MainModelEvaluator
from advanced_evaluator import AdvancedEvaluator
from results_manager import ResultsManager


class ExperimentOrchestrator:
    """Orchestrates the complete Amazon Reviews experimental pipeline"""
    
    def __init__(self, config_overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize the experiment orchestrator
        
        Args:
            config_overrides: Optional configuration overrides
        """
        # Load configuration
        from config import get_default_config
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent.parent.parent
        self.exp_config, self.path_config = get_default_config(project_root, **(config_overrides or {}))
        
        # Backward compatibility - keep self.config for existing code
        self.config = self.exp_config
        
        # Initialize components
        self.data_loader = None
        self.baseline_evaluator = None
        self.mainmodel_evaluator = None
        self.advanced_evaluator = None
        self.results_manager = None
        
        # Experiment state
        self.experiment_start_time = None
        self.all_results = {}
        
        print("🎯 Amazon Reviews Experiment Orchestrator Initialized")
        print(f"📊 Configuration: {self.config.n_classes} classes, seed={self.config.random_seed}")
    
    def run_full_experiment(self, 
                          include_baseline: bool = True,
                          include_mainmodel: bool = True,
                          include_advanced: bool = True,
                          include_hyperparameter_search: bool = True) -> Dict[str, Any]:
        """
        Run the complete experimental pipeline
        
        Args:
            include_baseline: Whether to run baseline model evaluation
            include_mainmodel: Whether to run MainModel evaluation
            include_advanced: Whether to run advanced analyses
            include_hyperparameter_search: Whether to run hyperparameter optimization
        
        Returns:
            Complete experimental results
        """
        print("\n🚀 STARTING AMAZON REVIEWS FULL EXPERIMENT")
        print("=" * 70)
        
        self.experiment_start_time = time.time()
        
        try:
            # Phase 1: Data Loading and Preparation
            self._phase_1_data_preparation()
            
            # Phase 2: Baseline Model Evaluation
            if include_baseline:
                self._phase_2_baseline_evaluation()
            
            # Phase 3: MainModel Evaluation
            if include_mainmodel:
                self._phase_3_mainmodel_evaluation(include_hyperparameter_search)
            
            # Phase 4: Advanced Analysis
            if include_advanced:
                self._phase_4_advanced_analysis()
            
            # Phase 5: Comprehensive Reporting
            self._phase_5_comprehensive_reporting()
            
            # Phase 6: Results Management
            self._phase_6_results_management()
            
            # Finalize experiment
            self._finalize_experiment()
            
            print("\n🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
            self._print_experiment_summary()
            
            return self.all_results
            
        except Exception as e:
            print(f"\n❌ EXPERIMENT FAILED: {str(e)}")
            traceback.print_exc()
            
            # Save partial results if possible
            try:
                self._save_partial_results(str(e))
            except:
                print("⚠️ Could not save partial results")
            
            return self.all_results
    
    def run_quick_experiment(self) -> Dict[str, Any]:
        """
        Run a quick experimental evaluation (subset of models for fast feedback)
        """
        print("\n⚡ STARTING AMAZON REVIEWS QUICK EXPERIMENT")
        print("=" * 70)
        
        self.experiment_start_time = time.time()
        
        try:
            # Phase 1: Data preparation
            self._phase_1_data_preparation()
            
            # Phase 2: Quick baseline evaluation (selected models only)
            self._quick_baseline_evaluation()
            
            # Phase 3: Quick MainModel test with hyperparameter search
            self._quick_mainmodel_with_hyperparameters()
            
            # Phase 4: Advanced Analysis (ablation studies, robustness testing)
            self._phase_4_advanced_analysis()
            
            # Phase 5: Comprehensive Reporting
            self._phase_5_comprehensive_reporting()
            
            # Phase 6: Results Management
            self._phase_6_results_management()
            
            print("\n⚡ QUICK EXPERIMENT COMPLETED!")
            self._print_quick_summary()
            
            return self.all_results
            
        except Exception as e:
            print(f"\n❌ QUICK EXPERIMENT FAILED: {str(e)}")
            traceback.print_exc()
            return self.all_results
    
    def run_baseline_only(self) -> Dict[str, Any]:
        """Run only baseline model evaluation"""
        print("\n📊 RUNNING BASELINE MODELS ONLY")
        print("=" * 50)
        
        self.experiment_start_time = time.time()
        
        try:
            self._phase_1_data_preparation()
            self._phase_2_baseline_evaluation()
            self._basic_reporting()
            self._quick_results_management()
            
            return self.all_results
            
        except Exception as e:
            print(f"\n❌ BASELINE EXPERIMENT FAILED: {str(e)}")
            return self.all_results
    
    def run_mainmodel_only(self, include_hyperparameter_search: bool = True) -> Dict[str, Any]:
        """Run only MainModel evaluation"""
        print("\n🧠 RUNNING MAINMODEL ONLY")
        print("=" * 30)
        
        self.experiment_start_time = time.time()
        
        try:
            self._phase_1_data_preparation()
            self._phase_3_mainmodel_evaluation(include_hyperparameter_search)
            self._basic_reporting()
            self._quick_results_management()
            
            return self.all_results
            
        except Exception as e:
            print(f"\n❌ MAINMODEL EXPERIMENT FAILED: {str(e)}")
            return self.all_results
    
    def _phase_1_data_preparation(self):
        """Phase 1: Load and prepare data"""
        print("\n📁 PHASE 1: DATA PREPARATION")
        print("-" * 40)
        
        start_time = time.time()
        
        # Initialize data loader
        self.data_loader = AmazonReviewsDataLoader(self.exp_config, self.path_config)
        
        # Load and prepare data
        self.data_loader.load_raw_data()
        self.data_loader.apply_sampling()
        
        # Initialize evaluators
        self.baseline_evaluator = BaselineEvaluator(self.exp_config, self.data_loader)
        self.mainmodel_evaluator = MainModelEvaluator(self.exp_config, self.data_loader)
        self.advanced_evaluator = AdvancedEvaluator(self.exp_config, self.data_loader)
        self.results_manager = ResultsManager(self.path_config, self.exp_config)
        
        phase_time = time.time() - start_time
        
        self.all_results['data_preparation'] = {
            'phase_time': phase_time,
            'data_info': self.data_loader.get_data_summary(),
            'feature_info': self.data_loader.get_data_summary()
        }
        
        print(f"✅ Data preparation completed in {phase_time:.1f}s")
    
    def _phase_2_baseline_evaluation(self):
        """Phase 2: Evaluate baseline models"""
        print("\n📊 PHASE 2: BASELINE MODEL EVALUATION")
        print("-" * 45)
        
        start_time = time.time()
        
        # Run comprehensive baseline evaluation
        baseline_results = self.baseline_evaluator.run_all_baselines()
        
        phase_time = time.time() - start_time
        baseline_results['phase_time'] = phase_time
        
        self.all_results['baseline_results'] = baseline_results
        
        print(f"✅ Baseline evaluation completed in {phase_time:.1f}s")
    
    def _phase_3_mainmodel_evaluation(self, include_hyperparameter_search: bool = True):
        """Phase 3: Evaluate MainModel"""
        print("\n🧠 PHASE 3: MAINMODEL EVALUATION")
        print("-" * 35)
        
        start_time = time.time()
        
        mainmodel_results = {}
        
        # Hyperparameter optimization (if requested)
        if include_hyperparameter_search:
            try:
                hp_results = self.mainmodel_evaluator.run_hyperparameter_search()
                mainmodel_results['hyperparameter_search'] = hp_results
                best_params = hp_results.get('best_params')
            except Exception as e:
                print(f"⚠️ Hyperparameter search failed: {str(e)}")
                best_params = None
        else:
            best_params = None
        
        # Simple test with default parameters
        try:
            simple_result = self.mainmodel_evaluator.run_simple_test()
            mainmodel_results['simple_test'] = simple_result
        except Exception as e:
            print(f"⚠️ Simple test failed: {str(e)}")
            mainmodel_results['simple_test'] = None
        
        # Optimized test (if we have optimal parameters)
        if best_params:
            try:
                optimized_result = self.mainmodel_evaluator.run_optimized_test(best_params)
                mainmodel_results['optimized_test'] = optimized_result
            except Exception as e:
                print(f"⚠️ Optimized test failed: {str(e)}")
                mainmodel_results['optimized_test'] = None
        else:
            mainmodel_results['optimized_test'] = None
        
        phase_time = time.time() - start_time
        mainmodel_results['phase_time'] = phase_time
        
        self.all_results['mainmodel_results'] = mainmodel_results
        
        print(f"✅ MainModel evaluation completed in {phase_time:.1f}s")
    
    def _phase_4_advanced_analysis(self):
        """Phase 4: Advanced analysis and robustness testing"""
        print("\n🔬 PHASE 4: ADVANCED ANALYSIS")
        print("-" * 30)
        
        start_time = time.time()
        
        advanced_results = {}
        
        # Cross-validation analysis
        try:
            cv_results = self.advanced_evaluator.run_cross_validation_analysis()
            advanced_results['cv_results'] = cv_results
            self.all_results['cv_results'] = cv_results
        except Exception as e:
            print(f"⚠️ Cross-validation analysis failed: {str(e)}")
        
        # Robustness testing
        try:
            robustness_results = self.advanced_evaluator.run_robustness_testing()
            advanced_results['robustness_results'] = robustness_results
            self.all_results['robustness_results'] = robustness_results
        except Exception as e:
            print(f"⚠️ Robustness testing failed: {str(e)}")
        
        # Interpretability analysis
        try:
            interpretability_results = self.advanced_evaluator.run_interpretability_analysis()
            advanced_results['interpretability_results'] = interpretability_results
            self.all_results['interpretability_results'] = interpretability_results
        except Exception as e:
            print(f"⚠️ Interpretability analysis failed: {str(e)}")
        
        # Modality ablation study
        try:
            ablation_results = self.advanced_evaluator.run_modality_ablation_study()
            advanced_results['ablation_results'] = ablation_results
            self.all_results['ablation_results'] = ablation_results
        except Exception as e:
            print(f"⚠️ Ablation study failed: {str(e)}")
        
        phase_time = time.time() - start_time
        advanced_results['phase_time'] = phase_time
        
        self.all_results['advanced_analysis'] = advanced_results
        
        print(f"✅ Advanced analysis completed in {phase_time:.1f}s")
    
    def _phase_5_comprehensive_reporting(self):
        """Phase 5: Generate comprehensive analysis report"""
        print("\n📋 PHASE 5: COMPREHENSIVE REPORTING")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            comprehensive_report = self.advanced_evaluator.generate_comprehensive_report(self.all_results)
            self.all_results['comprehensive_report'] = comprehensive_report
        except Exception as e:
            print(f"⚠️ Comprehensive reporting failed: {str(e)}")
            self.all_results['comprehensive_report'] = {'error': str(e)}
        
        phase_time = time.time() - start_time
        
        print(f"✅ Comprehensive reporting completed in {phase_time:.1f}s")
    
    def _phase_6_results_management(self):
        """Phase 6: Save and manage results"""
        print("\n💾 PHASE 6: RESULTS MANAGEMENT")
        print("-" * 35)
        
        start_time = time.time()
        
        try:
            # Ensure results manager is initialized
            if self.results_manager is None:
                self.results_manager = ResultsManager(self.path_config, self.exp_config)
            
            # Save all results
            saved_files = self.results_manager.save_all_results(self.all_results)
            self.all_results['saved_files'] = saved_files
            
            # Export for publication
            publication_files = self.results_manager.export_for_publication(self.all_results)
            self.all_results['publication_files'] = publication_files
            
        except Exception as e:
            print(f"⚠️ Results management failed: {str(e)}")
            self.all_results['results_management_error'] = str(e)
        
        phase_time = time.time() - start_time
        
        print(f"✅ Results management completed in {phase_time:.1f}s")
    
    def _quick_baseline_evaluation(self):
        """Quick baseline evaluation with selected models"""
        print("\n📊 QUICK BASELINE EVALUATION")
        print("-" * 35)
        
        start_time = time.time()
        
        # Run evaluation with selected models only
        quick_models = [
            'RandomForestClassifier', 'XGBoostClassifier', 'LogisticRegression',
            'VotingClassifier', 'StackingClassifier'
        ]
        
        baseline_results = self.baseline_evaluator.run_all_baselines()
        
        phase_time = time.time() - start_time
        baseline_results['phase_time'] = phase_time
        
        self.all_results['baseline_results'] = baseline_results
        
        print(f"✅ Quick baseline evaluation completed in {phase_time:.1f}s")
    
    def _quick_mainmodel_test(self):
        """Quick MainModel test without hyperparameter search"""
        print("\n🧠 QUICK MAINMODEL TEST")
        print("-" * 25)
        
        start_time = time.time()
        
        try:
            simple_result = self.mainmodel_evaluator.run_simple_test()
            mainmodel_results = {
                'simple_test': simple_result,
                'phase_time': time.time() - start_time
            }
        except Exception as e:
            print(f"⚠️ Quick MainModel test failed: {str(e)}")
            mainmodel_results = {
                'simple_test': None,
                'error': str(e),
                'phase_time': time.time() - start_time
            }
        
        self.all_results['mainmodel_results'] = mainmodel_results
        
        print(f"✅ Quick MainModel test completed in {mainmodel_results['phase_time']:.1f}s")
    
    def _quick_mainmodel_with_hyperparameters(self):
        """Quick MainModel test with hyperparameter search"""
        print("\n🧠 QUICK MAINMODEL WITH HYPERPARAMETERS")
        print("-" * 45)
        
        start_time = time.time()
        
        # Run hyperparameter search (uses predefined parameter grid)
        hp_results = self.mainmodel_evaluator.run_hyperparameter_search()
        
        # Test with optimized parameters
        optimized_results = self.mainmodel_evaluator.run_optimized_test(hp_results['best_params'])
        
        mainmodel_results = {
            'hyperparameter_search': hp_results,
            'optimized_test': optimized_results,
            'phase_time': time.time() - start_time
        }
        
        self.all_results['mainmodel_results'] = mainmodel_results
        
        print(f"✅ Quick MainModel with hyperparameters completed in {mainmodel_results['phase_time']:.1f}s")
    
    def _basic_reporting(self):
        """Basic reporting for quick experiments"""
        print("\n📋 BASIC REPORTING")
        print("-" * 20)
        
        start_time = time.time()
        
        # Generate basic performance summary
        basic_report = self._create_basic_report()
        self.all_results['basic_report'] = basic_report
        
        phase_time = time.time() - start_time
        print(f"✅ Basic reporting completed in {phase_time:.1f}s")
    
    def _quick_results_management(self):
        """Quick results management"""
        print("\n💾 QUICK RESULTS SAVE")
        print("-" * 25)
        
        start_time = time.time()
        
        try:
            # Save essential results only
            saved_files = self.results_manager.save_all_results(self.all_results)
            self.all_results['saved_files'] = saved_files
        except Exception as e:
            print(f"⚠️ Quick save failed: {str(e)}")
            self.all_results['save_error'] = str(e)
        
        phase_time = time.time() - start_time
        print(f"✅ Quick save completed in {phase_time:.1f}s")
    
    def _finalize_experiment(self):
        """Finalize the experiment"""
        total_time = time.time() - self.experiment_start_time
        
        self.all_results['experiment_metadata'] = {
            'total_time': total_time,
            'start_time': self.experiment_start_time,
            'config': {
                'n_classes': self.config.n_classes,
                'random_seed': self.config.random_seed,
                'cv_folds': self.config.cv_folds,
                'train_sample_size': self.config.train_sample_size,
                'test_sample_size': self.config.test_sample_size
            }
        }
    
    def _print_experiment_summary(self):
        """Print experiment summary"""
        total_time = self.all_results.get('experiment_metadata', {}).get('total_time', 0)
        
        print(f"\n📊 EXPERIMENT SUMMARY")
        print("=" * 30)
        print(f"⏱️ Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        
        # Best baseline model
        baseline_summary = self.all_results.get('baseline_results', {}).get('summary', {})
        if baseline_summary:
            print(f"🥇 Best baseline: {baseline_summary.get('best_model', 'N/A')} "
                  f"(Accuracy: {baseline_summary.get('best_accuracy', 0):.4f})")
        
        # MainModel performance
        mainmodel_results = self.all_results.get('mainmodel_results', {})
        optimized_test = mainmodel_results.get('optimized_test')
        if optimized_test:
            print(f"🧠 MainModel (Optimized): Accuracy {optimized_test.get('accuracy', 0):.4f}")
        else:
            simple_test = mainmodel_results.get('simple_test')
            if simple_test:
                print(f"🧠 MainModel (Default): Accuracy {simple_test.get('accuracy', 0):.4f}")
        
        # Results location
        saved_files = self.all_results.get('saved_files', {})
        if saved_files:
            print(f"📁 Results saved to: {self.results_manager.run_dir}")
        
        print("\n🎯 Experiment completed successfully!")
    
    def _print_quick_summary(self):
        """Print quick experiment summary"""
        total_time = time.time() - self.experiment_start_time
        
        print(f"\n⚡ QUICK EXPERIMENT SUMMARY")
        print("=" * 35)
        print(f"⏱️ Total runtime: {total_time:.1f} seconds")
        
        # Best baseline from quick evaluation
        baseline_summary = self.all_results.get('baseline_results', {}).get('summary', {})
        if baseline_summary:
            print(f"🥇 Best model: {baseline_summary.get('best_model', 'N/A')} "
                  f"(Accuracy: {baseline_summary.get('best_accuracy', 0):.4f})")
        
        # MainModel quick test
        mainmodel_simple = self.all_results.get('mainmodel_results', {}).get('simple_test')
        if mainmodel_simple:
            print(f"🧠 MainModel: Accuracy {mainmodel_simple.get('accuracy', 0):.4f}")
        
        print("\n⚡ Quick experiment completed!")
    
    def _create_basic_report(self) -> Dict[str, Any]:
        """Create basic performance report"""
        
        basic_report = {
            'dataset_info': {
                'name': 'Amazon Reviews 5-Class Rating Prediction',
                'n_classes': self.config.n_classes,
                'train_size': len(self.data_loader.train_labels),
                'test_size': len(self.data_loader.test_labels)
            },
            'best_models': {}
        }
        
        # Best baseline
        baseline_summary = self.all_results.get('baseline_results', {}).get('summary', {})
        if baseline_summary:
            basic_report['best_models']['baseline'] = {
                'name': baseline_summary.get('best_model'),
                'accuracy': baseline_summary.get('best_accuracy', 0)
            }
        
        # MainModel performance
        mainmodel_results = self.all_results.get('mainmodel_results', {})
        if 'optimized_test' in mainmodel_results and mainmodel_results['optimized_test']:
            result = mainmodel_results['optimized_test']
            basic_report['best_models']['mainmodel'] = {
                'name': 'MainModel (Optimized)',
                'accuracy': result.get('accuracy', 0),
                'star_mae': result.get('star_mae', 0)
            }
        elif 'simple_test' in mainmodel_results and mainmodel_results['simple_test']:
            result = mainmodel_results['simple_test']
            basic_report['best_models']['mainmodel'] = {
                'name': 'MainModel (Default)',
                'accuracy': result.get('accuracy', 0),
                'star_mae': result.get('star_mae', 0)
            }
        
        return basic_report
    
    def _save_partial_results(self, error_message: str):
        """Save partial results in case of experiment failure"""
        print(f"\n💾 Saving partial results due to error: {error_message}")
        
        try:
            if self.results_manager:
                # Add error information
                self.all_results['experiment_error'] = {
                    'error_message': error_message,
                    'partial_completion': True,
                    'completed_phases': list(self.all_results.keys())
                }
                
                # Save what we have
                self.results_manager.save_all_results(self.all_results)
                print(f"✅ Partial results saved to: {self.results_manager.run_dir}")
            else:
                print("⚠️ Results manager not initialized, cannot save partial results")
                
        except Exception as save_error:
            print(f"❌ Failed to save partial results: {str(save_error)}")
    
    def _apply_config_overrides(self, overrides: Dict[str, Any]):
        """Apply configuration overrides"""
        
        for key, value in overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                print(f"📝 Config override: {key} = {value}")
            else:
                print(f"⚠️ Unknown config parameter: {key}")
    
    def get_experiment_status(self) -> Dict[str, Any]:
        """Get current experiment status"""
        
        if not self.experiment_start_time:
            return {'status': 'not_started'}
        
        current_time = time.time()
        elapsed_time = current_time - self.experiment_start_time
        
        completed_phases = list(self.all_results.keys())
        
        return {
            'status': 'running',
            'elapsed_time': elapsed_time,
            'completed_phases': completed_phases,
            'current_results': len(self.all_results)
        }
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of current results"""
        
        summary = {
            'experiment_started': self.experiment_start_time is not None,
            'phases_completed': list(self.all_results.keys()),
            'data_loaded': 'data_preparation' in self.all_results,
            'baseline_completed': 'baseline_results' in self.all_results,
            'mainmodel_completed': 'mainmodel_results' in self.all_results,
            'advanced_completed': 'advanced_analysis' in self.all_results
        }
        
        # Add performance highlights if available
        if 'baseline_results' in self.all_results:
            baseline_summary = self.all_results['baseline_results'].get('summary', {})
            summary['best_baseline_accuracy'] = baseline_summary.get('best_accuracy', 0)
            summary['best_baseline_model'] = baseline_summary.get('best_model')
        
        if 'mainmodel_results' in self.all_results:
            mainmodel_optimized = self.all_results['mainmodel_results'].get('optimized_test')
            if mainmodel_optimized:
                summary['mainmodel_accuracy'] = mainmodel_optimized.get('accuracy', 0)
            else:
                mainmodel_simple = self.all_results['mainmodel_results'].get('simple_test')
                if mainmodel_simple:
                    summary['mainmodel_accuracy'] = mainmodel_simple.get('accuracy', 0)
        
        return summary
