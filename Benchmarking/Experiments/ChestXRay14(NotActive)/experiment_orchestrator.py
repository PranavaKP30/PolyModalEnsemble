#!/usr/bin/env python3
"""
Experiment orchestrator for ChestX-ray14 experiments
Coordinates all evaluation components and manages the complete experimental pipeline
"""

import time
import traceback
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np

from config import ExperimentConfig
from data_loader import ChestXRayDataLoader
from baseline_evaluator import BaselineEvaluator
from mainmodel_evaluator import MainModelEvaluator
from advanced_evaluator import AdvancedEvaluator
from results_manager import ResultsManager


class ExperimentOrchestrator:
    """Orchestrates the complete ChestX-ray14 experimental pipeline"""
    
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
        
        print("🏥 ChestX-ray14 Experiment Orchestrator Initialized")
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
        print("\n🚀 STARTING CHESTX-RAY14 FULL EXPERIMENT")
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
            
            return self.all_results
            
        except Exception as e:
            print(f"\n❌ EXPERIMENT FAILED: {str(e)}")
            self._handle_experiment_failure(e)
            raise
    
    def run_quick_experiment(self) -> Dict[str, Any]:
        """
        Run a quick experimental evaluation (subset of models for fast feedback)
        """
        print("\n⚡ STARTING CHESTX-RAY14 QUICK EXPERIMENT")
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
    
    def _handle_experiment_failure(self, error: Exception):
        """Handle experiment failures gracefully"""
        print(f"\n🚨 EXPERIMENT FAILURE HANDLING")
        print("=" * 50)
        print(f"❌ Error: {str(error)}")
        print(f"🔍 Error type: {type(error).__name__}")
        
        # Log the error details
        import traceback
        error_traceback = traceback.format_exc()
        print(f"📋 Full traceback:")
        print(error_traceback)
        
        # Save error information
        error_info = {
            'error_message': str(error),
            'error_type': type(error).__name__,
            'traceback': error_traceback,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'experiment_config': {
                'dataset': self.exp_config.dataset_name,
                'use_small_sample': self.exp_config.use_small_sample,
                'random_seed': self.exp_config.random_seed
            }
        }
        
        # Try to save error info to file
        try:
            error_file = self.path_config.output_path / f"error_log_{int(time.time())}.json"
            import json
            with open(error_file, 'w') as f:
                json.dump(error_info, f, indent=2)
            print(f"📄 Error details saved to: {error_file}")
        except Exception as save_error:
            print(f"⚠️ Could not save error details: {save_error}")
        
        # Provide debugging suggestions
        print(f"\n🔧 DEBUGGING SUGGESTIONS:")
        if "ImportError" in str(type(error)):
            print("   • Check if all required packages are installed")
            print("   • Verify import paths are correct")
            print("   • Ensure MainModel components are available")
        elif "FileNotFoundError" in str(type(error)):
            print("   • Check if data files exist in expected locations")
            print("   • Verify data preprocessing has been completed")
            print("   • Ensure correct data paths in configuration")
        elif "ValueError" in str(type(error)):
            print("   • Check data shapes and dimensions")
            print("   • Verify label format and class counts")
            print("   • Ensure data preprocessing is correct")
        else:
            print("   • Check the full traceback above for specific issues")
            print("   • Verify all dependencies are properly installed")
            print("   • Ensure sufficient system resources")

    # Phase implementations
    def _phase_1_data_preparation(self):
        """Phase 1: Load and prepare data"""
        print("\n📁 PHASE 1: DATA PREPARATION")
        print("-" * 40)
        
        start_time = time.time()
        
        # Initialize data loader
        self.data_loader = ChestXRayDataLoader(self.exp_config, self.path_config)
        
        # Load and prepare data
        self.data_loader.load_raw_data()
        self.data_loader.apply_sampling()
        
        # Initialize evaluators
        self.baseline_evaluator = BaselineEvaluator(self.exp_config, self.data_loader)
        self.mainmodel_evaluator = MainModelEvaluator(self.exp_config, self.data_loader)
        self.advanced_evaluator = AdvancedEvaluator(self.exp_config, self.data_loader)
        self.results_manager = ResultsManager(self.exp_config, self.path_config)
        
        phase_time = time.time() - start_time
        
        self.all_results['data_preparation'] = {
            'phase_time': phase_time,
            'data_info': self.data_loader.get_data_summary(),
            'feature_info': self.data_loader.get_data_summary()
        }
        
        print(f"✅ Data preparation completed in {phase_time:.1f}s")
    
    def _phase_2_baseline_evaluation(self):
        """Phase 2: Comprehensive baseline model evaluation"""
        print("\n📊 COMPREHENSIVE BASELINE EVALUATION")
        print("-" * 40)
        
        # Initialize baseline evaluator
        self.baseline_evaluator = BaselineEvaluator(self.exp_config, self.data_loader)
        
        # Run comprehensive evaluation
        # Run baseline evaluation
        baseline_results = self.baseline_evaluator.run_all_baselines()
        
        self.all_results['baseline_results'] = baseline_results
        
        # Print summary
        summary = baseline_results.get('summary', {})
        total_models = summary.get('total_models', 0)
        successful_models = summary.get('successful_models', 0)
        
        print(f"✅ Comprehensive baseline evaluation completed in {baseline_results.get('total_time', 0):.1f}s")
        print(f"🎯 Evaluated {successful_models}/{total_models} models successfully")
    

    
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
    
    def _quick_mainmodel_with_hyperparameters(self):
        """Quick MainModel test with hyperparameter search"""
        print("\n🧠 QUICK MAINMODEL WITH HYPERPARAMETERS")
        print("-" * 45)
        
        start_time = time.time()
        
        # Initialize MainModel evaluator
        self.mainmodel_evaluator = MainModelEvaluator(self.exp_config, self.data_loader)
        
        try:
            # Run hyperparameter search (uses predefined parameter grid)
            hp_results = self.mainmodel_evaluator.run_hyperparameter_search()
            
            # Test with optimized parameters
            optimized_results = self.mainmodel_evaluator.run_optimized_test(hp_results['best_params'])
            
            mainmodel_results = {
                'hyperparameter_search': hp_results,
                'optimized_test': optimized_results,
                'phase_time': time.time() - start_time
            }
        except Exception as e:
            print(f"⚠️ MainModel hyperparameter search failed: {str(e)}")
            # Try simple test as fallback
            try:
                simple_result = self.mainmodel_evaluator.run_simple_test()
                mainmodel_results = {
                    'simple_test': simple_result,
                    'phase_time': time.time() - start_time,
                    'note': 'Used simple test due to hyperparameter search failure'
                }
            except Exception as e2:
                print(f"⚠️ MainModel simple test also failed: {str(e2)}")
                mainmodel_results = {
                    'error': str(e2),
                    'phase_time': time.time() - start_time
                }
        
        self.all_results['mainmodel_results'] = mainmodel_results
        
        print(f"✅ Quick MainModel with hyperparameters completed in {mainmodel_results['phase_time']:.1f}s")
    
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
        """Phase 4: Advanced analysis including CV, robustness, and ablation"""
        print("\n🔬 PHASE 4: ADVANCED ANALYSIS")
        print("-" * 30)
        
        # Initialize advanced evaluator
        self.advanced_evaluator = AdvancedEvaluator(self.exp_config, self.data_loader)
        
        advanced_results = {}
        
        # Cross-validation analysis
        if self.config.enable_cv:
            print("🔄 Running cross-validation analysis...")
            try:
                cv_results = self.advanced_evaluator.run_cross_validation_analysis()
                advanced_results['cv_results'] = cv_results
                print("   ✅ Cross-validation analysis completed")
            except Exception as e:
                print(f"   ❌ Cross-validation analysis failed: {str(e)}")
                advanced_results['cv_results'] = {'error': str(e)}
        
        # Robustness testing
        print("🛡️ Running robustness testing...")
        try:
            robustness_results = self.advanced_evaluator.run_robustness_testing()
            advanced_results['robustness_results'] = robustness_results
            print("   ✅ Robustness testing completed")
        except Exception as e:
            print(f"   ❌ Robustness testing failed: {str(e)}")
            advanced_results['robustness_results'] = {'error': str(e)}
        
        # Interpretability analysis
        print("🔍 Running interpretability analysis...")
        try:
            interpretability_results = self.advanced_evaluator.run_interpretability_analysis()
            advanced_results['interpretability_results'] = interpretability_results
            print("   ✅ Interpretability analysis completed")
        except Exception as e:
            print(f"   ❌ Interpretability analysis failed: {str(e)}")
            advanced_results['interpretability_results'] = {'error': str(e)}
        
        # Ablation studies
        print("🎭 Running ablation studies...")
        try:
            ablation_results = self.advanced_evaluator.run_ablation_studies()
            advanced_results['ablation_results'] = ablation_results
            print("   ✅ Ablation studies completed")
        except Exception as e:
            print(f"   ❌ Ablation studies failed: {str(e)}")
            advanced_results['ablation_results'] = {'error': str(e)}
        
        self.all_results['advanced_analysis'] = advanced_results
        print("✅ Advanced analysis completed")
    

    
    def _phase_5_comprehensive_reporting(self):
        """Phase 5: Comprehensive reporting and visualization"""
        print("\n📋 PHASE 5: COMPREHENSIVE REPORTING")
        print("-" * 40)
        
        # Initialize results manager
        self.results_manager = ResultsManager(self.exp_config, self.path_config)
        
        # Generate comprehensive report
        try:
            report_results = self.advanced_evaluator.generate_comprehensive_report(self.all_results)
            self.all_results['comprehensive_report'] = report_results
            print("✅ Comprehensive reporting completed")
        except Exception as e:
            print(f"❌ Comprehensive reporting failed: {str(e)}")
            self.all_results['comprehensive_report'] = {'error': str(e)}
    

    
    def _phase_6_results_management(self):
        """Phase 6: Complete results management and export"""
        print("\n💾 PHASE 6: RESULTS MANAGEMENT")
        print("-" * 35)
        
        # Save all results
        try:
            saved_files = self.results_manager.save_all_results(self.all_results)
            self.all_results['saved_files'] = saved_files
            print("✅ All results saved successfully")
        except Exception as e:
            print(f"❌ Results saving failed: {str(e)}")
            self.all_results['saved_files'] = {'error': str(e)}
        
        # Export for publication
        try:
            pub_files = self.results_manager.export_for_publication(self.all_results)
            self.all_results['publication_files'] = pub_files
            print("✅ Publication materials exported")
        except Exception as e:
            print(f"❌ Publication export failed: {str(e)}")
            self.all_results['publication_files'] = {'error': str(e)}
    
    def _finalize_experiment(self):
        """Finalize experiment and print summary"""
        print("\n🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        self._print_experiment_summary()
    
    def _print_experiment_summary(self):
        """Print comprehensive experiment summary"""
        print("\n📊 EXPERIMENT SUMMARY")
        print("=" * 40)
        
        total_time = time.time() - self.experiment_start_time
        
        # Baseline summary
        if 'baseline_results' in self.all_results:
            baseline_summary = self.all_results['baseline_results'].get('summary', {})
            best_model = baseline_summary.get('best_model', 'N/A')
            best_f1 = baseline_summary.get('best_f1', 0)
            print(f"🏆 Best Baseline: {best_model} (F1: {best_f1:.3f})")
        
        # MainModel summary
        if 'mainmodel_results' in self.all_results:
            main_results = self.all_results['mainmodel_results']
            if main_results is not None:
                if 'optimized_test' in main_results and main_results['optimized_test'] is not None:
                    opt_metrics = main_results['optimized_test'].get('metrics', {})
                    print(f"🧠 MainModel Optimized: F1={opt_metrics.get('f1_macro', 0):.3f}")
                if 'simple_test' in main_results and main_results['simple_test'] is not None:
                    simple_metrics = main_results['simple_test'].get('metrics', {})
                    print(f"🧠 MainModel Simple: F1={simple_metrics.get('f1_macro', 0):.3f}")
            else:
                print("🧠 MainModel: Failed to complete")
        else:
            print("🧠 MainModel: Not executed")
        
        # Advanced analysis summary
        if 'advanced_analysis' in self.all_results:
            advanced = self.all_results['advanced_analysis']
            if 'robustness_results' in advanced:
                print(f"🛡️ Robustness tests: {len(advanced['robustness_results'])} scenarios")
            if 'ablation_results' in advanced:
                print(f"🎭 Ablation studies: {len(advanced['ablation_results'].get('ablation_experiments', {}))} experiments")
        
        print(f"⏱️ Total experiment time: {total_time:.1f}s")
        print(f"📁 Results saved to: {self.path_config.output_path}")
    
    def _save_partial_results(self, error_message: str):
        """Save partial results when experiment fails"""
        try:
            if self.results_manager is None:
                self.results_manager = ResultsManager(self.exp_config, self.path_config)
            
            # Add error information
            self.all_results['experiment_error'] = {
                'error_message': error_message,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'partial_results': True
            }
            
            # Save what we have
            saved_files = self.results_manager.save_all_results(self.all_results)
            print(f"💾 Partial results saved: {saved_files}")
            
        except Exception as save_error:
            print(f"⚠️ Could not save partial results: {save_error}")

