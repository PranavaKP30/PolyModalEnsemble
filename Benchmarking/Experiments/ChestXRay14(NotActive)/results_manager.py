#!/usr/bin/env python3
"""
Enhanced Results management module for ChestX-ray14 experiments
Saves comprehensive predictions, metrics, and analysis for all test components
"""

import json
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import pickle

from config import ExperimentConfig, PathConfig


class ResultsManager:
    """Enhanced manager for experimental results with comprehensive saving"""
    
    def __init__(self, exp_config: ExperimentConfig, path_config: PathConfig):
        self.exp_config = exp_config
        self.path_config = path_config
        self.results = {}
        
        # Create results directory structure like AmazonReviews
        self.results_dir = Path(path_config.output_path)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this experiment run
        from datetime import datetime
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        print(f"📁 Results will be saved to: {self.run_dir}")
        
        # Create structured directories for comprehensive saving
        self._create_output_directories()
    
    def _create_output_directories(self) -> None:
        """Create structured directories for different result types"""
        self.predictions_path = self.run_dir / "predictions"
        self.metrics_path = self.run_dir / "metrics"
        self.analysis_path = self.run_dir / "analysis"
        self.models_path = self.run_dir / "models"
        
        # Create all directories
        for path in [self.predictions_path, self.metrics_path, self.analysis_path, self.models_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def add_results(self, experiment_name: str, results: Dict[str, Any]) -> None:
        """Add results for an experiment with comprehensive saving"""
        self.results[experiment_name] = results
        
        # Save component-specific results immediately
        self._save_component_results(experiment_name, results)
    
    def save_all_results(self, all_results: Dict[str, Any]) -> Dict[str, str]:
        """Save all experimental results in organized structure (AmazonReviews style)"""
        print("\n💾 SAVING ALL EXPERIMENTAL RESULTS")
        print("=" * 60)
        
        saved_files = {}
        
        try:
            # Create organized structure like AmazonReviews
            self._create_amazonreviews_structure()
            
            # Check if this is a multi-seed experiment
            if 'seed_results' in all_results:
                print(f"🌱 Multi-seed experiment detected with {len(all_results['seed_results'])} seeds")
                
                # Save results by seed and analysis type
                for seed_idx, seed_results in enumerate(all_results['seed_results']):
                    seed_key = f"seed_{seed_idx + 1}"
                    try:
                        saved_files[seed_key] = self._save_seed_results(seed_idx, seed_results)
                    except Exception as e:
                        print(f"⚠️ Failed to save {seed_key} results: {e}")
                        saved_files[seed_key] = {"error": str(e)}
                
                # Save aggregate results
                try:
                    aggregate_files = self._save_aggregate_results(all_results)
                    saved_files['aggregate'] = aggregate_files
                except Exception as e:
                    print(f"⚠️ Failed to save aggregate results: {e}")
                    saved_files['aggregate'] = {"error": str(e)}
                
                # Save experiment summary
                try:
                    summary_files = self._save_experiment_summary(all_results)
                    saved_files['summary'] = summary_files
                except Exception as e:
                    print(f"⚠️ Failed to save summary: {e}")
                    saved_files['summary'] = {"error": str(e)}
                
            else:
                # Single experiment (quick mode or single seed)
                print("🌱 Single experiment detected")
                
                # Save baseline results
                if 'baseline_results' in all_results:
                    baseline_files = self._save_baseline_results_organized(all_results['baseline_results'])
                    saved_files['baseline'] = baseline_files
                
                # Save hyperparameter search results
                if 'mainmodel_results' in all_results and 'hyperparameter_search' in all_results['mainmodel_results']:
                    hp_files = self._save_hyperparameter_results_organized(all_results['mainmodel_results']['hyperparameter_search'])
                    saved_files['hyperparameter'] = hp_files
                
                # Save main model results
                if 'mainmodel_results' in all_results:
                    main_files = self._save_mainmodel_results_organized(all_results['mainmodel_results'])
                    saved_files['mainmodel'] = main_files
                
                # Save cross-validation results
                if 'cv_results' in all_results:
                    cv_files = self._save_cv_results_organized(all_results['cv_results'])
                    saved_files['cross_validation'] = cv_files
                
                # Save interpretability results
                if 'interpretability_results' in all_results:
                    interp_files = self._save_interpretability_results_organized(all_results['interpretability_results'])
                    saved_files['interpretability'] = interp_files
                
                # Save ablation results
                if 'ablation_results' in all_results:
                    ablation_files = self._save_ablation_results_organized(all_results['ablation_results'])
                    saved_files['ablation'] = ablation_files
                
                # Save robustness results
                if 'robustness_results' in all_results:
                    robust_files = self._save_robustness_results_organized(all_results['robustness_results'])
                    saved_files['robustness'] = robust_files
            
            # Create experiment manifest
            manifest_path = self._create_experiment_manifest(saved_files, all_results)
            saved_files['manifest'] = str(manifest_path)
            
            print(f"\n📊 All results saved to: {self.run_dir}")
            print(f"🏷️ Experiment ID: {self.timestamp}")
            
        except Exception as e:
            print(f"❌ Error saving all results: {e}")
            import traceback
            traceback.print_exc()
            saved_files = {"error": str(e)}
        
        return saved_files
    
    def export_for_publication(self, all_results: Dict[str, Any]) -> Dict[str, str]:
        """Export results for publication (AmazonReviews style)"""
        print("\n📄 EXPORTING FOR PUBLICATION")
        print("=" * 50)
        
        export_files = {}
        
        try:
            # Create publication directory
            pub_dir = self.run_dir / "publication"
            pub_dir.mkdir(exist_ok=True)
            
            # Export summary statistics
            summary_file = pub_dir / "experiment_summary.txt"
            self._export_summary_statistics(all_results, summary_file)
            export_files['summary'] = str(summary_file)
            
            # Export performance comparison
            perf_file = pub_dir / "performance_comparison.csv"
            self._export_performance_comparison(all_results, perf_file)
            export_files['performance'] = str(perf_file)
            
            # Export key metrics
            metrics_file = pub_dir / "key_metrics.json"
            self._export_key_metrics(all_results, metrics_file)
            export_files['metrics'] = str(metrics_file)
            
            print(f"✅ Publication exports saved to: {pub_dir}")
            
        except Exception as e:
            print(f"❌ Error exporting for publication: {e}")
            export_files = {"error": str(e)}
        
        return export_files
    
    def _create_amazonreviews_structure(self):
        """Create organized folder structure like AmazonReviews"""
        # Create seed-based structure (for single seed experiments)
        seed_dir = self.run_dir / "seed_1"
        seed_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each analysis component
        self.analysis_dirs = {
            'baseline_models': seed_dir / "01_baseline_models",
            'hyperparameter_tuning': seed_dir / "02_hyperparameter_tuning", 
            'main_model': seed_dir / "03_main_model",
            'cross_validation': seed_dir / "04_cross_validation",
            'interpretability': seed_dir / "05_interpretability",
            'ablation_studies': seed_dir / "06_ablation_studies",
            'robustness_analysis': seed_dir / "07_robustness_analysis",
            'visualizations': seed_dir / "08_visualizations",
            'raw_data': seed_dir / "09_raw_data"
        }
        
        for analysis_type, analysis_dir in self.analysis_dirs.items():
            analysis_dir.mkdir(exist_ok=True)
            # Create individual_predictions subdirectory for baseline models
            if analysis_type == 'baseline_models':
                (analysis_dir / "individual_predictions").mkdir(exist_ok=True)
            if analysis_type == 'main_model':
                (analysis_dir / "test_predictions").mkdir(exist_ok=True)
        
        # Create aggregate results directory
        self.aggregate_dir = self.run_dir / "aggregate_results"
        self.aggregate_dir.mkdir(exist_ok=True)
        
        # Create summary directory
        self.summary_dir = self.run_dir / "experiment_summary"
        self.summary_dir.mkdir(exist_ok=True)
        
        print(f"📂 Organized structure created like AmazonReviews")
    
    def _save_seed_results(self, seed_idx: int, seed_results: Dict[str, Any]) -> Dict[str, str]:
        """Save results for a specific seed in organized folders"""
        seed_files = {}
        
        # Create seed-specific directory structure
        seed_dir = self.run_dir / f"seed_{seed_idx + 1}"
        seed_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each analysis component
        analysis_dirs = {
            'baseline_models': seed_dir / "01_baseline_models",
            'hyperparameter_tuning': seed_dir / "02_hyperparameter_tuning", 
            'main_model': seed_dir / "03_main_model",
            'cross_validation': seed_dir / "04_cross_validation",
            'interpretability': seed_dir / "05_interpretability",
            'ablation_studies': seed_dir / "06_ablation_studies",
            'robustness_analysis': seed_dir / "07_robustness_analysis",
            'visualizations': seed_dir / "08_visualizations",
            'raw_data': seed_dir / "09_raw_data"
        }
        
        for analysis_type, analysis_dir in analysis_dirs.items():
            analysis_dir.mkdir(exist_ok=True)
            if analysis_type == 'baseline_models':
                (analysis_dir / "individual_predictions").mkdir(exist_ok=True)
            if analysis_type == 'main_model':
                (analysis_dir / "test_predictions").mkdir(exist_ok=True)
        
        # 1. Baseline Models
        if 'baseline_results' in seed_results:
            baseline_files = self._save_baseline_results_organized(seed_results['baseline_results'])
            seed_files['baseline'] = baseline_files
        
        # 2. Hyperparameter Tuning  
        if 'mainmodel_results' in seed_results and 'hyperparameter_search' in seed_results['mainmodel_results']:
            hp_files = self._save_hyperparameter_results_organized(seed_results['mainmodel_results']['hyperparameter_search'])
            seed_files['hyperparameter'] = hp_files
        
        # 3. Main Model
        if 'mainmodel_results' in seed_results:
            main_files = self._save_mainmodel_results_organized(seed_results['mainmodel_results'])
            seed_files['main_model'] = main_files
        
        # 4. Cross Validation
        if 'cv_results' in seed_results:
            cv_files = self._save_cv_results_organized(seed_results['cv_results'])
            seed_files['cross_validation'] = cv_files
        
        # 5. Interpretability
        if 'interpretability_results' in seed_results:
            interp_files = self._save_interpretability_results_organized(seed_results['interpretability_results'])
            seed_files['interpretability'] = interp_files
        
        # 6. Ablation Studies
        if 'ablation_results' in seed_results:
            ablation_files = self._save_ablation_results_organized(seed_results['ablation_results'])
            seed_files['ablation'] = ablation_files
        
        # 7. Robustness Analysis
        if 'robustness_results' in seed_results:
            robust_files = self._save_robustness_results_organized(seed_results['robustness_results'])
            seed_files['robustness'] = robust_files
        
        print(f"✅ Saved seed {seed_idx + 1} results in organized folders")
        return seed_files
    
    def _save_aggregate_results(self, all_results: Dict[str, Any]) -> Dict[str, str]:
        """Save aggregate results across all seeds"""
        aggregate_files = {}
        
        # Create aggregate results directory
        aggregate_dir = self.run_dir / "aggregate_results"
        aggregate_dir.mkdir(exist_ok=True)
        
        # Aggregate baseline performance across seeds
        baseline_performances = []
        mainmodel_performances = []
        
        for seed_idx, seed_results in enumerate(all_results['seed_results']):
            # Baseline performance
            if 'baseline_results' in seed_results:
                baseline_summary = seed_results['baseline_results'].get('summary', {})
                baseline_performances.append({
                    'seed': seed_idx + 1,
                    'best_model': baseline_summary.get('best_model', 'N/A'),
                    'best_f1': baseline_summary.get('best_f1', 0),
                    'best_accuracy': baseline_summary.get('best_accuracy', 0)
                })
            
            # MainModel performance
            if 'mainmodel_results' in seed_results:
                main_results = seed_results['mainmodel_results']
                if 'optimized_test' in main_results:
                    opt_metrics = main_results['optimized_test'].get('metrics', {})
                    mainmodel_performances.append({
                        'seed': seed_idx + 1,
                        'f1_macro': opt_metrics.get('f1_macro', 0),
                        'f1_micro': opt_metrics.get('f1_micro', 0),
                        'accuracy': opt_metrics.get('accuracy', 0)
                    })
        
        # Save aggregate baseline performance
        if baseline_performances:
            baseline_file = aggregate_dir / "baseline_performance_across_seeds.json"
            with open(baseline_file, 'w') as f:
                json.dump(baseline_performances, f, indent=2)
            aggregate_files['baseline_performance'] = str(baseline_file)
        
        # Save aggregate MainModel performance
        if mainmodel_performances:
            mainmodel_file = aggregate_dir / "mainmodel_performance_across_seeds.json"
            with open(mainmodel_file, 'w') as f:
                json.dump(mainmodel_performances, f, indent=2)
            aggregate_files['mainmodel_performance'] = str(mainmodel_file)
        
        # Calculate and save statistical summary
        if mainmodel_performances:
            f1_scores = [p['f1_macro'] for p in mainmodel_performances]
            statistical_summary = {
                'mainmodel_f1_macro': {
                    'mean': float(np.mean(f1_scores)),
                    'std': float(np.std(f1_scores)),
                    'min': float(np.min(f1_scores)),
                    'max': float(np.max(f1_scores)),
                    'n_seeds': len(f1_scores)
                }
            }
            
            stats_file = aggregate_dir / "statistical_summary.json"
            with open(stats_file, 'w') as f:
                json.dump(statistical_summary, f, indent=2)
            aggregate_files['statistical_summary'] = str(stats_file)
        
        print(f"✅ Aggregate results saved to {aggregate_dir}")
        return aggregate_files
    
    def _save_experiment_summary(self, all_results: Dict[str, Any]) -> Dict[str, str]:
        """Save experiment summary across all seeds"""
        summary_files = {}
        
        # Create summary directory
        summary_dir = self.run_dir / "experiment_summary"
        summary_dir.mkdir(exist_ok=True)
        
        # Generate comprehensive summary
        summary = {
            'experiment_id': self.timestamp,
            'dataset': 'ChestX-ray14',
            'task': '14-class multi-label pathology classification',
            'n_seeds': len(all_results['seed_results']) if 'seed_results' in all_results else 1,
            'seeds_used': [42, 123, 456, 789, 999][:len(all_results['seed_results'])] if 'seed_results' in all_results else [42],
            'timestamp': self.timestamp
        }
        
        # Add performance summary
        if 'seed_results' in all_results:
            baseline_f1s = []
            mainmodel_f1s = []
            
            for seed_results in all_results['seed_results']:
                # Baseline performance
                if 'baseline_results' in seed_results:
                    baseline_summary = seed_results['baseline_results'].get('summary', {})
                    if 'best_f1' in baseline_summary:
                        baseline_f1s.append(baseline_summary['best_f1'])
                
                # MainModel performance
                if 'mainmodel_results' in seed_results:
                    main_results = seed_results['mainmodel_results']
                    if 'optimized_test' in main_results:
                        opt_metrics = main_results['optimized_test'].get('metrics', {})
                        if 'f1_macro' in opt_metrics:
                            mainmodel_f1s.append(opt_metrics['f1_macro'])
            
            if baseline_f1s:
                summary['baseline_performance'] = {
                    'mean_f1': float(np.mean(baseline_f1s)),
                    'std_f1': float(np.std(baseline_f1s)),
                    'best_f1': float(np.max(baseline_f1s))
                }
            
            if mainmodel_f1s:
                summary['mainmodel_performance'] = {
                    'mean_f1_macro': float(np.mean(mainmodel_f1s)),
                    'std_f1_macro': float(np.std(mainmodel_f1s)),
                    'best_f1_macro': float(np.max(mainmodel_f1s))
                }
        
        # Save summary
        summary_file = summary_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        summary_files['experiment_summary'] = str(summary_file)
        
        # Save human-readable summary
        readable_file = summary_dir / "experiment_summary.txt"
        with open(readable_file, 'w') as f:
            f.write("ChestX-ray14 Experiment Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Experiment ID: {summary['experiment_id']}\n")
            f.write(f"Dataset: {summary['dataset']}\n")
            f.write(f"Task: {summary['task']}\n")
            f.write(f"Number of Seeds: {summary['n_seeds']}\n")
            f.write(f"Seeds Used: {summary['seeds_used']}\n\n")
            
            if 'baseline_performance' in summary:
                f.write("Baseline Performance:\n")
                f.write(f"  Mean F1: {summary['baseline_performance']['mean_f1']:.3f}\n")
                f.write(f"  Std F1: {summary['baseline_performance']['std_f1']:.3f}\n")
                f.write(f"  Best F1: {summary['baseline_performance']['best_f1']:.3f}\n\n")
            
            if 'mainmodel_performance' in summary:
                f.write("MainModel Performance:\n")
                f.write(f"  Mean F1 Macro: {summary['mainmodel_performance']['mean_f1_macro']:.3f}\n")
                f.write(f"  Std F1 Macro: {summary['mainmodel_performance']['std_f1_macro']:.3f}\n")
                f.write(f"  Best F1 Macro: {summary['mainmodel_performance']['best_f1_macro']:.3f}\n")
        
        summary_files['readable_summary'] = str(readable_file)
        
        print(f"✅ Experiment summary saved to {summary_dir}")
        return summary_files
    
    def _save_baseline_results_organized(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Save baseline results in organized structure"""
        saved_files = {}
        baseline_dir = self.analysis_dirs['baseline_models']
        pred_dir = baseline_dir / "individual_predictions"
        
        # Save individual model predictions and results
        individual_results = []
        all_predictions = {}
        
        for model_name, model_results in results.get('individual_models', {}).items():
            if isinstance(model_results, dict):
                # Save predictions
                pred_file = pred_dir / f"{model_name}_predictions.npy"
                if 'predictions' in model_results:
                    np.save(pred_file, model_results['predictions'])
                    saved_files[f"{model_name}_predictions"] = str(pred_file)
                
                # Save CSV version
                csv_file = pred_dir / f"{model_name}_predictions.csv"
                if 'predictions' in model_results and 'true_labels' in model_results:
                    # Handle multi-label data properly
                    true_labels = model_results['true_labels']
                    predictions = model_results['predictions']
                    
                    # For multi-label data, create separate columns for each label
                    if true_labels.ndim == 2 and predictions.ndim == 2:
                        # Multi-label: create columns for each label
                        df_data = {}
                        for i in range(true_labels.shape[1]):
                            df_data[f'true_label_{i}'] = true_labels[:, i]
                            df_data[f'prediction_{i}'] = predictions[:, i]
                        df = pd.DataFrame(df_data)
                    else:
                        # Single-label: use original format
                        df = pd.DataFrame({
                            'true_labels': true_labels,
                            'predictions': predictions
                        })
                    df.to_csv(csv_file, index=False)
                    saved_files[f"{model_name}_predictions_csv"] = str(csv_file)
                
                # Save JSON version
                json_file = pred_dir / f"{model_name}_predictions.json"
                json_data = {
                    'model_name': model_name,
                    'predictions': model_results.get('predictions', []).tolist() if hasattr(model_results.get('predictions', []), 'tolist') else model_results.get('predictions', []),
                    'true_labels': model_results.get('true_labels', []).tolist() if hasattr(model_results.get('true_labels', []), 'tolist') else model_results.get('true_labels', []),
                    'metrics': {}
                }
                
                # Convert metrics to JSON-serializable format
                for k, v in model_results.items():
                    if k not in ['predictions', 'true_labels']:
                        if isinstance(v, np.ndarray):
                            json_data['metrics'][k] = v.tolist()
                        elif isinstance(v, (np.integer, np.floating)):
                            json_data['metrics'][k] = float(v)
                        else:
                            json_data['metrics'][k] = v
                
                with open(json_file, 'w') as f:
                    json.dump(json_data, f, indent=2)
                saved_files[f"{model_name}_predictions_json"] = str(json_file)
                
                individual_results.append({
                    'model_name': model_name,
                    'accuracy': model_results.get('accuracy', 0),
                    'f1_score': model_results.get('f1_score', 0),
                    'f1_macro': model_results.get('f1_macro', 0),
                    'f1_micro': model_results.get('f1_micro', 0)
                })
                
                all_predictions[model_name] = model_results.get('predictions', [])
        
        # Save summary
        summary_file = baseline_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results.get('summary', {}), f, indent=2)
        saved_files['summary'] = str(summary_file)
        
        # Save individual results
        individual_file = baseline_dir / "individual_results.json"
        with open(individual_file, 'w') as f:
            json.dump(individual_results, f, indent=2)
        saved_files['individual_results'] = str(individual_file)
        
        # Save all predictions
        all_pred_file = baseline_dir / "all_predictions.pkl"
        with open(all_pred_file, 'wb') as f:
            pickle.dump(all_predictions, f)
        saved_files['all_predictions'] = str(all_pred_file)
        
        print(f"✅ Baseline results saved to {baseline_dir}")
        return saved_files
    
    def _save_hyperparameter_results_organized(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Save hyperparameter search results in organized structure"""
        saved_files = {}
        hp_dir = self.analysis_dirs['hyperparameter_tuning']
        
        # Save best parameters
        best_params_file = hp_dir / "best_parameters.json"
        with open(best_params_file, 'w') as f:
            json.dump(results.get('best_params', {}), f, indent=2)
        saved_files['best_parameters'] = str(best_params_file)
        
        # Save all trials
        trials_file = hp_dir / "all_trials.json"
        with open(trials_file, 'w') as f:
            json.dump(results.get('trials', []), f, indent=2)
        saved_files['all_trials'] = str(trials_file)
        
        # Save optimization history
        history_file = hp_dir / "optimization_history.csv"
        if 'trials' in results:
            history_data = []
            for trial in results['trials']:
                history_data.append({
                    'trial': trial.get('trial', 0),
                    'f1_score': trial.get('f1_score', 0),
                    'accuracy': trial.get('accuracy', 0),
                    'n_bags': trial.get('params', {}).get('n_bags', 0),
                    'epochs': trial.get('params', {}).get('epochs', 0),
                    'batch_size': trial.get('params', {}).get('batch_size', 0),
                    'dropout_strategy': trial.get('params', {}).get('dropout_strategy', '')
                })
            df = pd.DataFrame(history_data)
            df.to_csv(history_file, index=False)
            saved_files['optimization_history'] = str(history_file)
        
        print(f"✅ Hyperparameter results saved to {hp_dir}")
        return saved_files
    
    def _save_mainmodel_results_organized(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Save main model results in organized structure"""
        saved_files = {}
        main_dir = self.analysis_dirs['main_model']
        pred_dir = main_dir / "test_predictions"
        
        # Save simple test results
        if 'simple_test' in results:
            simple_results = results['simple_test']
            
            # Save predictions
            if 'predictions' in simple_results:
                pred_file = pred_dir / "simple_test_predictions.npy"
                np.save(pred_file, simple_results['predictions'])
                saved_files['simple_test_predictions'] = str(pred_file)
            
            # Save metrics
            metrics_file = main_dir / "simple_test_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(simple_results.get('metrics', {}), f, indent=2)
            saved_files['simple_test_metrics'] = str(metrics_file)
        
        # Save optimized test results
        if 'optimized_test' in results:
            opt_results = results['optimized_test']
            
            # Save predictions
            if 'predictions' in opt_results:
                pred_file = pred_dir / "optimized_test_predictions.npy"
                np.save(pred_file, opt_results['predictions'])
                saved_files['optimized_test_predictions'] = str(pred_file)
            
            # Save metrics
            metrics_file = main_dir / "optimized_test_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(opt_results.get('metrics', {}), f, indent=2)
            saved_files['optimized_test_metrics'] = str(metrics_file)
        
        # Save all predictions
        all_pred_file = main_dir / "all_predictions.pkl"
        all_predictions = {
            'simple_test': results.get('simple_test', {}).get('predictions', []),
            'optimized_test': results.get('optimized_test', {}).get('predictions', [])
        }
        with open(all_pred_file, 'wb') as f:
            pickle.dump(all_predictions, f)
        saved_files['all_predictions'] = str(all_pred_file)
        
        print(f"✅ Main model results saved to {main_dir}")
        return saved_files
    
    def _save_cv_results_organized(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Save cross-validation results in organized structure"""
        saved_files = {}
        cv_dir = self.analysis_dirs['cross_validation']
        
        # Save CV statistics
        cv_stats_file = cv_dir / "cv_statistics.json"
        with open(cv_stats_file, 'w') as f:
            json.dump(results, f, indent=2)
        saved_files['cv_statistics'] = str(cv_stats_file)
        
        # Save CV analysis report
        cv_report_file = cv_dir / "cv_analysis.txt"
        with open(cv_report_file, 'w') as f:
            f.write("Cross-Validation Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            for model_name, model_results in results.get('cv_results', {}).items():
                f.write(f"Model: {model_name}\n")
                f.write(f"Mean Accuracy: {model_results.get('mean_accuracy', 0):.3f}\n")
                f.write(f"Std Accuracy: {model_results.get('std_accuracy', 0):.3f}\n\n")
        saved_files['cv_analysis'] = str(cv_report_file)
        
        print(f"✅ Cross-validation results saved to {cv_dir}")
        return saved_files
    
    def _save_interpretability_results_organized(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Save interpretability results in organized structure"""
        saved_files = {}
        interp_dir = self.analysis_dirs['interpretability']
        
        # Save feature importance
        if 'feature_importance' in results:
            fi_file = interp_dir / "feature_importance.json"
            with open(fi_file, 'w') as f:
                json.dump(results['feature_importance'], f, indent=2)
            saved_files['feature_importance'] = str(fi_file)
        
        # Save confidence analysis
        if 'confidence_analysis' in results:
            conf_file = interp_dir / "confidence_analysis.json"
            with open(conf_file, 'w') as f:
                json.dump(results['confidence_analysis'], f, indent=2)
            saved_files['confidence_analysis'] = str(conf_file)
        
        # Save error analysis
        if 'error_analysis' in results:
            error_file = interp_dir / "error_analysis.json"
            with open(error_file, 'w') as f:
                json.dump(results['error_analysis'], f, indent=2)
            saved_files['error_analysis'] = str(error_file)
        
        print(f"✅ Interpretability results saved to {interp_dir}")
        return saved_files
    
    def _save_ablation_results_organized(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Save ablation results in organized structure"""
        saved_files = {}
        ablation_dir = self.analysis_dirs['ablation_studies']
        
        # Save all ablation results
        ablation_file = ablation_dir / "ablation_results.json"
        with open(ablation_file, 'w') as f:
            json.dump(results, f, indent=2)
        saved_files['ablation_results'] = str(ablation_file)
        
        # Save comprehensive analysis
        if 'comprehensive_analysis' in results:
            comp_file = ablation_dir / "comprehensive_analysis.json"
            with open(comp_file, 'w') as f:
                json.dump(results['comprehensive_analysis'], f, indent=2, default=str)
            saved_files['comprehensive_analysis'] = str(comp_file)
        
        print(f"✅ Ablation results saved to {ablation_dir}")
        return saved_files
    
    def _save_robustness_results_organized(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Save robustness results in organized structure"""
        saved_files = {}
        robust_dir = self.analysis_dirs['robustness_analysis']
        
        # Save robustness tests
        robust_file = robust_dir / "robustness_tests.json"
        with open(robust_file, 'w') as f:
            json.dump(results, f, indent=2)
        saved_files['robustness_tests'] = str(robust_file)
        
        # Save robustness report
        report_file = robust_dir / "robustness_report.txt"
        with open(report_file, 'w') as f:
            f.write("Robustness Testing Report\n")
            f.write("=" * 30 + "\n\n")
            for scenario, scenario_results in results.items():
                if scenario != 'pathology_name':
                    f.write(f"Scenario: {scenario}\n")
                    f.write(f"Accuracy: {scenario_results.get('accuracy', 0):.3f}\n")
                    f.write(f"Method: {scenario_results.get('method', 'unknown')}\n\n")
        saved_files['robustness_report'] = str(report_file)
        
        print(f"✅ Robustness results saved to {robust_dir}")
        return saved_files
    
    def _create_experiment_manifest(self, saved_files: Dict[str, Any], all_results: Dict[str, Any]) -> Path:
        """Create experiment manifest file"""
        manifest_file = self.run_dir / "experiment_manifest.json"
        
        manifest = {
            'experiment_id': self.timestamp,
            'dataset': 'ChestX-ray14',
            'task': '14-class multi-label pathology classification',
            'timestamp': self.timestamp,
            'saved_files': saved_files,
            'experiment_summary': {
                'baseline_models': len(all_results.get('baseline_results', {}).get('individual_models', {})),
                'hyperparameter_trials': len(all_results.get('mainmodel_results', {}).get('hyperparameter_search', {}).get('trials', [])),
                'robustness_scenarios': len(all_results.get('robustness_results', {})),
                'ablation_studies': len(all_results.get('ablation_results', {}).get('ablation_experiments', {}))
            }
        }
        
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return manifest_file
    
    def _export_summary_statistics(self, all_results: Dict[str, Any], output_file: Path):
        """Export summary statistics for publication"""
        with open(output_file, 'w') as f:
            f.write("ChestX-ray14 Experiment Summary\n")
            f.write("=" * 40 + "\n\n")
            
            # Baseline summary
            if 'baseline_results' in all_results:
                baseline_summary = all_results['baseline_results'].get('summary', {})
                f.write("Baseline Models:\n")
                f.write(f"Best Model: {baseline_summary.get('best_model', 'N/A')}\n")
                f.write(f"Best F1: {baseline_summary.get('best_f1', 0):.3f}\n\n")
            
            # Main model summary
            if 'mainmodel_results' in all_results:
                main_summary = all_results['mainmodel_results']
                f.write("Main Model:\n")
                if 'simple_test' in main_summary:
                    simple_metrics = main_summary['simple_test'].get('metrics', {})
                    f.write(f"Simple Test F1: {simple_metrics.get('f1_macro', 0):.3f}\n")
                if 'optimized_test' in main_summary:
                    opt_metrics = main_summary['optimized_test'].get('metrics', {})
                    f.write(f"Optimized Test F1: {opt_metrics.get('f1_macro', 0):.3f}\n\n")
    
    def _export_performance_comparison(self, all_results: Dict[str, Any], output_file: Path):
        """Export performance comparison for publication"""
        comparison_data = []
        
        # Add baseline models
        if 'baseline_results' in all_results:
            for model_name, model_results in all_results['baseline_results'].get('individual_models', {}).items():
                comparison_data.append({
                    'Model': model_name,
                    'Type': 'Baseline',
                    'Accuracy': model_results.get('accuracy', 0),
                    'F1_Macro': model_results.get('f1_macro', 0),
                    'F1_Micro': model_results.get('f1_micro', 0)
                })
        
        # Add main model results
        if 'mainmodel_results' in all_results:
            main_results = all_results['mainmodel_results']
            if 'simple_test' in main_results:
                metrics = main_results['simple_test'].get('metrics', {})
                comparison_data.append({
                    'Model': 'MainModel_Simple',
                    'Type': 'MainModel',
                    'Accuracy': metrics.get('accuracy', 0),
                    'F1_Macro': metrics.get('f1_macro', 0),
                    'F1_Micro': metrics.get('f1_micro', 0)
                })
            if 'optimized_test' in main_results:
                metrics = main_results['optimized_test'].get('metrics', {})
                comparison_data.append({
                    'Model': 'MainModel_Optimized',
                    'Type': 'MainModel',
                    'Accuracy': metrics.get('accuracy', 0),
                    'F1_Macro': metrics.get('f1_macro', 0),
                    'F1_Micro': metrics.get('f1_micro', 0)
                })
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(output_file, index=False)
    
    def _export_key_metrics(self, all_results: Dict[str, Any], output_file: Path):
        """Export key metrics for publication"""
        key_metrics = {
            'experiment_id': self.timestamp,
            'dataset': 'ChestX-ray14',
            'baseline_best_f1': 0,
            'mainmodel_simple_f1': 0,
            'mainmodel_optimized_f1': 0,
            'robustness_baseline_accuracy': 0,
            'robustness_missing_image_accuracy': 0,
            'robustness_missing_text_accuracy': 0,
            'robustness_missing_metadata_accuracy': 0,
            'robustness_noisy_image_accuracy': 0
        }
        
        # Extract key metrics
        if 'baseline_results' in all_results:
            key_metrics['baseline_best_f1'] = all_results['baseline_results'].get('summary', {}).get('best_f1', 0)
        
        if 'mainmodel_results' in all_results:
            main_results = all_results['mainmodel_results']
            if 'simple_test' in main_results:
                key_metrics['mainmodel_simple_f1'] = main_results['simple_test'].get('metrics', {}).get('f1_macro', 0)
            if 'optimized_test' in main_results:
                key_metrics['mainmodel_optimized_f1'] = main_results['optimized_test'].get('metrics', {}).get('f1_macro', 0)
        
        if 'robustness_results' in all_results:
            robust_results = all_results['robustness_results']
            key_metrics['robustness_baseline_accuracy'] = robust_results.get('baseline', {}).get('accuracy', 0)
            key_metrics['robustness_missing_image_accuracy'] = robust_results.get('missing_image', {}).get('accuracy', 0)
            key_metrics['robustness_missing_text_accuracy'] = robust_results.get('missing_text', {}).get('accuracy', 0)
            key_metrics['robustness_missing_metadata_accuracy'] = robust_results.get('missing_metadata', {}).get('accuracy', 0)
            key_metrics['robustness_noisy_image_accuracy'] = robust_results.get('noisy_image', {}).get('accuracy', 0)
        
        with open(output_file, 'w') as f:
            json.dump(key_metrics, f, indent=2)
    
    def _save_component_results(self, experiment_name: str, results: Dict[str, Any]) -> None:
        """Save comprehensive results for each experiment component"""
        try:
            if experiment_name == "baselines":
                self._save_baseline_results(results)
            elif experiment_name == "hyperparameter_search":
                self._save_hyperparameter_results(results)
            elif experiment_name in ["mainmodel_simple", "mainmodel_optimized"]:
                self._save_mainmodel_results(experiment_name, results)
            elif experiment_name == "ablation_studies":
                self._save_ablation_results(results)
            elif experiment_name == "robustness_testing":
                self._save_robustness_results(results)
            elif experiment_name == "hypothesis_testing":
                self._save_hypothesis_results(results)
                
        except Exception as e:
            print(f"⚠️ Warning: Could not save {experiment_name} results: {e}")
    
    def _save_baseline_results(self, results: Dict[str, Any]) -> None:
        """Save comprehensive baseline model results"""
        print("💾 Saving baseline results...")
        
        # Save predictions for each baseline model
        baseline_predictions = {}
        baseline_metrics = []
        
        for model_name, model_results in results.items():
            if isinstance(model_results, dict) and 'predictions' in model_results:
                # Save individual model predictions
                pred_file = self.predictions_path / f"baseline_{model_name}_predictions.npz"
                np.savez_compressed(pred_file, 
                                  predictions=model_results['predictions'],
                                  true_labels=model_results.get('true_labels', []))
                baseline_predictions[model_name] = str(pred_file)
                
                # Collect metrics
                baseline_metrics.append({
                    'model_name': model_name,
                    'f1_score': model_results.get('f1_score', 0),
                    'f1_macro': model_results.get('f1_macro', 0),
                    'f1_micro': model_results.get('f1_micro', 0),
                    'accuracy': model_results.get('accuracy', 0),
                    'precision_macro': model_results.get('precision_macro', 0),
                    'recall_macro': model_results.get('recall_macro', 0),
                    'hamming_loss': model_results.get('hamming_loss', 1.0),
                    'jaccard_score': model_results.get('jaccard_score', 0),
                    'training_time': model_results.get('training_time', 0),
                    'inference_time': model_results.get('inference_time', 0),
                    'error': model_results.get('error', '')
                })
        
        # Save baseline metrics summary
        if baseline_metrics:
            metrics_df = pd.DataFrame(baseline_metrics)
            metrics_file = self.metrics_path / "baseline_models_metrics.csv"
            metrics_df.to_csv(metrics_file, index=False)
            
            # Save baseline ranking analysis
            successful_models = metrics_df[metrics_df['f1_score'] > 0].sort_values('f1_score', ascending=False)
            analysis_file = self.analysis_path / "baseline_analysis.json"
            
            analysis = {
                'total_models_tested': len(results),
                'successful_models': len(successful_models),
                'failed_models': len(results) - len(successful_models),
                'best_model': {
                    'name': successful_models.iloc[0]['model_name'] if len(successful_models) > 0 else None,
                    'f1_score': float(successful_models.iloc[0]['f1_score']) if len(successful_models) > 0 else 0,
                    'accuracy': float(successful_models.iloc[0]['accuracy']) if len(successful_models) > 0 else 0
                },
                'model_categories': self._analyze_baseline_categories(baseline_metrics),
                'performance_distribution': {
                    'f1_mean': float(successful_models['f1_score'].mean()) if len(successful_models) > 0 else 0,
                    'f1_std': float(successful_models['f1_score'].std()) if len(successful_models) > 0 else 0,
                    'f1_min': float(successful_models['f1_score'].min()) if len(successful_models) > 0 else 0,
                    'f1_max': float(successful_models['f1_score'].max()) if len(successful_models) > 0 else 0
                }
            }
            
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
                
        print(f"✅ Baseline results saved: {len(baseline_metrics)} models")
    
    def _save_hyperparameter_results(self, results: Dict[str, Any]) -> None:
        """Save comprehensive hyperparameter optimization results"""
        print("💾 Saving hyperparameter optimization results...")
        
        # Save all trials data
        if 'all_trials' in results:
            trials_df = pd.DataFrame(results['all_trials'])
            trials_file = self.metrics_path / "hyperparameter_trials.csv"
            trials_df.to_csv(trials_file, index=False)
            
        # Save best parameters and analysis
        hp_analysis = {
            'best_params': results.get('best_params', {}),
            'best_score': results.get('best_score', 0),
            'total_trials': results.get('total_trials', 0),
            'optimization_time': results.get('optimization_time', 0),
            'parameter_importance': results.get('parameter_importance', {}),
            'convergence_analysis': results.get('convergence_data', [])
        }
        
        analysis_file = self.analysis_path / "hyperparameter_optimization_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(hp_analysis, f, indent=2)
            
        print("✅ Hyperparameter optimization results saved")
    
    def _save_mainmodel_results(self, experiment_name: str, results: Dict[str, Any]) -> None:
        """Save comprehensive MainModel results"""
        print(f"💾 Saving {experiment_name} results...")
        
        # Save predictions
        if 'predictions' in results:
            pred_file = self.predictions_path / f"{experiment_name}_predictions.npz"
            np.savez_compressed(pred_file,
                              predictions=results['predictions'],
                              true_labels=results.get('true_labels', []),
                              prediction_probabilities=results.get('prediction_probabilities', []),
                              confidence_scores=results.get('confidence_scores', []))
        
        # Save detailed metrics
        mainmodel_metrics = {
            'experiment_type': experiment_name,
            'pathology_name': results.get('pathology_name', 'unknown'),
            'model_parameters': results.get('optimized_params', results.get('model_params', {})),
            'performance_metrics': {
                'accuracy': results.get('accuracy', 0),
                'precision': results.get('precision', 0),
                'recall': results.get('recall', 0),
                'f1_score': results.get('f1_score', 0),
                'auc_score': results.get('auc_score', 0),
                'per_pathology_metrics': results.get('per_pathology_metrics', {})
            },
            'training_info': {
                'train_time': results.get('train_time', 0),
                'inference_time': results.get('inference_time', 0),
                'n_bags': results.get('n_bags', 0),
                'epochs': results.get('epochs', 0),
                'convergence_info': results.get('convergence_info', {})
            },
            'interpretability': {
                'modality_importance': results.get('modality_importance', {}),
                'feature_importance': results.get('feature_importance', {}),
                'attention_weights': results.get('attention_weights', {})
            }
        }
        
        metrics_file = self.metrics_path / f"{experiment_name}_detailed_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(mainmodel_metrics, f, indent=2, default=str)
            
        print(f"✅ {experiment_name} results saved")
    
    def _save_ablation_results(self, results: Dict[str, Any]) -> None:
        """Save comprehensive ablation study results"""
        print("💾 Saving comprehensive ablation study results...")
        
        # Check if this is the new comprehensive format
        if 'ablation_experiments' in results and 'comprehensive_analysis' in results:
            self._save_comprehensive_ablation_results(results)
        else:
            self._save_legacy_ablation_results(results)
    
    def _save_comprehensive_ablation_results(self, results: Dict[str, Any]) -> None:
        """Save new comprehensive ablation study results"""
        
        ablation_experiments = results['ablation_experiments']
        comprehensive_analysis = results['comprehensive_analysis']
        
        # Save basic modality ablation
        if 'basic_modality_ablation' in ablation_experiments:
            self._save_basic_modality_ablation(ablation_experiments['basic_modality_ablation'])
        
        # Save feature ablations
        feature_ablations = [
            ('image_feature_ablation', 'Image Feature'),
            ('clinical_text_feature_ablation', 'Clinical Text Feature'),
            ('medical_metadata_feature_ablation', 'Medical Metadata Feature')
        ]
        
        for ablation_key, ablation_name in feature_ablations:
            if ablation_key in ablation_experiments:
                self._save_feature_ablation_results(
                    ablation_experiments[ablation_key], ablation_key, ablation_name
                )
        
        # Save fusion strategy ablation
        if 'medical_fusion_strategy_ablation' in ablation_experiments:
            self._save_fusion_strategy_results(ablation_experiments['medical_fusion_strategy_ablation'])
        
        # Save algorithm sensitivity ablation
        if 'algorithm_sensitivity_ablation' in ablation_experiments:
            self._save_algorithm_sensitivity_results(ablation_experiments['algorithm_sensitivity_ablation'])
        
        # Save comprehensive analysis
        analysis_file = self.analysis_path / "comprehensive_ablation_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(comprehensive_analysis, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_ablation_summary_report(results)
        
        print("✅ Comprehensive ablation study results saved")
    
    def _save_basic_modality_ablation(self, basic_ablation: Dict[str, Any]) -> None:
        """Save basic modality ablation results"""
        
        if 'modality_results' in basic_ablation:
            modality_data = []
            
            for modality_combo, combo_results in basic_ablation['modality_results'].items():
                if 'algorithm_results' in combo_results:
                    best_result = combo_results.get('best_performance', {})
                    modality_data.append({
                        'modality_combination': modality_combo,
                        'modalities_used': ', '.join(combo_results.get('modalities_used', [])),
                        'best_f1_score': best_result.get('f1_score', 0),
                        'best_accuracy': best_result.get('accuracy', 0),
                        'n_modalities': len(combo_results.get('modalities_used', []))
                    })
            
            if modality_data:
                modality_df = pd.DataFrame(modality_data)
                modality_file = self.metrics_path / "basic_modality_ablation.csv"
                modality_df.to_csv(modality_file, index=False)
        
        # Save contribution analysis
        if 'contribution_analysis' in basic_ablation:
            contrib_file = self.analysis_path / "modality_contribution_analysis.json"
            with open(contrib_file, 'w') as f:
                json.dump(basic_ablation['contribution_analysis'], f, indent=2, default=str)
    
    def _save_feature_ablation_results(self, feature_ablation: Dict[str, Any], 
                                     ablation_key: str, ablation_name: str) -> None:
        """Save feature ablation results"""
        
        if 'feature_subset_results' in feature_ablation:
            feature_data = []
            
            for subset_name, subset_results in feature_ablation['feature_subset_results'].items():
                if 'f1_score' in subset_results:
                    feature_data.append({
                        'feature_subset': subset_name,
                        'f1_score': subset_results['f1_score'],
                        'accuracy': subset_results.get('accuracy', 0),
                        'n_features': subset_results.get('n_features', 0),
                        'importance_drop': subset_results.get('importance_drop', 0)
                    })
            
            if feature_data:
                feature_df = pd.DataFrame(feature_data)
                feature_file = self.metrics_path / f"{ablation_key}_results.csv"
                feature_df.to_csv(feature_file, index=False)
        
        elif 'feature_removal_results' in feature_ablation:
            removal_data = []
            
            for removal_name, removal_results in feature_ablation['feature_removal_results'].items():
                if 'f1_score' in removal_results:
                    removal_data.append({
                        'removal_condition': removal_name,
                        'f1_score': removal_results['f1_score'],
                        'accuracy': removal_results.get('accuracy', 0),
                        'features_used': removal_results.get('features_used', 0),
                        'removed_feature': removal_results.get('removed_feature', 'none'),
                        'importance_drop': removal_results.get('importance_drop', 0)
                    })
            
            if removal_data:
                removal_df = pd.DataFrame(removal_data)
                removal_file = self.metrics_path / f"{ablation_key}_results.csv"
                removal_df.to_csv(removal_file, index=False)
        
        # Save feature importance analysis
        if 'most_important_features' in feature_ablation:
            importance_file = self.analysis_path / f"{ablation_key}_importance.json"
            with open(importance_file, 'w') as f:
                json.dump({
                    'most_important_features': feature_ablation['most_important_features'],
                    'analysis': feature_ablation.get('analysis', {})
                }, f, indent=2, default=str)
    
    def _save_fusion_strategy_results(self, fusion_ablation: Dict[str, Any]) -> None:
        """Save fusion strategy ablation results"""
        
        if 'fusion_strategy_results' in fusion_ablation:
            fusion_data = []
            
            for strategy_name, strategy_results in fusion_ablation['fusion_strategy_results'].items():
                if 'f1_score' in strategy_results:
                    fusion_data.append({
                        'fusion_strategy': strategy_name,
                        'f1_score': strategy_results['f1_score'],
                        'accuracy': strategy_results.get('accuracy', 0)
                    })
            
            if fusion_data:
                fusion_df = pd.DataFrame(fusion_data)
                fusion_file = self.metrics_path / "fusion_strategy_ablation.csv"
                fusion_df.to_csv(fusion_file, index=False)
                
                # Save best strategy analysis
                best_strategy_file = self.analysis_path / "best_fusion_strategy.json"
                with open(best_strategy_file, 'w') as f:
                    json.dump({
                        'best_fusion_strategy': fusion_ablation.get('best_fusion_strategy', 'unknown'),
                        'strategy_performance': fusion_ablation['fusion_strategy_results']
                    }, f, indent=2, default=str)
    
    def _save_algorithm_sensitivity_results(self, algo_ablation: Dict[str, Any]) -> None:
        """Save algorithm sensitivity ablation results"""
        
        if 'algorithm_sensitivity_results' in algo_ablation:
            algo_data = []
            
            for modality, algo_results in algo_ablation['algorithm_sensitivity_results'].items():
                for algorithm, performance in algo_results.items():
                    algo_data.append({
                        'modality': modality,
                        'algorithm': algorithm,
                        'f1_score': performance
                    })
            
            if algo_data:
                algo_df = pd.DataFrame(algo_data)
                algo_file = self.metrics_path / "algorithm_sensitivity_ablation.csv"
                algo_df.to_csv(algo_file, index=False)
                
                # Save algorithm preferences
                preferences_file = self.analysis_path / "algorithm_preferences.json"
                with open(preferences_file, 'w') as f:
                    json.dump({
                        'modality_algorithm_preferences': algo_ablation.get('modality_algorithm_preferences', {}),
                        'sensitivity_results': algo_ablation['algorithm_sensitivity_results']
                    }, f, indent=2, default=str)
    
    def _generate_ablation_summary_report(self, results: Dict[str, Any]) -> None:
        """Generate a comprehensive ablation summary report"""
        
        comprehensive_analysis = results.get('comprehensive_analysis', {})
        
        summary = {
            'experiment_type': 'Comprehensive Medical Ablation Study',
            'total_ablation_categories': len(results.get('ablation_experiments', {})),
            'key_findings': comprehensive_analysis.get('key_findings', []),
            'modality_insights': comprehensive_analysis.get('modality_insights', {}),
            'feature_insights': comprehensive_analysis.get('feature_insights', {}),
            'fusion_insights': comprehensive_analysis.get('fusion_insights', {}),
            'algorithm_insights': comprehensive_analysis.get('algorithm_insights', {}),
            'clinical_recommendations': comprehensive_analysis.get('clinical_recommendations', []),
            'pathology_tested': results.get('ablation_experiments', {}).get('basic_modality_ablation', {}).get('pathology_tested', 'Unknown')
        }
        
        summary_file = self.analysis_path / "ablation_study_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create human-readable report
        self._create_ablation_text_report(summary)
    
    def _create_ablation_text_report(self, summary: Dict[str, Any]) -> None:
        """Create a human-readable text report of ablation results"""
        
        report_lines = [
            "=" * 80,
            "COMPREHENSIVE MEDICAL ABLATION STUDY REPORT",
            "=" * 80,
            f"Pathology Tested: {summary.get('pathology_tested', 'Unknown')}",
            f"Total Ablation Categories: {summary.get('total_ablation_categories', 0)}",
            "",
            "KEY FINDINGS:",
            "-" * 40
        ]
        
        for finding in summary.get('key_findings', []):
            report_lines.append(f"• {finding}")
        
        report_lines.extend([
            "",
            "CLINICAL RECOMMENDATIONS:",
            "-" * 40
        ])
        
        for recommendation in summary.get('clinical_recommendations', []):
            report_lines.append(f"• {recommendation}")
        
        # Add modality insights
        modality_insights = summary.get('modality_insights', {})
        if modality_insights:
            report_lines.extend([
                "",
                "MODALITY PERFORMANCE:",
                "-" * 40,
                f"• Image Performance: {modality_insights.get('image_performance', 0):.3f}",
                f"• Text Performance: {modality_insights.get('text_performance', 0):.3f}",
                f"• Metadata Performance: {modality_insights.get('metadata_performance', 0):.3f}",
                f"• Multimodal Benefit: {modality_insights.get('multimodal_benefit', 0):.3f}",
                f"• Best Single Modality: {modality_insights.get('best_single_modality', 'Unknown')}"
            ])
        
        # Add fusion insights
        fusion_insights = summary.get('fusion_insights', {})
        if fusion_insights:
            report_lines.extend([
                "",
                "FUSION STRATEGY ANALYSIS:",
                "-" * 40,
                f"• Best Medical Strategy: {fusion_insights.get('best_medical_strategy', 'Unknown')}"
            ])
        
        report_lines.append("=" * 80)
        
        # Save text report
        report_file = self.analysis_path / "ablation_study_report.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
    
    def _save_legacy_ablation_results(self, results: Dict[str, Any]) -> None:
        """Save legacy format ablation study results"""
        
        ablation_data = []
        ablation_predictions = {}
        
        for component_name, component_results in results.items():
            if isinstance(component_results, dict):
                # Save predictions for each ablation
                if 'predictions' in component_results:
                    pred_file = self.predictions_path / f"ablation_{component_name}_predictions.npz"
                    np.savez_compressed(pred_file,
                                      predictions=component_results['predictions'],
                                      true_labels=component_results.get('true_labels', []))
                    ablation_predictions[component_name] = str(pred_file)
                
                # Collect ablation metrics
                ablation_data.append({
                    'component_removed': component_name,
                    'f1_score': component_results.get('f1_score', 0),
                    'accuracy': component_results.get('accuracy', 0),
                    'precision': component_results.get('precision', 0),
                    'recall': component_results.get('recall', 0),
                    'performance_drop': component_results.get('performance_drop', 0),
                    'training_time': component_results.get('training_time', 0),
                    'component_importance': component_results.get('component_importance', 0)
                })
        
        # Save ablation metrics
        if ablation_data:
            ablation_df = pd.DataFrame(ablation_data)
            metrics_file = self.metrics_path / "ablation_study_metrics.csv"
            ablation_df.to_csv(metrics_file, index=False)
            
            # Analysis of component importance
            ablation_analysis = {
                'component_ranking': ablation_df.sort_values('performance_drop', ascending=False).to_dict('records'),
                'most_important_component': ablation_df.loc[ablation_df['performance_drop'].idxmax()]['component_removed'] if len(ablation_df) > 0 else None,
                'least_important_component': ablation_df.loc[ablation_df['performance_drop'].idxmin()]['component_removed'] if len(ablation_df) > 0 else None,
                'average_performance_drop': float(ablation_df['performance_drop'].mean()) if len(ablation_df) > 0 else 0,
                'hypothesis_5_validation': {
                    'all_components_important': all(ablation_df['performance_drop'] > 0.01),  # 1% threshold
                    'significant_drops': len(ablation_df[ablation_df['performance_drop'] > 0.05]),  # 5% threshold
                    'validation_score': float(ablation_df['performance_drop'].mean()) if len(ablation_df) > 0 else 0
                }
            }
            
            analysis_file = self.analysis_path / "ablation_study_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(ablation_analysis, f, indent=2)
                
        print("✅ Legacy ablation study results saved")
    
    def _save_robustness_results(self, results: Dict[str, Any]) -> None:
        """Save comprehensive robustness testing results"""
        print("💾 Saving robustness testing results...")
        
        # Save predictions for each robustness test
        robustness_data = []
        for test_name, test_results in results.items():
            if isinstance(test_results, dict):
                # Save predictions
                if 'predictions' in test_results:
                    pred_file = self.predictions_path / f"robustness_{test_name}_predictions.npz"
                    np.savez_compressed(pred_file,
                                      predictions=test_results['predictions'],
                                      true_labels=test_results.get('true_labels', []))
                
                # Collect robustness metrics
                robustness_data.append({
                    'test_scenario': test_name,
                    'f1_score': test_results.get('f1_score', 0),
                    'accuracy': test_results.get('accuracy', 0),
                    'performance_degradation': test_results.get('performance_degradation', 0),
                    'missing_modality_percent': test_results.get('missing_modality_percent', 0),
                    'robustness_score': test_results.get('robustness_score', 0)
                })
        
        # Save robustness metrics
        if robustness_data:
            robustness_df = pd.DataFrame(robustness_data)
            metrics_file = self.metrics_path / "robustness_testing_metrics.csv"
            robustness_df.to_csv(metrics_file, index=False)
            
            # Robustness analysis
            robustness_analysis = {
                'overall_robustness_score': float(robustness_df['robustness_score'].mean()) if len(robustness_df) > 0 else 0,
                'best_scenario': robustness_df.loc[robustness_df['f1_score'].idxmax()].to_dict() if len(robustness_df) > 0 else {},
                'worst_scenario': robustness_df.loc[robustness_df['f1_score'].idxmin()].to_dict() if len(robustness_df) > 0 else {},
                'hypothesis_2_validation': {
                    'maintains_performance': all(robustness_df['performance_degradation'] < 0.20),  # <20% degradation
                    'average_degradation': float(robustness_df['performance_degradation'].mean()) if len(robustness_df) > 0 else 0,
                    'validation_score': 1.0 - float(robustness_df['performance_degradation'].mean()) if len(robustness_df) > 0 else 0
                }
            }
            
            analysis_file = self.analysis_path / "robustness_testing_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(robustness_analysis, f, indent=2)
                
        print("✅ Robustness testing results saved")
    
    def _save_hypothesis_results(self, results: Dict[str, Any]) -> None:
        """Save comprehensive hypothesis testing results"""
        print("💾 Saving hypothesis testing results...")
        
        hypothesis_data = []
        for hyp_name, hyp_results in results.items():
            if hyp_name != 'overall' and isinstance(hyp_results, dict):
                hypothesis_data.append({
                    'hypothesis': hyp_name,
                    'score': hyp_results.get('score', 0),
                    'status': hyp_results.get('status', 'UNKNOWN'),
                    'evidence': hyp_results.get('evidence', {}),
                    'validation_metrics': hyp_results.get('validation_metrics', {}),
                    'supporting_data': hyp_results.get('supporting_data', {})
                })
        
        # Save hypothesis validation metrics
        if hypothesis_data:
            hyp_df = pd.DataFrame(hypothesis_data)
            metrics_file = self.metrics_path / "hypothesis_validation_metrics.csv"
            hyp_df.to_csv(metrics_file, index=False)
            
        # Save comprehensive hypothesis analysis
        hypothesis_analysis = {
            'individual_hypotheses': results,
            'validation_summary': {
                'total_hypotheses': len(hypothesis_data),
                'validated_hypotheses': len([h for h in hypothesis_data if h['status'] == 'VALIDATED']),
                'partially_validated': len([h for h in hypothesis_data if h['status'] == 'PARTIALLY_VALIDATED']),
                'rejected_hypotheses': len([h for h in hypothesis_data if h['status'] == 'REJECTED']),
                'overall_validation_score': results.get('overall', {}).get('overall_score', 0)
            },
            'research_contributions': self._generate_research_contributions(results)
        }
        
        analysis_file = self.analysis_path / "comprehensive_hypothesis_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(hypothesis_analysis, f, indent=2, default=str)
            
        print("✅ Hypothesis testing results saved")
    
    def _analyze_baseline_categories(self, baseline_metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze baseline model performance by category"""
        categories = {
            'simple_models': ['logistic_regression', 'random_forest', 'svm_linear', 'naive_bayes', 'knn'],
            'ensemble_methods': ['gradient_boosting', 'extra_trees', 'ada_boost', 'bagging_classifier', 'xgboost', 'lightgbm', 'voting_ensemble'],
            'deep_learning': ['mlp_small', 'mlp_medium', 'mlp_large', 'mlp_deep', 'mlp_wide', 'sgd_classifier'],
            'fusion_models': ['early_fusion', 'late_fusion', 'weighted_fusion', 'attention_fusion'],
            'sota_models': ['stacking_ensemble', 'advanced_mlp', 'meta_ensemble']
        }
        
        category_analysis = {}
        for category, model_names in categories.items():
            category_models = [m for m in baseline_metrics if m['model_name'] in model_names and m['f1_score'] > 0]
            if category_models:
                f1_scores = [m['f1_score'] for m in category_models]
                category_analysis[category] = {
                    'count': len(category_models),
                    'best_f1': max(f1_scores),
                    'avg_f1': sum(f1_scores) / len(f1_scores),
                    'best_model': max(category_models, key=lambda x: x['f1_score'])['model_name']
                }
        
        return category_analysis
    
    def _generate_research_contributions(self, hypothesis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate research contributions based on hypothesis validation"""
        contributions = {
            'novel_architecture': {
                'validated': hypothesis_results.get('h5_novel_features', {}).get('status') == 'VALIDATED',
                'contribution': 'Modality-aware adaptive bagging ensemble with transformer-based meta-learner'
            },
            'performance_improvements': {
                'validated': hypothesis_results.get('h1_performance', {}).get('status') == 'VALIDATED',
                'contribution': 'Statistically significant improvements over traditional fusion methods'
            },
            'robustness_advances': {
                'validated': hypothesis_results.get('h2_robustness', {}).get('status') == 'VALIDATED',
                'contribution': 'Superior handling of missing modality scenarios through strategic dropout'
            },
            'interpretability_enhancements': {
                'validated': hypothesis_results.get('h3_interpretability', {}).get('status') == 'VALIDATED',
                'contribution': 'Attention-based modality importance and uncertainty estimation'
            },
            'computational_efficiency': {
                'validated': hypothesis_results.get('h4_scalability', {}).get('status') == 'VALIDATED',
                'contribution': 'Scalable multimodal learning without prohibitive overhead'
            }
        }
        return contributions
    
    def save_comprehensive_results(self) -> bool:
        """Save comprehensive experimental results"""
        print(f"\n💾 SAVING COMPREHENSIVE RESULTS")
        print("=" * 50)
        
        try:
            # Save main results
            results_file = self.path_config.output_path / "comprehensive_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"✅ Main results saved: {results_file}")
            
            # Save experiment metadata
            self._save_experiment_metadata()
            
            # Save comparative analysis
            self._save_comparative_analysis()
            
            # Save final metrics summary
            self._save_final_metrics_summary()
            
            # Save hypothesis summary
            self._save_hypothesis_csv()
            
            print("✅ All comprehensive results saved successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_experiment_metadata(self) -> None:
        """Save experiment configuration and metadata"""
        metadata = {
            'experiment_config': {
                'dataset_name': self.exp_config.dataset_name,
                'task_type': self.exp_config.task_type,
                'use_small_sample': self.exp_config.use_small_sample,
                'random_seed': self.exp_config.random_seed,
                'n_pathologies': self.exp_config.n_pathologies,
                'pathologies': self.exp_config.pathologies,
                'hyperparameter_search_trials': self.exp_config.hyperparameter_search_trials
            },
            'sampling_config': {
                'small_sample_train_size': self.exp_config.small_sample_train_size,
                'small_sample_test_size': self.exp_config.small_sample_test_size
            },
            'model_config': {
                'default_n_bags': self.exp_config.default_n_bags,
                'default_epochs': self.exp_config.default_epochs,
                'default_batch_size': self.exp_config.default_batch_size
            },
            'paths': {
                'data_path': str(self.path_config.data_path),
                'output_path': str(self.path_config.output_path),
                'predictions_path': str(self.predictions_path),
                'metrics_path': str(self.metrics_path),
                'analysis_path': str(self.analysis_path)
            }
        }
        
        metadata_file = self.path_config.output_path / "experiment_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Experiment metadata saved: {metadata_file}")
    
    def _save_comparative_analysis(self) -> None:
        """Save comparative analysis across all methods"""
        comparative_data = []
        
        # Collect all method results
        methods = {}
        
        # MainModel results
        for key in ['mainmodel_optimized', 'mainmodel_simple']:
            if key in self.results:
                methods['MainModel'] = self.results[key]
                break
        
        # Best baseline
        if 'baselines' in self.results:
            baseline_results = self.results['baselines']
            best_baseline_name = None
            best_f1 = 0
            
            for name, result in baseline_results.items():
                if isinstance(result, dict) and 'f1_score' in result:
                    if result['f1_score'] > best_f1:
                        best_f1 = result['f1_score']
                        best_baseline_name = name
                        methods['Best_Baseline'] = result
        
        # Create comparative analysis
        for method_name, results in methods.items():
            comparative_data.append({
                'method': method_name,
                'f1_score': results.get('f1_score', 0),
                'accuracy': results.get('accuracy', 0),
                'precision': results.get('precision', 0),
                'recall': results.get('recall', 0),
                'training_time': results.get('training_time', results.get('train_time', 0)),
                'inference_time': results.get('inference_time', 0)
            })
        
        if comparative_data:
            comp_df = pd.DataFrame(comparative_data)
            comp_file = self.analysis_path / "comparative_analysis.csv"
            comp_df.to_csv(comp_file, index=False)
            
            # Calculate performance gains
            if len(comp_df) > 1:
                mainmodel_row = comp_df[comp_df['method'] == 'MainModel']
                baseline_row = comp_df[comp_df['method'] == 'Best_Baseline']
                
                if len(mainmodel_row) > 0 and len(baseline_row) > 0:
                    performance_gains = {
                        'f1_improvement': float(mainmodel_row['f1_score'].iloc[0] - baseline_row['f1_score'].iloc[0]),
                        'accuracy_improvement': float(mainmodel_row['accuracy'].iloc[0] - baseline_row['accuracy'].iloc[0]),
                        'relative_f1_gain': float((mainmodel_row['f1_score'].iloc[0] - baseline_row['f1_score'].iloc[0]) / baseline_row['f1_score'].iloc[0] * 100) if baseline_row['f1_score'].iloc[0] > 0 else 0,
                        'training_time_ratio': float(mainmodel_row['training_time'].iloc[0] / baseline_row['training_time'].iloc[0]) if baseline_row['training_time'].iloc[0] > 0 else 0
                    }
                    
                    gains_file = self.analysis_path / "performance_gains.json"
                    with open(gains_file, 'w') as f:
                        json.dump(performance_gains, f, indent=2)
            
            print(f"✅ Comparative analysis saved: {comp_file}")
    
    def _save_final_metrics_summary(self) -> None:
        """Save final comprehensive metrics summary"""
        all_metrics = []
        
        # Collect metrics from all experiments
        for exp_name, exp_results in self.results.items():
            if isinstance(exp_results, dict):
                if exp_name in ['mainmodel_simple', 'mainmodel_optimized']:
                    all_metrics.append({
                        'experiment': exp_name,
                        'category': 'MainModel',
                        'f1_score': exp_results.get('f1_score', 0),
                        'accuracy': exp_results.get('accuracy', 0),
                        'training_time': exp_results.get('train_time', exp_results.get('training_time', 0)),
                        'status': 'SUCCESS' if exp_results.get('f1_score', 0) > 0 else 'FAILED'
                    })
                
                elif exp_name == 'baselines':
                    for model_name, model_results in exp_results.items():
                        if isinstance(model_results, dict):
                            all_metrics.append({
                                'experiment': f'baseline_{model_name}',
                                'category': 'Baseline',
                                'f1_score': model_results.get('f1_score', model_results.get('f1_macro', 0)),
                                'accuracy': model_results.get('accuracy', 0),
                                'training_time': model_results.get('training_time', 0),
                                'status': 'SUCCESS' if model_results.get('f1_score', model_results.get('f1_macro', 0)) > 0 else 'FAILED'
                            })
        
        if all_metrics:
            final_df = pd.DataFrame(all_metrics)
            final_file = self.metrics_path / "final_comprehensive_metrics.csv"
            final_df.to_csv(final_file, index=False)
            print(f"✅ Final metrics summary saved: {final_file}")
    
    def _save_metrics_csv(self) -> None:
        """Save metrics summary as CSV"""
        metrics_file = self.path_config.metrics_path / "evaluation_metrics.csv"
        metrics_data = []
        
        # MainModel metrics (check both simple and optimized)
        mainmodel_key = 'mainmodel_optimized' if 'mainmodel_optimized' in self.results else 'mainmodel_simple'
        if mainmodel_key in self.results:
            mm_results = self.results[mainmodel_key]
            metrics_data.append({
                'experiment': 'mainmodel_optimized' if mainmodel_key == 'mainmodel_optimized' else 'mainmodel',
                'pathology': mm_results.get('pathology_name', 'unknown'),
                'accuracy': mm_results.get('accuracy', 0),
                'precision': mm_results.get('precision', 0),
                'recall': mm_results.get('recall', 0),
                'f1_score': mm_results.get('f1_score', 0),
                'train_time': mm_results.get('train_time', 0),
                'used_optimal_params': 'optimized_params' in mm_results
            })
        
        # Baseline metrics (top 5)
        if 'baselines' in self.results:
            baseline_results = self.results['baselines']
            # Sort by F1-macro and take top 5
            sorted_baselines = sorted(baseline_results.items(), 
                                    key=lambda x: x[1].get('f1_macro', 0), 
                                    reverse=True)[:5]
            
            for baseline_name, results in sorted_baselines:
                if 'f1_macro' in results and not isinstance(results['f1_macro'], str):
                    metrics_data.append({
                        'experiment': f'baseline_{baseline_name}',
                        'pathology': 'multi_label',
                        'accuracy': results.get('label_accuracy', 0),
                        'precision': 0,  # Not directly comparable
                        'recall': 0,     # Not directly comparable
                        'f1_score': results.get('f1_macro', 0),
                        'train_time': results.get('train_time', 0)
                    })
        
        # Save metrics CSV
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            df.to_csv(metrics_file, index=False)
            print(f"✅ Metrics saved: {metrics_file}")
    
    def _save_hypothesis_csv(self) -> None:
        """Save hypothesis summary as CSV"""
        if 'hypothesis_testing' not in self.results:
            return
        
        hypothesis_file = self.path_config.metrics_path / "hypothesis_summary.csv"
        hyp_data = []
        
        hyp_results = self.results['hypothesis_testing']
        for hyp_name, hyp_result in hyp_results.items():
            if hyp_name != 'overall' and 'score' in hyp_result:
                hyp_data.append({
                    'hypothesis': hyp_name,
                    'score': hyp_result.get('score', 0),
                    'status': hyp_result.get('status', 'UNKNOWN'),
                    'description': self._get_hypothesis_description(hyp_name)
                })
        
        if hyp_data:
            df_hyp = pd.DataFrame(hyp_data)
            df_hyp.to_csv(hypothesis_file, index=False)
            print(f"✅ Hypotheses saved: {hypothesis_file}")
    
    def _get_hypothesis_description(self, hyp_name: str) -> str:
        """Get human-readable hypothesis description"""
        descriptions = {
            'h1_performance': 'H1: Performance Gains vs Baselines',
            'h2_robustness': 'H2: Robustness under Missing Modalities',
            'h3_interpretability': 'H3: Interpretability and Modality Importance',
            'h4_scalability': 'H4: Computational Scalability',
            'h5_novel_features': 'H5: Novel Features Implementation'
        }
        return descriptions.get(hyp_name, hyp_name)
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of all experiments"""
        report = []
        report.append("🏥 ChestX-ray14 Experiment Summary Report")
        report.append("=" * 60)
        
        # Dataset info
        if hasattr(self, 'data_stats'):
            report.append(f"\n📊 Dataset Information:")
            report.append(f"   Mode: {'Small Sample' if self.exp_config.use_small_sample else 'Full Dataset'}")
            report.append(f"   Pathologies: {self.exp_config.n_pathologies}")
            report.append(f"   Random Seed: {self.exp_config.random_seed}")
        
        # MainModel results (check both simple and optimized)
        mainmodel_key = 'mainmodel_optimized' if 'mainmodel_optimized' in self.results else 'mainmodel_simple'
        if mainmodel_key in self.results:
            mm_results = self.results[mainmodel_key]
            is_optimized = mainmodel_key == 'mainmodel_optimized'
            report.append(f"\n🚀 MainModel Results ({('Optimized' if is_optimized else 'Default')} Parameters):")
            report.append(f"   Pathology: {mm_results.get('pathology_name', 'N/A')}")
            if is_optimized and 'optimized_params' in mm_results:
                report.append(f"   Optimal Params: {mm_results['optimized_params']}")
            report.append(f"   F1-Score: {mm_results.get('f1_score', 0):.3f}")
            report.append(f"   Accuracy: {mm_results.get('accuracy', 0):.3f}")
            report.append(f"   Training Time: {mm_results.get('train_time', 0):.1f}s")
        
        # Baseline results
        if 'baselines' in self.results:
            baseline_results = self.results['baselines']
            successful_baselines = [name for name, result in baseline_results.items() if 'f1_macro' in result]
            
            if successful_baselines:
                best_baseline = max(successful_baselines, 
                                  key=lambda x: baseline_results[x].get('f1_macro', 0))
                best_f1 = baseline_results[best_baseline]['f1_macro']
                
                report.append(f"\n📈 Baseline Results:")
                report.append(f"   Successful Baselines: {len(successful_baselines)}")
                report.append(f"   Best Baseline: {best_baseline}")
                report.append(f"   Best F1-macro: {best_f1:.3f}")
        
        # Hypothesis testing
        if 'hypothesis_testing' in self.results:
            hyp_results = self.results['hypothesis_testing']
            if 'overall' in hyp_results:
                overall = hyp_results['overall']
                report.append(f"\n🧪 Hypothesis Testing:")
                report.append(f"   Overall Score: {overall.get('overall_score', 0):.2f}/1.0")
                report.append(f"   Support Level: {overall.get('support_level', 'UNKNOWN')}")
                report.append(f"   Hypotheses Tested: {overall.get('hypotheses_tested', 0)}")
        
        return "\n".join(report)
    
    def print_summary(self) -> None:
        """Print experiment summary"""
        print(self.generate_summary_report())
