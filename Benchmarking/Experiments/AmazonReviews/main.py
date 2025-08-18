#!/usr/bin/env python3
"""
Main execution script for Amazon Reviews experiments
Provides command-line interface and execution entry points
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from .experiment_orchestrator import ExperimentOrchestrator


def main():
    """Main execution function with command-line interface"""
    
    parser = argparse.ArgumentParser(
        description="Amazon Reviews 5-Class Rating Prediction Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment
  python main.py --mode full
  
  # Run quick experiment for fast feedback
  python main.py --mode quick
  
  # Run only baseline models
  python main.py --mode baseline
  
  # Run only MainModel
  python main.py --mode mainmodel
  
  # Run full experiment with custom settings
  python main.py --mode full --seed 123 --train-size 5000 --no-cv
  
  # Run with custom hyperparameter search
  python main.py --mode mainmodel --hp-search
        """
    )
    
    # Execution mode
    parser.add_argument(
        '--mode', 
        choices=['full', 'quick', 'baseline', 'mainmodel'],
        default='full',
        help='Experiment mode to run (default: full)'
    )
    
    # Configuration overrides
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--train-size',
        type=int,
        help='Training sample size'
    )
    
    parser.add_argument(
        '--test-size',
        type=int,
        help='Test sample size'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        help='Number of cross-validation folds'
    )
    
    parser.add_argument(
        '--no-cv',
        action='store_true',
        help='Disable cross-validation'
    )
    
    parser.add_argument(
        '--hp-search',
        action='store_true',
        help='Enable hyperparameter search (for mainmodel mode)'
    )
    
    parser.add_argument(
        '--no-hp-search',
        action='store_true',
        help='Disable hyperparameter search'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Custom output directory for results'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress non-essential output'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Prepare configuration overrides
    config_overrides = prepare_config_overrides(args)
    
    # Create and run experiment
    try:
        orchestrator = ExperimentOrchestrator(config_overrides)
        
        if args.mode == 'full':
            results = run_full_experiment(orchestrator, args)
        elif args.mode == 'quick':
            results = run_quick_experiment(orchestrator, args)
        elif args.mode == 'baseline':
            results = run_baseline_experiment(orchestrator, args)
        elif args.mode == 'mainmodel':
            results = run_mainmodel_experiment(orchestrator, args)
        else:
            print(f"❌ Unknown mode: {args.mode}")
            sys.exit(1)
        
        # Print final summary
        print_final_summary(results, orchestrator)
        
        print("\n🎉 Amazon Reviews experiment completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def print_header():
    """Print experiment header"""
    print("=" * 80)
    print("🎯 AMAZON REVIEWS 5-CLASS RATING PREDICTION EXPERIMENTS")
    print("=" * 80)
    print("📊 Comprehensive evaluation of multimodal machine learning models")
    print("📝 Text + Metadata modalities for rating prediction (1⭐ to 5⭐)")
    print("🔬 Baseline models vs. MainModel ensemble comparison")
    print("=" * 80)


def prepare_config_overrides(args) -> Dict[str, Any]:
    """Prepare configuration overrides from command-line arguments"""
    
    overrides = {}
    
    if args.seed is not None:
        overrides['random_seed'] = args.seed
    
    if args.train_size is not None:
        overrides['train_sample_size'] = args.train_size
    
    if args.test_size is not None:
        overrides['test_sample_size'] = args.test_size
    
    if args.cv_folds is not None:
        overrides['cv_folds'] = args.cv_folds
    
    if args.no_cv:
        overrides['enable_cv'] = False
    
    if args.output_dir:
        overrides['results_dir'] = args.output_dir
    
    if overrides:
        print("📝 Configuration overrides:")
        for key, value in overrides.items():
            print(f"   {key}: {value}")
        print()
    
    return overrides


def run_full_experiment(orchestrator: ExperimentOrchestrator, args) -> Dict[str, Any]:
    """Run full comprehensive experiment"""
    
    print("🚀 RUNNING FULL COMPREHENSIVE EXPERIMENT")
    print("   ✅ Baseline model evaluation")
    print("   ✅ MainModel evaluation with hyperparameter optimization")
    print("   ✅ Advanced analysis (CV, robustness, ablation)")
    print("   ✅ Comprehensive reporting and publication export")
    print()
    
    include_hp_search = not args.no_hp_search
    
    return orchestrator.run_full_experiment(
        include_baseline=True,
        include_mainmodel=True,
        include_advanced=True,
        include_hyperparameter_search=include_hp_search
    )


def run_quick_experiment(orchestrator: ExperimentOrchestrator, args) -> Dict[str, Any]:
    """Run quick experiment for fast feedback"""
    
    print("⚡ RUNNING QUICK EXPERIMENT")
    print("   ✅ Selected baseline models")
    print("   ✅ MainModel with default parameters")
    print("   ✅ Basic reporting")
    print("   ⏱️ Optimized for speed")
    print()
    
    return orchestrator.run_quick_experiment()


def run_baseline_experiment(orchestrator: ExperimentOrchestrator, args) -> Dict[str, Any]:
    """Run baseline models only"""
    
    print("📊 RUNNING BASELINE MODELS EXPERIMENT")
    print("   ✅ Comprehensive baseline evaluation")
    print("   ✅ 30+ models across 5 categories")
    print("   ✅ Single & multimodal approaches")
    print()
    
    return orchestrator.run_baseline_only()


def run_mainmodel_experiment(orchestrator: ExperimentOrchestrator, args) -> Dict[str, Any]:
    """Run MainModel only"""
    
    include_hp_search = args.hp_search or not args.no_hp_search
    
    print("🧠 RUNNING MAINMODEL EXPERIMENT")
    print("   ✅ MainModel ensemble evaluation")
    if include_hp_search:
        print("   ✅ Hyperparameter optimization")
    print("   ✅ Multimodal dropout strategy")
    print()
    
    return orchestrator.run_mainmodel_only(include_hyperparameter_search=include_hp_search)


def print_final_summary(results: Dict[str, Any], orchestrator: ExperimentOrchestrator):
    """Print final experiment summary"""
    
    print("\n" + "=" * 80)
    print("📊 FINAL EXPERIMENT SUMMARY")
    print("=" * 80)
    
    # Experiment metadata
    metadata = results.get('experiment_metadata', {})
    total_time = metadata.get('total_time', 0)
    
    print(f"⏱️ Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"🎯 Random seed: {metadata.get('config', {}).get('random_seed', 'N/A')}")
    print(f"📊 Dataset: {metadata.get('config', {}).get('n_classes', 5)}-class Amazon Reviews")
    
    # Performance highlights
    print("\n🏆 PERFORMANCE HIGHLIGHTS")
    print("-" * 30)
    
    # Best baseline
    baseline_summary = results.get('baseline_results', {}).get('summary', {})
    if baseline_summary:
        best_model = baseline_summary.get('best_model', 'N/A')
        best_acc = baseline_summary.get('best_accuracy', 0)
        fastest_model = baseline_summary.get('fastest_model', 'N/A')
        
        print(f"🥇 Best baseline model: {best_model} (Accuracy: {best_acc:.4f})")
        print(f"⚡ Fastest baseline: {fastest_model}")
    
    # MainModel performance
    mainmodel_results = results.get('mainmodel_results', {})
    if 'optimized_test' in mainmodel_results and mainmodel_results['optimized_test']:
        opt_result = mainmodel_results['optimized_test']
        print(f"🧠 MainModel (Optimized): Accuracy {opt_result.get('accuracy', 0):.4f}, "
              f"Star MAE {opt_result.get('star_mae', 0):.4f}")
    elif 'simple_test' in mainmodel_results and mainmodel_results['simple_test']:
        simple_result = mainmodel_results['simple_test']
        print(f"🧠 MainModel (Default): Accuracy {simple_result.get('accuracy', 0):.4f}, "
              f"Star MAE {simple_result.get('star_mae', 0):.4f}")
    
    # Advanced analysis highlights
    if 'ablation_results' in results:
        ablation = results['ablation_results'].get('modality_contributions', {})
        if ablation:
            multimodal_benefit = ablation.get('multimodal_benefit', 0)
            best_single = ablation.get('best_single_modality', 'N/A')
            
            print(f"🎭 Multimodal benefit: {multimodal_benefit:.4f}")
            print(f"📈 Best single modality: {best_single}")
    
    # Results location
    saved_files = results.get('saved_files', {})
    if saved_files:
        print(f"\n📁 Results saved to: {orchestrator.results_manager.run_dir}")
        print(f"🏷️ Experiment ID: {orchestrator.results_manager.timestamp}")
    
    # Publication materials
    pub_files = results.get('publication_files', {})
    if pub_files:
        print(f"📝 Publication materials: {len(pub_files)} files generated")
    
    print("\n" + "=" * 80)


def run_interactive_experiment():
    """Run experiment in interactive mode"""
    
    print_header()
    
    print("🎯 INTERACTIVE EXPERIMENT MODE")
    print("=" * 40)
    
    # Get user preferences
    mode = input("Select experiment mode (full/quick/baseline/mainmodel) [full]: ").strip() or "full"
    
    if mode not in ['full', 'quick', 'baseline', 'mainmodel']:
        print(f"❌ Invalid mode: {mode}")
        return
    
    # Custom seed
    seed_input = input("Random seed (press Enter for default): ").strip()
    seed = int(seed_input) if seed_input else None
    
    # Prepare config
    config_overrides = {}
    if seed:
        config_overrides['random_seed'] = seed
    
    # Run experiment
    orchestrator = ExperimentOrchestrator(config_overrides)
    
    if mode == 'full':
        results = orchestrator.run_full_experiment()
    elif mode == 'quick':
        results = orchestrator.run_quick_experiment()
    elif mode == 'baseline':
        results = orchestrator.run_baseline_only()
    elif mode == 'mainmodel':
        results = orchestrator.run_mainmodel_only()
    
    print_final_summary(results, orchestrator)


def compare_experiments():
    """Compare results from multiple experiments"""
    
    print("🔍 EXPERIMENT COMPARISON MODE")
    print("=" * 40)
    
    from .results_manager import ResultsManager
    from .config import ExperimentConfig
    
    config = ExperimentConfig()
    results_manager = ResultsManager(config)
    
    # List available experiments
    experiments = results_manager.list_experiments()
    
    if not experiments:
        print("❌ No experiments found to compare")
        return
    
    print("📊 Available experiments:")
    for i, exp in enumerate(experiments):
        print(f"  {i+1}. {exp['experiment_id']} - {exp.get('summary', {})}")
    
    # Get user selection
    try:
        selection = input("\nEnter experiment numbers to compare (comma-separated): ").strip()
        indices = [int(x.strip()) - 1 for x in selection.split(',')]
        
        selected_experiments = [experiments[i]['experiment_id'] for i in indices if 0 <= i < len(experiments)]
        
        if len(selected_experiments) < 2:
            print("❌ Please select at least 2 experiments to compare")
            return
        
        # Run comparison
        comparison = results_manager.compare_experiments(selected_experiments)
        
        # Print comparison results
        print("\n📊 COMPARISON RESULTS")
        print("=" * 30)
        
        # Model performance comparison
        model_comparison = comparison.get('model_comparison', {})
        for model_name, performances in model_comparison.items():
            print(f"\n{model_name}:")
            for exp_id, accuracy in performances.items():
                print(f"  {exp_id}: {accuracy:.4f}")
        
        # Best results
        best_results = comparison.get('best_results', {})
        if best_results:
            print(f"\n🏆 Best overall accuracy: {best_results['best_overall_accuracy']}")
            print(f"⚡ Fastest training: {best_results['fastest_training']}")
        
    except (ValueError, IndexError):
        print("❌ Invalid selection")


if __name__ == "__main__":
    # Check if running interactively
    if len(sys.argv) == 1:
        # No arguments provided, run interactive mode
        try:
            choice = input("Select mode:\n1. Run experiment\n2. Compare experiments\nChoice [1]: ").strip() or "1"
            
            if choice == "1":
                run_interactive_experiment()
            elif choice == "2":
                compare_experiments()
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
    else:
        # Command-line arguments provided
        main()
