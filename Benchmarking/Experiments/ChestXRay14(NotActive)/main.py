
#!/usr/bin/env python3
"""
Main execution script for ChestXRay14 experiments
Provides command-line interface and execution entry points
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from experiment_orchestrator import ExperimentOrchestrator

def main():
    """Main execution function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="ChestXRay14 14-Class Multi-Label Pathology Classification Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode full
  python main.py --mode quick
  python main.py --mode baseline
  python main.py --mode mainmodel
  python main.py --mode full --seed 123 --no-cv
  python main.py --mode mainmodel --hp-search
        """
    )

    parser.add_argument(
        '--mode',
        choices=['full', 'quick', 'baseline', 'mainmodel'],
        default='full',
        help='Experiment mode to run (default: full)'
    )
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--no-cv', action='store_true', help='Disable cross-validation')
    parser.add_argument('--hp-search', action='store_true', help='Enable hyperparameter search (for mainmodel mode)')
    parser.add_argument('--no-hp-search', action='store_true', help='Disable hyperparameter search')
    parser.add_argument('--output-dir', type=str, help='Custom output directory for results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--quiet', action='store_true', help='Suppress non-essential output')

    args = parser.parse_args()
    print_header()
    config_overrides = prepare_config_overrides(args)

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
        print_final_summary(results, orchestrator)
        print("\n🎉 ChestXRay14 experiment completed successfully!")
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
    print("=" * 80)
    print("🫁 CHESTXRAY14 14-CLASS MULTI-LABEL PATHOLOGY CLASSIFICATION EXPERIMENTS")
    print("=" * 80)
    print("📊 Comprehensive evaluation of multimodal machine learning models")
    print("🖼️ Image + Text + Metadata modalities for pathology prediction")
    print("🔬 Baseline models vs. MainModel ensemble comparison")
    print("=" * 80)

def prepare_config_overrides(args) -> Dict[str, Any]:
    overrides = {}
    if args.seed is not None:
        overrides['random_seed'] = args.seed
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

def run_full_experiment(orchestrator: ExperimentOrchestrator, args):
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

def run_quick_experiment(orchestrator: ExperimentOrchestrator, args):
    print("⚡ RUNNING QUICK EXPERIMENT")
    print("   ✅ Selected baseline models")
    print("   ✅ MainModel with default parameters")
    print("   ✅ Basic reporting")
    print("   ⏱️ Optimized for speed")
    print()
    return orchestrator.run_quick_experiment()

def run_baseline_experiment(orchestrator: ExperimentOrchestrator, args):
    print("📊 RUNNING BASELINE MODELS EXPERIMENT")
    print("   ✅ Comprehensive baseline evaluation")
    print("   ✅ 20+ models across 5 categories")
    print("   ✅ Single & multimodal approaches")
    print()
    return orchestrator.run_baseline_only()

def run_mainmodel_experiment(orchestrator: ExperimentOrchestrator, args):
    include_hp_search = args.hp_search or not args.no_hp_search
    print("🧠 RUNNING MAINMODEL EXPERIMENT")
    print("   ✅ MainModel ensemble evaluation")
    if include_hp_search:
        print("   ✅ Hyperparameter optimization")
    print("   ✅ Multimodal dropout strategy")
    print()
    return orchestrator.run_mainmodel_only(include_hyperparameter_search=include_hp_search)

def print_final_summary(results: Dict[str, Any], orchestrator: ExperimentOrchestrator):
    print("\n" + "=" * 80)
    print("📊 FINAL EXPERIMENT SUMMARY")
    print("=" * 80)
    metadata = results.get('experiment_metadata', {})
    total_time = metadata.get('total_time', 0)
    print(f"⏱️ Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"🎯 Random seed: {metadata.get('config', {}).get('random_seed', 'N/A')}")
    print(f"📊 Dataset: {metadata.get('config', {}).get('n_pathologies', 14)}-class ChestXRay14")
    print("\n🏆 PERFORMANCE HIGHLIGHTS")
    print("-" * 30)
    baseline_summary = results.get('baseline_results', {}).get('summary', {})
    if baseline_summary:
        best_model = baseline_summary.get('best_model', 'N/A')
        best_f1 = baseline_summary.get('best_f1', 0)
        print(f"🥇 Best baseline model: {best_model} (F1: {best_f1:.4f})")
    mainmodel_results = results.get('mainmodel_results', {})
    if 'optimized_test' in mainmodel_results and mainmodel_results['optimized_test']:
        opt_result = mainmodel_results['optimized_test']
        print(f"🧠 MainModel (Optimized): F1 {opt_result.get('f1_score', 0):.4f}")
    elif 'simple_test' in mainmodel_results and mainmodel_results['simple_test']:
        simple_result = mainmodel_results['simple_test']
        print(f"🧠 MainModel (Default): F1 {simple_result.get('f1_score', 0):.4f}")
    saved_files = results.get('saved_files', {})
    if saved_files:
        print(f"\n📁 Results saved to: {orchestrator.results_manager.run_dir}")
        print(f"🏷️ Experiment ID: {orchestrator.results_manager.timestamp}")
    pub_files = results.get('publication_files', {})
    if pub_files:
        print(f"📝 Publication materials: {len(pub_files)} files generated")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
