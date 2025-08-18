#!/usr/bin/env python3
"""
Amazon Reviews Experiment Launcher - Interactive Mode Selection
==            print("\n🚀 Running Quick Run Mode...")
            print("📊 Configuration: Small dataset, 1 run, 27 hyperparameter trials")===========================================================================

5-class rating prediction experiment using the Modality-Aware Adaptive Bagging Ensemble
on the Amazon Reviews dataset with text and metadata modalities.

Features:
- Quick Test Mode: 500 train/100 test samples for fast development
- Full Evaluation Mode: Complete dataset for publication-quality results  
- Multi-Seed Statistical Evaluation: 5-seed statistical validation
- Interactive mode selection with clear parameter display
"""

import os
import sys
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add MainModel to path
project_root = Path(__file__).parent.parent.parent.parent
mainmodel_path = project_root / "MainModel"
sys.path.append(str(mainmodel_path))
sys.path.append(str(project_root))

print(f"📍 MainModel path: {mainmodel_path} (exists: {mainmodel_path.exists()})")

# Import experiment components
try:
    from experiment_orchestrator import ExperimentOrchestrator
except ImportError:
    # Fallback for direct execution - add current directory to path
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Import with absolute imports
    import experiment_orchestrator
    ExperimentOrchestrator = experiment_orchestrator.ExperimentOrchestrator


def main():
    """Interactive main execution with mode selection"""
    print("\n🛒 Amazon Reviews Experiment Launcher")
    print("=" * 50)
    print("Select experiment mode:")
    print()
    print("1. 🚀 Quick Run Mode (FAST)")
    print("   • Small dataset subset (500 train, 100 test)")
    print("   • 1 test run only")
    print("   • All hyperparameters (27 trials)")
    print("   • Full pipeline execution")
    print("   • Perfect for development & debugging")
    print()
    print("2. 🎯 Full Run Mode (COMPREHENSIVE)")
    print("   • Entire dataset")
    print("   • 5 test runs (multi-seed)")
    print("   • Extended hyperparameters (52 trials)")
    print("   • Full pipeline execution")
    print("   • Publication-quality results")
    print()
    
    choice = input("Enter choice (1-2) [default: 1]: ").strip()
    
    try:
        if choice == "2":
            print("\n🎯 Running Full Run Mode...")
            print("📊 Configuration: Entire dataset, 5 seeds, 52 hyperparameter trials")
            
            # Full Run Mode: 5 test runs with different seeds
            seeds = [42, 123, 456, 789, 999]
            all_results = {}
            
            for i, seed in enumerate(seeds, 1):
                print(f"\n🌱 Test Run {i}/5 with seed {seed}...")
                config_overrides = {
                    'use_small_sample': False,  # Use entire dataset
                    'random_seed': seed,
                    'full_hyperparameter_search_trials': 52  # Extended hyperparameters
                }
                orchestrator = ExperimentOrchestrator(config_overrides)
                results = orchestrator.run_full_experiment()
                all_results[f'seed_{seed}'] = results
                
            print("\n✅ Full Run Mode completed successfully!")
            print(f"📊 Completed 5 test runs with {len(all_results)} result sets")
            
        else:
            print("\n� Running Quick Run Mode...")
            print("📊 Configuration: Small dataset, 1 run, 12 hyperparameter trials")
            
            # Quick Run Mode: 1 test run with all hyperparameters
            config_overrides = {
                'use_small_sample': True,  # Small dataset subset
                'random_seed': 42,
                'hyperparameter_search_trials': 27  # All hyperparameters
            }
            orchestrator = ExperimentOrchestrator(config_overrides)
            results = orchestrator.run_full_experiment()  # Still runs full pipeline
            
            print("\n✅ Quick Run Mode completed successfully!")
            print("📊 Completed 1 test run with 27 hyperparameter trials")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()


def quick_mode():
    """Run quick mode directly (1 test run, small dataset, all hyperparameters)"""
    print("🚀 Amazon Reviews Quick Run Mode")
    config_overrides = {
        'use_small_sample': True,
        'random_seed': 42,
        'hyperparameter_search_trials': 27
    }
    orchestrator = ExperimentOrchestrator(config_overrides)
    return orchestrator.run_full_experiment()


def full_mode():
    """Run full mode directly (5 test runs, entire dataset, extended hyperparameters)"""
    print("🎯 Amazon Reviews Full Run Mode")
    seeds = [42, 123, 456, 789, 999]
    all_results = {}
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n🌱 Test Run {i}/5 with seed {seed}...")
        config_overrides = {
            'use_small_sample': False,
            'random_seed': seed,
            'full_hyperparameter_search_trials': 52
        }
        orchestrator = ExperimentOrchestrator(config_overrides)
        results = orchestrator.run_full_experiment()
        all_results[f'seed_{seed}'] = results
    
    return all_results


if __name__ == "__main__":
    print("🛒 Amazon Reviews 5-Class Rating Prediction Framework")
    print("=" * 80)
    print("📊 Dataset: Amazon Product Reviews")
    print("🎯 Task: 5-class rating prediction (1★ to 5★)")
    print("🔬 Framework: Modality-Aware Adaptive Bagging Ensemble")
    print("🔧 Architecture: Modular, maintainable, and extensible")
    print("=" * 80)
    
    main()
