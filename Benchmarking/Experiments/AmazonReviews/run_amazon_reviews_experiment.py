#!/usr/bin/env python3
"""
AmazonReviews Experiment Pipeline
Main orchestration script for running the complete 8-phase experimentation pipeline.

Test Modes:
- Quick: Subset dataset, half hyperparameter trials, single seed
- Full: Full dataset, all hyperparameter trials, multi-seed
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

# Add project root to path (3 levels up to reach main project directory)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AmazonReviewsExperiment:
    """Main experiment orchestrator for AmazonReviews dataset."""
    
    def __init__(self, test_mode: str = "quick"):
        """
        Initialize the experiment.
        
        Args:
            test_mode: "quick" or "full"
        """
        self.test_mode = test_mode.lower()
        if self.test_mode not in ["quick", "full"]:
            raise ValueError("test_mode must be 'quick' or 'full'")
        
        # Experiment configuration based on mode
        self.config = self._get_experiment_config()
        
        # Create experiment directory
        self.experiment_dir = self._create_experiment_directory()
        
        # Save experiment configuration
        self._save_experiment_config()
        
        logger.info(f"Initialized AmazonReviews experiment in {self.test_mode} mode")
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def _get_experiment_config(self) -> Dict[str, Any]:
        """Get experiment configuration based on test mode."""
        
        if self.test_mode == "quick":
            config = {
                # Dataset configuration - Phase 1 will handle sampling
                "test_mode": "quick",  # Phase 1 will use 10k samples (8k train, 2k test)
                "hyperparameter_trials": 50,  # Quick mode trials
                "seeds": [42],  # Single seed
                
                # HIGH-IMPACT PARAMETERS ONLY (Option 1)
                
                # Core ensemble parameters (Most Important)
                "n_bags": [5, 10, 15],  # Number of ensemble bags
                "dropout_strategy": ["linear", "adaptive"],  # Modality dropout strategy
                "max_dropout_rate": [0.3, 0.5],  # Maximum dropout rate
                "sample_ratio": [0.7, 0.8],  # Sample ratio per bag
                "diversity_target": [0.6, 0.7],  # Diversity target
                
                # Training parameters (High Impact)
                "epochs": [20, 40],  # Training epochs
                "batch_size": [32, 64],  # Batch size
                "early_stopping_patience": [8, 12],  # Early stopping patience
                "cross_validation_folds": 5,  # CV folds
                
                # Optimization parameters (High Impact)
                "optimization_strategy": ["balanced", "accuracy"],  # Learner selection strategy
                "performance_threshold": [0.6, 0.7],  # Performance threshold
                
                # Training pipeline parameters (Medium Impact)
                "optimizer_type": ["adamw", "adam"],  # Optimizer type
                "dropout_rate": [0.2, 0.3],  # Dropout rate
                
                # Ensemble aggregation parameters (High Impact)
                "aggregation_strategy": ["weighted_vote", "confidence_weighted"],  # Aggregation strategy
                "uncertainty_method": ["entropy", "variance"],  # Uncertainty method
                
                # Data preprocessing parameters (Medium Impact)
                "normalize_data": [True, False],  # Normalize data
                "remove_outliers": [True, False],  # Remove outliers
                
                # Other parameters
                "ablation_trials": 2,  # Reduced ablation trials
                "robustness_tests": ["missing_modality", "noise"],  # Core tests only
                "interpretability_depth": "basic",  # Basic interpretability
                "production_tests": False,  # Skip production tests
            }
        else:  # full mode
            config = {
                # Dataset configuration - Phase 1 will handle sampling
                "test_mode": "full",  # Phase 1 will use 100k samples (80k train, 20k test)
                "hyperparameter_trials": 400,  # Increased trials for better coverage
                "seeds": [42, 123, 456, 789, 999],  # Multiple seeds
                
                # HIGH-IMPACT PARAMETERS ONLY (Option 1) - Extended for full mode
                
                # Core ensemble parameters (Most Important)
                "n_bags": [5, 10, 15, 20],  # Number of ensemble bags
                "dropout_strategy": ["linear", "exponential", "adaptive"],  # Key dropout strategies
                "max_dropout_rate": [0.2, 0.3, 0.4, 0.5],  # Dropout rate range
                "sample_ratio": [0.6, 0.7, 0.8, 0.9],  # Sample ratio per bag
                "diversity_target": [0.5, 0.6, 0.7, 0.8],  # Diversity target
                
                # Training parameters (High Impact)
                "epochs": [20, 30, 40, 50],  # Training epochs
                "batch_size": [16, 32, 64],  # Batch size
                "early_stopping_patience": [8, 10, 12],  # Early stopping patience
                "cross_validation_folds": 5,  # CV folds
                
                # Optimization parameters (High Impact)
                "optimization_strategy": ["balanced", "accuracy", "speed"],  # Key strategies
                "performance_threshold": [0.6, 0.7, 0.8],  # Performance threshold
                
                # Training pipeline parameters (Medium Impact)
                "optimizer_type": ["adamw", "adam", "sgd"],  # Optimizer types
                "scheduler_type": ["cosine_restarts", "step", "exponential"],  # Key schedulers
                "dropout_rate": [0.1, 0.2, 0.3, 0.4],  # Dropout rate
                "label_smoothing": [0.05, 0.1, 0.15],  # Label smoothing
                
                # Ensemble aggregation parameters (High Impact)
                "aggregation_strategy": ["weighted_vote", "confidence_weighted", "stacking"],  # Key strategies
                "uncertainty_method": ["entropy", "variance", "monte_carlo"],  # Uncertainty methods
                
                # Data preprocessing parameters (Medium Impact)
                "normalize_data": [True, False],  # Normalize data
                "remove_outliers": [True, False],  # Remove outliers
                "outlier_std": [2.0, 2.5, 3.0],  # Outlier standard deviation
                
                # Other parameters
                "ablation_trials": 5,  # Full ablation trials
                "robustness_tests": ["missing_modality", "noise", "adversarial", "distribution_shift"],  # Core tests
                "interpretability_depth": "comprehensive",  # Full interpretability
                "production_tests": True,  # Include production tests
            }
        
        # Common configuration
        config.update({
            "dataset_name": "AmazonReviews",
            "task_type": "classification",
            "modalities": ["text", "metadata"],
            "test_mode": self.test_mode,
            "timestamp": datetime.now().isoformat(),
        })
        
        return config
    
    def _create_experiment_directory(self) -> Path:
        """Create the experiment directory structure."""
        
        # Create main results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"amazon_reviews_{self.test_mode}_{timestamp}"
        experiment_dir = Path("results") / experiment_name
        
        # Create main directory
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create seed directories
        for seed in self.config["seeds"]:
            seed_dir = experiment_dir / f"seed_{seed}"
            seed_dir.mkdir(exist_ok=True)
            
            # Create phase directories - Currently only Phase 1 is implemented
            phases = [
                "phase_1_data_validation"
            ]
            
            for phase in phases:
                phase_dir = seed_dir / phase
                phase_dir.mkdir(exist_ok=True)
        
        # Create comprehensive report directory
        report_dir = experiment_dir / "comprehensive_report"
        report_dir.mkdir(exist_ok=True)
        
        logger.info(f"Created experiment directory structure: {experiment_dir}")
        return experiment_dir
    
    def _save_experiment_config(self):
        """Save experiment configuration to file."""
        config_file = self.experiment_dir / "experiment_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        logger.info(f"Saved experiment configuration: {config_file}")
    
    def run_phase(self, phase_name: str, seed: int, processed_data: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Run a single phase of the experiment.
        
        Args:
            phase_name: Name of the phase to run
            seed: Random seed for this run
            processed_data: Processed data from previous phase (optional)
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (success_status, phase_results)
        """
        logger.info(f"Starting {phase_name} for seed {seed}")
        
        # Create phase-specific configuration
        phase_config = {
            **self.config,
            "seed": seed,
            "phase_name": phase_name,
            "phase_dir": self.experiment_dir / f"seed_{seed}" / phase_name,
            "dataset_path": str(project_root / "Benchmarking" / "ProcessedData" / "AmazonReviews"),
        }
        
        try:
            # Import and run phase script
            phase_script = Path(__file__).parent / "phase_scripts" / f"{phase_name}.py"
            
            if not Path(phase_script).exists():
                logger.warning(f"Phase script not found: {phase_script}")
                logger.info(f"Creating placeholder for {phase_name}")
                self._create_phase_placeholder(phase_name, phase_config)
                return True, {"status": "placeholder"}
            
            # Execute phase script
            logger.info(f"Executing {phase_script}")
            
            # Import the phase module dynamically
            import importlib.util
            spec = importlib.util.spec_from_file_location(phase_name, phase_script)
            phase_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(phase_module)
            
            # Run the phase function with processed data if available
            phase_function_name = f"run_{phase_name}"
            if hasattr(phase_module, phase_function_name):
                phase_function = getattr(phase_module, phase_function_name)
                
                # Pass processed data to phases that support it
                if processed_data is not None and phase_name != "phase_1_data_validation":
                    phase_results = phase_function(phase_config, processed_data)
                else:
                    phase_results = phase_function(phase_config)
                
                logger.info(f"Phase {phase_name} completed with status: {phase_results.get('status', 'unknown')}")
                return phase_results.get('status') == 'completed', phase_results
            else:
                logger.warning(f"Phase function {phase_function_name} not found in {phase_script}")
                self._create_phase_placeholder(phase_name, phase_config)
                return True, {"status": "placeholder"}
            
        except Exception as e:
            logger.error(f"Error in {phase_name} for seed {seed}: {str(e)}")
            return False, {"status": "failed", "error": str(e)}
    
    def _create_phase_placeholder(self, phase_name: str, config: Dict[str, Any]):
        """Create placeholder output for a phase."""
        phase_dir = config["phase_dir"]
        
        # Create placeholder results file
        placeholder_results = {
            "phase": phase_name,
            "seed": config["seed"],
            "test_mode": config["test_mode"],
            "timestamp": datetime.now().isoformat(),
            "status": "placeholder",
            "message": f"Placeholder for {phase_name} - implementation pending"
        }
        
        results_file = phase_dir / f"{phase_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(placeholder_results, f, indent=2)
        
        logger.info(f"Created placeholder for {phase_name}: {results_file}")
    
    def run_experiment(self) -> bool:
        """
        Run the complete experiment pipeline.
        
        Returns:
            bool: True if all phases completed successfully, False otherwise
        """
        logger.info(f"Starting AmazonReviews experiment in {self.test_mode} mode")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2, default=str)}")
        
        start_time = time.time()
        success_count = 0
        total_phases = 0
        
        # Run experiment for each seed
        for seed in self.config["seeds"]:
            logger.info(f"Running experiment for seed {seed}")
            
            # Define phases to run - Currently Phase 1 and 2 are implemented
            phases = [
                "phase_1_data_validation",
                "phase_2_baseline"
            ]
            
            # Run phases sequentially with data passing
            processed_data = None  # Initialize processed data
            
            for phase in phases:
                total_phases += 1
                phase_success, phase_results = self.run_phase(phase, seed, processed_data)
                
                if phase_success:
                    success_count += 1
                    
                    # Extract processed data from Phase 1 for subsequent phases
                    if phase == "phase_1_data_validation" and phase_results.get("status") == "completed":
                        processed_data = phase_results.get("processed_data")
                        logger.info(f"Extracted processed data from Phase 1 for subsequent phases")
                    
                else:
                    logger.error(f"Phase {phase} failed for seed {seed}")
                    # Continue with next phase instead of stopping
        
        # Calculate experiment statistics
        end_time = time.time()
        duration = end_time - start_time
        success_rate = (success_count / total_phases) * 100 if total_phases > 0 else 0
        
        # Save experiment summary
        summary = {
            "experiment_name": self.experiment_dir.name,
            "test_mode": self.test_mode,
            "seeds": self.config["seeds"],
            "total_phases": total_phases,
            "successful_phases": success_count,
            "success_rate": success_rate,
            "duration_seconds": duration,
            "duration_hours": duration / 3600,
            "timestamp": datetime.now().isoformat(),
        }
        
        summary_file = self.experiment_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Experiment completed!")
        logger.info(f"Success rate: {success_rate:.1f}% ({success_count}/{total_phases} phases)")
        logger.info(f"Duration: {duration/3600:.2f} hours")
        logger.info(f"Results saved to: {self.experiment_dir}")
        
        return success_count == total_phases

def main():
    """Main entry point for the AmazonReviews experiment."""
    
    parser = argparse.ArgumentParser(description="AmazonReviews Experiment Pipeline")
    parser.add_argument(
        "--mode", 
        choices=["quick", "full"], 
        help="Test mode: quick (subset, half trials, single seed) or full (full dataset, all trials, multi-seed)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Interactive mode selection if not provided via command line
    test_mode = args.mode
    if test_mode is None:
        print("\n" + "="*60)
        print("🚀 AmazonReviews Experiment Pipeline")
        print("="*60)
        print("\nPlease select the test mode:")
        print("1. Quick Test - Fast testing with subset data")
        print("   • Uses 10,000 samples (8,000 train, 2,000 test)")
        print("   • Stratified sampling maintains original distribution")
        print("   • Single seed")
        print("   • Estimated time: 5-10 minutes (Phase 1 only)")
        print()
        print("2. Full Test - Complete evaluation")
        print("   • Uses 100,000 samples (80,000 train, 20,000 test)")
        print("   • Stratified sampling maintains original distribution")
        print("   • Multiple seeds")
        print("   • Estimated time: 15-30 minutes (Phase 1 only)")
        print()
        print("Note: Currently only Phase 1 (Data Validation) is implemented.")
        print("Phases 2-8 will be added in future updates.")
        print()
        
        while True:
            choice = input("Enter your choice (1 for Quick, 2 for Full): ").strip()
            if choice == "1":
                test_mode = "quick"
                break
            elif choice == "2":
                test_mode = "full"
                break
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
        
        print(f"\n✅ Selected: {test_mode.upper()} mode")
        print("Starting experiment...\n")
    
    try:
        # Create and run experiment
        experiment = AmazonReviewsExperiment(test_mode=test_mode)
        success = experiment.run_experiment()
        
        if success:
            logger.info("🎉 All phases completed successfully!")
            sys.exit(0)
        else:
            logger.warning("⚠️ Some phases failed. Check logs for details.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
