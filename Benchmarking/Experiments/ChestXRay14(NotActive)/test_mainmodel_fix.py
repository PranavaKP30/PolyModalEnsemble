#!/usr/bin/env python3
"""
Quick test script to verify MainModel multi-label classification fixes
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import directly to avoid package import issues
sys.path.append(str(Path(__file__).parent))

from config import ExperimentConfig, PathConfig
from data_loader import ChestXRayDataLoader
from mainmodel_evaluator import MainModelEvaluator

def main():
    print("🧪 Quick MainModel Multi-label Test")
    print("=" * 50)
    
    # Initialize configs
    config = ExperimentConfig()
    path_config = PathConfig(project_root=project_root)
    
    # Load data first
    print("📊 Loading data...")
    data_loader = ChestXRayDataLoader(config, path_config)
    data_loader.load_raw_data()
    data_loader.apply_sampling()
    
    # Create evaluator with data_loader
    evaluator = MainModelEvaluator(config, data_loader)
    
    print("📊 Running single simple test (no hyperparameter optimization)...")
    
    # Run only the simple test (much faster)
    try:
        results = evaluator.run_simple_test()
        
        if results:
            print("\n✅ SUCCESS: MainModel test completed!")
            print(f"📈 Results: {results}")
        else:
            print("\n❌ FAILED: MainModel test returned None")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
