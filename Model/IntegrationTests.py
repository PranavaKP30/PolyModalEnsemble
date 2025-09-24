#!/usr/bin/env python3
"""
Integration Tests for PolyModal Ensemble Learning Pipeline

This module provides comprehensive integration tests to validate:
- Data flow between stages
- Cross-stage compatibility
- Error handling and recovery
- Performance and robustness
"""

import os
import sys
import logging
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
import unittest
from unittest.mock import patch, MagicMock

# Add the Model directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the pipeline components
from ModelAPI import ModelAPI
from DataIntegration import SimpleDataLoader
from BagGeneration import BagGeneration
from BagLearnerParing import BagLearnerParing
from BagTraining import BagTraining
from ErrorHandler import ErrorHandler, PipelineError

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDataFlow(unittest.TestCase):
    """Test data flow between stages."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "test_data"
        self.test_data_dir.mkdir()
        
        # Create mock data
        self._create_mock_data()
        
        # Initialize API
        self.api = ModelAPI(
            cache_dir=self.temp_dir,
            device='cpu',
            fast_mode=True,
            max_samples=100,
            n_bags=5,
            dropout_strategy='adaptive',
            configuration_method='predefined',
            output_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_mock_data(self):
        """Create mock data for testing."""
        # Create mock labels
        labels = np.random.randint(0, 3, 100)
        np.savetxt(self.test_data_dir / "labels.csv", labels, delimiter=',', fmt='%d')
        
        # Create mock image data (simulate EuroSAT)
        images_dir = self.test_data_dir / "images"
        images_dir.mkdir()
        
        # Create mock spectral data
        spectral_data = np.random.randn(100, 64)
        np.save(self.test_data_dir / "spectral.npy", spectral_data)
        
        # Create mock tabular data
        tabular_data = np.random.randn(100, 10)
        np.savetxt(self.test_data_dir / "tabular.csv", tabular_data, delimiter=',')
    
    def test_stage1_to_stage2_data_flow(self):
        """Test data flow from Stage 1 to Stage 2."""
        logger.info("Testing Stage 1 to Stage 2 data flow...")
        
        # Stage 1: Load data
        self.api.load_custom_data(
            label_file=str(self.test_data_dir / "labels.csv"),
            modality_files={
                'spectral': str(self.test_data_dir / "spectral.npy"),
                'tabular': str(self.test_data_dir / "tabular.csv")
            },
            modality_types={
                'spectral': 'spectral',
                'tabular': 'tabular'
            }
        )
        
        # Verify Stage 1 output
        train_data, train_labels = self.api.get_train_data()
        test_data, test_labels = self.api.get_test_data()
        
        self.assertIsInstance(train_data, dict)
        self.assertIsInstance(train_labels, np.ndarray)
        self.assertGreater(len(train_data), 0)
        self.assertGreater(len(train_labels), 0)
        
        # Stage 2: Generate bags
        self.api.generate_bags()
        
        # Verify Stage 2 output
        bags = self.api.get_bags()
        self.assertGreater(len(bags), 0)
        
        # Verify bag data structure
        for bag in bags:
            self.assertIn('bag_id', bag.__dict__)
            self.assertIn('modality_mask', bag.__dict__)
            self.assertIn('data_indices', bag.__dict__)
        
        logger.info("✅ Stage 1 to Stage 2 data flow test passed")
    
    def test_stage2_to_stage3_data_flow(self):
        """Test data flow from Stage 2 to Stage 3."""
        logger.info("Testing Stage 2 to Stage 3 data flow...")
        
        # Run Stage 1 and 2
        self.api.load_custom_data(
            label_file=str(self.test_data_dir / "labels.csv"),
            modality_files={
                'spectral': str(self.test_data_dir / "spectral.npy"),
                'tabular': str(self.test_data_dir / "tabular.csv")
            },
            modality_types={
                'spectral': 'spectral',
                'tabular': 'tabular'
            }
        )
        self.api.generate_bags()
        
        # Stage 3: Select learners
        self.api.select_learners()
        
        # Verify Stage 3 output
        learner_configs = self.api.get_learner_configs()
        self.assertGreater(len(learner_configs), 0)
        
        # Verify learner configuration structure
        for bag_id, config in learner_configs.items():
            self.assertIn('learner_type', config)
            self.assertIn('hyperparameters', config)
            self.assertIsInstance(config['hyperparameters'], dict)
        
        logger.info("✅ Stage 2 to Stage 3 data flow test passed")
    
    def test_stage3_to_stage4_data_flow(self):
        """Test data flow from Stage 3 to Stage 4."""
        logger.info("Testing Stage 3 to Stage 4 data flow...")
        
        # Run Stages 1, 2, and 3
        self.api.load_custom_data(
            label_file=str(self.test_data_dir / "labels.csv"),
            modality_files={
                'spectral': str(self.test_data_dir / "spectral.npy"),
                'tabular': str(self.test_data_dir / "tabular.csv")
            },
            modality_types={
                'spectral': 'spectral',
                'tabular': 'tabular'
            }
        )
        self.api.generate_bags()
        self.api.select_learners()
        
        # Stage 4: Train models
        self.api.train_bags()
        
        # Verify Stage 4 output
        trained_models = self.api.get_trained_models()
        self.assertGreater(len(trained_models), 0)
        
        # Verify trained model structure
        for bag_id, model in trained_models.items():
            self.assertIn('model', model.__dict__)
            self.assertIn('training_metrics', model.__dict__)
            self.assertIn('training_config', model.__dict__)
        
        logger.info("✅ Stage 3 to Stage 4 data flow test passed")

class TestErrorHandling(unittest.TestCase):
    """Test error handling and recovery mechanisms."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.error_handler = ErrorHandler(
            log_file=str(Path(self.temp_dir) / "errors.log"),
            enable_rollback=True
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_error_classification(self):
        """Test error classification system."""
        logger.info("Testing error classification...")
        
        # Test different error types
        test_errors = [
            (FileNotFoundError("test.txt"), "file_io", "medium"),
            (MemoryError("Out of memory"), "memory", "critical"),
            (ValueError("Invalid shape"), "data_validation", "high"),
            (TypeError("Invalid type"), "unknown", "low")
        ]
        
        for error, expected_category, expected_severity in test_errors:
            error_info = self.error_handler.handle_error(error, "test_stage")
            
            self.assertEqual(error_info.category.value, expected_category)
            self.assertEqual(error_info.severity.value, expected_severity)
        
        logger.info("✅ Error classification test passed")
    
    def test_rollback_mechanism(self):
        """Test rollback mechanism."""
        logger.info("Testing rollback mechanism...")
        
        # Register rollback actions
        rollback_called = []
        
        def rollback_action1():
            rollback_called.append("action1")
        
        def rollback_action2():
            rollback_called.append("action2")
        
        self.error_handler.register_rollback_action(rollback_action1, "Test action 1")
        self.error_handler.register_rollback_action(rollback_action2, "Test action 2")
        
        # Simulate critical error
        critical_error = MemoryError("Critical memory error")
        error_info = self.error_handler.handle_error(critical_error, "test_stage")
        
        # Verify rollback was executed
        self.assertEqual(len(rollback_called), 2)
        self.assertIn("action1", rollback_called)
        self.assertIn("action2", rollback_called)
        
        logger.info("✅ Rollback mechanism test passed")
    
    def test_pipeline_health_check(self):
        """Test pipeline health monitoring."""
        logger.info("Testing pipeline health check...")
        
        # Initially healthy
        self.assertTrue(self.error_handler.is_pipeline_healthy())
        
        # Add low severity errors (should still be healthy)
        for _ in range(5):
            self.error_handler.handle_error(ValueError("Minor error"), "test_stage")
        
        self.assertTrue(self.error_handler.is_pipeline_healthy())
        
        # Add critical error (should be unhealthy)
        self.error_handler.handle_error(MemoryError("Critical error"), "test_stage")
        self.assertFalse(self.error_handler.is_pipeline_healthy())
        
        logger.info("✅ Pipeline health check test passed")

class TestDataConsistency(unittest.TestCase):
    """Test data consistency across stages."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "test_data"
        self.test_data_dir.mkdir()
        
        # Create consistent mock data
        self._create_consistent_mock_data()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_consistent_mock_data(self):
        """Create consistent mock data."""
        n_samples = 200
        
        # Create consistent labels
        labels = np.random.randint(0, 3, n_samples)
        np.savetxt(self.test_data_dir / "labels.csv", labels, delimiter=',', fmt='%d')
        
        # Create consistent modality data
        spectral_data = np.random.randn(n_samples, 64)
        tabular_data = np.random.randn(n_samples, 10)
        
        np.save(self.test_data_dir / "spectral.npy", spectral_data)
        np.savetxt(self.test_data_dir / "tabular.csv", tabular_data, delimiter=',')
    
    def test_sample_count_consistency(self):
        """Test that sample counts remain consistent across stages."""
        logger.info("Testing sample count consistency...")
        
        api = ModelAPI(
            cache_dir=self.temp_dir,
            device='cpu',
            fast_mode=True,
            max_samples=200,
            n_bags=3,
            dropout_strategy='adaptive',
            configuration_method='predefined',
            output_dir=self.temp_dir
        )
        
        # Stage 1: Load data
        api.load_custom_data(
            label_file=str(self.test_data_dir / "labels.csv"),
            modality_files={
                'spectral': str(self.test_data_dir / "spectral.npy"),
                'tabular': str(self.test_data_dir / "tabular.csv")
            },
            modality_types={
                'spectral': 'spectral',
                'tabular': 'tabular'
            }
        )
        
        train_data, train_labels = api.get_train_data()
        test_data, test_labels = api.get_test_data()
        
        # Verify initial consistency
        train_sample_count = len(train_labels)
        test_sample_count = len(test_labels)
        
        for modality_data in train_data.values():
            self.assertEqual(len(modality_data), train_sample_count)
        
        for modality_data in test_data.values():
            self.assertEqual(len(modality_data), test_sample_count)
        
        # Stage 2: Generate bags
        api.generate_bags()
        bags = api.get_bags()
        
        # Verify bag consistency
        for bag in bags:
            bag_data = api.bag_generator.bag_data[bag.bag_id]
            bag_sample_count = len(bag_data['labels'])
            
            for modality_data in bag_data['train_data'].values():
                self.assertEqual(len(modality_data), bag_sample_count)
        
        logger.info("✅ Sample count consistency test passed")
    
    def test_modality_mask_consistency(self):
        """Test that modality masks are consistent with actual data."""
        logger.info("Testing modality mask consistency...")
        
        api = ModelAPI(
            cache_dir=self.temp_dir,
            device='cpu',
            fast_mode=True,
            max_samples=200,
            n_bags=5,
            dropout_strategy='adaptive',
            configuration_method='predefined',
            output_dir=self.temp_dir
        )
        
        # Run pipeline
        api.load_custom_data(
            label_file=str(self.test_data_dir / "labels.csv"),
            modality_files={
                'spectral': str(self.test_data_dir / "spectral.npy"),
                'tabular': str(self.test_data_dir / "tabular.csv")
            },
            modality_types={
                'spectral': 'spectral',
                'tabular': 'tabular'
            }
        )
        api.generate_bags()
        
        bags = api.get_bags()
        
        # Verify modality mask consistency
        for bag in bags:
            modality_mask = bag.modality_mask
            bag_data = api.bag_generator.bag_data[bag.bag_id]
            
            # Check that active modalities have data
            for modality_name, is_active in modality_mask.items():
                if is_active:
                    self.assertIn(modality_name, bag_data['train_data'])
                else:
                    self.assertNotIn(modality_name, bag_data['train_data'])
        
        logger.info("✅ Modality mask consistency test passed")

class TestPerformance(unittest.TestCase):
    """Test performance and scalability."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "test_data"
        self.test_data_dir.mkdir()
        
        # Create larger mock data for performance testing
        self._create_performance_test_data()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_performance_test_data(self):
        """Create data for performance testing."""
        n_samples = 1000  # Larger dataset
        
        # Create labels
        labels = np.random.randint(0, 5, n_samples)
        np.savetxt(self.test_data_dir / "labels.csv", labels, delimiter=',', fmt='%d')
        
        # Create modality data
        spectral_data = np.random.randn(n_samples, 128)
        tabular_data = np.random.randn(n_samples, 20)
        
        np.save(self.test_data_dir / "spectral.npy", spectral_data)
        np.savetxt(self.test_data_dir / "tabular.csv", tabular_data, delimiter=',')
    
    def test_stage1_performance(self):
        """Test Stage 1 performance with larger dataset."""
        logger.info("Testing Stage 1 performance...")
        
        import time
        
        api = ModelAPI(
            cache_dir=self.temp_dir,
            device='cpu',
            fast_mode=True,
            max_samples=1000,
            n_bags=10,
            dropout_strategy='adaptive',
            configuration_method='predefined',
            output_dir=self.temp_dir
        )
        
        start_time = time.time()
        
        api.load_custom_data(
            label_file=str(self.test_data_dir / "labels.csv"),
            modality_files={
                'spectral': str(self.test_data_dir / "spectral.npy"),
                'tabular': str(self.test_data_dir / "tabular.csv")
            },
            modality_types={
                'spectral': 'spectral',
                'tabular': 'tabular'
            }
        )
        
        end_time = time.time()
        load_time = end_time - start_time
        
        # Verify performance (should complete within reasonable time)
        self.assertLess(load_time, 30.0)  # 30 seconds max
        
        # Verify data integrity
        train_data, train_labels = api.get_train_data()
        self.assertGreater(len(train_labels), 0)
        
        logger.info(f"✅ Stage 1 performance test passed (load time: {load_time:.2f}s)")
    
    def test_memory_usage(self):
        """Test memory usage during pipeline execution."""
        logger.info("Testing memory usage...")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        api = ModelAPI(
            cache_dir=self.temp_dir,
            device='cpu',
            fast_mode=True,
            max_samples=1000,
            n_bags=10,
            dropout_strategy='adaptive',
            configuration_method='predefined',
            output_dir=self.temp_dir
        )
        
        # Run pipeline
        api.load_custom_data(
            label_file=str(self.test_data_dir / "labels.csv"),
            modality_files={
                'spectral': str(self.test_data_dir / "spectral.npy"),
                'tabular': str(self.test_data_dir / "tabular.csv")
            },
            modality_types={
                'spectral': 'spectral',
                'tabular': 'tabular'
            }
        )
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Verify reasonable memory usage (less than 1GB increase)
        self.assertLess(memory_increase, 1024)  # 1GB max increase
        
        logger.info(f"✅ Memory usage test passed (memory increase: {memory_increase:.2f}MB)")

class TestRobustness(unittest.TestCase):
    """Test robustness under various conditions."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "test_data"
        self.test_data_dir.mkdir()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        logger.info("Testing missing data handling...")
        
        # Create data with missing values
        n_samples = 100
        labels = np.random.randint(0, 3, n_samples)
        
        # Create data with NaN values
        spectral_data = np.random.randn(n_samples, 64)
        spectral_data[10:20, :] = np.nan  # Introduce NaN values
        
        tabular_data = np.random.randn(n_samples, 10)
        tabular_data[30:40, :] = np.nan  # Introduce NaN values
        
        # Save data
        np.savetxt(self.test_data_dir / "labels.csv", labels, delimiter=',', fmt='%d')
        np.save(self.test_data_dir / "spectral.npy", spectral_data)
        np.savetxt(self.test_data_dir / "tabular.csv", tabular_data, delimiter=',')
        
        api = ModelAPI(
            cache_dir=self.temp_dir,
            device='cpu',
            fast_mode=True,
            max_samples=100,
            n_bags=3,
            dropout_strategy='adaptive',
            configuration_method='predefined',
            output_dir=self.temp_dir
        )
        
        # Should handle missing data gracefully
        try:
            api.load_custom_data(
                label_file=str(self.test_data_dir / "labels.csv"),
                modality_files={
                    'spectral': str(self.test_data_dir / "spectral.npy"),
                    'tabular': str(self.test_data_dir / "tabular.csv")
                },
                modality_types={
                    'spectral': 'spectral',
                    'tabular': 'tabular'
                }
            )
            
            # Should complete without crashing
            train_data, train_labels = api.get_train_data()
            self.assertGreater(len(train_labels), 0)
            
        except Exception as e:
            # If it fails, it should be a recoverable error
            self.assertIsInstance(e, (ValueError, Warning))
        
        logger.info("✅ Missing data handling test passed")
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        logger.info("Testing invalid parameters handling...")
        
        # Test with invalid parameters
        with self.assertRaises((ValueError, TypeError)):
            api = ModelAPI(
                cache_dir=self.temp_dir,
                device='cpu',
                fast_mode=True,
                max_samples=-1,  # Invalid negative value
                n_bags=0,  # Invalid zero value
                dropout_strategy='invalid_strategy',  # Invalid strategy
                configuration_method='invalid_method',  # Invalid method
                output_dir=self.temp_dir
            )
        
        logger.info("✅ Invalid parameters handling test passed")

def run_integration_tests():
    """Run all integration tests."""
    logger.info("Starting PolyModal Ensemble Integration Tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDataFlow,
        TestErrorHandling,
        TestDataConsistency,
        TestPerformance,
        TestRobustness
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Report results
    logger.info(f"Integration Tests Results:")
    logger.info(f"  Tests run: {result.testsRun}")
    logger.info(f"  Failures: {len(result.failures)}")
    logger.info(f"  Errors: {len(result.errors)}")
    
    if result.failures:
        logger.error("Test Failures:")
        for test, traceback in result.failures:
            logger.error(f"  {test}: {traceback}")
    
    if result.errors:
        logger.error("Test Errors:")
        for test, traceback in result.errors:
            logger.error(f"  {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    logger.info(f"Integration Tests {'PASSED' if success else 'FAILED'}")
    
    return success

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
