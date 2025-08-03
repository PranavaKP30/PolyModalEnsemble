"""
test_dataIntegration.py
Unit and integration tests for Stage 1: Data Integration (dataIntegration.py)
Covers: loading, validation, cleaning, export, and builder utilities.
"""

import sys
import os
import numpy as np
import pytest
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from MainModel import dataIntegration as di

def test_modality_config_validation():
    # Valid config
    c = di.ModalityConfig(name="text", data_type="text", feature_dim=768, is_required=True, priority=1.0, min_feature_ratio=0.2, max_feature_ratio=0.8)
    assert c.name == "text"
    # Invalid config
    with pytest.raises(AssertionError):
        di.ModalityConfig(name="bad", data_type="tabular", feature_dim=-1)
    with pytest.raises(AssertionError):
        di.ModalityConfig(name="bad", data_type="tabular", min_feature_ratio=1.1)
    with pytest.raises(AssertionError):
        di.ModalityConfig(name="bad", data_type="tabular", min_feature_ratio=0.8, max_feature_ratio=0.5)

def test_generic_loader_add_and_validate():
    loader = di.GenericMultiModalDataLoader(validate_data=True)
    arr1 = np.random.randn(100, 10)
    arr2 = np.random.randn(100, 20)
    loader.add_modality("mod1", arr1, "tabular", is_required=True)
    loader.add_modality("mod2", arr2, "image", is_required=False)
    loader.add_labels(np.arange(100))
    report = loader.get_data_quality_report()
    assert report["overall"]["total_samples"] == 100
    assert len(report["modalities"]) == 2
    assert "mod1" in report["modalities"]
    assert "mod2" in report["modalities"]
    # Test summary
    loader.summary()

def test_clean_data_and_export():
    loader = di.GenericMultiModalDataLoader(validate_data=True)
    arr = np.random.randn(50, 5)
    arr[0, 0] = np.nan
    arr[1, 1] = np.inf
    loader.add_modality("mod", arr, "tabular")
    loader.clean_data(handle_nan='fill_mean', handle_inf='fill_zero')
    assert not np.isnan(loader.data["mod"]).any()
    assert not np.isinf(loader.data["mod"]).any()
    data, configs, meta = loader.export_for_ensemble_generation()
    assert "mod" in data
    assert isinstance(configs, list)
    assert isinstance(meta, dict)

def test_quick_dataset_builder_from_arrays():
    arrs = {"a": np.random.randn(10, 3), "b": np.random.randn(10, 2)}
    labels = np.arange(10)
    loader = di.QuickDatasetBuilder.from_arrays(arrs, modality_types={"a": "tabular", "b": "image"}, labels=labels, required_modalities=["a"])
    assert "a" in loader.data and "b" in loader.data
    assert "labels" in loader.data
    assert any(c.is_required for c in loader.modality_configs)

def test_create_synthetic_dataset():
    specs = {"x": (4, "tabular"), "y": (2, "image")}
    loader = di.create_synthetic_dataset(specs, n_samples=20, n_classes=3, noise_level=0.2, missing_data_rate=0.1, random_state=42)
    assert set(loader.data.keys()) >= {"x", "y", "labels"}
    assert loader.data["x"].shape == (20, 4)
    assert loader.data["y"].shape == (20, 2)
    assert loader.data["labels"].shape == (20,)
    # Check missing data
    assert np.isnan(loader.data["x"]).sum() > 0 or np.isnan(loader.data["y"]).sum() > 0

def test_auto_preprocess_dataset():
    specs = {"x": (4, "tabular"), "y": (2, "image")}
    loader = di.create_synthetic_dataset(specs, n_samples=10, n_classes=2, noise_level=0.1, missing_data_rate=0.2, random_state=1)
    pre = di.auto_preprocess_dataset(loader, normalize=True, handle_missing='mean', remove_outliers=True, outlier_std=2.5)
    for k, arr in pre.data.items():
        if arr.ndim == 2:
            assert not np.isnan(arr).any()
    # Labels copied
    assert any('label' in k.lower() for k in pre.data)

def test_directory_and_file_builders(tmp_path):
    # Create dummy files
    arr = np.random.randn(5, 2)
    np.save(tmp_path / "mod1.npy", arr)
    np.savetxt(tmp_path / "mod2.csv", arr, delimiter=",")
    files = {"mod1": str(tmp_path / "mod1.npy"), "mod2": str(tmp_path / "mod2.csv")}
    loader = di.QuickDatasetBuilder.from_files(files, modality_types={"mod1": "tabular", "mod2": "tabular"})
    assert "mod1" in loader.data and "mod2" in loader.data
    # Directory builder
    patterns = {"mod1": "mod1.npy", "mod2": "mod2.csv"}
    loader2 = di.QuickDatasetBuilder.from_directory(str(tmp_path), patterns, modality_types={"mod1": "tabular", "mod2": "tabular"})
    assert "mod1" in loader2.data and "mod2" in loader2.data
