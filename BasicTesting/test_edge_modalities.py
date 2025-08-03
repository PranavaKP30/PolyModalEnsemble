import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MainModel import mainModel, dataIntegration, performanceMetrics

def test_single_modality():
    # Only one modality (text)
    N_train, N_test = 40, 10
    np.random.seed(1)
    loader = dataIntegration.GenericMultiModalDataLoader()
    loader.add_modality_split('text', np.random.randn(N_train, 6), np.random.randn(N_test, 6))
    labels_train = np.random.randint(0, 2, N_train)
    labels_test = np.random.randint(0, 2, N_test)
    loader.add_labels_split(labels_train, labels_test)
    model = mainModel.MultiModalEnsembleModel(data_loader=loader, n_bags=2, epochs=2, batch_size=8)
    model.load_and_integrate_data()
    model.fit()
    report = model.evaluate()
    print("Single modality (text) test report:", report)

def test_imbalanced_feature_dims():
    # Highly imbalanced feature dimensions
    N_train, N_test = 40, 10
    np.random.seed(2)
    loader = dataIntegration.GenericMultiModalDataLoader()
    loader.add_modality_split('text', np.random.randn(N_train, 2), np.random.randn(N_test, 2))
    loader.add_modality_split('image', np.random.randn(N_train, 50), np.random.randn(N_test, 50))
    labels_train = np.random.randint(0, 2, N_train)
    labels_test = np.random.randint(0, 2, N_test)
    loader.add_labels_split(labels_train, labels_test)
    model = mainModel.MultiModalEnsembleModel(data_loader=loader, n_bags=2, epochs=2, batch_size=8)
    model.load_and_integrate_data()
    model.fit()
    report = model.evaluate()
    print("Imbalanced feature dims test report:", report)

if __name__ == "__main__":
    print("Testing single modality...")
    test_single_modality()
    print("\nTesting imbalanced feature dimensions...")
    test_imbalanced_feature_dims()
