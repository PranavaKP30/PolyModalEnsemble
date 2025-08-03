import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MainModel import mainModel, dataIntegration, performanceMetrics

def test_modality_missing_all_test():
    # Modality missing for all test samples
    N_train, N_test = 40, 10
    np.random.seed(3)
    loader = dataIntegration.GenericMultiModalDataLoader()
    loader.add_modality_split('text', np.random.randn(N_train, 5), np.random.randn(N_test, 5))
    loader.add_modality_split('image', np.random.randn(N_train, 5), np.zeros((N_test, 5)))  # All zeros for test
    labels_train = np.random.randint(0, 2, N_train)
    labels_test = np.random.randint(0, 2, N_test)
    loader.add_labels_split(labels_train, labels_test)
    model = mainModel.MultiModalEnsembleModel(data_loader=loader, n_bags=2, epochs=2, batch_size=8)
    model.load_and_integrate_data()
    model.fit()
    report = model.evaluate()
    print("All test samples missing 'image' modality report:", report)

def test_modality_half_missing():
    # >50% missing in one modality (text)
    N_train, N_test = 40, 10
    np.random.seed(4)
    loader = dataIntegration.GenericMultiModalDataLoader()
    text_train = np.random.randn(N_train, 5)
    text_test = np.random.randn(N_test, 5)
    text_train[:25] = 0  # Zero out >50%
    text_test[:6] = 0
    loader.add_modality_split('text', text_train, text_test)
    loader.add_modality_split('image', np.random.randn(N_train, 5), np.random.randn(N_test, 5))
    labels_train = np.random.randint(0, 2, N_train)
    labels_test = np.random.randint(0, 2, N_test)
    loader.add_labels_split(labels_train, labels_test)
    model = mainModel.MultiModalEnsembleModel(data_loader=loader, n_bags=2, epochs=2, batch_size=8)
    model.load_and_integrate_data()
    model.fit()
    report = model.evaluate()
    print(">50% missing in 'text' modality report:", report)

if __name__ == "__main__":
    print("Testing all test samples missing a modality...")
    test_modality_missing_all_test()
    print("\nTesting >50% missing in one modality...")
    test_modality_half_missing()
