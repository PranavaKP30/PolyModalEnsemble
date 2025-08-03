import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MainModel import mainModel, dataIntegration, performanceMetrics

def test_noisy_labels():
    # Add label noise to training data
    N_train, N_test = 40, 10
    np.random.seed(5)
    loader = dataIntegration.GenericMultiModalDataLoader()
    loader.add_modality_split('text', np.random.randn(N_train, 5), np.random.randn(N_test, 5))
    loader.add_modality_split('image', np.random.randn(N_train, 5), np.random.randn(N_test, 5))
    labels_train = np.random.randint(0, 2, N_train)
    labels_test = np.random.randint(0, 2, N_test)
    # Flip 25% of training labels
    flip_idx = np.random.choice(N_train, size=N_train // 4, replace=False)
    labels_train[flip_idx] = 1 - labels_train[flip_idx]
    loader.add_labels_split(labels_train, labels_test)
    model = mainModel.MultiModalEnsembleModel(data_loader=loader, n_bags=2, epochs=2, batch_size=8)
    model.load_and_integrate_data()
    model.fit()
    report = model.evaluate()
    print("Noisy labels test report:", report)

if __name__ == "__main__":
    print("Testing with noisy labels...")
    test_noisy_labels()
