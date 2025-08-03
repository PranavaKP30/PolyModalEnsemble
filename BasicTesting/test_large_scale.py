import sys
import os
import numpy as np
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MainModel import mainModel, dataIntegration, performanceMetrics

def test_large_scale():
    # Large-scale data test (moderate size for local test)
    N_train, N_test = 2000, 500
    n_features = 20
    np.random.seed(6)
    loader = dataIntegration.GenericMultiModalDataLoader()
    loader.add_modality_split('text', np.random.randn(N_train, n_features), np.random.randn(N_test, n_features))
    loader.add_modality_split('image', np.random.randn(N_train, n_features), np.random.randn(N_test, n_features))
    labels_train = np.random.randint(0, 3, N_train)
    labels_test = np.random.randint(0, 3, N_test)
    loader.add_labels_split(labels_train, labels_test)
    model = mainModel.MultiModalEnsembleModel(data_loader=loader, n_bags=3, epochs=2, batch_size=64)
    model.load_and_integrate_data()
    start = time.time()
    model.fit()
    fit_time = time.time() - start
    start = time.time()
    report = model.evaluate()
    eval_time = time.time() - start
    print(f"Large-scale data test report: {report}")
    print(f"Training time: {fit_time:.2f}s, Evaluation time: {eval_time:.2f}s")

if __name__ == "__main__":
    print("Testing large-scale data...")
    test_large_scale()
