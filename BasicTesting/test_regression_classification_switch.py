import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MainModel import mainModel, dataIntegration, performanceMetrics

def test_regression_classification_switch():
    # Classification
    N_train, N_test = 40, 10
    np.random.seed(8)
    loader = dataIntegration.GenericMultiModalDataLoader()
    loader.add_modality_split('text', np.random.randn(N_train, 5), np.random.randn(N_test, 5))
    loader.add_modality_split('image', np.random.randn(N_train, 5), np.random.randn(N_test, 5))
    labels_train = np.random.randint(0, 2, N_train)
    labels_test = np.random.randint(0, 2, N_test)
    loader.add_labels_split(labels_train, labels_test)
    model = mainModel.MultiModalEnsembleModel(data_loader=loader, n_bags=2, epochs=2, batch_size=8)
    model.load_and_integrate_data()
    model.fit()
    report_class = model.evaluate()
    print("Classification test report:", report_class)

    # Regression
    N_train, N_test = 40, 10
    np.random.seed(9)
    loader = dataIntegration.GenericMultiModalDataLoader()
    loader.add_modality_split('text', np.random.randn(N_train, 5), np.random.randn(N_test, 5))
    loader.add_modality_split('image', np.random.randn(N_train, 5), np.random.randn(N_test, 5))
    labels_train = np.random.uniform(-5, 5, N_train)
    labels_test = np.random.uniform(-5, 5, N_test)
    loader.add_labels_split(labels_train, labels_test)
    model = mainModel.MultiModalEnsembleModel(data_loader=loader, n_bags=2, epochs=2, batch_size=8)
    model.load_and_integrate_data()
    model.fit()
    report_reg = model.evaluate()
    print("Regression test report:", report_reg)

if __name__ == "__main__":
    print("Testing classification and regression switch...")
    test_regression_classification_switch()
