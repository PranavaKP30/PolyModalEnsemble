class ModalityDictDummyClassifier:
    def __init__(self, strategy='most_frequent'):
        from sklearn.dummy import DummyClassifier
        self.clf = DummyClassifier(strategy=strategy)
    def fit(self, X_dict, y):
        # X_dict: dict of modality -> array
        X = np.hstack([X_dict[k] for k in sorted(X_dict)])
        self.clf.fit(X, y)
        return self
    def predict(self, X_dict):
        X = np.hstack([X_dict[k] for k in sorted(X_dict)])
        return self.clf.predict(X)
    def predict_proba(self, X_dict):
        X = np.hstack([X_dict[k] for k in sorted(X_dict)])
        return self.clf.predict_proba(X)
import sys
import os
import numpy as np
from sklearn.dummy import DummyClassifier
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MainModel import mainModel, dataIntegration, performanceMetrics

def test_custom_learner():
    # Use a custom sklearn DummyClassifier as a base learner
    N_train, N_test = 40, 10
    np.random.seed(7)
    loader = dataIntegration.GenericMultiModalDataLoader()
    loader.add_modality_split('text', np.random.randn(N_train, 5), np.random.randn(N_test, 5))
    loader.add_modality_split('image', np.random.randn(N_train, 5), np.random.randn(N_test, 5))
    labels_train = np.random.randint(0, 2, N_train)
    labels_test = np.random.randint(0, 2, N_test)
    loader.add_labels_split(labels_train, labels_test)
    # Manually set up a pipeline and replace all trained learners with ModalityDictDummyClassifier
    model = mainModel.MultiModalEnsembleModel(data_loader=loader, n_bags=2, epochs=2, batch_size=8)
    model.load_and_integrate_data()
    model.fit()
    for i in range(len(model.trained_learners)):
        clf = ModalityDictDummyClassifier()
        clf.fit(model.train_data, model.train_labels)
        model.trained_learners[i] = clf
    report = model.evaluate()
    print("Custom learner (DummyClassifier) test report:", report)

if __name__ == "__main__":
    print("Testing custom learner integration...")
    test_custom_learner()
