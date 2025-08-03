import sys
import os
import numpy as np
from sklearn.model_selection import KFold
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MainModel import mainModel, dataIntegration, performanceMetrics

def cross_val_test(n_splits=3):
    N = 90
    n_features = 6
    n_classes = 3
    np.random.seed(123)
    X_text = np.random.randn(N, n_features)
    X_image = np.random.randn(N, n_features)
    y = np.random.randint(0, n_classes, N)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in kf.split(X_text):
        loader = dataIntegration.GenericMultiModalDataLoader()
        loader.add_modality_split('text', X_text[train_idx], X_text[test_idx])
        loader.add_modality_split('image', X_image[train_idx], X_image[test_idx])
        loader.add_labels_split(y[train_idx], y[test_idx])
        model = mainModel.MultiModalEnsembleModel(data_loader=loader, n_bags=2, epochs=2, batch_size=8)
        model.load_and_integrate_data()
        model.fit()
        report = model.evaluate()
        scores.append(report.accuracy)
    print(f"Cross-validation accuracies: {scores}")
    print(f"Mean accuracy: {np.mean(scores):.3f}")

def hyperparam_sensitivity_test():
    N = 60
    n_features = 5
    n_classes = 2
    np.random.seed(42)
    X_text = np.random.randn(N, n_features)
    X_image = np.random.randn(N, n_features)
    y = np.random.randint(0, n_classes, N)
    results = []
    for n_bags in [1, 2, 4]:
        for dropout_strategy in ['random', 'linear']:
            loader = dataIntegration.GenericMultiModalDataLoader()
            loader.add_modality_split('text', X_text[:40], X_text[40:])
            loader.add_modality_split('image', X_image[:40], X_image[40:])
            loader.add_labels_split(y[:40], y[40:])
            model = mainModel.MultiModalEnsembleModel(
                data_loader=loader, n_bags=n_bags, epochs=2, batch_size=8, dropout_strategy=dropout_strategy
            )
            model.load_and_integrate_data()
            model.fit()
            report = model.evaluate()
            results.append((n_bags, dropout_strategy, report.accuracy))
    print("Hyperparameter sensitivity results:")
    for n_bags, dropout_strategy, acc in results:
        print(f"n_bags={n_bags}, dropout_strategy={dropout_strategy}, accuracy={acc:.3f}")

if __name__ == "__main__":
    print("Running cross-validation test...")
    cross_val_test()
    print("\nRunning hyperparameter sensitivity test...")
    hyperparam_sensitivity_test()
