import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MainModel import mainModel, dataIntegration, performanceMetrics

def test_metrics_consistency():
    N_train, N_test = 40, 10
    np.random.seed(10)
    loader = dataIntegration.GenericMultiModalDataLoader()
    loader.add_modality_split('text', np.random.randn(N_train, 5), np.random.randn(N_test, 5))
    loader.add_modality_split('image', np.random.randn(N_train, 5), np.random.randn(N_test, 5))
    labels_train = np.random.randint(0, 2, N_train)
    labels_test = np.random.randint(0, 2, N_test)
    loader.add_labels_split(labels_train, labels_test)
    model = mainModel.MultiModalEnsembleModel(data_loader=loader, n_bags=2, epochs=2, batch_size=8)
    model.load_and_integrate_data()
    model.fit()
    pred_result = model.predict()
    # Pipeline metrics
    evaluator = performanceMetrics.PerformanceEvaluator(task_type="classification")
    report = evaluator.evaluate_model(lambda X: pred_result.predictions, model.test_data, model.test_labels, model_name="ConsistencyTest")
    print("Pipeline accuracy:", report.accuracy)
    print("Pipeline f1_score:", report.f1_score)
    # Sklearn metrics
    y_true = model.test_labels
    y_pred = pred_result.predictions
    print("Sklearn accuracy:", accuracy_score(y_true, y_pred))
    print("Sklearn f1_score:", f1_score(y_true, y_pred, average='weighted'))

if __name__ == "__main__":
    print("Testing performance metrics consistency...")
    test_metrics_consistency()
