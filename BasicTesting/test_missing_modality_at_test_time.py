
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MainModel import mainModel, dataIntegration, performanceMetrics

def test_model_with_missing_modality_at_test_time():
    # Test: All modalities present during training, one modality missing at test time
    N_train, N_test = 60, 20
    np.random.seed(42)
    loader = dataIntegration.GenericMultiModalDataLoader()
    # Add two modalities
    loader.add_modality_split('text', np.random.randn(N_train, 10), np.random.randn(N_test, 10))
    loader.add_modality_split('image', np.random.randn(N_train, 8), np.random.randn(N_test, 8))
    # Balanced class labels
    labels_train = np.random.choice([0, 1], size=N_train)
    labels_test = np.random.choice([0, 1], size=N_test)
    loader.add_labels_split(labels_train, labels_test)

    model = mainModel.MultiModalEnsembleModel(
        data_loader=loader, n_bags=2, epochs=2, batch_size=6
    )
    model.load_and_integrate_data()
    model.fit()

    # Simulate missing 'image' modality at test time
    test_data, _ = loader.get_split('test')
    test_data.pop('image')
    # Predict with missing modality
    pred_result = model.predict(test_data)
    print("Predictions (missing modality):", pred_result.predictions)
    print("Confidence (missing modality):", pred_result.confidence)
    print("Uncertainty (missing modality):", pred_result.uncertainty)
    print("Metadata (missing modality):", pred_result.metadata)

    evaluator = performanceMetrics.PerformanceEvaluator(task_type="classification")
    report = evaluator.evaluate_model(lambda X: pred_result.predictions, model.test_data, model.test_labels, model_name="MissingModalityTest")
    print("Performance report (missing modality):", report)

test_model_with_missing_modality_at_test_time()
