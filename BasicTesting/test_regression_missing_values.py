import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MainModel import mainModel, dataIntegration, performanceMetrics

def test_model_regression_with_missing_values():
    # Test: Regression, multiple modalities, random missing values (NaNs)
    N_train, N_test = 100, 30
    np.random.seed(123)
    loader = dataIntegration.GenericMultiModalDataLoader()
    # Add two modalities
    text_train = np.random.randn(N_train, 5)
    text_test = np.random.randn(N_test, 5)
    image_train = np.random.randn(N_train, 12)
    image_test = np.random.randn(N_test, 12)
    # Introduce random NaNs in both modalities
    nan_mask_text = np.random.rand(*text_train.shape) < 0.1
    nan_mask_image = np.random.rand(*image_train.shape) < 0.1
    text_train[nan_mask_text] = np.nan
    image_train[nan_mask_image] = np.nan
    loader.add_modality_split('text', text_train, text_test)
    loader.add_modality_split('image', image_train, image_test)
    # Continuous regression targets
    y_train = np.random.uniform(-5, 5, size=N_train)
    y_test = np.random.uniform(-5, 5, size=N_test)
    loader.add_labels_split(y_train, y_test)

    model = mainModel.MultiModalEnsembleModel(
        data_loader=loader, n_bags=2, epochs=2, batch_size=10
    )
    model.load_and_integrate_data()
    model.fit()

    pred_result = model.predict()
    print("Predictions (regression, missing values):", pred_result.predictions)
    print("Confidence (regression, missing values):", pred_result.confidence)
    print("Uncertainty (regression, missing values):", pred_result.uncertainty)
    print("Metadata (regression, missing values):", pred_result.metadata)

    evaluator = performanceMetrics.PerformanceEvaluator(task_type="regression")
    report = evaluator.evaluate_model(lambda X: pred_result.predictions, model.test_data, model.test_labels, model_name="RegressionMissingValuesTest")
    print("Performance report (regression, missing values):", report)

test_model_regression_with_missing_values()
