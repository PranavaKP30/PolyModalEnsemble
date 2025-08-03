import numpy as np
from MainModel import mainModel, dataIntegration, performanceMetrics

def test_model_with_unbalanced_modalities_and_labels():
    # Test: Unbalanced modalities (some bags only have one modality), unbalanced class distribution
    N_train, N_test = 80, 25
    np.random.seed(7)
    loader = dataIntegration.GenericMultiModalDataLoader()
    # Add modalities with different feature dims
    loader.add_modality_split('text', np.random.randn(N_train, 7), np.random.randn(N_test, 7))
    loader.add_modality_split('image', np.random.randn(N_train, 15), np.random.randn(N_test, 15))
    # Unbalanced class labels (mostly class 0)
    labels_train = np.random.choice([0, 1, 2], size=N_train, p=[0.8, 0.15, 0.05])
    labels_test = np.random.choice([0, 1, 2], size=N_test, p=[0.8, 0.15, 0.05])
    loader.add_labels_split(labels_train, labels_test)

    # Remove 'text' modality from half the training samples to simulate missing modality in some bags
    text_data = loader.get_split('train')[0]['text']
    text_data[:N_train//2] = 0  # Zero out text features for half
    test_text_data = loader.get_split('test')[0]['text']
    loader.add_modality_split('text', text_data, test_text_data)

    model = mainModel.MultiModalEnsembleModel(
        data_loader=loader, n_bags=3, epochs=3, batch_size=8
    )
    model.load_and_integrate_data()
    model.fit()

    pred_result = model.predict()
    print("Predictions (unbalanced):", pred_result.predictions)
    print("Confidence (unbalanced):", pred_result.confidence)
    print("Uncertainty (unbalanced):", pred_result.uncertainty)
    print("Metadata (unbalanced):", pred_result.metadata)

    evaluator = performanceMetrics.PerformanceEvaluator(task_type="classification")
    report = evaluator.evaluate_model(lambda X: pred_result.predictions, model.test_data, model.test_labels, model_name="UnbalancedTest")
    print("Performance report (unbalanced):", report)

test_model_with_unbalanced_modalities_and_labels()
