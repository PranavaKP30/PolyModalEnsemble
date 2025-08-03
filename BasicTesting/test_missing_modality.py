import numpy as np
from MainModel import mainModel, dataIntegration

def test_mainModel_with_missing_modalities():
    # Test: Some bags have missing modalities, and test set is missing a modality
    N_train, N_test = 60, 20
    np.random.seed(42)
    loader = dataIntegration.GenericMultiModalDataLoader()
    # Add all modalities for train
    loader.add_modality_split('text', np.random.randn(N_train, 6), np.random.randn(N_test, 6))
    loader.add_modality_split('image', np.random.randn(N_train, 8), np.random.randn(N_test, 8))
    loader.add_modality_split('metadata', np.random.randn(N_train, 4), np.random.randn(N_test, 4))
    n_classes = 3
    loader.add_labels_split(np.random.randint(0, n_classes, N_train), np.random.randint(0, n_classes, N_test))

    # Remove 'image' modality from test set to simulate missing modality at inference
    test_data = loader.get_split('test')[0]
    test_data_missing = dict(test_data)
    del test_data_missing['image']
    test_labels = loader.get_split('test')[1]

    model = mainModel.MultiModalEnsembleModel(
        data_loader=loader, n_bags=3, epochs=2, batch_size=8
    )
    model.load_and_integrate_data()
    model.fit()

    # Predict with missing modality
    pred_result = model.predict(data_dict=test_data_missing)
    print("Predictions with missing modality:", pred_result.predictions)
    print("Confidence:", pred_result.confidence)
    print("Uncertainty:", pred_result.uncertainty)
    print("Metadata:", pred_result.metadata)

    # Evaluate with missing modality
    from MainModel import performanceMetrics
    evaluator = performanceMetrics.PerformanceEvaluator(task_type="classification")
    report = evaluator.evaluate_model(lambda X: pred_result.predictions, test_data_missing, test_labels, model_name="Ensemble")
    print("Performance report with missing modality:", report)

test_mainModel_with_missing_modalities()
