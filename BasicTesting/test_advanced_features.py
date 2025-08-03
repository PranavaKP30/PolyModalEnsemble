import numpy as np
from MainModel import mainModel, dataIntegration, performanceMetrics, ensemblePrediction

def test_advanced_features():
    N_train, N_test = 120, 40
    np.random.seed(2025)
    loader = dataIntegration.GenericMultiModalDataLoader()
    loader.add_modality_split('text', np.random.randn(N_train, 12), np.random.randn(N_test, 12))
    loader.add_modality_split('image', np.random.randn(N_train, 20), np.random.randn(N_test, 20))
    loader.add_modality_split('metadata', np.random.randn(N_train, 6), np.random.randn(N_test, 6))
    n_classes = 5
    loader.add_labels_split(np.random.randint(0, n_classes, N_train), np.random.randint(0, n_classes, N_test))

    # Custom config: enable denoising, curriculum, transformer fusion
    model = mainModel.MultiModalEnsembleModel(
        data_loader=loader,
        n_bags=5,
        epochs=4,
        batch_size=16
    )
    model.load_and_integrate_data()
    model.fit()

    # Use transformer meta-learner for fusion
    model.ensemble = ensemblePrediction.create_ensemble_predictor(
        task_type='classification',
        aggregation_strategy='transformer_fusion'
    )
    for learner, meta in zip(model.trained_learners, model.learner_metadata):
        model.ensemble.add_trained_learner(learner, meta.get('metrics', {}), meta.get('modalities', []), meta.get('pattern', ''))
    # Setup transformer fusion: input_dim = n_bags (must be divisible by num_heads)
    # Use num_heads=1 to avoid embed_dim/num_heads mismatch
    model.ensemble.transformer_meta_learner = ensemblePrediction.TransformerMetaLearner(
        input_dim=model.n_bags, num_heads=1, num_classes=n_classes, task_type='classification'
    ).to(model.ensemble.device)

    # --- Custom transformer fusion prediction ---
    # Collect predictions from all learners as features (n_learners, n_samples)
    all_preds = []
    for learner in model.trained_learners:
        if hasattr(learner, 'predict_proba'):
            pred = learner.predict_proba(model.test_data)
        else:
            pred = learner.predict(model.test_data)
        # Use class probabilities if available, else class index
        if pred.ndim == 2:
            # Use max prob as feature
            all_preds.append(np.max(pred, axis=1))
        else:
            all_preds.append(pred)
    # Stack as (n_samples, n_learners)
    X_fusion = np.stack(all_preds, axis=1)
    import torch
    X_fusion_torch = torch.tensor(X_fusion, dtype=torch.float32).unsqueeze(1).to(model.ensemble.device)  # (n_samples, 1, n_learners)
    logits, attn_weights = model.ensemble.transformer_meta_learner(X_fusion_torch)
    pred_classes = torch.argmax(logits, dim=-1).cpu().numpy()
    print("Predictions:", pred_classes)
    print("Attention weights:", attn_weights.cpu().detach().numpy())
    # Confidence and uncertainty can be derived from logits if needed

    # Evaluate with advanced metrics
    evaluator = performanceMetrics.PerformanceEvaluator(task_type="classification")
    report = evaluator.evaluate_model(lambda X: pred_classes, model.test_data, model.test_labels, model_name="Ensemble", return_probabilities=True)
    print("Performance report (advanced):", report)

test_advanced_features()
