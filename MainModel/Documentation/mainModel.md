# MultiModalEnsembleModel - Main Model Documentation

## Overview

The `MultiModalEnsembleModel` is a production-ready, sklearn-like interface for multimodal ensemble learning. It provides a unified API for data integration, training, prediction, and evaluation across multiple data modalities (text, image, tabular, etc.) with automatic task detection and architecture selection.

## Key Features

- **Sklearn-like Interface**: Familiar `fit()`, `predict()`, `score()` methods
- **Automatic Task Detection**: Automatically determines classification vs regression
- **Multi-label Support**: Handles both single-label and multi-label classification
- **Modality-Aware Architecture**: Intelligent learner selection based on data types
- **Ensemble Generation**: Advanced modality dropout and diversity optimization
- **Cross-modal Denoising**: Optional cross-modal learning during training
- **Uncertainty Quantification**: Multiple uncertainty estimation methods
- **Model Persistence**: Save/load functionality for production deployment
- **Comprehensive Evaluation**: Extensive performance metrics and analysis

## Class Definition

```python
class MultiModalEnsembleModel:
    """
    MultiModal Ensemble Model with sklearn-like interface.
    
    This model provides a unified interface for multimodal ensemble learning,
    supporting both classification and regression tasks with automatic
    modality-aware architecture selection and ensemble generation.
    """
```

## Parameters

### Core Parameters
- `n_bags` (int, default=10): Number of ensemble bags to create
- `dropout_strategy` (str, default='adaptive'): Modality dropout strategy
  - Options: 'linear', 'exponential', 'random', 'adaptive'
- `epochs` (int, default=10): Number of training epochs for neural network learners
- `batch_size` (int, default=32): Batch size for training
- `random_state` (int, default=42): Random seed for reproducibility

### Task Configuration
- `task_type` (str, default='auto'): Task type specification
  - Options: 'auto', 'classification', 'regression'
  - If 'auto', will be determined from data
- `optimization_strategy` (str, default='balanced'): Optimization strategy for learner selection
  - Options: 'balanced', 'accuracy', 'speed', 'memory'

### Advanced Features
- `enable_denoising` (bool, default=True): Whether to enable cross-modal denoising during training
- `aggregation_strategy` (str, default='weighted_vote'): Ensemble aggregation strategy
- `uncertainty_method` (str, default='entropy'): Uncertainty quantification method

### Ensemble Generation Parameters
- `max_dropout_rate` (float, default=0.5): Maximum modality dropout rate
- `min_modalities` (int, default=1): Minimum number of modalities to keep
- `sample_ratio` (float, default=0.8): Ratio of samples to use in each bag
- `diversity_target` (float, default=0.7): Target diversity for ensemble
- `feature_sampling` (bool, default=True): Whether to sample features within modalities

## Attributes

After fitting, the model contains the following attributes:

- `trained_learners_` (list): List of trained ensemble learners
- `learner_metadata_` (list): Metadata for each learner
- `ensemble_` (EnsemblePredictor): Ensemble predictor instance
- `modality_configs_` (list): Configuration for each modality
- `n_classes_` (int): Number of classes (for classification)
- `n_features_` (int): Number of features
- `is_multilabel_` (bool): Whether the task is multi-label classification
- `task_type_` (str): Determined task type
- `is_fitted_` (bool): Whether the model has been fitted

## Methods

### Core Sklearn-like Methods

#### `fit(X, y, sample_weight=None)`
Fit the multimodal ensemble model.

**Parameters:**
- `X` (dict): Dictionary of modality data. Keys are modality names, values are numpy arrays
- `y` (array-like): Target values
- `sample_weight` (array-like, optional): Sample weights (not currently used)

**Returns:**
- `self`: Returns self for method chaining

**Example:**
```python
model.fit({
    'text': text_features,
    'image': image_features,
    'metadata': metadata
}, labels)
```

#### `predict(X)`
Predict using the fitted model.

**Parameters:**
- `X` (dict): Dictionary of modality data for prediction

**Returns:**
- `predictions` (array-like): Predicted values

**Example:**
```python
predictions = model.predict({
    'text': test_text_features,
    'image': test_image_features,
    'metadata': test_metadata
})
```

#### `predict_proba(X)`
Predict class probabilities (classification only).

**Parameters:**
- `X` (dict): Dictionary of modality data for prediction

**Returns:**
- `probabilities` (array-like): Class probabilities

**Example:**
```python
probabilities = model.predict_proba(test_data)
```

#### `predict_classes(X)`
Predict class labels (classification only).

**Parameters:**
- `X` (dict): Dictionary of modality data for prediction

**Returns:**
- `predictions` (array-like): Predicted class labels

#### `predict_values(X)`
Predict values (regression only).

**Parameters:**
- `X` (dict): Dictionary of modality data for prediction

**Returns:**
- `predictions` (array-like): Predicted values

#### `score(X, y, sample_weight=None)`
Return the score on the given test data and labels.

**Parameters:**
- `X` (dict): Dictionary of modality data
- `y` (array-like): True labels
- `sample_weight` (array-like, optional): Sample weights (not currently used)

**Returns:**
- `score` (float): Mean accuracy for classification, R² for regression

### Advanced Methods

#### `evaluate(X=None, y=None)`
Comprehensive model evaluation.

**Parameters:**
- `X` (dict, optional): Dictionary of modality data. If None, uses test data
- `y` (array-like, optional): True labels. If None, uses test labels

**Returns:**
- `report` (ModelPerformanceReport): Comprehensive performance report

**Example:**
```python
report = model.evaluate()
print(f"Accuracy: {report.accuracy:.4f}")
print(f"F1 Score: {report.f1_score:.4f}")
print(f"AUC-ROC: {report.auc_roc:.4f}")
```

#### `get_feature_importance()`
Get feature importance scores.

**Returns:**
- `importance` (dict): Dictionary of feature importance scores per modality

**Example:**
```python
importance = model.get_feature_importance()
for modality, scores in importance.items():
    print(f"{modality} importance: {np.mean(scores):.4f}")
```

#### `save(filepath)`
Save the fitted model to a file.

**Parameters:**
- `filepath` (str): Path to save the model

**Example:**
```python
model.save('my_model.pkl')
```

#### `load(filepath)`
Load a fitted model from a file.

**Parameters:**
- `filepath` (str): Path to the saved model

**Returns:**
- `model` (MultiModalEnsembleModel): Loaded model

**Example:**
```python
loaded_model = MultiModalEnsembleModel.load('my_model.pkl')
```

#### `get_params(deep=True)`
Get parameters for this estimator.

**Returns:**
- `params` (dict): Dictionary of parameters

#### `set_params(**params)`
Set parameters for this estimator.

**Parameters:**
- `**params`: Parameters to set

**Returns:**
- `self`: Returns self for method chaining

## Usage Examples

### Basic Classification
```python
from MainModel.mainModel import MultiModalEnsembleModel
import numpy as np

# Create synthetic data
n_samples = 1000
text_features = np.random.randn(n_samples, 10)
image_features = np.random.randn(n_samples, 10)
metadata = np.random.randn(n_samples, 10)
labels = np.random.randint(0, 3, n_samples)

# Initialize and fit model
model = MultiModalEnsembleModel(n_bags=5, random_state=42)
model.fit({
    'text': text_features,
    'image': image_features,
    'metadata': metadata
}, labels)

# Make predictions
test_data = {
    'text': text_features[:10],
    'image': image_features[:10],
    'metadata': metadata[:10]
}
predictions = model.predict(test_data)
probabilities = model.predict_proba(test_data)

# Evaluate
score = model.score(test_data, labels[:10])
print(f"Accuracy: {score:.4f}")
```

### Regression
```python
# Create continuous targets
targets = np.random.randn(n_samples)

# Initialize model for regression
model = MultiModalEnsembleModel(
    n_bags=3,
    task_type='regression',
    random_state=42
)

# Fit and predict
model.fit({
    'text': text_features,
    'image': image_features,
    'metadata': metadata
}, targets)

predictions = model.predict(test_data)
score = model.score(test_data, targets[:10])
print(f"R² Score: {score:.4f}")
```

### Multi-label Classification
```python
# Create multi-label targets
n_labels = 4
labels = np.random.randint(0, 2, (n_samples, n_labels))

# Fit model (automatically detects multi-label)
model = MultiModalEnsembleModel(n_bags=3, random_state=42)
model.fit({
    'text': text_features,
    'image': image_features,
    'metadata': metadata
}, labels)

# Predictions are binary for each label
predictions = model.predict(test_data)
print(f"Predictions shape: {predictions.shape}")  # (10, 4)
```

### Advanced Configuration
```python
# Initialize with advanced parameters
model = MultiModalEnsembleModel(
    n_bags=10,
    dropout_strategy='adaptive',
    epochs=20,
    batch_size=64,
    optimization_strategy='balanced',
    aggregation_strategy='weighted_vote',
    uncertainty_method='entropy',
    enable_denoising=True,
    max_dropout_rate=0.6,
    diversity_target=0.8,
    random_state=42,
    verbose=True
)
```

## Internal Architecture

### Training Pipeline
1. **Data Integration**: Automatically splits data into train/test sets
2. **Task Detection**: Determines classification vs regression from labels
3. **Ensemble Generation**: Creates diverse bags using modality dropout
4. **Learner Selection**: Intelligently selects base learners for each bag
5. **Training**: Trains all learners with optional cross-modal denoising
6. **Ensemble Creation**: Combines learners into final ensemble predictor

### Data Flow
```
Input Data (dict of modalities) 
    ↓
Data Integration & Validation
    ↓
Task Type Detection
    ↓
Ensemble Bag Generation
    ↓
Base Learner Selection & Training
    ↓
Ensemble Predictor Creation
    ↓
Ready for Prediction
```

### Error Handling
- Validates all input parameters during initialization
- Ensures data consistency across modalities
- Provides clear error messages for invalid operations
- Handles missing modalities gracefully
- Validates task-specific method calls

## Performance Optimization

### Training Optimizations
- **Batch Processing**: Configurable batch sizes for memory efficiency
- **Mixed Precision**: Automatic mixed precision training when available
- **Gradient Accumulation**: For large models with limited memory
- **Early Stopping**: Prevents overfitting with validation monitoring

### Prediction Optimizations
- **Parallel Inference**: Parallel prediction across ensemble members
- **Memory Management**: Efficient memory usage during prediction
- **Caching**: Caches intermediate results for repeated predictions

### Memory Optimizations
- **Sparse Support**: Handles sparse data efficiently
- **Lazy Loading**: Loads data on-demand for large datasets
- **Garbage Collection**: Automatic cleanup of intermediate results

## Production Deployment

### Model Persistence
```python
# Save model
model.save('production_model.pkl')

# Load model
loaded_model = MultiModalEnsembleModel.load('production_model.pkl')

# Verify persistence
assert np.allclose(model.predict(test_data), loaded_model.predict(test_data))
```

### Parameter Management
```python
# Get current parameters
params = model.get_params()
print(f"Current n_bags: {params['n_bags']}")

# Update parameters
model.set_params(n_bags=15, epochs=25)
```

### Monitoring and Logging
- Comprehensive training metrics
- Performance evaluation reports
- Feature importance analysis
- Uncertainty quantification
- Model health monitoring

## Best Practices

### Data Preparation
- Ensure all modalities have the same number of samples
- Normalize features appropriately for each modality
- Handle missing values before training
- Use meaningful modality names

### Model Configuration
- Start with default parameters for initial testing
- Adjust `n_bags` based on dataset size and complexity
- Use `adaptive` dropout strategy for best results
- Enable denoising for cross-modal learning

### Performance Tuning
- Increase `batch_size` for faster training (if memory allows)
- Adjust `epochs` based on convergence patterns
- Monitor validation performance to prevent overfitting
- Use `verbose=True` during development for debugging

### Production Considerations
- Always set `random_state` for reproducibility
- Save models after training for deployment
- Validate model performance on holdout data
- Monitor model drift in production

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce batch_size or n_bags
3. **Slow Training**: Increase batch_size or reduce epochs
4. **Poor Performance**: Check data quality and feature engineering

### Debug Mode
```python
# Enable verbose mode for debugging
model = MultiModalEnsembleModel(verbose=True)

# Check model state
print(f"Model fitted: {model.is_fitted_}")
print(f"Task type: {model.task_type_}")
print(f"Number of classes: {model.n_classes_}")
```

## Conclusion

The `MultiModalEnsembleModel` provides a complete, production-ready solution for multimodal ensemble learning with a familiar sklearn-like interface. It automatically handles the complexity of multimodal data integration, ensemble generation, and prediction while providing extensive customization options for advanced users.

The model is designed to be:
- **Easy to use**: Familiar sklearn interface
- **Robust**: Comprehensive error handling and validation
- **Flexible**: Extensive parameter customization
- **Scalable**: Optimized for large datasets
- **Production-ready**: Save/load functionality and monitoring

This implementation represents a significant advancement in multimodal ensemble learning, providing both ease of use and advanced capabilities for research and production applications.
