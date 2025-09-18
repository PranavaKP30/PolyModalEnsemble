
🔄 IN PROGRESS: Implement Real SOTA Baselines (Priority 5) - Current baselines need SOTA upgrade


# Comprehensive Experimentation Pipeline for Real Datasets

## 🎯 **Overview**
This document outlines a complete experimentation pipeline for evaluating the Modality-Aware Adaptive Bagging Ensemble on real datasets (AmazonReviews, CocoCaptions, YelpOpen). Each dataset will undergo systematic testing across multiple phases to ensure comprehensive evaluation.

## 📊 **Dataset Information**
- **AmazonReviews**: Text + Metadata classification
- **CocoCaptions**: Image + Text multimodal classification  
- **Dataset #3**: TBD

## 🏗️ **Pipeline Structure**

### **Phase 1: Data Validation & Preprocessing**
**Purpose**: Ensure data quality and consistency before experimentation

**Implementation Steps**:

#### **Step 1: Dataset Loading & Stratified Sampling**
- **Load AmazonReviews dataset** from processed data files (`train/text_features.npy`, `train/metadata_features.npy`, `train/labels.npy`, etc.)
- **Apply stratified sampling** based on test mode:
  - **Quick mode**: 10,000 total samples (8,000 train, 2,000 test)
  - **Full mode**: 300,000 total samples (240,000 train, 60,000 test)
- **Maintain class balance** across modalities using `train_test_split` with stratification
- **Convert labels** from 1-5 range to 0-4 range (0-indexed for sklearn compatibility)

#### **Step 2: Modality Separation**
- **Keep modalities separate** (NOT combined):
  - Create `train_text` DataFrame with text features + labels
  - Create `train_metadata` DataFrame with metadata features + labels
  - Create `test_text` DataFrame with text features + labels
  - Create `test_metadata` DataFrame with metadata features + labels

#### **Step 3: Data Quality Assessment**
- **Sample count verification** - ensure expected sample sizes
- **Feature analysis** - count features, check data types
- **Missing value analysis** - identify and quantify missing data
- **Data type analysis** - verify numeric vs categorical columns
- **Modality analysis** - detect text vs metadata modalities
- **Cross-modal alignment** - verify modalities have matching sample counts

#### **Step 4: Modality-Specific Validation**
- **Text modality validation**:
  - Feature correlation analysis
  - Statistical analysis (mean, std, min, max)
  - High correlation pair detection
- **Metadata modality validation**:
  - Outlier detection using IQR method
  - Feature correlation analysis
- **Cross-modal validation**:
  - Label distribution analysis
  - Modality presence verification

#### **Step 5: Data Preprocessing**
- **Handle NaN values** using specified method (fill_mean, fill_zero, drop)
- **Handle Inf values** using specified method (fill_max, fill_zero, drop)
- **Optional normalization** (exclude target columns)
- **Optional outlier removal** (exclude target columns)

#### **Step 6: Quality Score Calculation**
- Calculate overall quality score (0-100)
- Deduct points for missing values, validation failures, inconsistent dimensions
- Generate validation checks (passed/failed/warning)

#### **Step 7: Data Persistence**
- **Save sampled datasets** to `sampled_{test_mode}_{seed}/` directory
- **Save sampling info** with metadata about the sampling process
- **Save quality reports** and preprocessing summaries
- **Create reusable data** for subsequent phases

#### **Step 8: Results Generation**
- Generate comprehensive data quality report
- Create preprocessing summary
- Calculate memory usage statistics
- Verify label conversion (0-4 range)
- Save all results to JSON files

**Key Features**:
- **Memory-efficient loading** using memory mapping for large datasets
- **Reproducible sampling** with controlled random seeds
- **Comprehensive validation** covering all aspects of data quality
- **Modality preservation** - keeps text and metadata separate as required by MainModel
- **Quality scoring** provides objective assessment of data readiness

**Outputs**:
- Data quality report (`data_quality_report.json`)
- Preprocessing validation summary (`preprocessing_summary.json`)
- Memory usage statistics
- **Processed data files**:
  - `sampled_{test_mode}_{seed}/train_text/train_text_data.csv` (8k samples × 1,001 columns)
  - `sampled_{test_mode}_{seed}/train_metadata/train_metadata_data.csv` (8k samples × 15 columns)
  - `sampled_{test_mode}_{seed}/test_text/test_text_data.csv` (2k samples × 1,001 columns)
  - `sampled_{test_mode}_{seed}/test_metadata/test_metadata_data.csv` (2k samples × 15 columns)
- Sampling metadata (`sampling_info.json`, `metadata.json`
---

### **Phase 2: Baseline Model Evaluation**
**Purpose**: Establish performance baselines for comparison

**INPUTS:**
- **Processed Data from Phase 1**: `processed_data` dictionary containing:
  - `train_text` / `test_text`: Text data (pandas Series)
  - `train_metadata` / `test_metadata`: Metadata features (pandas DataFrame)
  - `train_labels` / `test_labels`: Target labels (numpy arrays)
  - `task_type`: "classification" (from Phase 1)
  - `num_classes`: 5 (from Phase 1)
  - `class_names`: ["1-star", "2-star", "3-star", "4-star", "5-star"] (from Phase 1)

**PROCESSING STEPS:**

#### **Step 1: Data Preparation**
- **Load processed data** from Phase 1 results
- **Validate data format** (check for missing features/labels)
- **Prepare data structures** for different baseline types
- **Handle missing values** (fill with mean/zero/drop)
- **Scale features** if needed (StandardScaler)

#### **Step 2: Text Feature Extraction**
- **TF-IDF Vectorization**: Extract text features with parameters:
  - `max_features`: 1000-5000 (varies by model)
  - `ngram_range`: (1, 2) - unigrams + bigrams
  - `stop_words`: 'english'
  - `min_df`: 2, `max_df`: 0.95
- **BERT Embeddings**: (if transformers available)
  - Load DistilBERT tokenizer and model
  - Extract [CLS] token embeddings
  - Limit to 100 samples for efficiency

#### **Step 3: Model Training & Evaluation (5 Steps)**

**Step 3.1: Text Baselines**
- Train: TF-IDF + SVM, Logistic Regression, Random Forest, XGBoost, BERT
- Evaluate: Predict on train/test sets
- Calculate: Accuracy, F1, Precision, Recall, Balanced Accuracy, AUC-ROC

**Step 3.2: Metadata Baselines**
- Train: Random Forest, XGBoost, SVM, Logistic Regression (metadata only)
- Evaluate: Predict on train/test sets
- Calculate: All classification metrics

**Step 3.3: Traditional Fusion Baselines**
- **Early Fusion**: Concatenate text + metadata features
- Train: Random Forest, SVM on combined features
- **Late Fusion**: Train separate models, combine predictions via voting/weighted average
- Calculate: All classification metrics

**Step 3.4: Ensemble Baselines**
- **Bagging**: Random Forest with modality-specific features
- **XGBoost**: Multimodal features with boosting
- **Stacking**: Meta-learner on base model predictions
- Calculate: All classification metrics

**Step 3.5: SoTA Baselines** ⭐
- **BERT + Metadata Fusion**: DistilBERT + metadata + Random Forest
- **XGBoost + BERT-like**: Enhanced TF-IDF + metadata + XGBoost
- **LightGBM + Multimodal**: Advanced LightGBM with optimized hyperparameters
- **TabNet + Text**: TabNet with text + metadata (if available)
- **Advanced Ensemble**: Voting ensemble of diverse models
- Calculate: All classification metrics

#### **Step 4: Performance Analysis**
- **Calculate metrics** for each model:
  - **Classification**: Accuracy, F1-Score, Precision, Recall, Balanced Accuracy, AUC-ROC
  - **Efficiency**: Training time, prediction time, memory usage
- **Statistical significance tests** between models
- **Performance ranking** and comparison

#### **Step 5: Results Compilation**
- **Create results dictionary** with:
  - Model name, metrics, training time, model type, description
  - Train/test performance breakdown
  - Error handling information
- **Generate performance summary** DataFrame
- **Save results** to files

**OUTPUTS:**
- **`baseline_results.json`**: Detailed results for each model including:
  - Metrics (accuracy, F1, precision, recall, etc.)
  - Training time and efficiency metrics
  - Model type and description
  - Error information (if any)
- **`baseline_summary.csv`**: Performance comparison summary
- **Logs**: Detailed logging of each step and model training
- **Return**: Results dictionary for Phase 3 input

**ERROR HANDLING:**
- **Missing dependencies**: Skip models if libraries not available
- **Data validation**: Check for missing features/labels
- **Model training errors**: Try-catch blocks for each model
- **Resource management**: Memory usage monitoring
- **Timeout handling**: Prevent hanging on slow models

**DEPENDENCIES:**
- **Required**: `sklearn`, `pandas`, `numpy`, `xgboost`
- **Optional**: `transformers`, `lightgbm`, `tabnet`, `torch`

---

### **Phase 3: MainModel Hyperparameter Optimization**
**Purpose**: Find optimal hyperparameter configuration for each dataset

**Loading data**
load data from phase 1 as is. For training and testing load the lables and the individual modality data

**Hyperparameter Search Space using only 1k samples (sample out the 1k from the sampled phase 1 data) -- will use Optuna optimization with intelligent search:**

**Optuna Trial Configuration:**
- **Quick Mode**: 75 trials (intelligent search of full space)
- **Full Mode**: 300 trials (comprehensive search of full space)
- **Search Space**: Same comprehensive space for both modes (Optuna TPE sampler handles intelligent exploration)

**VARY HYPERPARAMETERS (19 total across all stages) - UPDATED FOR IMBALANCED DATA HANDLING + LEARNER TYPE SELECTION:**

**🚨 IMBALANCED DATA OPTIMIZATIONS:**
The hyperparameter space has been specifically optimized to handle severely imbalanced datasets (e.g., Amazon Reviews with 60% Class 4, 20% Class 3, 8% Class 2, 5% Class 1, 6% Class 0). Key changes include:

1. **Added Class Imbalance Handling**: Automatic class weighting in loss function using inverse frequency weighting
2. **Optimized Learning Rates**: Moderate rates (1e-4 to 2e-3) for better gradient estimates on minority classes
3. **Increased Training Duration**: Higher epoch counts (50-150) to allow proper minority class learning
4. **Smaller Batch Sizes**: Reduced to 16-64 for better gradient estimates with minority samples
5. **Higher Denoising Weights**: Increased to 0.1-0.25 to improve minority class feature representation
6. **More Patience**: Extended early stopping patience (10-25) for minority class convergence
7. **Label Smoothing**: NEW parameter (0.0-0.2) to prevent overconfidence on majority class
8. **Normalization Enabled**: Set to True for proper feature scaling across modalities

These optimizations address the root cause of poor balanced accuracy (~20%) by ensuring the model learns meaningful patterns for all classes, not just the majority class.
**Stage 1:
🔧 CORE DATA INTEGRATION HYPERPARAMETERS
verbose
Description: Enables verbose logging and progress information during data loading and processing
Range: [True, False]
Default: Truetep and
Testing: FIXED - Keep True for debugging and monitoring
�� DATA CLEANING HYPERPARAMETERS
handle_nan
Description: Strategy for handling missing values (NaN) in the data - can fill with mean values, zeros, or drop samples with missing data
Range: ['fill_mean', 'fill_zero', 'drop']
Default: 'fill_mean'
Testing: FIXED - Keep 'fill_mean' for robust handling
handle_inf
Description: Strategy for handling infinite values (Inf) in the data - can fill with maximum values, zeros, or drop samples with infinite values
Range: ['fill_max', 'fill_zero', 'drop']
Default: 'fill_max'
Testing: FIXED - Keep 'fill_max' for robust handling
🔧 DATA PREPROCESSING HYPERPARAMETERS
normalize
Description: Enables data normalization (standardization) to scale features to zero mean and unit variance, critical for imbalanced multimodal data
Range: [True, False]
Default: True
Testing: FIXED - Set to True for proper feature scaling across TF-IDF and metadata modalities (UPDATED FOR IMBALANCED DATA)
remove_outliers
Description: Enables outlier removal using statistical thresholds to remove samples that are more than N standard deviations from the mean
Range: [True, False]
Default: False
Testing: FIXED - Keep False (preserve data integrity)
outlier_std
Description: Standard deviation threshold for outlier detection - samples beyond this threshold are considered outliers and removed
Range: [1.0, 2.0, 3.0, 4.0, 5.0]
Default: 3.0
Testing: FIXED - Keep 3.0 (standard threshold)

**Stage 2: Ensemble Generation (1 VARY, 5 FIXED parameters)
🎯 CORE ENSEMBLE HYPERPARAMETERS
1. n_bags
Description: Number of ensemble bags to create - determines the size of the ensemble and affects diversity and computational cost
Range: [10, 15, 20, 25, 30]
Default: 20
Testing: VARY - Key parameter for ensemble diversity and computational cost
Range: [15, 20, 25] (3 parameters) - Removed 10 (too small for effective ensemble)
2. dropout_strategy
Description: Modality dropout strategy for creating diverse bags - includes your novel adaptive strategy that dynamically adjusts based on ensemble diversity
Range: ['linear', 'exponential', 'random', 'adaptive']
Default: 'adaptive' (your novel feature)
Testing: FIXED - Keep 'adaptive' for main experiments, vary in ablation studies
3. max_dropout_rate
Description: Maximum dropout rate for modality removal - controls the upper bound of how many modalities can be dropped from each bag
Range: [0.3, 0.4, 0.5, 0.6, 0.7]
Default: 0.5
Testing: VARY - Critical for controlling ensemble diversity
Range: [0.4, 0.5, 0.6] (3 parameters) - Removed 0.3 (too low for effective diversity)
4. min_modalities
Description: Minimum number of modalities that must remain in each bag - ensures no bag becomes too sparse
Range: [1, 2, 3] (depends on total modalities)
Default: 1
Testing: FIXED - Keep 1 for maximum flexibility
🎯 DIVERSITY OPTIMIZATION HYPERPARAMETERS
5. sample_ratio
Description: Bootstrap sampling ratio for each bag - controls how much of the training data is sampled for each bag
Range: [0.7, 0.8, 0.9, 1.0]
Default: 0.8
Testing: VARY - Critical for bootstrap sampling diversity
Range: [0.8, 0.9, 1.0] (3 parameters) - Removed 0.7 (too low for sufficient data)
🎯 FEATURE SAMPLING HYPERPARAMETERS
6. feature_sampling_ratio
Description: Ratio of features to sample within each active modality - enables hierarchical feature sampling within modalities
Range: [0.7, 0.8, 0.9, 1.0]
Default: 0.8
Testing: VARY - Critical for feature-level diversity
Range: [0.7, 0.8, 0.9, 1.0] (4 parameters)
🎯 VALIDATION AND CONTROL HYPERPARAMETERS
7. random_state
Description: Random seed for reproducible ensemble generation and bag creation - set based on test run number during experiments
Range: [42, 123, 456, 789, 1000, ...] (any integer)
Default: 42
Testing: EXPERIMENT-DEPENDENT - Set based on test run number, not varied per test

**Stage 3: Base Learner Selection (2 VARY, 11 FIXED parameters)
🎯 CORE SELECTION HYPERPARAMETERS
task_type
Description: Task type for the learner selection - automatically determined from dataset in Phase 1
Range: ['classification', 'regression']
Default: 'classification'
Testing: FIXED - Determined by dataset characteristics
Fixed based on dataset
optimization_mode
Description: Optimization strategy for learner selection - balances accuracy, performance, and efficiency
Range: ['accuracy', 'performance', 'efficiency']
Default: 'accuracy'
Testing: VARY - Key parameter for learner selection strategy
Range: ['accuracy', 'performance'] (2 parameters) - Removed 'efficiency' (causes low accuracy)
learner_type
Description: Base learner implementation type - choose between sklearn and PyTorch learners for different datasets
Range: ['sklearn', 'pytorch']
Default: 'sklearn'
Testing: VARY - NEW HYPERPARAMETER for choosing learner implementation
Range: ['sklearn', 'pytorch'] (2 parameters) - Sklearn better for tabular/text, PyTorch better for images/sequences
n_classes
Description: Number of classes for classification tasks - only relevant when task_type is classification
Range: [2, 3, 4, 5, 10, 20]
Default: 2
Testing: FIXED - Determined by dataset characteristics
Fixed based on dataset
random_state
Description: Random seed for reproducible learner selection - uses same seed as Stage 2 for consistency
Range: [42, 123, 456, 789, 1000, ...] (any integer)
Default: 42
Testing: EXPERIMENT-DEPENDENT - Set based on test run number, not varied per test
🎯 ABLATION STUDY HYPERPARAMETERS
modality_aware
Description: Enable modality-aware learner selection - novel feature that selects learners based on modality combinations
Range: [True, False]
Default: True
Testing: FIXED - Keep True for main experiments, vary in ablation studies
bag_learner_pairing
Description: Enable complete bag-learner pairing storage - stores comprehensive pairing information
Range: [True, False]
Default: True
Testing: FIXED - Keep True for main experiments
metadata_level
Description: Level of metadata storage for bag-learner configurations - affects interpretability and analysis depth
Range: ['minimal', 'complete', 'enhanced']
Default: 'complete'
Testing: FIXED - Keep 'complete' for main experiments, vary in ablation studies
pairing_focus
Description: Focus for pairing optimization - determines how bag-learner pairs are optimized
Range: ['performance', 'diversity', 'efficiency']
Default: 'performance'
Testing: FIXED - Keep 'performance' for main experiments, vary in ablation studies
🎯 MODALITY WEIGHTAGE HYPERPARAMETERS
feature_ratio_weight
Description: Weight for feature selection ratio in modality importance calculation - controls how feature sampling affects modality importance
Range: [0.2, 0.3, 0.4, 0.5, 0.6]
Default: 0.4
Testing: FIXED - Keep 0.4 for main experiments, vary in ablation studies
variance_weight
Description: Weight for data variance in modality importance calculation - controls how data variance affects modality importance
Range: [0.2, 0.3, 0.4, 0.5, 0.6]
Default: 0.3
Testing: FIXED - Keep 0.3 for main experiments, vary in ablation studies
dimensionality_weight
Description: Weight for dimensionality in modality importance calculation - controls how feature dimensionality affects modality importance
Range: [0.2, 0.3, 0.4, 0.5, 0.6]
Default: 0.3
Testing: FIXED - Keep 0.3 for main experiments, vary in ablation studies
🎯 PERFORMANCE PREDICTION HYPERPARAMETERS
base_performance
Description: Base performance score for learner performance prediction - starting point for performance estimation
Range: [0.5, 0.6, 0.7, 0.8]
Default: 0.6
Testing: FIXED - Keep consistent baseline
diversity_bonus
Description: Bonus multiplier for modality diversity in performance prediction - rewards diverse modality combinations
Range: [0.05, 0.1, 0.15, 0.2]
Default: 0.1
Testing: FIXED - Keep consistent bonus
weightage_bonus
Description: Bonus multiplier for modality weightage in performance prediction - rewards balanced modality usage
Range: [0.05, 0.1, 0.15, 0.2]
Default: 0.1
Testing: FIXED - Keep consistent bonus
dropout_penalty
Description: Penalty multiplier for dropout rate in performance prediction - penalizes high dropout rates
Range: [0.05, 0.1, 0.15, 0.2]
Default: 0.1
Testing: FIXED - Keep consistent penalty
🎯 LEARNER CONFIGURATION HYPERPARAMETERS
n_estimators
Description: Number of estimators for tree-based learners - controls model complexity and training time
Range: [50, 100, 150, 200, 250]
Default: 100
Testing: VARY - Critical for model complexity
Range: [50, 100, 150, 200] (4 parameters)
max_depth
Description: Maximum depth for tree-based learners - controls model complexity and overfitting
Range: [5, 10, 15, 20, 25]
Default: 10
Testing: VARY - Critical for model complexity
Range: [5, 10, 15, 20] (4 parameters)
learning_rate
Description: Learning rate for gradient-based learners - controls training speed and convergence
Range: [0.01, 0.05, 0.1, 0.15, 0.2]
Default: 0.1
Testing: VARY - Critical for training dynamics
Range: [0.01, 0.05, 0.1, 0.15] (4 parameters)
early_stopping
Description: Enable early stopping for training - prevents overfitting and improves generalization
Range: [True, False]
Default: True
Testing: FIXED - Always enable early stopping
�� NEURAL NETWORK ARCHITECTURE HYPERPARAMETERS
hidden_layers
Description: Hidden layer configuration for neural network learners - fixed architecture for consistency
Range: [(128, 64), (256, 128), (512, 256)]
Default: (128, 64)
Testing: FIXED - Keep consistent architecture
dropout
Description: Dropout rate for neural network learners - fixed for consistency
Range: [0.1, 0.2, 0.3, 0.4]
Default: 0.2
Testing: FIXED - Keep consistent dropout
max_features
Description: Maximum features for tree-based learners - fixed as 'auto' for consistency
Range: ['sqrt', 'log2', 'auto']
Default: 'auto'
Testing: FIXED - Keep as 'auto'
min_samples_split
Description: Minimum samples to split for tree-based learners - fixed for consistency
Range: [2, 5, 10, 15]
Default: 5
Testing: FIXED - Keep consistent splitting
subsample
Description: Subsample ratio for gradient boosting learners - fixed for consistency
Range: [0.6, 0.8, 1.0]
Default: 0.8
Testing: FIXED - Keep consistent subsampling
min_samples_leaf
Description: Minimum samples per leaf for tree-based learners - fixed for consistency
Range: [1, 2, 5, 10]
Default: 2
Testing: FIXED - Keep consistent leaf size
fusion_type
Description: Fusion strategy for multi-modality learners - fixed for consistency
Range: ['weighted', 'attention', 'concatenation']
Default: 'weighted'
Testing: FIXED - Keep consistent fusion
cross_modal_learning
Description: Enable cross-modal learning for fusion learners - fixed as True for consistency
Range: [True, False]
Default: True
Testing: FIXED - Always enable cross-modal learning
hidden_dim
Description: Hidden dimension for neural network learners - fixed for consistency
Range: [128, 256, 512, 1024]
Default: 256
Testing: FIXED - Keep consistent hidden dimension
batch_norm
Description: Enable batch normalization for neural network learners - fixed as True for consistency
Range: [True, False]
Default: True
Testing: FIXED - Always enable batch normalization
layers
Description: Number of layers for neural network learners - fixed for consistency
Range: [2, 3, 4, 5]
Default: 3
Testing: FIXED - Keep consistent layer count
fusion_layers
Description: Number of fusion layers for multi-modality learners - fixed for consistency
Range: [1, 2, 3, 4]
Default: 2
Testing: FIXED - Keep consistent fusion layers

**Stage 4:
🎯 VARY HYPERPARAMETERS (7 total) - **OPTIMIZED FOR IMBALANCED DATA**:
learning_rate
Description: Learning rate for optimization - controls convergence speed and final performance, optimized for imbalanced datasets
Range: [1e-4, 5e-4, 1e-3, 2e-3]
Default: 1e-3
Testing: VARY - Critical for convergence speed and stability on imbalanced data
Range: [1e-4, 5e-4, 1e-3, 2e-3] (4 parameters) - Moderate rates for better gradient estimates on minority classes
epochs
Description: Number of training epochs - determines training completeness, increased for imbalanced data learning
Range: [50, 75, 100, 150]
Default: 75
Testing: VARY - Critical for training duration, more epochs needed for minority class learning
Range: [50, 75, 100, 150] (4 parameters) - Higher epoch counts for imbalanced data convergence
denoising_weight
Description: Weight for denoising loss - balances primary loss vs cross-modal denoising loss, higher for imbalanced data
Range: [0.1, 0.15, 0.2, 0.25]
Default: 0.15
Testing: VARY - Core novel feature parameter, higher weights help with minority class representation
Range: [0.1, 0.15, 0.2, 0.25] (4 parameters) - Higher denoising weights for better minority class features
batch_size
Description: Batch size for training - affects training stability, smaller batches better for imbalanced data
Range: [16, 32, 64]
Default: 32
Testing: VARY - Smaller batches provide better gradient estimates for minority classes
Range: [16, 32, 64] (3 parameters) - Smaller batch sizes for better imbalanced data handling
weight_decay
Description: L2 regularization strength - prevents overfitting and improves generalization
Range: [1e-5, 1e-4, 1e-3]
Default: 1e-4
Testing: VARY - Important for preventing overfitting, range maintained for imbalanced data
Range: [1e-5, 1e-4, 1e-3] (3 parameters) - Consistent regularization range
early_stopping_patience
Description: Epochs to wait before early stopping - increased patience for imbalanced data convergence
Range: [10, 15, 20, 25]
Default: 15
Testing: VARY - More patience needed for minority class learning convergence
Range: [10, 15, 20, 25] (4 parameters) - Higher patience for imbalanced data training
label_smoothing
Description: Label smoothing factor - reduces overconfidence on majority class, critical for imbalanced data
Range: [0.0, 0.1, 0.2]
Default: 0.1
Testing: VARY - NEW PARAMETER for imbalanced data handling
Range: [0.0, 0.1, 0.2] (3 parameters) - Prevents overconfidence on majority class
🎯 FIXED HYPERPARAMETERS (39 total):
task_type
Description: Type of learning task - automatically determined from dataset
Range: ['classification', 'regression']
Default: 'classification'
Testing: FIXED - Determined by dataset characteristics
mixed_precision
Description: Enable mixed precision training - improves training speed and memory efficiency
Range: [True, False]
Default: True
Testing: FIXED - Performance optimization
gradient_accumulation_steps
Description: Steps for gradient accumulation - enables larger effective batch sizes
Range: [1, 2, 4, 8]
Default: 1
Testing: FIXED - Memory optimization
num_workers
Description: Number of data loading workers - improves data loading performance
Range: [2, 4, 8, 16]
Default: 4
Testing: FIXED - Performance optimization
validation_split
Description: Fraction of data for validation - standard validation split
Range: [0.1, 0.15, 0.2, 0.25]
Default: 0.2
Testing: FIXED - Standard validation split
cross_validation_folds
Description: Number of cross-validation folds - standard CV setup
Range: [3, 5, 10]
Default: 5
Testing: FIXED - Standard CV setup
gradient_clipping
Description: Gradient clipping threshold - prevents gradient explosion
Range: [0.5, 1.0, 2.0, 5.0]
Default: 1.0
Testing: FIXED - Prevents gradient explosion
optimizer_type
Description: Type of optimizer - AdamW for stable optimization
Range: ['adamw', 'sgd', 'adam']
Default: 'adamw'
Testing: FIXED - Stable optimization
scheduler_type
Description: Type of learning rate scheduler - cosine restarts for better convergence
Range: ['cosine_restarts', 'onecycle', 'plateau']
Default: 'cosine_restarts'
Testing: FIXED - Optimal learning rate strategy
label_smoothing
Description: Label smoothing factor - improves generalization
Range: [0.0, 0.1, 0.2, 0.3]
Default: 0.1
Testing: FIXED - Improves generalization
dropout_rate
Description: Dropout rate for regularization - prevents overfitting
Range: [0.1, 0.2, 0.3, 0.4]
Default: 0.2
Testing: FIXED - Prevents overfitting
enable_denoising
Description: Enable cross-modal denoising - novel feature for improved representation learning
Range: [True, False]
Default: True
Testing: FIXED - Novel feature enabled
denoising_strategy
Description: Denoising strategy - adaptive strategy for optimal denoising
Range: ['adaptive', 'fixed', 'progressive']
Default: 'adaptive'
Testing: FIXED - Optimal denoising approach
denoising_objectives
Description: Denoising objectives - reconstruction and alignment for comprehensive denoising
Range: [['reconstruction'], ['alignment'], ['consistency'], ['reconstruction', 'alignment']]
Default: ['reconstruction', 'alignment']
Testing: FIXED - Comprehensive denoising objectives
denoising_modalities
Description: Modalities to apply denoising to - all modalities for comprehensive denoising
Range: [['text'], ['metadata'], ['text', 'metadata']]
Default: ['text', 'metadata']
Testing: FIXED - All modalities
gradient_scaling
Description: Enable gradient scaling - stability optimization for mixed precision
Range: [True, False]
Default: True
Testing: FIXED - Stability optimization
loss_scale
Description: Loss scaling strategy - dynamic scaling for mixed precision
Range: ['dynamic', 'fixed']
Default: 'dynamic'
Testing: FIXED - Mixed precision optimization
eval_interval
Description: Epochs between evaluations - frequent evaluation for monitoring
Range: [1, 2, 5, 10]
Default: 1
Testing: FIXED - Evaluation frequency
save_checkpoints
Description: Save model checkpoints - model preservation for recovery
Range: [True, False]
Default: True
Testing: FIXED - Model preservation
modal_specific_tracking
Description: Enable modal-specific tracking - novel feature for detailed modal analysis
Range: [True, False]
Default: True
Testing: FIXED - Novel feature enabled
track_modal_reconstruction
Description: Track reconstruction metrics - detailed modal reconstruction tracking
Range: [True, False]
Default: True
Testing: FIXED - Detailed tracking enabled
track_modal_alignment
Description: Track alignment metrics - detailed modal alignment tracking
Range: [True, False]
Default: True
Testing: FIXED - Detailed tracking enabled
track_modal_consistency
Description: Track consistency metrics - detailed modal consistency tracking
Range: [True, False]
Default: True
Testing: FIXED - Detailed tracking enabled
modal_tracking_frequency
Description: Frequency of modal tracking - maximum granularity for detailed analysis
Range: ['every_epoch', 'every_5_epochs', 'every_10_epochs']
Default: 'every_epoch'
Testing: FIXED - Maximum tracking granularity
log_interval
Description: Epochs between logging - standard logging frequency
Range: [5, 10, 20, 50]
Default: 10
Testing: FIXED - Logging frequency
profile_training
Description: Enable training profiling - performance analysis
Range: [True, False]
Default: False
Testing: FIXED - Performance analysis
preserve_bag_characteristics
Description: Enable bag characteristics preservation - novel feature for interpretability
Range: [True, False]
Default: True
Testing: FIXED - Novel feature enabled
save_modality_mask
Description: Save modality masks - preserve modality information
Range: [True, False]
Default: True
Testing: FIXED - Preservation enabled
save_modality_weights
Description: Save modality weights - preserve modality importance
Range: [True, False]
Default: True
Testing: FIXED - Preservation enabled
save_bag_id
Description: Save bag identifiers - preserve bag identification
Range: [True, False]
Default: True
Testing: FIXED - Preservation enabled
save_training_metrics
Description: Save training metrics - preserve training history
Range: [True, False]
Default: True
Testing: FIXED - Preservation enabled
save_learner_config
Description: Save learner configurations - preserve learner settings
Range: [True, False]
Default: True
Testing: FIXED - Preservation enabled
track_only_primary_modalities
Description: Track only primary modalities - comprehensive tracking scope
Range: [True, False]
Default: False
Testing: FIXED - Comprehensive tracking
preserve_only_primary_modalities
Description: Preserve only primary modalities - comprehensive preservation scope
Range: [True, False]
Default: False
Testing: FIXED - Comprehensive preservation
enable_cross_validation
Description: Enable cross-validation - validation strategy
Range: [True, False]
Default: True
Testing: FIXED - Validation strategy
cv_folds
Description: Number of cross-validation folds - CV configuration
Range: [3, 5, 10]
Default: 5
Testing: FIXED - CV configuration
verbose
Description: Enable verbose logging - logging control
Range: [True, False]
Default: True
Testing: FIXED - Logging control
distributed_training
Description: Enable distributed training - training strategy
Range: [True, False]
Default: True
Testing: FIXED - Training strategy
compile_model
Description: Compile model for optimization - performance optimization
Range: [True, False]
Default: True
Testing: FIXED - Performance optimization

**Stage 5:
VARY HYPERPARAMETERS (3 total):
🎯 NOVEL FEATURE - TRANSFORMER ARCHITECTURE
transformer_num_heads
Description: Number of attention heads in transformer - controls attention diversity and capacity
Range: [4, 8, 12]
Default: 8
Testing: VARY - High impact, core transformer fusion innovation
Range: [8, 12] (2 parameters) - Removed 4 (too low for effective attention)
transformer_hidden_dim
Description: Hidden dimension size in transformer - determines model capacity and representational power
Range: [128, 256, 512]
Default: 256
Testing: VARY - High impact, core transformer fusion innovation
Range: [256, 512] (2 parameters) - Removed 128 (too low for effective capacity)
HIGH IMPACT - UNCERTAINTY QUANTIFICATION
uncertainty_method
Description: Method for estimating prediction uncertainty - critical for model reliability and interpretability
Range: ['entropy', 'variance', 'confidence']
Default: 'entropy'
Testing: VARY - High impact, advanced uncertainty quantification innovation
Range: ['entropy', 'variance', 'confidence'] (3 parameters)
🔧 FIXED HYPERPARAMETERS (6 total):
🎯 CORE CONFIGURATION
aggregation_strategy
Description: Method for combining individual learner predictions - transformer fusion is the novel innovation
Range: ['simple_average', 'weighted_average', 'transformer_fusion']
Default: 'transformer_fusion'
Testing: FIXED - Novel feature, keep enabled for optimization
task_type
Description: Type of machine learning task - automatically determined from dataset
Range: ['classification', 'regression']
Default: 'classification'
Testing: FIXED - Determined by dataset characteristics
TRANSFORMER ARCHITECTURE (Fixed Components)
transformer_token_dim
Description: Token dimension for transformer input - consistent input dimension
Range: [32, 64, 128, 256]
Default: 64
Testing: FIXED - Keep consistent for stability
transformer_num_layers
Description: Number of transformer encoder layers - sufficient depth for most cases
Range: [1, 2, 3, 4, 6]
Default: 2
Testing: FIXED - Sufficient depth, more layers = diminishing returns
transformer_temperature
Description: Temperature scaling for attention weights - standard temperature
Range: [0.5, 1.0, 1.5, 2.0, 3.0]
Default: 1.0
Testing: FIXED - Standard temperature, less critical than architecture
⚙️ ADVANCED CONFIGURATION (Fixed Components)
confidence_threshold
Description: Minimum confidence threshold for predictions - standard threshold
Range: [0.1, 0.3, 0.5, 0.7, 0.9]
Default: 0.5
Testing: FIXED - Standard threshold, less impact than core parameters
weight_normalization
Description: Method for normalizing ensemble weights - standard softmax normalization
Range: ['softmax', 'l1', 'l2', 'none']
Default: 'softmax'
Testing: FIXED - Standard method, less impact than core parameters
modality_importance_weight
Description: Weight for modality importance in aggregation - balanced importance
Range: [0.1, 0.3, 0.5, 0.7, 0.9]
Default: 0.5
Testing: FIXED - Balanced weight, less critical than architecture

**Select best hyperparameter combination trial based on the following Selection Criteria**:

### **Classification Balanced Dataset**:
1. **Primary metric**: Accuracy (higher is better)
2. **Secondary metrics**: Balanced Accuracy, Weighted F1-Score, Weighted Precision, Weighted Recall
3. **Efficiency metrics**: Training time, prediction time, memory usage
4. **Robustness**: Cross-validation stability (low variance across folds)

### **Classification Unbalanced Dataset**:
1. **Primary metric**: **Balanced Accuracy** (higher is better) - *Critical for unbalanced datasets*
2. **Secondary metrics**: Weighted F1-Score, Weighted Precision, Weighted Recall, Accuracy
3. **Efficiency metrics**: Training time, prediction time, memory usage
4. **Robustness**: Cross-validation stability (low variance across folds)

1. **Primary metric**: Mean Squared Error (MSE) - *lower is better*
2. **Secondary metrics**: Mean Absolute Error (MAE), R² Score, Root Mean Squared Error (RMSE)
3. **Efficiency metrics**: Training time, prediction time, memory usage
4. **Robustness**: Cross-validation stability (low variance across folds)

### **Regression Unbalanced Dataset**:
1. **Primary metric**: **Mean Absolute Error (MAE)** - *lower is better, more robust to outliers*
2. **Secondary metrics**: Root Mean Squared Error (RMSE), R² Score, Mean Squared Error (MSE)
3. **Efficiency metrics**: Training time, prediction time, memory usage
4. **Robustness**: Cross-validation stability (low variance across folds)

### **Selection Algorithm**:
1. **Filter trials**: Remove trials with accuracy < 0.5 (classification) or MSE > baseline*2 (regression)
2. **Rank by primary metric**: Sort trials by primary metric (descending for accuracy, ascending for MSE/MAE)
3. **Tie-breaking**: Use secondary metrics in order of importance
4. **Efficiency check**: Ensure training time < 2x average time across all trials
5. **Robustness check**: Ensure CV stability (std/mean) < 0.2 for final selection

**Main MainModel Run**:
Use the best hyperparameter trial chosen through the selection criteria and use its parameter for each hyperparameter to train and predict using the full sampled dataset from phase 1(quick test - 10k samples and full test 300k samples)

**Outputs**:
- Hyperparameter trial results (`mainmodel_trials.json`)
- Best configuration summary (`mainmodel_best.json`)
- Optimization history and convergence analysis

**📊 OPTIMIZED HYPERPARAMETER SPACE FOR IMBALANCED DATA**:
Based on analysis of imbalanced dataset performance (Amazon Reviews), the hyperparameter space has been specifically optimized to handle class imbalance:

**Stage 2 (Ensemble Generation)**: 3 × 3 × 3 = 27 combinations
- `n_bags`: [15, 20, 25] (removed 10 - too small)
- `max_dropout_rate`: [0.4, 0.5, 0.6] (removed 0.3 - too low)
- `sample_ratio`: [0.8, 0.9, 1.0] (removed 0.7 - too low)

**Stage 3 (Base Learner Selection)**: 2 × 4 = 8 combinations
- `optimization_mode`: ['accuracy', 'performance'] (removed 'efficiency' - causes low accuracy)
- `learner_type`: ['sklearn', 'pytorch', 'deep_learning', 'transformer'] (NEW - choose learner implementation)

**Stage 4 (Training Pipeline - OPTIMIZED FOR IMBALANCED DATA)**: 4 × 4 × 4 × 3 × 3 × 4 × 3 = 13,824 combinations
- `learning_rate`: [1e-4, 5e-4, 1e-3, 2e-3] (moderate rates for minority classes)
- `epochs`: [50, 75, 100, 150] (more epochs for minority class learning)
- `denoising_weight`: [0.1, 0.15, 0.2, 0.25] (higher weights for minority class features)
- `batch_size`: [16, 32, 64] (smaller batches for better gradients)
- `weight_decay`: [1e-5, 1e-4, 1e-3] (maintained range)
- `early_stopping_patience`: [10, 15, 20, 25] (more patience for convergence)
- `label_smoothing`: [0.0, 0.1, 0.2] (NEW - prevents majority class overconfidence)

**Stage 5 (Ensemble Prediction)**: 2 × 2 × 3 = 12 combinations
- `transformer_num_heads`: [8, 12] (removed 4 - too low)
- `transformer_hidden_dim`: [256, 512] (removed 128 - too low)
- `uncertainty_method`: ['entropy', 'variance', 'confidence'] (unchanged)

**Total Optimized Combinations**: 27 × 8 × 13,824 × 12 = **35,880,960 combinations**

**🚨 Key Imbalanced Data Optimizations**:
- **Automatic class weighting** in loss function (inverse frequency)
- **Feature normalization** enabled for proper TF-IDF + metadata scaling
- **Label smoothing** parameter added to prevent majority class bias

**🚀 NEW: Advanced Learner Type Selection Benefits**:
- **Dataset-adaptive learning**: Choose sklearn for tabular/text data, PyTorch for images/sequences, deep learning for complex patterns, transformers for state-of-the-art performance
- **Performance optimization**: Sklearn learners achieve 77.8% accuracy vs PyTorch 45.2% on Amazon Reviews
- **Advanced architectures**: Deep learning learners with LSTM/CNN/MLP for modality-specific processing
- **State-of-the-art transformers**: BERT, Vision Transformers, and cross-modal transformers for cutting-edge performance
- **Modality awareness**: Each learner type has sophisticated modality-aware selection and fusion capabilities
- **Universal applicability**: Works across different dataset types and modalities with specialized architectures
- **Quadrupled search space** from 8.97M to 35.88M combinations for comprehensive optimization

**📈 PERFORMANCE ACHIEVEMENTS**:
- **Current accuracy**: 78.0% (target: 80-90%)
- **Gap to target**: Only 2.0 percentage points to reach 80%
- **Improvement from baseline**: +30-35% accuracy improvement from initial ~45% performance
- **Ensemble optimization**: 8 bags, 0.0 dropout rate, 0.8 sample ratio for optimal performance
**Optuna Trials**: 75 (quick test), 300 (full test)

---

### **Phase 4: Ablation Studies**
**Purpose**: Quantify the contribution of each novel component by using the best hypermodel combination from phase 3 and use the individual mainModel pipeline files rather than the API for testing

**Ablation Tests**:

**Stage 2:
🎯 ABLATION STUDY 1: Adaptive Strategy vs Traditional Methods
Purpose: Prove your novel 'adaptive' dropout strategy is superior to traditional methods
Comparison:
Your Novel Method: dropout_strategy='adaptive' (importance-based + distinct combinations)
Traditional Methods: dropout_strategy='linear', 'exponential', 'random'
What This Tests: Your core innovation of intelligent, importance-based modality dropout
Expected Result: Adaptive should show better ensemble diversity and prediction accuracy
🎯 ABLATION STUDY 2: Modality Dropout vs Traditional Bootstrap
Purpose: Prove that modality dropout (your approach) is more effective than traditional bootstrap sampling alone
Comparison:
Your Novel Method: Full modality dropout system (max_dropout_rate=0.5)
Traditional Method: No modality dropout (max_dropout_rate=0.0) - only bootstrap sampling
What This Tests: Your core innovation of strategic modality removal for ensemble diversity
Expected Result: Modality dropout should improve multimodal learning and ensemble diversity
🎯 ABLATION STUDY 3: Feature-Level Sampling Effectiveness
Purpose: Validate your hierarchical feature sampling within modalities
Comparison:
Your Novel Method: feature_sampling_ratio=0.8 (feature-level sampling)
Traditional Method: feature_sampling_ratio=1.0 (no feature sampling)
What This Tests: Your innovation of multi-level sampling (modality + feature level)
Expected Result: Feature sampling should improve diversity without significant accuracy loss

**Stage 3:
�� ABLATION STUDY 1: MODALITY-AWARE SELECTION
Novel Feature: Modality-aware learner selection that chooses different learner types based on modality combinations
Parameters to Vary:
modality_aware: [True, False]
optimization_mode: ['accuracy', 'performance', 'efficiency'] (fixed)
Test Scenarios:
Modality-Aware ON: Learners selected based on modality combinations (text→TextLearner, image→ImageLearner, etc.)
Modality-Aware OFF: Fixed learner selection regardless of modality combinations
Research Questions:
Does modality-aware selection improve performance over fixed selection?
How does modality-aware selection affect ensemble diversity?
Which modality combinations benefit most from specialized learners?
Expected Impact: High - Tests the core novel feature of Stage 3
🎯 ABLATION STUDY 2: LEARNER TYPE SPECIALIZATION
Novel Feature: Specialized learners for different modality types (TextLearner, ImageLearner, TabularLearner, FusionLearner)
Parameters to Vary:
optimization_mode: ['accuracy', 'performance', 'efficiency']
modality_aware: [True, False]
Test Scenarios:
Specialized + Accuracy: [True, 'accuracy']
Specialized + Performance: [True, 'performance']
Specialized + Efficiency: [True, 'efficiency']
Fixed + Accuracy: [False, 'accuracy']
Fixed + Performance: [False, 'performance']
Research Questions:
Do specialized learners outperform generic learners?
Which optimization mode works best with specialized learners?
How does specialization affect training time vs accuracy?
Expected Impact: High - Tests the core learner specialization feature
🎯 ABLATION STUDY 3: MODALITY WEIGHTAGE ANALYSIS
Novel Feature: Dynamic modality importance calculation using feature ratio, variance, and dimensionality weights
Parameters to Vary:
feature_ratio_weight: [0.2, 0.3, 0.4, 0.5, 0.6]
variance_weight: [0.2, 0.3, 0.4, 0.5, 0.6]
dimensionality_weight: [0.2, 0.3, 0.4, 0.5, 0.6]
Test Scenarios:
Balanced Weights: [0.4, 0.3, 0.3] (default)
Feature-Ratio Dominant: [0.6, 0.2, 0.2]
Variance Dominant: [0.2, 0.6, 0.2]
Dimensionality Dominant: [0.2, 0.2, 0.6]
Equal Weights: [0.33, 0.33, 0.34]
Research Questions:
Which modality characteristics are most important for learner selection?
How does weightage balance affect ensemble performance?
Do different datasets favor different weightage strategies?
Expected Impact: Medium-High - Tests the modality analysis algorithm
🎯 ABLATION STUDY 4: BAG-LEARNER PAIRING STRATEGIES
Novel Feature: Different pairing optimization strategies for bag-learner assignments
Parameters to Vary:
pairing_focus: ['performance', 'diversity', 'efficiency']
metadata_level: ['minimal', 'complete', 'enhanced']
Test Scenarios:
Performance-Focused: ['performance', 'complete']
Diversity-Focused: ['diversity', 'complete']
Efficiency-Focused: ['efficiency', 'complete']
Minimal Metadata: ['performance', 'minimal']
Enhanced Metadata: ['performance', 'enhanced']
Research Questions:
Which pairing strategy produces the best ensemble performance?
How does metadata level affect pairing quality?
Is there a trade-off between pairing strategies?
Expected Impact: Medium - Tests the pairing optimization

**Stage 4:
🎯 ABLATION STUDY 1: CROSS-MODAL DENOISING SYSTEM
Purpose: Test the effectiveness of the cross-modal denoising system
Novel Feature: Cross-modal denoising for improved representation learning
Method: Compare training with and without denoising, different denoising strategies
Algorithm:
Baseline: Standard training without denoising
Variant 1: Fixed denoising strategy
Variant 2: Adaptive denoising strategy
Variant 3: Curriculum denoising strategy
Output: Performance comparison, denoising effectiveness scores
Usage: run_stage4_ablation_study('cross_modal_denoising')
Question Answered: Does cross-modal denoising improve ensemble performance?
🎯 ABLATION STUDY 2: MODAL-SPECIFIC METRICS TRACKING
Purpose: Test the impact of detailed modal-specific metrics tracking
Novel Feature: Modal-specific metrics tracking for interpretability
Method: Compare different tracking frequencies and granularity levels
Algorithm:
Baseline: No modal-specific tracking
Variant 1: Track every epoch
Variant 2: Track every 5 epochs
Variant 3: Track only primary modalities
Output: Tracking effectiveness, interpretability scores, performance impact
Usage: run_stage4_ablation_study('modal_metrics_tracking')
Question Answered: Does modal-specific tracking provide valuable insights without performance cost?
🎯 ABLATION STUDY 3: BAG CHARACTERISTICS PRESERVATION
Purpose: Test the value of preserving bag characteristics during training
Novel Feature: Bag characteristics preservation for interpretability
Method: Compare training with different preservation levels
Algorithm:
Baseline: No bag characteristics preservation
Variant 1: Preserve only modality masks
Variant 2: Preserve modality masks and weights
Variant 3: Preserve all bag characteristics
Output: Preservation effectiveness, interpretability scores, memory usage
Usage: run_stage4_ablation_study('bag_characteristics_preservation')
Question Answered: Does preserving bag characteristics enhance interpretability without performance degradation?
🎯 ABLATION STUDY 4: DENOISING OBJECTIVE COMBINATIONS
Purpose: Test different combinations of denoising objectives
Novel Feature: Multi-objective denoising system
Method: Compare different objective combinations
Algorithm:
Baseline: No denoising
Variant 1: Reconstruction only
Variant 2: Alignment only
Variant 3: Consistency only
Variant 4: Reconstruction + Alignment
Variant 5: All objectives combined
Output: Objective effectiveness scores, performance impact analysis
Usage: run_stage4_ablation_study('denoising_objective_combinations')
Question Answered: Which denoising objectives provide the most benefit?
🎯 ABLATION STUDY 5: TRAINING STRATEGY COMPARISON
Purpose: Test different training strategies for ensemble optimization
Novel Feature: Adaptive training strategies
Method: Compare different optimization approaches
Algorithm:
Baseline: Standard training with fixed parameters
Variant 1: Adaptive denoising weights
Variant 2: Curriculum learning with progressive denoising
Variant 3: Multi-task training with denoising objectives
Output: Strategy effectiveness, convergence analysis, performance comparison
Usage: run_stage4_ablation_study('training_strategy_comparison')
Question Answered: Which training strategy provides the best ensemble performance?
🎯 ABLATION STUDY 6: MODALITY-SPECIFIC DENOISING
Purpose: Test denoising effectiveness on different modality combinations
Novel Feature: Modality-aware denoising
Method: Compare denoising on different modality subsets
Algorithm:
Baseline: No denoising
Variant 1: Denoise text modality only
Variant 2: Denoise metadata modality only
Variant 3: Denoise both modalities
Variant 4: Adaptive modality selection for denoising
Output: Modality-specific denoising effectiveness, cross-modal learning scores
Usage: run_stage4_ablation_study('modality_specific_denoising')
Question Answered: Which modalities benefit most from denoising?

**Stage 5:
✅ ABLATION STUDY 1: BAG RECONSTRUCTION SYSTEM
Purpose: Test the effectiveness of the bag reconstruction novel feature
Novel Feature: Bag reconstruction for faithful training bag representation
Method: Compare prediction with and without bag reconstruction
Algorithm:
Baseline: Direct prediction on full test data (no bag reconstruction)
Variant 1: With bag reconstruction (faithful reconstruction of training bags)
Output: Bag reconstruction effectiveness, accuracy improvement, bag fidelity scores
Usage: run_bag_reconstruction_ablation(test_data, test_labels)
Question Answered: Does faithful bag reconstruction improve prediction accuracy?
✅ ABLATION STUDY 2: UNCERTAINTY METHOD COMPARISON
Purpose: Test the effectiveness of different uncertainty quantification methods
Novel Feature: Multi-method uncertainty quantification system
Method: Compare different uncertainty calculation strategies
Algorithm:
Baseline: Entropy-based uncertainty
Variant 1: Variance-based uncertainty
Variant 2: Confidence-based uncertainty
Output: Uncertainty calibration scores, prediction reliability, method effectiveness
Usage: run_uncertainty_method_ablation(test_data, test_labels)
Question Answered: Which uncertainty method provides best calibration and reliability?
✅ ABLATION STUDY 3: TRANSFORMER ARCHITECTURE OPTIMIZATION
Purpose: Test different transformer architectures for modality-aware aggregation
Novel Feature: Modality-aware transformer aggregation system
Method: Compare different transformer configurations
Algorithm:
Baseline: Small transformer (4 heads, 128 hidden dim)
Variant 1: Medium transformer (8 heads, 256 hidden dim)
Variant 2: Large transformer (12 heads, 512 hidden dim)
Output: Architecture effectiveness, modality awareness scores, attention diversity
Usage: run_transformer_architecture_ablation(test_data, test_labels)
Question Answered: What transformer architecture provides optimal modality-aware aggregation?
✅ ABLATION STUDY 4: AGGREGATION STRATEGY COMPARISON
Purpose: Test different ensemble aggregation strategies
Novel Feature: Multi-strategy ensemble aggregation system
Method: Compare different aggregation approaches
Algorithm:
Baseline: Simple averaging
Variant 1: Weighted averaging
Variant 2: Transformer fusion
Variant 3: Attention-based fusion
Output: Strategy effectiveness, aggregation quality, performance comparison
Usage: run_stage5_ablation_study(test_data, test_labels, ablation_configs)
Question Answered: Which aggregation strategy provides the best ensemble performance?

---

### **Phase 5: Interpretability Studies**
**Purpose**: Understand model decisions and provide actionable insights based on the best hyperparameter combination result from phase 3 on the respective dataset

**Interpretability Tests**:

**Stage 2:
🎯 MODALITY IMPORTANCE ANALYSIS
1. Modality Importance Computation
What it does: Computes predictive importance scores for each modality using variance-based analysis
Method: _compute_modality_importance()
Algorithm:
For multi-dimensional data: np.mean(np.var(data, axis=0)) (mean variance across features)
For 1D data: np.var(data) (direct variance)
Higher variance = higher importance score
Output: Dictionary {'text': 0.85, 'metadata': 0.42} (importance scores)
Usage: Drives adaptive dropout strategy decisions, determines which modalities to prioritize
Question it answers: "Which modalities are most predictive and should be preserved in more bags?"
2. Modality Usage Statistics
What it does: Tracks how frequently each modality appears across all generated bags
Method: _collect_interpretability_data() → modality usage calculation
Algorithm:
Counts active vs total occurrences per modality across all bags
Calculates usage ratio: active_count / total_count
Output: Dictionary {'text': 0.8, 'metadata': 0.6} (80% of bags use text, 60% use metadata)
Usage: Validates adaptive strategy effectiveness, compares different dropout strategies
Question it answers: "How well does the adaptive strategy balance modality usage across the ensemble?"
🎯 ENSEMBLE DIVERSITY ANALYSIS
3. Ensemble-Level Statistics
What it does: Provides comprehensive statistics about the entire ensemble's dropout behavior
Method: get_ensemble_statistics()
Algorithm:
Calculates mean, std, min, max of dropout rates across all bags
Tracks total bags and strategy used
Output: Dictionary with mean_dropout_rate, std_dropout_rate, min_dropout_rate, max_dropout_rate, total_bags, strategy
Usage: Validates ensemble diversity, compares different strategies' effectiveness
Question it answers: "How diverse is the ensemble in terms of dropout rates and what strategy was used?"
4. Bag-Level Detailed Information
What it does: Provides complete configuration details for each individual bag
Method: get_bag_info(bag_id) or _collect_interpretability_data() → detailed_bags
Algorithm:
Extracts all BagConfig attributes for each bag
Includes modality masks, feature masks, dropout rates, sample counts, timestamps
Output: List of dictionaries with complete bag configurations
Usage: Individual bag analysis, debugging, understanding bag generation patterns
Question it answers: "What is the exact configuration of each bag and how do they differ?"
🎯 FEATURE SELECTION ANALYSIS
5. Feature Selection Statistics
What it does: Analyzes how features are sampled within each modality across all bags
Method: get_feature_statistics()
Algorithm:
For each modality, tracks feature selection ratios across all bags
Calculates statistical summaries: mean, std, min, max selection ratios
Counts selected features per bag per modality
Output: Dictionary with per-modality feature selection statistics
Usage: Validates feature-level diversity, optimizes feature sampling ratios
Question it answers: "How effectively does feature sampling create diversity within each modality?"
🎯 COMPREHENSIVE ANALYSIS METHODS
6. Complete Interpretability Data
What it does: Provides access to all interpretability data in one comprehensive dataset
Method: get_interpretability_data()
Algorithm:
Combines all interpretability data: modality importance, ensemble stats, bag details, feature stats
Returns deep copy of complete interpretability dataset
Output: Complete dictionary with all interpretability information
Usage: Full ensemble analysis, research insights, comprehensive reporting
Question it answers: "What is the complete picture of how the ensemble was generated and how diverse is it?"
7. Modality Importance Access
What it does: Provides quick access to modality importance scores only
Method: get_modality_importance()
Algorithm:
Extracts modality importance from interpretability data
Returns copy of importance scores
Output: Dictionary with modality importance scores
Usage: Quick modality ranking analysis, adaptive strategy validation
Question it answers: "Which modalities are most important according to the adaptive strategy?"
8. Feature Statistics Access
What it does: Provides quick access to feature selection statistics only
Method: get_feature_statistics()
Algorithm:
Extracts feature statistics from interpretability data
Returns copy of feature selection statistics
Output: Dictionary with feature selection statistics
Usage: Feature-level diversity analysis, feature sampling optimization
Question it answers: "How effectively does feature sampling create diversity within modalities?"
9. Ensemble Statistics Access
What it does: Provides quick access to ensemble-level statistics only
Method: get_ensemble_statistics()
Algorithm:
Extracts ensemble statistics from interpretability data
Returns copy of ensemble-level statistics
Output: Dictionary with ensemble-level statistics
Usage: Overall ensemble diversity analysis, strategy comparison
Question it answers: "What are the overall diversity characteristics of the generated ensemble?"
10. Individual Bag Analysis
What it does: Provides detailed information about a specific bag
Method: get_bag_info(bag_id)
Algorithm:
Retrieves specific bag from bags list
Extracts all bag configuration details
Output: Dictionary with specific bag configuration
Usage: Individual bag debugging, specific bag analysis
Question it answers: "What is the exact configuration of bag X and how does it contribute to ensemble diversity?"

**Stage 3:
🎯 BAG-LEARNER ASSIGNMENT ANALYSIS
1. Bag-Learner Summary
What it does: Provides comprehensive summary of bag-learner assignments across the ensemble
Method: get_bag_learner_summary()
Algorithm:
Counts learner types across all bags
Tracks optimization modes and task types
Calculates average expected performance and performance range
Aggregates all bag-learner configurations
Output: Dictionary with total_bags, learner_type_distribution, optimization_mode, task_type, average_expected_performance, performance_range, bag_learner_configs
Usage: Overall ensemble analysis, learner type distribution validation, performance expectation analysis
Question it answers: "What is the overall distribution of learner types and expected performance across the ensemble?"
2. Individual Bag Configuration
What it does: Provides detailed configuration for a specific bag-learner pair
Method: get_bag_learner_config(bag_id)
Algorithm:
Searches through bag_learner_configs for specific bag_id
Returns complete BagLearnerConfig object
Includes all configuration details for that specific bag
Output: BagLearnerConfig object with bag_id, learner_type, optimization_mode, modality_weights, learner_config, expected_performance, modality_mask, feature_mask, bag_data
Usage: Individual bag analysis, debugging specific bag-learner pairs, detailed configuration inspection
Question it answers: "What is the exact configuration of bag X and which learner was assigned to it?"
🎯 PAIRING QUALITY ANALYSIS
3. Bag-Learner Pairing Statistics
What it does: Analyzes the quality and effectiveness of bag-learner pairings
Method: get_pairing_statistics()
Algorithm:
Calculates modality-learner match rate (how well learners match their assigned modalities)
Tracks performance predictions and consistency scores
Groups learners by type and calculates consistency within groups
Measures pairing quality based on modality patterns
Output: Dictionary with total_pairs, modality_learner_match_rate, average_expected_performance, performance_std, pairing_consistency, learner_type_distribution, pairing_focus, metadata_level
Usage: Validates pairing strategy effectiveness, measures pairing quality, compares different pairing approaches
Question it answers: "How well do the bag-learner pairings match and how consistent are similar pairings?"
🎯 METADATA COMPLETENESS ANALYSIS
4. Metadata Completeness Statistics
What it does: Analyzes the completeness and quality of metadata stored for each bag-learner configuration
Method: get_metadata_completeness()
Algorithm:
Checks completeness based on metadata_level (minimal, complete, enhanced)
Scores each configuration based on available fields
Calculates average completeness and standard deviation
Counts configurations with high completeness scores
Output: Dictionary with total_configs, average_completeness, completeness_std, metadata_level, min_completeness, max_completeness, complete_configs
Usage: Validates metadata storage quality, optimizes metadata_level settings, ensures comprehensive configuration tracking
Question it answers: "How complete and comprehensive is the metadata stored for each bag-learner configuration?"
🎯 ENSEMBLE COHERENCE ANALYSIS
5. Ensemble Coherence Statistics
What it does: Analyzes the coherence and consistency of the entire ensemble's bag-learner assignments
Method: get_ensemble_coherence()
Algorithm:
Calculates learner type diversity and entropy
Measures optimization mode consistency
Analyzes performance coherence across similar bags
Evaluates relationship coherence between modality patterns and learner types
Output: Dictionary with total_bags, learner_type_diversity, learner_type_entropy, optimization_mode_consistency, performance_coherence, relationship_coherence, overall_coherence
Usage: Validates ensemble consistency, measures coherence quality, ensures balanced learner distribution
Question it answers: "How coherent and consistent is the ensemble in terms of learner assignments and performance expectations?"
�� ADVANCED INTERPRETABILITY TESTING
6. Modality Importance Interpretability Test
What it does: Tests how different weightage parameters affect modality importance in learner selection
Method: run_stage3_interpretability_test('modality_importance', test_scenarios)
Algorithm:
Creates test selectors with different weightage parameters
Compares modality importance calculations
Analyzes impact on learner selection decisions
Measures weightage parameter sensitivity
Output: Dictionary with test results, weightage comparisons, selection impact analysis
Usage: Validates modality importance calculation, optimizes weightage parameters, tests sensitivity
Question it answers: "How do different weightage parameters affect modality importance and learner selection?"
7. Learner Selection Interpretability Test
What it does: Tests modality-aware vs fixed learner selection strategies
Method: run_stage3_interpretability_test('learner_selection', test_scenarios)
Algorithm:
Compares modality-aware vs fixed selection approaches
Analyzes learner type distribution changes
Measures selection consistency and quality
Evaluates selection strategy effectiveness
Output: Dictionary with selection strategy comparisons, distribution analysis, consistency metrics
Usage: Validates modality-aware selection, compares selection strategies, measures selection quality
Question it answers: "How does modality-aware selection compare to fixed selection in terms of quality and consistency?"
8. Performance Prediction Interpretability Test
What it does: Tests the accuracy and effectiveness of performance prediction system
Method: run_stage3_interpretability_test('performance_prediction', test_scenarios)
Algorithm:
Tests different performance prediction parameters
Analyzes base performance, bonuses, and penalties
Measures prediction vs actual performance correlation
Evaluates prediction system accuracy
Output: Dictionary with prediction accuracy metrics, parameter sensitivity analysis, correlation results
Usage: Validates performance prediction system, optimizes prediction parameters, measures prediction accuracy
Question it answers: "How accurate is the performance prediction system and how do different parameters affect it?"
9. Bag-Learner Pairing Interpretability Test
What it does: Tests different bag-learner pairing strategies and their effectiveness
Method: run_stage3_interpretability_test('bag_learner_pairing', test_scenarios)
Algorithm:
Tests different pairing strategies (performance, diversity, efficiency)
Analyzes pairing quality metrics
Measures pairing consistency and effectiveness
Compares pairing strategy outcomes
Output: Dictionary with pairing strategy comparisons, quality metrics, consistency analysis
Usage: Validates pairing strategies, optimizes pairing approach, measures pairing effectiveness
Question it answers: "Which bag-learner pairing strategy produces the best quality and consistency?"
10. Ensemble Coherence Interpretability Test
What it does: Tests ensemble coherence and consistency under different configurations
Method: run_stage3_interpretability_test('ensemble_coherence', test_scenarios)
Algorithm:
Tests coherence under different configurations
Analyzes learner type diversity and consistency
Measures relationship coherence between bags and learners
Evaluates overall ensemble coherence
Output: Dictionary with coherence metrics, diversity analysis, consistency measurements
Usage: Validates ensemble coherence, measures consistency quality, ensures balanced distribution
Question it answers: "How coherent and consistent is the ensemble under different configurations?"
11. Optimization Mode Interpretability Test
What it does: Tests different optimization strategies and their trade-offs
Method: run_stage3_interpretability_test('optimization_mode', test_scenarios)
Algorithm:
Tests accuracy vs performance vs efficiency optimization modes
Analyzes trade-offs between different optimization strategies
Measures optimization mode effectiveness
Evaluates strategy impact on learner selection
Output: Dictionary with optimization strategy comparisons, trade-off analysis, effectiveness metrics
Usage: Validates optimization strategies, measures trade-offs, optimizes strategy selection
Question it answers: "What are the trade-offs between different optimization modes and which is most effective?"
🎯 COMPREHENSIVE ANALYSIS METHODS
12. Complete Stage 3 Interpretability Data
What it does: Provides access to all Stage 3 interpretability data in one comprehensive dataset
Method: get_stage3_interpretability_data()
Algorithm:
Combines all Stage 3 interpretability data
Includes bag-learner summaries, pairing statistics, metadata completeness, ensemble coherence
Returns deep copy of complete interpretability dataset
Output: Complete dictionary with all Stage 3 interpretability information
Usage: Full Stage 3 analysis, research insights, comprehensive reporting
Question it answers: "What is the complete picture of Stage 3's bag-learner selection and configuration quality?"

**Stage 4:
�� INTERPRETABILITY TEST 1: CROSS-MODAL DENOISING EFFECTIVENESS
Name: Cross-Modal Denoising Effectiveness Analysis
What it does: Analyzes the effectiveness of the cross-modal denoising system
Method: analyze_cross_modal_denoising_effectiveness(trained_learners)
Algorithm:
Extracts denoising loss progression from training history
Calculates reconstruction accuracy and alignment consistency
Computes denoising effectiveness scores per bag
Analyzes objective contribution and modality benefit
Output:
denoising_loss_progression: Loss progression per bag
reconstruction_accuracy: Reconstruction accuracy scores
alignment_consistency: Alignment consistency scores
denoising_effectiveness_score: Overall effectiveness metrics
objective_contribution: Contribution of each denoising objective
modality_benefit_analysis: Benefits per modality
Usage: training_pipeline.analyze_cross_modal_denoising_effectiveness(trained_learners)
Question Answered: How effective is the cross-modal denoising system?
🎯 INTERPRETABILITY TEST 2: MODAL-SPECIFIC METRICS GRANULARITY
Name: Modal-Specific Metrics Granularity Analysis
What it does: Analyzes granularity and insights from modal-specific metrics tracking
Method: analyze_modal_specific_metrics_granularity(trained_learners)
Algorithm:
Collects modal-specific metrics across all learners and epochs
Analyzes modal performance progression over time
Calculates improvement rates and correlation analysis
Identifies critical modalities and tracking frequency impact
Output:
modal_performance_progression: Performance trends per modality
modal_improvement_rates: Improvement rates per modality
modal_correlation_analysis: Correlations between modalities
critical_modality_identification: Most important modalities
tracking_frequency_impact: Impact of tracking frequency
bag_configuration_modal_variation: Modal variation across bags
Usage: training_pipeline.analyze_modal_specific_metrics_granularity(trained_learners)
Question Answered: What insights do modal-specific metrics provide?
�� INTERPRETABILITY TEST 3: BAG CHARACTERISTICS PRESERVATION TRACEABILITY
Name: Bag Characteristics Preservation Traceability Analysis
What it does: Analyzes traceability and insights from bag characteristics preservation
Method: analyze_bag_characteristics_preservation_traceability(trained_learners)
Algorithm:
Analyzes correlation between bag characteristics and performance
Calculates modality weight effectiveness
Evaluates audit trail completeness
Assesses ensemble behavior analysis
Output:
bag_characteristics_performance_correlation: Correlations between bag features and performance
modality_weight_effectiveness: Effectiveness of modality weights
bag_configuration_impact: Impact of bag configuration on performance
audit_trail_insights: Completeness and benefits of audit trail
performance_prediction_accuracy: Accuracy of performance predictions
ensemble_behavior_analysis: Analysis of ensemble behavior patterns
Usage: training_pipeline.analyze_bag_characteristics_preservation_traceability(trained_learners)
Question Answered: How traceable and interpretable are the bag characteristics?
🎯 INTERPRETABILITY TEST 4: TRAINING METRICS COMPREHENSIVE ANALYSIS
Name: Training Metrics Comprehensive Analysis
What it does: Provides comprehensive training metrics and performance analysis
Method: get_stage4_interpretability_data() (API wrapper)
Algorithm:
Collects training metrics from all trained learners
Aggregates bag characteristics and learner performance
Provides unified access to interpretability data
Output:
training_metrics: Complete training history for all learners
bag_characteristics: Preserved bag characteristics
learner_performance: Performance metrics per learner
Usage: model.get_stage4_interpretability_data()
Question Answered: What comprehensive training insights are available?

**Stage 5:
🎯 INTERPRETABILITY TEST 1: ENSEMBLE AGGREGATION INTERPRETABILITY
Name: Ensemble Aggregation Interpretability Analysis
What it does: Analyzes how the ensemble aggregation system makes decisions and combines predictions
Method: run_ensemble_aggregation_interpretability_test(test_data, test_labels)
Algorithm:
Extracts aggregation strategy and mixture weight patterns
Calculates decision consistency and prediction stability
Computes ensemble coherence and learner agreement
Analyzes mixture weight statistics and attention patterns
Output:
aggregation_analysis: Strategy type, learner count, mixture weight statistics
decision_consistency: Prediction confidence, uncertainty patterns
ensemble_coherence: Learner agreement, prediction stability, aggregation effectiveness
interpretability_score: Overall ensemble aggregation interpretability
Usage: model.run_ensemble_aggregation_interpretability_test(test_data, test_labels)
Question Answered: How does the ensemble aggregation system make decisions?
🎯 INTERPRETABILITY TEST 2: MODALITY IMPORTANCE GRANULARITY
Name: Modality Importance Granularity Analysis
What it does: Analyzes how different modalities contribute to ensemble decisions
Method: run_modality_importance_interpretability_test(test_data, test_labels)
Algorithm:
Collects modality importance weights and contribution patterns
Analyzes modality diversity, balance, and dominant modality identification
Calculates cross-modal interaction strength and synergistic effects
Evaluates modality-specific performance and reliability
Output:
modality_analysis: Modality weights, diversity, balance, dominant modality
cross_modal_analysis: Modality correlations, interaction strength, synergistic effects
modality_performance: Modality reliability, consistency, effectiveness
interpretability_score: Overall modality importance interpretability
Usage: model.run_modality_importance_interpretability_test(test_data, test_labels)
Question Answered: Which modalities contribute most to ensemble decisions?
�� INTERPRETABILITY TEST 3: UNCERTAINTY CALIBRATION ANALYSIS
Name: Uncertainty Calibration Analysis
What it does: Analyzes how well the uncertainty estimates correlate with actual prediction errors
Method: run_uncertainty_calibration_interpretability_test(test_data, test_labels)
Algorithm:
Calculates correlation between uncertainty and prediction errors
Computes calibration accuracy using uncertainty bins and error rates
Analyzes confidence vs uncertainty relationships
Evaluates uncertainty method effectiveness and characteristics
Output:
calibration_analysis: Uncertainty method, calibration correlation, accuracy, reliability
confidence_uncertainty_analysis: Confidence-uncertainty correlation, reliability, consistency
method_effectiveness: Method characteristics, effectiveness score
interpretability_score: Overall uncertainty calibration interpretability
Usage: model.run_uncertainty_calibration_interpretability_test(test_data, test_labels)
Question Answered: How well do uncertainty estimates predict actual errors?
�� INTERPRETABILITY TEST 4: ATTENTION PATTERN ANALYSIS
Name: Attention Pattern Analysis
What it does: Analyzes attention patterns in transformer-based aggregation
Method: run_attention_pattern_interpretability_test(test_data, test_labels)
Algorithm:
Extracts attention diversity, consistency, and focus patterns
Analyzes transformer architecture effectiveness and design impact
Calculates learner attention weights and importance ranking
Evaluates attention concentration and spread across samples
Output:
attention_analysis: Attention diversity, consistency, focus, pattern details
transformer_analysis: Architecture type, effectiveness, design parameters
learner_attention_analysis: Learner attention weights, ranking, stability
interpretability_score: Overall attention pattern interpretability
Usage: model.run_attention_pattern_interpretability_test(test_data, test_labels)
Question Answered: How does transformer attention distribute across learners?
🎯 INTERPRETABILITY TEST 5: BAG RECONSTRUCTION FIDELITY
Name: Bag Reconstruction Fidelity Analysis
What it does: Analyzes how faithfully the bag reconstruction process preserves training bag characteristics
Method: run_bag_reconstruction_fidelity_interpretability_test(test_data, test_labels)
Algorithm:
Analyzes bag characteristics preservation and reconstruction consistency
Compares reconstruction vs direct prediction effectiveness
Calculates bag-learner alignment quality and consistency
Evaluates reconstruction necessity and fidelity components
Output:
reconstruction_analysis: Reconstruction fidelity, characteristics preservation, consistency
reconstruction_comparison: Reconstruction advantage, fidelity benefit, necessity
bag_learner_alignment: Alignment quality, consistency, effectiveness
fidelity_metrics: Overall fidelity score and component breakdown
Usage: model.run_bag_reconstruction_fidelity_interpretability_test(test_data, test_labels)
Question Answered: How faithfully does bag reconstruction preserve training characteristics?
🎯 INTERPRETABILITY TEST 6: INTEGRATED STAGE 5 ANALYSIS
Name: Integrated Stage 5 Interpretability Analysis
What it does: Provides comprehensive interpretability analysis combining all Stage 5 features
Method: run_stage5_interpretability_test(test_data, test_labels)
Algorithm:
Combines modality importance diversity and mixture weight entropy
Calculates uncertainty calibration and consistency metrics
Provides unified interpretability scoring across all features
Output:
integrated_interpretability_score: Combined interpretability metric
modality_importance: Modality weight distribution
mixture_weight_entropy: Ensemble attention diversity
uncertainty_std: Uncertainty consistency measure
Usage: model.run_stage5_interpretability_test(test_data, test_labels)
Question Answered: What is the overall interpretability of Stage 5 ensemble prediction?

**Outputs**:
- Interpretability report (`interpretability_report.json`)
- Visualization files (attention maps, importance plots)
- Sample-level explanations
- Uncertainty calibration curves

---

### **Phase 6: Robustness Tests**
**Purpose**: Evaluate model performance under various challenging conditions based on the best hyperparameter combination result from phase 3 on the respective dataset

**Robustness Tests**:

**Stage 2:
🎯 PARAMETER VALIDATION ROBUSTNESS
1. Input Parameter Validation
What it does: Validates all input parameters to prevent invalid configurations
Method: _validate_params()
Validation Rules:
n_bags: Must be between 1 and 1000
dropout_strategy: Must be one of ['adaptive', 'linear', 'exponential', 'random']
max_dropout_rate: Must be between 0.0 and 0.9
sample_ratio: Must be between 0.1 and 1.0
feature_sampling_ratio: Must be between 0.1 and 1.0
min_modalities: Must be between 1 and total number of modalities
Robustness Test: "Does the system handle invalid parameter inputs gracefully?"
Expected Result: Raises clear assertion errors for invalid parameters
🎯 BOUNDARY CONDITION ROBUSTNESS
2. Value Clipping and Bounds Checking
What it does: Ensures all calculated values stay within valid ranges
Method: np.clip() and boundary checks throughout
Examples:
adaptive_rate = np.clip(base_rate + variation, 0, self.max_dropout_rate)
n_drop = min(n_drop, n_modalities - self.min_modalities)
n_selected = max(1, int(self.feature_sampling_ratio * n_features))
Robustness Test: "Does the system handle edge cases and prevent out-of-bounds values?"
Expected Result: All values stay within valid ranges
3. Empty Data Handling
What it does: Handles cases where no features are selected or data is empty
Method: Conditional checks for empty arrays
Examples:
if len(feature_mask) > 0: - Prevents empty feature selection
selection_ratio = selected_count / len(feature_mask) if len(feature_mask) > 0 else 0.0
Robustness Test: "Does the system handle empty feature masks gracefully?"
Expected Result: No division by zero errors, graceful handling of empty data
🎯 INFINITE LOOP PREVENTION
4. Adaptive Strategy Loop Prevention
What it does: Prevents infinite loops in adaptive strategy when generating distinct combinations
Method: attempts counter with maximum limit
Algorithm:
attempts = 0
while attempts < 10: (maximum 10 attempts)
attempts += 1 if no distinct combination found
Robustness Test: "Does the adaptive strategy prevent infinite loops when generating distinct combinations?"
Expected Result: Maximum 10 attempts before falling back to any valid combination
🎯 ERROR HANDLING ROBUSTNESS
5. Bag Access Validation
What it does: Validates bag ID access to prevent index errors
Method: get_bag_data() and get_bag_info()
Validation:
if bag_id not in self.bag_data: raise ValueError(f"Bag {bag_id} not found")
if bag_id >= len(self.bags): raise ValueError(f"Bag {bag_id} not found")
Robustness Test: "Does the system handle invalid bag ID access gracefully?"
Expected Result: Clear error messages for invalid bag access
6. Empty Ensemble Handling
What it does: Handles cases where no bags have been generated
Method: get_ensemble_stats()
Validation: if not self.bags: return {}
Robustness Test: "Does the system handle empty ensemble gracefully?"
Expected Result: Returns empty dictionary instead of crashing
🎯 MATHEMATICAL ROBUSTNESS
7. Division by Zero Prevention
What it does: Prevents division by zero in statistical calculations
Method: Conditional checks before division
Examples:
selection_ratio = selected_count / len(feature_mask) if len(feature_mask) > 0 else 0.0
base_rate = self.max_dropout_rate * (bag_id / max(1, self.n_bags - 1))
Robustness Test: "Does the system prevent division by zero errors?"
Expected Result: No mathematical errors, graceful handling of edge cases
8. Statistical Calculation Robustness
What it does: Ensures statistical calculations handle edge cases
Method: Safe statistical operations
Examples:
max(1, self.n_bags - 1) - Prevents division by zero
max(1, int(self.feature_sampling_ratio * n_features)) - Ensures at least 1 feature
Robustness Test: "Do statistical calculations handle edge cases robustly?"
Expected Result: All statistical operations complete without errors
🎯 DATA CONSISTENCY ROBUSTNESS
9. Modality Configuration Validation
What it does: Ensures modality configurations are consistent
Method: Parameter validation and data structure checks
Validation: assert 1 <= self.min_modalities <= len(self.modality_configs)
Robustness Test: "Does the system validate modality configuration consistency?"
Expected Result: Clear errors for inconsistent modality configurations
10. Random State Robustness
What it does: Ensures reproducible results with random state
Method: self._rng = np.random.default_rng(random_state)
Robustness Test: "Does the system produce reproducible results with same random state?"
Expected Result: Identical results with same random state

**Stage 3:
🎯 OPTIMIZATION MODE ROBUSTNESS
1. Optimization Strategy Robustness
What it does: Tests robustness of different optimization modes (accuracy, performance, efficiency) in learner selection
Method: test_optimization_mode_robustness()
Algorithm:
Creates test selectors with different optimization modes
Compares bag-learner assignment quality across modes
Measures modality-learner match rates and pairing consistency
Output: Dictionary with robustness scores, pairing statistics, ensemble coherence for each optimization mode
Usage: Validates optimization strategy effectiveness, measures mode-specific robustness
Question it answers: "How robust is the learner selection system across different optimization strategies?"
🎯 MODALITY WEIGHTAGE ROBUSTNESS
2. Weightage Parameter Sensitivity
What it does: Tests robustness of modality weightage parameters (feature_ratio_weight, variance_weight, dimensionality_weight)
Method: test_modality_weightage_robustness()
Algorithm:
Creates test selectors with different weightage parameter combinations
Analyzes impact on modality importance calculations
Measures weightage sensitivity and selection consistency
Output: Dictionary with robustness scores, weightage sensitivity analysis, selection impact metrics
Usage: Validates weightage parameter robustness, optimizes parameter sensitivity
Question it answers: "How sensitive is the modality importance calculation to weightage parameter changes?"
🎯 BAG-LEARNER PAIRING ROBUSTNESS
3. Pairing Strategy Robustness
What it does: Tests robustness of bag-learner pairing strategies (performance, diversity, efficiency focus)
Method: test_bag_learner_pairing_robustness()
Algorithm:
Tests different pairing focuses and metadata levels
Analyzes pairing quality and consistency across strategies
Measures metadata completeness and pairing effectiveness
Output: Dictionary with robustness scores, pairing statistics, metadata completeness, ensemble coherence
Usage: Validates pairing strategy effectiveness, measures pairing quality robustness
Question it answers: "How robust are different bag-learner pairing strategies in maintaining quality?"
🎯 PERFORMANCE PREDICTION ROBUSTNESS
4. Prediction System Stability
What it does: Tests robustness of performance prediction system parameters (base_performance, bonuses, penalties)
Method: test_performance_prediction_robustness()
Algorithm:
Tests different prediction parameter configurations
Analyzes performance prediction stability and accuracy
Measures prediction consistency across parameter variations
Output: Dictionary with robustness scores, performance stability metrics, prediction consistency analysis
Usage: Validates prediction system robustness, measures parameter sensitivity
Question it answers: "How stable and robust is the performance prediction system across different parameter settings?"
🎯 ENSEMBLE SIZE ROBUSTNESS
5. Scalability Robustness
What it does: Tests robustness across different ensemble sizes (small, medium, large ensembles)
Method: test_ensemble_size_robustness()
Algorithm:
Tests learner selection with different ensemble sizes
Analyzes size efficiency and diversity maintenance
Measures scalability and performance consistency
Output: Dictionary with robustness scores, size efficiency metrics, diversity maintenance analysis
Usage: Validates ensemble scalability, measures size-dependent robustness
Question it answers: "How well does the learner selection system scale across different ensemble sizes?"
🎯 MODALITY-AWARE ROBUSTNESS
6. Modality Awareness Robustness
What it does: Tests robustness of modality-aware vs fixed learner selection approaches
Method: test_modality_aware_robustness() (via main robustness test method)
Algorithm:
Compares modality-aware vs fixed selection strategies
Analyzes selection quality and consistency differences
Measures modality awareness effectiveness
Output: Dictionary with robustness scores, selection strategy comparisons, modality awareness metrics
Usage: Validates modality-aware selection robustness, measures awareness effectiveness
Question it answers: "How robust is the modality-aware learner selection compared to fixed selection?"
�� COMPREHENSIVE ROBUSTNESS TESTING
7. Main Robustness Test Orchestrator
What it does: Provides unified interface for running all Stage 3 robustness tests
Method: run_stage3_robustness_test()
Algorithm:
Routes to specific robustness test methods based on test_type
Supports all 6 robustness test types
Provides consistent error handling and result formatting
Output: Results from specific robustness test method
Usage: Unified robustness testing interface, comprehensive Stage 3 robustness validation
Question it answers: "What is the overall robustness of Stage 3's learner selection system?"
🎯 ROBUSTNESS METRICS AND VALIDATION
Core Robustness Metrics:
Robustness Score: Overall robustness assessment (0-1 scale)
Modality-Learner Match Rate: Quality of modality-learner pairings
Pairing Consistency: Consistency of similar bag-learner pairings
Ensemble Coherence: Overall ensemble consistency and diversity
Performance Stability: Stability of performance predictions
Size Efficiency: Efficiency across different ensemble sizes
Weightage Sensitivity: Sensitivity to weightage parameter changes
Validation Features:
Error Handling: Comprehensive error handling with detailed error messages
Parameter Validation: Input parameter validation and bounds checking
Edge Case Handling: Graceful handling of edge cases and boundary conditions
Reproducibility: Consistent results with same random state
Comprehensive Coverage: Tests all major Stage 3 novel features

**Stage 4:
🎯 ROBUSTNESS TEST 1: CROSS-MODAL DENOISING ROBUSTNESS
Name: Cross-Modal Denoising Robustness Test
What it does: Tests robustness of the cross-modal denoising system under various conditions
Method: test_cross_modal_denoising_robustness(test_scenarios)
Algorithm:
Tests different denoising strategies (adaptive, cross_modal, modal_specific)
Tests different denoising weights (0.05, 0.1, 0.2, 0.3)
Tests different denoising objectives (reconstruction, alignment, consistency)
Evaluates robustness under various training configurations
Output:
denoising_strategy_robustness: Robustness scores for different strategies
denoising_weight_robustness: Robustness scores for different weights
denoising_objective_robustness: Robustness scores for different objectives
noise_level_robustness: Robustness under different noise levels
modality_combination_robustness: Robustness for different modality combinations
training_configuration_robustness: Robustness under different training configs
overall_robustness_score: Overall robustness score
Usage: training_pipeline.test_cross_modal_denoising_robustness(test_scenarios)
Question Answered: How robust is the cross-modal denoising system?
🎯 ROBUSTNESS TEST 2: MODAL-SPECIFIC METRICS ROBUSTNESS
Name: Modal-Specific Metrics Robustness Test
What it does: Tests robustness of the modal-specific metrics tracking system
Method: test_modal_specific_metrics_robustness(test_scenarios)
Algorithm:
Tests different tracking frequencies (every_epoch, every_5_epochs, every_10_epochs)
Tests different tracking combinations (reconstruction, alignment, consistency)
Tests modality selection robustness
Tests data size and computational constraint robustness
Output:
tracking_frequency_robustness: Robustness scores for different frequencies
tracking_combination_robustness: Robustness scores for different combinations
modality_selection_robustness: Robustness for different modality selections
data_size_robustness: Robustness under different data sizes
computational_constraint_robustness: Robustness under computational constraints
training_duration_robustness: Robustness under different training durations
overall_robustness_score: Overall robustness score
Usage: training_pipeline.test_modal_specific_metrics_robustness(test_scenarios)
Question Answered: How robust is the modal-specific metrics tracking system?
🎯 ROBUSTNESS TEST 3: BAG CHARACTERISTICS ROBUSTNESS
Name: Bag Characteristics Robustness Test
What it does: Tests robustness of the bag characteristics preservation system
Method: test_bag_characteristics_robustness(test_scenarios)
Algorithm:
Tests different preservation combinations (modality masks, weights, IDs, metrics)
Tests memory constraint robustness (low_memory, medium_memory, high_memory)
Tests bag size robustness
Tests modality complexity robustness
Output:
preservation_combination_robustness: Robustness scores for different preservation combinations
memory_constraint_robustness: Robustness under different memory constraints
bag_size_robustness: Robustness for different bag sizes
modality_complexity_robustness: Robustness under different modality complexities
training_metrics_level_robustness: Robustness for different metrics levels
learner_config_robustness: Robustness for different learner configurations
overall_robustness_score: Overall robustness score
Usage: training_pipeline.test_bag_characteristics_robustness(test_scenarios)
Question Answered: How robust is the bag characteristics preservation system?
🎯 ROBUSTNESS TEST 4: INTEGRATED STAGE 4 ROBUSTNESS
Name: Integrated Stage 4 Robustness Test
What it does: Tests robustness of integrated Stage 4 novel features
Method: test_integrated_stage4_robustness(test_scenarios)
Algorithm:
Tests different feature combinations (denoising + tracking + preservation)
Tests stress condition robustness (high_load, low_memory, network_issues)
Tests dataset variation robustness
Tests hardware configuration robustness
Output:
feature_combination_robustness: Robustness scores for different feature combinations
stress_condition_robustness: Robustness under stress conditions
dataset_variation_robustness: Robustness under dataset variations
hardware_configuration_robustness: Robustness under different hardware configs
integration_robustness: Robustness of feature integration
performance_robustness: Robustness of performance under various conditions
overall_robustness_score: Overall robustness score
Usage: training_pipeline.test_integrated_stage4_robustness(test_scenarios)
Question Answered: How robust are the integrated Stage 4 novel features?

**Stage 5:
�� ROBUSTNESS TEST 1: BAG RECONSTRUCTION ROBUSTNESS
Name: Bag Reconstruction Robustness Test
What it does: Tests robustness of the bag reconstruction system under various conditions
Method: run_bag_reconstruction_robustness_test(test_data, test_labels)
Algorithm:
Tests noise perturbation robustness (levels: 0.01, 0.05, 0.1, 0.2)
Tests modality dropout robustness (dropping individual modalities)
Tests feature corruption robustness (levels: 0.1, 0.2, 0.3)
Evaluates bag reconstruction fidelity under stress conditions
Output:
noise_robustness: Robustness scores for different noise levels
modality_dropout_robustness: Robustness when dropping individual modalities
feature_corruption_robustness: Robustness under feature corruption
overall_robustness_score: Overall robustness score combining all tests
Usage: model.run_bag_reconstruction_robustness_test(test_data, test_labels)
Question Answered: How robust is the bag reconstruction system under various stress conditions?
🎯 ROBUSTNESS TEST 2: UNCERTAINTY METHOD ROBUSTNESS
Name: Uncertainty Method Robustness Test
What it does: Tests robustness of different uncertainty quantification methods under stress
Method: run_uncertainty_method_robustness_test(test_data, test_labels)
Algorithm:
Tests different uncertainty methods (entropy, variance, confidence)
Tests under various stress conditions (baseline, noise 0.05, 0.1, 0.2)
Evaluates uncertainty calibration under stress
Compares method performance degradation
Output:
method_results: Results for each uncertainty method under each condition
robustness_scores: Robustness scores for each method
best_method: Most robust uncertainty method
overall_robustness_score: Overall robustness across all methods
Usage: model.run_uncertainty_method_robustness_test(test_data, test_labels)
Question Answered: How robust are different uncertainty quantification methods under stress?
🎯 ROBUSTNESS TEST 3: TRANSFORMER ARCHITECTURE ROBUSTNESS
Name: Transformer Architecture Robustness Test
What it does: Tests robustness of different transformer architectures under various conditions
Method: run_transformer_architecture_robustness_test(test_data, test_labels)
Algorithm:
Tests different transformer configurations (Small, Medium, Large)
Tests under stress conditions (baseline, noise, modality dropout)
Evaluates attention diversity and mixture weight entropy
Compares architecture performance under stress
Output:
architecture_results: Results for each transformer architecture
robustness_scores: Robustness scores for each architecture
best_architecture: Most robust transformer configuration
overall_robustness_score: Overall robustness across all architectures
Usage: model.run_transformer_architecture_robustness_test(test_data, test_labels)
Question Answered: How robust are different transformer architectures under stress conditions?
🎯 ROBUSTNESS TEST 4: ENSEMBLE AGGREGATION ROBUSTNESS
Name: Ensemble Aggregation Robustness Test
What it does: Tests robustness of different aggregation strategies under stress
Method: run_ensemble_aggregation_robustness_test(test_data, test_labels)
Algorithm:
Tests different aggregation strategies (simple_average, weighted_average, transformer_fusion)
Tests under stress conditions (baseline, noise, modality dropout)
Evaluates ensemble coherence and prediction confidence
Compares strategy performance degradation
Output:
strategy_results: Results for each aggregation strategy
robustness_scores: Robustness scores for each strategy
best_strategy: Most robust aggregation method
overall_robustness_score: Overall robustness across all strategies
Usage: model.run_ensemble_aggregation_robustness_test(test_data, test_labels)
Question Answered: How robust are different ensemble aggregation strategies under stress?
🎯 ROBUSTNESS TEST 5: INTEGRATED STAGE 5 ROBUSTNESS
Name: Integrated Stage 5 Robustness Test
What it does: Tests robustness of integrated Stage 5 features working together
Method: run_integrated_stage5_robustness_test(test_data, test_labels)
Algorithm:
Tests integrated system under comprehensive stress conditions
Tests noise robustness (levels: 0.05, 0.1, 0.2)
Tests modality dropout and feature corruption robustness
Evaluates ensemble coherence and modality importance diversity
Output:
stress_condition_results: Results for each stress condition
baseline_accuracy: Baseline performance
stress_accuracies: Performance under stress conditions
accuracy_degradation: Performance degradation measures
robustness_breakdown: Breakdown by stress type (noise, dropout, corruption)
overall_robustness_score: Overall integrated robustness score
Usage: model.run_integrated_stage5_robustness_test(test_data, test_labels)
Question Answered: How robust are the integrated Stage 5 novel features working together?
🎯 ROBUSTNESS TEST 6: BASIC STAGE 5 ROBUSTNESS
Name: Basic Stage 5 Robustness Test
What it does: Tests basic robustness of Stage 5 ensemble prediction system
Method: run_stage5_robustness_test(test_data, test_labels)
Algorithm:
Tests with noise perturbation (levels: 0.01, 0.05, 0.1)
Evaluates accuracy degradation under noise
Provides baseline robustness assessment
Output:
overall_robustness_score: Overall robustness score
noise_levels: Tested noise levels
robustness_scores: Robustness scores for each noise level
Usage: model.run_stage5_robustness_test(test_data, test_labels)
Question Answered: What is the basic robustness of the Stage 5 ensemble prediction system?

**Outputs**:
- Comparative analysis report (`comparative_analysis.json`)
- Statistical significance tables
- Performance ranking summary
- Efficiency comparison charts

---

## 📁 **File Structure for Each Dataset**

```
results/
├── run_YYYYMMDD_HHMMSS/
│   ├── seed_42/
│   │   ├── phase_1_data_validation/
│   │   │   ├── data_quality_report.json
│   │   │   └── preprocessing_summary.json
│   │   ├── phase_2_baseline/
│   │   │   ├── baseline_results.json
│   │   │   └── baseline_summary.csv
│   │   ├── phase_3_mainmodel/
│   │   │   ├── mainmodel_trials.json
│   │   │   ├── mainmodel_best.json
│   │   │   └── optimization_history.json
│   │   ├── phase_4_ablation/
│   │   │   ├── ablation_results.json
│   │   │   └── component_importance.json
│   │   ├── phase_5_interpretability/
│   │   │   ├── interpretability_report.json
│   │   │   ├── modality_importance.json
│   │   │   └── uncertainty_analysis.json
│   │   ├── phase_6_robustness/
│   │   │   ├── robustness_results.json
│   │   │   ├── missing_modality_analysis.json
│   │   │   └── adversarial_robustness.json
│   └── comprehensive_report/
│       ├── executive_summary.md
│       ├── detailed_analysis.md
│       └── recommendations.md
```

## 🚀 **Implementation Priority**

### **High Priority (Core Evaluation)**:
1. Phase 1: Data Validation
2. Phase 2: Baseline Models
3. Phase 3: Hyperparameter Optimization
4. Phase 4: Ablation Studies

### **Medium Priority (Advanced Analysis)**:
5. Phase 5: Interpretability Studies
6. Phase 6: Robustness Tests

## 📊 **Success Metrics**

### **Performance Targets**:
- **Accuracy Improvement**: >5% over best baseline
- **Robustness**: <10% performance drop under missing modalities
- **Efficiency**: <2x training time compared to best baseline
- **Interpretability**: >80% explanation accuracy

### **Statistical Requirements**:
- **Significance Level**: p < 0.05 for all comparisons
- **Effect Size**: Cohen's d > 0.5 for meaningful differences
- **Confidence Intervals**: 95% CI for all performance metrics
- **Multiple Testing**: FDR correction for multiple comparisons

This comprehensive pipeline ensures thorough evaluation of your multimodal ensemble model across all critical dimensions! 🎯
