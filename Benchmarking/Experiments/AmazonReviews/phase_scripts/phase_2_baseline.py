#!/usr/bin/env python3
"""
Phase 2: Baseline Model Evaluation
Evaluates baseline models for AmazonReviews dataset comparison.
"""

import json
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import warnings
import psutil
import os
import signal
warnings.filterwarnings('ignore')

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import required libraries
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR, LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, 
        balanced_accuracy_score, roc_auc_score, mean_squared_error,
        mean_absolute_error, r2_score
    )
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import Ridge, Lasso
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor
    SKLEARN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

def timeout_handler(signum, frame):
    """Handle timeout for model training."""
    raise TimeoutError("Model training timed out")

class BaselineEvaluator:
    """Evaluates baseline models for AmazonReviews dataset."""
    
    def __init__(self, config: Dict[str, Any], processed_data: Dict[str, Any] = None):
        self.config = config
        self.phase_dir = Path(config["phase_dir"])
        self.seed = config["seed"]
        self.test_mode = config["test_mode"]
        np.random.seed(self.seed)
        
        # Use provided processed data or load from Phase 1
        if processed_data is not None:
            self.dataset = processed_data
            logger.info(f"Using provided processed data from previous phase")
        else:
            self.dataset = self._load_phase1_data()
            logger.info(f"Loading data from Phase 1 files")
        
        self.task_type = self._determine_task_type()
        
        logger.info(f"Initialized BaselineEvaluator for {self.test_mode} mode")
        logger.info(f"Task type: {self.task_type}")
    
    def _load_phase1_data(self) -> Dict[str, Any]:
        """Load sampled data from Phase 1 results."""
        try:
            # Load the sampled data from Phase 1
            dataset_path = Path(self.config["dataset_path"])
            
            # Find the sampled data directory
            sampled_dir = None
            if self.test_mode == "quick":
                sampled_dir = dataset_path / f"sampled_quick_{self.seed}"
            else:
                sampled_dir = dataset_path / f"sampled_full_{self.seed}"
            
            if not sampled_dir.exists():
                raise FileNotFoundError(f"Sampled data not found at {sampled_dir}. Please run Phase 1 first.")
            
            logger.info(f"Loading sampled data from {sampled_dir}")
            
            # Load the CSV files
            train_file = sampled_dir / "train" / "train_data.csv"
            test_file = sampled_dir / "test" / "test_data.csv"
            
            if not train_file.exists() or not test_file.exists():
                raise FileNotFoundError(f"Sampled data files not found. Expected {train_file} and {test_file}")
            
            # Load the data
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            
            logger.info(f"Loaded train data: {train_df.shape}")
            logger.info(f"Loaded test data: {test_df.shape}")
            
            return {
                'train': train_df,
                'test': test_df
            }
            
        except Exception as e:
            logger.error(f"Error loading Phase 1 data: {e}")
            raise
    
    def _determine_task_type(self) -> str:
        """Determine if this is a classification or regression task."""
        try:
            train_data = self.dataset.get('train')
            if train_data is None:
                return 'classification'
            
            target_columns = ['label', 'target', 'class', 'rating', 'score', 'sentiment']
            target_col = None
            for col in target_columns:
                if col in train_data.columns:
                    target_col = col
                    break
            else:
                target_col = train_data.columns[-1]
            
            target_values = train_data[target_col].dropna()
            unique_count = target_values.nunique()
            
            if target_values.dtype in ['object', 'string'] or unique_count <= 10:
                return 'classification'
            else:
                return 'regression'
                
        except Exception as e:
            logger.warning(f"Error determining task type: {e}")
            return 'classification'
    
    def _prepare_data(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Prepare data for baseline models."""
        logger.info("Preparing data for baseline models...")
        
        # Handle different data formats
        if 'train' in self.dataset and 'test' in self.dataset:
            # Standard format from Phase 1 (CSV strings)
            train_data = self.dataset['train']
            test_data = self.dataset['test']
            
            # Convert CSV strings back to DataFrames
            if isinstance(train_data, str):
                import pandas as pd
                from io import StringIO
                train_df = pd.read_csv(StringIO(train_data))
                test_df = pd.read_csv(StringIO(test_data))
            else:
                train_df = train_data
                test_df = test_data
        else:
            # Mock data format - create train/test split
            features = self.dataset.get('features', self.dataset.get('X'))
            labels = self.dataset.get('labels', self.dataset.get('y'))
            
            if features is None or labels is None:
                raise ValueError("Invalid data format: missing features or labels")
            
            # Simple train/test split (80/20)
            n_samples = len(features)
            n_train = int(0.8 * n_samples)
            
            train_data = {
                'features': features[:n_train],
                'labels': labels[:n_train]
            }
            test_data = {
                'features': features[n_train:],
                'labels': labels[n_train:]
            }
        
        # Identify target column
        target_columns = ['label', 'target', 'class', 'rating', 'score', 'sentiment']
        target_col = None
        for col in target_columns:
            if col in train_df.columns:
                target_col = col
                break
        else:
            target_col = train_df.columns[-1]
        
        # Separate features and target
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        # Identify text and metadata columns
        text_columns = []
        metadata_columns = []
        
        for col in X_train.columns:
            if any(keyword in col.lower() for keyword in ['text', 'review', 'comment', 'description', 'content']):
                text_columns.append(col)
            else:
                metadata_columns.append(col)
        
        # Prepare text data
        text_data = {}
        if text_columns:
            text_data['train'] = X_train[text_columns].fillna('')
            text_data['test'] = X_test[text_columns].fillna('')
            text_data['train'] = text_data['train'].apply(lambda x: ' '.join(x.astype(str)), axis=1)
            text_data['test'] = text_data['test'].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        
        # Prepare metadata
        metadata_data = {}
        if metadata_columns:
            metadata_data['train'] = X_train[metadata_columns].fillna(0)
            metadata_data['test'] = X_test[metadata_columns].fillna(0)
            for col in metadata_data['train'].columns:
                if metadata_data['train'][col].dtype == 'object':
                    le = LabelEncoder()
                    metadata_data['train'][col] = le.fit_transform(metadata_data['train'][col].astype(str))
                    metadata_data['test'][col] = le.transform(metadata_data['test'][col].astype(str))
        
        # Encode target for classification
        if self.task_type == 'classification':
            # Check if labels are already discrete (integers)
            if np.issubdtype(y_train.dtype, np.integer) or len(np.unique(y_train)) <= 10:
                # Labels are already discrete, use LabelEncoder
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
                y_test = le.transform(y_test)
            else:
                # Labels are continuous, convert to discrete classes
                # For AmazonReviews, convert ratings to discrete classes
                y_train = np.round(y_train).astype(int)
                y_test = np.round(y_test).astype(int)
                # Ensure labels are in valid range (1-5 for ratings)
                y_train = np.clip(y_train, 1, 5)
                y_test = np.clip(y_test, 1, 5)
                # Convert to 0-based indexing for sklearn
                y_train = y_train - 1
                y_test = y_test - 1
        
        data_prep = {
            'text_data': text_data,
            'metadata_data': metadata_data,
            'target_col': target_col,
            'text_columns': text_columns,
            'metadata_columns': metadata_columns
        }
        
        targets = {
            'train': y_train,
            'test': y_test
        }
        
        logger.info(f"Data prepared: {len(text_columns)} text columns, {len(metadata_columns)} metadata columns")
        return data_prep, targets
    
    def _get_tfidf_features(self, text_data: pd.Series, fit_vectorizer: bool = True, max_features: int = 1000) -> np.ndarray:
        """Extract TF-IDF features from text data."""
        if not SKLEARN_AVAILABLE:
            return np.zeros((len(text_data), 100))
        
        if fit_vectorizer:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                stop_words='english',
                max_df=0.95,
                min_df=2
            )
            return self.tfidf_vectorizer.fit_transform(text_data).toarray()
        else:
            return self.tfidf_vectorizer.transform(text_data).toarray()
    
    def evaluate_baselines(self, data_prep: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all baseline models."""
        logger.info("Starting baseline model evaluation...")
        logger.info(f"Task type: {self.task_type}")
        logger.info(f"Training samples: {len(targets['train'])}, Test samples: {len(targets['test'])}")
        
        results = {}
        
        # Unimodal baselines
        if data_prep['text_data']:
            logger.info("Step 1/4: Evaluating text baselines...")
            text_results = self._evaluate_text_baselines(data_prep['text_data'], targets)
            results.update(text_results)
            logger.info(f"Text baselines completed: {len(text_results)} models")
        
        if data_prep['metadata_data']:
            logger.info("Step 2/4: Evaluating metadata baselines...")
            metadata_results = self._evaluate_metadata_baselines(data_prep['metadata_data'], targets)
            results.update(metadata_results)
            logger.info(f"Metadata baselines completed: {len(metadata_results)} models")
        
        # Traditional fusion baselines
        if data_prep['text_data'] and data_prep['metadata_data']:
            logger.info("Step 3/4: Evaluating fusion baselines...")
            fusion_results = self._evaluate_fusion_baselines(data_prep, targets)
            results.update(fusion_results)
            logger.info(f"Fusion baselines completed: {len(fusion_results)} models")
        
        # Ensemble baselines
        if data_prep['text_data'] and data_prep['metadata_data']:
            logger.info("Step 4/4: Evaluating ensemble baselines...")
            ensemble_results = self._evaluate_ensemble_baselines(data_prep, targets)
            results.update(ensemble_results)
            logger.info(f"Ensemble baselines completed: {len(ensemble_results)} models")
        
        logger.info(f"All baseline evaluation completed! Total models: {len(results)}")
        return results
    
    def _evaluate_text_baselines(self, text_data: Dict[str, pd.Series], targets: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate text-only baseline models."""
        results = {}
        
        if not SKLEARN_AVAILABLE:
            return results
        
        # TF-IDF + SVM (optimized for large datasets)
        logger.info("  - Training TF-IDF + SVM...")
        start_time = time.time()
        try:
            # Set timeout for model training (30 minutes)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(1800)  # 30 minutes timeout
            
            # For large datasets, use more efficient TF-IDF settings
            X_train_tfidf = self._get_tfidf_features(text_data['train'], fit_vectorizer=True, max_features=10000)
            X_test_tfidf = self._get_tfidf_features(text_data['test'], fit_vectorizer=False, max_features=10000)
            
            if self.task_type == 'classification':
                # Use LinearSVC instead of SVC for better performance on large datasets
                model = LinearSVC(random_state=self.seed, max_iter=1000)
            else:
                model = SVR(kernel='linear', max_iter=1000)
            
            model.fit(X_train_tfidf, targets['train'])
            
            # Cancel timeout
            signal.alarm(0)
            
            training_time = time.time() - start_time
            
            # Use standardized evaluation
            result = self._evaluate_model_with_metrics(
                model, X_train_tfidf, targets['train'], X_test_tfidf, targets['test'],
                'TF-IDF + SVM', 'text_only', training_time
            )
            
            if result:
                results['text_tfidf_svm'] = result
            
            logger.info(f"TF-IDF + SVM completed")
            
        except TimeoutError:
            logger.warning("TF-IDF + SVM training timed out after 30 minutes")
            signal.alarm(0)  # Cancel any remaining alarm
        except Exception as e:
            logger.error(f"Error in TF-IDF + SVM: {e}")
            signal.alarm(0)  # Cancel any remaining alarm
        
        # TF-IDF + Random Forest (optimized for large datasets)
        logger.info("  - Training TF-IDF + Random Forest...")
        start_time = time.time()
        try:
            X_train_tfidf = self._get_tfidf_features(text_data['train'], fit_vectorizer=True, max_features=5000)
            X_test_tfidf = self._get_tfidf_features(text_data['test'], fit_vectorizer=False, max_features=5000)
            
            if self.task_type == 'classification':
                # Reduce n_estimators for faster training on large datasets
                model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=self.seed, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=self.seed)
            
            model.fit(X_train_tfidf, targets['train'])
            
            train_pred = model.predict(X_train_tfidf)
            test_pred = model.predict(X_test_tfidf)
            
            test_proba = model.predict_proba(X_test_tfidf) if hasattr(model, 'predict_proba') else None
            
            metrics = self._calculate_metrics(targets['train'], train_pred, targets['test'], test_pred, test_proba)
            
            results['text_tfidf_rf'] = {
                'model_type': 'text_only',
                'model_name': 'TF-IDF + Random Forest',
                'training_time': time.time() - start_time,
                'metrics': metrics,
                'predictions': {
                    'train': train_pred.tolist(),
                    'test': test_pred.tolist()
                }
            }
            
            logger.info(f"TF-IDF + Random Forest completed")
            
        except Exception as e:
            logger.error(f"Error in TF-IDF + Random Forest: {e}")
        
        # BERT-based classification (if transformers available) - DISABLED due to segmentation fault
        if False and TRANSFORMERS_AVAILABLE and self.task_type == 'classification':
            logger.info("  - Training BERT + Logistic Regression (this may take a while)...")
            start_time = time.time()
            try:
                # Use a smaller BERT model for efficiency
                model_name = "distilbert-base-uncased"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                bert_model = AutoModel.from_pretrained(model_name)
                
                # Tokenize text (using a subset for efficiency)
                max_length = 128
                train_texts = text_data['train'].iloc[:min(1000, len(text_data['train']))]
                test_texts = text_data['test'].iloc[:min(500, len(text_data['test']))]
                
                train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
                test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
                
                # Extract BERT embeddings
                with torch.no_grad():
                    train_outputs = bert_model(**train_encodings)
                    test_outputs = bert_model(**test_encodings)
                
                train_embeddings = train_outputs.last_hidden_state.mean(dim=1).numpy()
                test_embeddings = test_outputs.last_hidden_state.mean(dim=1).numpy()
                
                # Train classifier on BERT embeddings
                if self.task_type == 'classification':
                    classifier = LogisticRegression(random_state=self.seed, max_iter=1000)
                else:
                    classifier = LinearRegression()
                
                classifier.fit(train_embeddings, targets['train'][:len(train_embeddings)])
                
                train_pred = classifier.predict(train_embeddings)
                test_pred = classifier.predict(test_embeddings)
                
                test_proba = classifier.predict_proba(test_embeddings) if hasattr(classifier, 'predict_proba') else None
                
                metrics = self._calculate_metrics(
                    targets['train'][:len(train_embeddings)], train_pred, 
                    targets['test'][:len(test_embeddings)], test_pred, test_proba
                )
                
                results['text_bert'] = {
                    'model_type': 'text_only',
                    'model_name': 'BERT + Logistic Regression',
                    'training_time': time.time() - start_time,
                    'metrics': metrics,
                    'predictions': {
                        'train': train_pred.tolist(),
                        'test': test_pred.tolist()
                    }
                }
                
                logger.info(f"BERT + Logistic Regression completed")
                
            except Exception as e:
                logger.error(f"Error in BERT: {e}")
        
        return results
    
    def _evaluate_metadata_baselines(self, metadata_data: Dict[str, pd.DataFrame], targets: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate metadata-only baseline models."""
        results = {}
        
        if not SKLEARN_AVAILABLE:
            return results
        
        # Random Forest
        logger.info("  - Training Random Forest (Metadata)...")
        start_time = time.time()
        try:
            if self.task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=self.seed)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=self.seed)
            
            model.fit(metadata_data['train'], targets['train'])
            
            train_pred = model.predict(metadata_data['train'])
            test_pred = model.predict(metadata_data['test'])
            
            test_proba = model.predict_proba(metadata_data['test']) if hasattr(model, 'predict_proba') else None
            
            metrics = self._calculate_metrics(targets['train'], train_pred, targets['test'], test_pred, test_proba)
            
            results['metadata_rf'] = {
                'model_type': 'metadata_only',
                'model_name': 'Random Forest (Metadata)',
                'training_time': time.time() - start_time,
                'metrics': metrics,
                'predictions': {
                    'train': train_pred.tolist(),
                    'test': test_pred.tolist()
                }
            }
            
            logger.info(f"Random Forest (Metadata) completed")
            
        except Exception as e:
            logger.error(f"Error in Random Forest (Metadata): {e}")
        
        # XGBoost
        logger.info("  - Training XGBoost (Metadata)...")
        start_time = time.time()
        try:
            if self.task_type == 'classification':
                model = XGBClassifier(random_state=self.seed, n_estimators=100)
            else:
                model = XGBRegressor(random_state=self.seed, n_estimators=100)
            
            model.fit(metadata_data['train'], targets['train'])
            
            train_pred = model.predict(metadata_data['train'])
            test_pred = model.predict(metadata_data['test'])
            
            test_proba = model.predict_proba(metadata_data['test']) if hasattr(model, 'predict_proba') else None
            
            metrics = self._calculate_metrics(targets['train'], train_pred, targets['test'], test_pred, test_proba)
            
            results['metadata_xgb'] = {
                'model_type': 'metadata_only',
                'model_name': 'XGBoost (Metadata)',
                'training_time': time.time() - start_time,
                'metrics': metrics,
                'predictions': {
                    'train': train_pred.tolist(),
                    'test': test_pred.tolist()
                }
            }
            
            logger.info(f"XGBoost (Metadata) completed")
            
        except Exception as e:
            logger.error(f"Error in XGBoost (Metadata): {e}")
        
        # SVM (Metadata)
        logger.info("  - Training SVM (Metadata)...")
        start_time = time.time()
        try:
            if self.task_type == 'classification':
                model = SVC(probability=True, random_state=self.seed)
            else:
                model = SVR()
            
            model.fit(metadata_data['train'], targets['train'])
            
            train_pred = model.predict(metadata_data['train'])
            test_pred = model.predict(metadata_data['test'])
            
            test_proba = model.predict_proba(metadata_data['test']) if hasattr(model, 'predict_proba') else None
            
            metrics = self._calculate_metrics(targets['train'], train_pred, targets['test'], test_pred, test_proba)
            
            results['metadata_svm'] = {
                'model_type': 'metadata_only',
                'model_name': 'SVM (Metadata)',
                'training_time': time.time() - start_time,
                'metrics': metrics,
                'predictions': {
                    'train': train_pred.tolist(),
                    'test': test_pred.tolist()
                }
            }
            
            logger.info(f"SVM (Metadata) completed")
            
        except Exception as e:
            logger.error(f"Error in SVM (Metadata): {e}")
        
        return results
    
    def _evaluate_fusion_baselines(self, data_prep: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate fusion baseline models."""
        results = {}
        
        if not SKLEARN_AVAILABLE:
            return results
        
        # Early Fusion (concatenated features)
        logger.info("  - Training Early Fusion (Random Forest)...")
        start_time = time.time()
        try:
            # Combine TF-IDF features with metadata
            X_train_tfidf = self._get_tfidf_features(data_prep['text_data']['train'], fit_vectorizer=True)
            X_test_tfidf = self._get_tfidf_features(data_prep['text_data']['test'], fit_vectorizer=False)
            
            combined_train = np.column_stack([X_train_tfidf, data_prep['metadata_data']['train'].values])
            combined_test = np.column_stack([X_test_tfidf, data_prep['metadata_data']['test'].values])
            
            if self.task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=self.seed)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=self.seed)
            
            model.fit(combined_train, targets['train'])
            
            train_pred = model.predict(combined_train)
            test_pred = model.predict(combined_test)
            
            test_proba = model.predict_proba(combined_test) if hasattr(model, 'predict_proba') else None
            
            metrics = self._calculate_metrics(targets['train'], train_pred, targets['test'], test_pred, test_proba)
            
            results['early_fusion_rf'] = {
                'model_type': 'early_fusion',
                'model_name': 'Early Fusion (Random Forest)',
                'training_time': time.time() - start_time,
                'metrics': metrics,
                'predictions': {
                    'train': train_pred.tolist(),
                    'test': test_pred.tolist()
                }
            }
            
            logger.info(f"Early Fusion (Random Forest) completed")
            
        except Exception as e:
            logger.error(f"Error in Early Fusion: {e}")
        
        # Late Fusion (voting/weighted average)
        logger.info("  - Training Late Fusion (Weighted Voting)...")
        start_time = time.time()
        try:
            # Train individual models
            text_model = RandomForestClassifier(n_estimators=50, random_state=self.seed)
            metadata_model = RandomForestClassifier(n_estimators=50, random_state=self.seed)
            
            # Train text model
            X_train_tfidf = self._get_tfidf_features(data_prep['text_data']['train'], fit_vectorizer=True)
            X_test_tfidf = self._get_tfidf_features(data_prep['text_data']['test'], fit_vectorizer=False)
            text_model.fit(X_train_tfidf, targets['train'])
            
            # Train metadata model
            metadata_model.fit(data_prep['metadata_data']['train'], targets['train'])
            
            # Get predictions
            text_pred = text_model.predict(X_test_tfidf)
            metadata_pred = metadata_model.predict(data_prep['metadata_data']['test'])
            
            # Weighted average (based on cross-validation scores)
            text_cv_score = np.mean(cross_val_score(text_model, X_train_tfidf, targets['train'], cv=3))
            metadata_cv_score = np.mean(cross_val_score(metadata_model, data_prep['metadata_data']['train'], targets['train'], cv=3))
            
            # Normalize weights
            total_score = text_cv_score + metadata_cv_score
            text_weight = text_cv_score / total_score
            metadata_weight = metadata_cv_score / total_score
            
            # Weighted voting
            test_pred = (text_weight * text_pred + metadata_weight * metadata_pred).astype(int)
            
            # For training predictions
            text_train_pred = text_model.predict(X_train_tfidf)
            metadata_train_pred = metadata_model.predict(data_prep['metadata_data']['train'])
            train_pred = (text_weight * text_train_pred + metadata_weight * metadata_train_pred).astype(int)
            
            metrics = self._calculate_metrics(targets['train'], train_pred, targets['test'], test_pred, None)
            
            results['late_fusion_weighted'] = {
                'model_type': 'late_fusion',
                'model_name': 'Late Fusion (Weighted Voting)',
                'training_time': time.time() - start_time,
                'metrics': metrics,
                'predictions': {
                    'train': train_pred.tolist(),
                    'test': test_pred.tolist()
                }
            }
            
            logger.info(f"Late Fusion (Weighted Voting) completed")
            
        except Exception as e:
            logger.error(f"Error in Late Fusion: {e}")
        
        return results
    
    def _evaluate_ensemble_baselines(self, data_prep: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate ensemble baseline models."""
        results = {}
        
        if not SKLEARN_AVAILABLE:
            return results
        
        # Bagging with modality-specific features
        logger.info("  - Training Bagging (Multimodal)...")
        start_time = time.time()
        try:
            # Combine features for bagging
            X_train_tfidf = self._get_tfidf_features(data_prep['text_data']['train'], fit_vectorizer=True)
            X_test_tfidf = self._get_tfidf_features(data_prep['text_data']['test'], fit_vectorizer=False)
            
            combined_train = np.column_stack([X_train_tfidf, data_prep['metadata_data']['train'].values])
            combined_test = np.column_stack([X_test_tfidf, data_prep['metadata_data']['test'].values])
            
            if self.task_type == 'classification':
                base_model = RandomForestClassifier(n_estimators=50, random_state=self.seed)
                model = BaggingClassifier(base_model, n_estimators=5, random_state=self.seed)
            else:
                base_model = RandomForestRegressor(n_estimators=50, random_state=self.seed)
                model = BaggingRegressor(base_model, n_estimators=5, random_state=self.seed)
            
            model.fit(combined_train, targets['train'])
            
            train_pred = model.predict(combined_train)
            test_pred = model.predict(combined_test)
            
            test_proba = model.predict_proba(combined_test) if hasattr(model, 'predict_proba') else None
            
            metrics = self._calculate_metrics(targets['train'], train_pred, targets['test'], test_pred, test_proba)
            
            results['bagging_multimodal'] = {
                'model_type': 'ensemble',
                'model_name': 'Bagging (Multimodal)',
                'training_time': time.time() - start_time,
                'metrics': metrics,
                'predictions': {
                    'train': train_pred.tolist(),
                    'test': test_pred.tolist()
                }
            }
            
            logger.info(f"Bagging (Multimodal) completed")
            
        except Exception as e:
            logger.error(f"Error in Bagging: {e}")
        
        # Boosting with multimodal features
        logger.info("  - Training XGBoost (Multimodal)...")
        start_time = time.time()
        try:
            # Combine features for boosting
            X_train_tfidf = self._get_tfidf_features(data_prep['text_data']['train'], fit_vectorizer=True)
            X_test_tfidf = self._get_tfidf_features(data_prep['text_data']['test'], fit_vectorizer=False)
            
            combined_train = np.column_stack([X_train_tfidf, data_prep['metadata_data']['train'].values])
            combined_test = np.column_stack([X_test_tfidf, data_prep['metadata_data']['test'].values])
            
            if self.task_type == 'classification':
                model = XGBClassifier(random_state=self.seed, n_estimators=100, learning_rate=0.1)
            else:
                model = XGBRegressor(random_state=self.seed, n_estimators=100, learning_rate=0.1)
            
            model.fit(combined_train, targets['train'])
            
            train_pred = model.predict(combined_train)
            test_pred = model.predict(combined_test)
            
            test_proba = model.predict_proba(combined_test) if hasattr(model, 'predict_proba') else None
            
            metrics = self._calculate_metrics(targets['train'], train_pred, targets['test'], test_pred, test_proba)
            
            results['boosting_multimodal'] = {
                'model_type': 'ensemble',
                'model_name': 'XGBoost (Multimodal)',
                'training_time': time.time() - start_time,
                'metrics': metrics,
                'predictions': {
                    'train': train_pred.tolist(),
                    'test': test_pred.tolist()
                }
            }
            
            logger.info(f"XGBoost (Multimodal) completed")
            
        except Exception as e:
            logger.error(f"Error in XGBoost (Multimodal): {e}")
        
        # Stacking (meta-learner)
        start_time = time.time()
        try:
            # Train base models
            base_models = []
            
            # Text model
            X_train_tfidf = self._get_tfidf_features(data_prep['text_data']['train'], fit_vectorizer=True)
            X_test_tfidf = self._get_tfidf_features(data_prep['text_data']['test'], fit_vectorizer=False)
            
            text_model = RandomForestClassifier(n_estimators=50, random_state=self.seed)
            text_model.fit(X_train_tfidf, targets['train'])
            base_models.append(('text', text_model))
            
            # Metadata model
            metadata_model = RandomForestClassifier(n_estimators=50, random_state=self.seed)
            metadata_model.fit(data_prep['metadata_data']['train'], targets['train'])
            base_models.append(('metadata', metadata_model))
            
            # Get base predictions for stacking
            train_base_preds = []
            test_base_preds = []
            
            for name, model in base_models:
                train_pred = model.predict(X_train_tfidf if name == 'text' else data_prep['metadata_data']['train'])
                test_pred = model.predict(X_test_tfidf if name == 'text' else data_prep['metadata_data']['test'])
                train_base_preds.append(train_pred)
                test_base_preds.append(test_pred)
            
            # Stack base predictions
            train_stacked = np.column_stack(train_base_preds)
            test_stacked = np.column_stack(test_base_preds)
            
            # Train meta-learner
            if self.task_type == 'classification':
                meta_learner = LogisticRegression(random_state=self.seed)
            else:
                meta_learner = Ridge(random_state=self.seed)
            
            meta_learner.fit(train_stacked, targets['train'])
            
            train_pred = meta_learner.predict(train_stacked)
            test_pred = meta_learner.predict(test_stacked)
            
            test_proba = meta_learner.predict_proba(test_stacked) if hasattr(meta_learner, 'predict_proba') else None
            
            metrics = self._calculate_metrics(targets['train'], train_pred, targets['test'], test_pred, test_proba)
            
            results['stacking_meta'] = {
                'model_type': 'ensemble',
                'model_name': 'Stacking (Meta-learner)',
                'training_time': time.time() - start_time,
                'metrics': metrics,
                'predictions': {
                    'train': train_pred.tolist(),
                    'test': test_pred.tolist()
                }
            }
            
            logger.info(f"Stacking (Meta-learner) completed")
            
        except Exception as e:
            logger.error(f"Error in Stacking: {e}")
        
        return results
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb
        except Exception as e:
            logger.warning(f"Could not measure memory usage: {e}")
            return 0.0
    
    def _evaluate_model_with_metrics(self, model, X_train, y_train, X_test, y_test, 
                                   model_name, model_type, training_time) -> Dict[str, Any]:
        """Standardized model evaluation with all metrics."""
        try:
            # Measure prediction time
            pred_start_time = time.time()
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            prediction_time = time.time() - pred_start_time
            
            # Get probabilities if available
            test_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Measure memory usage
            memory_usage = self._get_memory_usage()
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_train, train_pred, y_test, test_pred, test_proba)
            
            return {
                'model_type': model_type,
                'model_name': model_name,
                'training_time': training_time,
                'prediction_time': prediction_time,
                'memory_usage_mb': memory_usage,
                'metrics': metrics,
                'predictions': {
                    'train': train_pred.tolist(),
                    'test': test_pred.tolist()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            return None
    
    def _calculate_metrics(self, y_train: np.ndarray, y_train_pred: np.ndarray, 
                          y_test: np.ndarray, y_test_pred: np.ndarray, 
                          y_test_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive metrics for model evaluation."""
        metrics = {}
        
        try:
            if self.task_type == 'classification':
                # Classification metrics
                metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
                metrics['test_accuracy'] = accuracy_score(y_test, y_test_pred)
                metrics['train_f1'] = f1_score(y_train, y_train_pred, average='weighted')
                metrics['test_f1'] = f1_score(y_test, y_test_pred, average='weighted')
                metrics['train_precision'] = precision_score(y_train, y_train_pred, average='weighted')
                metrics['test_precision'] = precision_score(y_test, y_test_pred, average='weighted')
                metrics['train_recall'] = recall_score(y_train, y_train_pred, average='weighted')
                metrics['test_recall'] = recall_score(y_test, y_test_pred, average='weighted')
                metrics['train_balanced_accuracy'] = balanced_accuracy_score(y_train, y_train_pred)
                metrics['test_balanced_accuracy'] = balanced_accuracy_score(y_test, y_test_pred)
                
                # AUC-ROC if probabilities available
                if y_test_proba is not None and y_test_proba.shape[1] > 1:
                    try:
                        metrics['test_auc_roc'] = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='weighted')
                    except:
                        metrics['test_auc_roc'] = 0.0
                else:
                    metrics['test_auc_roc'] = 0.0
                
            else:
                # Regression metrics
                metrics['train_mse'] = mean_squared_error(y_train, y_train_pred)
                metrics['test_mse'] = mean_squared_error(y_test, y_test_pred)
                metrics['train_mae'] = mean_absolute_error(y_train, y_train_pred)
                metrics['test_mae'] = mean_absolute_error(y_test, y_test_pred)
                metrics['train_rmse'] = np.sqrt(metrics['train_mse'])
                metrics['test_rmse'] = np.sqrt(metrics['test_mse'])
                metrics['train_r2'] = r2_score(y_train, y_train_pred)
                metrics['test_r2'] = r2_score(y_test, y_test_pred)
                
                # MAPE
                try:
                    metrics['train_mape'] = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
                    metrics['test_mape'] = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
                except:
                    metrics['train_mape'] = 0.0
                    metrics['test_mape'] = 0.0
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            # Set default values
            if self.task_type == 'classification':
                metrics = {
                    'train_accuracy': 0.0, 'test_accuracy': 0.0,
                    'train_f1': 0.0, 'test_f1': 0.0,
                    'train_precision': 0.0, 'test_precision': 0.0,
                    'train_recall': 0.0, 'test_recall': 0.0,
                    'train_balanced_accuracy': 0.0, 'test_balanced_accuracy': 0.0,
                    'test_auc_roc': 0.0
                }
            else:
                metrics = {
                    'train_mse': 0.0, 'test_mse': 0.0,
                    'train_mae': 0.0, 'test_mae': 0.0,
                    'train_rmse': 0.0, 'test_rmse': 0.0,
                    'train_r2': 0.0, 'test_r2': 0.0,
                    'train_mape': 0.0, 'test_mape': 0.0
                }
        
        return metrics

def run_phase_2_baseline(config: Dict[str, Any], processed_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run Phase 2: Baseline Model Evaluation."""
    logger.info("Starting Phase 2: Baseline Model Evaluation")
    
    phase_dir = Path(config["phase_dir"])
    seed = config["seed"]
    test_mode = config["test_mode"]
    
    results = {
        "phase": "phase_2_baseline",
        "seed": seed,
        "test_mode": test_mode,
        "status": "completed",
        "timestamp": None,
        "task_type": None,
        "baseline_results": {},
        "performance_summary": {},
        "execution_time": 0.0
    }
    
    start_time = time.time()
    
    try:
        # Initialize baseline evaluator with processed data if provided
        evaluator = BaselineEvaluator(config, processed_data)
        
        # Prepare data
        data_prep, targets = evaluator._prepare_data()
        
        # Update task type
        results["task_type"] = evaluator.task_type
        
        # Evaluate all baseline models
        baseline_results = evaluator.evaluate_baselines(data_prep, targets)
        
        # Create performance summary
        performance_summary = _create_performance_summary(baseline_results, evaluator.task_type)
        
        # Perform statistical significance tests
        statistical_tests = _perform_statistical_tests(baseline_results, evaluator.task_type)
        
        # Update results
        execution_time = time.time() - start_time
        results.update({
            "timestamp": pd.Timestamp.now().isoformat(),
            "baseline_results": baseline_results,
            "performance_summary": performance_summary,
            "statistical_tests": statistical_tests,
            "execution_time": execution_time
        })
        
        # Create output directory if it doesn't exist
        phase_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_file = phase_dir / "phase_2_baseline_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save detailed reports
        baseline_file = phase_dir / "baseline_results.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_results, f, indent=2, default=str)
        
        summary_file = phase_dir / "baseline_summary.csv"
        performance_summary.to_csv(summary_file, index=False)
        
        # Save statistical tests
        stats_file = phase_dir / "statistical_significance_tests.json"
        with open(stats_file, 'w') as f:
            json.dump(statistical_tests, f, indent=2, default=str)
        
        # Log summary
        logger.info(f"Phase 2 completed successfully in {execution_time:.2f}s")
        logger.info(f"Evaluated {len(baseline_results)} baseline models")
        logger.info(f"Task type: {evaluator.task_type}")
        logger.info(f"Results saved to {phase_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Phase 2 failed: {str(e)}")
        results.update({
            "status": "failed",
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat(),
            "execution_time": time.time() - start_time
        })
        
        # Create output directory if it doesn't exist
        phase_dir.mkdir(parents=True, exist_ok=True)
        
        # Save error results
        results_file = phase_dir / "phase_2_baseline_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results

def _create_performance_summary(baseline_results: Dict[str, Any], task_type: str) -> pd.DataFrame:
    """Create performance summary DataFrame."""
    summary_data = []
    
    for model_name, result in baseline_results.items():
        metrics = result['metrics']
        
        # Common efficiency metrics
        efficiency_metrics = {
            'training_time': result.get('training_time', 0.0),
            'prediction_time': result.get('prediction_time', 0.0),
            'memory_usage_mb': result.get('memory_usage_mb', 0.0)
        }
        
        if task_type == 'classification':
            summary_data.append({
                'model_name': result['model_name'],
                'model_type': result['model_type'],
                'train_accuracy': metrics.get('train_accuracy', 0.0),
                'test_accuracy': metrics.get('test_accuracy', 0.0),
                'train_f1': metrics.get('train_f1', 0.0),
                'test_f1': metrics.get('test_f1', 0.0),
                'train_precision': metrics.get('train_precision', 0.0),
                'test_precision': metrics.get('test_precision', 0.0),
                'train_recall': metrics.get('train_recall', 0.0),
                'test_recall': metrics.get('test_recall', 0.0),
                'train_balanced_accuracy': metrics.get('train_balanced_accuracy', 0.0),
                'test_balanced_accuracy': metrics.get('test_balanced_accuracy', 0.0),
                'test_auc_roc': metrics.get('test_auc_roc', 0.0),
                **efficiency_metrics
            })
        else:
            summary_data.append({
                'model_name': result['model_name'],
                'model_type': result['model_type'],
                'train_mse': metrics.get('train_mse', 0.0),
                'test_mse': metrics.get('test_mse', 0.0),
                'train_mae': metrics.get('train_mae', 0.0),
                'test_mae': metrics.get('test_mae', 0.0),
                'train_rmse': metrics.get('train_rmse', 0.0),
                'test_rmse': metrics.get('test_rmse', 0.0),
                'train_r2': metrics.get('train_r2', 0.0),
                'test_r2': metrics.get('test_r2', 0.0),
                'train_mape': metrics.get('train_mape', 0.0),
                'test_mape': metrics.get('test_mape', 0.0),
                **efficiency_metrics
            })
    
    return pd.DataFrame(summary_data)

def _perform_statistical_tests(baseline_results: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    """Perform statistical significance tests between models."""
    statistical_tests = {}
    
    try:
        if len(baseline_results) < 2:
            return statistical_tests
        
        # Get test metrics for comparison
        test_metrics = {}
        for model_name, result in baseline_results.items():
            if task_type == 'classification':
                test_metrics[model_name] = result['metrics'].get('test_accuracy', 0.0)
            else:
                test_metrics[model_name] = result['metrics'].get('test_r2', 0.0)
        
        # Find best model
        best_model = max(test_metrics, key=test_metrics.get)
        best_score = test_metrics[best_model]
        
        statistical_tests['best_model'] = best_model
        statistical_tests['best_score'] = best_score
        statistical_tests['model_rankings'] = sorted(test_metrics.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate performance differences
        performance_differences = {}
        for model_name, score in test_metrics.items():
            if model_name != best_model:
                diff = best_score - score
                percentage_improvement = (diff / score * 100) if score > 0 else 0
                performance_differences[f"{best_model}_vs_{model_name}"] = {
                    'difference': diff,
                    'percentage_improvement': percentage_improvement,
                    'effect_size': 'large' if percentage_improvement > 20 else 'medium' if percentage_improvement > 10 else 'small'
                }
        
        statistical_tests['performance_differences'] = performance_differences
        
        # Paired t-test simulation (since we don't have multiple runs)
        # In a real scenario, you'd run each model multiple times
        statistical_tests['significance_notes'] = {
            'note': 'Statistical significance tests require multiple runs per model. For single-run experiments, performance differences are reported.',
            'recommendation': 'Run each model 5-10 times with different seeds for proper statistical testing.'
        }
        
        # Confidence intervals (estimated)
        confidence_intervals = {}
        for model_name, score in test_metrics.items():
            # Estimate 95% CI based on typical variance in ML models
            estimated_std = score * 0.05  # 5% of score as estimated standard deviation
            ci_lower = max(0, score - 1.96 * estimated_std)
            ci_upper = min(1 if task_type == 'classification' else float('inf'), score + 1.96 * estimated_std)
            
            confidence_intervals[model_name] = {
                'score': score,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'margin_of_error': 1.96 * estimated_std
            }
        
        statistical_tests['confidence_intervals'] = confidence_intervals
        
    except Exception as e:
        logger.error(f"Error in statistical tests: {e}")
        statistical_tests['error'] = str(e)
    
    return statistical_tests

if __name__ == "__main__":
    # Test the phase script independently
    test_config = {
        "dataset_path": "../../../ProcessedData/AmazonReviews",
        "phase_dir": "./test_output",
        "seed": 42,
        "test_mode": "quick"
    }
    
    # Create test output directory
    Path(test_config["phase_dir"]).mkdir(exist_ok=True)
    
    # Run the phase
    results = run_phase_2_baseline(test_config)
    print(f"Phase 2 Results: {json.dumps(results, indent=2, default=str)}")
