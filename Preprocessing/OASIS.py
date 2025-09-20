"""
OASIS Dataset Preprocessing Script

This script preprocesses the OASIS dataset into a standardized format
for multimodal learning. It creates a single tabular modality with
multiple feature categories:
- Demographic features (gender, handedness, age, education, SES)
- Clinical features (MMSE, CDR)
- Brain volume features (eTIV, nWBV, ASF)
- Temporal features (visit, MR delay)

Based on the documentation in PreprocessingDocumentation.md
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_oasis_data():
    """
    Preprocess OASIS dataset into standardized format
    """
    logger.info("Starting OASIS preprocessing...")
    
    # Create output directory
    processed_path = "ProcessedData/OASIS"
    os.makedirs(processed_path, exist_ok=True)
    logger.info(f"Created output directory: {processed_path}")
    
    # Load raw data
    try:
        cross_df = pd.read_csv('Data/OASIS/oasis_cross-sectional.csv')
        long_df = pd.read_csv('Data/OASIS/oasis_longitudinal.csv')
        logger.info(f"Loaded cross-sectional data: {len(cross_df)} subjects")
        logger.info(f"Loaded longitudinal data: {len(long_df)} visits")
    except FileNotFoundError as e:
        logger.error(f"Raw data files not found: {e}")
        raise
    
    # Create labels for cross-sectional data
    logger.info("Creating labels for cross-sectional data...")
    cross_labels = cross_df[['ID', 'CDR']].copy()
    cross_labels['sample_id'] = 'cross_' + cross_labels['ID'].astype(str)
    cross_labels['dementia_binary'] = (cross_labels['CDR'] > 0).astype(int)
    cross_labels['dementia_severity'] = cross_labels['CDR'].map({
        0.0: 'Normal', 0.5: 'Very_Mild', 1.0: 'Mild', 2.0: 'Moderate'
    })
    
    # Create labels for longitudinal data
    logger.info("Creating labels for longitudinal data...")
    long_labels = long_df[['Subject ID', 'CDR']].copy()
    long_labels['sample_id'] = 'long_' + long_df['MRI ID'].astype(str)
    long_labels['dementia_binary'] = (long_labels['CDR'] > 0).astype(int)
    long_labels['dementia_severity'] = long_labels['CDR'].map({
        0.0: 'Normal', 0.5: 'Very_Mild', 1.0: 'Mild', 2.0: 'Moderate'
    })
    
    # Combine all labels
    all_labels = pd.concat([cross_labels, long_labels], ignore_index=True)
    logger.info(f"Combined labels: {len(all_labels)} total samples")
    
    # Extract features
    logger.info("Extracting tabular features...")
    features_df = extract_all_tabular_features(cross_df, long_df)
    logger.info(f"Extracted features for {len(features_df)} samples")
    
    # Create metadata
    logger.info("Creating metadata...")
    metadata = create_oasis_metadata(features_df, cross_labels, long_labels)
    
    # Save processed data
    logger.info("Saving processed data...")
    all_labels.to_csv(f"{processed_path}/labels.csv", index=False)
    features_df.to_csv(f"{processed_path}/tabular_features.csv", index=False)
    
    with open(f"{processed_path}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("OASIS preprocessing completed!")
    logger.info(f"Processed data saved to: {processed_path}")
    logger.info(f"Total samples: {len(features_df)}")
    
    return all_labels, features_df, metadata

def extract_all_tabular_features(cross_df, long_df):
    """
    Extract all features and consolidate into a single tabular features file
    """
    all_features_data = []
    
    logger.info("Processing cross-sectional data...")
    # Process cross-sectional data
    for idx, row in cross_df.iterrows():
        sample_id = f"cross_{row['ID']}"
        
        # Handle missing values with sensible defaults
        educ = row['Educ'] if pd.notna(row['Educ']) else 3.0  # Default to middle education
        ses = row['SES'] if pd.notna(row['SES']) else 3.0     # Default to middle SES
        mmse = row['MMSE'] if pd.notna(row['MMSE']) else 27.0 # Default to normal MMSE
        cdr = row['CDR'] if pd.notna(row['CDR']) else 0.0     # Default to normal CDR
        
        # Encode categorical variables
        gender_encoded = 1 if row['M/F'] == 'M' else 0
        hand_encoded = 1 if row['Hand'] == 'R' else 0
        
        all_features_data.append({
            'sample_id': sample_id,
            # Demographic features (6)
            'gender': gender_encoded,
            'handedness': hand_encoded,
            'age': row['Age'],
            'education': educ,
            'ses': ses,
            # Clinical features (2)
            'mmse': mmse,
            'cdr': cdr,
            # Brain volume features (3)
            'etiv': row['eTIV'],
            'nwbv': row['nWBV'],
            'asf': row['ASF'],
            # Temporal features (2) - set to 0 for cross-sectional
            'visit': 1,
            'mr_delay': 0,
            # Dataset indicator
            'dataset_type': 'cross_sectional'
        })
    
    logger.info("Processing longitudinal data...")
    # Process longitudinal data
    for idx, row in long_df.iterrows():
        sample_id = f"long_{row['MRI ID']}"
        
        # Handle missing values
        ses = row['SES'] if pd.notna(row['SES']) else 3.0
        mmse = row['MMSE'] if pd.notna(row['MMSE']) else 27.0
        
        # Encode categorical variables
        gender_encoded = 1 if row['M/F'] == 'M' else 0
        hand_encoded = 1 if row['Hand'] == 'R' else 0
        
        all_features_data.append({
            'sample_id': sample_id,
            # Demographic features (6)
            'gender': gender_encoded,
            'handedness': hand_encoded,
            'age': row['Age'],
            'education': row['EDUC'],
            'ses': ses,
            # Clinical features (2)
            'mmse': mmse,
            'cdr': row['CDR'],
            # Brain volume features (3)
            'etiv': row['eTIV'],
            'nwbv': row['nWBV'],
            'asf': row['ASF'],
            # Temporal features (2)
            'visit': row['Visit'],
            'mr_delay': row['MR Delay'],
            # Dataset indicator
            'dataset_type': 'longitudinal'
        })
    
    return pd.DataFrame(all_features_data)

def create_oasis_metadata(features_df, cross_labels, long_labels):
    """
    Create metadata file for OASIS dataset
    """
    metadata = {
        "dataset_info": {
            "name": "OASIS Processed Dataset",
            "version": "1.0",
            "creation_date": datetime.now().isoformat(),
            "source": "OASIS Raw Dataset",
            "preprocessing_version": "1.0"
        },
        "data_summary": {
            "total_samples": len(features_df),
            "cross_sectional_samples": len(cross_labels),
            "longitudinal_samples": len(long_labels),
            "modalities": ["tabular"],
            "feature_categories": ["demographic", "clinical", "brain_volume", "temporal"]
        },
        "modality_details": {
            "tabular": {
                "type": "structured_data",
                "format": "CSV",
                "features": [
                    "gender", "handedness", "age", "education", "ses",
                    "mmse", "cdr", "etiv", "nwbv", "asf", "visit", "mr_delay"
                ],
                "feature_categories": {
                    "demographic": ["gender", "handedness", "age", "education", "ses"],
                    "clinical": ["mmse", "cdr"],
                    "brain_volume": ["etiv", "nwbv", "asf"],
                    "temporal": ["visit", "mr_delay"]
                }
            }
        },
        "labels": {
            "cdr_distribution": {
                "0.0": int(cross_labels['CDR'].value_counts().get(0.0, 0) + 
                           long_labels['CDR'].value_counts().get(0.0, 0)),
                "0.5": int(cross_labels['CDR'].value_counts().get(0.5, 0) + 
                           long_labels['CDR'].value_counts().get(0.5, 0)),
                "1.0": int(cross_labels['CDR'].value_counts().get(1.0, 0) + 
                           long_labels['CDR'].value_counts().get(1.0, 0)),
                "2.0": int(cross_labels['CDR'].value_counts().get(2.0, 0) + 
                           long_labels['CDR'].value_counts().get(2.0, 0))
            },
            "dementia_binary_distribution": {
                "normal": int((cross_labels['dementia_binary'] == 0).sum() + 
                             (long_labels['dementia_binary'] == 0).sum()),
                "dementia": int((cross_labels['dementia_binary'] == 1).sum() + 
                               (long_labels['dementia_binary'] == 1).sum())
            }
        },
        "preprocessing_info": {
            "missing_data_handling": "Default values for missing Educ, SES, MMSE, CDR",
            "categorical_encoding": "Binary encoding for gender and handedness",
            "feature_scaling": "Not applied (raw values preserved)",
            "data_splits": "Not created (to be determined by user)"
        },
        "computational_requirements": {
            "processing_time": "~1-2 minutes (small dataset)",
            "memory_usage": "~100MB RAM",
            "storage_requirements": "~5MB for processed data",
            "dependencies": ["pandas", "numpy", "json"]
        }
    }
    
    return metadata

def validate_oasis_preprocessing(processed_path):
    """
    Comprehensive validation of OASIS preprocessing results
    """
    logger.info("Running comprehensive validation...")
    
    validation_results = {
        'overall_status': 'PASS',
        'errors': [],
        'warnings': [],
        'sample_counts': {},
        'quality_metrics': {},
        'missing_data_percentages': {}
    }
    
    try:
        # Check required files exist
        required_files = ['labels.csv', 'tabular_features.csv', 'metadata.json']
        for file in required_files:
            if not os.path.exists(f"{processed_path}/{file}"):
                validation_results['errors'].append(f"Missing required file: {file}")
        
        if validation_results['errors']:
            validation_results['overall_status'] = 'FAIL'
            return validation_results
        
        # Load and validate labels
        labels_df = pd.read_csv(f"{processed_path}/labels.csv")
        validation_results['sample_counts']['labels'] = len(labels_df)
        
        # Load and validate features
        features_df = pd.read_csv(f"{processed_path}/tabular_features.csv")
        validation_results['sample_counts']['features'] = len(features_df)
        
        # Check sample count consistency
        if len(labels_df) != len(features_df):
            validation_results['errors'].append(
                f"Sample count mismatch: labels={len(labels_df)}, features={len(features_df)}"
            )
        
        # Validate label distribution
        if 'cdr' in labels_df.columns:
            cdr_counts = labels_df['cdr'].value_counts()
            validation_results['quality_metrics']['cdr_distribution'] = cdr_counts.to_dict()
            
            # Check for expected CDR distribution
            expected_cdr_values = [0.0, 0.5, 1.0, 2.0]
            for cdr_val in expected_cdr_values:
                if cdr_val not in cdr_counts.index:
                    validation_results['warnings'].append(f"Missing CDR value: {cdr_val}")
        
        # Validate feature quality
        validation_results['quality_metrics']['feature_columns'] = len(features_df.columns)
        validation_results['quality_metrics']['feature_rows'] = len(features_df)
        
        # Check for missing data
        missing_data = features_df.isnull().sum().sum()
        total_cells = len(features_df) * len(features_df.columns)
        missing_percentage = (missing_data / total_cells) * 100
        validation_results['missing_data_percentages']['features'] = missing_percentage
        
        if missing_percentage > 5:  # More than 5% missing data
            validation_results['warnings'].append(f"High missing data in features: {missing_percentage:.2f}%")
        
        # Validate data types
        expected_numeric_columns = ['age', 'mmse', 'etiv', 'nwbv', 'asf']
        for col in expected_numeric_columns:
            if col in features_df.columns:
                if not pd.api.types.is_numeric_dtype(features_df[col]):
                    validation_results['warnings'].append(f"Column {col} should be numeric but is {features_df[col].dtype}")
        
        # Validate value ranges
        if 'age' in features_df.columns:
            age_range = features_df['age'].min(), features_df['age'].max()
            if age_range[0] < 18 or age_range[1] > 100:
                validation_results['warnings'].append(f"Unusual age range: {age_range}")
        
        if 'mmse' in features_df.columns:
            mmse_range = features_df['mmse'].min(), features_df['mmse'].max()
            if mmse_range[0] < 0 or mmse_range[1] > 30:
                validation_results['warnings'].append(f"Unusual MMSE range: {mmse_range}")
        
        # Calculate overall quality score
        total_checks = 12  # Approximate number of checks
        error_penalty = len(validation_results['errors']) * 3
        warning_penalty = len(validation_results['warnings']) * 1
        validation_results['quality_score'] = max(0, (total_checks - error_penalty - warning_penalty) / total_checks * 100)
        
        if validation_results['errors']:
            validation_results['overall_status'] = 'FAIL'
        elif validation_results['warnings']:
            validation_results['overall_status'] = 'WARN'
        
    except Exception as e:
        validation_results['errors'].append(f"Validation failed with exception: {e}")
        validation_results['overall_status'] = 'FAIL'
    
    return validation_results

class OASISDataset:
    """
    PyTorch-style Dataset for OASIS processed data
    """
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        
        # Load data
        self.labels_df = pd.read_csv(f"{data_path}/labels.csv")
        self.features_df = pd.read_csv(f"{data_path}/tabular_features.csv")
        
        # Merge labels and features
        self.data = pd.merge(self.labels_df, self.features_df, on='sample_id', how='inner')
        
        logger.info(f"Loaded OASIS dataset: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Extract features (exclude sample_id, labels, and non-numeric columns)
        feature_columns = [col for col in self.features_df.columns 
                          if col not in ['sample_id', 'dataset_type']]
        features = row[feature_columns].values.astype(np.float32)
        
        # Apply transform if provided
        if self.transform:
            features = self.transform(features)
        
        return {
            'features': features,
            'label': row['cdr'],
            'dementia_binary': row['dementia_binary'],
            'dementia_severity': row['dementia_severity'],
            'sample_id': row['sample_id']
        }

def load_oasis_for_mainmodel(data_path="ProcessedData/OASIS"):
    """
    Load preprocessed OASIS data in a format suitable for the MainModel API
    """
    if not os.path.exists(data_path):
        logger.error(f"Processed data path not found: {data_path}. Please run preprocessing first.")
        return None
    
    # Create dataset
    dataset = OASISDataset(data_path)
    
    # Load metadata
    with open(f"{data_path}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return {
        'dataset': dataset,
        'labels': dataset.labels_df,
        'features': dataset.features_df,
        'metadata': metadata,
        'modality_types': ['tabular']
    }

if __name__ == "__main__":
    logger.info("✅ OASIS preprocessing script imported successfully!")
    
    # Test raw data access
    raw_data_path = "Data/OASIS"
    if os.path.exists(raw_data_path):
        logger.info("✅ OASIS raw data found")
        
        # Check for required files
        cross_file = f"{raw_data_path}/oasis_cross-sectional.csv"
        long_file = f"{raw_data_path}/oasis_longitudinal.csv"
        
        if os.path.exists(cross_file) and os.path.exists(long_file):
            logger.info("✅ Both OASIS CSV files found")
            
            # Test loading
            try:
                cross_df = pd.read_csv(cross_file)
                long_df = pd.read_csv(long_file)
                logger.info(f"✅ Cross-sectional data: {len(cross_df)} subjects")
                logger.info(f"✅ Longitudinal data: {len(long_df)} visits")
            except Exception as e:
                logger.error(f"❌ Error loading OASIS data: {e}")
        else:
            logger.error("❌ Missing OASIS CSV files")
    else:
        logger.error("❌ OASIS raw data directory not found")
    
    # Example of running the full preprocessing pipeline
    # Uncomment the following lines to run the full preprocessing
    # logger.info("\n🚀 Running full OASIS preprocessing pipeline...")
    # try:
    #     labels_df, features_df, metadata = preprocess_oasis_data()
    #     logger.info("✅ Full preprocessing completed successfully!")
    #     
    #     # Run validation
    #     validation_results = validate_oasis_preprocessing("ProcessedData/OASIS")
    #     logger.info(f"Validation Status: {validation_results['overall_status']}")
    #     logger.info(f"Quality Score: {validation_results['quality_score']:.1f}%")
    #     
    # except Exception as e:
    #     logger.error(f"❌ Full preprocessing failed: {e}")
    
    # Example of loading processed data
    logger.info("\n🚀 Testing OASIS processed data loading...")
    processed_data_path = "ProcessedData/OASIS"
    if os.path.exists(processed_data_path):
        logger.info("✅ Processed data already exists, testing loading...")
        try:
            oasis_data = load_oasis_for_mainmodel()
            if oasis_data:
                logger.info(f"✅ Successfully loaded {len(oasis_data['dataset'])} samples")
                logger.info(f"✅ Available modalities: {oasis_data['modality_types']}")
                
                # Test loading a sample from the dataset
                dataset = oasis_data['dataset']
                if len(dataset) > 0:
                    sample = dataset[0]
                    logger.info(f"✅ Sample loaded: {sample['sample_id']}")
                    logger.info(f"✅ Features shape: {sample['features'].shape}")
                    logger.info(f"✅ CDR label: {sample['label']}")
                    logger.info(f"✅ Dementia severity: {sample['dementia_severity']}")
                else:
                    logger.warning("No samples in dataset to load.")
            else:
                logger.error("❌ Failed to load OASIS data for MainModel.")
        except Exception as e:
            logger.error(f"❌ Error during processed data loading test: {e}")
    else:
        logger.warning(f"❌ Processed data directory not found: {processed_data_path}. Please run preprocessing first.")
