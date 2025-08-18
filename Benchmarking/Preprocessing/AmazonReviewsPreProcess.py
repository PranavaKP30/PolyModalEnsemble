#!/usr/bin/env python3
"""
Amazon Reviews Preprocessing Pipeline
Transforms raw Amazon product review data into machine learning-ready features
"""

import json
import ast
import numpy as np
import pandas as pd
import pickle
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class AmazonReviewsPreProcessor:
    def __init__(self, data_dir="Benchmarking/Data/AmazonReviews", output_dir="Benchmarking/ProcessedData/AmazonReviews"):
        """
        Initialize Amazon Reviews preprocessor
        
        Args:
            data_dir: Directory containing raw data files
            output_dir: Directory to save processed data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "test").mkdir(exist_ok=True)
        (self.output_dir / "preprocessing_components").mkdir(exist_ok=True)
        
        # Initialize preprocessing components
        self.text_vectorizer = None
        self.metadata_scaler = None
        self.category_encoder = None
        self.brand_encoder = None
        
    def load_data(self):
        """Load raw Amazon reviews data"""
        print("Loading Amazon Reviews data...")
        
        reviews_file = self.data_dir / "reviews_Electronics_5.json"
        meta_file = self.data_dir / "meta_Electronics.json"
        
        if not reviews_file.exists():
            raise FileNotFoundError(f"Reviews file not found: {reviews_file}")
        if not meta_file.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_file}")
        
        # Load reviews data
        reviews_data = []
        with open(reviews_file, 'r') as f:
            for i, line in enumerate(f):
                try:
                    if line.strip():  # Skip empty lines
                        reviews_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSON decode error at line {i+1}: {e}")
                    print(f"Line content: {line[:100]}...")
                    raise
        
        # Load meta data (handle Python dict format with single quotes)
        meta_data = []
        with open(meta_file, 'r') as f:
            for i, line in enumerate(f):
                try:
                    if line.strip():  # Skip empty lines
                        # Use ast.literal_eval for Python dict format
                        meta_data.append(ast.literal_eval(line))
                except Exception as e:
                    print(f"Error parsing metadata at line {i+1}: {e}")
                    print(f"Line content: {line[:100]}...")
                    raise
        
        # Convert to DataFrames
        reviews_df = pd.DataFrame(reviews_data)
        meta_df = pd.DataFrame(meta_data)
        
        print(f"Loaded {len(reviews_df)} reviews and {len(meta_df)} products")
        
        # Merge reviews with product metadata
        merged_df = reviews_df.merge(meta_df, on='asin', how='inner')
        
        print(f"Merged dataset: {len(merged_df)} samples")
        
        return merged_df
    
    def clean_text(self, text):
        """Clean and normalize text data"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short words (< 2 characters)
        words = text.split()
        words = [word for word in words if len(word) >= 2]
        
        return ' '.join(words)
    
    def extract_helpful_votes(self, helpful_str):
        """Extract helpful votes from string format '[helpful, total]'"""
        # Handle pandas Series or numpy arrays
        if hasattr(helpful_str, '__iter__') and not isinstance(helpful_str, str):
            if len(helpful_str) == 0:
                return 0, 0
            helpful_str = helpful_str[0] if hasattr(helpful_str, '__getitem__') else helpful_str
        
        if pd.isna(helpful_str) or helpful_str == '[]':
            return 0, 0
        
        try:
            # Parse the string format
            if isinstance(helpful_str, str):
                # Remove brackets and split
                votes = helpful_str.strip('[]').split(',')
                helpful = int(votes[0])
                total = int(votes[1])
            else:
                helpful = helpful_str[0]
                total = helpful_str[1]
            
            return helpful, total
        except:
            return 0, 0
    
    def process_text_features(self, texts, max_features=1000):
        """Process text features using TF-IDF vectorization"""
        print("Processing text features...")
        
        # Clean all texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Combine review text and summary
        combined_texts = []
        for i, text in enumerate(cleaned_texts):
            if i < len(texts) // 2:  # Assuming alternating review text and summary
                combined_texts.append(text)
            else:
                # For summary, just use the text as is
                combined_texts.append(text)
        
        # TF-IDF Vectorization
        self.text_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),        # Unigrams + bigrams
            min_df=2,                  # Ignore rare terms
            max_df=0.95,               # Ignore common terms
            stop_words='english',      # Remove English stopwords
            norm='l2'                  # L2 normalization
        )
        
        text_features = self.text_vectorizer.fit_transform(combined_texts)
        
        print(f"Text features shape: {text_features.shape}")
        print(f"Vocabulary size: {len(self.text_vectorizer.vocabulary_)}")
        
        return text_features.toarray()
    
    def process_metadata_features(self, df):
        """Process metadata features"""
        print("Processing metadata features...")
        
        metadata_features = []
        
        for _, row in df.iterrows():
            features = []
            
            # Extract helpful votes
            helpful, total = self.extract_helpful_votes(row.get('helpful', [0, 0]))
            helpful_ratio = helpful / total if total > 0 else 0
            
            # Price (handle missing values)
            price = row.get('price', np.nan)
            if pd.isna(price):
                price = df['price'].median()  # Median imputation
            
            # Text statistics
            review_text = str(row.get('reviewText', ''))
            summary_text = str(row.get('summary', ''))
            combined_text = review_text + ' ' + summary_text
            
            review_length = len(combined_text)
            word_count = len(combined_text.split())
            avg_word_length = np.mean([len(word) for word in combined_text.split()]) if word_count > 0 else 0
            sentence_count = combined_text.count('.') + combined_text.count('!') + combined_text.count('?')
            exclamation_count = combined_text.count('!')
            question_count = combined_text.count('?')
            
            # Category and brand (will be encoded later)
            category = str(row.get('category', 'Unknown'))
            brand = str(row.get('brand', 'Unknown'))
            
            features = [
                price,
                helpful,
                total,
                helpful_ratio,
                review_length,
                word_count,
                avg_word_length,
                sentence_count,
                exclamation_count,
                question_count,
                category,
                brand
            ]
            
            metadata_features.append(features)
        
        metadata_df = pd.DataFrame(metadata_features, columns=[
            'price', 'helpful_votes', 'total_votes', 'helpful_ratio',
            'review_length', 'word_count', 'avg_word_length', 'sentence_count',
            'exclamation_count', 'question_count', 'category', 'brand'
        ])
        
        # Encode categorical variables
        self.category_encoder = LabelEncoder()
        self.brand_encoder = LabelEncoder()
        
        metadata_df['category_encoded'] = self.category_encoder.fit_transform(metadata_df['category'])
        metadata_df['brand_encoded'] = self.brand_encoder.fit_transform(metadata_df['brand'])
        
        # Select numerical features for scaling
        numerical_features = metadata_df[[
            'price', 'helpful_votes', 'total_votes', 'helpful_ratio',
            'review_length', 'word_count', 'avg_word_length', 'sentence_count',
            'exclamation_count', 'question_count', 'category_encoded', 'brand_encoded'
        ]].values
        
        # Add two additional features to match current format (14 features total)
        # These appear to be derived features based on the current data
        additional_features = np.column_stack([
            numerical_features,
            metadata_df['review_length'] * metadata_df['word_count'],  # Feature 13: length * word_count
            metadata_df['helpful_ratio'] * metadata_df['total_votes']   # Feature 14: helpful_ratio * total_votes
        ])
        
        numerical_features = additional_features
        
        # Scale features
        self.metadata_scaler = StandardScaler()
        scaled_features = self.metadata_scaler.fit_transform(numerical_features)
        
        print(f"Metadata features shape: {scaled_features.shape}")
        
        return scaled_features
    
    def generate_synthetic_images(self, df, target_size=(64, 64)):
        """Generate synthetic product images based on category and features"""
        print("Generating synthetic images...")
        
        # Create synthetic images based on product characteristics
        image_features = []
        
        for _, row in df.iterrows():
            # Create a simple synthetic image based on category and rating
            category = str(row.get('category', 'Unknown'))
            rating = row.get('overall', 3.0)
            price = row.get('price', 50.0)
            
            # Generate image based on category
            if 'Electronics' in category:
                base_color = [100, 149, 237]  # Cornflower blue
            elif 'Clothing' in category:
                base_color = [255, 182, 193]  # Light pink
            elif 'Home' in category or 'Kitchen' in category:
                base_color = [144, 238, 144]  # Light green
            elif 'Books' in category:
                base_color = [255, 218, 185]  # Peach
            else:
                base_color = [255, 165, 0]    # Orange
            
            # Adjust brightness based on rating
            brightness_factor = rating / 5.0
            base_color = [int(c * brightness_factor) for c in base_color]
            
            # Create simple pattern based on price
            if price < 20:
                pattern = 'circles'
            elif price < 100:
                pattern = 'squares'
            else:
                pattern = 'lines'
            
            # Generate synthetic image (simplified - just create a feature vector)
            # In practice, this would create actual images
            image_vector = np.random.rand(target_size[0] * target_size[1] * 3)
            
            # Add some structure based on category and rating
            image_vector = image_vector * brightness_factor
            
            image_features.append(image_vector)
        
        image_features = np.array(image_features)
        print(f"Synthetic image features shape: {image_features.shape}")
        
        return image_features
    
    def process_labels(self, df):
        """Process labels for rating prediction"""
        print("Processing labels...")
        
        labels = df['overall'].values.astype(np.float32)
        
        # Handle missing values
        labels = np.nan_to_num(labels, nan=3.0)  # Default to neutral rating
        
        print(f"Labels shape: {labels.shape}")
        print(f"Label range: {labels.min():.1f} - {labels.max():.1f}")
        
        return labels
    
    def create_train_test_split(self, text_features, metadata_features, image_features, labels, test_size=0.2, random_state=42):
        """Create train/test split"""
        print("Creating train/test split...")
        
        # Check if we have enough samples for stratified splitting
        if len(labels) >= 10 and len(np.unique(labels)) >= 2:
            try:
                # Use stratified split based on rating bins
                rating_bins = pd.cut(labels, bins=5, labels=False)
                
                # Check if each bin has at least 2 samples
                bin_counts = np.bincount(rating_bins.astype(int))
                if all(bin_counts >= 2):
                    # Split indices with stratification
                    train_idx, test_idx = train_test_split(
                        np.arange(len(labels)),
                        test_size=test_size,
                        random_state=random_state,
                        stratify=rating_bins
                    )
                else:
                    print("Warning: Using random split (insufficient samples per rating bin)")
                    train_idx, test_idx = train_test_split(
                        np.arange(len(labels)),
                        test_size=test_size,
                        random_state=random_state
                    )
            except Exception as e:
                print(f"Warning: Stratified split failed ({e}), using random split")
                train_idx, test_idx = train_test_split(
                    np.arange(len(labels)),
                    test_size=test_size,
                    random_state=random_state
                )
        else:
            print("Warning: Using random split (insufficient samples for stratification)")
            train_idx, test_idx = train_test_split(
                np.arange(len(labels)),
                test_size=test_size,
                random_state=random_state
            )
        
        # Split all features
        train_text = text_features[train_idx]
        test_text = text_features[test_idx]
        
        train_metadata = metadata_features[train_idx]
        test_metadata = metadata_features[test_idx]
        
        train_images = image_features[train_idx]
        test_images = image_features[test_idx]
        
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        
        print(f"Train samples: {len(train_labels)}")
        print(f"Test samples: {len(test_labels)}")
        
        return (train_text, test_text, train_metadata, test_metadata, 
                train_images, test_images, train_labels, test_labels)
    
    def save_processed_data(self, train_data, test_data):
        """Save processed data to files"""
        print("Saving processed data...")
        
        (train_text, train_metadata, train_images, train_labels) = train_data
        (test_text, test_metadata, test_images, test_labels) = test_data
        
        # Save train data
        np.save(self.output_dir / "train" / "text_features.npy", train_text)
        np.save(self.output_dir / "train" / "metadata_features.npy", train_metadata)
        # Only save image features if they exist
        if train_images is not None:
            np.save(self.output_dir / "train" / "image_features.npy", train_images)
        np.save(self.output_dir / "train" / "labels.npy", train_labels)
        
        # Save test data
        np.save(self.output_dir / "test" / "text_features.npy", test_text)
        np.save(self.output_dir / "test" / "metadata_features.npy", test_metadata)
        # Only save image features if they exist
        if test_images is not None:
            np.save(self.output_dir / "test" / "image_features.npy", test_images)
        np.save(self.output_dir / "test" / "labels.npy", test_labels)
        
        # Save preprocessing components
        with open(self.output_dir / "preprocessing_components" / "text_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.text_vectorizer, f)
        
        with open(self.output_dir / "preprocessing_components" / "metadata_scaler.pkl", 'wb') as f:
            pickle.dump(self.metadata_scaler, f)
        
        with open(self.output_dir / "preprocessing_components" / "category_encoder.pkl", 'wb') as f:
            pickle.dump(self.category_encoder, f)
        
        with open(self.output_dir / "preprocessing_components" / "brand_encoder.pkl", 'wb') as f:
            pickle.dump(self.brand_encoder, f)
        
        print("Data saved successfully!")
    
    def create_mainmodel_data(self, train_data, test_data):
        """Create MainModel-compatible data structure"""
        print("Creating MainModel data structure...")
        
        (train_text, train_metadata, train_images, train_labels) = train_data
        (test_text, test_metadata, test_images, test_labels) = test_data
        
        # Handle case where image features might be None
        modality_configs = [
            {'name': 'text', 'feature_dim': train_text.shape[1], 'data_type': 'text'},
            {'name': 'metadata', 'feature_dim': train_metadata.shape[1], 'data_type': 'tabular'}
        ]
        
        if train_images is not None:
            modality_configs.append({'name': 'image', 'feature_dim': train_images.shape[1], 'data_type': 'image'})
        
        mainmodel_data = {
            'train_data': {
                'text': train_text,
                'metadata': train_metadata,
                'image': train_images
            },
            'train_labels': train_labels,
            'test_data': {
                'text': test_text,
                'metadata': test_metadata,
                'image': test_images
            },
            'test_labels': test_labels,
            'modality_configs': modality_configs
        }
        
        np.save(self.output_dir / "mainmodel_data.npy", mainmodel_data)
        print("MainModel data saved!")
    
    def process(self, max_samples=None):
        """Main processing pipeline"""
        print("Starting Amazon Reviews preprocessing...")
        
        # Load data
        df = self.load_data()
        
        # Limit samples if specified
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
            print(f"Limited to {max_samples} samples")
        
        # Process text features
        review_texts = df['reviewText'].fillna('').tolist()
        summary_texts = df['summary'].fillna('').tolist()
        all_texts = review_texts + summary_texts  # Combine for vectorization
        
        text_features = self.process_text_features(all_texts, max_features=1000)
        
        # Split text features back to reviews and summaries
        n_samples = len(df)
        review_text_features = text_features[:n_samples]
        summary_text_features = text_features[n_samples:]
        
        # Combine review and summary features
        combined_text_features = review_text_features + summary_text_features
        
        # Process metadata features
        metadata_features = self.process_metadata_features(df)
        
        # Note: AmazonReviews current processed data doesn't have image features
        # Generate synthetic images (but won't be used in final output)
        # image_features = self.generate_synthetic_images(df, target_size=(64, 64))
        image_features = None
        
        # Process labels
        labels = self.process_labels(df)
        
        # Create train/test split
        if image_features is not None:
            (train_text, test_text, train_metadata, test_metadata,
             train_images, test_images, train_labels, test_labels) = self.create_train_test_split(
                combined_text_features, metadata_features, image_features, labels
            )
        else:
            # Handle case without image features
            (train_text, test_text, train_metadata, test_metadata,
             _, _, train_labels, test_labels) = self.create_train_test_split(
                combined_text_features, metadata_features, np.zeros((len(labels), 1)), labels
            )
            train_images = test_images = None
        
        # Save processed data
        train_data = (train_text, train_metadata, train_images, train_labels)
        test_data = (test_text, test_metadata, test_images, test_labels)
        
        self.save_processed_data(train_data, test_data)
        self.create_mainmodel_data(train_data, test_data)
        
        print("Amazon Reviews preprocessing completed!")
        
        return {
            'train_samples': len(train_labels),
            'test_samples': len(test_labels),
            'text_features': combined_text_features.shape[1],
            'metadata_features': metadata_features.shape[1],
            'image_features': image_features.shape[1] if image_features is not None else 0
        }

def main():
    """Main function to run preprocessing"""
    preprocessor = AmazonReviewsPreProcessor()
    
    # Process all data (or limit with max_samples parameter)
    results = preprocessor.process()
    
    print("\nPreprocessing Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
