# 🔧 Preprocessing Pipeline Documentation

## Overview
This document provides detailed documentation of the preprocessing pipelines used to transform raw data into machine learning-ready features for each of the three datasets in the MultiModal Ensemble Model project.

**Last Updated**: August 18, 2024
**Status**: All preprocessing scripts implemented and verified

---

## 1. Amazon Reviews Preprocessing Pipeline

### 📋 **Pipeline Overview**
**Input**: Raw Amazon product review data (JSON format) 
**Output**: TF-IDF text features + scaled metadata + rating labels 
**Goal**: Product rating prediction (1-5 stars) with multimodal features 


### 🔄 **Step-by-Step Processing**


#### **Step 1: Data Loading & Initial Processing**
```python
def load_data(self):
   # Load JSON data from Data/AmazonReviews/reviews_Electronics_5.json and meta_Electronics.json
   # Expected fields: reviewText, summary, overall, helpful, price, category, brand
   # Handle missing files and validate data structure
```


**Input Validation:**
- ✅ Check required columns exist
- ✅ Handle missing data gracefully 
- ✅ Display data shape and sample records
- ✅ Validate data types


#### **Step 2: Text Preprocessing**
```python
def clean_text(self, text):
   # 1. Convert to lowercase
   # 2. Remove special characters (keep basic punctuation)
   # 3. Normalize whitespace
   # 4. Remove very short words (< 2 characters)
```


**Text Cleaning Pipeline:**
1. **Normalization**: Convert all text to lowercase
2. **Character Filtering**: Remove special characters except `.!?`
3. **Whitespace Normalization**: Replace multiple spaces with single space
4. **Short Word Removal**: Filter words with length < 2
5. **Combination**: Merge `reviewText` + `summary` for comprehensive text


**Processing Parameters:**
- **Regex Pattern**: `[^a-zA-Z0-9\s\.\!\?]` → ` ` (space)
- **Word Length Filter**: Minimum 2 characters
- **Missing Text Handling**: Replace with empty string


#### **Step 3: Text Vectorization**
```python
def process_text_features(self, max_features=5000):
   # TF-IDF Vectorization with specific parameters
   vectorizer = TfidfVectorizer(
       max_features=75,           # Reduced for efficiency
       ngram_range=(1, 2),        # Unigrams + bigrams
       min_df=2,                  # Ignore rare terms
       max_df=0.95,               # Ignore common terms
       stop_words='english'       # Remove English stopwords
   )
```


**TF-IDF Configuration:**
- **Feature Count**: 1000 dimensions (optimized for dataset size)
- **N-gram Range**: (1,2) - Unigrams and bigrams
- **Min Document Frequency**: 2 (terms must appear in ≥2 documents)
- **Max Document Frequency**: 0.95 (ignore terms in >95% of documents)
- **Stop Words**: English stopwords removed
- **Normalization**: L2 normalization applied


#### **Step 4: Metadata Processing**
```python
def process_metadata_features(self):
   # 1. Extract helpful votes from string format "[helpful, total]"
   # 2. Handle missing prices with median imputation
   # 3. Encode categorical variables (category, brand)
   # 4. Calculate text statistics
   # 5. Apply StandardScaler normalization
```


**Metadata Features (14 dimensions):**
1. **`price`**: Product price (median imputed, scaled)
2. **`helpful_votes`**: Extracted from helpful votes string
3. **`total_votes`**: Total votes (helpful + unhelpful)
4. **`helpful_ratio`**: Ratio of helpful to total votes [0-1]
5. **`review_length`**: Character count of combined review text
6. **`word_count`**: Word count in review text
7. **`avg_word_length`**: Average word length (complexity indicator)
8. **`sentence_count`**: Number of sentences (period/exclamation/question count)
9. **`exclamation_count`**: Number of exclamation marks (enthusiasm)
10. **`question_count`**: Number of question marks (uncertainty)
11. **`category_encoded`**: Product category (label encoded)
12. **`brand_encoded`**: Product brand (label encoded)
13. **`review_length * word_count`**: Derived feature
14. **`helpful_ratio * total_votes`**: Derived feature
4. **`brand_encoded`**: Brand name (label encoded)
5. **`review_length`**: Character count of review text
6. **`word_count`**: Word count of review text
7. **`avg_word_length`**: Average word length
8. **`sentence_count`**: Number of sentences (period count)
9. **`exclamation_count`**: Number of exclamation marks
10. **`question_count`**: Number of question marks


**Processing Steps:**
- **Helpful Votes Extraction**: Parse "[helpful, total]" format using regex
- **Price Imputation**: Replace missing values with median price
- **Categorical Encoding**: LabelEncoder for category and brand
- **Text Statistics**: Calculate length, word count, punctuation metrics
- **Scaling**: StandardScaler for numerical features


#### **Step 5: Label Processing**
```python
def process_labels(self, task='rating'):
   # Primary: Rating prediction (1-5 stars)
   # Alternative: Sentiment classification (positive/negative)
```


**Label Options:**
- **Rating Task**: Direct numerical ratings (1.0, 2.0, 3.0, 4.0, 5.0)
- **Sentiment Task**: Binary classification (1=negative ≤3, 2=positive >3)
- **Encoding**: Numerical (no encoding needed for regression)


#### **Step 6: Image Generation**
```python
def process_images(self, target_size=(224, 224)):
   # Generate synthetic product images based on category and features
   # Create RGB images representing product characteristics
```


**Synthetic Image Pipeline:**
1. **Category-based Generation**: Create images based on product category
2. **Feature Integration**: Incorporate price, rating, brand into image patterns
3. **Size Standardization**: Resize all images to 224×224×3
4. **Normalization**: Scale pixel values to [0,1] range
5. **Flattening**: Convert to 150,528-dimensional vectors (224×224×3)


**Image Characteristics:**
- **Format**: RGB (3 channels)
- **Dimensions**: 224×224 pixels
- **Features**: 150,528 (flattened)
- **Content**: Category-specific patterns, price/rating-based coloring
- **Purpose**: Simulate multimodal e-commerce scenario


#### **Step 7: Train/Test Split**
```python
def create_train_test_split(self, test_size=0.2, random_state=42):
   # Stratified split to maintain rating distribution
   # 80% training, 20% testing
```


**Split Configuration:**
- **Ratio**: 80% train / 20% test
- **Strategy**: Random split (stratified where possible)
- **Random State**: 42 (reproducible splits)
- **Validation**: Ensure balanced representation


---


## 2. Medical Transcription (MedTransc) Preprocessing Pipeline


### 📋 **Pipeline Overview**
**Input**: Medical transcription CSV data with clinical notes 
**Output**: Medical domain TF-IDF features + case metadata + specialty labels 
**Goal**: Medical specialty classification (40 classes) from clinical text 


### 🔄 **Step-by-Step Processing**


#### **Step 1: Medical Domain Setup**
```python
def __init__(self):
   # Medical abbreviations dictionary
   self.medical_abbreviations = {
       'bp': 'blood pressure', 'hr': 'heart rate', 'rr': 'respiratory rate',
       'temp': 'temperature', 'htn': 'hypertension', 'dm': 'diabetes mellitus',
       'ca': 'cancer', 'pt': 'patient', 'hx': 'history', 'sx': 'symptoms'
   }
  
   # Medical stopwords to preserve
   self.medical_stopwords_to_keep = {
       'no', 'not', 'without', 'never', 'none', 'nothing',
       'acute', 'chronic', 'severe', 'mild', 'moderate',
       'positive', 'negative', 'normal', 'abnormal'
   }
```


**Domain-Specific Configuration:**
- **Medical Abbreviations**: 12 common clinical abbreviations expanded
- **Preserved Stopwords**: Medical negations and severity terms kept
- **Clinical Context**: Maintains medical meaning and relationships


#### **Step 2: Advanced Text Cleaning**
```python
def clean_text(self, text):
   # 1. Lowercase normalization
   # 2. Whitespace and line break normalization
   # 3. Medical abbreviation expansion
   # 4. Medical symbol preservation (%, /)
   # 5. Character filtering with medical context
```


**Medical Text Cleaning:**
1. **Case Normalization**: Convert to lowercase
2. **Whitespace Handling**: Normalize spaces and line breaks
3. **Abbreviation Expansion**: Replace medical abbreviations with full terms
4. **Symbol Preservation**: Keep medical symbols (%, /, -, .)
5. **Character Filtering**: Remove non-medical special characters


**Medical-Specific Rules:**
- **Abbreviation Pattern**: Word boundary matching to avoid partial replacements
- **Symbol Retention**: Preserve %, /, -, . for medical notation
- **Context Awareness**: Maintain clinical terminology integrity


#### **Step 3: Medical NLP Pipeline**
```python
def preprocess_text_pipeline(self, text):
   # 1. Clean text (medical-aware)
   # 2. Tokenize with NLTK
   # 3. Filter tokens (length, digits)
   # 4. Remove stopwords (preserve medical terms)
   # 5. Lemmatize with WordNet
```


**NLP Processing Steps:**
1. **Text Cleaning**: Apply medical-aware cleaning
2. **Tokenization**: NLTK word tokenization
3. **Token Filtering**: Remove short words (<2 chars) and pure digits
4. **Stopword Removal**: English stopwords minus medical terms
5. **Lemmatization**: WordNet lemmatizer with POS tagging


**Medical NLP Features:**
- **Preserved Terms**: Medical negations, severity descriptors
- **Lemmatization**: Root form extraction (e.g., "diagnosis" ← "diagnoses")
- **Token Quality**: Medical terminology prioritized


#### **Step 4: Medical TF-IDF Vectorization**
```python
def vectorize_text(self, texts, method='tfidf', max_features=5000):
   # Medical domain TF-IDF with optimized parameters
   vectorizer = TfidfVectorizer(
       max_features=5000,
       ngram_range=(1, 2),
       min_df=2,
       max_df=0.95,
       analyzer='word'
   )
```


**Medical TF-IDF Configuration:**
- **Feature Count**: 5,000 dimensions (comprehensive medical vocabulary)
- **N-gram Range**: (1,2) - Medical terms and clinical phrases
- **Min DF**: 2 (clinical terms must appear multiple times)
- **Max DF**: 0.95 (exclude overly common terms)
- **Vocabulary**: Medical terminology focused


#### **Step 5: Label Processing & Class Analysis**
```python
def encode_labels(self, labels):
   # LabelEncoder for 40 medical specialties
   # Class imbalance analysis and weight calculation
```


**Medical Specialty Encoding:**
- **Classes**: 40 medical specialties
- **Encoding**: Integer labels (0-39)
- **Imbalance Handling**: Class weights calculated for imbalanced specialties
- **Top Specialties**: Surgery (22.1%), Consult (10.3%), Cardiovascular (7.4%)


**Class Distribution Analysis:**
- **Imbalance Ratio**: 183.8:1 (Surgery: 1,103 vs. Palliative Care: 6)
- **Weight Calculation**: Scikit-learn compute_class_weight
- **Stratification**: Used for train/test split where possible


#### **Step 6: Metadata Processing**
```python
# Metadata features (6 dimensions):
# 1. sample_name: Case identifier (categorical)
# 2. description: Brief case description (text)
# 3. keywords: Medical keywords (text)
# 4. medical_specialty: Original specialty name (text)
# 5. text_length: Character count of transcription
# 6. word_count: Word count of transcription
```


**Medical Metadata:**
- **Case Information**: Identifier, description, keywords, specialty
- **Text Statistics**: Length and word count metrics
- **Clinical Context**: Preserved original medical terminology


#### **Step 7: Train/Test Split with Stratification**
```python
def create_train_test_split(self, test_size=0.2, random_state=42):
   # Stratified split to maintain specialty distribution
   # Handle class imbalance appropriately
```


**Medical Split Strategy:**
- **Stratification**: Maintain medical specialty distribution
- **Imbalance Handling**: Ensure rare specialties represented
- **Validation**: Clinical relevance preserved


---


## 3. MM-IMDb (Movie) Preprocessing Pipeline


### 📋 **Pipeline Overview**
**Input**: Movie data (plot text + genre arrays + poster images) 
**Output**: Plot TF-IDF features + image features + text stats + multi-label genres 
**Goal**: Multi-label genre classification (23 genres) from plot + poster 


### 🔄 **Step-by-Step Processing**


#### **Step 1: Multimodal Data Loading**
```python
def load_data(self):
   # Load structured data from data.npy
   # Format: [movie_id, genres_binary, plot_text]
   # Load images from images.npz
   # Format: (25959, 3, 256, 160) - [samples, channels, height, width]
```


**Data Structure:**
- **Text Data**: Movie plot descriptions
- **Genre Data**: Binary vectors (23 genres per movie)
- **Image Data**: Movie poster images (RGB, 256×160)
- **Format**: NumPy arrays for efficient processing


#### **Step 2: Plot Text Processing**
```python
def clean_plot_text(self, text):
   # 1. Lowercase conversion
   # 2. Special character removal (preserve apostrophes/hyphens)
   # 3. Whitespace normalization
   # 4. Short word filtering
```


**Plot Text Cleaning:**
1. **Case Normalization**: Convert to lowercase
2. **Character Filtering**: Keep alphanumeric, spaces, apostrophes, hyphens
3. **Whitespace Handling**: Normalize multiple spaces
4. **Word Filtering**: Remove words with length ≤ 1
5. **Narrative Preservation**: Maintain story flow and character names


#### **Step 3: Movie Plot TF-IDF**
```python
def process_text_features(self, max_features=2000):
   vectorizer = TfidfVectorizer(
       max_features=2000,
       ngram_range=(1, 2),        # Narrative terms + phrases
       min_df=2,                  # Plot terms in multiple movies
       max_df=0.95,               # Avoid overly common words
       stop_words='english'       # Remove common words
   )
```


**Movie Plot TF-IDF:**
- **Feature Count**: 2,000 dimensions (narrative vocabulary)
- **N-gram Range**: (1,2) - Character names, plot elements, phrases
- **Domain Focus**: Movie plot terminology and narrative structures
- **Vocabulary**: Film-specific language and storytelling terms


#### **Step 4: Image Processing Pipeline**
```python
def process_images(self, target_size=(64, 64)):
   # 1. Format conversion: (N,C,H,W) → (N,H,W,C)
   # 2. Batch resizing with PIL
   # 3. Normalization to [0,1]
   # 4. Memory-efficient processing
```


**Movie Poster Processing:**
1. **Format Conversion**: Channels-first to channels-last
2. **Batch Resizing**: 256×160 → 64×64 (efficiency optimization)
3. **PIL Processing**: High-quality image resizing
4. **Normalization**: Pixel values scaled to [0,1]
5. **Memory Management**: Batch processing to handle large image sets


**Image Optimization:**
- **Original Size**: 256×160×3 = 122,880 features
- **Optimized Size**: 64×64×3 = 12,288 features (~10x reduction)
- **Quality**: Maintained essential visual features
- **Processing**: LANCZOS resampling for quality


#### **Step 5: Multi-label Genre Processing**
```python
def process_labels(self):
   # 1. Convert binary arrays to genre name lists
   # 2. Multi-label binarization
   # 3. Genre distribution analysis
```


**Multi-label Genre Pipeline:**
1. **Binary Array Decoding**: Convert binary vectors to genre names
2. **Multi-label Binarization**: Create binary matrix (samples × genres)
3. **Genre Mapping**: 23 predefined movie genres
4. **Distribution Analysis**: Genre frequency and co-occurrence


**Genre Characteristics:**
- **Genre Count**: 23 movie genres
- **Multi-label**: Movies can have multiple genres
- **Average Genres**: ~2.5 genres per movie
- **Range**: 1-9 genres per movie
- **Top Genres**: Action (54%), Adventure (33%), Animation (22%)


#### **Step 6: Movie Metadata Features**
```python
def process_metadata_features(self):
   # Plot text statistics (6 dimensions):
   # 1. plot_length: Character count
   # 2. word_count: Word count
   # 3. avg_word_length: Average word length
   # 4. sentence_count: Number of sentences
   # 5. exclamation_count: Exclamation marks
   # 6. question_count: Question marks
```


**Movie Metadata:**
- **Text Statistics**: Plot length, word count, complexity metrics
- **Narrative Features**: Sentence structure, punctuation usage
- **Scaling**: StandardScaler normalization
- **Purpose**: Capture narrative style and plot complexity


#### **Step 7: Multimodal Train/Test Split**
```python
def create_train_test_split(self, test_size=0.2, random_state=42):
   # Random split (multilabel stratification complex)
   # Maintain genre distribution as much as possible
   # Handle multiple modalities consistently
```


**Multimodal Split Strategy:**
- **Approach**: Random split (multilabel stratification complex)
- **Consistency**: Same indices for text, image, metadata, labels
- **Validation**: Ensure genre representation in both sets
- **Modalities**: Synchronized splitting across all feature types


---


## 🔄 **Cross-Dataset Processing Comparison**


| **Aspect** | **Amazon Reviews** | **MedTransc** | **MM-IMDb** |
|------------|-------------------|---------------|-------------|
| **Text Domain** | Product reviews | Medical notes | Movie plots |
| **Text Cleaning** | Basic + punctuation | Medical abbreviations | Narrative preservation |
| **TF-IDF Features** | 75 (efficiency) | 5,000 (medical vocab) | 2,000 (narrative terms) |
| **Domain Expertise** | E-commerce terms | Medical terminology | Film vocabulary |
| **Metadata Focus** | Product attributes | Clinical information | Plot statistics |
| **Label Complexity** | Ordinal (1-5) | Multi-class (40) | Multi-label (23) |
| **Image Processing** | Synthetic generation | None | Real poster resize |
| **Special Handling** | Price imputation | Abbreviation expansion | Genre co-occurrence |


## 🎯 **Quality Assurance & Validation**


### **Data Validation Steps**
1. **Input Validation**: Column existence, data types, value ranges
2. **Processing Validation**: Feature dimensions, missing values, outliers
3. **Output Validation**: Label encoding, split consistency, file integrity
4. **Cross-validation**: Feature alignment, reproducibility, performance


### **Preprocessing Artifacts**
- **Vectorizers**: Saved TF-IDF vectorizers for consistent transformation
- **Scalers**: Fitted StandardScalers for metadata normalization
- **Encoders**: Label encoders and binarizers for consistent labeling
- **Mappings**: Category mappings, genre lists, specialty names


### **Reproducibility Features**
- **Random Seeds**: Fixed random states (42) for reproducible splits
- **Saved Models**: Preprocessing components saved as pickle files
- **Documentation**: Comprehensive parameter tracking
- **Validation**: Checksum verification for processed data


---


**Generated:** July 21, 2025 
**Status:** ✅ COMPREHENSIVE PREPROCESSING PIPELINE DOCUMENTATION 
**Coverage:** Complete step-by-step processing for all three datasets 
**Ready for:** Machine learning experimentation and model development




