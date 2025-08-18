# 📊 Dataset Documentation - 4-Dataset Benchmarking Portfolio

## Overview
This document provides comprehensive information about the **four carefully selected datasets** used for benchmarking the Modality-Aware Ensemble Learning Framework. These datasets provide diverse multimodal scenarios across different domains, task types, and modality combinations to thoroughly evaluate the framework's capabilities.

**Last Updated**: August 18, 2024
**Status**: All datasets downloaded and preprocessing scripts implemented

## 🎯 **Strategic Dataset Selection**
Our 4-dataset portfolio covers:
- **Different modality combinations**: 2-modality and 3-modality scenarios
- **Different domains**: E-commerce, Medical, Computer Vision, Local Business
- **Different task types**: Regression, Classification, Retrieval, Multi-label
- **Different data characteristics**: Clean vs. noisy, professional vs. user-generated

--- 

## 1. Amazon Product Reviews Dataset

### � **Download Instructions**
```bash
# Option 1: Amazon Review Data (Complete dataset)
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz

# Option 2: Fine Food Reviews (Smaller subset)
wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Grocery_and_Gourmet_Food_5.json.gz

# Official source: https://nijianmo.github.io/amazon/index.html
```

### �📈 **Prediction Goal**
**Sentiment Analysis & Rating Prediction** - Predict customer rating (1-5 stars) based on review text, product metadata, and synthetic product images.

### 🎯 **Task Type**
- **Primary**: Regression/Ordinal Classification (rating prediction)
- **Secondary**: Sentiment analysis (positive/negative classification)

### 🏷️ **Output Labels**
- **Label Column**: `overall` (rating)
- **Label Type**: Numerical (1-5 scale)
- **Distribution**: 1=Very Negative, 2=Negative, 3=Neutral, 4=Positive, 5=Very Positive
- **Encoding**: Direct numerical values (1, 2, 3, 4, 5)

### 📝 **Raw Features (Data/AmazonReviews/)**
| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `reviewText` | Text | Customer review content | "This product is amazing! I love it so much." |
| `summary` | Text | Review title/summary | "Amazing product" |
| `overall` | Numerical | Rating score (1-5) | 5 |
| `helpful` | List | Helpfulness votes [helpful, total] | "[2, 3]" |
| `asin` | String | Product identifier | "B000JVER7W" |
| `price` | Numerical | Product price in dollars | 75.92 |
| `category` | Categorical | Product category | "Electronics" |
| `brand` | Categorical | Product brand name | "BrandA" |

### 🔧 **Processed Features**
#### Text Features (1000 dimensions)
- **Method**: TF-IDF Vectorization
- **Features**: Top 1000 most important n-grams (1-gram and 2-gram)
- **Source**: Combined `reviewText` + `summary`
- **Preprocessing**: Lowercasing, stopword removal, special character cleaning

#### Metadata Features (14 dimensions)
- `price`: Product price (scaled)
- `helpful_votes`: Number of helpful votes (scaled)
- `total_votes`: Total votes (helpful + unhelpful)
- `helpful_ratio`: Ratio of helpful to total votes [0-1]
- `category_encoded`: Product category (label encoded)
- `brand_encoded`: Brand name (label encoded)
- `review_length`: Character count of review text
- `word_count`: Word count of review text
- `avg_word_length`: Average word length
- `sentence_count`: Number of sentences
- `exclamation_count`: Number of exclamation marks
- `question_count`: Number of question marks
- `review_length * word_count`: Derived feature
- `helpful_ratio * total_votes`: Derived feature

#### Image Features (None)
- **Note**: Current processed data does not include image features
- **Reason**: Matches existing processed data format

### 🎛️ **Modalities Present**
1. **Text Modality**: Review content and summaries
2. **Structured Data**: Product metadata (price, category, brand, helpfulness)
3. **Image Modality**: Synthetic product images
4. **Temporal Features**: Text statistics and engagement metrics

### 📊 **Dataset Statistics**
- **Total Samples**: 1,689,188 (raw data)
- **Current Processed**: 200,000 samples
- **Train Split**: 160,000 samples (80%)
- **Test Split**: 40,000 samples (20%)
- **Missing Values**: Handled during preprocessing
- **Class Balance**: Distributed across rating levels

---

## 2. COCO Captions Dataset

### 📥 **Download Instructions**
```bash
# Images and annotations
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract files
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

# Official source: https://cocodataset.org/#download
# Captions: https://cocodataset.org/#captions-2017
```

### 📈 **Prediction Goal**
**Image-Text Retrieval & Caption Generation** - Given an image, retrieve relevant captions or generate new captions. Given a caption, retrieve relevant images.

### 🎯 **Task Type**
- **Primary**: Image-to-text retrieval
- **Secondary**: Text-to-image retrieval  
- **Alternative**: Caption generation/ranking

### 🏷️ **Output Labels**
- **Label Type**: Binary classification (image-caption matching)
- **Format**: 0.0 (negative pair) or 1.0 (positive pair)
- **Task**: Distinguish between correct and incorrect image-caption pairs
- **Evaluation**: Binary classification metrics (accuracy, precision, recall, F1)

### 📝 **Raw Features**
| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `image_id` | Numerical | Unique image identifier | 12345 |
| `image_path` | String | Path to image file | "train2017/000000012345.jpg" |
| `caption` | Text | Human-written description | "A red bus driving down a city street" |
| `caption_id` | Numerical | Unique caption identifier | 67890 |

### 🔧 **Processed Features**
#### Image Features (2048 dimensions)
- **Method**: ResNet-50 CNN features (pre-trained)
- **Extraction**: Final pooling layer before classification
- **Preprocessing**: Images resized to 224×224, normalized
- **Alternative**: Vision Transformer (ViT) features

#### Text Features (2000 dimensions)
- **Method**: TF-IDF Vectorization
- **Source**: Caption text
- **Preprocessing**: Lowercasing, stopword removal, special character cleaning
- **Features**: Top 2000 most important n-grams (1-gram and 2-gram)

#### Metadata Features (8 dimensions)
- `image_width`: Original image width
- `image_height`: Original image height  
- `caption_length`: Character count of caption
- `word_count`: Word count in caption
- `noun_count`: Number of nouns in caption
- `verb_count`: Number of verbs in caption
- `adj_count`: Number of adjectives in caption
- `sentence_count`: Number of sentences in caption

### 🎛️ **Modalities Present**
1. **Visual Modality**: Natural scene images (118K training images)
2. **Text Modality**: Human-written captions (590K captions)
3. **Structural Metadata**: Image dimensions and text statistics

### 📊 **Dataset Statistics**
- **Training Images**: 118,287
- **Validation Images**: 5,000  
- **Total Captions**: 616,767 (raw data)
- **Current Processed**: 1,600 samples
- **Train Split**: 1,280 samples (80%)
- **Test Split**: 320 samples (20%)
- **Average Caption Length**: 10.5 words
- **Vocabulary Size**: ~27,000 unique words
- **Image Resolution**: Variable (resized to 224×224 for processing)

---

## 3. ChestX-ray14 Dataset (NIH Clinical Center)

### 📥 **Download Instructions**
```bash
# Method 1: Official NIH Download (Recommended)
# Visit: https://nihcc.app.box.com/v/ChestXray-NIHCC
# Manual download required due to frequent link changes

# Method 2: Kaggle Dataset (Alternative)
# Install Kaggle CLI: pip install kaggle
# Setup API credentials: https://www.kaggle.com/docs/api
kaggle datasets download -d nih-chest-xrays/data
unzip data.zip

# Method 3: Academic Torrents (Reliable)
# Download via: https://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da5104048741
# torrent file: ChestXray-NIHCC.torrent

# Method 4: Google Drive Mirror (if available)
# Check research community forums for current mirrors

# Required files structure after download:
# images/ (112,120 PNG files)
# Data_Entry_2017_v2020.csv (labels and metadata)
# BBox_List_2017.csv (bounding boxes - optional)
# README_ChestXray.pdf (documentation)

# Official source: https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community
```

### 📈 **Prediction Goal**
**Multi-label Medical Diagnosis** - Predict presence/absence of 14 pathological findings in chest X-rays using imaging data and extracted metadata features.

### 🎯 **Task Type**
- **Type**: Multi-label binary classification
- **Labels**: 14 thoracic pathologies
- **Approach**: Binary classification for each pathology

### 🏷️ **Output Labels**
- **Label Format**: Binary matrix (14 columns, one per pathology)
- **Pathologies**: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax
- **Encoding**: 1=Present, 0=Absent
- **Multi-label**: Each image can have multiple pathologies
- **Example**: [1,0,0,1,0,0,0,0,1,0,0,0,1,0] = Atelectasis + Consolidation + Infiltration + Pneumonia

### 📝 **Raw Features**
| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `Image Index` | String | Image filename identifier | "00000013_005.png" |
| `Finding Labels` | String | Pipe-separated pathology labels | "Cardiomegaly\|Emphysema" |
| `Follow-up #` | Integer | Follow-up number for patient | 5 |
| `Patient ID` | Integer | De-identified patient ID | 13 |
| `Patient Age` | Integer | Patient age at time of imaging | 58 |
| `Patient Gender` | String | Patient gender (M/F) | "M" |
| `View Position` | String | X-ray view position | "PA" (posteroanterior) |
| `OriginalImage[Width\|Height]` | Integer | Original image dimensions | 1024, 1024 |
| `OriginalImagePixelSpacing[X\|Y]` | Float | Pixel spacing in mm | 0.139, 0.139 |

### 🔧 **Processed Features**
#### Image Features (2048 dimensions)
- **Method**: DenseNet-121 pre-trained on ImageNet + fine-tuned on chest X-rays
- **Source**: Chest X-ray PNG images (1024×1024 original resolution)
- **Preprocessing**: Resize to 224×224, normalization, CLAHE enhancement for contrast
- **Alternative**: ResNet-50 or Vision Transformer features

#### Text Features (512 dimensions)
- **Method**: TF-IDF with medical terminology focus
- **Source**: Extracted pathology labels and metadata text
- **Preprocessing**: Medical abbreviation expansion, synonym mapping
- **Features**: Pathology terms, anatomical references, severity indicators

#### Metadata Features (10 dimensions)
- `patient_age`: Patient age (normalized to 0-1)
- `gender_encoded`: Patient gender (binary encoding)
- `view_position`: X-ray view type (PA, AP, lateral) 
- `image_width`: Original image width (normalized)
- `image_height`: Original image height (normalized)
- `pixel_spacing_x`: Horizontal pixel spacing
- `pixel_spacing_y`: Vertical pixel spacing
- `followup_number`: Follow-up study number
- `pathology_count`: Number of pathologies present
- `image_quality_score`: Computed image quality metric

### 🎛️ **Modalities Present**
1. **Medical Imaging**: Chest X-ray images (112K+ images)
2. **Pathology Labels**: Multi-label disease classifications
3. **Patient Metadata**: Demographics, imaging parameters, clinical context

### 📊 **Dataset Statistics**
- **Total Images**: 112,120
- **Patients**: 30,805  
- **Pathology Labels**: 14 thoracic disease categories
- **Images per Patient**: 1-5 (average: 3.6)
- **Multi-label Distribution**: 60% have 1 pathology, 25% have 2+, 15% normal
- **Most Common**: Infiltration (19%), Atelectasis (11%), Effusion (13%)
- **Training Split**: ~70% / Validation: ~15% / Test: ~15%
- **Image Format**: PNG (1024×1024 pixels)
- **Dataset Size**: ~45GB total

---

## 4. Yelp Open Dataset

### 📥 **Download Instructions**
```bash
# Download the complete Yelp Open Dataset
wget https://www.yelp.com/dataset/download/yelp_dataset.tar

# Extract the dataset
tar -xf yelp_dataset.tar

# Key files:
# - yelp_academic_dataset_business.json (business information)
# - yelp_academic_dataset_review.json (review text and ratings)
# - yelp_academic_dataset_user.json (user information)  
# - yelp_academic_dataset_photo.json (photo metadata)
# - photos.tar (actual business photos)

# Official source: https://www.yelp.com/dataset
# Note: Requires account registration and agreement to terms
```

### 📈 **Prediction Goal**
**Business Rating Prediction & Category Classification** - Predict business star ratings and classify business categories using review text, business metadata, and uploaded photos.

### � **Task Type**
- **Primary**: Rating prediction (1-5 stars, regression)
- **Secondary**: Business category classification (multi-label)
- **Tertiary**: Review sentiment analysis

### 🏷️ **Output Labels**
- **Rating Labels**: Numerical (1.0-5.0 stars, 0.5 increments)
- **Category Labels**: Multi-label (e.g., "Restaurants", "Italian", "Pizza", "Delivery")
- **Distribution**: Most businesses 3.5-4.5 stars
- **Categories**: 1000+ possible business categories

### 📝 **Raw Features**
| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `business_id` | String | Unique business identifier | "b1234567890abcdef" |
| `name` | String | Business name | "Tony's Pizza Palace" |
| `categories` | String | Business categories (pipe-separated) | "Restaurants\|Pizza\|Italian" |
| `stars` | Float | Average business rating | 4.5 |
| `review_count` | Integer | Number of reviews | 127 |
| `text` | String | Review text content | "Amazing pizza! Great service..." |
| `photo_id` | String | Photo identifier | "p1234567890abcdef" |

### � **Processed Features**
#### Text Features (1000 dimensions)
- **Method**: TF-IDF Vectorization (review-optimized)
- **Source**: Aggregated review text per business
- **Preprocessing**: Restaurant/service domain stopwords, sentiment preservation
- **Features**: Food terms, service quality, ambiance descriptors

#### Image Features (2048 dimensions)  
- **Method**: ResNet-50 features from business photos
- **Source**: User-uploaded business photos (food, interior, exterior)
- **Preprocessing**: Resize to 224×224, normalization, quality filtering
- **Content**: Food images, restaurant ambiance, storefront photos

#### Business Metadata Features (15 dimensions)
- `review_count`: Number of reviews (log-scaled)
- `checkin_count`: Number of check-ins
- `price_range`: Price level (1-4 $)
- `is_open`: Whether business is currently open
- `latitude/longitude`: Geographic coordinates (normalized)
- `hours_per_week`: Total operating hours
- `weekend_hours`: Weekend operating hours  
- `accepts_credit_cards`: Payment options
- `delivery_available`: Delivery service
- `takeout_available`: Takeout service
- `parking_available`: Parking availability
- `wifi_available`: WiFi availability
- `alcohol_served`: Alcohol service
- `noise_level`: Establishment noise level
- `ambiance_score`: Computed ambiance rating

### 🎛️ **Modalities Present**
1. **Text Modality**: User reviews and business descriptions
2. **Visual Modality**: Business photos (food, interior, exterior)
3. **Structured Metadata**: Business attributes, hours, location, amenities
4. **Geographic Data**: Location coordinates and neighborhood context

### 📊 **Dataset Statistics**
- **Businesses**: 150,346 (raw data)
- **Reviews**: 6,990,280 (raw data)
- **Current Processed**: 51,703 samples
- **Train Split**: 41,362 samples (80%)
- **Test Split**: 10,341 samples (20%)
- **Photos**: 200,100
- **Users**: 1.98 million
- **Cities**: 1,174 (across 8 countries)
- **Average Reviews per Business**: 46
- **Average Rating**: 3.7 stars

---

## 🔄 **Cross-Dataset Comparison**

| Aspect | Amazon Reviews | COCO Captions | MIMIC-CXR | Yelp Dataset |
|--------|----------------|---------------|-----------|--------------|
| **Domain** | E-commerce | Computer Vision | Healthcare | Local Business |
| **Task** | Rating Prediction | Image-Text Retrieval | Medical Diagnosis | Rating/Category Prediction |
| **Label Type** | Ordinal (1-5) | Text Captions | Multi-label (14) | Ordinal + Multi-label |
| **Text Source** | Reviews | Captions | Pathology Labels | Reviews |
| **Text Features** | 1000 (TF-IDF) | 2000 (TF-IDF) | 512 (TF-IDF) | 1000 (TF-IDF) |
| **Image Type** | Product Photos | Natural Scenes | Medical X-rays | Business Photos |
| **Image Features** | 2048 (ResNet) | 2048 (ResNet) | 2048 (DenseNet) | 2048 (ResNet) |
| **Metadata** | 14 features | 8 features | 10 features | 15 features |
| **Train Samples** | 160K | 1,280 | 112K images | 41K businesses |
| **Modalities** | Text + Meta | Image + Text + Meta | Image + Labels + Meta | Text + Image + Meta |
| **Complexity** | Medium | High | High | High |

---

## 🎯 **Research Evaluation Framework**

### **Performance Metrics by Dataset**
- **Amazon Reviews**: MSE, MAE, R², Accuracy (if classified)
- **COCO Captions**: BLEU-4, METEOR, CIDEr, SPICE, Recall@K
- **ChestX-ray14**: AUC-ROC, F1-score, Precision, Recall (per pathology)  
- **Yelp**: MSE, MAE, R² (rating), Hamming Loss (categories)

### **Multimodal-Specific Evaluations**
- **Missing Modality Robustness**: Performance degradation when modalities unavailable
- **Cross-Modal Consistency**: Prediction stability across modality combinations
- **Attention Analysis**: Modality importance scores and attention weights
- **Computational Efficiency**: Training time, inference speed, memory usage

### **Baseline Comparisons**
1. **Single-Modality Models**: Text-only, Image-only baselines
2. **Simple Fusion**: Early fusion (concatenation), Late fusion (voting)
3. **State-of-the-Art**: Best published results on each dataset
4. **Traditional Ensembles**: Random Forest, Gradient Boosting on concatenated features

### **Statistical Validation**
- **Cross-Validation**: 5-fold CV for robust performance estimates
- **Significance Testing**: Paired t-tests, Wilcoxon signed-rank tests
- **Effect Size**: Cohen's d for practical significance
- **Confidence Intervals**: 95% CI for all reported metrics

---

## 📚 **Dataset Access & Licensing**

### **Data Usage Guidelines**
- **Amazon Reviews**: Academic use, cite original papers
- **COCO Captions**: Creative Commons, commercial use allowed
- **ChestX-ray14**: Public domain, cite original NIH paper
- **Yelp Dataset**: Academic/educational use, terms of service apply

### **Citation Requirements**
```bibtex
# Amazon Product Data
@inproceedings{ni2019justifying,
  title={Justifying recommendations using distantly-labeled reviews and fined-grained aspects},
  author={Ni, Jianmo and Li, Jiacheng and McAuley, Julian},
  booktitle={Empirical Methods in Natural Language Processing},
  year={2019}
}

# COCO Dataset
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Dollár, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision},
  year={2014}
}

# ChestX-ray14
@article{wang2017chestx,
  title={ChestX-ray8: Hospital-scale chest x-ray database and benchmarks for weakly-supervised classification and localization of common thorax diseases},
  author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M},
  journal={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2017}
}

# Yelp Dataset
@misc{yelp2021dataset,
  title={Yelp Open Dataset},
  author={{Yelp Inc.}},
  year={2021},
  url={https://www.yelp.com/dataset}
}
```

---

## 🚀 **Implementation Timeline**

### **Phase 1: Data Acquisition (Week 1)**
- ✅ Download Amazon Reviews data
- ✅ Download COCO Captions dataset  
- ✅ Download ChestX-ray14 dataset
- ✅ Download Yelp Open Dataset

### **Phase 2: Preprocessing Pipelines (Week 2-3)**
- ✅ Implement Amazon Reviews preprocessor
- ✅ Implement COCO Captions preprocessor
- 🔧 Implement ChestX-ray14 preprocessor (Not Active)
- ✅ Implement Yelp Dataset preprocessor

### **Phase 3: Benchmarking Framework (Week 4)**
- 📊 Unified evaluation framework
- 📈 Statistical testing implementation
- 🔄 Cross-validation setup
- 📝 Results reporting automation

### **Phase 4: Experimental Evaluation (Week 5-6)**
- 🧪 Baseline model implementation
- 🔬 Ensemble framework evaluation
- 📊 Comprehensive result analysis
- 📋 Performance comparison tables

---

**Generated:** August 4, 2025  
**Last Updated:** August 18, 2024
**Status:** ✅ 3-DATASET BENCHMARKING PORTFOLIO COMPLETE  
**Next Steps:** Run full data preprocessing to convert all raw data to processed format

---

## 🔧 **Preprocessing Scripts**

### **Available Scripts**
All preprocessing scripts are located in `Benchmarking/Preprocessing/`:

- **`AmazonReviewsPreProcess.py`**: Processes 1.69M Amazon reviews
- **`CocoCaptionsPreProcess.py`**: Processes 616K COCO captions + images  
- **`YelpPreProcess.py`**: Processes 150K Yelp businesses + reviews
- **`run_all_preprocessing.py`**: Master script to run all preprocessing

### **Usage**
```bash
# Run all preprocessing
cd Benchmarking/Preprocessing
python3 run_all_preprocessing.py

# Run individual datasets
python3 AmazonReviewsPreProcess.py
python3 CocoCaptionsPreProcess.py
python3 YelpPreProcess.py
```

### **Output Structure**
Each script creates:
```
Benchmarking/ProcessedData/[DatasetName]/
├── train/
│   ├── text_features.npy
│   ├── image_features.npy (if applicable)
│   ├── metadata_features.npy
│   └── labels.npy
├── test/
├── preprocessing_components/
└── mainmodel_data.npy
```

### **Processing Time Estimates**
- **AmazonReviews**: 30-60 minutes (text + metadata only)
- **CocoCaptions**: 2-4 hours (text + images + metadata)
- **YelpOpen**: 4-8 hours (text + images + metadata)
- **Total**: 6-12 hours (recommended: run overnight)
