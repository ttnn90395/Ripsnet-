# Twitter Bot Detection - Deep Learning Project

A comprehensive machine learning pipeline for detecting Twitter bots using fine-tuned RoBERTa embeddings, traditional ML classifiers, and contrastive learning approaches.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Core Modules](#core-modules)
- [Model Training Scripts](#model-training-scripts)
- [Notebooks](#notebooks)
- [Model Checkpoints](#model-checkpoints)
- [Outputs](#outputs)
- [Installation & Usage](#installation--usage)

---

## 🎯 Project Overview

This project tackles the Twitter bot detection problem using a hybrid approach combining:
- **Fine-tuned RoBERTa models** for tweet text and user description embeddings
- **Traditional ML classifiers** (Random Forest, XGBoost, LightGBM) on hand-crafted features
- **User-based train/validation splits** to prevent data leakage
- **Separate models** for users with/without descriptions
- **Contrastive learning** approaches for improved representations

The goal is to classify Twitter accounts as bots (1) or humans (0) based on tweet content and user metadata.

---

## 📁 Repository Structure

```
DL-Competitive-Project/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── data/                              # Data directory
│   ├── train.jsonl                    # Training data (tweets + labels)
│   ├── kaggle_test.jsonl             # Test data for Kaggle submission
│   ├── X_train_roberta_text_scores.pkl          # RoBERTa scores for tweet text
│   ├── X_kaggle_roberta_text_scores.pkl         # RoBERTa scores for test set
│   ├── X_train_roberta_user_desc_scores.pkl     # RoBERTa scores for user descriptions
│   ├── X_kaggle_roberta_user_desc_scores.pkl    # RoBERTa scores for test descriptions
│   └── embeddings/
│       ├── X_train_roberta_embeddings.pkl       # 768-dim embeddings for training
│       └── X_kaggle_roberta_embeddings.pkl      # 768-dim embeddings for test
│
├── model_checkpoints/                 # Fine-tuned model weights
│   ├── twitter_roberta_best/         # Best performing RoBERTa checkpoint
│   └── twitter_roberta_final/        # Final RoBERTa checkpoint
│
├── outputs_for_kaggle/                # Kaggle submission CSV files
│   ├── model_2random_forest_combined_desc_no_desc.csv
│   ├── model_xgboost4new.csv
│   └── ...                           # Various model predictions
│
├── statistics_features/               # Feature statistics
│   └── all_data_feature_statistics.csv
│
├── preprocessing.py                   # Data loading and preprocessing utilities
├── utils.py                          # Evaluation metrics and visualization
├── contrastive_learning.py           # Supervised contrastive loss implementation
│
├── train_roberta_text.py             # Fine-tune RoBERTa on tweet text
├── train_roberta_user_desc.py        # Fine-tune RoBERTa on user descriptions
├── train_roberta_text_union.py       # Fine-tune on multiple tweets per user
│
├── baseline.ipynb                    # Main training notebook (RF, XGBoost, embeddings)
├── data_analysis.ipynb               # Exploratory data analysis
└── user_desc.png                     # Visualization asset

```

---

## 📊 Data

### Input Files

**`data/train.jsonl`**
- JSONL format with nested tweet objects
- Contains tweet metadata, user info, and binary labels (0=human, 1=bot)
- Key fields: `text`, `extended_tweet.full_text`, `user.id`, `user.description`, `user.created_at`, etc.

**`data/kaggle_test.jsonl`**
- Test data for Kaggle competition submission
- Same format as training data but without labels

### Preprocessed Features

**RoBERTa Scores (`.pkl` files)**
- Binary classification scores from fine-tuned RoBERTa models
- `*_text_scores.pkl`: Scores based on tweet text content
- `*_user_desc_scores.pkl`: Scores based on user profile descriptions

**RoBERTa Embeddings (`.pkl` files)**
- 768-dimensional dense representations from RoBERTa's `[CLS]` token
- Used for visualization (PCA/t-SNE) and embedding-based classifiers

---

## 🔧 Core Modules

### `preprocessing.py`

Centralized data loading and preprocessing pipeline.

**Key Functions:**
- `extract_full_text(tweet)`: Extracts full text from tweet, prioritizing `extended_tweet.full_text`
- `load_and_preprocess_data()`: Loads train/test JSONL files, creates initial features:
  - `full_text`: Complete tweet text
  - `user.id`: Unique user identifier
  - `user.created_at.float`: Account creation timestamp (float)
  - `description_missing`: Binary indicator for missing user descriptions
- `train_val_split_by_user()`: Splits data by unique user IDs to prevent data leakage
  - Uses sorted user IDs for reproducibility across platforms
  - Default: 80/20 train/val split with `random_state=42`

**Returns:** `X_train`, `y_train`, `X_kaggle`

### `utils.py`

Evaluation and visualization utilities.

**Key Functions:**
- `evaluate_binary_classifier(y_true, y_pred, threshold=0.5)`:
  - Computes accuracy, precision, recall, F1, AUC-ROC
  - Finds optimal threshold based on accuracy
  - Visualizes confusion matrix, ROC curve, and prediction histograms
- `pred_tweet_averaging(predictions, user_ids)`:
  - Aggregates tweet-level predictions to user-level by averaging
  - Ensures consistent predictions for all tweets from same user

### `contrastive_learning.py`

Supervised contrastive learning implementation for learning discriminative embeddings.

**Components:**
- `RobertaWithProjection`: Neural network wrapping fine-tuned RoBERTa with projection head
  - Loads from `model_checkpoints/twitter_roberta_best`
  - Projects 768-dim embeddings to lower-dimensional space (default: 128)
  - Outputs L2-normalized embeddings
- `supervised_contrastive_loss(embeddings, labels, temperature=0.1)`:
  - Pulls together embeddings of same class (bot/human)
  - Pushes apart embeddings of different classes
  - Uses temperature-scaled cosine similarity

---

## 🚀 Model Training Scripts

### `train_roberta_text.py`

Fine-tunes RoBERTa on **tweet text** for binary classification.

**Approach:**
- Uses `extended_tweet.full_text` when available, else `text`
- Fine-tunes on individual tweets
- Outputs binary classification scores and 768-dim embeddings

### `train_roberta_user_desc.py`

Fine-tunes RoBERTa on **user profile descriptions**.

**Approach:**
- Extracts `user.description` field
- Handles missing descriptions
- Learns user-level representations

### `train_roberta_text_union.py`

Fine-tunes RoBERTa on **concatenated tweets per user**.

**Approach:**
- Groups tweets by user ID
- Samples up to 5 tweets per user
- Creates permutations for data augmentation
- Trains on user-level aggregated text

**Benefits:** Captures user-level patterns across multiple tweets

---

## 📓 Notebooks

### `baseline.ipynb`

Main training and evaluation notebook implementing the full pipeline.

**Sections:**
1. **Data Loading**: Loads preprocessed data with `load_and_preprocess_data()`
2. **Train/Val Split**: User-based splitting with `train_val_split_by_user()`
3. **Feature Engineering**: 
   - Loads RoBERTa scores (text + user descriptions)
   - Creates `description_missing` indicator
   - Separate datasets for users with/without descriptions
4. **Random Forest Models**:
   - Separate models for desc/no_desc subsets
   - Features: `user.listed_count`, `user.favourites_count`, `user.statuses_count`, `user.created_at.float`, `roberta_text_score`, (`roberta_user_desc_scores` for desc model)
   - Tweet averaging for user-level predictions
5. **XGBoost/LightGBM Optimization**: Optuna hyperparameter tuning (commented out)
6. **Embedding Analysis**:
   - Loads 768-dim RoBERTa embeddings
   - PCA (50 components) + t-SNE visualization
   - Random Forest and XGBoost on embeddings
7. **Kaggle Submission**: Generates CSV predictions

**Key Variables:**
- `X_tr`, `y_tr`, `X_val`, `y_val`: User-based train/val splits
- `X_tr_desc`, `X_tr_no_desc`: Subsets with/without user descriptions
- `X_tr_embeddings`, `X_val_embeddings`: Split embedding arrays

### `data_analysis.ipynb`

Exploratory data analysis notebook.

**Contents:**
- Feature distribution analysis
- Missing value statistics
- Correlation analysis
- Histogram plots by class (bot vs human)
- Feature statistics export to CSV

**Key Functions:**
- `plot_histograms_numerical_feature()`: Visualizes feature distributions by class
- `feature_statistics()`: Generates comprehensive feature stats

---

## 🏷️ Model Checkpoints

### `model_checkpoints/twitter_roberta_best/`

Fine-tuned RoBERTa model (best validation performance).

**Contents:**
- `config.json`: Model configuration
- `model.safetensors`: Model weights
- `vocab.json`, `merges.txt`: Tokenizer vocabulary
- `tokenizer.json`, `tokenizer_config.json`: Tokenizer settings
- `special_tokens_map.json`: Special token definitions

**Usage:**
```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("model_checkpoints/twitter_roberta_best")
tokenizer = AutoTokenizer.from_pretrained("model_checkpoints/twitter_roberta_best")
```

### `model_checkpoints/twitter_roberta_final/`

Final fine-tuned RoBERTa checkpoint (end of training).

---

## 📤 Outputs

### `outputs_for_kaggle/`

Kaggle competition submission files (CSV format).

**Format:**
```csv
ID,Prediction
0,1
1,0
...
```

**Notable Submissions:**
- `model_2random_forest_combined_desc_no_desc.csv`: Ensemble of separate desc/no_desc models
- `model_xgboost4new.csv`: XGBoost with 7 features
- `model_random_forest_averaged_4num_2roberta_new.csv`: RF with tweet averaging

---

## 🛠️ Installation & Usage

### Prerequisites

```bash
Python 3.12+
conda or venv
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/JosephdRf/DL-Competitive-Project.git
cd DL-Competitive-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure data files are in `data/` directory:
   - `train.jsonl`
   - `kaggle_test.jsonl`

### Training Pipeline

1. **Preprocess and explore data:**
```bash
jupyter notebook data_analysis.ipynb
```

2. **Fine-tune RoBERTa models** (optional, checkpoints provided):
```bash
python train_roberta_text.py
python train_roberta_user_desc.py
python train_roberta_text_union.py
```

3. **Train baseline models:**
```bash
jupyter notebook baseline.ipynb
```
   - Run cells sequentially
   - Generates predictions in `outputs_for_kaggle/`

4. **Experiment with contrastive learning:**
```python
from contrastive_learning import RobertaWithProjection, supervised_contrastive_loss
# Implement training loop with custom dataset
```

### Key Configuration

**Train/Val Split:**
- Modify `test_size` and `random_state` in `train_val_split_by_user()` call
- **Warning:** Changing split breaks reproducibility with existing embeddings

**Feature Selection:**
- Edit `numeric_features` and `categorical_features` lists in `baseline.ipynb`
- Separate feature sets for desc/no_desc models

**Model Hyperparameters:**
- Random Forest: `n_estimators`, `max_depth`, `random_state`
- XGBoost: Use Optuna optimization cell (commented in notebook)

---

## 📈 Performance Metrics

Models are evaluated using:
- **Accuracy** (primary metric)
- **Precision, Recall, F1-score**
- **AUC-ROC**
- **Confusion Matrix**
- **Tweet averaging** for user-level predictions

Best threshold optimization based on validation accuracy.

---

## 🔬 Methodology Highlights

1. **User-based splitting**: Prevents data leakage from same user appearing in train/val
2. **Separate models for desc/no_desc**: Handles missing user descriptions without imputation
3. **Fine-tuned RoBERTa**: Domain-adapted embeddings for Twitter bot detection
4. **Tweet averaging**: Aggregates tweet-level predictions to user-level
5. **Embedding visualization**: PCA + t-SNE for interpretability
6. **Hyperparameter optimization**: Optuna for XGBoost/LightGBM tuning

---

## 📝 Notes

- All random seeds set to `42` for reproducibility
- User-based splits maintain sorted user IDs for cross-platform consistency
- Embeddings are split using index alignment to match train/val splits
- `description_missing` feature acts as a gate for using `roberta_user_desc_scores`

---

## 👥 Contributors

Joseph de Roffignac
Roméo Nazaret 
Ten Nguyen Hanaoka
