# 📧 Spam Email Detector

NLP-based spam classifier using TF-IDF feature extraction and multiple scikit-learn classifiers on the UCI SMS Spam Collection dataset.

## Overview
| Detail | Value |
|--------|-------|
| Type | Binary Text Classification |
| Dataset | UCI SMS Spam Collection (5,572 messages, auto-downloaded) |
| Framework | scikit-learn |
| Models | Naive Bayes, Logistic Regression, Linear SVM, Random Forest |

## Getting Started
```bash
git clone https://github.com/Dnshitobu/spam-email-detector.git
cd spam-email-detector
pip install -r requirements.txt
python spam_detector.py
```

## What It Does
1. Auto-downloads SMS Spam dataset
2. EDA: class distribution, message length analysis
3. Text preprocessing: lowercase, remove punctuation
4. TF-IDF with bigrams (5,000 features)
5. Trains 4 models, evaluates with accuracy + ROC-AUC
6. Shows most discriminative spam/ham words

## Results
| Model | Accuracy | ROC-AUC |
|-------|:---:|:---:|
| Naive Bayes | ~0.982 | ~0.993 |
| **Linear SVM** | **~0.988** | **~0.997** |

## Concepts Covered
TF-IDF · N-grams · Naive Bayes · Precision vs Recall · Text preprocessing pipeline
