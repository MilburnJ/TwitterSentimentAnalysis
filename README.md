# X Sentiment Analysis

This project presents a BERT-based deep learning model for sentiment classification on social media data from X (formerly Twitter). It fine-tunes a pretrained BERT encoder to accurately classify tweets into one of three sentiment classes: positive, negative, or neutral.

## Overview

The goal is to automate sentiment classification of social media posts to enable large-scale opinion analysis, reduce the cost of manual labeling, and provide insights for applications like political monitoring, consumer feedback, and cultural trend tracking.

## Dataset

- Source: [Kaggle Twitter Entity Sentiment Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data)
- Size: ~70,000 tweets
- Labels: Positive, Negative, Neutral (after merging “Irrelevant” into “Neutral”)
- Preprocessing:
  - Removal of numbers, special characters, and stopwords
  - Label mapping to integers
  - Tokenization using `bert-base-cased` with attention masks
  - Train/val/test split: 90% / 5% / 5%

## Methodology

- **Model**: BERT (bert-base-cased) + linear classifier head
- **Fine-tuning Pipeline**:
  - Encodings and masks generated via `encode_plus`
  - Feedforward classification layer on top of BERT embeddings
  - Dropout (p=0.3) for regularization
  - CrossEntropyLoss and Adam optimizer with a linear scheduler
- **Training Parameters**:
  - Epochs: 20
  - Batch size: 16
  - Learning rate: 2e-5 with linear scheduler

## Results

| Model                | Accuracy |
|---------------------|----------|
| BERT-based model     | 92.5%    |
| Random Forest (baseline) | 90.5%    |

- Accuracy remained consistently above 90% across various hyperparameter configurations.
- The biggest improvement came from merging “Irrelevant” into “Neutral”, resolving ambiguity in those labels.
- The model demonstrated strong generalization to modern (2023+) tweet samples despite being trained on 2021 data.

## Limitations

- Class imbalance caused by combining “Neutral” and “Irrelevant” may affect precision.
- Social media language evolves rapidly; slang and emoji use can degrade model performance over time.
- BERT does not natively interpret emojis or cultural context.

## Future Directions

- Expand label granularity beyond 3 classes to allow for more nuanced predictions.
- Incorporate emoji-to-text conversion to capture emotional signals more effectively.
- Retrain with more recent tweet data for higher relevancy and robustness.

## References

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Sentiment Analysis with BERT using HuggingFace](https://medium.com/@manjindersingh_10145/sentiment-analysis-with-bert-using-huggingface-88e99deeec9a)
- [Speech and Language Processing (Jurafsky & Martin)](https://web.stanford.edu/~jurafsky/slp3/)
- [Random Forest Sentiment Analysis - Kaggle](https://www.kaggle.com/code/omaradel1221/nlp-sentiment-analysis)

---

**Author**: Jakeb Milburn  
