# Sentiment Analysis on E-Commerce Reviews

## Introduction

This project performs sentiment analysis on e-commerce product reviews using deep learning models from the Transformers library. The goal is to classify the sentiment of each review as **negative**, **neutral**, or **positive**.

## Data

- Main fields in the dataset include:
  - `review_text`: the text of the customer review.
  - `label`: sentiment labels with values:
    - 0 = negative
    - 1 = neutral
    - 2 = positive

## Data Preprocessing

- Clean the text by removing special characters, extra whitespace, etc.
- Convert numeric labels to descriptive strings using:

```python
label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
```

## Model

- Uses `transformers.AutoModelForSequenceClassification` for the sentiment classification task.
- The corresponding tokenizer is loaded using `AutoTokenizer`.
- Fine-tuning is performed on the training dataset using Hugging Face's `Trainer`.

## Training

- Training parameters:
  - Epochs: 5
  - Batch size: 32
  - Optimizer: AdamW (default)
- Early stopping is used via `EarlyStoppingCallback` to prevent overfitting.
- The `Trainer` is configured with callbacks and evaluation metrics such as accuracy and F1-score.

## Evaluation

- Metrics used:
  - Accuracy
  - F1-score
- Final evaluation is printed on the test dataset.

## How to Use

1. Install required libraries:

```bash
pip install transformers torch scikit-learn gdown
```

2. Run the notebook to:
   - Download and preprocess the data
   - Train the model
   - Make predictions and evaluate the results
