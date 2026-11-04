# IMDB Sentiment Analysis

Classifies movie reviews as positive or negative. Data: IMDB reviews from Hugging Face
(5,000 for training, 1,000 for testing).

| Model | Accuracy |
|---|---|
| TF-IDF + Logistic Regression | 0.849 |
| Pretrained DistilBERT (no training) | 0.881 |
| Fine-tuned DistilBERT | 0.897 |
