# IMDB Sentiment Analysis

Live app: https://sentiment-analysis-jxx9ihdebzziztgfngpue5.streamlit.app

Classifies movie reviews as positive or negative. Data: IMDB reviews from Hugging Face
(5,000 for training, 1,000 for testing).

| Model | Accuracy |
|---|---|
| TF-IDF + Logistic Regression | 0.849 |
| Pretrained DistilBERT (no training) | 0.881 |
| Fine-tuned DistilBERT | 0.897 |


## Limitations
- Sarcasm is hard: "Oh great, the battery died" uses positive words for a negative opinion.
- Mixed reviews (good acting, bad plot) and neutral reviews are often misclassified.
- Only a subset of the data was used, so results could improve with more.
