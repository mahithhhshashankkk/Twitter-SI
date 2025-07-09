# Twitter-SI

 Suicidal Ideation Detection from Text

This project uses Natural Language Processing (NLP) and Machine Learning to detect suicidal ideation in user-generated text. It aims to provide early insights into high-risk language patterns from social media or user inputs.

 Features

- Text classification using **TF-IDF** and **Logistic Regression**
- Handles class imbalance using `class_weight='balanced'`
- Model evaluation with precision, recall, F1-score, and confusion matrix
- Predicts suicidal vs. non-suicidal labels for custom input text

 Example

```text
Text: I don’t want to live anymore
Prediction: Suicidal

Text: Life is beautiful and worth living
Prediction: Not Suicidal


