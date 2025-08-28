📰 Fake News Detection using Logistic Regression, LSTM, and BERT

This project focuses on detecting fake news by applying Machine Learning (Logistic Regression), Deep Learning (LSTM), and Transformer Models (BERT). The goal is to compare traditional ML, sequence-based deep learning, and transformer-based language models.
Features

Preprocessing of text data (cleaning, tokenization, stopword removal).

Feature extraction using TF-IDF for Logistic Regression.

Word embeddings with LSTM for sequence modeling.

Fine-tuning BERT model for advanced language understanding.

Comparative analysis of all models.

Visualization of results (accuracy graphs).
Dataset

We used the FakeNewsNet Dataset
, which provides real-world examples of true and fake news.

Type: Labeled dataset

Labels: Real and Fake

Data: News text, metadata, social context.
nstallation

Clone the repository:

git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
Models Implemented

Logistic Regression

Feature extraction using TF-IDF.

Lightweight and interpretable.

Accuracy: ~85%

LSTM (Long Short-Term Memory)

Recurrent Neural Network (RNN) model.

Handles sequential dependencies in text.

Word embeddings used as input (e.g., GloVe/Word2Vec).

Accuracy: ~88%

BERT (Bidirectional Encoder Representations from Transformers)

Transformer-based pre-trained language model.

Captures contextual meaning and long-range dependencies.

Fine-tuned on FakeNewsNet dataset.

Accuracy: ~93%
| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | \~85%    | 0.83      | 0.84   | 0.83     |
| LSTM                | \~88%    | 0.86      | 0.87   | 0.86     |
| BERT                | \~93%    | 0.92      | 0.93   | 0.92     |
✅ BERT performs the best, but LSTM provides a balance between traditional ML and transformers.
✅ Logistic Regression remains a strong baseline with TF-IDF.
▶️ Usage

Run Logistic Regression:

python logistic_regression.py


Run LSTM model:

python lstm_model.py


Run BERT model:

python bert_model.py

Untitled27 : BERT vs Logistic Regression
Untitled28.ipnyb : LSTM on Fakenewsnet Split 70/30 with time and accuracy comparison
FakeNewsNet- master.zip: Data Set Details
Logistic_regression_fakenewsnet : Logistic Regression on Fakenewsnet
bert.py.py : BERT on Fakenewsnet





