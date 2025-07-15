# Install required libraries
!pip install pandas scikit-learn nltk

# Clone the FakeNewsNet repository
!git clone https://github.com/KaiDMML/FakeNewsNet.git

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab') # Ensure this is downloaded

# Define paths to the dataset files
dataset_path = '/content/FakeNewsNet/dataset/'

# Load the datasets
politifact_fake = pd.read_csv(os.path.join(dataset_path, 'politifact_fake.csv'))
politifact_real = pd.read_csv(os.path.join(dataset_path, 'politifact_real.csv'))
gossipcop_fake = pd.read_csv(os.path.join(dataset_path, 'gossipcop_fake.csv'))
gossipcop_real = pd.read_csv(os.path.join(dataset_path, 'gossipcop_real.csv'))

# Add labels: 1 for fake, 0 for real
politifact_fake['label'] = 1
politifact_real['label'] = 0
gossipcop_fake['label'] = 1
gossipcop_real['label'] = 0

# Combine the datasets
data = pd.concat([politifact_fake, politifact_real, gossipcop_fake, gossipcop_real], ignore_index=True)

# Text preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.isna(text):  # Handle missing values
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs, special characters, and numbers
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\W|\d', ' ', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the 'title' column
data['cleaned_text'] = data['title'].apply(preprocess_text)

# Remove empty or invalid entries
data = data[data['cleaned_text'] != '']

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit and transform the text data
X = tfidf.fit_transform(data['cleaned_text'])
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))