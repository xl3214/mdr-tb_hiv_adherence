'''
Raw Counseling Session Notes Pre-analysis Processing Script
1. Stopword Removal using nltk stopword list
2. Lemmatization
'''

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Initialize
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^A-Za-z\s]', ' ', text)
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def load_and_preprocess(file_path):
    df = pd.read_excel(file_path)
    documents = df.iloc[:, 3].astype(str).tolist()
    return [preprocess_text(doc) for doc in documents]
