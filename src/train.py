import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import sys
import os

# Add the parent directory to sys.path to import from src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import TextPreprocessor

# Load knowledge base
kb = pd.read_csv(os.path.join(BASE_DIR, "data", "knowledge_base.csv"))

# Initialize preprocessor
pre = TextPreprocessor()

# Preprocess questions
processed_questions = [pre.preprocess(q) for q in kb["question"]]

# Train TF-IDF vectorizer
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(processed_questions)

# Save models
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)
pickle.dump(vectorizer, open(os.path.join(models_dir, "tfidf_vectorizer.pkl"), "wb"))
pickle.dump(matrix, open(os.path.join(models_dir, "tfidf_matrix.pkl"), "wb"))

print("Models retrained and saved successfully.")
